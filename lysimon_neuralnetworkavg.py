# Import all required libraries
import pandas
from pandas import *
import numpy
from datetime import datetime

# Initialize values

env = None
sample_size = None
# Set env, if env = test, will only be run locally and display the result
env = "prod"
#env = "test"

enable_scaler=True
nb_models = 10

nb_year_min = 1
nb_year_max = 0

nb_week_min = 15
nb_week_max = 0
# Number of value on which to train, if null, train on all value
#sample_size = 120000
# Read training data + test data
df_data = pandas.read_csv("../input/train.csv")
df_test = pandas.read_csv("../input/test.csv")


# Display basic information
display(df_data.head(5))
print(df_data.describe())
df_data.columns

df_test.describe()
if env == "test":
    # Take the last 3 months of 2017 as testing data
    df_test = df_data[df_data.date >= '2017-10-01']
    # Remove the last 3 months, as it would not be fair to train on those
    df_train = df_data[df_data.date < '2017-10-01']
    if nb_week_max > 0 or nb_year_max >0:
        df_train = df_tain[df_train.date >= '2015-01-01']
    df_train = df_train[df_train.date >= '2016-01-01']
else:
    if nb_week_max > 0 or nb_year_max >0:
        df_train = df_data[df_data.date >= '2015-01-01']
    df_test['sales'] = 0

# Only select a small sample, faster local testing
if sample_size is not None and sample_size > 0:
    df_train = df_train.sample(sample_size)
    
display(df_train.head(5))
df_test.describe()
# Add time to df_data
from datetime import timedelta
from dateutil.relativedelta import relativedelta

# This can be quite slow, but is important for next steps

if nb_week_max > 0 or nb_year_max >0:
    df_data = df_data.assign(time = df_data.date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d")))

# Let's add the value for week 15 to 20
for i in range(nb_week_min, nb_week_min+nb_week_max):
    print("Adding weeks")
    df_data['weekplus{0}'.format(i)] = (df_data['time'] + timedelta(days=i*7)).astype(str)

for i in range(nb_year_min, nb_year_min+nb_year_max):
    print("Adding years")
    df_data['yearplus{0}'.format(i)] = (df_data['time'] + pandas.DateOffset(years=i)).astype(str)

df_data.head(5)
df_test.describe()
# First preparation step, adding some values to both dataframe
def prepare_initial(df, df_data, nb_year_max = 2, nb_week_max = 5):
    for i in range(nb_year_min, nb_year_min+nb_year_max):
        df = df.merge(df_data[['yearplus{0}'.format(i), 'store', 'item', 'sales']], how='inner', 
                           left_on=['date', 'store', 'item'],
                           right_on=['yearplus{0}'.format(i), 'store', 'item'],
                           suffixes=('', '_saleminus{0}year'.format(i)))
        
        df['sales_saleminus{0}year'.format(i)] = df['sales_saleminus{0}year'.format(i)].fillna(0)

        df = df.drop(columns=['yearplus{0}'.format(i)])


    for i in range(nb_week_min, nb_week_min+nb_week_max):
        df = df.merge(df_data[['weekplus{0}'.format(i), 'store', 'item', 'sales']], how='inner', 
                           left_on=['date', 'store', 'item'],
                           right_on=['weekplus{0}'.format(i), 'store', 'item'],
                           suffixes=('', '_saleminus{0}week'.format(i)))
        
        df['sales_saleminus{0}week'.format(i)] = df['sales_saleminus{0}week'.format(i)].fillna(0)
        
        df = df.drop(columns=['weekplus{0}'.format(i)])

    print("Start")
    return df
   

df_train = prepare_initial(df_train, df_data, nb_year_max, nb_week_max)
df_test = prepare_initial(df_test, df_data, nb_year_max, nb_week_max)

if env != "test":
    # Sometimes, we have duplicates, so they should be removed:
    df_test = df_test.drop_duplicates(subset='id', keep="last")
df_train.sample(5)
df_test.describe()
# Second preparation step
# We need to add all might be useful information from df_test and df_train
# Extracting some variable from date

def prepare_data(df, df_data):
    # Add time column, easier for later step
    df = df.assign(time = df.date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d")))
    
    # nb days since the beginning of the data, as traffic grows by store
    df = df.assign(days = df.time.apply(lambda x: (x - datetime(2012,12,31)).days))
    
    # Week day should be used 
    df = df.assign(weekday = df.time.apply(lambda x: x.weekday()))
    
    df = df.assign(dom = df.time.apply(lambda x: x.day))
    
    df = df.assign(cw = df.time.apply(lambda x: x.isocalendar()[1]))

    df = df.assign(month = df.time.apply(lambda x: x.month))
    
    df = pandas.get_dummies(df, prefix=['store', 'item', 'dom', 'cw', 'weekday', 'month'], 
                            columns=['store', 'item', 'dom', 'cw', 'weekday', 'month'])
    
    # Should we give more information? Like nb sales previous X weeks?
    # Previous sales of last 2/3 year at the same date (bank holiday, black friday, etc.)
    # Probably yes as this information is available, would be a good try
    return df

df_train = prepare_data(df_train, df_data)
df_test = prepare_data(df_test, df_data)

# Initialize column that do not exist in test with value 0, to avoid dummy not creating enough columns
# For instance, non existing month like month_5 will not exist in test set
for train_col in df_train.columns.values:
    if train_col not in df_test.columns.values:
        df_test[train_col] = 0

# Also train the other way around
for test_col in df_test.columns.values:
    if test_col not in df_train.columns.values:
        df_train[test_col] = 0
        
display(df_train.head(5))
display(df_test.head(5))
# Generate our training/validation datasets
from sklearn import model_selection

# Name of the result column
result_cols = ['sales']
result_excl_cols = 'sales_'
# Removing input_cols = ['store', 'item',
# dom, cw, 

# best model contained:
# days, store, item, weekday, month, cw, dom
input_cols = [
#    'sales_',
    'store_',
    'item_',
    #'day', always out
    #'day_',
    #'weekday', always out
    'dom_',
    'cw_',
    'weekday_',
    #'month', always out
    'month_',
    
    'days'
]

# Get the final values
def get_values(df, cols=[], excl_cols = "doqwidjoqwidjqwoidjqwoidjqwodijqw"):
    columns = df.columns.values
    # Remove all columns that are not inside the list
    for column in columns:
        find = False
        if column.startswith(excl_cols):
            print("Ignoring {0}".format(column))
        else:
            for col in cols:
                if column.startswith(col):
                    find = True
        if not find:
            df = df.drop(columns=[column])
    new_order = sorted(df.columns.values)
    print(new_order)
    # Same order for both training and testing set
    df = df[new_order]
    return df.values

X_train = get_values(df_train, input_cols)
Y_train = get_values(df_train, result_cols, result_excl_cols).ravel()
X_test = get_values(df_test, input_cols)

# In test env, we calculate it for the test only
if env == "test":
    Y_test = get_values(df_test, result_cols, result_excl_cols).ravel()
df_test.describe()
# Normalize the data


X_all = [x + y for x, y in zip(X_train, X_test)]
#print(len(X_all))
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler() 

# Don't cheat - fit only on training data
# Def adding x_train + X_test + X_validation to fit all of them

if enable_scaler:
    scaler.fit(X_train)  

    X_train = scaler.transform(X_train) 
    X_test = scaler.transform(X_test) 
# Custom function to calculate the SMAPE
def get_smape(Y_validation, Y_validation_predict):
    result = 0
    for i in range(0, len(Y_validation)):
        result += (abs(Y_validation[i] - Y_validation_predict[i]))/(abs(Y_validation[i])+abs(Y_validation_predict[i]))
    return result / len(Y_validation) * 200
# Import algorithm
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

models = []

for i in range(5, 5 + nb_models):
    models.append(('MLPRegressor_adam_{0}'.format(i), MLPRegressor(hidden_layer_sizes=(8,),  activation='relu', solver='adam',
                                                                   alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=False,
    random_state=i, tol=0.0001, verbose=False, warm_start=False, nesterovs_momentum=True,
    beta_1=0.9, beta_2=0.999, epsilon=1e-08)))
    
    #models.append(('MLPRegressor_adam_{0}'.format(i), MLPRegressor(hidden_layer_sizes=(8,),  activation='relu', solver='adam',
    #                                                               alpha=0.001, batch_size='auto',
    #learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    #random_state=i, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    #early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)))
    

# High value until first model get solved
max_score = 10000
best_model = "UNKNOWN"

res = []
# Testing all models, one by one
for name, model in models:
    print("Executing for model {0}".format(name))
    time_start = datetime.now()

    # Training the model
    model.fit(X_train, Y_train)
    
    print("Finish fit for {0}".format(name))

    Y_test_result = model.predict(X_test)
    res.append(Y_test_result)
    if env == "test":
        # We can calculate the avg error
        score = get_smape(Y_test, Y_test_result)
        print("Model {0} got score of {1}, time: {2}".format(name, score, datetime.now() - time_start))
    #else:
        # Let's write an output file, with the name of the model
        #print("Writing output file {0}.csv for model {0}".format(name))
        
        #df_test['sales'] = Y_test_result
        #result_df = df_test[['id', 'sales']]
        #result_df['sales'] = Y_test_result
        
        #result_df.to_csv("{0}.csv".format(name), index=False)
# For all result in res, if test, display the result, if not, write it to a file
final_res = []
nb_variable = len(res[0])
for variable in range(0, nb_variable):
    final_res.append(0.0)
    for i in range(0, len(res)):
        final_res[variable] += res[i][variable]
    final_res[variable] = final_res[variable] / len(res)

if env == "test":
    # We can calculate the avg error
    score = get_smape(Y_test, final_res)
    print("avg model got score of {0}".format(score))
else:
    print("Writing output file merged.csv".format(name))

    df_test['sales'] = final_res
    result_df = df_test[['id', 'sales']]
    result_df['sales'] = final_res

    result_df.to_csv("merged.csv".format(name), index=False)
