import numpy as np 

import pandas as pd 



import warnings

warnings.filterwarnings("ignore")



import seaborn as sns

sns.set_palette('Set2')

import matplotlib.pyplot as plt

%matplotlib inline



# Supress Scientific notation in python

pd.set_option('display.float_format', lambda x: '%.2f' % x)



# Display all columns of long dataframe

pd.set_option('display.max_columns', None)



import re



from math import sqrt 

from sklearn.metrics import mean_squared_log_error



import lightgbm as lgb

from sklearn.model_selection import KFold



import pandas_profiling
# Import datasets

train = pd.read_excel('../input/Data_Train.xlsx')

test = pd.read_excel('../input/Data_Test.xlsx')
# Checkout the shape of datasets

train.shape, test.shape
train.profile_report()
train.sample(5)
test.sample(5)
# Define categorical features

categorical_feature = ['Name','Location','Fuel_Type','Transmission','Owner_Type']
def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"+"There are " + str(mis_val_table_ren_columns.shape[0]) +" columns that have missing values.")        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(train)
missing_values_table(test)
train[categorical_feature].nunique()
test[categorical_feature].nunique()
# Check the Fuel_Type which is not in test set

list(train.Fuel_Type[~train.Fuel_Type.isin(test.Fuel_Type)].unique())
train = train[train.Fuel_Type != 'Electric']
train["Full_name"] = train.Name.copy()

test["Full_name"] = test.Name.copy()

train.Name.sample(20)
temp1 = list(train.Name.str.split(' ').str[0].unique())

temp2 = list(test.Name.str.split(' ').str[0].unique())

temp3 = [item for item in temp1 if item not in temp2]

temp3
train = train[~train.Name.str.contains('|'.join(temp3))]
def remove_year(data):

    result = re.search("([0-9]+[-]+[0-9]+)",data)

    if result:

        arr = data.replace(result.group(1),"")

        return arr

    else:

        return data
train.Name = train.Name.apply(lambda x: remove_year(x))

test.Name = test.Name.apply(lambda x: remove_year(x))
def remove_char(str):

    arr = ' '.join(str.split()) #replace multiple spaces to single space

    arr = re.sub(r"[-(){}<>/\.,]","", arr) #remove special characters

    return arr.lower() #lowercase all characters
train.Full_name = train.Name.apply(lambda x: remove_char(x))

test.Full_name = test.Name.apply(lambda x: remove_char(x))
train.Name = train.Full_name.apply(lambda x: " ".join(x.split(' ')[:2]))

test.Name = test.Full_name.apply(lambda x: " ".join(x.split(' ')[:2]))



# Filter brand name for a more generic aggregation in further calculations

train['brand'] = train.Name.apply(lambda x: " ".join(x.split(' ')[:1]))

test['brand'] = test.Name.apply(lambda x: " ".join(x.split(' ')[:1]))

train.Name.sample(10)
# Define function to correct the New_Price value

def price_correct(x):

    if str(x).endswith('Lakh'):

        return float(str(x).split()[0])*100000

    elif str(x).endswith('Cr'):

        return float(str(x).split()[0])*10000000

    else:

        return x



train.New_Price = train.New_Price.apply(price_correct)

test.New_Price = test.New_Price.apply(price_correct)
train.Mileage = train.Mileage.replace('0.0 kmpl', np.NaN).apply(lambda x: str(x).split()[0]).astype(float).round(2) # Convert 0 value to Nan, remove unit and convert to float type and round off to 2 decimal place.

train.Engine = train.Engine.apply(lambda x: str(x).split()[0]).astype(float) # Remove the CC part

train.Power = train.Power.replace('null bhp', np.NaN).apply(lambda x: str(x).split()[0]).astype(float).round(2) # convert null value to NaN than as above



test.Mileage = test.Mileage.replace('0.0 kmpl', np.NaN).apply(lambda x: str(x).split()[0]).astype(float).round(2)

test.Engine = test.Engine.apply(lambda x: str(x).split()[0]).astype(float)

test.Power = test.Power.replace('null bhp', np.NaN).apply(lambda x: str(x).split()[0]).astype(float).round(2)
# Fill missing values aggregating by Name mean and median

train.Engine = train.groupby('Name').Engine.apply(lambda x: x.fillna(x.median()))

train.Power = train.groupby('Name').Power.apply(lambda x: x.fillna(x.mean()))

train.Mileage = train.groupby('Name').Mileage.apply(lambda x: x.fillna(x.mean()))

train.Seats = train.groupby('Name').Seats.apply(lambda x: x.fillna(x.median()))

train.New_Price = train.groupby('Name').New_Price.apply(lambda x: x.fillna(x.mean()))



test.Engine = test.groupby('Name').Engine.apply(lambda x: x.fillna(x.median()))

test.Power = test.groupby('Name').Power.apply(lambda x: x.fillna(x.mean()))

test.Mileage = test.groupby('Name').Mileage.apply(lambda x: x.fillna(x.mean()))

test.Seats = test.groupby('Name').Seats.apply(lambda x: x.fillna(x.median()))

test.New_Price = test.groupby('Name').New_Price.apply(lambda x: x.fillna(x.mean()))



# Fill remaining missing values aggregating by brand mean and median

train.Power = train.groupby('brand').Power.apply(lambda x: x.fillna(x.mean()))

train.Mileage = train.groupby('brand').Mileage.apply(lambda x: x.fillna(x.mean()))

train.Seats = train.groupby('brand').Seats.apply(lambda x: x.fillna(x.median()))

train.New_Price = train.groupby('brand').New_Price.apply(lambda x: x.fillna(x.mean()))



test.Power = test.groupby('brand').Power.apply(lambda x: x.fillna(x.mean()))

test.New_Price = test.groupby('brand').New_Price.apply(lambda x: x.fillna(x.mean()))



# Fill remaining missing values aggregating by whole column mean

train.New_Price = train.New_Price.fillna(train.New_Price.mean())

test.New_Price = test.New_Price.fillna(test.New_Price.mean())

missing_values_table(train)
missing_values_table(test)
test[test.Power.isnull()]
test.Power.fillna(test[test.Engine.between(1900,2000)].Power.mean(), inplace=True)
# Define a function to plot the distribution of various features

def count_plot(data,col,figx,figy,rotate = 'N', order = 'Y'):

    plt.figure(figsize=(figx, figy));

    if order == 'Y':

        g = sns.countplot(x=col, data=data, order = data[col].value_counts().index)

    else:

        g = sns.countplot(x=col, data=data)

    plt.title('Distribution of %s' %col);

    if rotate == 'Y':

        plt.xticks(rotation=45);

    ax=g.axes

    for p in ax.patches:

         ax.annotate(f"{p.get_height() * 100 / data.shape[0]:.2f}%",

                     (p.get_x() + p.get_width() / 2., p.get_height()),

                     ha='center', 	# horizontal alignment

                     va='top',		# Vertical alignment

                     fontsize=10,	# Fontsize

                     color='black',	# Color set

                     rotation=0,	# Rotation type

                     xytext=(0,10),	# caption position

                     textcoords='offset points' # Caption placement

                    ) 
# Check the Year make distribution of the Training data

count_plot(train,'Year',20,5,rotate = 'Y', order = 'N')
# Check the Year make distribution of the Test data

count_plot(test,'Year',20,5,rotate = 'Y', order = 'N')
# Check the Fule type distribution of the Training data

count_plot(train,'Fuel_Type',6,6)
# Check the Fule type distribution of the Test data

count_plot(test,'Fuel_Type',6,6)
# Check the Transmission distribution of the Training data

count_plot(train,'Transmission',5,8)
# Check the Transmission distribution of the Test data

count_plot(test,'Transmission',5,8)
# Check the Transmission distribution of the Training data

count_plot(train,'Owner_Type',7,8)
# Check the Transmission distribution of the Test data

count_plot(test,'Owner_Type',7,8)
# Check the Transmission distribution of the Training data

count_plot(train,'Location',20,7)
# Check the Transmission distribution of the Test data

count_plot(test,'Location',20,7)
# Check the Transmission distribution of the Train data

count_plot(train,'Seats',10,6)
# Check the Transmission distribution of the Test data

count_plot(test,'Seats',10,6)
train[train.Seats.isin([0,2,9,10])]
train.loc[3999,'Seats'] = 5



train.Seats[train.Seats == 9] = 10
# Define function for the next set of graph distributions.

def dist_plot(data, col, bins, color, figx, figy, kde = True):

    plt.figure(figsize=(figx,figy))

    sns.distplot(data[col].values, bins=bins, color=color, kde_kws={"shade": True}, label="Low", kde=kde)

    plt.title("Histogram of %s Distribution"%col)

    plt.xlabel('%s'%col, fontsize=12)

    plt.ylabel('Vehicle Count', fontsize=12)

    plt.show();
# Check the Power distribution of the Train data

dist_plot(train,'Power', 50, 'blue', 20, 5)
# Check the Power distribution of the Test data

dist_plot(test,'Power', 50, 'blue', 20, 5)
# Check the Power distribution of the Train data

dist_plot(train,'Engine', 25, 'brown', 20, 5)
# Check the Power distribution of the Test data

dist_plot(test,'Engine', 25, 'brown', 20, 5)
# Check the Power distribution of the Train data

dist_plot(train,'Mileage', 50, 'green', 20, 5)
# Check the Power distribution of the Test data

dist_plot(test,'Mileage', 50, 'green', 20, 5)
# Check the Power distribution of the Training data

dist_plot(train,'Kilometers_Driven', 50, 'magenta', 20, 5)
# Check the Power distribution of the Test data

dist_plot(test,'Kilometers_Driven', 50, 'magenta', 20, 5)
col = 'Kilometers_Driven'

from scipy import stats

outliers = train[col][(np.abs(stats.zscore(train[col])) > 3)]

outliers
train[train.Kilometers_Driven.isin(outliers)]
train = train[~train.Kilometers_Driven.isin(outliers)]
dist_plot(train,'Kilometers_Driven', 50, 'magenta', 20, 5, kde = True)
# Record the age of the car

import datetime

train['Age'] = datetime.datetime.now().year - train['Year']

test['Age'] = datetime.datetime.now().year - test['Year']
# Record the number of words in the Full_name of the car

train['Name_length'] = train.Full_name.apply(lambda x: len(str(x).split(' ')))

test['Name_length'] = test.Full_name.apply(lambda x: len(str(x).split(' ')))
# Define categorical features

categorical_features = ['Location','Fuel_Type','Transmission','Owner_Type','Seats']



# Define function for dummy operation

def get_dummies(dataframe,feature_name):

  dummy = pd.get_dummies(dataframe[feature_name], prefix=feature_name)

  dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap

  return pd.concat([dataframe,dummy], axis = 1)



# Dummify categorical features

for i in categorical_features:

    train = get_dummies(train, i)

    test = get_dummies(test, i)

# Define function to aggregate metrics for different features

def aggregate_features(data):   

    

    aggregate_dict = {  'Age' : ['count'],

                        'Mileage' : ['sum','max','min','mean','std','median','skew'],

                        'Power' : ['sum','max','min','mean','std','median','skew'],

                        'Engine' : ['sum','max','min','mean','std','median','skew']}

    

    data_agg = data.groupby(['Name']).agg(aggregate_dict)

    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns.values]

    data_agg.reset_index(inplace=True)    

    data_agg = pd.merge(data, data_agg, on='Name', how='left')    

    return data_agg
# Create aggregated features

train = aggregate_features(train)

test = aggregate_features(test)
missing_values_table(train)
missing_values_table(test)
train.Mileage_skew = train.Mileage_skew.fillna(train.Mileage_skew.mean())

train.Power_skew = train.Power_skew.fillna(train.Power_skew.mean())

train.Engine_skew = train.Engine_skew.fillna(train.Engine_skew.mean())

train.Mileage_std = train.Mileage_std.fillna(train.Mileage_std.mean())

train.Power_std = train.Power_std.fillna(train.Power_std.mean())

train.Engine_std = train.Engine_std.fillna(train.Engine_std.mean())



test.Mileage_skew = test.Mileage_skew.fillna(test.Mileage_skew.mean())

test.Power_skew = test.Power_skew.fillna(test.Power_skew.mean())

test.Engine_skew = test.Engine_skew.fillna(test.Engine_skew.mean())

test.Mileage_std = test.Mileage_std.fillna(test.Mileage_std.mean())

test.Power_std = test.Power_std.fillna(test.Power_std.mean())

test.Engine_std = test.Engine_std.fillna(test.Engine_std.mean())
import gensim

import multiprocessing

cores = multiprocessing.cpu_count()



# Define function to add/aggregate embeddings of single token text

def word_vector(tokens, size):

    vec = np.zeros(size).reshape((1, size))

    count = 0

    for word in tokens:

        try:

            vec += model_w2v.wv[word].reshape((1, size))

            count += 1

        except KeyError:  # handling the case where the token is not in vocabulary

            continue

    if count != 0:

        vec /= count

    return vec
total_names = pd.concat([train.Full_name, test.Full_name], ignore_index=True)

tokens = total_names.apply(lambda x: x.split()) # tokenizing text

train_tokens = train.Full_name.apply(lambda x: x.split()) # tokenizing text

test_tokens = test.Full_name.apply(lambda x: x.split()) # tokenizing text

tokens_size = len(tokens)

train_tokens_size = len(train_tokens)

test_tokens_size = len(test_tokens)
model_w2v = gensim.models.Word2Vec(

            tokens,

            size=200, # desired no. of features/independent variables

            window=2, # context window size

            min_count=2,

            sg = 1, # 1 for skip-gram model

            hs = 0, # off heirarchichal softmax

            negative = 1, # for negative sampling

            workers= cores-1, # no.of cores

#             sample=.1,

            alpha=0.009, 

            min_alpha=0.0009,

#             seed=0,

#             hashfxn=hash

) 



model_w2v.train(tokens,

                total_examples= tokens_size,

                epochs=40)
wordvec_train_array = np.zeros((train_tokens_size, 200)) 

wordvec_test_array = np.zeros((test_tokens_size, 200))



for i in range(train_tokens_size):

    wordvec_train_array[i,:] = word_vector(train_tokens[i], 200)

wordvec_train_df = pd.DataFrame(wordvec_train_array)



for i in range(test_tokens_size):

    wordvec_test_array[i,:] = word_vector(test_tokens[i], 200)

wordvec_test_df = pd.DataFrame(wordvec_test_array)
wordvec_train_df.shape, wordvec_test_df.shape
model_w2v.wv.most_similar(positive="hyundai")
# Define function for multiple aggregation on a dataframe rows

def agg_df(df):

    return pd.DataFrame(

                        {'Name_sum':df.sum(axis=1),

                         'Name_mean':df.mean(axis=1),

                         'Name_std':df.std(axis=1),

                         'Name_max':df.max(axis=1),

                         'Name_min':df.min(axis=1),

                         'Name_median':df.median(axis=1),

                         'Name_skew':df.skew(axis=1)

                        }

                       )
# Add aggregated features on the Name word2vec vctors.

train = pd.concat([train, agg_df(wordvec_train_df)], axis=1) 

test = pd.concat([test, agg_df(wordvec_test_df)], axis=1) 
train.head()
features = ['Year','Kilometers_Driven','Mileage','Engine','Power','Age','Price']



# Through CORRMAT

from mlens.visualization import corrmat

corrmat(train[features].corr(), inflate=False)

plt.show();
# Engine and Power 

plt.figure(figsize=(8,6))

plt.scatter(train.Power, train.Engine, c='blue')

plt.xlabel('Power(bhp)', fontsize=12)

plt.ylabel('Engine(cc)', fontsize=12)

plt.show();
# Engine and Mileage 

plt.figure(figsize=(8,6))

plt.scatter(train.Mileage, train.Engine, c='blue')

plt.xlabel('Mileage(kmpl)', fontsize=12)

plt.ylabel('Engine(cc)', fontsize=12)

plt.show();
# Power and Price

plt.figure(figsize=(8,6))

plt.scatter(train.Power, train.Price, c='red')

plt.xlabel('Power(bhp)', fontsize=12)

plt.ylabel('Price(lacs)', fontsize=12)

plt.show();
# Engine and Price

plt.figure(figsize=(8,6))

plt.scatter(train.Engine, train.Price, c='green')

plt.xlabel('Engine(cc)', fontsize=12)

plt.ylabel('Price(lacs)', fontsize=12)

plt.show();
# Age and Price

plt.figure(figsize=(8,6))

plt.scatter(train.Age, train.Price, c='orange')

plt.xlabel('Age(years)', fontsize=12)

plt.ylabel('Price(lacs)', fontsize=12)

plt.show();
# Take backup before dropping some features

train_backup = train.copy() 

test_backup = test.copy() 



# Drop irrelevant features

drop_features = ['Location','Fuel_Type','Transmission','Owner_Type','Seats','Full_name','Name','brand']

backup_train = train.drop(drop_features, axis=1, inplace=True)

backup_test = test.drop(drop_features, axis=1, inplace=True)
# Assign values to variables for training and testing



X_train = train.drop(labels=['Price'], axis=1) # Assign all features except Price to X

y_train = np.log1p(train['Price'].values) # Convert Price to log scale

X_test = test.copy()
# Scale the train and test set before feeding to the model



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = pd.DataFrame(sc.fit_transform(X_train),columns = X_train.columns)

X_test = pd.DataFrame(sc.transform(X_test),columns = X_test.columns)
train_X = X_train.copy()

train_y = y_train.copy()

test_X = X_test.copy()



# Define LGBM function

def runLGB(train_X, train_y, val_X=None, val_y=None, test_X=None, dep=-1, seed=0, data_leaf=5):

    params = {}

    params["objective"] = "regression"

    params['metric'] = 'l2_root'

    params['boosting'] = 'gbdt'

#     params["max_depth"] = dep

#     params["num_leaves"] = 39

#     params["min_data_in_leaf"] = data_leaf

    params["learning_rate"] = 0.009

    params["bagging_fraction"] = 0.75

    params["feature_fraction"] = 0.75

    params["feature_fraction_seed"] = seed

    params["bagging_freq"] = 1

    params["bagging_seed"] = seed

#     params["lambda_l2"] = 5

#     params["lambda_l1"] = 5

    params["silent"] = True

    params["random_state"] = seed,

    num_rounds = 3000

    

    lgtrain = lgb.Dataset(train_X, label=train_y)



    if val_y is not None:

        lgtest = lgb.Dataset(val_X, label=val_y)

        model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtest], early_stopping_rounds=50, verbose_eval=100)

    else:

        lgtest = lgb.DMatrix(val_X)

        model = lgb.train(params, lgtrain, num_rounds)



    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

  

    loss = 0

    

    if val_y is not None:

        loss = sqrt(mean_squared_log_error(np.expm1(val_y), np.expm1(pred_val_y)))

        return model, loss, pred_test_y

    else:

        return model, loss, pred_test_y



## K-FOLD train



cv_scores = [] # array for keeping cv-scores for each fold.

pred_test_full = 0 # array to keep predictions of each fold.

pred_train = np.zeros(train_X.shape[0])

n_fold = 10

print(f"Building model over {n_fold} folds\n")

kf = KFold(n_splits=n_fold, shuffle=True, random_state=4)



feature_importance = pd.DataFrame()

for fold_n, (dev_index, val_index) in enumerate(kf.split(train_X, train_y)):    

    dev_X, val_X = train_X.iloc[dev_index,:], train_X.iloc[val_index,:]

    dev_y, val_y = train_y[dev_index], train_y[val_index]



    model, loss, pred_t = runLGB(dev_X, dev_y, val_X, val_y, test_X, dep=8, seed=0)

      

    pred_test_full += pred_t

    print(f"\n>>>>RMSLE for fold {fold_n+1} is: {loss}<<<<\n")

    cv_scores.append(loss)

    

    # feature importance aggregation over n folds

    fold_importance = pd.DataFrame()

    fold_importance["feature"] = X_train.columns

    fold_importance["importance"] = model.feature_importance()

    fold_importance["fold"] = fold_n + 1

    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
print(f"Mean RMSLE score over folds is: {np.mean(cv_scores)}")



# Aggregate mean prediction over 10 folds.

pred_test_full /= n_fold

pred_test_final = np.expm1(pred_test_full)
# Plot feature importance mean aggregated over 10 folds

plt.figure(figsize=(20, 20));

feature_importance = pd.DataFrame(feature_importance.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index())[:50]

sns.barplot(x="importance", y="feature", data=feature_importance);

plt.title('Feature Importance (average over folds)');
# Create submission file.

Predict_submission = pd.DataFrame(data=pred_test_final, columns=['Price'])

writer = pd.ExcelWriter('Output.xlsx', engine='xlsxwriter')

Predict_submission.to_excel(writer,sheet_name='Sheet1', index=False)

writer.save()