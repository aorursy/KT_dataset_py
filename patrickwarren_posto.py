#####NOTES#####
'''
This produces one of the final submissions for team Posto.
Kaggle out of sample MAPE = 11.57115.

We:
-drop all data from 2010
-model all three countries separately
-log vehicle price for modeling (and convert back at the end)
-use a ridge regression to generate an initial prediction from our features.
-let a random forest evaluate all those features plus the ridge prediction

Final MAPE = 11.32097 after averaging predictions from our two best submissions.
'''
#packages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#file paths
data = '/kaggle/input/ihsmarkit-hackathon-june2020/train_data.csv'
out = '/kaggle/input/ihsmarkit-hackathon-june2020/oos_data.csv'
#####FUNCTIONS#####

#scatter plot
def scatter(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(16,8))
    plt.scatter(
            x,
            y
            )
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=20)
    plt.show()

#check shape of residuals, e.g. miss('MPG')
def miss(x):
    pe = 100 * (predictions - test_labels) / test_labels
    scatter(dftest[x], pe, x, 'Percent Error', mod)
    
#compare prediction from most recently run model against actuals
def compare():
    scatter(predictions, test_labels, 'Predictions', 'Actuals', mod)
#set seed
seed = 1992
geo = input()    #options: China, Germany, USA

#import data
df = pd.read_csv(data, index_col='vehicle_id')
df = df.loc[df['year'] != 2010]     #drop all obs from 2010

drops = ['Nameplate', 'Global_Sales_Sub-Segment']
df = df.drop(drops, axis=1) #drop variables we don't care about

xso = pd.read_csv(out, index_col='vehicle_id')   #out of sample
xso = xso.drop(drops, axis=1) #drop variables we don't care about

xs = df.drop(['Price_USD'], axis=1)
y = np.log(df['Price_USD'])               #predict logs and convert back later

df.columns.values

#create data transformations
#e.g. logs, trends, interactions, etc.

#variables to dummy
dummies = ['Body_Type',
           'Brand',
           'Transmission',
           'Turbo',
           'Fuel_Type',
           'PropSysDesign',
           'Registration_Type',
           'No_of_Gears',
           'month',
           'Driven_Wheels',
           'Plugin'
           ]

#dummy to capture extra high-end cars
super_lux = ['aston martin',
             'bentley',
             'bugatti',
             'ferrari',
             'fisker',
             'koenigsegg',
             'lamborghini',
             'lotus',
             'maserati',
             'maybach',
             'mclaren',
             'porsche',
             'rolls-royce'
             ]

def transformer(frame):
    frame['volume'] = frame['Length'] * frame['Height'] * frame['Width']
    frame['lnkw'] = np.log(frame['Engine_KW'])
    #strip month from date for dummies
    frame['month'] = frame.apply(lambda x: x['date'][5:7], axis=1)
    frame['No_of_Gears'] = frame['No_of_Gears'].astype(str)
    #transform generation_year to vehicle age
    frame['age'] = frame['year'] - frame['Generation_Year']
    frame['lnweight'] = np.log(frame['Curb_Weight'])
    frame['sqweight'] = frame['Curb_Weight'] ** 2
    frame['kw_weight'] = frame['Engine_KW'] / frame['Curb_Weight']
    if geo != 'USA':
        #interact super_lux with volume
        frame['lux_vol'] = frame.apply(lambda x: x['volume'] if x['Brand'] in super_lux else 0, axis=1)
    
#transform x variables, trimmed x variables, and out of sample x variables
for f in [xs, xso]:
    transformer(f)

#categories to dummies
xs = pd.get_dummies(xs, columns=dummies)
xso = pd.get_dummies(xso, columns=dummies)

#drop date, which we got monthly dummies from, and Generation_Year
xs = xs.drop(['date', 'Generation_Year'], axis=1)
xso = xso.drop(['date', 'Generation_Year'], axis=1)


#print(xs.columns.values)


#cleanup
#not all brands from the training data actually show up out of sample! let's drop those
for x in xs.columns.values:
    if x not in xso.columns.values:
        xs = xs[xs[x] == 0]    #drop those observations
        xs = xs.drop([x], axis=1) #drop the offending column

#now remove those obs from the y dataframes
for x in y.index:
    if x not in xs.index:
        y = y.drop(x)

print(len(xs.index))
print(len(y.index))
#####COUNTRY SELECTION#####
if geo == 'USA':
    xs = xs.loc[xs['country_name'] == geo]
    xso_u = xso.loc[xso['country_name'] == geo]
    xso_u = xso_u.drop('country_name', axis=1)
elif geo == 'China':
    xs = xs.loc[xs['country_name'] == geo]
    xso_c = xso.loc[xso['country_name'] == geo]
    xso_c = xso_c.drop('country_name', axis=1)
elif geo == 'Germany':
    xs = xs.loc[xs['country_name'] == geo]
    xso_g = xso.loc[xso['country_name'] == geo]
    xso_g = xso_g.drop('country_name', axis=1)
else:
    print('ERROR: please enter a valid geography')
    
xs = xs.drop('country_name', axis=1) #drop the country column
xso = xso.drop('country_name', axis=1) #drop the country column

#now remove those obs from the y dataframes
y = y[y.index.isin(xs.index)]

print(len(xs.index))
print(len(y.index))


#####split into training and testing#####
print('splitting ' + geo)

train_features,test_features, train_labels, test_labels = train_test_split(xs,
                                                                            y,
                                                                            test_size = 0.25,
                                                                            random_state = seed)
dftest = pd.DataFrame(test_features)
#####RIDGE REGRESSION#####
#For China and Germany, start with a ridge regression (doesn't help for USA)
if geo != 'USA':
    mod = 'Ridge Regression'

    #parameter grid to search
    parms = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
    parameters = {'alpha': parms}
    ridge = Ridge()
    #grid search for optimal alpha
    ridge_reg = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=25)

    #fit on trimmed training data
    ridge_reg.fit(train_features, train_labels)

    #predict
    predictions = ridge_reg.predict(test_features)
    predictions = np.exp(predictions) #undo logs
    test_labels = np.exp(test_labels) #undo logs
    #mean absolute percentage error
    errors = abs(predictions - test_labels)
    mape = 100 * (errors / test_labels)
    print('MAPE: {}%.'.format(round(np.mean(mape), 2)))

    #add ridge prediction as a column in xs and xso_*
    #we'll let the random forest consider the ridge regression predictions
    xs['ridge'] = np.exp(ridge_reg.predict(xs))
    if geo == 'Germany':
        xso_g['ridge'] = np.exp(ridge_reg.predict(xso_g))
    if geo == 'USA':
        xso_u['ridge'] = np.exp(ridge_reg.predict(xso_u))
    if geo == 'China':
        xso_c['ridge'] = np.exp(ridge_reg.predict(xso_c))
#####RANDOM FOREST#####
mod = 'Random Forest'
#parameters to search for optimal values
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf = RandomizedSearchCV(estimator = rf,
                        param_distributions = random_grid,
                        n_iter = 10,
                        cv = 5,
                        verbose = 10,
                        random_state = seed,
                        n_jobs = -1)
#fit on training data
rf.fit(train_features, train_labels);

#predict y values
predictions = rf.predict(test_features)
predictions = np.exp(predictions)           #undo logs
#mean absolute percentage error
errors = abs(predictions - test_labels)
mape = 100 * (errors / test_labels)
print('MAPE: {}%.'.format(round(np.mean(mape), 2)))
print(rf.best_params_)


#####POOLING PREDICTIONS, pt 1#####
#predictions_* and test_labels_* allow us to evaluate performance on the full sample
#(instead of country by country)
#saving the rf_* so we can predict OOS later
if geo == 'Germany':
    predictions_g = predictions
    test_labels_g = test_labels
    rf_g = rf
if geo == 'USA':
    predictions_u = predictions
    test_labels_u = test_labels
    rf_u = rf
if geo == 'China':
    predictions_c = predictions
    test_labels_c = test_labels
    rf_c = rf
#vizualize
compare()
miss('age')
#####POOL OUT OF SAMPLE PREDICTIONS#####
fpredict_g = np.exp(rf_g.predict(xso_g))
fpredict_u = np.exp(rf_u.predict(xso_u))
fpredict_c = np.exp(rf_c.predict(xso_c))
fpredict = np.concatenate([fpredict_g, fpredict_u, fpredict_c])

id_g = xso_g.index.values
id_u = xso_u.index.values
id_c = xso_c.index.values

fid = np.concatenate([id_g, id_u, id_c])
#####SUBMISSION#####
df_out = pd.DataFrame(data=fpredict, index=fid)
df_out.columns = ['Price_USD']
df_out.index.name = 'vehicle_id'
df_out.to_csv(root + 'submission.csv', index=True, header=True)