import re

import numpy as np

import pandas as pd

from scipy import stats

from scipy.stats import normaltest



import seaborn as sn

import matplotlib.pyplot as plt

import matplotlib.style as style

get_ipython().run_line_magic('matplotlib', 'inline')

style.use('seaborn-colorblind')



from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")

test_index = df_test['PassengerId']



df = df_train.append(df_test, ignore_index=True, sort=True) 

#df = df.drop(['Cabin', 'PassengerId', 'Ticket'], axis=1)
df.head(5)
print(df.isnull().sum())
df.describe()
#------------------#

#FEATURE ENGINEERING

#------------------#
#Extract, Clean, and Combine titles

def get_titles(name):

    

    #compile the regex for possible titles

    #to use, call this funciton on the pandas column that needs sorting

    titles = re.compile('Mr\.|Mrs\.|Miss\.|Mlle\.|Ms\.|Mme\.|Lady\.|Countess\.|Capt\.|Col\.|Don\.|Major\.|Sir\.|Jonkheer\.|Dona\.|Master\.|Dr\.|Rev\.')

    

    Title = []

    #search through each name in column to find titles and append to list

    for x in name:

        

        if(titles.search(x)):

            m = titles.search(x)

            Title.append(m.group())

            

        else:

            Title.append('None')

    #list titles in index order

    return Title



df['Title'] = get_titles(df['Name'])



# Combine strange titles into one title

# Mlle means mademoiselle and Mme means madam, pretty cool!, a Jonkheer is untitled nobility too.

df['Title'] = df.Title.replace(['Countess.', 'Dona.', 'Jonkheer.', 'Don.', 'Capt.', 'Major.', 'Col.', 'Dr.', 'Rev.'], 'Uncommon' )

df['Title'] = df.Title.replace('Sir.', 'Mr.')

df['Title'] = df.Title.replace(['Mme.', 'Lady.'], 'Mrs.')

df['Title'] = df.Title.replace(['Mlle.', 'Ms.'], 'Miss.')



# Find the number of unique titles

df.Title.value_counts()
# Graphing survivability of each Title by age

grid = sn.FacetGrid(df, col='Survived', row='Title', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
#Predict missing ages using regression

#One hot encode categorical variables using dummy variables

df1 = pd.get_dummies(data=df, columns=['Pclass', 'Sex', 'Embarked', 'Title'])

df1 = df1.drop(['Survived', 'Cabin', 'PassengerId', 'Ticket'], axis=1)



#replace NaN value for 'Fare'

df1['Fare'].fillna(df1['Fare'].mean(), inplace=True)



#create index to use when combining values later on

df1['index_col'] = df.index



#Scale numerical data between 0 and 1

#Don't need to do this for tree based estimators

#scaler = MinMaxScaler(feature_range=(0,1))

#df1[['Fare']] = scaler.fit_transform(df1[['Fare']])



#Split into [x,y] train and test

train = df1[df1['Age'].notnull()].drop('Name', axis=1)

x_train = train.drop('Age', axis=1)

y_train = train['Age']



test = df1[df1['Age'].isnull()].drop('Name', axis=1)

x_test = test.drop('Age', axis=1)



##Random Forest##

from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=1000, random_state=1234)

rf.fit(x_train, y_train)



#Get Accuracy metric and assign results to y_test

acc = rf.score(x_train, y_train)

y_test = rf.predict(x_test)



#Merge back predicted age values into new dataframe 

x_test['Age'] = y_test



age_pred = x_test[['index_col', 'Age']]

age_known = train[['index_col', 'Age']]



#combine known and predicted ages, sort by index, and merge into new dataframe

age_corrected = age_known.append(age_pred).sort_index()

df2 = df.drop(['Age'], axis = 1)

df2['Age'] = age_corrected['Age']



#impute 2 remaining missing values in the 'Embarked' column and 1 missing value in 'Fare' column

df2['Embarked'] = df2.Embarked.replace(np.nan, 'S', regex=True)

df2['Embarked'].fillna(df2['Embarked'].mode(), inplace=True)

df2['Fare'].fillna(df2['Fare'].mean(), inplace=True)
# Get the count of each unique cabin value, then convert this series into a pandas dataframe

# If a count is greater than 1 the cabin is shared between tickets, shared cabins are assigned a different value

# Create a dictionary of cabins to their respective values

cab_values = df2.Cabin.value_counts()

cab_values = pd.DataFrame({'Cabin':cab_values.index, 'Count':cab_values.values})

cab_values['Value'] = cab_values.Count.replace([2,3,4,5,6], 2)

cab_dict = dict(zip(cab_values.Cabin, cab_values.Value))



# Apply the cabin dictionary to a new column in the original dataframe

# 0: No cabin, 1: single cabin, 2: shared cabin

df2['Cabin_value'] = df2['Cabin'].map(cab_dict).fillna(0).astype(int)



# Find Family Size

df2['Family_size'] = df2['Parch'] + df2['SibSp'] + 1



# drop non predictive columns

#df2 = df2.drop(['Cabin', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1)
# Data Distribution, Probability Plots and Transformations: Fare

fig,ax = plt.subplots(ncols=2,nrows=3)

fig.set_size_inches(15,15)



# Need to impute 0 values in order to apply boxcox transformations.

# Impute the smallest non zero value divided by 2

m = min(i for i in df2['Fare'] if i > 0)/2

df2['Fare'] = df2.Fare.replace(0,m)



# Visualizations, Distribution and Q-Q plots

sn.distplot(df2['Fare'], ax=ax[0][0])

stats.probplot(df2["Fare"], dist='norm', fit=True, plot=ax[0][1])

sn.distplot(stats.boxcox(df2['Fare'], 0), ax=ax[1][0])

stats.probplot(stats.boxcox(df2["Fare"], 0), dist='norm', fit=True, plot=ax[1][1])

sn.distplot(np.cbrt(df2['Fare']), ax=ax[2][0])

stats.probplot(np.cbrt(df2["Fare"]), dist='norm', fit=True, plot=ax[2][1])



# Create transformed column in original dataframe

df2['log_fare'] = stats.boxcox(df2['Fare'], 0)



# Checking to see if transformation decreased skewness using D'Agnostino's K^2 Test

stat, p = normaltest(df2['Fare'])

stat1, p1 = normaltest(df2['log_fare'])
# Data Distributions, Probability Plots and Transformations: Age

fig,ax = plt.subplots(ncols=2,nrows=3)

fig.set_size_inches(15,15)



# Visualizations, Distributions and Q-Q plots

sn.distplot(df2['Age'], ax=ax[0][0])

stats.probplot(df2['Age'], dist='norm', fit=True, plot=ax[0][1])

sn.distplot(stats.boxcox(df2['Age'], 0), ax=ax[1][0])

stats.probplot(stats.boxcox(df2['Age'], 0), dist='norm', fit=True, plot=ax[1][1])

sn.distplot(np.cbrt(df2['Age']), ax=ax[2][0])

stats.probplot(np.cbrt(df2['Age']), dist='norm', fit=True, plot=ax[2][1])



#Age is relatively normally distributed, it requires no Transformation
#Scale continuous data (age and fare) between 0 and 1 to improve predictive accuracy

scaler = MinMaxScaler(feature_range=(0,1))



df2[['log_fare']] = scaler.fit_transform(df2[['log_fare']])

df2[['Age']] = scaler.fit_transform(df2[['Age']])



# drop non predictive columns

df2 = df2.drop(['Cabin', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'PassengerId'], axis=1)



#Correlation Heatmap

cormat= df2[:].corr()

mask = np.array(cormat)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(10,10)

sn.heatmap(data=cormat, cmap="BuPu", vmax=0.5, mask=mask, square=True, annot=True, cbar=True)



#One hot encode categorical variables using dummies

df2 = pd.get_dummies(data=df2, columns=['Embarked','Pclass','Sex','Title','Cabin_value','Family_size'])
#Seperate data to original format

data_test = df2[test_index.min()-1:]

data = df2[:test_index.min()-1]



# Split training data into Train and Test, Train = 70% Test = 30%

x = data.drop(['Survived'], axis=1)

y = data['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# LogisticRegression #

lr = LogisticRegression()

lr.fit(x_train, y_train)



y_pred = lr.predict(x_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr.score(x_test, y_test)))
#tuned xgboost model

import xgboost as xgb

dtrain = xgb.DMatrix(x_train, label=y_train)

dtest = xgb.DMatrix(x_test, label=y_test)
params = {

    "max_depth":6,

    "min_child_weight":1,

    'eta':.3,

    'subsample':1,

    'colsample_bytree':1,

    'objective':'binary:logistic',

}



params['eval_metric']='rmse'

num_boost_round=999



model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, 'Test')],

    early_stopping_rounds=10

)



print("Best RMSE: {:.2f} with {} rounds".format(

        model.best_score,

        model.best_iteration+1))
cv_results = xgb.cv(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    seed=42,

    nfold=5,

    metrics={'rmse'},

    early_stopping_rounds=10

)



cv_results
cv_results['test-rmse-mean'].min()
gridsearch_params = [

    (max_depth, min_child_weight)

    for max_depth in range (9,12)

    for min_child_weight in range (5,8)

]



min_rmse = float("Inf")

best_params = None

for max_depth, min_child_weight in gridsearch_params:

    print ('CV with max_depth={}, min_child_weight={}'.format(

                max_depth,

                min_child_weight))

    params['max_depth']=max_depth

    params['min_child_weight']=min_child_weight

    

    cv_results=xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'rmse'},

        early_stopping_rounds=10

    )

    

    mean_rmse = cv_results['test-rmse-mean'].min()

    boost_rounds = cv_results['test-rmse-mean'].argmin()

    print('\tRMSE {} for {} rounds'.format(mean_rmse, boost_rounds))

    if mean_rmse < min_rmse:

        min_rmse = mean_rmse

        best_params = (max_depth, min_child_weight)



print("Best params: {}, {}, RMSE: {}".format(best_params[0],

best_params[1], min_rmse))
params['max_depth']=10

params['min_child_weight']=5
gridsearch_params = [

    (subsample, colsample)

    for subsample in [i/10. for i in range(7,11)]

    for colsample in [i/10. for i in range(7,11)]

]



min_rmse = float("Inf")

best_params = None



for subsample, colsample in reversed(gridsearch_params):

    print("CV with subsample={}, colsample={}".format(

        subsample,

        colsample))

    params['subsample'] = subsample

    params['colsample_bytree'] = colsample

    

    cv_results = xgb.cv(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        seed=42,

        nfold=5,

        metrics={'rmse'},

        early_stopping_rounds=10

    )

    

    mean_rmse = cv_results['test-rmse-mean'].min()

    boost_rounds = cv_results['test-rmse-mean'].argmin()

    print('\tRMSE {} for {} rounds'.format(mean_rmse, boost_rounds))

    if mean_rmse < min_rmse:

        min_rmse = mean_rmse

        best_params = (subsample,colsample)

        

print("Best params: {}, {}, RMSE: {}".format(best_params[0],

                                            best_params[1], min_rmse))
params['subsample'] = 1.

params['colsample_bytree'] = 1.
min_rmse = float('Inf')

best_params = None



for eta in [.3, .2, .1, .05, .01, .005]:

    print('CV with eta={}'.format(eta))

    

    params['eta'] = eta

    

    %time cv_results = xgb.cv(params, dtrain, num_boost_round=num_boost_round, seed=42, nfold=5, metrics=['rmse'], early_stopping_rounds=10)

   

    mean_rmse = cv_results['test-rmse-mean'].min()

    boost_rounds = cv_results['test-rmse-mean'].argmin()

    print('\tRMSE {} for {} rounds\n'.format(mean_rmse, boost_rounds))

    if mean_rmse < min_rmse:

        min_rmse = mean_rmse

        best_params = eta



print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
params['eta'] = 0.1
model = xgb.train(

    params,

    dtrain,

    num_boost_round=num_boost_round,

    evals=[(dtest, "Test")],

    early_stopping_rounds=10

)



print("Best RMSE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))
preds = model.predict(dtest)



bin = []

for x in preds:

    if x >= .5:

        bin.append(1)

    else:

        bin.append(0)



accuracy_score(y_test, bin)
x_test_submission = data_test.drop(['Survived'], axis=1)

y_test_submission = data_test['Survived']



dtest_submission = xgb.DMatrix(x_test_submission, label=y_test_submission)
preds = model.predict(dtest_submission)



submission = []

for x in preds:

    if x >= .5:

        submission.append(1)

    else:

        submission.append(0)
submission_df = pd.DataFrame({'PassengerId': test_index, 'Survived': submission})



submission_df.to_csv('submission.csv', index = False)