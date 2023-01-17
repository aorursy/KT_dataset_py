# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



%matplotlib inline 

#present the figure in program lines without typing show()

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns
dt_1 = pd.read_csv('/kaggle/input/loanprediction/train_ctrUa4K.csv')

print('Print the first 5 recirds in the dataset')

print(dt_1.head())

print(' ')

print('Basic descriptive statistics of all the variables')

print(dt_1.describe())

print(' ')

print('Present the features attributes')

print(dt_1.info())
dt_1.hist(bins=50, figsize=(15, 10))

plt.show()
dt_obj = dt_1.select_dtypes(include='object')

#print(dt_obj.head())



#print(dt_obj.iloc[:, 1:].columns)



for col in dt_obj.iloc[:, 1:].columns:

    print(sns.countplot(x=col, data=dt_obj))

    plt.show()

#sns.countplot(x='Gender', data=dt_obj)

#sns.countplot(x='Married', data=dt_obj)

#sns.countplot(x='Dependents', data=dt_obj)

#sns.countplot(x='Education', data=dt_obj)

#sns.countplot(x='Self_Employed', data=dt_obj)

#sns.countplot(x='Property_Area', data=dt_obj)

#sns.countplot(x='Loan_Status', data=dt_obj)

#plt.show()
#print(dt_Y['LoanAmount'][dt_Y['LoanAmount'].isnull()==False])

#print(dt_Y['Loan_Amount_Term'])
dt_Y = dt_1[dt_1['Loan_Status']=='Y']

dt_N = dt_1[dt_1['Loan_Status']=='N']



sns.distplot(dt_Y['ApplicantIncome'], label='Loan_Statue = Y')

sns.distplot(dt_N['ApplicantIncome'], label='Loan_Statue = N')

plt.legend()

plt.show()

sns.distplot(dt_Y['CoapplicantIncome'], label='Loan_Statue = Y')

sns.distplot(dt_N['CoapplicantIncome'], label='Loan_Statue = N')

plt.legend()

plt.show()

sns.distplot(dt_Y['LoanAmount'][dt_Y['LoanAmount'].isnull()==False], label='Loan_Statue = Y')

sns.distplot(dt_N['LoanAmount'][dt_N['LoanAmount'].isnull()==False], label='Loan_Statue = N')

plt.legend()

plt.show()

#ValueError: cannot convert float NaN to integer => Not sure why



sns.distplot(dt_Y['Loan_Amount_Term'], label='Loan_Statue = Y')

sns.distplot(dt_N['Loan_Amount_Term'], label='Loan_Statue = N')

plt.legend()

plt.show()
for col in dt_obj.iloc[:, 1:6].columns:

    print(sns.countplot(x=col, data=dt_obj, hue='Loan_Status'))

    plt.show()
def fill_mode(x):

    dt_1[x].fillna(dt_1[x].mode()[0], inplace=True)
#dt_1['Gender'].fillna(dt_1['Gender'].mode()[0], inplace=True)

fill_mode('Gender')

fill_mode('Married')

fill_mode('Dependents')

fill_mode('Self_Employed')

fill_mode('LoanAmount')

fill_mode('Loan_Amount_Term')

fill_mode('Credit_History')
#Make sure all the null have been filled

for columns in dt_1.columns:

    print('Variable', columns, '/with missing entry:', sum(dt_1[columns].isnull()))
dt_obj = dt_1.select_dtypes(include='object')



from sklearn.preprocessing import OneHotEncoder, LabelEncoder



labelencoder = LabelEncoder()

for col in dt_obj.iloc[:, 1:].columns:

    dt_obj[col]= labelencoder.fit_transform(dt_obj[col])



print(dt_obj.head())
#print(dt_obj.iloc[:, 1:7])
onehotencoder = OneHotEncoder(categorical_features = 'all')

data_ohe=onehotencoder.fit_transform(dt_obj.iloc[:, 1:7]).toarray()

dt_objx = pd.DataFrame(data_ohe)

print(dt_1.select_dtypes(exclude='object').head())

print('')

print(dt_obj.iloc[:, 7].head()) #Target_variable

print('')

print(dt_objx.head())
dt_1.head()

dt_1x = dt_1.select_dtypes(exclude='object')

#dt_1x.merge(dt_objx)

X = dt_1x.reset_index().merge(dt_objx.reset_index(), left_on='index', right_on='index')

y = dt_obj.iloc[:, 7]
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier





clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)

scores = cross_val_score(clf, X, y, cv=5)

print("Decision Tree Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#print(scores.mean())

#Default score for Decision Tree: Mean accuracy of self.predict(X) wrt. y. 

#accuracy = # of correct / # of prediction



clrf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

scores = cross_val_score(clrf, X, y, cv=5)

print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Default score for Random Forest: Mean accuracy of self.predict(X) wrt. y. 

from sklearn.model_selection import RandomizedSearchCV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 700, num = 10)]

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



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model

rf_random.fit(X, y)
rf_random.best_params_
clrf_grid = RandomForestClassifier(n_estimators=86, min_samples_split=2, min_samples_leaf= 4, max_features= 'sqrt', max_depth=10, random_state=0, bootstrap= True)

scores = cross_val_score(clrf_grid, X, y, cv=5)

print("Grid Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Default score for Random Forest: Mean accuracy of self.predict(X) wrt. y. 
from xgboost import XGBClassifier

from sklearn.model_selection import KFold
xgb = XGBClassifier()

#kfold = KFold(n_splits=5, random_state=7)

results = cross_val_score(xgb, X, y, cv=5)

print("XGBoost Accuracy: %.2f (+/- %.2f)" % (results.mean(), results.std()* 2))
test = pd.read_csv('/kaggle/input/loanprediction/test_lAUu6dG.csv')



def tfill_mode(x):

    test[x].fillna(test[x].mode()[0], inplace=True)



tfill_mode('Gender')

tfill_mode('Married')

tfill_mode('Dependents')

tfill_mode('Self_Employed')

tfill_mode('LoanAmount')

tfill_mode('Loan_Amount_Term')

tfill_mode('Credit_History')
test_obj = test.select_dtypes(include='object')



labelencoder = LabelEncoder()

for col in dt_obj.iloc[:, 1:6].columns:

    test_obj[col]= labelencoder.fit_transform(test_obj[col])



print(test_obj.head())
onehotencoder = OneHotEncoder(categorical_features = 'all')

test_ohe=onehotencoder.fit_transform(test_obj.iloc[:, 1:7]).toarray()

test_objx = pd.DataFrame(test_ohe)



testx = test.select_dtypes(exclude='object')

test_X = testx.reset_index().merge(test_objx.reset_index(), left_on='index', right_on='index')

#print(test_X)
clrf_grid.fit(X, y)



result_RF = clrf_grid.predict(test_X)

SJ_submitt=pd.DataFrame({"Loan_ID": test['Loan_ID'], "Loan_Status":result_RF})

print(SJ_submitt.head())
pd.DataFrame(SJ_submitt).to_csv("submit_SJ.csv", index=False)