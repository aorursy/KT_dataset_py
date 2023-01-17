# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import pandas_profiling as pp

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
#Once the data is loaded,we can view the data. Instead of viewing the entire data , we can view the first five rows

df.head()
# We can view the last five rows of the data

df.tail()
#we can view the total number of rows and columns of data

df.shape
# Now we got the information about the total size of the data . We can further explore the detailed information about each column using the info method on datadrame

df.info()
profile = pp.ProfileReport(df)

profile
# We can get the summary statistics (min,max,count...) of each column by using the describe() method

df.describe()
df.corr()
#Checking the column names

df.columns
# Rename the columns of dataframe to more meaningful

df=df.rename(columns={'age':'Age','sex':'Sex','cp':'Cp','trestbps':'Trestbps','chol':'Chol','fbs':'Fbs','restecg':'Restecg','thalach':'Thalach','exang':'Exang','oldpeak':'Oldpeak','slope':'Slope','ca':'Ca','thal':'Thal','target':'Target'})
#Recheck the column names

df.columns
# Check for missing values

#df.isnull().sum()

df.isnull().mean()
# Calculating the IQR for entire dataset, to detect outliers

Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
# Filtering the columns by removing the outliers

print((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))  )
# Try to delete the rows with outliers and check if this impacts our prediction

df_out = df[(df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))] #viewing the outliers

print(df_out)

#df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
#df_out.shape # We can see that by deleting the rows with outliers , we may lose a large amount of data
# We will now try perform imputation on these outliers , As all columns are numerical we can perform median imputation

#df.out = df[(df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))]
# We will try to check and drop duplicate values

#Incase of duplicate rows we use drop_duplicates() method

df_dup = df[df.duplicated()]

df_dup
#As we have one duplicate row , Delete the duplicated rows

df = df.drop_duplicates()

df
df['Target'].value_counts()
sns.countplot(df['Target'])

plt.show()
df.hist()
dataset = pd.get_dummies(df, columns = ['Sex', 'Cp', 'Fbs', 'Restecg', 'Exang', 'Slope', 'Ca', 'Thal'])
standardScaler = StandardScaler()

columns_to_scale = ['Age', 'Trestbps', 'Chol', 'Thalach', 'Oldpeak']

df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])
df.head()
y = df['Target']

X = df.drop(['Target'], axis = 1)
logreg = LogisticRegression()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 42)

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

print("The Score is",logreg.score(X_test,y_test))

cv_score_five = cross_val_score(logreg,X,y,cv=5)

cv_score_ten = cross_val_score(logreg,X,y,cv=10)

print("Cross validation score - Five Folds",cv_score_five)

print("Cross validation score - Ten Folds",cv_score_ten)

print("Mean cross validation score",cv_score_ten.mean())

print("Confusion Matrix",confusion_matrix(y_test,y_pred))

print("Classification Report")

print(classification_report(y_test,y_pred))
dec_clf = DecisionTreeClassifier(max_depth = 3 , random_state=1)

dec_clf.fit(X_train,y_train)

y_pred = dec_clf.predict(X_test)

print("The accuarcy score of decision tree classifier is ",accuracy_score(y_test,y_pred))

cv_dec_tree_clf = cross_val_score(dec_clf,X,y,cv=10)

print("The Cross validation score ",cv_dec_tree_clf.mean())
randomforest_classifier= RandomForestClassifier(n_estimators=10)

randomforest_classifier.fit(X_train,y_train.ravel())

y_pred = randomforest_classifier.predict(X_test)

cv_score=cross_val_score(randomforest_classifier,X,y,cv=10)

print("The accuarcy score of random tree classifier is ",accuracy_score(y_test,y_pred))

print("The mean cross validation score is ",cv_score.mean())




## Hyper Parameter Optimization



params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

    

}



## Hyperparameter optimization using RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV

import xgboost
classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X,y.ravel())

random_search.best_estimator_



classifier = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.3, gamma=0.3,

              learning_rate=0.2, max_delta_step=0, max_depth=15,

              min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)
score=cross_val_score(classifier,X,y.ravel(),cv=10)

score
score.mean()