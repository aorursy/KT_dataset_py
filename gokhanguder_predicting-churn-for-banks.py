# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# data analysis libraries:
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# to display all columns:
pd.set_option('display.max_columns', None)

#timer
import time
from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} done in {:.0f}s".format(title, time.time() - t0))

# Importing modelling libraries
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
df = pd.read_csv('/kaggle/input/predicting-churn-for-bank-customers/Churn_Modelling.csv')
df.info()
df = df.astype({"Age": float})
df.head()
df = df.astype({"Age": float})
df.shape
df['Exited'].value_counts()
df['Gender'].value_counts()
df['Geography'].value_counts()
df['NumOfProducts'].value_counts()
df['HasCrCard'].value_counts()
df['IsActiveMember'].value_counts()
df.groupby('IsActiveMember')['Exited'].value_counts()
print('The ratio of retention of active members',end=": ")
print(round(4416/5151*100,2))
print('The ratio of retention of passive members',end=": ")
print(round(3547/4849*100,2))
df['Gender'].value_counts()
df.groupby('Gender')['Exited'].value_counts()
print('The ratio of retention of men',end=": ")
print(round(4559/5457*100,2))
print('The ratio of retention of women',end=": ")
print(round(3404/4543*100,2))
df['Geography'].value_counts()
df.groupby('Geography')['Exited'].value_counts()
print('The ratio of retention of customers from France',end=": ") 
print(round(4204/5014*100,2))
print('The ratio of retention of  customers from Spain',end=": ")
print(round(2064/2477*100,2))
print('The ratio of retention of customers from Germany',end=": ")
print(round(1695/2509*100,2))
df['NumOfProducts'].value_counts()
df.groupby('NumOfProducts')['Exited'].value_counts()
print('The ratio of retention of customers with 2 products',end=": ") 
print(round(4242/4590*100,2))
print('The ratio of retention of  customers with 1 product',end=": ")
print(round(3675/5084*100,2))
df['HasCrCard'].value_counts()
df.groupby('HasCrCard')['Exited'].value_counts()
print('The ratio of retention of customers with credit card',end=": ") 
print(round(2332/2945*100,2))
print('The ratio of retention of  customers without credit card',end=": ")
print(round(5631/7055*100,2))
df.describe().T
g = sns.FacetGrid(df, col = "Exited")
g.map(sns.distplot, "Age", bins = 25)
plt.show()
g = sns.FacetGrid(df, col = "Exited")
g.map(sns.distplot, "CreditScore", bins = 25)
plt.show()
g = sns.FacetGrid(df, col = "Exited")
g.map(sns.distplot, "Tenure", bins = 25)
plt.show()
g = sns.FacetGrid(df, col = "Exited")
g.map(sns.distplot, "Balance", bins = 25)
plt.show()
g = sns.FacetGrid(df, col = "Exited")
g.map(sns.distplot, "EstimatedSalary", bins = 25)
plt.show()
# Let's visualize the correlations between numerical features of data.
fig, ax = plt.subplots(figsize=(12,6)) 
sns.heatmap(df.iloc[:,1:len(df)].corr(), annot = True, fmt = ".2f", linewidths=0.5, ax=ax) 
plt.show()
df.drop(['RowNumber'], axis = 1, inplace = True)
df.drop(['Surname'], axis = 1, inplace = True)
df.drop(['CustomerId'], axis = 1, inplace = True)
df.head()
for d in [df]:
    d["Gender"]=d["Gender"].map(lambda x: 0 if x=='Female' else 1)
df.head()
df = pd.get_dummies(df, columns=["Tenure"])
df = pd.get_dummies(df, columns=["NumOfProducts"])
df = pd.get_dummies(df, columns=["Geography"])
df.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
predictors = df.drop(['Exited'], axis=1)
target = df["Exited"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)
x_train.shape
y_train.shape
y_train.head()
x_val.shape
r=1309
models = [LogisticRegression(random_state=r),GaussianNB(), KNeighborsClassifier(),
          SVC(random_state=r,probability=True),DecisionTreeClassifier(random_state=r),
          RandomForestClassifier(random_state=r), GradientBoostingClassifier(random_state=r),
          XGBClassifier(random_state=r), MLPClassifier(random_state=r),
          CatBoostClassifier(random_state=r,verbose = False)]
names = ["LogisticRegression","GaussianNB","KNN","SVC",
             "DecisionTree","Random_Forest","GBM","XGBoost","Art.Neural_Network","CatBoost"]
print('Default model validation accuracies for the train data:', end = "\n\n")
for name, model in zip(names, models):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val) 
    print(name,':',"%.3f" % accuracy_score(y_pred, y_val))
results = []
print('10 fold cross validation accuracy scores of the default models:', end = "\n\n")
for name, model in zip(names, models):
    kfold = KFold(n_splits=10, random_state=1001)
    cv_results = cross_val_score(model, predictors, target, cv = kfold, scoring = "accuracy")
    results.append(cv_results)
    print("{}: {} ({})".format(name, "%.3f" % cv_results.mean() ,"%.3f" %  cv_results.std()))
# Tuning by Cross Validation  
rf_params = {"max_features": ["log2","Auto","None"],
                "min_samples_split":[2,3,5],
                "min_samples_leaf":[1,3,5],
                "bootstrap":[True,False],
                "n_estimators":[50,100,150],
                "criterion":["gini","entropy"]}
rf = RandomForestClassifier()
rf_cv_model = GridSearchCV(rf, rf_params, cv = 5, n_jobs = -1, verbose = 2)
rf_cv_model.fit(x_train, y_train)
rf_cv_model.best_params_
rf = RandomForestClassifier(bootstrap = True, criterion = 'entropy' , max_features = 'log2', min_samples_leaf = 3, min_samples_split = 3,
 n_estimators = 100)
rf_tuned = rf.fit(x_train,y_train)
y_pred = rf_tuned.predict(x_val) 
acc_rf = round(accuracy_score(y_pred, y_val) * 100, 2) 
print(acc_rf)
predictions = y_pred
output = pd.DataFrame({ 'Exited': predictions }) 
output.to_csv('submission.csv', index=False)
output.head()
output.describe().T
output["Exited"].value_counts()
Retention_Rate = 1734/2000*100
print(str(Retention_Rate)+'%')
