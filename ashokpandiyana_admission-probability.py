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
data = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict.csv")
data.head()
data = data.drop("Serial No." , axis = 1)
data.describe().T
print(f"The data has {data.shape[0]} rows and {data.shape[1]} columns")
data.head()
data.isnull().sum()
data.columns
chance = []
for val in list(data['Chance of Admit ']):
    if (val >= 0.70):
        chance.append(1)
    else:
        chance.append(0)
data['admit'] = chance
data_copy = data.copy()
data_copy.drop('Chance of Admit ', axis = 1 , inplace = True)
data_copy.columns
import matplotlib.pyplot as plt
import seaborn as sns

fig = sns.distplot(data_copy['GRE Score'], kde=False)
plt.title("Distribution of GRE Scores")
plt.show()

fig = sns.distplot(data_copy['TOEFL Score'], kde=False)
plt.title("Distribution of TOEFL Scores")
plt.show()

fig = sns.distplot(data_copy['University Rating'], kde=False)
plt.title("Distribution of University Rating")
plt.show()

fig = sns.distplot(data_copy['SOP'], kde=False)
plt.title("Distribution of SOP Ratings")
plt.show()

fig = sns.distplot(data_copy['CGPA'], kde=False)
plt.title("Distribution of CGPA")
plt.show()

plt.show()
data_copy['admit'].value_counts()
fig = sns.regplot(x="GRE Score", y="TOEFL Score", data=data_copy)
plt.title("GRE Score vs TOEFL Score")
plt.show()
fig = sns.regplot(x="GRE Score", y="CGPA", data=data_copy)
plt.title("GRE Score vs CGPA")
plt.show()
fig = sns.lmplot(x="CGPA", y="LOR ", data=data_copy, hue="Research")
plt.title("GRE Score vs CGPA")
plt.show()
fig = sns.lmplot(x="GRE Score", y="LOR ", data=data_copy, hue="Research")
plt.title("GRE Score vs CGPA")
plt.show()
corr = data_copy.corr()
fig, ax = plt.subplots(figsize=(8, 8))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))
data_copy['Research'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Students Research')
ax[0].set_ylabel('Student Count')
sns.countplot('Research',data=data_copy,ax=ax[1])
ax[1].set_title('Students Research')
plt.show()
sns.factorplot('Research','admit',data=data_copy)
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))
data_copy['admit'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Admitted')
ax[0].set_ylabel('')
sns.countplot('admit',data=data_copy,ax=ax[1])
ax[1].set_title('Admitted')
plt.show()
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators = 1000,random_state = 123)
X = data_copy.drop('admit',axis = 1)
y = data_copy['admit']
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = .25,random_state = 123)
rf_model = RandomForestRegressor(n_estimators = 1000,random_state = 123)
rf_model.fit(X_train,y_train)
feature_importance = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns)), columns=['Value','Feature'])
plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))
plt.xlabel('Value',fontsize=20)
plt.ylabel('Feature',fontsize=20)
plt.title('Random Forest Feature Importance',fontsize=25)
plt.grid()
plt.ioff()
plt.tight_layout()
X = data_copy.drop('admit',axis = 1)
y = data_copy['admit']
from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
Y = lab_enc.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0) 
from sklearn.preprocessing import StandardScaler 
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor
from sklearn.metrics import mean_squared_error

models = [['DecisionTree :',DecisionTreeRegressor()],
           ['Linear Regression :', LinearRegression()],
           ['RandomForest :',RandomForestRegressor()],
           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],
           ['SVM :', SVR()],
           ['AdaBoostClassifier :', AdaBoostRegressor()],
           ['GradientBoostingClassifier: ', GradientBoostingRegressor()],
           ['Xgboost: ', XGBRegressor()],
           ['CatBoost: ', CatBoostRegressor(logging_level='Silent')],
           ['Lasso: ', Lasso()],
           ['Ridge: ', Ridge()],
           ['BayesianRidge: ', BayesianRidge()],
           ['ElasticNet: ', ElasticNet()],
           ['HuberRegressor: ', HuberRegressor()]]

for name,model in models:
    model = model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))