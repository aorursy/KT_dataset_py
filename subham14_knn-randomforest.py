# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import seaborn as sns

import matplotlib.pyplot as plt





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path='/kaggle/input/pima-indians-diabetes-database/diabetes.csv'
data=pd.read_csv(path)
data.head()
data.columns
data.describe()
data.Outcome.unique()
sns.countplot(data.Outcome)
f, axes = plt.subplots( figsize=(15, 9), sharex=True)



sns.distplot(data.Insulin , color="red")

plt.show()

f, axes = plt.subplots( figsize=(15, 9), sharex=True)

sns.distplot(data.Age)

f, axes = plt.subplots( figsize=(15, 9), sharex=True)

sns.distplot( data.Pregnancies , color="olive",)
f, axes = plt.subplots( figsize=(15, 9))

sns.distplot( data.DiabetesPedigreeFunction , color="black")
f, axes = plt.subplots(2, 2, figsize=(40, 18), sharex=True)





sns.distplot( data.SkinThickness , color="olive", ax=axes[0, 0])

sns.distplot(  data.Glucose, color="gold", ax=axes[0, 1])

sns.distplot( data.BloodPressure , color="teal", ax=axes[1, 0])

sns.distplot( data.BMI , color="red", ax=axes[1, 1])

print("Skewness: %f" % data.Insulin.skew())

print("Kurtosis: %f" % data.Insulin.kurt())
cols=[]

for i in data.columns:

    cols.append(str(i))

    
data[cols].skew()
data[cols].kurt()
f, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(data=data.BloodPressure)
print((data.BloodPressure<1).sum())

sns.boxplot(data=data.Pregnancies)
s=data[data.Pregnancies>13]

s
sns.boxplot(data=data.Insulin)
sns.boxplot(data=data.Glucose)
g=data[data.Glucose==0]

g
data.isnull().sum()


data_copy = data.copy(deep = True)

data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
import missingno as msno

p=msno.bar(data_copy)
sns.boxplot(data=data.Glucose)
data.Glucose = data.Glucose.fillna(data.Glucose.mean())
sns.boxplot(data=data.BloodPressure,orient = 'v',color='y')
data.BloodPressure = data.BloodPressure.fillna(data.BloodPressure.median())
sns.boxplot(data=data.SkinThickness,orient = 'v',color='r')
data.SkinThickness = data.SkinThickness.fillna(data.SkinThickness.mean())
sns.boxplot(data=data.Insulin,orient = 'v',color='b')
data.Insulin = data.Insulin.fillna(data.Insulin.median())
sns.boxplot(data=data.BMI,orient = 'v',color='g')
data.BMI = data.BMI.fillna(data.BMI.mean())
import missingno as msno

p=msno.bar(data)
sns.pairplot(data, hue = 'Outcome')
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(data.corr(), annot=True,cmap ='RdYlGn')


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(data.drop(["Outcome"],axis = 1),),

        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age'])



Y=data.Outcome
from sklearn.feature_selection import f_classif

from sklearn.feature_selection import SelectKBest
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)
print(fit.scores_)
features = fit.transform(X)
features
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.20, random_state = 0)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



knn = KNeighborsClassifier()



param_grid = {'n_neighbors':[5,10,15,25,30,50]}



grid_knn = GridSearchCV(knn,param_grid,scoring='accuracy',cv = 10,refit = True)
grid_knn.fit(x_train,y_train)

print("Best Score ==> ", grid_knn.best_score_)

print("Tuned Paramerers ==> ",grid_knn.best_params_)

print("Accuracy on Train set ==> ", grid_knn.score(x_train,y_train))

print("Accuracy on Test set ==> ", grid_knn.score(x_test,y_test))
from sklearn.model_selection import train_test_split

x_train_alova,x_test_alova,y_train_alova,y_test_alova = train_test_split(features,Y,test_size = 0.20, random_state = 0)
grid_knn.fit(x_train_alova,y_train_alova)

print("Best Score ==> ", grid_knn.best_score_)

print("Tuned Paramerers ==> ",grid_knn.best_params_)

print("Accuracy on Train set ==> ", grid_knn.score(x_train_alova,y_train_alova))

print("Accuracy on Test set ==> ", grid_knn.score(x_test_alova,y_test_alova))
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
model = LogisticRegression(solver='lbfgs')

rfe = RFE(model, 4)

fit = rfe.fit(X, Y)

print("Num Features: %d" % fit.n_features_)

print("Selected Features: %s" % fit.support_)

print("Feature Ranking: %s" % fit.ranking_)
X_new =  pd.DataFrame(sc_X.fit_transform(data.drop(["Outcome","Insulin","BloodPressure","SkinThickness","Age"],axis = 1),),

        columns=['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction' ])
x_train_rfe,x_test_rfe,y_train_rfe,y_test_rfe = train_test_split(X_new,Y,test_size = 0.20, random_state = 0)
grid_knn.fit(x_train_rfe,y_train_rfe)

print("Best Score ==> ", grid_knn.best_score_)

print("Tuned Paramerers ==> ",grid_knn.best_params_)

print("Accuracy on Train set ==> ", grid_knn.score(x_train_rfe,y_train_rfe))

print("Accuracy on Test set ==> ", grid_knn.score(x_test_rfe,y_test_rfe))
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()



param_grid = {'n_estimators':[200,500,1000],

              'max_depth':[2,3,4,5],

              'min_samples_leaf':[0.2,0.4,0.6,0.8,1],

              'max_features':['auto','sqrt'],

              'criterion':['gini','entropy']}



grid_rfc = RandomizedSearchCV(rfc,param_grid,n_iter=20,scoring='accuracy',cv = 10,refit = True)
grid_rfc.fit(x_train_rfe,y_train_rfe)

print("Best Score ==> ", grid_rfc.best_score_)

print("Tuned Paramerers ==> ",grid_rfc.best_params_)

print("Accuracy on Train set ==> ", grid_rfc.score(x_train_rfe,y_train_rfe))

print("Accuracy on Test set ==> ", grid_rfc.score(x_train_rfe,y_train_rfe))