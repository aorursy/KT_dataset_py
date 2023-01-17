#Enivornment Setup
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
#Data Read, Data Visualization,EDA Analysis,Data Pre-Processing,Data Splitting
#Data Read

file_path = '../input/fish-market'

train=pd.read_csv(f'{file_path}/Fish.csv')
df=train.copy()
import pandas_profiling
# preparing profile report



profile_report = pandas_profiling.ProfileReport(df,minimal=True)

profile_report
df.info()
df.describe()
df.shape
df.Species.value_counts()
df.apply(lambda x: sum(x.isnull()),axis=0)
def correlation_matrix(d):

    from matplotlib import pyplot as plt

    from matplotlib import cm as cm



    fig = plt.figure(figsize=(16,12))

    ax1 = fig.add_subplot(111)

    cmap = cm.get_cmap('jet', 30)

    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)

    ax1.grid(True)

    plt.title('Fish Speciecs Market dataset features correlation\n',fontsize=15)

    labels=df.columns

    ax1.set_xticklabels(labels,fontsize=9)

    ax1.set_yticklabels(labels,fontsize=9)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels

    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])

    plt.show()



correlation_matrix(df)
#Plotting data 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px
plt.figure(figsize=(20,14))

sns.heatmap(df.corr(),annot=True,linecolor='red',linewidths=3,cmap = 'plasma')
sns.pairplot(df,diag_kind="kde")

plt.show()
i=1

plt.figure(figsize=(25,20))

for c in df.describe().columns[:]:

    plt.subplot(4,2,i)

    plt.title(f"Histogram of {c}",fontsize=10)

    plt.yticks(fontsize=12)

    plt.xticks(fontsize=12)

    plt.hist(df[c],bins=20,color='blue',edgecolor='k')

    i+=1

plt.show()
from sklearn.preprocessing import LabelEncoder





X = df.iloc[:,1:].copy()

y = df.iloc[:,0].copy()

le = LabelEncoder()

ylab= le.fit_transform(y)



labels = pd.DataFrame({"y":y,"ylabel":ylab})

labels.drop_duplicates(inplace=True)

labels = labels.sort_values(by="ylabel")

labels


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,ylab,test_size=0.2,random_state=42)

X_train.shape
X_train
y_train
pd.Series(y_train).value_counts()
dims = X_train.shape[1]

print(dims, 'dims')
from sklearn.preprocessing import RobustScaler, StandardScaler
# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#Supervised Machine Learning Algorithm:
#Using RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix
cr = classification_report(y_test,rfc_pred)
print(cr)
#For Support vector Algorithm

from sklearn.svm import SVC

model = SVC()

model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#Parameter tuning

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV



grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=1)



# May take a while!

grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
param_grid = {'C': [50,75,100,125,150], 'gamma': [1e-2,1e-3,1e-4,1e-5,1e-6], 'kernel': ['rbf']} 

grid = GridSearchCV(SVC(tol=1e-5),param_grid,refit=True,verbose=1)

grid.fit(X_train,y_train)
grid.best_estimator_
grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
#Models performance Analysis 
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

list_models=[]

list_scores=[]

x_train=sc.fit_transform(X_train)

lr=LogisticRegression(max_iter=1000)

lr.fit(X_train,y_train)

pred_1=lr.predict(sc.transform(X_test))

score_1=accuracy_score(y_test,pred_1)

list_scores.append(score_1)

list_models.append('LogisticRegression')
score_1
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,11):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    preds=knn.predict(sc.transform(X_test))

    scores=accuracy_score(y_test,preds)

    list_1.append(scores)
list_scores.append(max(list_1))

list_models.append('KNeighbors Classifier')
print(max(list_1))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(sc.transform(X_test))

score_2=accuracy_score(y_test,pred_2)

list_models.append('Randomforest Classifier')

list_scores.append(score_2)
score_2


from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

pred_3=svm.predict(sc.transform(X_test))

score_3=accuracy_score(y_test,pred_3)

list_scores.append(score_3)

list_models.append('Support vector machines')
score_3
from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(x_train,y_train)

pred_4=xgb.predict(sc.transform(X_test))

score_4=accuracy_score(y_test,pred_4)

list_models.append('XGboost')

list_scores.append(score_4)
score_4
plt.figure(figsize=(12,5))

plt.bar(list_models,list_scores)

plt.xlabel('classifiers')

plt.ylabel('accuracy scores')

plt.show()