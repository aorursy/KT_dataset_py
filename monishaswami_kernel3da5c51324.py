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
dmon=pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv',parse_dates=True,index_col='date')
dyear=pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_yearly_variables.csv',parse_dates=True,index_col='date')
dyear.index.name='Date'

dyear.head()

dyear.info()
#dyear['mean_salary']=dyear['mean_salary'].astype(float)

pd.unique(dyear['mean_salary'])

dyear['mean_salary']=pd.to_numeric(dyear['mean_salary'],errors='coerce')

dyear['recycling_pct']=pd.to_numeric(dyear['recycling_pct'],errors='coerce')



from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

dyear['area_n']=encoder.fit_transform(dyear['area'])

dyear['code_n']=encoder.fit_transform(dyear['code'])

dyear.drop(['area','code'],axis=1,inplace=True)
dyear.head()

dyear.info()
dyear.describe()
dyear.isnull().sum()
nullcol=['median_salary','life_satisfaction','mean_salary','recycling_pct','population_size','number_of_jobs','area_size','no_of_houses']

#mean_of_col=[]

for col in nullcol:

    mean_of_col=dyear[col].mean()

    dyear=dyear.fillna({col:mean_of_col})

        

         

         

        # =dyear['median_salary'].mean()

#mean_life_satisfaction=dyear['life_satisfaction'].mean()

#mean_recycling_pct=dyear['recycling_pct'].mean()

#mean_population_size=dyear['population_size'].mean()

#mean_number_of_jobs=dyear['number_of_jobs'].mean()

#mean_area_size=dyer
dyear.info()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()

_=sns.boxplot(x='borough_flag',y='population_size',data=dyear)

plt.xlabel('Borough flag')

plt.ylabel('Population size')

plt.yscale('log')

plt.show()
plt.plot(dyear)

plt.yscale('log')

plt.show()

plt.figure()

plt.scatter(dyear['life_satisfaction'],dyear['mean_salary'])

plt.xlabel('life satisfaction')

plt.ylabel('mean salary')

plt.show()
_=sns.boxplot(x='borough_flag',y='life_satisfaction',data=dyear)

plt.xlabel('Borough flag')

plt.ylabel('Life satisfaction')

#plt.yscale('log')

plt.show()
_=sns.boxplot(x='area_n',y='life_satisfaction',data=dyear)

#plt.xticks(range(len(dyear['area_n'])),rotation=60)

plt.xlabel('area')

plt.ylabel('Life satisfaction')

#plt.yscale('log')

plt.show()
X=dyear.drop('borough_flag',axis=1).values

y=dyear['borough_flag']
sns.heatmap(dyear.corr(),square=True,cmap='RdYlGn')
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=21,stratify=y)



from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

y_train=y_train.to_numpy()

y_test=y_test.to_numpy()
from sklearn.neighbors import KNeighborsClassifier

# Plotting Model complexity and overfitting/underfitting

neighbors=np.arange(1,12)

train_accuracy=np.empty(len(neighbors))

test_accuracy=np.empty(len(neighbors))

for i,k in enumerate(neighbors):

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    train_accuracy[i]=knn.score(X_train,y_train)

    test_accuracy[i]=knn.score(X_test,y_test)

plt.title('Model complexity \n knn:Varying number of Neighbors')

plt.plot(neighbors,train_accuracy,label='Train Accuracy')

plt.plot(neighbors,test_accuracy,label='Test Accuracy')

plt.legend()

plt.xlabel('No. of Neighbors')

plt.ylabel('Accuracy')

plt.show()

    

    

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

param_grid={'n_neighbors':np.arange(1,50)}

knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn,param_grid,cv=5)

knn_cv.fit(X_train,y_train)

print(knn_cv.best_params_)

print(knn_cv.best_score_)

y_pred=knn_cv.predict(X_test)

r2=knn_cv.score(X_test,y_test)

print(r2)

cm=confusion_matrix(y_test,y_pred)

print('The confusion matrics is \n {}'.format(cm))

print('Classification report is \n {}'.format(classification_report(y_test,y_pred)))

sns.heatmap(cm,annot=True)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

c_space=np.logspace(-5,8,15)

param_grid={'C':c_space}

logreg=LogisticRegression()

logreg_cv=GridSearchCV(logreg,param_grid,cv=5)

logreg_cv.fit(X_train,y_train)

print('Tuned Logistic Regression parameters:{}'.format(logreg_cv.best_params_))

print('Best score is {}'.format(logreg_cv.best_score_))

y_pred2=logreg_cv.predict(X_test)

print(logreg_cv.score(X_test,y_test))
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

y_pred_prob=logreg_cv.predict_proba(X_test)[:,1]

fpr,tpr,threshold=roc_curve(y_test,y_pred_prob)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr,label='Logistic Regression')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('Logistic Regression ROC curve')

plt.show()

print('ROC AUC score is {}'.format(roc_auc_score(y_test,y_pred_prob)))

cv_score=cross_val_score(logreg_cv,X_train,y_train,cv=5,scoring='roc_auc')

print('AUC scoress computed using 5 fold cross valdation is: {}'.format(cv_score))
dyear.head()
X=dyear.drop('life_satisfaction',axis=1).values

y=dyear['life_satisfaction']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=21,stratify=y)



from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

y_train=y_train.to_numpy()

y_test=y_test.to_numpy()
print(y_train.shape)

print(X_train.shape)

print(y_test.shape)

print(y_test.shape)
from sklearn.linear_model import Ridge

alpha_space=np.logspace(-4,0,50)

ridge=Ridge(normalize=True)

param_grid={'alpha':alpha_space}

ridge_cv=GridSearchCV(ridge,param_grid,cv=5)

ridge_cv.fit(X_train,y_train)

print('Tuned Ridge Regression parameter:{}'.format(logreg_cv.best_params_))

print('Best score is {}'.format(logreg_cv.best_score_))

y_pred2=ridge_cv.predict(X_test)

print(ridge_cv.score(X_test,y_test))
from sklearn.neighbors import KNeighborsRegressor

knnr=KNeighborsRegressor()

param_grid={'n_neighbors':np.arange(1,50)}

knnr_cv=GridSearchCV(knnr,param_grid,cv=5)

knnr_cv.fit(X_train,y_train)

#y_pred=knnr.predict(X_test)

print('Tuned KNeighborsRegressor best parameters are:{}'.format(knnr_cv.best_params_))

print('Tuned KNeighborsRegressor best score is:{}'.format( knnr_cv.best_score_))

r2=knnr_cv.score(X_test,y_test)

print('R2 :{}'.format(r2))

mse=mean_squared_error(y_test,y_pred)

print('Mean square error :{}'.format(mse))
