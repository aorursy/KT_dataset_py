# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from dateutil import parser

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 

from sklearn.svm import SVC 

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.model_selection import GridSearchCV

import pickle

from lightgbm import LGBMClassifier

import warnings

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv('../input/heart.csv')

data.head()
data.shape
data.isnull().sum()
cols = data.columns

cols
print("# Rows in the dataset {0}".format(len(data)))

print("---------------------------------------------------")

for col in cols:

    print("# Rows in {1} with ZERO value: {0}".format(len(data.loc[data[col] ==0]),col))
data.dtypes
print('Rows     :',data.shape[0])

print('Columns  :',data.shape[1])

print('\nFeatures :\n     :',data.columns.tolist())

print('\nMissing values    :',data.isnull().values.sum())

print('\nUnique values :  \n',data.nunique())
data.describe().T
data.hist(figsize = (12,12))

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

data['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Rate of Heart Disease')

ax[0].set_ylabel('Count')

sns.countplot('target',data=data,ax=ax[1],order=data['sex'].value_counts().index)

ax[1].set_title('Rate of Heart Disease')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

data['cp'].value_counts().plot.pie(explode=[0,0.05,0.05,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Chest Pain Type')

ax[0].set_ylabel('Count')

sns.countplot('cp',data=data,ax=ax[1],order=data['cp'].value_counts().index)

ax[1].set_title('Chest Pain Type')

plt.show()
sns.barplot(x="sex",y ='age',hue ='target',data=data)

pass
#Creating subplots

plt.figure(figsize=(15, 12))

#Subplot1 

plt.subplot(2,2,1)

plt.title('Distribution of Age')

sns.distplot(data['age'], rug = True)



#Subplot2

plt.subplot(2,2,2)

plt.title('Distribution of Resting Blood Pressure')

sns.distplot(data['trestbps'], rug = True)



#Subplot3

plt.subplot(2,2,3)

plt.title('Distribution of Cholesterol')

sns.distplot(data['chol'], rug = True)





#Subplot4

plt.subplot(2,2,4)

plt.title('Distribution of Max Hear Rate')

sns.distplot(data['thalach'], rug = True)

plt.ioff()

#plt.show()
fig,ax = plt.subplots(nrows=4, ncols=2, figsize=(18,18))

plt.suptitle('Violin Plots',fontsize=24)

sns.violinplot(x="ca", data=data,ax=ax[0,0],palette='Set3')

sns.violinplot(x="trestbps", data=data,ax=ax[0,1],palette='Set3')

sns.violinplot (x ='chol', data=data, ax=ax[1,0], palette='Set3')

sns.violinplot(x='fbs', data=data, ax=ax[1,1],palette='Set3')

sns.violinplot(x='restecg', data=data, ax=ax[2,0], palette='Set3')

sns.violinplot(x='thalach', data=data, ax=ax[2,1],palette='Set3')

sns.violinplot(x='exang', data=data, ax=ax[3,0],palette='Set3')

sns.violinplot(x='age', data=data, ax=ax[3,1],palette='Set3')

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot("sex","trestbps",hue="target", data=data,split=True)

plt.subplot(2,2,2)

sns.violinplot("sex","chol",hue="target", data=data,split=True)

plt.subplot(2,2,3)

sns.violinplot("sex","thalach",hue="target", data=data,split=True)

plt.subplot(2,2,4)

sns.violinplot("sex","fbs",hue="target", data=data,split=True)

#ax[0].set_title('Sex and trestbps vs target')

#ax[0].set_yticks(range(0,110,10))

#sns.violinplot("Sex","Age", hue="Survived", data=data,split=True,ax=ax[1])

#ax[1].set_title('Sex and Age vs Survived')

#ax[1].set_yticks(range(0,110,10))

plt.ioff()

plt.show()
pd.crosstab(data.age,data.target).plot(kind='bar',figsize=(20,6))

plt.title('Heart Disease Vs Age')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('HeartDiseaseAge.png')

plt.ioff()
# Bar chart for age with sorted index

# Changing title fontsize -> We have to use alternative matplotlib set_title function

plot = data[data.target == 1].age.value_counts().sort_index().plot(kind = "bar", figsize=(15,5), fontsize = 15)

plot.set_title("Heart disease: Age distribution", fontsize = 15)

plt.ioff()
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)

plt.scatter(x=data.age[data.target==1],y=data.thalach[data.target==1],c='red')

plt.scatter(x=data.age[data.target==0],y=data.thalach[data.target==0],c='green')

plt.xlabel('Age')

plt.ylabel('Max Heart Rate')

plt.legend(['Disease','No Disease'])



plt.subplot(1,3,2)

plt.scatter(x=data.age[data.target==1],y=data.chol[data.target==1],c='red')

plt.scatter(x=data.age[data.target==0],y=data.chol[data.target==0],c='green')

plt.xlabel('Age')

plt.ylabel('Cholesterol')

plt.legend(['Disease','No Disease'])



plt.subplot(1,3,3)

plt.scatter(x=data.age[data.target==1],y=data.trestbps[data.target==1],c='red')

plt.scatter(x=data.age[data.target==0],y=data.trestbps[data.target==0],c='green')

plt.xlabel('Age')

plt.ylabel('Resting Blood Pressure')

plt.legend(['Disease','No Disease'])



plt.tight_layout()

plt.ioff()
corrmat = data.corr()

fig = plt.figure(figsize = (16,16))

sns.heatmap(corrmat,vmax = 1,square = True,annot = True,vmin = -1)

plt.show()
cols
final_cols = cols

final_cols = list(final_cols)

final_cols.remove('ca')

final_cols.remove('cp')

final_cols.remove('exang')

final_cols.remove('fbs')

final_cols.remove('restecg')

final_cols.remove('sex')

final_cols.remove('slope')

final_cols.remove('target')

final_cols.remove('thal')

final_cols
X = data.drop('target',axis=1) #predictor feature columns

y = data.target

y.value_counts()
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

X_res_OS , Y_res_OS = sm.fit_resample(X,y)

pd.Series(Y_res_OS).value_counts()
X_res_OS = pd.DataFrame(X_res_OS,columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',

       'exang', 'oldpeak', 'slope', 'ca', 'thal'])

Y_res_OS = pd.DataFrame(Y_res_OS,columns=['target'])
X_train,X_test,y_train,y_test = train_test_split(X_res_OS,Y_res_OS,test_size = 0.1,random_state=10)

print('Training Set :',len(X_train))

print('Test Set :',len(X_test))

print('Training labels :',len(y_train))

print('Test labels :',len(y_test))
final_cols
from sklearn.impute import SimpleImputer 

fill = SimpleImputer(missing_values=np.nan, strategy='mean')



X_train = fill.fit_transform(X_train[final_cols])

X_test = fill.fit_transform(X_test[final_cols])
def FitModel(X_train,y_train,X_test,y_test,algo_name,algorithm,gridSearchParams,cv):

    np.random.seed(10)

    

    grid = GridSearchCV(

         estimator = algorithm,

         param_grid = gridSearchParams,

         cv=cv,scoring='accuracy',verbose=1,n_jobs=-1)

        

    grid_result = grid.fit(X_train,y_train)

    best_params = grid_result.best_params_

    pred = grid_result.predict(X_test)

    cm = confusion_matrix(y_test,pred)

    print(pred)

    

    print('Best Params :',best_params)

    print('Classification Report :',classification_report(y_test,pred))

    print('Accuracy Score :'+ str(accuracy_score(y_test,pred)))

    print('Confusion Matrix : \n',cm)
# Creating Regularization penalty

penalty = ['l1','l2']



# Create regularization hyperparameter space 

C = np.logspace(0,4,10)



# Create hyperparameter options 

hyperparameters = dict(C=C,penalty = penalty)



FitModel(X_train,y_train,X_test,y_test,'LogisticRegression',LogisticRegression(),hyperparameters,cv=5)
param = {

           'n_estimators':[100,500,1000,1500,2000],

           'max_depth':[2,3,4,5,6,7],

           'learning_rate':np.arange(0.01,0.1,0.01).tolist()

         }

FitModel(X_train,y_train,X_test,y_test,'XGBoost',XGBClassifier(),param,cv=5)
