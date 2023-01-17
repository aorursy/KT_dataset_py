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
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.head()
print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n     :',df.columns.tolist())

print('\nMissing values    :',df.isnull().values.sum())

print('\nUnique values :  \n',df.nunique())
df.isnull().sum()
df = df.drop(['id','Unnamed: 32'],axis=1)
df.shape
df['diagnosis'].value_counts()
#df.info()
df.diagnosis = df.diagnosis.astype('category')
plt.rcParams['figure.figsize']=(12,8)

sns.set(font_scale=1.4)

sns.heatmap(df.drop('diagnosis',axis=1).corr(),cmap='coolwarm')

#sns.heatmap(df.drop('diagnosis',axis=1).drop('id',axis=1).corr(),cmap='coolwarm')

pass
plt.rcParams['figure.figsize']=(10,5)

f,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)

sns.boxplot('diagnosis',y='radius_mean',data=df,ax=ax1)

sns.boxplot('diagnosis',y='texture_mean',data=df,ax=ax2)

sns.boxplot('diagnosis',y='perimeter_mean',data=df,ax=ax3)

sns.boxplot('diagnosis',y='area_mean',data=df,ax=ax4)

sns.boxplot('diagnosis',y='smoothness_mean',data=df,ax=ax5)

f.tight_layout()



f,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)

sns.boxplot('diagnosis',y='compactness_mean',data=df,ax=ax1)

sns.boxplot('diagnosis',y='concavity_mean',data=df,ax=ax2)

sns.boxplot('diagnosis',y='concave points_mean',data=df,ax=ax3)

sns.boxplot('diagnosis',y='symmetry_mean',data=df,ax=ax4)

sns.boxplot('diagnosis',y='fractal_dimension_mean',data=df,ax=ax5)

f.tight_layout()
g = sns.FacetGrid(df,col='diagnosis',hue='diagnosis')

g.map(sns.distplot,'radius_mean',hist=False,rug=True)



g = sns.FacetGrid(df,col='diagnosis',hue='diagnosis')

g.map(sns.distplot,'texture_mean',hist=False,rug=True)



g = sns.FacetGrid(df,col='diagnosis',hue='diagnosis')

g.map(sns.distplot,'perimeter_mean',hist=False,rug=True)



g = sns.FacetGrid(df,col='diagnosis',hue='diagnosis')

g.map(sns.distplot,'area_mean',hist=False,rug=True)



g = sns.FacetGrid(df,col='diagnosis',hue='diagnosis')

g.map(sns.distplot,'smoothness_mean',hist=False,rug=True)



g = sns.FacetGrid(df,col='diagnosis',hue='diagnosis')

g.map(sns.distplot,'compactness_mean',hist=False,rug=True)



g = sns.FacetGrid(df,col='diagnosis',hue='diagnosis')

g.map(sns.distplot,'concavity_mean',hist=False,rug=True)



g = sns.FacetGrid(df,col='diagnosis',hue='diagnosis')

g.map(sns.distplot,'concave points_mean',hist=False,rug=True)



g = sns.FacetGrid(df,col='diagnosis',hue='diagnosis')

g.map(sns.distplot,'symmetry_mean',hist=False,rug=True)



g = sns.FacetGrid(df,col='diagnosis',hue='diagnosis')

g.map(sns.distplot,'fractal_dimension_mean',hist=False,rug=True)



pass
X = df.drop(labels='diagnosis',axis=1)

y = df['diagnosis']

col=X.columns

#col
X.isnull().sum()
df_norm = (X-X.mean())/(X.max()-X.min())

df_norm = pd.concat([df_norm,y],axis=1)
df_norm.head(4)
X_norm = df_norm.drop(labels='diagnosis',axis=1)

Y_norm = df_norm['diagnosis']

col=X_norm.columns

#le = LabelEncoder()

#le.fit(Y_norm)

from sklearn import preprocessing 

le = preprocessing.LabelEncoder() 

Y_norm= le.fit_transform(Y_norm)

Y_norm = pd.DataFrame(Y_norm)

#Y_norm.head()
#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_norm,Y_norm,test_size=0.2,random_state= 10)
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

    pickle.dump(grid_result,open(algo_name,'wb'))

    

    print('Best Params :',best_params)

    print('Classification Report :',classification_report(y_test,pred))

    print('Accuracy Score :'+ str(accuracy_score(y_test,pred)))

    print('Confusion Matrix : \n',cm)

param = {

           'C':[0.1,1,100,1000],

           'gamma':[0.0001,0.001,0.005,0.1,1,3,5],

           

         }

FitModel(X_train,y_train,X_test,y_test,'SVC',SVC(),param,cv=5)
param = {

           'n_estimators':[100,500,1000,1500,2000],

           'max_depth':[2,3,4,5,6,7],

        

         }

FitModel(X_train,y_train,X_test,y_test,'Random Forest',RandomForestClassifier(),param,cv=5)
param = {

           'n_estimators':[100,500,1000,1500,2000],

           'max_depth':[2,3,4,5,6,7],

           'learning_rate':np.arange(0.01,0.1,0.01).tolist()

         }

FitModel(X_train,y_train,X_test,y_test,'XGBoost',XGBClassifier(),param,cv=5)

np.random.seed()

forest = RandomForestClassifier(n_estimators=1000)

fit = forest.fit(X_train,y_train)

accuracy = fit.score(X_test,y_test)

predict = fit.predict(X_test)

cmatrix = confusion_matrix(y_test,predict)



#-------------------------------------------------------------------------------------------------#

# Perform k Fold cross- validation 



print('Accuracy of Random Forest: %s'% "{0:.2%}".format(accuracy))
# Feature importance 

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]



print("Feature ranking")

for f in range(X.shape[1]):

    print("Feature %s (%f)" % (list(X)[f],importances[indices[f]]))
feat_imp = pd.DataFrame({'Feature':list(X),

                        'Gini importance':importances[indices]})

plt.rcParams['figure.figsize']=(12,12)

sns.set_style('whitegrid')

ax = sns.barplot(x='Gini importance',y='Feature',data=feat_imp)

ax.set(xlabel='Gini Importance')
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

X_res , Y_res = sm.fit_resample(X_norm,Y_norm)

#pd.Series(Y_res).value_counts()
#from sklearn.model_selection import train_test_split

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_res,Y_res,test_size=0.2,random_state= 10)
def FitModel_1(X_train_1,y_train_1,X_test_1,y_test_1,algo_name,algorithm,gridSearchParams,cv):

    np.random.seed(10)

    

    grid = GridSearchCV(

         estimator = algorithm,

         param_grid = gridSearchParams,

         cv=cv,scoring='accuracy',verbose=1,n_jobs=-1)

        

    grid_result = grid.fit(X_train_1,y_train_1)

    best_params = grid_result.best_params_

    pred = grid_result.predict(X_test_1)

    cm = confusion_matrix(y_test_1,pred)

    print(pred)

    

    print('Best Params :',best_params)

    print('Classification Report :',classification_report(y_test_1,pred))

    print('Accuracy Score :'+ str(accuracy_score(y_test_1,pred)))

    print('Confusion Matrix : \n',cm)
param = {

           'n_estimators':[100,500,1000,1500,2000],

           'max_depth':[2,3,4,5,6,7],

        

         }

FitModel(X_train_1,y_train_1,X_test_1,y_test_1,'Random Forest Balanced',RandomForestClassifier(),param,cv=5)
feat_imp.index = feat_imp.Feature

feat_to_keep = feat_imp.iloc[1:15].index

type(feat_to_keep),feat_to_keep
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

X_res , Y_res = sm.fit_resample(X_norm[feat_to_keep],Y_norm)

#pd.Series(Y_res).value_counts()
#from sklearn.model_selection import train_test_split

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_res,Y_res,test_size=0.2,random_state= 10)
def FitModel_2(X_train_2,y_train_2,X_test_2,y_test_2,algo_name,algorithm,gridSearchParams,cv):

    np.random.seed(10)

    

    grid = GridSearchCV(

         estimator = algorithm,

         param_grid = gridSearchParams,

         cv=cv,scoring='accuracy',verbose=1,n_jobs=-1)

        

    grid_result = grid.fit(X_train_2,y_train_2)

    best_params = grid_result.best_params_

    pred = grid_result.predict(X_test_2)

    cm = confusion_matrix(y_test_2,pred)

    print(pred)

    

    print('Best Params :',best_params)

    print('Classification Report :',classification_report(y_test_2,pred))

    print('Accuracy Score :'+ str(accuracy_score(y_test_2,pred)))

    print('Confusion Matrix : \n',cm)
param = {

           'n_estimators':[100,500,1000,1500,2000],

           'max_depth':[2,3,4,5,6,7],

        

         }

FitModel(X_train_2,y_train_2,X_test_2,y_test_2,'Random Forest FS',RandomForestClassifier(),param,cv=5)
loaded_model = pickle.load(open("Random Forest Balanced","rb"))
pred1 = loaded_model.predict(X_test)

loaded_model.best_params_