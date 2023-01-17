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
df=pd.read_csv('../input/heart-disease-uci/heart.csv')

df.head()
df.shape
df.isnull().sum()
cols = df.columns

cols
print("# Rows in the dataset {0}".format(len(df)))

print("---------------------------------------------------")

for col in cols:

    print("# Rows in {1} with ZERO value: {0}".format(len(df.loc[df[col] ==0]),col))
df.dtypes
corrmat = df.corr()

fig = plt.figure(figsize = (16,16))

sns.heatmap(corrmat,vmax = 1,square = True,annot = True,vmin = -1)

plt.show()
df.hist(figsize = (12,12))

plt.show()
sns.barplot(x="sex",y ='age',hue ='target',data=df)

pass
X = df.drop('target',axis =1)

from sklearn.manifold import TSNE

import time 

time_start = time.time()

df_tsne = TSNE(random_state =10).fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#df_tsne
import matplotlib.patheffects as PathEffects

def fashion_scatter(x,colors):

    # Choosing a color palette with seaborn 

    num_classes = len(np.unique(colors))

    palette = np.array(sns.color_palette("deep",num_classes))

    

    # Create a scatter plot

    f = plt.figure(figsize=(8,8))

    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0],x[:,1],lw=0,s=40,c=palette[colors.astype(np.int)])

    plt.xlim(-25,25)

    plt.ylim(-25,25)

    ax.axis('off')

    ax.axis('tight')

    

    #add the label for each digit corresponding to label 

    txts = []

    

    for i in range(num_classes):

        

        # Position of each label at median of data points

        

        xtext, ytext = np.median(x[colors== i,:],axis = 0)

        txt = ax.text(xtext, ytext,str(i),fontsize=24)

        txt.set_path_effects([

            PathEffects.Stroke(linewidth=5,foreground="w"),

            PathEffects.Normal()])

        txts.append(txt)

    

    return f,ax,sc,txts

        
fashion_scatter(df_tsne,df.target)

pass
f,ax=plt.subplots(1,2,figsize=(18,8))

df['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Rate of Heart Disease')

ax[0].set_ylabel('Count')

sns.countplot('target',data=df,ax=ax[1],order=df['sex'].value_counts().index)

ax[1].set_title('Rate of Heart Disease')

plt.show()
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
X = df.drop('target',axis=1) #predictor feature columns

y = df.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state= 10)



print('Training Set:',len(X_train))

print('Test Set:',len(X_test))

print('Training labels:',len(y_train))

print('Test labels:',len(y_test))
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
param = {

           'n_estimators':[100,500,1000,1500,2000],

           'max_depth':[2,3,4,5,6,7],

        

         }

FitModel(X_train,y_train,X_test,y_test,'Random Forest',RandomForestClassifier(),param,cv=5)
param = {

           'C':[0.1,1,100,1000],

           'gamma':[0.0001,0.001,0.005,0.1,1,3,5],

           

         }

FitModel(X_train,y_train,X_test,y_test,'SVC',SVC(),param,cv=5)
X = df.drop('target',axis=1)

y = df.target

y.value_counts()
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

X_res_OS , Y_res_OS = sm.fit_resample(X,y)

pd.Series(Y_res_OS).value_counts()
print(cols)
X_res_OS = pd.DataFrame(X_res_OS,columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',

       'exang', 'oldpeak', 'slope', 'ca', 'thal'])

Y_res_OS = pd.DataFrame(Y_res_OS,columns=['target'])
X_train,X_test,y_train,y_test = train_test_split(X_res_OS,Y_res_OS,test_size = 0.1,random_state=10)

print('Training Set :',len(X_train))

print('Test Set :',len(X_test))

print('Training labels :',len(y_train))

print('Test labels :',len(y_test))

print(final_cols)

type(X_train)
from sklearn.impute import SimpleImputer 

fill = SimpleImputer(missing_values=np.nan, strategy='mean')



X_train = fill.fit_transform(X_train[final_cols])

X_test = fill.fit_transform(X_test[final_cols])
param = {

           'n_estimators':[100,500,1000,1500,2000],

           'max_depth':[2,3,4,5,6,7],

           'learning_rate':np.arange(0.01,0.1,0.01).tolist()

         }

FitModel(X_train,y_train,X_test,y_test,'XGBoost',XGBClassifier(),param,cv=5)