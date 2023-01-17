# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as st



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')


df.head()
df.info()
pd.set_option('display.max_columns',30)

df.describe()
df1=df.drop('ID',axis=1)
df1['default.payment.next.month'].value_counts()
df1.corr()
cat=['SEX','EDUCATION','MARRIAGE','PAY_0',

'PAY_2',

'PAY_3',

'PAY_4',

'PAY_5',

'PAY_6',

'default.payment.next.month']

for i in cat:

    sns.countplot(df1[i],hue=df1['default.payment.next.month'])

    plt.show()
df1['PAY_2'].value_counts()
num=['LIMIT_BAL','AGE',

'BILL_AMT1',

'BILL_AMT2',

'BILL_AMT3',

'BILL_AMT4',

'BILL_AMT5',

'BILL_AMT6',

'PAY_AMT1',

'PAY_AMT2',

'PAY_AMT3',

'PAY_AMT4',

'PAY_AMT5',

'PAY_AMT6']

for i in num:

    sns.boxplot(y=df[i],x=df['default.payment.next.month'])

    plt.show()
x=df1.drop('default.payment.next.month',axis=1)

y=df1['default.payment.next.month']




from statsmodels.stats.outliers_influence import variance_inflation_factor



vif=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]



pd.DataFrame(vif,index=x.columns)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split,RandomizedSearchCV,KFold,cross_val_score

from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,classification_report,confusion_matrix
def mod_eval(algo,x,y):

    cv1= cross_val_score(algo,x,y,scoring='roc_auc',cv=10)

    cv2=cross_val_score(algo,x,y,scoring='accuracy',cv=10)

    

    print('10-fold auc_score',np.mean(cv1))

    print('10-fold accuracy',np.mean(cv2))

    

    
def rand_search(algo,params):

    rs=RandomizedSearchCV(algo,param_distributions=params,random_state=0,n_jobs=-1,n_iter=100,scoring='roc_auc',cv=5)

    mod=rs.fit(x,y)

    print(mod.best_score_)

    return mod.best_params_
rfc_params={'n_estimators':st.randint(50,300),

    'criterion':['gini','entropy'],

    'max_depth':st.randint(2,20),

    'min_samples_split':st.randint(2,100),

    'min_samples_leaf':st.randint(2,100)}

lgb_params={ 'num_leaves':st.randint(31,60),

   'max_depth':st.randint(2,20),

    'learning_rate':st.uniform(0,1),

    'n_estimators':st.randint(50,300),

    'min_split_gain':st.uniform(0,0.3)}
rbp=rand_search(RandomForestClassifier(),rfc_params)
lbp=rand_search(LGBMClassifier(),lgb_params)
models={'Logistic Regression':LogisticRegression(solver='liblinear'),'Random Forest':RandomForestClassifier(**rbp),

       'Light GBM(Boosting)':LGBMClassifier(**lbp),'Gausian Naive Bayes':GaussianNB()

       }
for i in models.keys():

    print(i,'\n')

    mod_eval(models[i],x,y)
from imblearn.over_sampling import SMOTE
sm=SMOTE(sampling_strategy=0.5,random_state=7)

x_sm,y_sm=sm.fit_resample(x,y)

print(x_sm.shape,y_sm.shape)
y_sm.value_counts()
def rand_search_sm(algo,params):

    rs=RandomizedSearchCV(algo,param_distributions=params,random_state=0,n_jobs=-1,n_iter=100,scoring='roc_auc',cv=10)

    mod=rs.fit(x_sm,y_sm)

    print(mod.best_score_)

    return mod.best_params_
rbp_sm=rand_search_sm(RandomForestClassifier(),rfc_params)
lbp_sm=rand_search_sm(LGBMClassifier(),lgb_params)
models_sm={'Logistic Regression':LogisticRegression(solver='liblinear'),'Random Forest':RandomForestClassifier(**rbp_sm),

       'Light GBM(Boosting)':LGBMClassifier(**lbp_sm),'Gausian Naive Bayes':GaussianNB()

       }
for i in models_sm.keys():

    print(i,'\n')

    mod_eval(models_sm[i],x_sm,y_sm)
def model_eval(algo,x,y):

    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=3)



    mod=algo.fit(x_train,y_train)



    train_pred=mod.predict(x_train)

    train_prob=mod.predict_proba(x_train)[:,1]



    print('overall accuracy -Train: ',accuracy_score(y_train,train_pred))

    print('confusion matrix:\n',confusion_matrix (y_train,train_pred))

    print('AUC-train:',roc_auc_score(y_train,train_prob))



    test_pred=mod.predict(x_test)

    test_prob=mod.predict_proba(x_test)[:,1]



    print('overall accuracy -Test: ',accuracy_score(y_test,test_pred))

    print('confusion matrix:\n',confusion_matrix (y_test,test_pred))

    print('AUC-Test:',roc_auc_score(y_test,test_prob))

    print('Classification Report \n',classification_report(y_test,test_pred))



    fpr,tpr,th=roc_curve(y_test,test_prob)

    fig,ax=plt.subplots()

    plt.plot(fpr,tpr)

    plt.plot(fpr,fpr)
model_eval(RandomForestClassifier(**rbp),x,y)
model_eval(RandomForestClassifier(**rbp_sm),x_sm,y_sm)
model_eval(LGBMClassifier(**lbp),x,y)
model_eval(LGBMClassifier(**lbp_sm),x_sm,y_sm)