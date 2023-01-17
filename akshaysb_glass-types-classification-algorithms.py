import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('../input/glass.csv')

df.head()
df.info()
df.isnull().sum()
## Let us check for five point summary of our data

df.describe()
features=df.columns[:-1]

cols=list(features)
for i in cols:

    skewness=df[i].skew()

    print('Skewness for ',i,'= ',skewness)
plt.figure(figsize=(10,10))

df.boxplot()
Feat=[]

U_C_L=[]

L_C_L=[]

for i in cols:

    q_25=np.percentile(df[i],25)

    q_75=np.percentile(df[i],75)

    IQR=q_75-q_25

    const=1.5*IQR

    UCL=round((q_75+const),4)

    LCL=round((q_25+const),4)

    Feat.append(i)

    U_C_L.append(UCL)

    L_C_L.append(LCL)
limits=pd.DataFrame({'Features':Feat,'Upper Limit':U_C_L,'Lower Limit':L_C_L})

limits
def outliers(df):

    outlier_indices=[]

    

    # iterate over features(columns)

    for col in df.columns.tolist():

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        

        # Interquartile rrange (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 2 )

    print(multiple_outliers)

    return multiple_outliers

print('The dataset contains %d observations with more than 2 outliers' %(len(outlier_indices)))  
outlier_indices = outliers(df[cols])

outlier_indices
plt.figure(figsize=(8,8))

cor_mat=df[cols].corr()

cor_mat

sns.heatmap(cor_mat,annot=True)
plt.figure(figsize=(50,50))

print(df.groupby(['Type'])['RI'].mean())
(df.groupby(['Type'])['Na'].mean()).plot(kind='bar')

plt.xlabel('Type of glass')

plt.ylabel('"Na" content')

plt.title('Sodium Content in various types of glass')
(df.groupby(['Type'])['Mg'].mean()).plot(kind='bar')

plt.xlabel('Type of glass')

plt.ylabel('"Mg" content')

plt.title('Magnesium Content in various types of glass')
(df.groupby(['Type'])['Al'].mean()).plot(kind='bar')

plt.xlabel('Type of glass')

plt.ylabel('"Al" content')

plt.title('Aluminum Content in various types of glass')
(df.groupby(['Type'])['Si'].mean()).plot(kind='bar')

plt.xlabel('Type of glass')

plt.ylabel('"Si" content')

plt.title('Silicon Content in various types of glass')
(df.groupby(['Type'])['K'].mean()).plot(kind='bar')

plt.xlabel('Type of glass')

plt.ylabel('"K" content')

plt.title('Potassium Content in various types of glass')
(df.groupby(['Type'])['Ca'].mean()).plot(kind='bar')

plt.xlabel('Type of glass')

plt.ylabel('"Ca" content')

plt.title('Calcium Content in various types of glass')
(df.groupby(['Type'])['Ba'].mean()).plot(kind='bar')

plt.xlabel('Type of glass')

plt.ylabel('"Ba" content')

plt.title('Barium Content in various types of glass')
(df.groupby(['Type'])['Fe'].mean()).plot(kind='bar')

plt.xlabel('Type of glass')

plt.ylabel('"Fe" content')

plt.title('Iron Content in various types of glass')
df1=df.drop(outlier_indices).reset_index(drop=True)
df1.shape
for i in cols:

    skewness=df1[i].skew()

    print('Skewness for ',i,'= ',skewness)
y=df1['Type']

x=df1.drop('Type',axis=1)

from sklearn.model_selection import train_test_split
from scipy import stats as st
for i in x.columns:

    x[i],lambda_val=st.boxcox(x[i]+1.0)
for i in x.columns:

    skewness=x[i].skew()

    print('Skewness for ',i,'= ',skewness)
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

xs=ss.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
x_train=ss.fit_transform(x_train)

x_test=ss.transform(x_test)
x_test.shape,y_test.shape
def model_eval(algo,xtrain,ytrain,xtest,ytest):

    algo.fit(xtrain,ytrain)

    ytrain_pred=algo.predict(xtrain)



    from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,classification_report



    print('Confusion matrix for train:','\n',confusion_matrix(ytrain,ytrain_pred))



    print('Overall accuracy of train dataset:',accuracy_score(ytrain,ytrain_pred))

    

    print('Classification matrix for train data','\n',classification_report(ytrain,ytrain_pred))



    ytest_pred=algo.predict(xtest)



    print('Test data accuracy:',accuracy_score(ytest,ytest_pred))



    print('Confusion matrix for test data','\n',confusion_matrix(ytest,ytest_pred))

    

    print('Classification matrix for train data','\n',classification_report(ytest,ytest_pred))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

model_eval(rfc,x_train,y_train,x_test,y_test)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint

rfc=RandomForestClassifier(random_state=3)

params={'n_estimators':sp_randint(50,200),'max_features':sp_randint(1,24),'max_depth':sp_randint(2,10),

       'min_samples_split':sp_randint(2,20),'min_samples_leaf':sp_randint(1,20),'criterion':['gini','entropy']}

rs=RandomizedSearchCV(rfc,param_distributions=params,n_iter=500,cv=3,scoring='accuracy',random_state=3,

                      return_train_score=True)

rs.fit(xs,y)
rfc_best_parameters=rs.best_params_

print(rfc_best_parameters)
rfc1=RandomForestClassifier(**rfc_best_parameters)

model_eval(rfc1,x_train,y_train,x_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

model_eval(knn,x_train,y_train,x_test,y_test)
knn_rs=KNeighborsClassifier()



params={'n_neighbors':sp_randint(1,30),'p':sp_randint(1,6)}



rs1=RandomizedSearchCV(knn_rs,param_distributions=params,cv=3,return_train_score=True,random_state=3,n_iter=500)



rs1.fit(xs,y)
knn_best_parameters=rs1.best_params_

print(knn_best_parameters)
knn1=KNeighborsClassifier(**knn_best_parameters)

model_eval(knn1,x_train,y_train,x_test,y_test)
import lightgbm as lgb

lgbm=lgb.LGBMClassifier()

model_eval(lgbm,x_train,y_train,x_test,y_test)
from scipy.stats import uniform as sp_uniform

params={'n_estimator':sp_randint(50,200),'max_depth':sp_randint(2,15),'learning_rate':sp_uniform(0.001,0.5),

       'num_leaves':sp_randint(20,50)}

lgbm_rs=lgb.LGBMClassifier()

rs_lgbm=RandomizedSearchCV(lgbm_rs,param_distributions=params,cv=3,random_state=3,n_iter=500,n_jobs=-1)

rs_lgbm.fit(xs,y)
lgbm_best_parameters=rs_lgbm.best_params_

print(lgbm_best_parameters)
lgbm_1=lgb.LGBMClassifier(**lgbm_best_parameters)

model_eval(lgbm_1,x_train,y_train,x_test,y_test)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(solver='liblinear')

model_eval(lr,x_train,y_train,x_test,y_test)
from sklearn.ensemble import VotingClassifier

lr=LogisticRegression(solver='liblinear')

rfc1=RandomForestClassifier(**rfc_best_parameters)

knn1=KNeighborsClassifier(**knn_best_parameters)

lgbm_1=lgb.LGBMClassifier(**lgbm_best_parameters)
clf=VotingClassifier(estimators=[('lr',lr),('knn',knn1),('rfc',rfc1),('lgbm',lgbm_1)],voting='hard')

model_eval(clf,x_train,y_train,x_test,y_test)
clf_sv=VotingClassifier(estimators=[('lr',lr),('knn',knn1),('rfc',rfc1),('lgbm',lgbm_1)],voting='soft')

model_eval(clf_sv,x_train,y_train,x_test,y_test)
clf_sv1=VotingClassifier(estimators=[('lr',lr),('knn',knn1),('rfc',rfc1),('lgbm',lgbm_1)],voting='soft',weights=[5,5,1,1])

model_eval(clf_sv1,x_train,y_train,x_test,y_test)
clf_sv2=VotingClassifier(estimators=[('lr',lr),('knn',knn1)],voting='soft',weights=[5,4])

model_eval(clf_sv2,x_train,y_train,x_test,y_test)