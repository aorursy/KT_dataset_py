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

import matplotlib.pyplot as plt

import pandas as pd

import scipy.stats as st

import seaborn as sb

from sklearn.preprocessing import MinMaxScaler,StandardScaler,PowerTransformer

from sklearn.model_selection import train_test_split,KFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,roc_auc_score

from sklearn.decomposition import PCA

import warnings

from xgboost import XGBClassifier

# from empiricaldist import Cdf,Pmf

import missingno as msno

from pprint import pprint

!pip install impyute

from impyute.imputation.cs import fast_knn

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

warnings.filterwarnings('ignore')

%matplotlib inline

heart=pd.read_csv('../input/framingham-heart-study-dataset/framingham.csv')

heart.drop(['education'],inplace=True,axis=1)

heart.head()
def remove_index(df,info=False):

    shape_0=df.shape[0]

    for i in df.columns:

        if len(df[i].unique())==shape_0:

            df.drop([i],inplace=True,axis=1)

    if (info == True):

        print (df.info())

    return df
heart=remove_index(heart,info=False)
heart_columns=heart.columns
def HandleCategory(df,info=False,describe=False):

    

    datatypes=dict(df.dtypes)

    dependentVariable=len(datatypes)-1

    countIndependentList=[]

    countIndependent=-1

    for key in datatypes:

      countIndependent+=1

      if (datatypes[key]==np.dtype('O')):

        key_new=pd.get_dummies(df[key],drop_first=True)

        df.drop([key],inplace=True,axis=1)

        df=pd.concat([df,key_new],axis=1)

        dependentVariable-=1

    if (info==True):

        pprint (df.info())

    elif (describe==True):

        pprint(df.describe())

    return df
heart=HandleCategory(heart,info=True)
pprint(heart.isnull().sum())

msno.matrix(heart)
x=heart[heart['BPMeds'].isnull()].index.tolist()
heart=heart.drop(x,axis=0)

heart.describe()
pprint(heart.isnull().sum())
x=heart[heart['heartRate'].isnull()].index.tolist()

heart=heart.drop(x,axis=0)
missin=IterativeImputer()

heart=pd.DataFrame(missin.fit_transform(heart),columns=heart_columns)
msno.matrix(heart)
counts=heart['TenYearCHD'].value_counts()

plt.figure(figsize=(10,5))

sb.barplot(counts.index, counts.values, alpha=0.8)

plt.show()
plt.figure(figsize=(20,15))

heart_corr=heart.corr()

sb.heatmap(heart_corr,cmap="Blues", vmin= -2.0, vmax=1,

           linewidth=0.1, cbar_kws={"shrink": .8},annot=True)

plt.show()
upper_tri = heart_corr.where(np.triu(np.ones(heart_corr.shape),k=1).astype(np.bool))

# print (upper_tri['prevalentStroke'])

print ([i for i in upper_tri if any(upper_tri[i]>0.70)],"columns would be dropped")

heart.drop([i for i in upper_tri if any(upper_tri[i]>0.68)],inplace=True,axis=1)
heart.describe()
from statsmodels.tools import add_constant

heart_constant = add_constant(heart)

heart_constant.head()

heart=heart_constant.copy()
# sb.set_style('darkgrid')

# sb.distplot(heart['male'])
plt.figure(figsize=(20,10),dpi=80, facecolor='gray', edgecolor='yellow')

heart.boxplot(column=[i for i in heart.columns])

plt.show()

print(heart.shape)
# plt.figure(figsize=(20,10),dpi=80, facecolor='gray', edgecolor='yellow')

# sb.boxplot(y='age',x='TenYearCHD',data=heart,whis=10)

# plt.yscale('log')  #to see data on log scale

# plt.show()
# plt.figure(figsize=(20,10),dpi=80, facecolor='gray', edgecolor='yellow')

# sb.kdeplot(data=heart['age'])

# plt.show()

# plt.figure(figsize=(20,10),dpi=80, facecolor='gray', edgecolor='yellow')

# sb.violinplot(x='age',y='heartRate',data=heart,inner=None)

# sb.despine(left=True,bottom=True)

# plt.show()
def quantile_trans(string,minn,maxx,name):

    max_threshold,min_threshold=name[str(string)].quantile([maxx,minn])

    name=name[(heart[str(string)]<max_threshold) & (name[str(string)] > min_threshold)]

    return name
def iqrFunc(string,minn,maxx,name):

    max_threshold,min_threshold=name[str(string)].quantile([maxx,minn])

    iqr=max_threshold-min_threshold

    upperLimit=max_threshold-1.5*iqr

    lowerLimit=min_threshold-1.5-iqr

    name=name[(heart[str(string)]<upperLimit) & (name[str(string)] > lowerLimit)]

    return name
TempIsolationHeart=heart
from sklearn.ensemble import IsolationForest

modelIsolation=IsolationForest(contamination=0.14,random_state=0)

modelIsolation.fit(TempIsolationHeart)

predictIsolation=modelIsolation.predict(TempIsolationHeart)
# mask=[]

# for i in range(len(predictIsolation)):

#     if (predictIsolation[i]==-1):

#         mask.append(i)

# TempIsolationHeart=TempIsolationHeart.drop(mask)

# TempIsolationHeart.describe()
power=PowerTransformer(method="yeo-johnson",standardize=True)
TempIsolationHeart=pd.DataFrame(TempIsolationHeart,columns=heart.columns)
counts=TempIsolationHeart['TenYearCHD'].value_counts()

plt.figure(figsize=(10,5))

sb.barplot(counts.index, counts.values, alpha=0.8)

plt.show()


# heart=quantile_trans('totChol',0.01,0.83,heart)

# #heart=iqrFunc('totChol',0.2,0.9,heart)

# print(heart.shape)

# plt.figure(figsize=(20,10),dpi=80, facecolor='gray', edgecolor='yellow')

# heart.boxplot(column=[i for i in heart.columns])

# plt.show()
from imblearn.combine import SMOTEENN

smt=SMOTEENN(random_state=0,sampling_strategy='minority')
from collections import Counter
heart_x=TempIsolationHeart.iloc[:,:-1].values

heart_y=TempIsolationHeart.iloc[:,-1].values

new_heart_x,new_heart_y=smt.fit_resample(heart_x,heart_y)

# new_heart_x=power.fit_transform(new_heart_x)

# sc=StandardScaler()

# columnsHeart_x=TempIsolationHeart.columns

# heart_x=sc.fit_transform(heart_x)

# pca=PCA(n_components=9)

# new_heart_x=pca.fit_transform(new_heart_x)

# plt.figure()

# plt.plot(np.cumsum(pca.explained_variance_ratio_))

# plt.xlim(0,7,1)

# plt.xlabel('Number of components')

# plt.ylabel('Cumulative explained variance')

# plt.show()



print (Counter(new_heart_y))

train_x,test_x,train_y,test_y=train_test_split(new_heart_x,new_heart_y,test_size=0.2,random_state=42)

train_y=train_y.reshape(-1,1)

test_y=test_y.reshape(-1,1)

counts=test_y

counts=pd.DataFrame(counts).value_counts()

plt.figure(figsize=(10,5))

sb.barplot(counts.index, counts.values, alpha=0.8)

plt.show()
score=[]



cv = KFold(n_splits=10, random_state=0)

maxx=0

for train_index, test_index in cv.split(train_x):

    

    regr=LogisticRegression(penalty='l2',solver='saga')

    y=regr.fit(train_x[train_index], train_y[train_index])

    prediction= y.predict(train_x[test_index])

    train_score=y.score(train_x[train_index], train_y[train_index])

    test_score=y.score(train_x[test_index],train_y[test_index])

    test_prediction=y.predict(test_x)

    accuracy_test=accuracy_score(test_y,test_prediction)

    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(test_y, test_prediction)



    #mse_test=mean_squared_error(test_y,test_prediction)

    

    if (maxx<accuracy_test):

        maxx=accuracy_test

        intercept=y.intercept_

        coef=pd.concat([pd.DataFrame(heart.columns),pd.DataFrame(np.transpose(y.coef_))], axis = 1)

        trainScore=train_score

        validationTestScore=test_score

        roc_auc=roc_auc_score(test_y, test_prediction)

        false_positive_rate=false_positive_rate1

        true_positive_rate=true_positive_rate1
print('training score:',trainScore,'\n','Test Score: ',maxx,'\n','intercept: ',intercept,'\n','coeficient: ',coef,'\n','Validation Test score:',validationTestScore,'\n','ROC-AUC Score:',roc_auc)



plt.figure(figsize=(15,10))

plt.plot(false_positive_rate1, true_positive_rate1)

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

print (confusion_matrix(test_y,test_prediction))


score=[]



cv = KFold(n_splits=10, random_state=42)

maxx=0

for train_index, test_index in cv.split(train_x):

    

    regr=RandomForestClassifier(n_estimators=30,criterion='entropy',random_state=42)

    y=regr.fit(train_x[train_index], train_y[train_index])

    prediction= y.predict(train_x[test_index])

    train_score=y.score(train_x[train_index], train_y[train_index])

    test_score=y.score(train_x[test_index],train_y[test_index])

    test_prediction=y.predict(test_x)

    accuracy_test=accuracy_score(test_y,test_prediction)

    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(test_y, test_prediction)



    #mse_test=mean_squared_error(test_y,test_prediction)

    

    if (maxx<accuracy_test):

        maxx=accuracy_test

        trainScore=train_score

        validationTestScore=test_score

        roc_auc=roc_auc_score(test_y, test_prediction)

        false_positive_rate=false_positive_rate1

        true_positive_rate=true_positive_rate1
print('training score:',trainScore,'\n','Test Score: ',maxx,'\n','intercept: ','intercept','\n','coeficient: ','coef','\n','Validation Test score:',validationTestScore,'\n','ROC-AUC Score:',roc_auc)



plt.figure(figsize=(15,10))

plt.plot(false_positive_rate1, true_positive_rate1)

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

print (confusion_matrix(test_y,test_prediction))
score=[]



cv = KFold(n_splits=10, random_state=0)

maxx=0

for train_index, test_index in cv.split(train_x):

    

    regr=make_pipeline(SVC(kernel="rbf",gamma="scale",degree=5,random_state=0))

    y=regr.fit(train_x[train_index], train_y[train_index])

    prediction= y.predict(train_x[test_index])

    train_score=y.score(train_x[train_index], train_y[train_index])

    test_score=y.score(train_x[test_index],train_y[test_index])

    test_prediction=y.predict(test_x)

    accuracy_test=accuracy_score(test_y,test_prediction)

    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(test_y, test_prediction)



    #mse_test=mean_squared_error(test_y,test_prediction)

    

    if (maxx<accuracy_test):

        maxx=accuracy_test

        trainScore=train_score

        validationTestScore=test_score

        roc_auc=roc_auc_score(test_y, test_prediction)

        false_positive_rate=false_positive_rate1

        true_positive_rate=true_positive_rate1
print('training score:',trainScore,'\n','Test Score: ',maxx,'\n','intercept: ','intercept','\n','coeficient: ','coef','\n','Validation Test score:',validationTestScore,'\n','ROC-AUC Score:',roc_auc)



plt.figure(figsize=(15,10))

plt.plot(false_positive_rate1, true_positive_rate1)

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

print (confusion_matrix(test_y,test_prediction))
score=[]



cv = KFold(n_splits=20, random_state=0)

maxx=0

for train_index, test_index in cv.split(train_x):

    

    regr=XGBClassifier(n_estimators=73,booster="gbtree",learning_rate=0.3)

    y=regr.fit(train_x[train_index], train_y[train_index])

    prediction= y.predict(train_x[test_index])

    train_score=y.score(train_x[train_index], train_y[train_index])

    test_score=y.score(train_x[test_index],train_y[test_index])

    test_prediction=y.predict(test_x)

    accuracy_test=accuracy_score(test_y,test_prediction)

    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(test_y, test_prediction)



    #mse_test=mean_squared_error(test_y,test_prediction)

    

    if (maxx<accuracy_test):

        maxx=accuracy_test

        trainScore=train_score

        validationTestScore=test_score

        roc_auc=roc_auc_score(test_y, test_prediction)

        false_positive_rate=false_positive_rate1

        true_positive_rate=true_positive_rate1
print('training score:',trainScore,'\n','Test Score: ',maxx,'\n','intercept: ','intercept','\n','coeficient: ','coef','\n','Validation Test score:',validationTestScore,'\n','ROC-AUC Score:',roc_auc)



plt.figure(figsize=(15,10))

plt.plot(false_positive_rate1, true_positive_rate1)

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

print (confusion_matrix(test_y,test_prediction))
def doPrediction(train_x,test_x,train_y,test_y,meth="logistic",LogisticPenalty='l2',):

    Meth={"logistic":LogisticRegression(penalty=LogisticPenalty)}

    score=[]



    cv = KFold(n_splits=10, random_state=42)

    maxx=0

    for train_index, test_index in cv.split(train_x):



        regr=LogisticRegression(penalty='l2',solver='saga',C=0.5,tol=0.1)

        y=regr.fit(train_x[train_index], train_y[train_index])

        prediction= y.predict(train_x[test_index])

        train_score=y.score(train_x[train_index], train_y[train_index])

        test_score=y.score(train_x[test_index],train_y[test_index])

        test_prediction=y.predict(test_x)

        accuracy_test=accuracy_score(test_y,test_prediction)

        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(test_y, test_prediction)



        #mse_test=mean_squared_error(test_y,test_prediction)



        if (maxx<accuracy_test):

            maxx=accuracy_test

            intercept=y.intercept_

            coef=pd.concat([pd.DataFrame(heart.columns),pd.DataFrame(np.transpose(y.coef_))], axis = 1)

            trainScore=train_score

            validationTestScore=test_score

            roc_auc=roc_auc_score(test_y, test_prediction)

            false_positive_rate=false_positive_rate1

            true_positive_rate=true_positive_rate1
from sklearn.manifold import TSNE

tsne=TSNE(n_components=2)

X_embedded = tsne.fit_transform(TempIsolationHeart)

sb.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=sb.color_palette("bright"))