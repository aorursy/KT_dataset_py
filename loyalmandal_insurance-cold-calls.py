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
import pandas as pd

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from scipy import stats

from scipy.stats import skew

from scipy.stats import mode

from scipy.optimize import minimize

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,GradientBoostingClassifier, VotingClassifier

from sklearn.naive_bayes import GaussianNB
train = pd.read_csv("../input/carinsurance/carInsurance_train.csv")

test = pd.read_csv("../input/carinsurance/carInsurance_test.csv")
train.describe()
train.shape
train.head()
import matplotlib.pyplot as plt

import seaborn as sns
# First check out correlations among numeric features

# Heatmap is a useful tool to get a quick understanding of which variables are important

colormap = plt.cm.viridis

cor = train.corr()

cor = cor.drop(['Id'],axis=1).drop(['Id'],axis=0)

plt.figure(figsize=(12,12))

sns.heatmap(cor,vmax=0.8,cmap=colormap,annot=True,fmt='.2f',square=True,annot_kws={'size':10},linecolor='black',linewidths=0.1)
train.dtypes
# Next, pair plot some important features

imp_feats = ['CarInsurance','Age','Balance','HHInsurance', 'CarLoan','NoOfContacts','DaysPassed','PrevAttempts']

sns.pairplot(train[imp_feats], palette = "viridis", size=2.5)

plt.show()
# Take a further look at Age

facet = sns.FacetGrid(train, hue='CarInsurance',size=5,aspect=3,palette='seismic')

facet.map(plt.hist,'Age',bins=30,alpha=0.5,normed=True)

facet.set(xlim=(0,train.Age.max()+10))

facet.add_legend()
# Next check out categorical features

cat_feats = train.select_dtypes(include=['object']).columns

plt_feats = cat_feats[(cat_feats!= 'CallStart') & (cat_feats!='CallEnd')]



for feature in plt_feats:

    plt.figure(figsize=(10,6))

    sns.barplot(feature,'CarInsurance', data=train,palette='Set2')
# Check outliers

# From the pairplot, we can see there is an outlier with extreme high balance. Drop that obs here.

train[train['Balance']>80000]

train = train.drop(train[train.index==1742].index)
# merge train and test data here in order to impute missing values all at once

all=pd.concat([train,test],keys=('train','test'))

all.drop(['CarInsurance','Id'],axis=1,inplace=True)

print(all.shape)
total = all.isnull().sum()

pct = total/all.isnull().count()

NAs = pd.concat([total,pct],axis=1,keys=('Total','Pct'))

NAs[NAs.Total>0].sort_values(by='Total',ascending=False)
all_df = all.copy()



# Fill missing outcome as not in previous campaign

all_df[all_df['DaysPassed']==-1].count()

all_df.loc[all_df['DaysPassed']==-1,'Outcome']='NoPrev'



# Fill missing communication with none 

all_df['Communication'].value_counts()

all_df['Communication'].fillna('None',inplace=True)



# Fill missing education with the most common education level by job type

all_df['Education'].value_counts()



# Create job-education level mode mapping

edu_mode=[]

job_types = all_df.Job.value_counts().index

for job in job_types:

    mode = all_df[all_df.Job==job]['Education'].value_counts().nlargest(1).index

    edu_mode = np.append(edu_mode,mode)

edu_map=pd.Series(edu_mode,index=all_df.Job.value_counts().index)



# Apply the mapping to missing eductaion obs

for j in job_types:

    all_df.loc[(all_df['Education'].isnull()) & (all_df['Job']==j),'Education'] = edu_map.loc[edu_map.index==j][0]

all_df['Education'].fillna('None',inplace=True)



# Fill missing job with none

all_df['Job'].fillna('None',inplace=True)



# Double check if there is still any missing value

all_df.isnull().sum().sum()
# First simplify some client features



# Create age group based on age bands

all_df['AgeBand']=pd.cut(all_df['Age'],5)

print(all_df['AgeBand'].value_counts())



all_df.loc[(all_df['Age']>=17) & (all_df['Age']<34),'AgeBin'] = 1

all_df.loc[(all_df['Age']>=34) & (all_df['Age']<49),'AgeBin'] = 2

all_df.loc[(all_df['Age']>=49) & (all_df['Age']<65),'AgeBin'] = 3

all_df.loc[(all_df['Age']>=65) & (all_df['Age']<80),'AgeBin'] = 4

all_df.loc[(all_df['Age']>=80) & (all_df['Age']<96),'AgeBin'] = 5

all_df['AgeBin'] = all_df['AgeBin'].astype(int)



# Create balance groups

all_df['BalanceBand']=pd.cut(all_df['Balance'],5)

print(all_df['BalanceBand'].value_counts())

all_df.loc[(all_df['Balance']>=-3200) & (all_df['Balance']<17237),'BalanceBin'] = 1

all_df.loc[(all_df['Balance']>=17237) & (all_df['Balance']<37532),'BalanceBin'] = 2

all_df.loc[(all_df['Balance']>=37532) & (all_df['Balance']<57827),'BalanceBin'] = 3

all_df.loc[(all_df['Balance']>=57827) & (all_df['Balance']<78122),'BalanceBin'] = 4

all_df.loc[(all_df['Balance']>=78122) & (all_df['Balance']<98418),'BalanceBin'] = 5

all_df['BalanceBin'] = all_df['BalanceBin'].astype(int)



all_df = all_df.drop(['AgeBand','BalanceBand','Age','Balance'],axis=1)



# Convert education level to numeric 

all_df['Education'] = all_df['Education'].replace({'None':0,'primary':1,'secondary':2,'tertiary':3})
# Next create some new communication Features. This is the place feature engineering coming into play



# Get call length

all_df['CallEnd'] = pd.to_datetime(all_df['CallEnd'])

all_df['CallStart'] = pd.to_datetime(all_df['CallStart'])

all_df['CallLength'] = ((all_df['CallEnd'] - all_df['CallStart'])/np.timedelta64(1,'m')).astype(float)

all_df['CallLenBand']=pd.cut(all_df['CallLength'],5)

print(all_df['CallLenBand'].value_counts())



# Create call length bins

all_df.loc[(all_df['CallLength']>= 0) & (all_df['CallLength']<11),'CallLengthBin'] = 1

all_df.loc[(all_df['CallLength']>=11) & (all_df['CallLength']<22),'CallLengthBin'] = 2

all_df.loc[(all_df['CallLength']>=22) & (all_df['CallLength']<33),'CallLengthBin'] = 3

all_df.loc[(all_df['CallLength']>=33) & (all_df['CallLength']<44),'CallLengthBin'] = 4

all_df.loc[(all_df['CallLength']>=44) & (all_df['CallLength']<55),'CallLengthBin'] = 5

all_df['CallLengthBin'] = all_df['CallLengthBin'].astype(int)

all_df = all_df.drop('CallLenBand',axis=1)



# Get call start hour

all_df['CallStartHour'] = all_df['CallStart'].dt.hour

print(all_df[['CallStart','CallEnd','CallLength','CallStartHour']].head())



# Get workday of last contact based on call day and month, assuming the year is 2016

all_df['LastContactDate'] = all_df.apply(lambda x:datetime.datetime.strptime("%s %s %s" %(2016,x['LastContactMonth'],x['LastContactDay']),"%Y %b %d"),axis=1)

all_df['LastContactWkd'] = all_df['LastContactDate'].dt.weekday

all_df['LastContactWkd'].value_counts()

all_df['LastContactMon'] = all_df['LastContactDate'].dt.month

all_df = all_df.drop('LastContactMonth',axis=1)



# Get week of last contact

all_df['LastContactWk'] = all_df['LastContactDate'].dt.week



# Get num of week in a month. There might be easier ways to do this, I will keep exploring. 

MonWk = all_df.groupby(['LastContactWk','LastContactMon'])['Education'].count().reset_index()

MonWk = MonWk.drop('Education',axis=1)

MonWk['LastContactWkNum']=0

for m in range(1,13):

    k=0

    for i,row in MonWk.iterrows():

        if row['LastContactMon']== m:

            k=k+1

            row['LastContactWkNum']=k

            

def get_num_of_week(df):

    for i,row in MonWk.iterrows():

        if (df['LastContactWk']== row['LastContactWk']) & (df['LastContactMon']== row['LastContactMon']):

            return row['LastContactWkNum']



all_df['LastContactWkNum'] = all_df.apply(lambda x: get_num_of_week(x),axis=1)

print(all_df[['LastContactWkNum','LastContactWk','LastContactMon']].head(10))
# Spilt numeric and categorical features

cat_feats = all_df.select_dtypes(include=['object']).columns

num_feats = all_df.select_dtypes(include=['float64','int64']).columns

num_df = all_df[num_feats]

cat_df = all_df[cat_feats]

print('There are %d numeric features and %d categorical features\n' %(len(num_feats),len(cat_feats)))

print('Numeric features:\n',num_feats.values)

print('Categorical features:\n',cat_feats.values)
# One hot encoding

cat_df = pd.get_dummies(cat_df)
# Merge all features

all_data = pd.concat([num_df,cat_df],axis=1)
# Split train and test

idx=pd.IndexSlice

train_df=all_data.loc[idx[['train',],:]]

test_df=all_data.loc[idx[['test',],:]]

train_label=train['CarInsurance']

print(train_df.shape)

print(len(train_label))

print(test_df.shape)
# Train test split

x_train, x_test, y_train, y_test = train_test_split(train_df,train_label,test_size = 0.3,random_state=3)
# Create a cross validation function 

def get_best_model(estimator, params_grid={}):

    

    model = GridSearchCV(estimator = estimator,param_grid = params_grid,cv=3, scoring="accuracy", n_jobs= -1)

    model.fit(x_train,y_train)

    print('\n--- Best Parameters -----------------------------')

    print(model.best_params_)

    print('\n--- Best Model -----------------------------')

    best_model = model.best_estimator_

    print(best_model)

    return best_model
# Create a model fitting function

def model_fit(model,feature_imp=True,cv=5):



    # model fit   

    clf = model.fit(x_train,y_train)

    

    # model prediction     

    y_pred = clf.predict(x_test)

    

    # model report     

    cm = confusion_matrix(y_test,y_pred)

    plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')



    print('\n--- Train Set -----------------------------')

    print('Accuracy: %.5f +/- %.4f' % (np.mean(cross_val_score(clf,x_train,y_train,cv=cv)),np.std(cross_val_score(clf,x_train,y_train,cv=cv))))

    print('AUC: %.5f +/- %.4f' % (np.mean(cross_val_score(clf,x_train,y_train,cv=cv,scoring='roc_auc')),np.std(cross_val_score(clf,x_train,y_train,cv=cv,scoring='roc_auc'))))

    print('\n--- Validation Set -----------------------------')    

    print('Accuracy: %.5f +/- %.4f' % (np.mean(cross_val_score(clf,x_test,y_test,cv=cv)),np.std(cross_val_score(clf,x_test,y_test,cv=cv))))

    print('AUC: %.5f +/- %.4f' % (np.mean(cross_val_score(clf,x_test,y_test,cv=cv,scoring='roc_auc')),np.std(cross_val_score(clf,x_test,y_test,cv=cv,scoring='roc_auc'))))

    print('-----------------------------------------------') 



    # feature importance 

    if feature_imp:

        feat_imp = pd.Series(clf.feature_importances_,index=all_data.columns)

        feat_imp = feat_imp.nlargest(15).sort_values()

        plt.figure()

        feat_imp.plot(kind="barh",figsize=(6,8),title="Most Important Features")
# The confusion matrix plotting function is from the sklearn documentation below:

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



class_names = ['Success','Failure']
from sklearn.model_selection import GridSearchCV
# Let's start with KNN. An accuracy of 0.76 is not very impressive. I will just take this as the model benchmark. 

knn = KNeighborsClassifier()

parameters = {'n_neighbors':[5,6,7], 

              'p':[1,2],

              'weights':['uniform','distance']}

clf_knn = get_best_model(knn,parameters)

model_fit(model=clf_knn, feature_imp=False)
# As expected, Naive Bayes classifier doesn't perform well here. 

# There are multiple reasons. Some of the numeric features are not normally distributed, which is a strong assemption hold by Naive Bayes. 

# Also, features are definitely not independent.  

clf_nb = GaussianNB()

model_fit(model=clf_nb,feature_imp=False)
# We're making progress here. Logistic regression performs better than KNN. 

lg = LogisticRegression(random_state=3)

parameters = {'C':[0.8,0.9,1], 

              'penalty':['l1','l2']}

clf_lg = get_best_model(lg,parameters)

model_fit(model=clf_lg, feature_imp=False)
# I did some manual parameter tuning here. This is the best model so far. 

# Based on the feature importance report, call length, last contact week, and previous success are strong predictors of cold call success

rf = RandomForestClassifier(random_state=3)

parameters={'n_estimators':[100],

            'max_depth':[10],

            'max_features':[13,14],

            'min_samples_split':[11]}

clf_rf= get_best_model(rf,parameters)

model_fit(model=clf_rf, feature_imp=True)
# try a SVM RBF model 

svc = svm.SVC(kernel='rbf', probability=True, random_state=3)

parameters = {'gamma': [0.005,0.01,0.02],

              'C': [0.5,1,5]}

clf_svc = get_best_model(svc, parameters)

model_fit(model=clf_svc,feature_imp=False)
# Finally let's try out XBGoost. As expected, it outperforms all other algorithms. 

# Also, based on feature importances, some of the newly created features such as call start hour, last contact week and weekday 

# have been picked as top features. 



import xgboost as xgb

xgb = xgb.XGBClassifier()

parameters={'n_estimators':[900,1000,1100],

            'learning_rate':[0.01],

            'max_depth':[8],

            'min_child_weight':[1],

            'subsample':[0.8],

            'colsample_bytree':[0.3,0.4,0.5]}

clf_xgb= get_best_model(xgb,parameters)

model_fit(model=clf_xgb, feature_imp=True)
# Compare model performance

clfs= [clf_knn, clf_nb, clf_lg, clf_rf, clf_svc, clf_xgb]

index =['K-Nearest Neighbors','Naive Bayes','Logistic Regression','Random Forest','Support Vector Machines','XGBoost']

scores=[]

for clf in clfs:

    score = np.mean(cross_val_score(clf,x_test,y_test,cv=5,scoring = 'accuracy'))

    scores = np.append(scores,score)

models = pd.Series(scores,index=index)

models.sort_values(ascending=False)
# XGBoost and Random Forest show different important features, implying that those models are capturing different aspects of the data

# To get the final model, I ensembled different classifiers based on majority voting.

# XGBoost and Random Forest are given larger weights due to their better performance. 



clf_vc = VotingClassifier(estimators=[('xgb', clf_xgb),                                       

                                      ('rf', clf_rf),

                                      ('lg', clf_lg), 

                                      ('svc', clf_svc)], 

                          voting='hard',

                          weights=[4,4,1,1])

clf_vc = clf_vc.fit(x_train, y_train)
print('Final Model Accuracy: %.5f'%(accuracy_score(y_test, clf_vc.predict(x_test))))