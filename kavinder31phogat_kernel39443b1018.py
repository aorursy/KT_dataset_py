# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore')

from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

df_test=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
print(df_train.shape)

df_train.head()
print(df_test.shape)

df_test.head()
df_train.info()
df_test.info()
df_train.describe()
df_test.describe()
#finding the null values



total=df_train.isnull().sum().sort_values(ascending = False)

percent=(df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)

missing_data=pd.concat([total,percent],axis =1,keys=['Total','Percent'])

missing_data.head(10)
df_train['target'].value_counts()
#visualizing percentage of target variable



df_train.target.value_counts().plot(kind='pie',autopct='%1.0f%%')
#skewness and kurtosis



sns.distplot(df_train['target'])

print("Skewness: %f" % df_train['target'].skew())

print("Kurtosis: %f" % df_train['target'].kurt())
# 1. Filter method



#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = df_train.corr()

#sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

#plt.show()
#Correlation with output variable

cor_target = abs(cor["target"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.03]

relevant_features=list(relevant_features.index)

type(relevant_features)
relevant_features=relevant_features[1:]

relevant_features
len(relevant_features)
#checking for correlation between filter_features



cor1=abs(df_train[relevant_features].corr())

cor1[cor1>0.01].stack()
a=['ID_code','target']

relevant_features.extend(a)

df_train1=df_train[relevant_features]

df_train1.head()
#Box plot of filter_variables1 for checking the distriution

plt.subplots(figsize=(18,8))

df_train1.boxplot(rot=90)
#Histogram for filter_variables1

df_train1.hist(figsize=(20,20));
one = df_train1[df_train1['target']==1]

zero = df_train1[df_train1['target']==0]
var=df_train1.columns[:-2]

fig, ax = plt.subplots(10,9,figsize=(20,20))

j=0

for i in var:

    j+=1

    plt.subplot(10,9,j)

    sns.distplot(one[i], color='r')

    sns.distplot(zero[i], color='b')

    plt.title(i)
from sklearn.model_selection import train_test_split

X = df_train1.drop(['ID_code','target'],axis=1)

y = df_train1['target']

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
#from sklearn.model_selection import GridSearchCV



#lr=[0.01,0.05,0.1,0.2,0.3]

#base_learners = [50,100,150,200,250,300,350]

#Depths = [3,5,7,9,11,13,15]

#childweight=[80,85,90]

#lossfunc=[0.1,0.2,0.3,0.4]

#subsampl=[0.5,1]



#param_grid = {'gamma':lossfunc}

              #'learning_rate': lr

              #'n_estimators': base_learners

              #'max_depth':Depths,

              #'min_child_weight':childweight

              #

              #'subsample': subsampl,

              

              

    

#from xgboost import XGBClassifier





#clf = XGBClassifier(objective="binary:logistic",max_delta_step=0,learning_rate=0.2,n_estimators=200,max_depth=3,min_child_weight=80)

#model = GridSearchCV(clf, param_grid, scoring = 'roc_auc', cv=3, n_jobs = -1,pre_dispatch=2)

#model.fit(train_X, train_y)





#print("Best: %f using %s" % (model.best_score_, model.best_params_))

#means = model.cv_results_['mean_test_score']

#stds = model.cv_results_['std_test_score']

#params = model.cv_results_['params']

#for mean, stdev, param in zip(means, stds, params):

#    print("%f (%f) with: %r" % (mean, stdev, param))
# Optimal value of number of base learners

optimal_learners = 2000

print("The optimal number of base learners is : ",optimal_learners)

# Optimal value of depth

optimal_depth = 3

print("\nThe optimal value of depth is : ",optimal_depth)

# Optimal value of childweight

optimal_childweight = 80

print("\nThe optimal value of childweight is : ",optimal_childweight)

# Optimal value of lossfunc

optimal_lossfunc = 0.2

print("\nThe optimal value of lossfunction is : ",optimal_lossfunc)

# Optimal value of subsampl

optimal_subsampl = 0.5

print("\nThe optimal value of subsample is : ",optimal_subsampl)

# Optimal value of lr

optimal_lr = 0.2

print("\nThe optimal value of lr is : ",optimal_lr)
from datetime import datetime

from xgboost import XGBClassifier



print(datetime.now())



clffinal = XGBClassifier(objective = "binary:logistic",n_estimators= optimal_learners,max_delta_step=0 ,max_depth=optimal_depth,

                         min_child_weight=optimal_childweight, gamma=optimal_lossfunc, subsample= optimal_subsampl,

                         learning_rate= optimal_lr,eval_metric ="auc",n_jobs = -1).fit(train_X,train_y)

print(datetime.now())
y_pred = clffinal.predict_proba(val_X)[:,1]
from sklearn.metrics import roc_auc_score

print("Training score :" + str(roc_auc_score(train_y,clffinal.predict(train_X))))

print("validation score :" + str(roc_auc_score(val_y,y_pred)))

b=relevant_features[:-2]
test_X = df_test.drop(['ID_code'],axis=1)

test_X = test_X[b]

Test_Prediction = clffinal.predict(test_X)



sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})

sub_df["target"] = Test_Prediction

sub_df.to_csv("submission_final.csv", index=False)
