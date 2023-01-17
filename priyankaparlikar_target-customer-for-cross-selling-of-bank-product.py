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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import scipy.stats as st

from sklearn.feature_selection import RFE

from sklearn.preprocessing import MinMaxScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score,roc_curve,classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB
pd.set_option('display.max_columns',None)

df_train=pd.read_csv('../input/banking-dataset-marketing-targets/train.csv')
df_train.head()
print('train shape: {}'.format(df_train.shape))
df_train.info()
df_train.drop(['ID','poutcome'],axis=1,inplace=True)
df_train.isna().sum()
df_train.describe()
df_train['subscribed'].value_counts().plot(kind='bar')

plt.xlabel('Subscribed')

plt.ylabel('No. of subscription')

plt.show()
df_train['subscribed'].value_counts()
plt.figure(figsize=(20,8))

sns.countplot(data=df_train,x='job',hue='subscribed')

plt.ylabel('No. of subscription')

plt.show()
sns.countplot(data=df_train,x='marital',hue='subscribed')

plt.ylabel('No. of subscription')

plt.show()
sns.countplot(data=df_train,x='education',hue='subscribed')

plt.ylabel('No. of subscription')

plt.show()
sns.countplot(data=df_train,x='housing',hue='subscribed')

plt.ylabel('No. of subscription')

plt.show()
sns.countplot(data=df_train,x='loan',hue='subscribed')

plt.ylabel('No. of subscription')

plt.show()
sns.countplot(data=df_train,x='contact',hue='subscribed')

plt.ylabel('No. of subscription')

plt.show()
plt.figure(figsize=(15,8))

sns.countplot(data=df_train,x='month',hue='subscribed')
final_train = pd.get_dummies(data=df_train,columns=['job','marital','education','default','housing','loan','contact','month'])
final_train['subscribed']=final_train['subscribed'].replace('no',0)

final_train['subscribed']=final_train['subscribed'].replace('yes',1)
final_train.head()
final_train.shape
cor=final_train.corr()

sub_cor=abs(cor['subscribed'])

sig_features=sub_cor[sub_cor>0.05]

print(sig_features)

print(sig_features.count())
X=final_train.drop('subscribed',axis=1)

y=final_train['subscribed']
vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif
x=X.drop(['age','job_admin.','marital_divorced','education_primary','default_no','loan_no','housing_no','contact_unknown','month_apr'],axis=1)

vif1 = pd.DataFrame()

vif1["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

vif1["features"] = x.columns

vif1


df= final_train.drop(['age','job_admin.','marital_divorced','education_primary','default_no','loan_no','housing_no','contact_unknown','month_apr'],axis=1)
df.shape
X=df.drop('subscribed',axis=1)

y=df['subscribed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
Xytrain = pd.concat([X_train,y_train],axis=1)



print('before oversampling: ','\n', Xytrain['subscribed'].value_counts())

Xytrain0 = Xytrain[Xytrain['subscribed']==0]

Xytrain1 = Xytrain[Xytrain['subscribed']==1]



len0 = len(Xytrain0)

len1 = len(Xytrain1)



Xytrain1_os = Xytrain1.sample(len0,replace = True, random_state=3)

Xytrain_os = pd.concat([Xytrain0, Xytrain1_os],axis=0)



print('after undersampling: ','\n',Xytrain_os['subscribed'].value_counts())



y_train_os = Xytrain_os['subscribed']

X_train_os = Xytrain_os.drop('subscribed',axis=1)
ss = StandardScaler()



Xtrains = ss.fit_transform(X_train_os)

Xtests = ss.transform(X_test)
def model_eval(algo, Xtrains, y_train_os, Xtests, y_test):

    algo.fit(Xtrains,y_train_os)

    ytrain_pred = algo.predict(Xtrains)

    ytrain_prob = algo.predict_proba(Xtrains)[:,1]



    print('Overall accuracy - train:' , accuracy_score(y_train_os, ytrain_pred))

    print('Confusion matrix - train: ','\n',confusion_matrix(y_train_os,ytrain_pred))

    print('AUC - train', roc_auc_score(y_train_os,ytrain_prob))

    print('\n')

    print('Classification report - train: ','\n',classification_report(y_train_os,ytrain_pred))



    ytest_pred = algo.predict(Xtests)

    ytest_prob = algo.predict_proba(Xtests)[:,1]



    print('\n')

    print('Overall accuracy - test:' , accuracy_score(y_test, ytest_pred))

    print('Confusion matrix - test: ','\n',confusion_matrix(y_test,ytest_pred))

    print('AUC - test', roc_auc_score(y_test,ytest_prob))

    print('Classification report - test: ','\n',classification_report(y_test,ytest_pred))



    fpr,tpr,thresholds = roc_curve(y_test,ytest_prob)

    plt.plot(fpr,tpr)

    plt.plot(fpr,fpr,'r')

    plt.xlabel('FPR')

    plt.ylabel('TPR')

    plt.show()
dt=DecisionTreeClassifier(max_depth = 5, criterion = 'gini',random_state=3)

model_eval(dt, Xtrains, y_train_os, Xtests, y_test)
rf = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=5,random_state=3)

model_eval(rf, Xtrains, y_train_os, Xtests, y_test)
lr=LogisticRegression(solver='liblinear', fit_intercept=True,random_state=3)

model_eval(lr, Xtrains, y_train_os, Xtests, y_test)
ada = AdaBoostClassifier(random_state = 3)

model_eval(ada, Xtrains, y_train_os, Xtests, y_test)
clf = GaussianNB()

model_eval(clf, Xtrains, y_train_os, Xtests, y_test)