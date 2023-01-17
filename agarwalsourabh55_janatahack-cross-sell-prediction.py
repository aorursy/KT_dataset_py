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
#import seaborn as sns
#train[train['Gender']=='Male']['Response'].sum()
#train[train['Gender']=='Female']['Response'].sum()
#col=['Gender','Response']
#sns.pairplot(train[col])
#col=train.columns
#print(col)
#col=[ 'Gender', 'Age', 'Driving_License', 'Region_Code',
#       'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
#       'Policy_Sales_Channel', 'Vintage', 'Response']

#sns.pairplot(train[col])
#col=['Response','Vintage']

#sns.pairplot(train[col])
#import matplotlib.pyplot as plt 
#var = 'Vehicle_Age'
#data = pd.concat([train['Response'], train[var]], axis=1)
#data.plot.scatter(x=var, y='Vehicle_Age', ylim=(0,1));

#train[train['Response']==0]['Vehicle_Age'].unique()

#train[train['Response']==1]['Vehicle_Age'].unique()

#print(len(train[train['Previously_Insured']==1][train['Response']==0]))
#print(len(train[train['Previously_Insured']==1]))
#col=['Annual_Premium','Response']
#sns.distplot(train[col])




#print("The number of classes before fit {}".format(Counter(target)))
#print("The number of classes after fit {}".format(Counter(y_train_ns)))
#X_train_ns['Response'].sum()
 
#target=X_train_ns['Response']
#X_train_ns=X_train_ns.drop(['Response'],axis=1)
'''
from imblearn.under_sampling import NearMiss
ns=NearMiss(0.8)
X_train_ns,y_train_ns=ns.fit_sample(X_train_ns,target)
print("The number of classes before fit {}".format(Counter(target)))
print("The number of classes after fit {}".format(Counter(y_train_ns)))

'''
#X_train_ns
'''
print(len(train[train['Driving_License']==1][target==0]))
print(len(train[train['Driving_License']==0][target==0]))
print(len(train[train['Driving_License']==1][target==1]))
print(len(train[train['Driving_License']==0][target==1]))

print(len(train[train['Previously_Insured']==1][target==0]))
print(len(train[train['Previously_Insured']==0][target==0]))
print(len(train[train['Previously_Insured']==1][target==1]))
print(len(train[train['Previously_Insured']==0][target==1]))

print(len(train[train['Gender']==1][target==0]))
print(len(train[train['Gender']==0][target==0]))
print(len(train[train['Gender']==1][target==1]))
print(len(train[train['Gender']==0][target==1]))

'''
#target=pd.concat([target,train['Driving_License']],axis=1)


#from imblearn.under_sampling import NearMiss
#nm=NearMiss()
#train_new,target_new=nm.fit_sample(train,target)
train=pd.read_csv('../input/janatahack-crosssell-prediction/train.csv')
test=pd.read_csv('../input/janatahack-crosssell-prediction/test.csv')

target=train['Response']
train=train.drop(['Response','id'],axis=1)
a={'Male':0,'Female':1}
train['Gender']=train['Gender'].map(a)
a={'> 2 Years':0, '1-2 Year':2, '< 1 Year':1}
train['Vehicle_Age']=train['Vehicle_Age'].map(a)
a={'Yes':1,'No':0}
train['Vehicle_Damage']=train['Vehicle_Damage'].map(a)



a={'Male':0,'Female':1}
test['Gender']=test['Gender'].map(a)
a={'> 2 Years':0, '1-2 Year':2, '< 1 Year':1}
test['Vehicle_Age']=test['Vehicle_Age'].map(a)
a={'Yes':1,'No':0}
test['Vehicle_Damage']=test['Vehicle_Damage'].map(a)

ids=test['id']
test=test.drop(['id'],axis=1)
#for i in train.columns:
#    print(train[i].unique())
#print(len(train.iloc[:,7].unique()))
#import seaborn as sns
len(train['Vintage'].unique())

train['Vintage'][train['Vintage']<150]=1
train['Vintage'][train['Vintage']>=150]=0


from imblearn.over_sampling import RandomOverSampler
from collections import Counter

os=RandomOverSampler(0.75)
train_new,target_new=os.fit_sample(train,target)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train,target,test_size=0.2,random_state=3)

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
#x = np.array([[0,0],[0,1],[1,1],[3,0]])
clf = LocalOutlierFactor(n_neighbors=20,algorithm='auto',leaf_size=30,metric='minkowski',p=2,metric_params=None)
y_pred = clf.fit_predict(X_train)
scores = clf.negative_outlier_factor_
print(-scores)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#y_pred=clf.predict(X_test)

print(confusion_matrix(y_train,y_pred))
print(accuracy_score(y_train,y_pred))
print(classification_report(y_train,y_pred))

from sklearn.ensemble import IsolationForest
IsolationForest(n_estimators=100, max_samples=len(X), contamination=outlier_fraction,random_state=state, verbose=0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)


y_pred=rfc.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

train['Vintage'].unique().max()



a
from imblearn.combine import SMOTETomek
os=SMOTETomek(0.75)
X_train_ns,y_train_ns=os.fit_sample(train,target)
#print("The number of classes before fit {}".format(Counter(y_train)))
#print("The number of classes after fit {}".format(Counter(y_train_ns)))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_train_ns,y_train_ns,test_size=0.4,random_state=3)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report






#from xgboost import XGBClassifier
#from sklearn.metrics import accuracy_score
#lr=XGBClassifier()
#lr.fit(X_train,y_train)

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)


X_train.columns
feature_imp = pd.Series(rfc.feature_importances_,index=X_train.columns).sort_values(ascending=False)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
cols=['Previously_Insured','Vintage','Age','Vehicle_Damage','Annual_Premium','Policy_Sales_Channel','Region_Code','Gender','Vehicle_Age']
X_train=X_train[cols]
rfc1=RandomForestClassifier()
rfc1.fit(X_train,y_train)
X_test=X_test[cols]
y_pred=rfc1.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
predictions=rfc.predict_proba(test)[:,1]

output = pd.DataFrame({'id': ids, 'Response': predictions})
output.to_csv('my_submission7.csv', index=False)
print("Your submission was successfully saved!")

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,y_pred)
pr=rfc.predict(test)



output = pd.DataFrame({'id': ids, 'Response': pr})
output.to_csv('my_submission3.csv', index=False)
print("Your submission was successfully saved!")
from sklearn.metrics import roc_auc_score
ns_auc = roc_auc_score(y_test,pred)
ns_auc
from sklearn.metrics import confusion_matrix
ns_auc = confusion_matrix(y_test,pred)
ns_auc
#true positive rate or sensitivity or recall
ns_auc[0][0] / (ns_auc[0][0] + ns_auc[1][0])
#true negatice rate or inverted specificity
ns_auc[0][1] / (ns_auc[0][1] + ns_auc[1][1])
print("no with labels as 0",len(target)-target.sum())
print("no wth label as 1",target.sum())
target.sum()/(len(target)-target.sum())
