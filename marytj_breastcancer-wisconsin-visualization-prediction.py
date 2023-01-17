# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import ExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn import svm







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

bcan=pd.read_csv('../input/data.csv')

print(bcan.head(2))
#Getting information about the dataset

print(bcan.columns.values)

bcan.describe()
bcan.drop(['Unnamed: 32'],axis=1,inplace=True)

print(bcan.columns.values)
bcan_null=bcan.isnull().sum()

print(bcan_null)

#bcan_null.plot(kind='bar',grid=False)

#plt.axhline(0)

#bcan.isnan()



#plt.plot(bcan_null)
print(bcan['diagnosis'].value_counts())

ax=plt.gca()

bcan['diagnosis'].value_counts().plot(kind='bar')

ax.set_xticklabels(['Benign','Malignant'],ha='right',rotation=0)
bcan['diagnosis']=bcan['diagnosis'].map({'B':0,'M':1})

#To check if the mapping is correct

print(bcan['diagnosis'].value_counts())

ax=plt.gca()

bcan['diagnosis'].value_counts().plot(kind='bar')

ax.set_xticklabels(['Benign','Malignant'],ha='right',rotation=0)
bcan_corr = bcan.corr()

sns.heatmap(bcan_corr, fmt='.2f')

fig=plt.figure(figsize=(12,18))
print(bcan_corr['diagnosis'].sort_values(ascending=False))
feature_selected=['diagnosis','concave points_worst','perimeter_worst','concave points_mean','radius_worst',\

    'perimeter_mean','area_worst','radius_mean','area_mean','concavity_mean','concavity_worst',\

    'compactness_mean','compactness_worst']
bcan_corr['diagnosis'].sort_values().plot(kind='bar',sort_columns=True)
#plt.rcParams['figure.figsize']=(1000,1000)



#g=sns.pairplot(bcan[feature_selected],hue='diagnosis')

bcan[feature_selected].corr().style.format("{:.2}").background_gradient(cmap='coolwarm',axis=1)
#splitting dataset into X & Y

Y_bcan=bcan['diagnosis']#.values

#print(Y_bcan)



X_bcan=bcan[feature_selected].iloc[:,1:]#.values

#print(X_bcan.columns.values)



X_train,X_test,Y_train,Y_test=train_test_split(X_bcan,Y_bcan,random_state=0,stratify=Y_bcan)

clf=RandomForestClassifier(n_estimators=100,random_state = 0)

#clf=ExtraTreeClassifier(random_state = 0)

clf.fit(X_train,Y_train)

print('Training Set Accuracy:{:.3f}'.format(clf.score(X_train,Y_train)))

print('Test Set Accuracy:{:.3f}'.format(clf.score(X_test,Y_test)))
#bcan_features=bcan.shape[1]

#print(bcan_features)

importance=clf.feature_importances_

print(clf.feature_importances_)

print(bcan[feature_selected].columns)

features=bcan[feature_selected].columns

indices = np.argsort(importance)

print(indices)

#ax=plt.gca()

plt.title('Feature Importances')



plt.barh(range(len(indices)), importance[indices], color='b', align='center')

plt.yticks(range(len(indices)), features) ## removed [indices]

plt.xlabel('Relative Importance')

plt.show()

feature_selected=['compactness_mean','concavity_mean','area_mean','radius_mean']

Y_bcan=bcan['diagnosis']#.values

X_bcan=bcan[feature_selected].iloc[:,1:]#.values





X_train,X_test,Y_train,Y_test=train_test_split(X_bcan,Y_bcan,random_state=0,stratify=Y_bcan)

clf_log=LogisticRegression(random_state = 0)

#clf=ExtraTreeClassifier(random_state = 0)

clf_log.fit(X_train,Y_train)

print('Training Set Accuracy:{:.3f}'.format(clf_log.score(X_train,Y_train)))

print('Test Set Accuracy:{:.3f}'.format(clf_log.score(X_test,Y_test)))

clf_log.predict()

#Trying svms

svc = svm.SVC(kernel='linear')

svc.fit(X_train, Y_train)    

print('Training Set Accuracy:{:.3f}'.format(svc.score(X_train,Y_train)))

print('Test Set Accuracy:{:.3f}'.format(svc.score(X_test,Y_test)))