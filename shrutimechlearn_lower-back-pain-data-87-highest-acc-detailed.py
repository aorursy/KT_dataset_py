# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import seaborn as sns

sns.set()

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

%matplotlib inline

#plt.style.use('ggplot')

from sklearn.preprocessing import StandardScaler



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Dataset_spine.csv')
data.info()
data['Unnamed: 13'][:20]
data.drop(['Unnamed: 13'],axis=1,inplace=True)
data.columns = ['pelvic_incidence','pelvic tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis','pelvic_slope','Direct_tilt','thoracic_slope','cervical_tilt','sacrum_angle','scoliosis_slope','State']
## Let's check how the data looks now

data.info()
data.describe(include="all")
data.State.value_counts()
p = data.plot(kind='box',figsize =(30,15))
fig,ax = plt.subplots(nrows = 3, ncols=4, figsize=(16,10))

row = 0

col = 0

for i in range(len(data.columns) -1):

    if col > 3:

        row += 1

        col = 0

    axes = ax[row,col]

    sns.boxplot(x = data['State'], y = data[data.columns[i]],ax = axes)

    col += 1

plt.tight_layout()

# plt.title("Individual Features by Class")

plt.show()
sns.countplot(y=data.dtypes ,data=data)

plt.xlabel("count of each data type")

plt.ylabel("data types")

plt.show()
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

data['State_Code'] = lb_make.fit_transform(data['State'])
data.State_Code.value_counts()
data.to_csv('Dataset_spine_clean.csv')
data.hist(figsize=(15,12),bins = 15, color="#107009AA")

plt.title("Features Distribution")

plt.show()
p=sns.pairplot(data, hue = 'State')
data.columns
plt.figure(figsize=(15,15))

p=sns.heatmap(data.corr(), annot=True,cmap='RdYlGn') 
## null count analysis before modelling to keep check

import missingno as msno

p=msno.bar(data)
sc_X = StandardScaler()

X =  pd.DataFrame(sc_X.fit_transform(data.drop(["State",'State_Code'],axis = 1)), columns = ['pelvic_incidence', 'pelvic tilt', 'lumbar_lordosis_angle',

       'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis',

       'pelvic_slope', 'Direct_tilt', 'thoracic_slope', 'cervical_tilt',

       'sacrum_angle', 'scoliosis_slope'])

#X = data.drop(["State",'State_Code'],axis = 1)

y = data.State_Code
X.head()
#importing train_test_split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42, stratify=y)

from sklearn.neighbors import KNeighborsClassifier





test_scores = []

train_scores = []



for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(X_train,y_train)

    

    train_scores.append(knn.score(X_train,y_train))

    test_scores.append(knn.score(X_test,y_test))
max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
import matplotlib

plt.figure(figsize=(15,5))

plt.title('k-NN Varying number of neighbors')

plt.plot(range(1,15),test_scores,label="Test", marker='*')

plt.plot(range(1,15),train_scores,label="Train",linestyle='--')

plt.legend()

plt.xticks(range(1,15))

plt.show()
#Setup a knn classifier with k neighbors

knn = KNeighborsClassifier(13)



knn.fit(X_train,y_train)

knn.score(X_test,y_test)
y_pred = knn.predict(X_test)
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
#import classification_report

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn.metrics import roc_curve

y_pred_proba = knn.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('Knn(n_neighbors=13) ROC curve')

plt.show()