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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df1= pd.read_csv('../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
df1.head()
#Let us check the dataset info

df1.info()
#How are the data distributed?

df1.describe().transpose()
#Check for data correlation

plt.figure(figsize=(12,10))

sns.heatmap(df1.corr(), annot=True, square=True)
#Let us check the balancing of the output data

sns.countplot(df1['diagnosis'])
#Data split between 2 output classifier

df1['diagnosis'].value_counts()/len(df1['diagnosis'])*100
#Further feature analysis

sns.pairplot(df1)
#We are checking the outliers in the data

for i in df1.columns:

    plt.figure(figsize=(12,10))

    sns.boxplot(df1[i])
#We shall re-order the columns and then drop the highly correlated columns

df1 = df1[['mean_radius','mean_texture','mean_smoothness','mean_perimeter','mean_area','diagnosis']]
from sklearn.preprocessing import StandardScaler

std_scale= StandardScaler()

X = std_scale.fit_transform(df1.drop(['mean_area','mean_perimeter','diagnosis'], axis=1))

y = df1['diagnosis']
X_data = pd.DataFrame(X, columns=df1.columns[:-3])
X_data.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X_data, y, test_size=0.3)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
# Let us find the right K value

error_plt=[]

for i in range(1,40):

    knn= KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    y_pred=knn.predict(X_test)

    error_plt.append(np.mean(y_pred !=y_test))
plt.figure(figsize=(12,10))

plt.plot(range(1,40), error_plt,marker='o', markerfacecolor='red')
knn1= KNeighborsClassifier(n_neighbors=25)

knn1.fit(X_train, y_train)

y1_pred=knn1.predict(X_test)
print(confusion_matrix(y_test, y1_pred))

print(classification_report(y_test,y1_pred))
#Naive Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y2_pred=gnb.predict(X_test)
print('Accuracy report for GNB ',accuracy_score(y_test, y2_pred) )

print('Confusion Matrix  ', confusion_matrix(y_test, y2_pred))

print('Classification Report  ',classification_report(y_test,y2_pred))
#Random Forest Classifier - using this approach to see for any better outcome than the KNN

#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

def random_forest(train, test, y_train, _test, n_estimators = 10, max_depth = 20, min_samples_split = 2, random_state=42):

    rfc = RandomForestClassifier(n_estimators=n_estimators,max_depth=20, min_samples_split=2,random_state=42)

    rfc.fit(X_train, y_train)

    y3_pred=rfc.predict(X_test)

    print('Estimator count   ',n_estimators)

    print('Minimum Samples Split   ',min_samples_split)

    print(accuracy_score(y_test, y3_pred))

    print(confusion_matrix(y_test, y3_pred))

    print(' ')

    print(' ')

    print('*****************')
for i in range(2,100,5):

    for j in range(2,10,2):

        random_forest(X_train, X_test, y_train, y_test, n_estimators=i, max_depth=20, min_samples_split=j, random_state=42)
#We chose estimator 92, as this had best Accuracy compared with rest

rfc_new = RandomForestClassifier(n_estimators=92,max_depth=20, min_samples_split=2,random_state=42 )

rfc_new.fit(X_train, y_train)

y6_pred=rfc_new.predict(X_test)

print(accuracy_score(y_test, y6_pred))

print(confusion_matrix(y_test, y6_pred))
from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression()

logReg.fit(X_train, y_train)

y4_pred = logReg.predict(X_test)

print(accuracy_score(y_test, y4_pred))

print(confusion_matrix(y_test, y4_pred))
# I am going to try the Voting Classifier, with the previous classifier algorithms - Logistic Regression

#Random Forest Classifier, KNN, Naive Bayes. Let us try Hard voting first

from sklearn.ensemble import VotingClassifier

vot_class = VotingClassifier(estimators=[('logReg',logReg),('Random',rfc_new),('KNN', knn1),('Naive', gnb)], voting='hard')

vot_class.fit(X_train, y_train)

y5_pred = vot_class.predict(X_test)

print(accuracy_score(y_test, y5_pred))

print(confusion_matrix(y_test, y5_pred))
# Voting Classifier using Soft voting

vot_class2 = VotingClassifier(estimators=[('logReg',logReg),('Random',rfc_new),('KNN', knn1),('Naive', gnb)], voting='soft')

vot_class2.fit(X_train, y_train)

y7_pred = vot_class2.predict(X_test)

print(accuracy_score(y_test, y7_pred))

print(confusion_matrix(y_test, y7_pred))
#Trying Stacking Classifier

from sklearn.ensemble import StackingClassifier

stck_class = StackingClassifier(estimators=[('logReg',logReg),('Random',rfc_new),('KNN', knn1),('Naive', gnb)], cv=5, final_estimator=LogisticRegression())

stck_class.fit(X_train, y_train)

y8_pred = stck_class.predict(X_test)

print(accuracy_score(y_test, y8_pred))

print(confusion_matrix(y_test, y8_pred))