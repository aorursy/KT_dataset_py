import pandas as pd

import numpy as np

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

# Set theme

sns.set_style('darkgrid') # 'whitegrid'
fd = pd.read_csv('../input/falldeteciton.csv')
fd.head()
fd = fd.replace({'ACTIVITY':{0:'Standing',1:'Walking',2:'Sitting',3:'Falling',4:'Cramps',5:'Running'}})

fd.head()
fd.shape
fd.describe()
fd.isna().sum()
cols=['TIME','SL','EEG','BP','HR','CIRCLUATION']

#Distribution

fig = plt.figure(figsize=(10, 20)) # (Breite,Länge)

for i in range (0,len(cols)):

    fig.add_subplot(len(cols),1,i+1)

    sns.distplot(fd[cols[i]]);
sns.boxplot(data=fd)
fd = fd[(fd['EEG']< fd['EEG'].quantile(0.999) ) & (fd['EEG']> fd['EEG'].quantile(0.001))]
fd = fd[(fd['SL']< fd['SL'].quantile(0.99) ) & (fd['SL']> fd['SL'].quantile(0.01))]
sns.boxplot(data=fd)
sns.pairplot(fd, hue='ACTIVITY', h=2.5);
from sklearn.model_selection import train_test_split

# Classifiier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



# Metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
Target = fd['ACTIVITY']

Features = fd[['TIME','SL','EEG','BP','HR','CIRCLUATION']]



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size = 0.3)
lr = LogisticRegression()

lr.fit(X_train, y_train)

predictions = lr.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

accuracy
dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

predictions = dtc.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

accuracy
rfc = RandomForestClassifier(

    n_estimators = 50,

    #min_samples_leaf=10,

    #min_samples_split=10

)

rfc.fit(X_train, y_train)

predictions = rfc.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

accuracy
knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

predictions = knn.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

accuracy
svc = SVC(gamma='auto')

svc.fit(X_train,y_train)

predictions = svc.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

accuracy