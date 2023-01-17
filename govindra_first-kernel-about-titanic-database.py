# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#### Import all the required libraries
import seaborn as sns #### Library for plotting graphs. This is a layer on top of matplotlib
import matplotlib.pyplot as plt #### Basic Library for plotting graphs
#### Configuring Matplotlib to show Plots inline
%matplotlib inline 
plt.rcParams['figure.figsize'] = (50, 50) ### Setting the size of the Plots
# Any results you write to the current directory are saved as output.
import os
import re
print(os.listdir("../input"))

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

#train_data.head(10)
test_data.info()
train_data.info()
#data = pd.concat([train_data,test_data],ignore_index=True,sort=False)
#da=data
#import copy
#data = copy.deepcopy(train_data)
data = train_data.copy(deep=True)
pattern = re.compile(r'\s+([A-Za-z]+)\.')

def get_title(name):
    match = pattern.search(name)
    if match:
        return match.group(1)
    return ""

def has_cabin(Cabin):
    if type(Cabin) == float:
        return 1
    else: 
        return 0

def family_size(SibSp,Parch):
    return SibSp + Parch + 1

data['Embarked'] = data['Embarked'].fillna(0)

age_avg = data['Age'].mean()
age_std = data['Age'].std()
age_null_count = data['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
data['Age'][np.isnan(data['Age'])] = age_null_random_list
data['Age'] = data['Age'].astype(int)

data['Fare'] = data['Fare'].fillna(train_data['Fare'].mean()) 

data['Title'] = data['Name'].apply(get_title)
data['Family_size'] = data.apply(lambda x: family_size(x['SibSp'], x['Parch']), axis=1)
data['Has_cabin'] = data['Cabin'].apply(has_cabin)


data.count()



clean_up_dict = {'Sex' :{'female': 1, 'male': 0},
                 'Title':{'Lady':0,'Countess':0,'Capt':0, 'Col':0,'Don':0, 
                          'Dr':0, 'Major':0, 'Rev':0, 'Sir':0, 'Jonkheer':0, 'Dona': 0,
                           "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4,'Mme':3,'Mlle':2,'Ms':2},
                 'Embarked': {'S': 0, 'C': 1, 'Q': 2},
                }
data.replace(clean_up_dict, inplace=True)
data['Title'] = data['Title'].astype(int)
data.describe()
data.info()
data.count()

drop_elements = [ 'Name', 'Ticket', 'Cabin', 'SibSp','PassengerId']
data = data.drop(drop_elements, axis = 1)
data.head(10)
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
cormat = data.corr()
sns.heatmap(cormat,linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
#data.Fare.hist()
data.Age.hist()
#data.Fare.hist()
#data.Age.hist()
#bins age by 8 and Fare by 50
#d=copy.deepcopy(data)
d = data.copy(deep=True)
age_bins = [-1,8,16,24,32,40,48,56,64,72,80]
group_names = [1,2,3,4,5,6,7,8,9,10]
out = pd.cut(d.Age, bins = age_bins, labels=group_names)
d['Age'] = out
d['Age'] = d['Age'].astype(int)

d['Fare'][np.isnan(d['Fare'])] = d['Fare'].fillna(d['Fare'].median())


Fare_bins = [-1,50,100,150,200,250,300,350,400,450,500]
group_names = [1,2,3,4,5,6,7,8,9,10]
out = pd.cut(d.Fare, bins = Fare_bins, labels=False)
#type(out)

d['Fare'] = out
d=d.dropna()
d.info()
y = d['Survived'].values
#d['Fare'] = d['Fare'].astype(int)
d.info()

#d.Fare.fillna(d.Fare.mean())
#d.info()
#d.head(10)
d.Fare.unique()
d.describe()
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
cormat = d.corr()
sns.heatmap(cormat,linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
#d.dropna('Fare',1)
nans = lambda data: data[data.isnull().any(axis=1)]
nans(d)
#d.head(10)

#d.query('Has_cabin ==0')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
D = sc.fit_transform(d)

from sklearn.model_selection import train_test_split
X_train, X_test_, y_train,y_test = train_test_split(D, y, test_size = 0.1, random_state = 0)
#_, X_test, _, y_test = train_test_split(test_data, y, test_size = 0.1, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc)
### Lets create a Confusion Matrix to See how valid our accuracy score is
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
labels =['Pr 0', 'Pr 1']
print(*labels)
for line in cm:
    print(*line)
#### Finally Lets Apply k-Fold Cross Validation to see how our model has performed
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())