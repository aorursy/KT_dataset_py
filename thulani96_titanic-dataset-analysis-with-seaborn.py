import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib import rcParams
%matplotlib inline
# figure size in inches
rcParams['figure.figsize'] = 15,6
data = pd.read_csv('../input/train.csv')
data.fillna(0, inplace = True)
# Convert the survived column to strings for easier reading
data['Survived'] = data['Survived'].map({
    0: 'Died',
    1: 'Survived'
})

# Convert the Embarked column to strings for easier reading
data['Embarked'] = data['Embarked'].map({
    'C':'Cherbourg',
    'Q':'Queenstown',
    'S':'Southampton',
})

data.head()
# fig, ax = plt.subplots(1,1, figsize = (12,10))
ax = sns.countplot(x = 'Pclass', hue = 'Survived', palette = 'Set1', data = data)
ax.set(title = 'Passenger status (Survived/Died) against Passenger Class', 
       xlabel = 'Passenger Class', ylabel = 'Total')
plt.show()
print(pd.crosstab(data["Sex"],data.Survived))
ax = sns.countplot(x = 'Sex', hue = 'Survived', palette = 'Set1', data = data)
ax.set(title = 'Total Survivors According to Sex', xlabel = 'Sex', ylabel='Total')
plt.show()
# We look at Age column and set Intevals on the ages and the map them to their categories as
# (Children, Teen, Adult, Old)
interval = (0,18,35,60,120)
categories = ['Children','Teens','Adult', 'Old']
data['Age_cats'] = pd.cut(data.Age, interval, labels = categories)

ax = sns.countplot(x = 'Age_cats',  data = data, hue = 'Survived', palette = 'Set1')

ax.set(xlabel='Age Categorical', ylabel='Total',
       title="Age Categorical Survival Distribution")

plt.show()
print(pd.crosstab(data['Embarked'], data.Survived))
ax = sns.countplot(x = 'Embarked', hue = 'Survived', palette = 'Set1', data = data)
ax.set(title = 'Survival distribution according to Embarking place')
plt.show()
# print(data.nunique())
data.head()
data.drop(['Name','Ticket','Cabin','PassengerId','Age_cats'], 1, inplace =True)
# data.fillna(0, inplace = True)
data.head()
data.Sex.replace(('male','female'), (0,1), inplace = True)
data.Embarked.replace(('Southampton','Cherbourg','Queenstown'), (0,1,2), inplace = True)
data.Survived.replace(('Died','Survived'), (0,1), inplace = True)
data.head()
plt.figure(figsize=(14,12))
sns.heatmap(data.astype(float).corr(),linewidths=0.1, 
            square=True,  linecolor='white', annot=True)
plt.show()

data.head()
X = np.array(data.drop(['Survived'],1))
y = np.array(data['Survived'])
print("Features shape: ", X.shape)
print("Labels: ", y.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
data.fillna(0, inplace = True)
sns.heatmap(data.isnull())
data.head()
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
