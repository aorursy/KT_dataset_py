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
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, neighbors
from sklearn.preprocessing import StandardScaler
def broad_analysis(data):
    print('shape of the dataset')
    print(data.shape)
    print('============================================================')
    print('============================================================')
    print('columns in the dataset')
    print(data.columns)
    print('============================================================')
    print('============================================================')
    print('infos on the dataset')
    i = 0
    features_list = df.columns
    while i < len(df.columns):
        print(features_list[i])
        print(df[str(features_list[i])].unique())
        i += 1
    print('============================================================')
    print('============================================================')
    print('infos on the type repartition')
    print(df.dtypes.value_counts())
    print('============================================================')
    print('============================================================')
    print(data.info())
    print('============================================================')
    print('============================================================')
    print('head')
    print(data.head())
    print('============================================================')
    print('============================================================')
    print('tail')
    print(data.tail())
    print('============================================================')
    print('============================================================')    
    print('null data')
    print(data.isnull().any())
    print('============================================================')
    print('============================================================')
    print('description')
    print(np.round(data.describe()))

def visualise_correlation(data):
    correlation = data.corr()
    plt.figure(figsize=(14, 12))
    heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")

def format_data(d,df,feature):
    df[feature] = df[feature].map(d)
    return df

def visualise_specific_correlation(df,x_axis,y_axis):
    specific_data = df[[str(y_axis), str(x_axis)]]
    gridA = sns.JointGrid(x=x_axis, y=y_axis, data=specific_data, size=6)
    gridA = gridA.plot_joint(sns.regplot, scatter_kws={"s": 10})
    gridA = gridA.plot_marginals(sns.distplot)
    
def visualise_correlation(data):
    correlation = data.corr()
    plt.figure(figsize=(14, 12))
    heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
    
def class_analysis(df,class_number):
    df = df.loc[df['Pclass'] == class_number]
    print(df.info())
    print(df.isnull().any())
    ax = df['Survived'].plot.hist(bins=12, alpha=0.5)
    plt.show()
    ax = df['Age'].plot.hist(bins=12, alpha=0.5)
    plt.show()
    ax = df['SibSp'].plot.hist(bins=12, alpha=0.5)
    plt.show()
    ax = df['Parch'].plot.hist(bins=12, alpha=0.5)
    plt.show()
    ax = df['Fare'].plot.hist(bins=12, alpha=0.5)
    plt.show()
    ax = df['Sex'].plot.hist(bins=12, alpha=0.5)
    plt.show()
    visualise_correlation(df[['Survived', 'Sex', 'Age', 'SibSp','Parch', 'Fare']])
df = pd.read_csv('/kaggle/input/titanic/train.csv')
broad_analysis(df)
df.dropna(subset = ["Embarked"], inplace=True)
df = df.drop_duplicates()
df.shape
ax = df['Survived'].plot.hist(bins=12, alpha=0.5)
ax = df['Pclass'].plot.hist(bins=12, alpha=0.5)
ax = df['Age'].plot.hist(bins=12, alpha=0.5)
ax = df['SibSp'].plot.hist(bins=12, alpha=0.5)
ax = df['Parch'].plot.hist(bins=12, alpha=0.5)
ax = df['Fare'].plot.hist(bins=12, alpha=0.5)
d = {'male' : 0, 'female' : 1}
feature = 'Sex'
format_data(d,df,feature)
i = 0
names = df.Name.values.tolist()
title_list = ["Mr", "Mrs", "Miss", "Master", "Don", "Rev"]
denomination = []
while i < len(names):
    matches = [c for c in title_list if c in names[i]]
    denomination.append(matches)
    i += 1
i = 0
while i < len(denomination):
    if len(denomination[i]) > 0:
        denomination[i] = denomination[i][-1]
    else:
        denomination[i] = 'No'
    i += 1
print(denomination)
ax = df['Sex'].plot.hist(bins=12, alpha=0.5)
import plotly.express as px
fig = px.scatter_matrix(df,
    dimensions=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare'],
    color="Survived")
fig.show()
visualise_correlation(df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare']])
class_analysis(df,1)
class_analysis(df,2)
class_analysis(df,3)
def age_bucket(age): 
    if age > 0 and age < 12:
        age = 0
    elif age > 12 and age < 40:
        age = 1
    elif age > 40 and age < 60:
        age = 2
    elif age > 60:
        age = 3
    else:
        age = -100
    return int(age)

age = df.Age.values.tolist()
bucket = list(map(age_bucket, age))
df = {'Survived':df.Survived.values.tolist(),'Sex':df.Sex.values.tolist(),'Age':bucket,'SibSp':df.SibSp.values.tolist(),'Pclass':df.Pclass.values.tolist(),'Parch':df.Parch.values.tolist(), 'Title':denomination}
df = pd.DataFrame.from_dict(df)
df.columns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
data = df[['Sex', 'Age', 'SibSp', 'Pclass', 'Parch']]
data = data.dropna()
data = data[data.Age != -100]
mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)
data_transformed.shape
Sum_of_squared_distances = []
inertia = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
    inertia.append(km.inertia_)
print(inertia)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
X = np.array(data.drop(['Age'], 1))
y = np.array(data['Age'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = neighbors.KNeighborsClassifier(n_neighbors=5)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)


clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
data_missing_age = df[df.Age == -100]
data_missing_age = data_missing_age[['Sex', 'Age', 'SibSp', 'Pclass', 'Parch']]
X_to_predict = np.array(data_missing_age.drop(['Age'], 1))
X_to_predict = sc.fit_transform(X_to_predict)
clf.predict(X_to_predict)

