import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore', category=FutureWarning)



plt.style.use("seaborn-dark")

np.random.seed(42)
pacific = pd.read_csv("../input/hurricane-database/pacific.csv").drop(['ID', 'Name'], axis=1)

pacific.head()
pacific.shape
pacific.columns
pacific.isna().sum().any()
pacific.Status.unique()
sns.countplot(x='Status', data=pacific)

print(pacific.Status.value_counts())
pacific.loc[:,'Status'] = pacific['Status'].str.strip()
pacific.Event.unique()
plt.figure(figsize=(17,5))

sns.countplot(pacific.Event, hue=pacific.Status)

plt.legend(loc='upper right')
pacific.Time.unique()
plt.figure(figsize=(17,5))

sns.countplot(pacific.Time, hue=pacific.Status)

plt.legend('')

plt.xticks(rotation=90)

plt.show()
a = pacific.groupby(by='Status')['Time'].mean()

sns.barplot(x=a.index, y=a)
date = pd.to_datetime(pacific['Date'], format='%Y%m%d')

pacific['Year'] = pd.DatetimeIndex(date).year

pacific['Month'] = pd.DatetimeIndex(date).month

pacific['Day'] = pd.DatetimeIndex(date).day
cols = ['Year', 'Month', 'Day']

for i, col in enumerate(cols):

    plt.figure(i)

    plt.figure(figsize=(17,5))

    sns.countplot(x=col, hue='Status', data=pacific)

    plt.legend('')

    plt.xticks(rotation=90)
cols = ['Year', 'Month', 'Day']

for i, col in enumerate(cols):

    plt.figure(i)

    sns.scatterplot(pacific[col], pacific['Status'])
a = pacific['Latitude'].unique()[0]

str(a).replace('N', '')
directions =['N', 'S', 'E', 'W']

for dir_ in directions:

    pacific.loc[:,'Latitude'] = pacific['Latitude'].apply(lambda x : str(x).replace(dir_,''))

    pacific.loc[:,'Longitude'] = pacific['Longitude'].apply(lambda x : str(x).replace(dir_,''))
sns.distplot(pacific['Latitude'].astype('float'))
sns.distplot(pacific['Longitude'].astype('float'))
sns.distplot(pacific['Maximum Wind'])
# colums = ['Minimum Pressure', 'Low Wind NE', 'Low Wind SE',

#        'Low Wind SW', 'Low Wind NW', 'Moderate Wind NE', 'Moderate Wind SE',

#        'Moderate Wind SW', 'Moderate Wind NW', 'High Wind NE', 'High Wind SE',

#        'High Wind SW', 'High Wind NW']



# for i, col in enumerate(colums):

#     plt.figure(i)

#     sns.distplot(pacific[col])
features = ['Time', 'Status', 'Latitude', 'Longitude',

       'Maximum Wind', 'Minimum Pressure', 'Low Wind NE', 'Low Wind SE',

       'Low Wind SW', 'Low Wind NW', 'Moderate Wind NE', 'Moderate Wind SE',

       'Moderate Wind SW', 'Moderate Wind NW', 'High Wind NE', 'High Wind SE',

       'High Wind SW', 'High Wind NW', 'Year', 'Month', 'Day']



label = 'Status'
from sklearn.preprocessing import LabelEncoder



y = pacific['Status']



le = LabelEncoder()

le.fit(y)

pacific.loc[:,'Status'] = le.transform(y)

print(le.classes_)
X = pacific[features]

y = pacific[label]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)

pca.fit(X)

print("---Explained Variance Ratio---")

print(pca.explained_variance_ratio_.sum()*100)

X_pca = pca.transform(X)
X_pca.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X, y,

                                                   stratify = y,

                                                   test_size = 0.20)
from sklearn.linear_model import LogisticRegression



model = LogisticRegression(max_iter=30000).fit(X_train, y_train)



# print(model.coef_)

# print(model.intercept_)



# y_pred = model.predict(X_test)





#Init

model_for_cv = model



from sklearn.model_selection import cross_val_score

scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='f1_macro')

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()



model_for_cv = clf



from sklearn.model_selection import cross_val_score

scores = cross_val_score(model_for_cv, X_train, y_train, cv=5, scoring='f1_macro')

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)



from sklearn.metrics import classification_report

print("----Classification Report----")

print(classification_report(y_test, y_pred))