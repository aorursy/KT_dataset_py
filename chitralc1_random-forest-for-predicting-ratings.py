import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/flavors_of_cacao.csv')
df.head()
df.info()
df.describe()
df.isnull()

sns.heatmap(df.isnull(), cbar = False, cmap='coolwarm')
df.columns
df['Bean\nType'].value_counts()
df['Bean\nType'].nunique()
sns.heatmap(df.corr())
df.columns
print('Unique Values:')

print('Company (Maker-if known): ',df['Company\xa0\n(Maker-if known)'].nunique())

print('Specific Bean Origin or Bar Name: ', df['Specific Bean Origin\nor Bar Name'].nunique())

print('Company Location: ',df['Company\nLocation'].nunique())

print('Bean Type: ', df['Bean\nType'].nunique())

print('Broad Bean Origin', df['Broad Bean\nOrigin'].nunique())

print('Review Date: ', df['Review\nDate'].nunique())

print('Cocoa Percent: ', df['Cocoa\nPercent'].nunique())
sns.countplot(x = df['Rating'])
sns.countplot(x = df['Review\nDate'])
sns.jointplot(x = 'Rating', y= 'Review\nDate', data = df, kind='kde', color = 'brown')
df['Cocoa\nPercent'] = df['Cocoa\nPercent'].str.replace('%', '')

df['Cocoa\nPercent'] = df['Cocoa\nPercent'].str.replace('.', '')

df['Cocoa\nPercent'] = df['Cocoa\nPercent'].astype(int)
plt.figure(figsize=(15,7))

sns.countplot(x= 'Cocoa\nPercent', data = df, color = 'brown')
def normalizeIt(percent):

    if percent > 100:

        percent = int(str(percent)[:2])

    return percent
df['Cocoa\nPercent'] = df['Cocoa\nPercent'].apply(normalizeIt)
plt.figure(figsize=(15,7))

sns.countplot(x= 'Cocoa\nPercent', data = df, color = 'brown')
df['Rating'] = (df['Rating']* 100).astype(int)

df['Rating'].head(5)
df.columns
company = pd.get_dummies(df['Company\xa0\n(Maker-if known)'],drop_first=True)

sbOrigin = pd.get_dummies(df['Specific Bean Origin\nor Bar Name'],drop_first=True)

companyLocation = pd.get_dummies(df['Company\nLocation'],drop_first=True)

bType = pd.get_dummies(df['Bean\nType'],drop_first=True)

bbOrigin = pd.get_dummies(df['Broad Bean\nOrigin'],drop_first=True)
df = pd.concat([df, company, sbOrigin, companyLocation, bType, bbOrigin], axis = 1)
df.drop(['Company\xa0\n(Maker-if known)', 'Specific Bean Origin\nor Bar Name','Company\nLocation', 'Bean\nType', 

         'Broad Bean\nOrigin'], axis = 1, inplace = True )
df = df.loc[:,~df.columns.duplicated()]

from sklearn.model_selection import train_test_split
X = df.drop('Rating', axis = 1) #Features

y = df['Rating']   # Target Variables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)
df.columns
df['Venezuela'].head(5)
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,rfc_pred))
print(accuracy_score(y_test,rfc_pred)*100)
sns.countplot(x = 'Rating', data=df)
def rating_to_stars(rating):

    

    rating = int(rating)

    

    if (rating == 0.0 ):

        return 0.0

    elif (rating > 0 ) and (rating <= 199 ):

        return 1.0

    elif (rating >= 200 ) and (rating <= 299 ):

        return 2.0

    elif (rating >= 300 ) and (rating <= 399 ):

        return 3.0

    else:

        return 4.0
df['Rating'] = df['Rating'].apply(rating_to_stars)
sns.countplot(x = 'Rating', data=df)
X = df.drop('Rating', axis = 1)

y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)
rfc = RandomForestClassifier(n_estimators=5000, min_weight_fraction_leaf= 0)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print(accuracy_score(y_test,rfc_pred)*100)