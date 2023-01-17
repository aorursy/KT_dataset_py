import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/flavors_of_cacao.csv')
df.head()
df.columns
df.columns = df.columns.str.replace("\\n","-").str.replace(" ","-").str.strip(" ")

df.columns
df.columns
df['Review-Date'] = pd.to_datetime(df['Review-Date'],format="%Y")
df.to_csv("ChocolateReviews.csv.gz",index=False,compression="gzip")
df.info()
df.describe()
df.isnull().sum()

sns.heatmap(df.isnull(), cbar = False, cmap='coolwarm')
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
rfc = RandomForestClassifier(n_estimators=20)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,rfc_pred))
print(accuracy_score(y_test,rfc_pred)*100)