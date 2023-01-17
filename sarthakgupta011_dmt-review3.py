# This Python 3 environment comes with many helpful analytics libraries installed

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/denver-crime-data/crime.csv', index_col='INCIDENT_ID')
print(df.shape)
df.head()
df.describe()
df.isna().sum()
df.drop(['OFFENSE_ID', 'GEO_X', 'GEO_Y', 'LAST_OCCURRENCE_DATE'], axis=1, inplace=True)
# feature engineering
df['FIRST_OCCURRENCE_DATE'] = pd.to_datetime(df['FIRST_OCCURRENCE_DATE'])
df['YEAR'] = df['FIRST_OCCURRENCE_DATE'].dt.year
df['MONTH'] = df['FIRST_OCCURRENCE_DATE'].dt.month
df['DAY'] = df['FIRST_OCCURRENCE_DATE'].dt.day
df['HOUR'] = df['FIRST_OCCURRENCE_DATE'].dt.hour
df['OFFENSE_CATEGORY_ID'].value_counts()[:15].sort_values(ascending=True).plot(kind='barh', 
                                                                               title='OFFENSE_CATEGORY_ID')
df = df[~df['OFFENSE_CATEGORY_ID'].isin(['traffic-accident', 'all-other-crimes'])]
df.shape
# Captures 504,098 out of 508,459 rows of data (99%). The rest was outliers and/or misclassified.

df = df[(df['GEO_LON'] < -50) & (df['GEO_LAT'] > 38)]

plt.figure(figsize=(12,10))
ax = sns.scatterplot(x='GEO_LON',y='GEO_LAT', data=df)
df.shape
## district separation ##
plt.figure(figsize=(10,10))
sns.scatterplot(x='GEO_LON', 
                y='GEO_LAT', 
                alpha=0.5,
                hue='DISTRICT_ID',
                palette=plt.get_cmap('jet'),
                legend='full',
                data=df
               )

## if data is numerical (not categorical) use this instead ##
# df.plot(kind='scatter', 
#         x='GEO_LON', 
#         y='GEO_LAT', 
#         figsize=(10,10),
#         alpha=0.5,
#         c='DISTRICT_ID',
#         cmap=plt.get_cmap('jet'),
#         colorbar=True,
#         sharex=False
#        )
offense_cats = df['OFFENSE_CATEGORY_ID'].value_counts()[:10].index

plt.figure(figsize=(12,10))
sns.scatterplot(x='GEO_LON',
                y='GEO_LAT', 
                hue='OFFENSE_CATEGORY_ID', 
                data=df[df['OFFENSE_CATEGORY_ID'].isin(offense_cats)])
plt.figure(figsize=(12,10))
sns.scatterplot(x='GEO_LON',
                y='GEO_LAT', 
                hue='DISTRICT_ID', 
                data=df[(df['OFFENSE_CATEGORY_ID'] == 'murder')])
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
fig, axes = plt.subplots(2, 2, figsize=(20, 10))

df.groupby('HOUR').count()['OFFENSE_CODE'].plot(kind='bar', title='Crimes Per Hour', ax=axes[0,0])
df.groupby('DAY').count()['OFFENSE_CODE'].plot(kind='bar', title='Crimes Per Day', ax=axes[0,1])
df.groupby('MONTH').count()['OFFENSE_CODE'].plot(kind='bar', title='Crimes Per Month', ax=axes[1,0])
df.groupby('YEAR').count()['OFFENSE_CODE'].plot(kind='bar', title='Crimes Per Year', ax=axes[1,1])
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

features = ['GEO_LAT', 'GEO_LON', 'DISTRICT_ID']
df2 = df[df['OFFENSE_CATEGORY_ID'] == 'murder'][features]

X = df2[features[:2]].values
y = df2[features[-1]].values
y = np.reshape(y, (df2.shape[0], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
params = {'n_neighbors':range(1,6)}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, params, cv=5)
clf.fit(X_train, y_train)

knn_model = clf.best_estimator_
y_pred = knn_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
df_pred = pd.DataFrame(np.column_stack((X_test, y_pred)), columns=['GEO_LAT', 'GEO_LON', 'DISTRICT_ID'])
df_test = pd.DataFrame(np.column_stack((X_test, y_test)), columns=['GEO_LAT', 'GEO_LON', 'DISTRICT_ID'])
cmap_pred = sns.cubehelix_palette(dark=.9, light=.1, as_cmap=True)

fig, axes = plt.subplots(2, 2, figsize=(20, 10))
plt.title('Murder Occurences by District (Prediction vs Test)')

axes[0,1].title.set_text('Murder Occurences by District (Prediction vs Test)')
axes[1,0].title.set_text('Murder Occurences by District (Test)')
axes[1,1].title.set_text('Murder Occurences by District (Prediction)')

sns.scatterplot(x='GEO_LON',
                y='GEO_LAT', 
                hue='DISTRICT_ID', 
                legend='full',
                palette='Set2',
                alpha=0.5,
                ax=axes[0,1],
                data=df_pred)

sns.scatterplot(x='GEO_LON',
                y='GEO_LAT', 
                hue='DISTRICT_ID', 
                legend='full',
                palette='Set1',
                alpha=0.5,
                ax=axes[0,1],
                data=df_test)

sns.scatterplot(x='GEO_LON',
                y='GEO_LAT', 
                hue='DISTRICT_ID', 
                legend='full',
                palette='Set2',
                ax=axes[1,0],
                data=df_pred)

sns.scatterplot(x='GEO_LON',
                y='GEO_LAT', 
                hue='DISTRICT_ID', 
                legend='full',
                palette='Set1',
                ax=axes[1,1],
                data=df_test)