# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('ggplot')
df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

df.head()
for col in df.columns.unique():

    print('\n', col ,'\n', df[col].unique())
df1 = df.copy()
df.isna().any()
df = df.fillna(0)
df.info()
fig, axs = plt.subplots(ncols=4,figsize=(20,5))

sns.countplot(df['gender'], ax = axs[0])

sns.countplot(df['ssc_b'], ax = axs[1], palette="vlag")

sns.countplot(df['hsc_b'], ax = axs[2], palette="rocket")

sns.countplot(df['hsc_s'], ax = axs[3], palette="deep")

fig, axs = plt.subplots(ncols=3,figsize=(20,5))

sns.countplot(df['workex'], ax = axs[0], palette="Paired")

sns.countplot(df['specialisation'], ax = axs[1], palette="muted")

sns.countplot(df['status'], ax = axs[2],palette="dark")

df = df.drop(['sl_no'], axis = 1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

df['ssc_b'] = le.fit_transform(df['ssc_b'])

df['workex'] = le.fit_transform(df['workex'])

df['specialisation'] = le.fit_transform(df['specialisation'])

df['status'] = le.fit_transform(df['status'])

df['hsc_b'] = le.fit_transform(df['hsc_b'])

df['hsc_s'] = le.fit_transform(df['hsc_s'])

df['degree_t'] = le.fit_transform(df['degree_t'])

plt.figure(figsize=(15,10))

corr = df.corr()

sns.heatmap(corr, annot = True)
plt.figure(figsize=(5,5))

sns.scatterplot(x='status', y = 'degree_p', hue ='gender', data = df1)
fig, axs = plt.subplots(ncols=5,figsize=(25,5))

sns.distplot(df1['degree_p'], ax= axs[0], color = 'g')

sns.distplot(df1['hsc_p'], ax= axs[1])

sns.distplot(df1['ssc_p'],  ax= axs[2], color = 'b')

sns.distplot(df1['etest_p'],  ax= axs[3], color = 'r')

sns.distplot(df1['mba_p'],  ax= axs[4], color = 'c')

plt.figure(figsize=(7,7))

plt.hist(df1['salary'], bins = 10)

plt.show()
fig, axs = plt.subplots(ncols=3,figsize=(20,5))

sns.scatterplot(x = 'degree_p',y='hsc_p',hue='status',data = df1, ax= axs[0])

sns.scatterplot(x = 'degree_p',y='hsc_p',hue='gender',data = df1, ax= axs[1], palette="muted")

sns.scatterplot(x = 'degree_p',y='hsc_p',hue='degree_t',data = df1, palette="dark", ax= axs[2])

fig, axs = plt.subplots(ncols=3,figsize=(20,5))

sns.scatterplot(x = 'degree_p',y='hsc_p',hue='hsc_s',data = df1, ax= axs[0])

sns.scatterplot(x = 'degree_p',hue='specialisation',y='mba_p',data = df1, ax= axs[1], palette="muted")

sns.scatterplot(x = 'degree_p',hue='workex',y='salary',data = df1, palette="dark", ax= axs[2])
df = pd.DataFrame(df)
df_placed = df1[df1['status'] == 'Placed']

df2 = df_placed.groupby(['degree_t','degree_p','mba_p','specialisation','hsc_p', 'hsc_s','salary', 'workex']).sum().sort_values(by ='salary')

df2
df_np = df1[df1.status == 'Not Placed']



df3 = df_np.groupby(['degree_t','degree_p','mba_p','specialisation','hsc_p', 'hsc_s','workex']).sum().sort_values(by ='degree_p')

df3
X = df.drop(['status'], axis = 1)

y = df['status']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 43)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 8)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

result = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")

print(result)

result1 = classification_report(y_test, y_pred)

print("Classification Report:",)

print (result1)

result2 = accuracy_score(y_test,y_pred)

print("Accuracy:",result2)