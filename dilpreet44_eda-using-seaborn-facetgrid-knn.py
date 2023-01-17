#important modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sea

import plotly.express as px

%matplotlib inline
df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df.info()
df = df.drop(['sl_no'], axis = 1)
df['salary'] = df['salary'].fillna(0)
#let's convert percentage into percentage bands.

# 0 - Less than 60

# 1 - Between 60 and 80

# 2 - Between 80 and 100



df.loc[df['ssc_p'] <= 60, 'ssc_p_band'] = 0

df.loc[(df['ssc_p'] > 60) & (df['ssc_p'] <= 80), 'ssc_p_band'] = 1

df.loc[(df['ssc_p'] > 80) & (df['ssc_p'] <= 100), 'ssc_p_band'] = 2

df = df.drop(['ssc_p'], axis = 1)



df.loc[df['hsc_p'] <= 60, 'hsc_p_band'] = 0

df.loc[(df['hsc_p'] > 60) & (df['hsc_p'] <= 80), 'hsc_p_band'] = 1

df.loc[(df['hsc_p'] > 80) & (df['hsc_p'] <= 100), 'hsc_p_band'] = 2

df = df.drop(['hsc_p'], axis = 1)



df.loc[df['degree_p'] <= 60, 'degree_p_band'] = 0

df.loc[(df['degree_p'] > 60) & (df['degree_p'] <= 80), 'degree_p_band'] = 1

df.loc[(df['degree_p'] > 80) & (df['degree_p'] <= 100), 'degree_p_band'] = 2

df = df.drop(['degree_p'], axis = 1)



df.loc[df['mba_p'] <= 60, 'mba_p_band'] = 0

df.loc[(df['mba_p'] > 60) & (df['mba_p'] <= 80), 'mba_p_band'] = 1

df.loc[(df['mba_p'] > 80) & (df['mba_p'] <= 100), 'mba_p_band'] = 2

df = df.drop(['mba_p'], axis = 1)



df.loc[df['etest_p'] <= 60, 'etest_p_band'] = 0

df.loc[(df['etest_p'] > 60) & (df['etest_p'] <= 80), 'etest_p_band'] = 1

df.loc[(df['etest_p'] > 80) & (df['etest_p'] <= 100), 'etest_p_band'] = 2

df = df.drop(['etest_p'], axis = 1)
df.head()
x = df[(df['gender'] == 'M') & (df['status'] == 'Placed')]

print('Total number of male who got placed = ',x.groupby(['gender'])['status'].count()[0])

print('Maximum salary offered to  male = ',x['salary'].max())

print('Minimum salary offered to  male = ',x['salary'].min())

print('Mean salary offered to  male = ',x['salary'].mean())

sea.kdeplot(x['salary'])



x = df[(df['gender'] == 'F') & (df['status'] == 'Placed')]

print('Total number of female who got placed = ',x.groupby(['gender'])['status'].count()[0])

print('Maximum salary offered to  female = ',x['salary'].max())

print('Minimum salary offered to  female = ',x['salary'].min())

print('Mean salary offered to  female = ',x['salary'].mean())

sea.kdeplot(x['salary'])

plt.legend(['Male', 'Female'])
print(df['ssc_b'].value_counts())

print(df.groupby(['ssc_p_band'])['salary'].mean())

grid = sea.FacetGrid(df, row = 'ssc_b', col = 'ssc_p_band')

grid.map(sea.countplot, 'status', order = ['Placed', 'Not Placed'], color = 'cyan')
print(df['hsc_b'].value_counts())

print(df['hsc_s'].value_counts())

print(df.groupby(['hsc_p_band'])['salary'].mean())

print(df.groupby(['hsc_s'])['salary'].mean())

grid = sea.FacetGrid(df, row = 'hsc_b', col = 'hsc_p_band', hue = 'hsc_s')

grid.map(sea.countplot, 'status', order = ['Placed', 'Not Placed']).add_legend()
print(df['degree_t'].value_counts())

print(df.groupby(['degree_p_band'])['salary'].mean())

print(df.groupby(['degree_p_band'])['salary'].mean())

fig, ax = plt.subplots(1,2, figsize = (10,4))

sea.countplot(df['degree_t'], hue = df['status'], ax = ax[0])

sea.countplot(df['degree_p_band'], hue = df['status'], ax = ax[1])

grid = sea.FacetGrid(df, col = 'degree_p_band', height = 4)

grid.map(plt.hist, 'salary', bins = 20, color = 'green')
print(df['workex'].value_counts())

print(df.groupby(['workex'])['salary'].mean())

sea.countplot('workex', hue = 'status', data = df)

grid = sea.FacetGrid(df, col = 'workex', height = 4)

grid.map(plt.hist, 'salary', bins = 20, color = 'red')
print(df['specialisation'].value_counts())

print(df.groupby(['specialisation'])['salary'].mean())

sea.countplot('specialisation', hue = 'status', data = df)

grid = sea.FacetGrid(df, col = 'specialisation', height = 4)

grid.map(plt.hist, 'salary', bins = 20, color = 'orange')
print(df.groupby(['mba_p_band'])['salary'].mean())

print(df.groupby(['etest_p_band'])['salary'].mean())

fig, ax = plt.subplots(1,2, figsize = (10,4))

sea.countplot(df['mba_p_band'], hue = df['status'], ax = ax[0], palette = 'Set2')

sea.countplot(df['etest_p_band'], hue = df['status'], ax = ax[1], palette = 'Set3')
df.drop(['ssc_b','hsc_b', 'salary'], axis=1, inplace=True)

df["gender"] = df.gender.map({"M":0,"F":1})

df["workex"] = df.workex.map({"No":0, "Yes":1})

df["specialisation"] = df.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})

df["status"] = df.status.map({"Not Placed":0, "Placed":1})

df["hsc_s"] = df.hsc_s.map({"Commerce":0, "Science":1, "Arts" : 2})

df["degree_t"] = df.degree_t.map({"Sci&Tech":0, "Comm&Mgmt":1, "Others" : 2})
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(df.drop(['status'], axis = 1), df['status'],test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(train_x,train_y)

    pred_i = knn.predict(test_x)

    error_rate.append(np.mean(pred_i != test_y))



plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 

         marker='o',markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))



acc = []

# Will take some time

from sklearn import metrics

for i in range(1,40):

    neigh = KNeighborsClassifier(n_neighbors = i).fit(train_x, train_y)

    yhat = neigh.predict(test_x)

    acc.append(metrics.accuracy_score(test_y, yhat))

    

plt.figure(figsize=(10,6))

plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 

         marker='o',markerfacecolor='red', markersize=10)

plt.title('accuracy vs. K Value')

plt.xlabel('K')

plt.ylabel('Accuracy')

print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))