import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split
df=pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()
df.info()
for col in df.columns.unique():

    print('\n', col ,'\n', df[col].unique())
df.columns
for cols in df.columns.unique():

    print('\n', cols, '\n', df[cols].unique())
df1=df.copy()
df.isna()
df.info()
df=df.fillna(0)
df.info()
fig, axs = plt.subplots(ncols=3,figsize=(20,5))

sns.countplot(df['workex'], ax = axs[0], palette="Paired")

sns.countplot(df['specialisation'], ax = axs[1], palette="muted")

sns.countplot(df['status'], ax = axs[2],palette="dark")
df=df.drop(["sl_no"],axis=1)
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
df.head()
sns.heatmap(df.corr())
df.corr()
sns.scatterplot(x='status', y = 'degree_p', hue ='gender', data = df1)
fig, axs = plt.subplots(ncols=5,figsize=(25,5))

sns.distplot(df1['degree_p'], ax= axs[0], color = 'g')

sns.distplot(df1['hsc_p'], ax= axs[1])

sns.distplot(df1['ssc_p'],  ax= axs[2], color = 'b')

sns.distplot(df1['etest_p'],  ax= axs[3], color = 'r')

sns.distplot(df1['mba_p'],  ax= axs[4], color = 'c')
plt.hist(df1['salary'], bins = 10)

plt.show()







#graph of salary of students ranging
fig, axs = plt.subplots(ncols=3,figsize=(20,5))

sns.scatterplot(x = 'degree_p',y='hsc_p',hue='status',data = df1, ax= axs[0])

sns.scatterplot(x = 'degree_p',y='hsc_p',hue='gender',data = df1, ax= axs[1], palette="muted")

sns.scatterplot(x = 'degree_p',y='hsc_p',hue='degree_t',data = df1, palette="dark", ax= axs[2])
fig, axs = plt.subplots(ncols=3,figsize=(20,5))

sns.scatterplot(x = 'degree_p',y='hsc_p',hue='hsc_s',data = df1, ax= axs[0])

sns.scatterplot(x = 'degree_p',hue='specialisation',y='mba_p',data = df1, ax= axs[1], palette="muted")

sns.scatterplot(x = 'degree_p',hue='workex',y='salary',data = df1, palette="dark", ax= axs[2])
# new dataframe to store salary





df=pd.DataFrame(df)

df1
df_placed = df1[df1['status'] == 'Placed']

df_placed
df2 = df_placed.groupby(['degree_t','degree_p','mba_p','specialisation','hsc_p', 'hsc_s','salary', 'workex']).sum().sort_values(by ='salary')

df2
df_unplaced = df1[df1['status'] != 'Placed']

df_unplaced
df3 = df_unplaced.groupby(['degree_t','degree_p','mba_p','specialisation','hsc_p', 'hsc_s','workex']).sum().sort_values(by ='degree_p')

df3
X = df.drop(['status'], axis = 1)

y = df['status']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 101)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 1)

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
error_rate = []



# Will take some time

for i in range(1,40):

    

    classifier = KNeighborsClassifier(n_neighbors=i)

    classifier.fit(X_train,y_train)

    pred_i = classifier.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')