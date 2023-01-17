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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/diabetes/diabetes.csv")
df.head()
df.shape
df.isnull().sum()
sns.heatmap(df.corr())
sns.countplot(df['Outcome'])
sns.pairplot(df, hue = "Outcome")
features = [df.columns]

features
plt.figure(figsize=(20,10))

sns.boxplot(data =df, orient='v', palette='rainbow')
index_s = df[df['SkinThickness'] == 0].index.values

for i in index_s:

    df['SkinThickness'][i] = df['SkinThickness'].mean()
index_I = df[df['Insulin'] == 0].index.values

for i in index_I:

    df['Insulin'][i] = df['Insulin'].mean()
index_g = df[df['Glucose'] == 0].index.values

for i in index_g:

    df['Glucose'][i] = df['Glucose'].mean()
index_b = df[df['BloodPressure'] == 0].index.values

for i in index_b:

    df['BloodPressure'][i] = df['BloodPressure'].mean()
index_bmi = df[df['BMI'] == 0].index.values

for i in index_bmi:

    df['BMI'][i] = df['BMI'].mean()
index_d = df[df['Pregnancies'] == 0].index.values

len(index_d)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Outcome', axis = 1))
scaled_features = scaler.transform(df.drop('Outcome', axis = 1))
scaled_data = pd.DataFrame(scaled_features, columns = df.columns[ : -1])
scaled_data.head()
plt.figure(figsize=(20,10))

sns.boxplot(data = scaled_data, orient='v', palette='rainbow')
scaled_data['Insulin'].describe()
feature_data = scaled_data.columns
for i in feature_data:

    q1 = scaled_data[i].describe()['25%']

    q3 = scaled_data[i].describe()['75%']

    IQR = q3 - q1

    high = q3 + (1.5*IQR)

    out_data_1 = scaled_data[scaled_data[i] > high]

    idx_1 = out_data_1.index.values

    

    for j in idx_1:

        scaled_data[i].iloc[j] = high

    

    low = q3 - (1.5*IQR)

    out_data_2 = scaled_data[scaled_data[i] < low]

    idx_2 = out_data_2.index.values

    

    for k in idx_2:

        scaled_data[i].iloc[k] = low
plt.figure(figsize=(20,10))

sns.boxplot(data = scaled_data, orient='v', palette='rainbow')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_data, df['Outcome'], test_size = 0.3)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
predict = knn.predict(X_test)
predict
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(X_train, y_train)

    prad_i = knn.predict(X_test)

    error_rate.append(np.mean(prad_i != y_test))
error_rate.index(min(error_rate))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,'g-', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, y_train)
predictt = knn.predict(X_test)

print(confusion_matrix(y_test, predictt))
print(classification_report(y_test, predictt))