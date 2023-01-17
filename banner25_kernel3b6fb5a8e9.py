import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm
data = pd.read_csv("/input/train.csv", sep=",");
Basic_data = data

data.head(5)
data.info()
#data = data.drop('Weaks', axis=1)
data.duplicated().sum()
data['Class'].unique()
data.head()
import seaborn as sns

f, ax = plt.subplots(figsize=(44, 42))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, annot = True);

plt.show()

colname = 'ID'

for i in range(len(corr.columns)):

    #print('-a')

    for j in range(i):

        #print('-b')

        if (corr.iloc[i, j] >= 0.75):

           # print('-c')

            colname = corr.columns[i]

            print(colname)

            #data = data.drop([corr.columns[i]], axis=1)            
data = data.drop(colname, axis=1)
data = data.drop('ID', axis=1)
data.head()

data = data.drop(['Enrolled', 'MLU', 'Reason', 'Area', 'State', 'Fill'], axis=1)
data = data.drop('PREV', axis=1)
y=data['Class']

x=data.drop(['Class'],axis=1)

x.head()

def convert(data):

    col = data.columns.values

    for i in col:

        null = {}

        def int_con(string):

            return null[string]

        if data[i].dtype != np.int64 and data[i].dtype != np.float64:

            val = set(data[i].values.tolist())

            a = 0

            for j in val:

                if j not in null:

                    null[j] = a

                    a = a+1

            data[i] = list(map(int_con,data[i]))

    return data
x = convert(x)

x.head()
min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(x)

x = pd.DataFrame(np_scaled)

x.head()
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.4, random_state = 40)

rf = RandomForestClassifier()

rf.fit(X_train, Y_train)
print("Accuracy: ",rf.score(X_test, Y_test))
test = pd.read_csv("/input/test_1.csv", sep=",");
test = test.drop(['Enrolled', 'MLU', 'Reason', 'Area', 'State', 'Fill'], axis=1)
test = test.drop(['PREV'], axis=1)
ID_test = test
test = test.drop(['ID'], axis=1)
test = test.drop(colname, axis=1)
test = convert(test)



min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(test)

test = pd.DataFrame(np_scaled)

test.head()
print(Basic_data['ID'])
ID = set()

prediction = rf.predict(test)



np.savetxt('/home/kapish/Documents/3-2/DM/Assignment2/assignment2.csv',prediction ,delimiter=',')