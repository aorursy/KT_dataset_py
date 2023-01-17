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
data = pd.read_csv("../input/heart-disease-uci/heart.csv")

data.head()
data = data.sort_values(by="age")
# check on None

data.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt



# Count patients

age_count = pd.DataFrame(data.groupby("age").sex.count()).reset_index()

age_count = age_count.rename(columns={'age':'age', 'sex':'count_person'})

age_count.head()
age_count[age_count.count_person == age_count.count_person.max()] #most patients aged
plt.figure(figsize=(20,6))

sns.barplot(x=age_count.age, y=age_count.count_person)
age = pd.DataFrame(data.groupby("age").target.sum()).reset_index()

age["ill"] = age.target / age_count.count_person # % ill patients

age.head()
plt.figure(figsize=(20,6))

sns.barplot(x=age.age, y=age.ill)
min_age = data.age.min()

max_age = data.age.max()

n = data.age.count()

print(min_age, max_age, n)



import math



# to determine the size of the interval, we use the Sturges formula



h = round((max_age - min_age) / (1 + math.log(n,2))) 



print(h)



# we divide the initial data into m intervals



m = 1 + math.ceil(math.log(n, 2))



print(m)



# initial value



x_start = min_age - round(h/2)



print(x_start)



# we get intervals



intervals = []

for i in range(m):

    

    interval = [x_start, x_start + h]

    intervals.append(interval)

    x_start = interval[1]



print(intervals)
X = pd.DataFrame()

i = 0

for interval in intervals:

    i = i + 1

    value = []

    if i == m:

        for value_col in data.age:

            if value_col <= int(interval[1]) and value_col >= int(interval[0]):

                val = 1

            else:

                val = 0

            value.append(val)

    else:

        for value_col in data.age:

            if value_col < int(interval[1]) and value_col >= int(interval[0]):

                val = 1

            else:

                val = 0

            value.append(val)

    

    X["["+str(interval[0])+" ,"+str(interval[1]) + ")"] = value
X.head()
X["sex"] = data.sex

X["cp"] = data.cp

X["trestbps"] = data.trestbps

X["chol"] = data.chol

X["fbs"] = data.fbs

X["restecg"] = data.restecg

X["thalach"] = data.thalach

X["exang"] = data.exang

X["oldpeak"] = data.oldpeak

X["slope"] = data.slope

X["ca"] = data.ca

X["thal"] = data.thal
X.head()
data.head()
data.target.unique()
def norm(column):

    x_min = column.min()

    x_max = column.max()

    

    column = (column - x_min) / (x_max - x_min)

    return column
X.cp = norm(X.cp)

X.trestbps = norm(X.trestbps)

X.chol = norm(X.chol)

X.restecg = norm(X.restecg)

X.thalach = norm(X.thalach)

X.oldpeak = norm(X.oldpeak)

X.slope = norm(X.slope)

X.ca = norm(X.ca)

X.thal = norm(X.thal)

X.head()
y = data.target
X.shape[0] == y.shape[0]
from sklearn.model_selection import train_test_split



train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.10, random_state=0)

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score



model = MLPClassifier(solver='adam', alpha=0.00001, hidden_layer_sizes=(150,), random_state=0)



score = cross_val_score(model, train_X, train_y, cv=3)

score
model.fit(train_X, train_y)
predict = model.predict(test_X)



summ = 0

n = predict.size

ys = np.array(test_y)

for i in range(n):

    if predict[i] == ys[i]:

        summ = summ + 1

summ/n

model.fit(X, y)



from joblib import dump

dump(model, 'classificator_heart_disease.joblib')


