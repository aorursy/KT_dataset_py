import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import animation

from IPython.display import clear_output
data = pd.read_csv('../input/train.csv')

data
gender_data = data['Sex'].values

gender_data = np.array([1 if i == 'male' else 0 for i in gender_data])

age_data = data['Age'].values

age_data = [0 if np.isnan(i) else i for i in age_data]

mean_age = sum(age_data)/len(age_data) # We will assign this mean age to all unavialable ages of passengers

age_data = np.array([mean_age if i == 0 else i for i in age_data])

alive_data = data['Survived'].values

fare_data = data['Fare'].values

sibsp_data = data['SibSp'].values

pclass_data = data['Pclass'].values

embarked_data = data['Embarked'].values



for _x in ['C','Q','S']:

    print(_x+':', len([i for i in embarked_data if i == _x]))

'''

Result:

C: 168

Q: 77

S: 644

'''

'Hence `S` has more number of lower class(income) people!'

'Hence setting all NaN valued embarkment of Passenger to `S`'

'And replacing Character with integer ID'

for i in range(len(embarked_data)):

    if embarked_data[i] == 'Q':

        embarked_data[i] = 0

    elif embarked_data[i] == 'C':

        embarked_data[i] = 1

    else:

        embarked_data[i] = 2
n_male = sum([1 if i == 1 else 0 for i in gender_data])

n_female = len(gender_data) - n_male

plt.title("Passengers Composition")

plt.bar(['male', 'female'], [n_male, n_female], color=['cyan', 'purple'])
n_male_alive = 0

n_female_alive = 0



for i in range(len(alive_data)):

    if alive_data[i] == 1:

        if gender_data[i] == 1:

            n_male_alive += 1

        elif gender_data[i] == 0:

            n_female_alive += 1

            

plt.title("Survival Rate of Passengers")

plt.bar(['male', 'female'], [n_male_alive/n_male, n_female_alive/n_female], color=['cyan', 'purple'])
plt.title(r"Age & Fare correlation with being alive")



x = [age_data[i] for i in range(len(age_data)) if alive_data[i] == 0]

y = [fare_data[i] for i in range(len(fare_data)) if alive_data[i] == 0]

color = [0]*(len(gender_data)-n_female_alive-n_male_alive)

dead_sc = plt.scatter(x, y, c='purple', alpha=0.8)



x = [age_data[i] for i in range(len(age_data)) if alive_data[i] == 1]

y = [fare_data[i] for i in range(len(fare_data)) if alive_data[i] == 1]

color = [1]*(n_female_alive+n_male_alive)

alive_sc = plt.scatter(x, y, c='c', alpha=0.5)



plt.legend(

    (alive_sc,dead_sc),

    ('Alive', 'Dead')

          )

plt.annotate('Rich Arses', xy=(38,490), xytext=(50, 400), arrowprops={'facecolor': 'red'})

plt.xlabel(r'Age $\longrightarrow$')

plt.ylabel(r'Fare $\longrightarrow$')
plt.title("Passenger Class correlation with Survival")

c1, c2, c3 = [0] * 3

alive_c1, alive_c2, alive_c3 = [0] * 3

for i in range(len(pclass_data)):

    if pclass_data[i] == 1:

        c1 += 1

        if alive_data[i]:

            alive_c1 += 1

    elif pclass_data[i] == 2:

        c2 += 1

        if alive_data[i]:

            alive_c2 += 1

    elif pclass_data[i] == 3:

        c3 += 1

        if alive_data[i]:

            alive_c3 += 1

        

plt.bar(['Class1', 'Class1 Alive', 'Class2', 'Class2 Alive', 'Class3', 'Class3 Alive'], [c1, alive_c1, c2, alive_c2, c3, alive_c3], color=['green', 'purple']*3)

plt.xlabel(r"Classes $\longrightarrow$")

plt.ylabel(r"Number of Alive passengers $\longrightarrow$")

plt.show()
# Preparing Data for Crunching

from sklearn.model_selection import train_test_split

y = alive_data

X1 = pclass_data

X2 = gender_data

X3 = age_data

X4 = fare_data

X5 = embarked_data

X6 = sibsp_data

X = np.array([X1,X2,X3,X4,X5,X6]).T



x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
X[0]
help(SVC)
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

#clf = DecisionTreeClassifier(max_features=3, max_depth=10).fit(x_train, y_train)

#clf = SVC(kernel='poly', verbose=True, gamma='scale')

clf = XGBClassifier(learning_rate=0.01, booster='dart', max_depth=7, n_estimators=2)

clf.fit(x_train, y_train)

print('Accuracy:', clf.score(x_test, y_test))
clf = XGBClassifier(learning_rate=0.01, booster='dart', max_depth=7, n_estimators=2)

clf.fit(X, y)
help(clf)
data_ = pd.read_csv('../input/test.csv')

data_

len(data_)
data = data_

gender_data = data['Sex'].values

gender_data = np.array([1 if i == 'male' else 0 for i in gender_data])

age_data = data['Age'].values

age_data = [0 if np.isnan(i) else i for i in age_data]

mean_age = sum(age_data)/len(age_data) # We will assign this mean age to all unavialable ages of passengers

age_data = np.array([mean_age if i == 0 else i for i in age_data])

fare_data = data['Fare'].values

sibsp_data = data['SibSp'].values

pclass_data = data['Pclass'].values

embarked_data = data['Embarked'].values



for _x in ['C','Q','S']:

    print(_x+':', len([i for i in embarked_data if i == _x]))

'''

Result:

C: 168

Q: 77

S: 644

'''

'Hence `S` has more number of lower class(income) people!'

'Hence setting all NaN valued embarkment of Passenger to `S`'

'And replacing Character with integer ID'

for i in range(len(embarked_data)):

    if embarked_data[i] == 'Q':

        embarked_data[i] = 0

    elif embarked_data[i] == 'C':

        embarked_data[i] = 1

    else:

        embarked_data[i] = 2

        



X1 = to_categorical(

    pclass_data-1, # -1 is done in order to bring data from [1<->3] to [0<->2]

    num_classes=3) 

X2 = to_categorical(gender_data)

X3 = np.array([age_data]).T /100

X4 = np.array([fare_data]).T /800

X5 = to_categorical(embarked_data)

X6 = to_categorical(sibsp_data, num_classes=10)
[X1[0],X2[0],X3[0],X4[0],X5[0],X6[0]]
res = clf.predict(X)

print(len(res))

res = list(zip(range(892, 1310), res))

res_ = []



for i in res:

    res_.append(','.join(map(str, i)))

    

res = '\n'.join(res_)

print(res)
with open("submission.csv", 'w+') as f:

    f.write('PassengerId,Survived\n'+res)