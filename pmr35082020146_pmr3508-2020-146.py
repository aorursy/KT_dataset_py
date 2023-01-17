import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import numpy as np
adtrain = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv',

        names = ["ID", "Age", "Workclass", "fnlwgt","Education", "Education-Num", "Martial Status","Occupation",

        "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country", "Income"],

        sep = r'\s*,\s*',

        engine = 'python',

        na_values = "?")
adtrain.shape
adtrain.head()
plt.figure(figsize=(13, 7))

adtrain['Age'].hist(color = 'blue')

plt.xlabel('Age')

plt.ylabel('Quantity')

plt.title('Age histogram')
plt.figure(figsize=(13, 7))

adtrain['Workclass'].value_counts().plot(kind = 'pie')
plt.figure(figsize=(13, 7))

adtrain['Education-Num'].value_counts().plot(kind = 'bar')

plt.ylabel('Quantity')

plt.title('Education bar plot')
plt.figure(figsize=(13, 7))

adtrain['Martial Status'].value_counts().plot(kind = 'pie')
plt.figure(figsize=(13, 7))

adtrain['Occupation'].value_counts().plot(kind = 'bar')

plt.ylabel('Quantity')

plt.title('Occupation bar plot')
plt.figure(figsize=(13, 7))

adtrain['Hours per week'].hist(color = 'blue')

plt.xlabel('Hours per week')

plt.ylabel('Quantity')

plt.title('Hours per week histogram')
plt.figure(figsize=(13, 7))

adtrain['Relationship'].value_counts().plot(kind = 'bar')

plt.ylabel('Quantity')

plt.title('Relationship bar plot')
plt.figure(figsize=(13, 7))

adtrain['Race'].value_counts().plot(kind = 'pie')
plt.figure(figsize=(13, 7))

adtrain['Sex'].value_counts().plot(kind = 'pie')
plt.figure(figsize=(13, 7))

adtrain['Capital Gain'].hist(color = 'blue')

plt.xlabel('Capital gain')

plt.ylabel('Quantity')

plt.title('Capital gain histogram')
plt.figure(figsize=(13, 7))

adtrain['Capital Loss'].hist(color = 'blue')

plt.xlabel('Capital loss')

plt.ylabel('Quantity')

plt.title('Capital loss histogram')
adtrain['Country'].value_counts()
marcivincome = [0,0]

nevermarincome = [0,0]

divorcedincome = [0,0]

separatedincome = [0,0]

widowedincome = [0,0]

marspouincome = [0,0]

marafincome = [0,0]

for i in range (32560):

    martialstatus = adtrain.iloc[i]['Martial Status']

    income = adtrain.iloc[i]['Income']

    if martialstatus == 'Married-civ-spouse':

        if income == '>50K':

            marcivincome[0] +=1

        else:

            marcivincome[1] +=1

    elif martialstatus == 'Never-married':

        if income == '>50K':

            nevermarincome[0] +=1

        else:

            nevermarincome[1] +=1

    elif martialstatus == 'Divorced':

        if income == '>50K':

            divorcedincome[0] +=1

        else:

            divorcedincome[1] +=1

    elif martialstatus == 'Separated':

        if income == '>50K':

            separatedincome[0] +=1

        else:

            separatedincome[1] +=1

    elif martialstatus == 'Widowed':

        if income == '>50K':

            widowedincome[0] +=1

        else:

            widowedincome[1] +=1

    elif martialstatus == 'Married-spouse-absent':

        if income == '>50K':

            marspouincome[0] +=1

        else:

            marspouincome[1] +=1

    elif martialstatus == 'Married-AF-spouse':

        if income == '>50K':

            marafincome[0] +=1

        else:

            marafincome[1] +=1

            

plt.figure(figsize=(13, 7))

x_indexes = np.arange(7)

width = 0.4

plt.xticks(ticks = x_indexes, labels = ['Married-civ-spouse','Never-married','Divorced','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'])

plt.bar(x_indexes, [marcivincome[0], nevermarincome[0], divorcedincome[0], separatedincome[0], widowedincome[0], marspouincome[0],marafincome[0]], width = width, color = 'blue')

plt.bar(x_indexes + width, [marcivincome[1], nevermarincome[1], divorcedincome[1], separatedincome[1], widowedincome[1], marspouincome[1],marafincome[1]], width = width, color = 'orange')
husbandincome = [0,0]

notincome = [0,0]

ownincome = [0,0]

unmarriedincome = [0,0]

wifeincome = [0,0]

otherincome = [0,0]

for i in range (32560):

    relationship = adtrain.iloc[i]['Relationship']

    income = adtrain.iloc[i]['Income']

    if relationship == 'Husband':

        if income == '>50K':

            husbandincome[0] +=1

        else:

            husbandincome[1] +=1

    elif relationship == 'Not-in-family':

        if income == '>50K':

            notincome[0] +=1

        else:

            notincome[1] +=1

    elif relationship == 'Own-child':

        if income == '>50K':

            ownincome[0] +=1

        else:

            ownincome[1] +=1

    elif relationship == 'Unmarried':

        if income == '>50K':

            unmarriedincome[0] +=1

        else:

            unmarriedincome[1] +=1

    elif relationship == 'Wife':

        if income == '>50K':

            wifeincome[0] +=1

        else:

            wifeincome[1] +=1

    elif relationship == 'Other':

        if income == '>50K':

            otherincome[0] +=1

        else:

            otherincome[1] +=1

            

plt.figure(figsize=(13, 7))

x_indexes = np.arange(6)

width = 0.4

plt.xticks(ticks = x_indexes, labels = ['Husband','Not-in-family','Own-child','Unmarried','Wife','Other'])

plt.bar(x_indexes, [husbandincome[0], notincome[0], ownincome[0], unmarriedincome[0], wifeincome[0], otherincome[0]], width = width, color = 'blue')

plt.bar(x_indexes + width, [husbandincome[1], notincome[1], ownincome[1], unmarriedincome[1], wifeincome[1], otherincome[1]], width = width, color = 'orange')
whiteincome = [0,0]

blackincome = [0,0]

asianpacincome = [0,0]

amerindincome = [0,0]

otherincome = [0,0]

for i in range (32560):

    race = adtrain.iloc[i]['Race']

    income = adtrain.iloc[i]['Income']

    if race == 'White':

        if income == '>50K':

            whiteincome[0] +=1

        else:

            whiteincome[1] +=1

    elif race == 'Black':

        if income == '>50K':

            blackincome[0] +=1

        else:

            blackincome[1] +=1

    elif race == 'Asian-Pac-Islander':

        if income == '>50K':

            asianpacincome[0] +=1

        else:

            asianpacincome[1] +=1

    elif race == 'Amer-Indian-Eskimo':

        if income == '>50K':

            amerindincome[0] +=1

        else:

            amerindincome[1] +=1

    elif race == 'Other':

        if income == '>50K':

            otherincome[0] +=1

        else:

            otherincome[1] +=1

            

plt.figure(figsize=(13, 7))

x_indexes = np.arange(5)

width = 0.4

plt.xticks(ticks = x_indexes, labels = ['White','Black','Asian-Pac-Islander','Amer-Indian-Eskimo','Other'])

plt.bar(x_indexes, [whiteincome[0], blackincome[0], asianpacincome[0], amerindincome[0], otherincome[0]], width = width, color = 'blue')

plt.bar(x_indexes + width, [whiteincome[1], blackincome[1], asianpacincome[1], amerindincome[1], otherincome[1]], width = width, color = 'orange')
maleincome = [0,0]

femaleincome = [0,0]

for i in range (32560):

    if adtrain.iloc[i]['Sex'] == 'Male':

        if adtrain.iloc[i]['Income'] == '>50K':

            maleincome[0] +=1

        else:

            maleincome[1] +=1

    else:

        if adtrain.iloc[i]['Income'] == '>50K':

            femaleincome[0] +=1

        else:

            femaleincome[1] +=1

            

plt.figure(figsize=(13, 7))

x_indexes = np.arange(len(maleincome))

width = 0.4

plt.xticks(ticks = x_indexes, labels = ["Male", "Female"])

plt.bar(x_indexes, [maleincome[0], femaleincome[0]], width = width, color = 'blue')

plt.bar(x_indexes + width, [maleincome[1], femaleincome[1]], width = width, color = 'orange')
nadtrain = adtrain.dropna()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

numadtrain = nadtrain.apply(le.fit_transform)

Xtrain = numadtrain[['Age', 'Education-Num', 'Martial Status', 'Occupation', 'Relationship', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week']]

Ytrain = numadtrain.Income
scoreintervalos = []

folds = 5

for k in [1,10,20,30,40]:

    knn = KNeighborsClassifier(n_neighbors = k)

    scores = cross_val_score(knn, Xtrain, Ytrain, cv = folds)

    scoreintervalos.append([k, scores.mean()])
scoreintervalos
scoreintervalos = []

folds = 8

for k in range(20,31):

    knn = KNeighborsClassifier(n_neighbors = k)

    scores = cross_val_score(knn, Xtrain, Ytrain, cv = folds)

    scoreintervalos.append([k, scores.mean()])
scoreintervalos
scoreintervalos = []

folds = 12

for k in [22, 26, 27]:

    knn = KNeighborsClassifier(n_neighbors = k)

    scores = cross_val_score(knn, Xtrain, Ytrain, cv = folds)

    scoreintervalos.append([k, scores.mean()])
scoreintervalos
knnfinal = KNeighborsClassifier(n_neighbors = 26)

knnfinal.fit(Xtrain, Ytrain)
adtest = pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv',

        names = ["ID", "Age", "Workclass", "fnlwgt","Education", "Education-Num", "Martial Status","Occupation",

        "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country"],

        sep = r'\s*,\s*',

        engine = 'python',

        na_values = "?")
adtest.head()
nadtest = adtest.dropna()

nadtest.head()
numadtest = nadtest.apply(le.fit_transform)

Xtest = numadtest[['Age', 'Education-Num', 'Martial Status', 'Occupation', 'Relationship', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week']]

Ytestpred = knnfinal.predict(Xtest)

temp_ = []

for i in Ytestpred:

    if i == 0:

        temp_.append(['>50k'])

    else:

        temp_.append(['<=50k'])

predictions = pd.DataFrame(temp_)

predictions.columns = ['income']

predictions
predictions.to_csv("output.csv", index = True, index_label = 'Id')