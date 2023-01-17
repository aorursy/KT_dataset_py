import pandas as pd

import sklearn

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



adult = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv'

,

        sep=',',

        engine='python',

        na_values="?")



adult.head()
adult.describe(include = 'all')
adult["age"].value_counts().plot(kind="bar")
adult["hours.per.week"].value_counts().plot(kind="bar")
adult["sex"].value_counts().plot(kind="bar")
adult["education.num"].value_counts().plot(kind="bar")
#We will change NaN values in the columns with these values, either wih the mean or the top (as seen in the .describe() method)

values = {'age': 38.581647, 'education.num': 10.080679, 'sex': 'Male', 'capital.gain': 1077.648844, 'capital.loss': 87.303830, 'hours.per.week': 40.437456} 

adult.fillna(values)



#And we will create a new column that shows if the person works more than 45 hours per week

more_hours = []

for i in range(len(adult)): 

  if adult["hours.per.week"][i] > 45:

    more_hours.append(1)

  else:

    more_hours.append(0)

adult[">45h"] = more_hours



#In order to use Male or Female (Sex) data, we will create a new column

Male_or_Female = []

for i in range(len(adult)): 

  if adult["sex"][i] == "Male":

    Male_or_Female.append(1)

  else:

    Male_or_Female.append(0)

    



adult["Male_or_Female"] = Male_or_Female
#Now, we will create our atributes that will be used to train the model

atributes = ["age", "Male_or_Female", "capital.gain", "capital.loss", "education.num", ">45h"]
adultTest = pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv'

,

        sep=',',

        engine='python',

        na_values="?")

values = {'age': 38.581647, 'education.num': 10.080679, 'sex': 'male', 'capital.gain': 1077.648844, 'capital.loss': 87.303830, 'hours.per.week': 40.437456} 

adultTest.fillna(values)
more_hours_Test = []

for i in range(len(adultTest)): 

  if adult["hours.per.week"][i] > 45:

    more_hours_Test.append(1)

  else:

    more_hours_Test.append(0)



adultTest[">45h"] = more_hours_Test



Male_or_Female_Test = []

for i in range(len(adultTest)): 

  if adultTest["sex"][i] == "Male":

    Male_or_Female_Test.append(1)

  else:

    Male_or_Female_Test.append(0)

    



adultTest["Male_or_Female"] = Male_or_Female_Test

#KNN, with K=3

Xadult = adult[atributes]

Yadult = adult.income
XtestAdult = adultTest[atributes]
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(knn, Xadult, Yadult, cv=10)
print("scores = {}". format(scores))

print("accuracy = {}".format(scores.mean()))
#now, let's find the best k for this case, from 10 to 30

accuracy_list = []

for i in range(10,30):

    knn = KNeighborsClassifier(n_neighbors=i)

    for j in range(5, 16, 5):

        scores = cross_val_score(knn, Xadult, Yadult, cv=j)

        accuracy_list.append(scores.mean())

best_k = accuracy_list.index(max(accuracy_list))//3

best_folds = accuracy_list.index(max(accuracy_list))%3*5

print("The best case is with k = {} and folds = {}".format(best_k, best_folds))
#SO, WE GET THAT THE HYPERPARAMETER K IS THE BEST WHEN K = 17 AND THE NUMER OF FOLDS IS BEST WHEN FOLDS = 10

knn = KNeighborsClassifier(n_neighbors=17)

knn.fit(Xadult,Yadult)

YadultTestPred = knn.predict(XtestAdult)

id_index = pd.DataFrame({'Id' : list(range(len(YadultTestPred)))})

income = pd.DataFrame({'income' : YadultTestPred})

result = income

result
result.to_csv("submission.csv", index = True, index_label = 'Id')