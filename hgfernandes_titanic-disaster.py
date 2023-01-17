import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

train.head()
test.head()
train_drop = train.drop(columns = ['Survived'])

y = train.Survived

n_train = train.shape[0]
print(train_drop .info())

print(test.info())
# grouping the ages every five years

age_group = train.Age // 5 + 1



# fill the NaN values in the feature

age_group.fillna(0, inplace = True)



# total people

num_people = train.shape[0]



# number of deaths and survivors by age

death_age_group = age_group[train["Survived"] == 0].value_counts()

survivor_age_group = age_group[train["Survived"] == 1].value_counts()



# sorting the index

ind = np.sort(death_age_group.index.values)



# the width of the bars

width = 0.8



# ploating graph

plt.figure(figsize=(12, 10))



death_bar = plt.bar(ind,death_age_group[ind]/num_people,width)

survivor_bar = plt.bar(ind,survivor_age_group[ind]/num_people,width, bottom = death_age_group[ind]/num_people)



plt.title('Percentage of Passengers by Age Group', fontdict={'fontsize':20})

age_groups_ticks = ['None'] + ['{}-{}'.format(int(i * 5), int((i + 1)* 5)) for i in ind[0:]]

plt.xticks(ind, age_groups_ticks)

plt.legend((death_bar[0], survivor_bar[0]), ('Deceased', 'Survived'))



plt.show()

train[train.Age <= 15].Survived.value_counts(normalize = True)
train[train.Age >= 15].Survived.value_counts(normalize = True)
# number of deaths and survivors by class

death_class = train.Pclass[train["Survived"] == 0].value_counts()

survivor_class = train.Pclass[train["Survived"] == 1].value_counts()



#index

ind = [1, 2, 3]



# ploating graph

plt.figure(figsize=(12, 10))



death_class_bar = plt.bar(ind,death_class[ind].values/num_people, width)

survivor_class_bar = plt.bar(ind,survivor_class[ind].values/num_people, width, bottom = death_class[ind].values/num_people)



plt.title('Percentage of Passengers by Class', fontdict={'fontsize':20})

plt.xticks(ind, ('First','Second','Third'))

plt.legend((death_class_bar[0], survivor_class_bar[0]), ('Deceased', 'Survived'))



plt.show()





train.groupby('Pclass').Survived.value_counts(normalize = True, sort = False)
# number of deaths and survivors by class

death_gender = train.Sex[train["Survived"] == 0].value_counts()

survivor_gender = train.Sex[train["Survived"] == 1].value_counts()



#index

ind = ['male','female']



# ploating graph

plt.figure(figsize=(12, 10))



death_gender_bar = plt.bar(ind,death_gender[ind].values/num_people, width)

survivor_gender_bar = plt.bar(ind,survivor_gender[ind].values/num_people, width, bottom = death_gender[ind].values/num_people)



plt.title('Percentage of Passengers by Gender', fontdict={'fontsize':20})

plt.xticks(ind, ('Male','Female'))

plt.legend((death_gender_bar[0], survivor_gender_bar[0]), ('Deceased', 'Survived'))



plt.show()
train.groupby('Sex').Survived.value_counts(normalize = True, sort = False)
all_data = pd.concat((train_drop , test), sort = False).reset_index(drop = True)
all_data["Title"] = all_data.Name.str.extract(r'\b(\w+)\.')
all_data.info()
all_data.Title.value_counts()
all_data.Title[all_data.Age.isna() == True].value_counts()
#creating a dictionary with the titles that have NaN values

missing_age = all_data.Title[all_data.Age.isna() == True].value_counts().index.values



#fill the dictionary with the mean of the ages

age_na_fills = {}

for title in missing_age:

    age_na_fills[title] = round(all_data[(all_data.Title == title).values].Age.mean())



#replacing the NaN values

all_data["Age"] = all_data.apply(lambda row: age_na_fills.get(row.Title) if np.isnan(row['Age']) else row['Age'], axis=1)
all_data.info()
all_data.Embarked.value_counts()
all_data.Embarked = all_data.Embarked.fillna('S')

all_data.Fare = all_data.Fare.fillna(all_data.Fare.mean())
#Creating dummies variables

all_data = pd.get_dummies(all_data, columns = ['Pclass','Sex', 'Embarked','Title'], drop_first = True)



#Dropping unecessary features

all_data = all_data.drop(columns = ['PassengerId','Name','Ticket','Cabin'])
X = all_data[0:n_train]

X_pred = all_data[n_train:]
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
#10-fold cross validation for logistic regression

logreg = LogisticRegression(solver = 'liblinear')

score_logreg = cross_val_score(logreg, X , y, cv = 10, scoring = 'accuracy')

score_logreg.mean()
#10-fold cross validation for SVM

svc = SVC(gamma = 'auto')

score_svc = cross_val_score(svc, X , y, cv = 10, scoring = 'accuracy')

score_svc.mean()
#10-fold cross validation for Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

score_rf = cross_val_score(random_forest, X , y, cv = 10, scoring = 'accuracy')

score_rf.mean()
#Serching for the optimal value of K for KNN

k_range = range(1,31)

k_scores = []



for k in k_range:

    knn = KNeighborsClassifier(n_neighbors = k)

    score_knn = cross_val_score(knn, X , y, cv = 10, scoring = 'accuracy')

    k_scores.append(score_knn.mean())
#plot the value of K for KNN versus the cross validation accuracy

plt.plot(k_range, k_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross Validation Accuracy')

#10-fold cross validation for KNN (K= 4)

knn = KNeighborsClassifier(n_neighbors = 4)

score_knn = cross_val_score(knn, X , y, cv = 10, scoring = 'accuracy')

score_knn.mean()
logreg.fit(X,y)

y_pred = logreg.predict(X_pred)
sub = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':y_pred})

sub.to_csv("titanic.csv", index = False)
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")