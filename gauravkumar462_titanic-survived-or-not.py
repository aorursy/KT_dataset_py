# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pt

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
pd.isnull(train_data).any()
#we can see that the Age and Cabin and Embarked are contaning NULL values so now lets handle them

#First by plotting them and then imputing them



sns.heatmap(train_data.isnull(), yticklabels = False, color='darkred')
#Lets handle the age and visualize with the relation with pclass

pt.figure(figsize = (10,5))

sns.boxplot(x = 'Pclass', y = 'Age', data=train_data)
def in_age(cols):

    age = cols[0]

    pclass = cols[1]

    if pd.isnull(age):

        if pclass == 1:

            return 38

        elif pclass ==2:

            return 29

        else:

            return 24

    else:

        return age
#Applying the above function

train_data['Age'] = train_data[['Age', 'Pclass']].apply(in_age, axis=1)
#checking heatmap of nullity

sns.heatmap(train_data.isnull(),  yticklabels = False, cbar=False)



#So below we can see that er have removed the nullity from age column
#Lets check survival based on Age

sns.distplot(train_data['Age'])
#Lets check the class wise survival of passenger

sns.countplot(train_data['Survived'],hue=train_data['Pclass'])
women = train_data.loc[train_data['Sex']=='female']['Survived']

wom_sur = sum(women)/len(women)*100



print("% of women survived : ", wom_sur)
men = train_data.loc[train_data['Sex']=='male']['Survived']

men_sur = sum(men)/len(men)*100



print("% of women survived : ", men_sur)
#Lets visualize the data

sns.countplot(train_data['Survived'], hue=train_data['Sex'])
train_data.describe()
#Now we will drop PassengerId, Name, Ticket, Fare, Cabin

train_data.drop(['PassengerId', 'Name', 'Ticket','Fare', 'Cabin'], inplace=True, axis=1)
train_data.head()
em = pd.get_dummies(train_data['Embarked'], drop_first = True)

train_data.drop("Embarked", axis=1, inplace=True)
train_data = train_data.join(em)

train_data.head()
#Separating dependent and independent variables

x = train_data.iloc[:,1:8].values

y = train_data.iloc[:,0].values
#Handling the categorical features

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label = LabelEncoder()

x[:,1] = label.fit_transform(x[:,1])

x[:,0] = label.fit_transform(x[:,0])
onehot = OneHotEncoder(categorical_features = [0])

x = onehot.fit_transform(x).toarray()

x
#Avoiding dummy variable trap

x =x[:,1:]
#Splitting the training set and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .33, random_state = 0)
def conf_mat(classifier, y_real, y_pred):

    from sklearn.metrics import confusion_matrix, accuracy_score

    cm = confusion_matrix(y_real, y_pred)

    acc = accuracy_score(y_real, y_pred)*100

    return cm, acc
#Building our models



# ---- LOGISTIC REGRESSION ----



from sklearn.linear_model import LogisticRegression

log_classifier = LogisticRegression(random_state = 0)

log_classifier.fit(x_train, y_train)



y_pred = log_classifier.predict(x_test)



log_cm, log_acc = conf_mat(log_classifier, y_test, y_pred)



print("Confusion Matrix is  : ",log_cm)

print("Accuracy Score is : ", log_acc)
# ---- K NEAREST NEIGHBORS ----



from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p=2)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

knn_cm, accuracy_knn = conf_mat(classifier, y_test, y_pred)



print("Confusion Matrix is  : ",knn_cm)

print("Accuracy Score is : ", accuracy_knn)
# ---- SUPPORT VECTOR MACHINE ---



from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)



lin_svm_cm, accuracy_svm = conf_mat(classifier, y_test, y_pred)



print("Confusion Matrix is  : ",lin_svm_cm)

print("Accuracy Score is : ", accuracy_svm)
# ---- GAUSSIAN SVM ----

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0, gamma='auto')

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)



svm_cm, accuracy_gsvm = conf_mat(classifier, y_test, y_pred)



print("Confusion Matrix is  : ",svm_cm)

print("Accuracy Score is : ", accuracy_gsvm)
# ---- NAIVE BAYES ALGORITHM ----



from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(x_train, y_train)



y_pred = classifier.predict(x_test)



bayes_cm, accuracy_bayes = conf_mat(classifier, y_test,  y_pred)



print("Confusion Matrix is  : ",bayes_cm)

print("Accuracy Score is : ", accuracy_bayes)
# ---- DECISION TREE ----



from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(x_train, y_train)



y_pred = classifier.predict(x_test)



decision_cm, accuracy_decision = conf_mat(classifier, y_test, y_pred)



print("Confusion Matrix is  : " ,decision_cm)

print("Accuracy Score is : ", accuracy_decision)
# ---- RANDOM FOREST ----



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state=0)

classifier.fit(x_train, y_train)



y_pred = classifier.predict(x_test)



random_cm, accuracy_random = conf_mat(classifier, y_test, y_pred)



print("Confusion Matrix is  : " ,random_cm)

print("Accuracy Score is : ", accuracy_random)

Score_Sheet = pd.DataFrame(

    {"Accuracy Score":[log_acc,accuracy_knn,accuracy_svm,accuracy_gsvm,accuracy_bayes

                       ,accuracy_decision, accuracy_random]},

    index = ["Logisic_Regression","K_Nearest","Support_Vector","Gaussian_SVM","Naive_Bayes","Decision_Tree", 

                           "Random_Forest"]

        )
Score_Sheet
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
temp = test_data.copy()

test_data.drop(["PassengerId", 'Name', 'Ticket', 'Fare', 'Cabin'],axis = 1, inplace=True)
pd.isnull(test_data).any()
sns.heatmap(test_data.isnull(), yticklabels = False, cbar = False)
test_data['Age'] = test_data[['Age', 'Pclass']].apply(in_age, axis = 1)

sns.heatmap(test_data.isnull(), yticklabels = False, cbar = False)
pclass = pd.get_dummies(test_data['Pclass'], drop_first = True)

sex = pd.get_dummies(test_data['Sex'], drop_first = True)

emb = pd.get_dummies(test_data['Embarked'], drop_first = True)

test_data.drop(['Pclass', 'Sex', 'Embarked'], axis = 1, inplace=True)
test_data = test_data.join([pclass, sex, emb])
test_data = test_data[[2,3,'male','Age', 'SibSp', 'Parch', 'Q', 'S']]

test_data.head()
x = test_data.iloc[:,:].values

x
y_pred = log_classifier.predict(x)
x1 = pd.Series(temp['PassengerId'])

x2 = pd.Series(y_pred, name='Survived')

submission = pd.DataFrame(x1)

submission = submission.join(x2)
submission.to_csv("Result.csv")
submission