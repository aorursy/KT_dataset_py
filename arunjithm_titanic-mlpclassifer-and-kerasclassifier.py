import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import random



from keras.models import Sequential

from keras.layers.core import Dense

from keras.optimizers import adam

from keras.wrappers.scikit_learn import KerasClassifier



from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler



from sklearn.neural_network import MLPClassifier



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score,confusion_matrix, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report



train_titanic = pd.read_csv("../input/titanic/train.csv")

real_test_titanic = pd.read_csv("../input/titanic/test.csv")

print("Dimension of training set : ", train_titanic.shape)

print("Dimension of test set : ", real_test_titanic.shape)

train_titanic.head()
real_test_titanic.head()
#Check if there are any null values in training set



train_titanic.isnull().sum()
age_mean = int((round(train_titanic['Age'].mean(),2)))

age_sd = int(round(train_titanic['Age'].std(),2))

print("The mean of age : ", age_mean)

print("The standard deviation of age : ", age_sd)
from random import choice

randomvalue = [i for i in range(age_sd,age_mean)]

for _ in range(train_titanic.isnull().sum().Age):

	number_to_insert = choice(randomvalue)

	train_titanic['Age'].fillna(number_to_insert, inplace = True) 
train_titanic.isnull().sum()
print("The most frequent value in 'Embarked' column is :", train_titanic['Embarked'].value_counts().idxmax())
train_titanic['Embarked'] = train_titanic['Embarked'].fillna(train_titanic['Embarked'].value_counts().idxmax())

train_titanic.isnull().sum()
real_test_titanic.isnull().sum()
age_mean_test = int((round(real_test_titanic['Age'].mean(),2)))

age_sd_test = int(round(real_test_titanic['Age'].std(),2))

randomvalue = [i for i in range(age_sd_test,age_mean_test)]

for _ in range(real_test_titanic.isnull().sum().Age):

    number_to_insert = choice(randomvalue)

    real_test_titanic['Age'].fillna(number_to_insert, inplace = True)
real_test_titanic.isnull().sum()
real_test_titanic['Fare'] = real_test_titanic['Fare'].fillna(int(real_test_titanic['Fare'].mean()))
real_test_titanic.isnull().sum()
#Total passengers and survival percentages based on gender

total_passengers = train_titanic['Sex'].count()

total_males = train_titanic['Sex'].value_counts()['male']

total_females = train_titanic['Sex'].value_counts()['female']

survived_males = train_titanic.query('Survived==1')['Sex'].value_counts()['male']

survived_females = train_titanic.query('Survived==1')['Sex'].value_counts()['female']

survived_total = survived_males + survived_females

print("=" * 70)

print("TRAINING SET ANALYSIS BASED ON GENDER")

print("=" * 70)

print("Total number of travellers : ",total_passengers )

print("Percentage of males : ", (100 * total_males /total_passengers))

print("Percentage of females : ", (100 * total_females /total_passengers))



print("=" * 70)

print("Total number of people survived : ", survived_total) 

print("Total number of survived males : ", survived_males )

print("Total number of survived females : ", survived_females)



print("=" * 70)

print("Percentage of males survived : ", 100 * (survived_males/total_passengers) )

print("Percentage of females survived : ", 100 * (survived_females/total_passengers))



print("=" * 70)

print("Percentage of total people survived : ", 100 * (survived_total/total_passengers)) 





#Total passengers and survival percentages based on passenger class

print("=" * 70)

print("*" * 70)

print("=" * 70)

total_1stclass_passengers = train_titanic['Pclass'].value_counts()[1]

total_2ndclass_passengers = train_titanic['Pclass'].value_counts()[2]

total_3rdclass_passengers = train_titanic['Pclass'].value_counts()[3]

survived_1stclass_passengers = train_titanic.query('Survived==1')['Pclass'].value_counts()[1]

survived_2ndclass_passengers = train_titanic.query('Survived==1')['Pclass'].value_counts()[2]

survived_3rdclass_passengers = train_titanic.query('Survived==1')['Pclass'].value_counts()[3]

print("TRAINING SET ANALYSIS BASED ON PASSENGER CLASS")

print("=" * 70)

print("Total number of first class passengers : ", total_1stclass_passengers )

print("Total number of second class passengers : ", total_2ndclass_passengers )

print("Total number of third class passengers : ", total_3rdclass_passengers )

print("=" * 70)

print("Total number of survived first class passengers : ", survived_1stclass_passengers )

print("Total number of survived second class passengers : ", survived_2ndclass_passengers )

print("Total number of survived third class passengers : ", survived_3rdclass_passengers )

print("Percentage of survived first class passengers : ", 100 * survived_1stclass_passengers/ total_passengers)

print("Percentage of survived second class passengers : ", 100 * survived_2ndclass_passengers/ total_passengers)

print("Percentage of survived first class passengers : ", 100 * survived_3rdclass_passengers/ total_passengers)

print("=" * 70)

#Plotting to see distribution



sns.countplot('Sex', data=train_titanic, palette=None)

plt.title("Total male and female passengers onboard Titanic")

plt.ylabel("Number of passengers")

plt.xlabel("Gender")

sns.countplot('Sex',data=train_titanic,hue='Survived')

plt.title("Genderwise survival diagram")

plt.ylabel("Number of passengers")

plt.xlabel("Gender")
sns.countplot('Pclass',data=train_titanic)

plt.title("Number of passengers in each class")

plt.ylabel("Number of passengers")

plt.xlabel("Gender")
sns.countplot('Pclass',data=train_titanic,hue='Survived')

plt.title("Passenger Class and Survival rate diagram")

plt.ylabel("Number of passengers")

plt.xlabel("Gender")
train_titanic = train_titanic.drop(['Ticket', 'Cabin','Name'], axis=1)

real_test_titanic = real_test_titanic.drop(['Ticket', 'Cabin','Name'], axis=1)
train_titanic.head()
real_test_titanic.head()
train_survived = train_titanic['Survived']
train_titanic['Sex'] = train_titanic['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

train_titanic['Embarked'] = train_titanic['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train_titanic.head()
real_test_titanic['Sex'] = real_test_titanic['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

real_test_titanic['Embarked'] = real_test_titanic['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
real_test_titanic.head()
train_titanic = train_titanic.set_index('PassengerId')

real_test_titanic = real_test_titanic.set_index('PassengerId')
real_test_titanic.head()
X = train_titanic.drop(['Survived'], axis = 1)

y = train_titanic["Survived"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

print("Dimension of Train data :", x_train.shape)

print("Dimension of Test data :", x_test.shape)
scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.transform(x_test)
mlp_model = MLPClassifier(hidden_layer_sizes=(150,150,150),activation ='relu', max_iter=500, alpha=0.0001,

                     solver='sgd', verbose=10, learning_rate = 'adaptive', momentum=0.9)



mlp_model.fit(x_train_scaled, y_train)

y_pred = mlp_model.predict(x_test_scaled)

confusion = confusion_matrix(y_test, y_pred)

print('Confusion Matrix \n', confusion)

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
classifier = MLPClassifier()

parameter_space = {

    'hidden_layer_sizes': [(50,50,50), (100,100,100), (150,150,150), (200,200,200)],

    'activation': ['tanh', 'relu', 'logistic'],

    'solver': ['sgd', 'adam'],

    'alpha': [0.0001, 0.05],

    'learning_rate': ['constant','adaptive'],

}
model = GridSearchCV(classifier, parameter_space, n_jobs=-1, cv=3)

model.fit(x_train_scaled, y_train)
print('Best parameters calculated :', model.best_params_)
predicted_y = model.predict(x_test_scaled)

confusion = confusion_matrix(y_test, predicted_y)

print('Confusion Matrix after hyperparameter optimization \n', confusion)

print('Accuracy after hyperparameter optimization: {:.2f}'.format(accuracy_score(y_test, predicted_y)))
layer1 = Dense(units = 10,activation = 'relu', input_dim = 7)

layer2 = Dense(units = 15, activation = 'relu')

layer3 = Dense(units = 2, activation = 'sigmoid')



model = Sequential([layer1, layer2, layer3])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])



clf = model.fit(x_train_scaled, y_train, batch_size = 25, epochs = 300)
predicted_on_test = model.predict_classes(x_test_scaled, batch_size = 25)

confusion = confusion_matrix(y_test, predicted_on_test)

print('Confusion Matrix \n', confusion)

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, predicted_on_test)))
real_test_titanic_scaled = scaler.transform(real_test_titanic)
predicted_on_actual = model.predict_classes(real_test_titanic_scaled, batch_size = 25)
predicted_on_actual
actual_test = pd.read_csv("../input/titanic/test.csv")



for_submission = pd.DataFrame({"PassengerId": actual_test['PassengerId'],

                      "Survived":predicted_on_actual.astype(int)})

for_submission.to_csv("prediction_file_by_arunjith.csv",index=False)