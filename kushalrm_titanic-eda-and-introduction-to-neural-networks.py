# importing required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

plt.rcParams['figure.figsize'] = (9.0,9.0)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

test_dataset =  pd.read_csv('/kaggle/input/titanic/test.csv')
# Shape of the datasets

print(train_dataset.shape, test_dataset.shape)
test_dataset.head()
train_dataset.head()
train_dataset.SibSp.unique()
train_dataset.describe()
train_dataset.isnull().sum()
sb.heatmap(train_dataset.isnull())
# writing a function for imputing the data

def imput(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train_dataset['Age'] = train_dataset[['Age','Pclass']].apply(imput, axis=1)
train_dataset['Embarked'].unique()
# a function for missing value imputation

def impu(Embarked):

    if pd.isnull(Embarked):

        return 'C'

    else:

        return Embarked
train_dataset['Embarked'] = train_dataset['Embarked'].apply(impu)
sb.heatmap(train_dataset.isnull())
train_dataset.info()
test_dataset.isnull().sum()
test_dataset.info()
test_dataset['Age'] = test_dataset[['Age','Pclass']].apply(imput, axis=1)
sb.heatmap(test_dataset.isnull())
train_dataset.groupby(train_dataset.Age//10*10).size().plot.bar(cmap='Set3', width=0.9)

plt.title('Age Group Size', fontsize = 20)

plt.show()
sb.countplot(x='Survived',data = train_dataset, palette = 'Dark2')

plt.title('Survival Count', fontsize = 20)

plt.show()
sb.countplot(x='Survived',hue = train_dataset['Sex'], data = train_dataset, palette = 'Reds')

plt.title('Survived vs sex', fontsize = 20)

plt.show()
sb.countplot(x='Survived', hue='Pclass',data=train_dataset)

plt.title('Survived vs Pclass', fontsize = 20)

plt.show()
sb.boxplot(x='Survived', y='Age',data=train_dataset, palette = 'winter')

plt.title('Survived vs Age', fontsize = 20)

plt.show()

train_dataset = train_dataset.set_index('PassengerId')

train_dataset = train_dataset.drop(columns=['Name','Fare', 'Ticket', 'Cabin'])

train_dataset = pd.get_dummies(train_dataset, columns=[ 'Pclass','Sex','Embarked'], drop_first = True)
train_dataset.head()
test_dataset = test_dataset.set_index('PassengerId')

test_dataset = test_dataset.drop(columns=['Name','Fare', 'Ticket', 'Cabin'])

test_dataset = pd.get_dummies(test_dataset, columns=['Pclass','Sex', 'Embarked'], drop_first = True)

test_dataset.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_dataset.drop(['Survived'],axis=1),

                                                    train_dataset['Survived'], test_size=0.1)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)

test_1 = sc.transform(test_dataset)
# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LeakyReLU,PReLU,ELU
#initialize model

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 15, init = 'he_uniform', activation = 'relu', input_dim = 8))



#Adding a second hidden layer 

classifier.add(Dense(output_dim = 20, init = 'he_uniform', activation = 'relu'))



#output layer

classifier.add(Dense(output_dim = 1 , init = 'glorot_uniform', activation = 'sigmoid'))



#compiling

classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



#fitting the data in ANN 

model_history=classifier.fit(x_train, y_train,validation_split=0.33, batch_size = 50, nb_epoch = 400)

y_pred = classifier.predict(x_test)

y_pred[0:10]
# predicting the test data for submission

test_predictions = classifier.predict(test_1)

test_predictions[:10]
# the output is continuous and need to converted.

test_predictions = (test_predictions > 0.5)
# function to covert the output 

def convert(insert):

     for x in insert:

            if x == False:

                return '0'

            if x == True:

                return '1'    
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df['PassengerId'] = test_dataset.index

submission_df['Survived'] = test_predictions
submission_df['Survived'] = submission_df[['Survived']].apply(convert, axis=1)
submission_df.head(10)
submission_df.to_csv('submit_it_1.csv', header = True, index = False)