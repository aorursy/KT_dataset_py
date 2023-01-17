# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sub_df = pd.read_csv('../input/genderclassmodel.csv')
train_df.head(2)
train_df.info()

print('---------------------------')

test_df.info()
def has_cabin(data):

    data['Has_Cabin'] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



def sex_map(data):

    data['Sex'] = data['Sex'].map({'female':0, 'male':1}).astype(int)



def familysize(data):

    data['Family_Size'] = data['SibSp'] + data['Parch'] + 1



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



def title_name(data):

    data['Title'] = data['Name'].apply(get_title)

    #-------------------training dataset-------------------------

    #Mr          517

    #Miss        182

    #Mrs         125

    #Master       40

    #Dr_7, Rev_6, Col_2, Major_2, Mlle_2, Lady_1

    #Sir_1, Countess_1, Don_1, Capt_1, Mme_1, Ms_1, Jonkheer_1

    data['Title'] = data['Title'].replace(['Countess','Capt', 'Col','Don', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    data['Title'] = data['Title'].replace(['Mlle', 'Lady', 'Ms'], 'Miss')

    data['Title'] = data['Title'].replace('Dr', 'Master')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    data['Title'] = data['Title'].replace('Sir', 'Mr')

    #Mapping title

    data['Title'] = data['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})

    #Remove all NULLS in the Title column

    data['Title'] = data['Title'].fillna(0)



def embarked_map(data):

    # Mapping Embarked

    data['Embarked'] = data['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} )

    #Remove all NULLS in the Embarked column

    data['Embarked'] = data['Embarked'].fillna(0).astype(int)

        
plt.figure(figsize=(12, 8))

sns.kdeplot(train_df.Fare, shade=True)

plt.title('Fare distribution', fontsize = 15)

plt.xlabel('Fare', fontsize = 12)

plt.ylabel('Percent', fontsize = 12)

plt.show()
train_fare = train_df.groupby('Fare')['Fare'].count()

for i in range(0, len(train_fare)):

    k = train_fare[0:train_fare.index[i]].sum()

    if ((k/train_fare.sum()* 100) >= 15) & ((k/train_fare.sum()* 100) <= 17):

        print('Fare that appear less than {} times: {}%'.format(train_fare.index[i], k/train_fare.sum()* 100))

    elif ((k/train_fare.sum()* 100) >= 49) & ((k/train_fare.sum()* 100) <= 51):

        print('Fare that appear less than {} times: {}%'.format(train_fare.index[i], k/train_fare.sum()* 100))

    elif ((k/train_fare.sum()* 100) >= 84) & ((k/train_fare.sum()* 100) <= 86):

        print('Fare that appear less than {} times: {}%'.format(train_fare.index[i], k/train_fare.sum()* 100))
def fare_map(data):

    # Mapping Fare

    data.loc[ data['Fare'] <= 7.8, 'Fare'] = 1

    data.loc[(data['Fare'] > 7.8) & (data['Fare'] <= 14.455 ), 'Fare'] = 2

    data.loc[(data['Fare'] > 14.455 ) & (data['Fare'] <= 54), 'Fare']   = 3

    data.loc[ data['Fare'] > 54, 'Fare'] = 4

    #Remove all NULLS in the Fare column

    data['Fare'] = data['Fare'].fillna(0)

    data['Fare'] = data['Fare'].astype(int)

    
plt.figure(figsize=(12, 8))

sns.kdeplot(train_df.Age, shade=True)

plt.title('Age distribution', fontsize = 15)

plt.xlabel('Age', fontsize = 12)

plt.ylabel('Percent', fontsize = 12)

plt.show()
train_age = train_df.groupby('Age')['Age'].count()

for i in range(0, len(train_age)):

    k = train_age[0:train_age.index[i]].sum()

    if ((k/train_age.sum()* 100) >= 15) & ((k/train_age.sum()* 100) <= 17):

        print('Age that appear less than {} times: {}%'.format(train_age.index[i], k/train_age.sum()* 100))

    elif ((k/train_age.sum()* 100) >= 49) & ((k/train_age.sum()* 100) <= 51):

        print('Age that appear less than {} times: {}%'.format(train_age.index[i], k/train_age.sum()* 100))

    elif ((k/train_age.sum()* 100) >= 84) & ((k/train_age.sum()* 100) <= 86):

        print('Age that appear less than {} times: {}%'.format(train_age.index[i], k/train_age.sum()* 100))
def age_map(data):

    # Mapping age

    data.loc[ data['Age'] <= 17, 'Age'] = 1

    data.loc[(data['Age'] > 17) & (data['Age'] <= 28 ), 'Age'] = 2

    data.loc[(data['Age'] > 28 ) & (data['Age'] <= 45), 'Age']   = 3

    data.loc[ data['Age'] > 45, 'Age'] = 4

    #Remove all NULLS in the Age column

    data['Age'] = data['Age'].fillna(0)

    data['Age'] = data['Age'].astype(int)
def data_map(data):

    has_cabin(data)

    sex_map(data)

    familysize(data)

    title_name(data)

    embarked_map(data)

    fare_map(data)

    age_map(data)

    return data

#---------------------------------

train_data = data_map(train_df)

test_data = data_map(test_df)

train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
train_data.head(3)
colormap = plt.cm.viridis

plt.figure(figsize=(10,10))

plt.title(' The Absolute Correlation Coefficient of Features', y=1.05, size=15)

sns.heatmap(abs(train_data.astype(float).corr()),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, )

plt.show()
plt.figure(figsize=(15,15))

sns.pairplot(train_data[[u'Survived', u'Pclass', u'Sex', u'Parch', u'Fare', u'Embarked', u'Has_Cabin', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10)).set(xticklabels=[])

plt.show()
from tpot import TPOTClassifier
import tensorflow as tf

import tflearn as tfl



def survived_logic(data):

    x = []

    for i in range(len(data.Survived)):

        if data.Survived[i] == 0:

            x.append(np.array([0, 1]))

        elif data.Survived[i] == 1:

            x.append(np.array([1, 0]))

    return x

#----------------------------dataset-----------------------------------

train_x = np.array(train_data.drop(['Survived'], axis = 1)).tolist()

train_y = survived_logic(train_data)

test_x = np.array(test_data).tolist()

#---------------------------------------------------------------------
#----------------------------model build---------------------------------------

def network_structure():

    network = tfl.layers.core.input_data(shape=[None, 10], name='Input')

    network = tfl.layers.core.fully_connected(network, 10, activation='relu')

    network = tfl.layers.core.fully_connected(network, 22, activation='tanh')

    network = tfl.layers.core.fully_connected(network, 35, activation='relu')

    network = tfl.layers.core.fully_connected(network, 44, activation='relu')

    network = tfl.layers.core.fully_connected(network, 25, activation='relu')

    network = tfl.layers.core.fully_connected(network, 10, activation='relu')

    network = tfl.layers.core.fully_connected(network, 2, activation='softmax')

    network = tfl.layers.estimator.regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='Targets')

    model = tfl.DNN(network)

    return model

#---------------------------------------------------------------------------------

model = network_structure()

#training model

import random

for _ in range(10):

    samp = random.sample(range(len(train_x)), 500)

    train_x_samp = []

    train_y_samp = []

    for j in range(len(samp)):

        train_x_samp.append(train_x[j])

        train_y_samp.append(train_y[j])

    model.fit({'Input': train_x_samp}, {'Targets': train_y_samp}, n_epoch=50, validation_set = 0.3, snapshot_step=100, show_metric=True)
#------------------------prediction-----------------------------------------

temp = model.predict(test_x)

test_y = []

for i in range(len(temp)):

    if temp[i][0] > temp[i][1]:

        test_y.append(1)

    else:

        test_y.append(0)

t_y = pd.Series(test_y).reset_index()

print("\n Predicted output: ")

print(test_y)

plt.figure(figsize=(12,8))

sns.barplot(t_y.index[0:50], test_y[0:50], alpha = 0.8, color=color[0])

plt.xlabel('Test ID', fontsize=12)

plt.ylabel('Survived', fontsize=12)

plt.title('Survival or not in testing data', fontsize=15)

plt.show()
#------------------------------Confusion Matrix---------------------------------

cnf_matrix = confusion_matrix(sub_df.Survived.tolist(), test_y)

df_cm = pd.DataFrame(cnf_matrix, index = ['Non Survived', 'Survived'], columns = ['Non Survived', 'Survived'])

plt.figure(figsize = (12,8))

sns.heatmap(df_cm, annot=True)

plt.ylabel('True label', fontsize=12)

plt.xlabel('Predicted label', fontsize=12)

plt.title('Confusion Matrix', fontsize=15)

plt.show()

sensitivity = cnf_matrix[1][1]/(cnf_matrix[1][1]+cnf_matrix[0][1])

specificity = cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0])

acc = (cnf_matrix[0][0]+cnf_matrix[1][1])/cnf_matrix.sum()

print('The Sensitivity of Survival is {}%'.format(sensitivity*100))

print('The Specificity of Survival is {}%'.format(specificity*100))

print('The Accuracy is {}%'.format(acc*100))
submissions = test_df['PassengerId'].reset_index()

submissions = pd.merge(submissions, t_y, on='index', how='left')

submissions = submissions.drop(['index'], axis = 1)

submissions.columns = ["PassengerId", "Survived"]

submissions.to_csv('my_submissions.csv', index=False)