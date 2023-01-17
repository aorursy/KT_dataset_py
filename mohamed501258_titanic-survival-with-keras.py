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
#Libs

import keras

import pandas as pd

import matplotlib.pyplot as plt
#Data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

full = pd.concat([train, test]).reset_index()

full.info()
# Visualization

# Survival/Death - Sex

survived_female, survived_male = train[train['Survived']==1]['Sex'].value_counts().sort_index()

dead_female, dead_male = train[train['Survived']==0]['Sex'].value_counts().sort_index()

width_bar= 0.25

plt.bar(['Survival of F','Survival of M'], [survived_female, survived_male], label='Survival', width=width_bar)

plt.bar(['Death of M','Death of F'], [dead_male, dead_female], label='Death', width=width_bar)



plt.title('Survival/Death - Sex')

plt.legend()
# Survival/Death - Fare

s_fare = train[train['Survived']==1]['Fare'].value_counts().sort_index()

d_fare = train[train['Survived']==0]['Fare'].value_counts().sort_index()

fig, ax = plt.subplots(2,1)





ax[0].bar(s_fare.index, s_fare, label='Survival', width=width_bar+6)        

ax[1].bar(d_fare.index, d_fare, label='Death', width=width_bar+6)



ax[0].set_ylim([0,full[full['Survived']==0]['Fare'].value_counts().max()+10])

ax[1].set_ylim([0,full[full['Survived']==0]['Fare'].value_counts().max()+10])

ax[0].title.set_text('Survival/Death - Fare')

plt.legend()

plt.show()
# Survival/Death - Pclass 

s_pclass1, s_pclass2, s_pclass3 = train[train['Survived']==1]['Pclass'].value_counts().sort_index()# survival of pclass 

d_pclass1, d_pclass2, d_pclass3 = train[train['Survived']==0]['Pclass'].value_counts().sort_index()



# Visualization

fig2, ax2 = plt.subplots(2,1)



ax2[0].bar(['pclass1', 'pclass2', 'pclass3'], [s_pclass1, s_pclass2, s_pclass3], label='Survival', width=width_bar)     

ax2[1].bar(['pclass1', 'pclass2', 'pclass3'], [d_pclass1, d_pclass2, d_pclass3], label='Death', width=width_bar)



ax2[0].title.set_text('Survival/Death - Pclass')



# Setting y-axis limit

ax2[0].set_ylim([0,full[full['Survived']==0]['Pclass'].value_counts().max()+10])

ax2[1].set_ylim([0,full[full['Survived']==0]['Pclass'].value_counts().max()+10])

plt.legend()

plt.show()
# Survival/Death - Age

s_age = train[train['Survived']==1]['Age'].value_counts().sort_index()# survival of age

d_age = train[train['Survived']==0]['Age'].value_counts().sort_index()



# Visualization

fig3, ax3 = plt.subplots(2,1)



ax3[0].bar(s_age.index, s_age, label='Survival', width=width_bar)     

ax3[1].bar(d_age.index, d_age, label='Death', width=width_bar)



ax3[0].title.set_text('Survival/Death - Age')



# Setting y-axis limit

ax3[0].set_ylim([0,full[full['Survived']==0]['Age'].value_counts().max()+10])

ax3[1].set_ylim([0,full[full['Survived']==0]['Age'].value_counts().max()+10])

plt.legend()

plt.show()
# Survival/Death - SibSp

s_sibsp = train[train['Survived']==1]['SibSp'].value_counts().sort_index()# survival of sibsp

d_sibsp = train[train['Survived']==0]['SibSp'].value_counts().sort_index()



# Visualization

fig4, ax4 = plt.subplots(2,1)



ax4[0].bar(s_sibsp.index, s_sibsp, label='Survival', width=width_bar)     

ax4[1].bar(d_sibsp.index, d_sibsp, label='Death', width=width_bar)



ax4[0].title.set_text('Survival/Death - SibSp')



# Setting y-axis limit

ax4[0].set_xlim([-1,9])

ax4[1].set_xlim([-1,9])

ax4[0].set_ylim([0,full[full['Survived']==0]['SibSp'].value_counts().max()+10])

ax4[1].set_ylim([0,full[full['Survived']==0]['SibSp'].value_counts().max()+10])



plt.legend()

plt.show()
# Survival/Death - SibSp

s_parch = train[train['Survived']==1]['Parch'].value_counts().sort_index()# survival of parch

d_parch = train[train['Survived']==0]['Parch'].value_counts().sort_index()



# Visualization

fig5, ax5 = plt.subplots(2,1)



ax5[0].bar(s_parch.index, s_parch, label='Survival', width=width_bar)     

ax5[1].bar(d_parch.index, d_parch, label='Death', width=width_bar)



ax5[0].title.set_text('Survival/Death - Parch')



# Setting y-axis limit

ax5[0].set_xlim([-1,7])

ax5[1].set_xlim([-1,7])

ax5[0].set_ylim([0,full[full['Survived']==0]['Parch'].value_counts().max()+10])

ax5[1].set_ylim([0,full[full['Survived']==0]['Parch'].value_counts().max()+10])



plt.legend()

plt.show()
# Survival/Death - Embarkation

s_embark = train[train['Survived']==1]['Embarked'].value_counts().sort_index()# survival of embarkation

d_embark = train[train['Survived']==0]['Embarked'].value_counts().sort_index()



# Visualization

fig6, ax6 = plt.subplots(2,1)



ax6[0].bar(s_embark.index, s_embark, label='Survival', width=width_bar)     

ax6[1].bar(d_embark.index, d_embark, label='Death', width=width_bar)



ax6[0].title.set_text('Survival/Death - Embarkation')



# Setting y-axis limit

#ax6[0].set_xlim([-1,7])

#ax6[1].set_xlim([-1,7])

ax6[0].set_ylim([0,full[full['Survived']==0]['Embarked'].value_counts().max()+10])

ax6[1].set_ylim([0,full[full['Survived']==0]['Embarked'].value_counts().max()+10])



plt.legend()

plt.show()
# Handling data for the model

# For Sex

sex_map = {'female':1, 'male':2}

full['dummy_sex'] = full['Sex'].map(sex_map)



# For Name Title

full['Title_of_the_name'] = full.Name.str.extract('([A-Za-z]+)\.')

title_mapping = {'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Master' : 4,'Don': 4, 'Rev' : 4,'Dr' : 4,'Mme': 4, 'Ms': 4, 'Major': 4,

 'Lady': 4, 'Sir': 4, 'Mlle': 4, 'Col': 4, 'Capt': 4, 'Countess': 4, 'Jonkheer': 4,'Dona': 4}



full['Title'] = full['Title_of_the_name'].map(title_mapping)

full['Title'] = full['Title'].fillna(0)

full['Title'].value_counts()
full.drop(['Title_of_the_name', 'Sex'], axis=1, inplace=True)

full.rename(columns={'dummy_sex':'Sex'}, inplace=True)



# filling missing age based on title and the pclass

full['Age'] = full.groupby(['Title', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
full.drop(['Cabin'], axis=1, inplace=True)

full.Age=  full.Age.astype(int)
full['FamSize'] = full['SibSp'] + full['Parch'] + 1
full.drop(['index'], axis=1, inplace=True)
full.Embarked.fillna('S', inplace=True)
full.Fare.fillna(full['Fare'].median(), inplace=True)
full.Embarked = full.Embarked.map({'S':1, 'C':2, 'Q':3})
full.drop(['Name','SibSp','Parch','Ticket'],axis=1, inplace=True)
full.drop('PassengerId',axis=1, inplace=True)
#normalize Data

full.Age = full['Age']/full.Age.max()

full.Fare = full['Fare']/full.Fare.max()
# After handling data we split them again into train and test

# Train dataset

train = full[ :len(train)+1]

# Test dataset

test = full[len(train): ]

test.drop('Survived',axis=1, inplace=True)
#train.Survived = train.Survived.astype(int)

train.Title.isnull().sum()

#train.Survived.fillna(1.0, inplace=True)

train.head()
# preparing model

from keras.models import Sequential

from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(10, activation='relu', input_shape=(7,)))

model.add(Dropout(0.2))

model.add(Dense(150, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(100, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())



history = model.fit(train.drop('Survived', axis=1), train['Survived'], validation_split=0.2, epochs=180, shuffle=True)
# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
pred = model.predict(test).round().astype(int)

pred
pred = pred.reshape(417,)

pred
passengerId= test.index.values

df = pd.DataFrame({'PassengerId': passengerId, 'Survived': pred})

df.to_csv('submission.csv', index=False)