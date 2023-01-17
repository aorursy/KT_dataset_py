# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import plotly.express as px

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

import wordcloud 

import collections

import pandas as pd
train_data=pd.read_csv('/kaggle/input/titanic/train.csv')

test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
# Observe the meaning of each label in the data

train_data.columns.values
#Check the data base

train_data.info()

#(Cabin)The cabin number information contains a large number of null values, which are not very helpful for analysis and prediction
train_data.describe(include=['O'])

#Discover that the name in the dataset is unique

#There were 577 male passengers

#The number of people under S boarding port is the largest, 644
#Observe the first five data and further screen the meaningless labels

train_data.head(5)

#PassengerId can be found to have no real meaning and should be excluded in the analysis
#Here we can use word clouds to perform a statistical rendering of passenger names

name_str_file=''

for i in range(len(train_data)):

    name_str_file+=train_data[train_data.index==i]['Name'].tolist()[0]

    

#Remove words that have no real meaning

STOPWORDS=['Mr','Miss','Mrs','Dr','Master']

wc = wordcloud.WordCloud(stopwords=STOPWORDS, width=800, height=600, mode='RGBA', background_color=None).generate(name_str_file)



plt.figure(figsize=(12,8))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()
name_str_file=name_str_file.split(' ')

for i in name_str_file:

    if i in ['Mr.','Mrs.','Miss.','Dr.','Master.']:

        name_str_file.remove(i)

#We can find that the following characters appear more frequently, and then analyze the death rate of each name

c=collections.Counter(name_str_file)

name_list=[]

for i in range(10):

    name_list.append(c.most_common(10)[i][0])

#most common name

name_list
#Calculate the death rate for each name

name_list=set(name_list)

top_name_dataframe=pd.DataFrame()

for i in range(len(train_data)):

    name_str=train_data[train_data.index==i]['Name'].values[0]

    name_str_copy=set(name_str.split(' '))

    if name_str_copy&name_list!=set():

        passage=train_data[train_data['Name']==name_str]

        passage['Name']=list(name_str_copy&name_list)[0]

        top_name_dataframe=pd.concat([top_name_dataframe,passage])
top_name_dataframe[['Name', 'Survived']].groupby(['Name'], as_index=False).mean().sort_values(by='Survived', ascending=False)
c.most_common(10)

#By comparison, passengers with 'Johan' in their names had the lowest survival rate of just 13 percent
#It can be seen from the above that the non-empty labels with meanings are Sex, Pclass, SibSp and Parch, which are analyzed one by one

train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#basic visualization

plt.style.use('dark_background')

grid = sns.FacetGrid(train_data, row='Pclass', col='Sex', height=4, aspect=2)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend(fontsize=12)

plt.show(grid)
train_data['Ticket'].unique()[:5]

#You can find that the Ticket tag data is too cluttered and you can eliminate it for analysis efficiency
# here is the gender numeric code for subsequent analysis and prediction

#train_data_pro['Sex'] = train_data_pro['Sex'].apply(lambda x: str(x).replace('female', '2') if 'female' in str(x) else x)

#train_data_pro['Sex'] = train_data_pro['Sex'].apply(lambda x: str(x).replace('male', '1') if 'male' in str(x) else x)

#train_data_pro['Sex'] = train_data_pro['Sex'].astype(int)



#Handle missing values

#Because there is less missing data of Age, the data that is missing from Age will have an acceptable impact on the data set

train_data=train_data[train_data['Age'].notna()]

train_data.reset_index(drop=True, inplace=True)
train_data[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# you can find that the age is too dispersed, we can set up an age tag to analyze the data
pd.cut(train_data['Fare'], 6)
train_data[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#You can see that the cost is too spread out,we can create a cost tag to analyze the data
pd.cut(train_data['Fare'], 6)
#Cross the Age level

#(0.34, 13.683] < (13.683, 26.947] < (26.947, 40.21] < (40.21, 53.473] < (53.473, 66.737] < (66.737, 80.0]

train_data.loc[ train_data['Age'] <= 13.683, 'Agerank'] = 1

train_data.loc[(train_data['Age'] > 13.683) & (train_data['Age'] <= 26.947), 'Agerank'] = 2

train_data.loc[(train_data['Age'] > 26.947) & (train_data['Age'] <= 40.21), 'Agerank'] = 3

train_data.loc[(train_data['Age'] > 40.21) & (train_data['Age'] <= 53.473), 'Agerank'] = 4

train_data.loc[(train_data['Age'] > 53.473) & (train_data['Age'] <= 66.737), 'Agerank'] = 5

train_data.loc[(train_data['Age'] > 66.737) & (train_data['Age'] <= 80.0), 'Agerank'] = 6

train_data['Agerank']=train_data['Agerank'].astype(int)

#Cross the Fare level

#(-0.512, 85.388] < (85.388, 170.776] < (170.776, 256.165] < (256.165, 341.553] < (341.553, 426.941] < (426.941, 512.329]

train_data.loc[ train_data['Fare'] <= 85.388, 'Farerank'] = 1

train_data.loc[(train_data['Fare'] > 85.388) & (train_data['Fare'] <= 170.776), 'Farerank'] = 2

train_data.loc[(train_data['Fare'] > 170.776) & (train_data['Fare'] <= 256.165), 'Farerank'] = 3

train_data.loc[(train_data['Fare'] > 256.165) & (train_data['Fare'] <= 341.533), 'Farerank'] = 4

train_data.loc[(train_data['Fare'] > 341.533) & (train_data['Fare'] <= 426.941), 'Farerank'] = 5

train_data.loc[(train_data['Fare'] > 426.941), 'Farerank'] = 6

train_data['Farerank']=train_data['Farerank'].astype(int)
fig1=px.bar(x=train_data.groupby('Farerank').count().index.tolist(),\

       y=train_data.groupby('Farerank').count()['Survived'].tolist())

fig1.update_traces(marker_line_width=4,marker_line_color='black',selector=dict(type="bar"))

fig1.update_layout(title='Age distribution',title_font_family='Courier New, monospace')

fig1.update_layout(xaxis_title='Age',yaxis_title='Amount')

         



fig2=px.bar(x=train_data.groupby('Farerank').count().index.tolist(),\

       y=train_data.groupby('Farerank').count()['Survived'].tolist())

fig2.update_traces(marker_line_width=4,marker_line_color='black',selector=dict(type="bar"),marker_color='red')

fig2.update_layout(title='Fare distribution',title_font_family='Courier New, monospace')

fig2.update_layout(xaxis_title='Fare',yaxis_title='Amount')



fig1.show()

fig2.show()
# here is the gender numeric code for subsequent analysis and prediction

for dataset in [train_data,test_data]:

    dataset['Sex'] = dataset['Sex'].apply(lambda x: str(x).replace('female', '0') if 'female' in str(x) else x)

    dataset['Sex'] = dataset['Sex'].apply(lambda x: str(x).replace('male', '1') if 'male' in str(x) else x)

    dataset['Sex'] = dataset['Sex'].astype(int)

    dataset['Embarked'] = dataset['Embarked'].apply(lambda x: str(x).replace('C','1') if 'C' in str(x) else x)

    dataset['Embarked'] = dataset['Embarked'].apply(lambda x: str(x).replace('S','2') if 'S' in str(x) else x)

    dataset['Embarked'] = dataset['Embarked'].apply(lambda x: str(x).replace('Q','3') if 'Q' in str(x) else x)

    dataset['Embarked'].fillna('2',inplace=True)

    dataset['Embarked']=dataset['Embarked'].astype(int)
train_data['Age'].fillna(0,inplace=True)

test_data['Age'].fillna(test_data['Age'].mean()-10.5,inplace=True)

train_data['Fare'].fillna(0,inplace=True)

test_data['Fare'].fillna(0,inplace=True)
train_data=train_data[[ 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked','Survived']]

#Merge Sibsp and Parch into the Family tag

train_data['Family']=train_data['SibSp']+train_data['Parch']

train_data=train_data.drop(['Parch', 'SibSp'], axis=1)

#Normalize the data form

train_data['Pclass']=train_data['Pclass'].astype(int)

train_data['Age']=train_data['Age'].astype(int)

train_data['Family']=train_data['Family'].astype(int)



test_data=test_data[[ 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked','PassengerId']]    

test_data['Family']=test_data['SibSp']+test_data['Parch']

test_data=test_data.drop(['Parch', 'SibSp'], axis=1)
import numpy as np

from keras.models import Sequential

from keras.optimizers import SGD

from keras.utils import np_utils

from keras.layers.core import Dense, Activation, Dropout
y = train_data["Survived"]



features = ["Pclass", "Sex", "Fare", "Embarked",'Family','Age']



X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])
X['Pclass']=X['Pclass']/X['Pclass'].max()

X['Fare']=X['Fare']/X['Fare'].max()

X['Age']=X['Age']/X['Age'].max()

X['Family']=X['Family']/X['Family'].max()

X['Embarked']=X['Embarked']/X['Embarked'].max()



X_test['Pclass']=X_test['Pclass']/X_test['Pclass'].max()

X_test['Fare']=X_test['Fare']/X_test['Fare'].max()

X_test['Age']=X_test['Age']/X_test['Age'].max()

X_test['Family']=X_test['Family']/X_test['Family'].max()

X_test['Embarked']=X_test['Embarked']/X_test['Embarked'].max()
train_data=pd.read_csv('/kaggle/input/titanic/train.csv')

test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
train_data=train_data[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

test_data=test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
train_data=train_data[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

test_data=test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

test_data['Sex']=test_data['Sex'].astype('category')

test_data['Embarked']=test_data['Embarked'].astype('category')

test_data['Age']=(test_data['Age']/test_data['Age'].max()).astype('float32')

test_data['Fare']=(test_data['Fare']/test_data['Fare'].max()).astype('float32')

test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)

test_data['Embarked'].fillna('C',inplace=True)

train_data['Sex']=train_data['Sex'].astype('category')

train_data['Embarked']=train_data['Embarked'].astype('category')

train_data['Age']=(train_data['Age']/train_data['Age'].max()).astype('float32')

train_data['Fare']=(train_data['Fare']/train_data['Fare'].max()).astype('float32')

train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)

train_data['Embarked'].fillna('C',inplace=True)



for dataset in [train_data,test_data]:

    dataset['Sex'] = dataset['Sex'].apply(lambda x: str(x).replace('female', '0') if 'female' in str(x) else x)

    dataset['Sex'] = dataset['Sex'].apply(lambda x: str(x).replace('male', '1') if 'male' in str(x) else x)

    dataset['Sex'] = dataset['Sex'].astype(int)

    dataset['Embarked'] = dataset['Embarked'].apply(lambda x: str(x).replace('C', '1') if 'C' in str(x) else x)

    dataset['Embarked'] = dataset['Embarked'].apply(lambda x: str(x).replace('S', '2') if 'S' in str(x) else x)

    dataset['Embarked'] = dataset['Embarked'].apply(lambda x: str(x).replace('Q', '3') if 'Q' in str(x) else x)

    dataset['Embarked'] = dataset['Embarked'].astype(int)
train_values=train_data.iloc[:,1:].values

test_values=test_data.iloc[:,:].values

train_y = np_utils.to_categorical(train_data.iloc[:,0].values,2)
model = Sequential()

model.add(Dense(output_dim=20, input_dim=7,activation='relu'))

model.add(Activation('tanh'))

model.add(Dropout(0.25))

model.add(Dense(20))    

model.add(Dropout(0.25))

model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_values, train_y, batch_size=100, epochs=525)
pre=model.predict_classes(test_values)
pre=model.predict_classes(test_values)
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pre})

print(output)

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
prediction=pd.read_csv('my_submission.csv')