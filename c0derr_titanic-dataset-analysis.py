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
import pandas as pd

titanic=pd.read_csv("/kaggle/input/mlcourse/titanic_train.csv")

titanic.head()
# tail shows last 5 rows

titanic.tail()
# columns gives column names of features

titanic.columns
# shape gives number of rows and columns in a tuble

titanic.shape

#Our dataSet has 12 columns ,891 rows
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

titanic.info()
# For example lets look frequency of gender in ship

print(titanic['Sex'].value_counts(dropna=False))  # if there are nan values that also be counted

# As it can be seen below there are 577 male and 314 female person
# For example max HP is 255 or min defense is 5

titanic.describe() #ignore null entries
# For example: compare attack of pokemons that are legendary  or not

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

titanic.boxplot(column='SibSp',by = 'Sex')
titanic.boxplot(column='Fare',by = 'Sex')
topTen=titanic.head(10)

topTen
melted = pd.melt(frame=topTen,id_vars = 'Name', value_vars= ['Age','Fare'])

melted
# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'Name', columns = 'variable',values='value')
#Vertical Concatenating

# Firstly lets create 2 data frame

data1 = titanic.head()

data2= titanic.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

#ignore_index=True :ignore the old indexis.Assign new indexis

conc_data_row
#Horizontal Concatenating

data1 = titanic['Name'].head()

data2= titanic['Age'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 1 : adds dataframes in column

conc_data_col



titanic.dtypes
# lets convert object(str) to categorical and int to float.

titanic['Sex'] = titanic['Sex'].astype('category')

titanic['Parch'] = titanic['Parch'].astype('float')

titanic.dtypes
titanic.info()


# Lets chech Age

titanic["Age"].value_counts(dropna =False)

# As you can see, there are 177 NAN value
# Lets drop nan values

data1=titanic   # also we will use data to fill missing value so I assign it to data1 variable

data1["Age"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable.

#Changes automatically assigned to data

# So does it work ?

#(Age i olmayan insanları çıkar.inplace=True :çıkar ve çıkarılmış halini data1 e kaydet)
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true

# assert 1==2 # return error because it is false
assert  titanic['Age'].notnull().all() # returns nothing because we drop nan values
titanic["Age"].fillna('empty',inplace = True)
assert  titanic['Age'].notnull().all() # returns nothing because we do not have nan values
# # With assert statement we can check a lot of thing. For example

assert titanic.columns[0] == 'PassengerId' #true

assert titanic.PassengerId.dtypes == np.int64 #true ,so it wont show us any error
titanic.info()
import numpy as np

import pandas as pd

import seaborn as sns

import timeit

titanic[

    (titanic.Sex=='female')

    &( titanic["Pclass"].isin([1,3]))

    &( titanic.Age >40)

    & (titanic.Survived == 0)

]
towns_dic = {

    'name': ['Southampton', 'Cherbourg', 'Queenstown', 'Montevideo'],

    'country': ['United Kingdom', 'France', 'United Kingdom', 'Uruguay'],

    'population': [236900, 37121, 12347, 1305000],

    'age': [np.random.randint(500, 1000) for _ in range(4)]

}

towns_df = pd.DataFrame(towns_dic)
sns.distplot(titanic.Age.dropna())

g = sns.FacetGrid(titanic, row='Survived', col='Pclass')

g.map(sns.distplot, "Age")
#sns.jointplot(data=titanic, x='Age', y='Pclass', kind='reg', color='g')

df = titanic.pivot_table(index='Embarked', columns='Survived', values='Fare', aggfunc=np.median)

sns.heatmap(df, annot=True, fmt=".1f")

#corelation matrix
data=titanic.copy()
data.tail()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import csv

import re



from keras.models import Sequential

from keras.layers import Dense,Activation

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam
data.Cabin
def preprocess(data):

    data.Cabin.fillna('0',inplace=True)

    data.loc[data.Cabin.str[0]=='A','Cabin']=1

    data.loc[data.Cabin.str[0]=='B','Cabin']=2

    data.loc[data.Cabin.str[0]=='C','Cabin']=3

    data.loc[data.Cabin.str[0]=='D','Cabin']=4

    data.loc[data.Cabin.str[0]=='E','Cabin']=5

    data.loc[data.Cabin.str[0]=='F','Cabin']=6

    data.loc[data.Cabin.str[0]=='G','Cabin']=7

    data.loc[data.Cabin.str[0]=='T','Cabin']=8

    

    data['Sex'].replace('female',1,inplace=True)

    data['Sex'].replace('male',2,inplace=True)



    data['Embarked'].replace('S',1,inplace=True)

    data['Embarked'].replace('C',2,inplace=True)

    data['Embarked'].replace('Q',3,inplace=True)

    

    #I wanna replace empty spaces with median value

    data['Age'].fillna(data['Age'].median(),inplace=True) 

    data['Fare'].fillna(data['Fare'].median(),inplace=True)

    data['Embarked'].fillna(data['Embarked'].median(),inplace=True)

    

    #If u wanna delete all empty values:

    #data.dropna(subset=['Fare','Embarked'],inplace=True,how='any')

    #BUT we dont prefer this option because when yu do it, u ll lose all info about these people

    

    return data
def group_titles(data):

    data['Names'] = data['Name'].map(lambda x: len(re.split(' ', x)))

    data['Title'] = data['Name'].map(lambda x: re.search(', (.+?) ', x).group(1))

    data['Title'].replace('Master.', 0, inplace=True)

    data['Title'].replace('Mr.', 1, inplace=True)

    data['Title'].replace(['Ms.','Mlle.', 'Miss.'], 2, inplace=True)

    data['Title'].replace(['Mme.', 'Mrs.'], 3, inplace=True)

    data['Title'].replace(['Dona.', 'Lady.', 'the Countess.', 'Capt.', 'Col.', 'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'the'], 4, inplace=True)



     

    
def data_subset(data):

    features = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Names', 'Title', 'Age', 'Cabin'] #, 'Fare', 'Embarked']

    lengh_features = len(features)

    subset = data[features]#.fillna(0)

    return subset, lengh_features
#Design the model

#batch_size= cluster size 

#Dense =layer

#activation function=Can be softplus,ReLU,Sigmoid ..etc.

#lr =learning rate



def create_model(train_set_size, input_length, num_epochs, batch_size):

    model = Sequential()

    model.add(Dense(7, input_dim=input_length, activation='softplus'))

    model.add(Dense(3, activation='softplus'))

    model.add(Dense(1, activation='softplus'))



    lr = .001

    adam0 = Adam(lr = lr)



    #Execute the model ,if you find better results save these weights

    

    model.compile(loss='binary_crossentropy', optimizer=adam0, metrics=['accuracy'])

    filepath = 'weights.best.hdf5'

    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

    callbacks_list = [checkpoint]



    history_model = model.fit(X_train[:train_set_size], Y_train[:train_set_size], callbacks=callbacks_list, epochs=num_epochs, batch_size=batch_size, verbose=0) #40, 32

    return model, history_model
def plots(history):

    loss_history = history.history['loss']

    acc_history = history.history['accuracy']

    epochs = [(i + 1) for i in range(num_epochs)]



    ax = plt.subplot(211)

    ax.plot(epochs, loss_history, color='red')

    ax.set_xlabel('Epochs')

    ax.set_ylabel('Error Rate\n')

    ax.set_title('Error Rate per Epoch\n')



    ax2 = plt.subplot(212)

    ax2.plot(epochs, acc_history, color='blue')

    ax2.set_xlabel('Epochs')

    ax2.set_ylabel('Accuracy\n')

    ax2.set_title('Accuracy per Epoch\n')



    plt.subplots_adjust(hspace=0.8)

    plt.savefig('Accuracy_loss.png')

    plt.close()


#test=pd.read_csv("/kaggle/input/mlcourse/titanic_test.csv")



def test(batch_size):

    test = pd.read_csv("/kaggle/input/mlcourse/titanic_test.csv", header=0)

    test_ids = test['PassengerId']

    test = preprocess(test)

    group_titles(test)

    testdata, _ = data_subset(test)



    X_test = np.array(testdata).astype(float)



    output = model.predict(X_test, batch_size=batch_size, verbose=0)

    output = output.reshape((418,))



    # Sonuçları ondalık sayı yerine 0-1 olarak değiştirebilirsiniz

    #outputBin = np.zeros(0)

    #for element in output:

    #    if element <= .5:

    #         outputBin = np.append(outputBin, 0)

    #    else:

    #        outputBin = np.append(outputBin, 1)

    #output = np.array(outputBin).astype(int)



    column_1 = np.concatenate((['PassengerId'], test_ids ), axis=0 )

    column_2 = np.concatenate( ( ['Survived'], output ), axis=0 )



    f = open("output.csv", "w")

    writer = csv.writer(f)

    for i in range(len(column_1)):

        writer.writerow( [column_1[i]] + [column_2[i]])

    f.close()
# results should be reproductible

seed = 7

np.random.seed(seed)





train = pd.read_csv('/kaggle/input/mlcourse/titanic_train.csv', header=0)





preprocess(train)

group_titles(train)





num_epochs = 100

batch_size = 32







traindata, lengh_features = data_subset(train)



Y_train = np.array(train['Survived']).astype(int)

X_train = np.array(traindata).astype(float)





train_set_size = int(.67 * len(X_train))





model, history_model = create_model(train_set_size, lengh_features, num_epochs, batch_size)



plots(history_model)





X_validation = X_train[train_set_size:]

Y_validation = Y_train[train_set_size:]





loss_and_metrics = model.evaluate(X_validation, Y_validation, batch_size=batch_size)

print ("loss_and_metrics")



test(batch_size)
#

#plt.show("./Accuracy_loss.png")



from IPython.display import Image

Image("./Accuracy_loss.png")


output = pd.read_csv('./output.csv', header=0)

output.head()
