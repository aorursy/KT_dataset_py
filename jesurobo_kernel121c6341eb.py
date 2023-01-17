# import all modules needed for this task

from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os
# ignore Deprecation Warning

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
# Load training and test data

print(os.listdir("../input"))

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")



# combine training and test datasets

df = df_train.append(df_test,ignore_index = True,sort=False )



# A quick look at the first 3 rows for train and test datasets

display(df_train.head(3))



display(df_test.head(3))

# A quick look at the first and last 5 rows for combined dataset to see if it properly aligns 

display(df.head())



display(df.tail())
# some quick inspections:check no.of rows and columns in train.csv,test.csv,df.. also check titles/headers in df

df_train.shape, df_test.shape,df.shape, df.columns.values
# Feature processing: Pclass , count the number of NaN for Pclass

# there is no null value in Pclass.



df["Pclass"].notnull().value_counts()

# check the correlation between Pclass and Survived.

"""Pclass seems to be a very useful feature....there seems to be a correlation"""



df[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean()



"""A more friendly approach with Neural network is to scale and normalize all variables between the range of 0 to 1.

More specifically if the output activation functon is sigmoid""" 

#Now we can use dummy encoding to Normalise Pclass and drop the main Pclass.



df = pd.concat([df, pd.get_dummies(df['Pclass'],prefix="Pclass")], axis=1).drop(labels="Pclass",axis=1)

df.head(3)
# Feature processing:Name

# count the number of NaN for Name

df["Name"].notnull().value_counts()

# see first 3 rows for column "Name"

df.Name.head(3)
"""Name itself does not gives any deduction, but the titles in the names is paramount since it contains information 

of gender/status"""



#Let's extract the titles from these names and assign it to a variable called "Title".



df['Title'] = df.Name.map( lambda x: x.split( ',' )[1].split(".")[0].strip())

df = df.drop(labels="Name",axis=1)



df.head()
# inspect the amount of people for each title

df['Title'].value_counts()
"""Looks like the main ones are "Mr", "Miss", "Mrs", "Master". 

Some of the others can be be merged into some of these four categories. For the rest, it will be called 'Others"""



df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace(['Mme','Lady','Ms'], 'Mrs')

df.Title.loc[ (df.Title !=  'Master') & (df.Title !=  'Mr') & (df.Title !=  'Miss') 

             & (df.Title !=  'Mrs')] = 'Others'



# inspect the correlation between Title and Survived

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#Now we can use dummy encoding to Normalise Title and drop the main Name

df = pd.concat([df, pd.get_dummies(df['Title'],prefix="Title")], axis=1).drop(labels="Title",axis=1)

df.head(3)
# Feature processing:Sex , check if there is any NaN



df.Sex.isnull().sum(axis=0)
df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
# Feature processing: sex, convert into binary

df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

display(df.head(3))
# Feature processing:Cabin, check if there is any NaN, and also the number of rows in Cabin

df.Cabin.isnull().sum(axis=0), df["Cabin"].shape
#Cabin: Too many NaN in Cabin,i.e 1014 out of 1309. so Cabin will also be dropped later
#Feature processing:SibSp and Parch, check if there is any NAN

"""SibSp and Parch already has values 0 and 1, no null values , so no further processing."""



df.SibSp.isnull().sum(axis=0), df.Parch.isnull().sum(axis=0)
#Feature processing:Embarked, check if there is any NAN

df.Embarked.isnull().sum(axis=0)
# check for most common categories of embarked

df.describe(include=['O']) 
# S is the most common category. fill the 2 NaN spaces with S.

df.Embarked.fillna('S' , inplace=True )

df.Embarked.isnull().sum(axis=0)
#Now we can use dummy encoding for Embarked and drop the main Embarked

df = pd.concat([df, pd.get_dummies(df['Embarked'],prefix="Embarked")], axis=1).drop(labels="Embarked",axis=1)

df.head()
#Removing insignificant features that do not have predictive power:

#name,ticket, passengerId, Fare

#Cabin may have predictive power but has too many NAs (1014 out of 1309),and so have to be removed as well.



df = df.drop(['Ticket','PassengerId','Cabin','Fare'],axis = 1)

display(df.head(2))
# Feature processing: Age, Age is a very important feature

"""there are many NaN value(263) in this feature so first we will replace them with a random number

   generated between (mean - std) and (mean + std) """



age_avg   = df['Age'].mean()

print (age_avg)

age_std    = df['Age'].std()

print (age_std)

age_null_count = df['Age'].isnull().value_counts()[1]

print (age_null_count)

    

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

print(age_null_random_list)

df['Age'][np.isnan(df['Age'])] = age_null_random_list

"""since the output layer uses a sigmoid activation function,it is in my opinion to scale and normalize the datasets

   so that the learning does not shift towards the higher weight"""

  

# normalize the dataset

scaler = MinMaxScaler(feature_range=(0, 1))

df = pd.DataFrame(scaler.fit_transform(df))

df.head()
df.head(2)

df.shape
#Modelling and Prediction.drop survived from X_train dataset.

X_train = df[0:891].drop([0], axis=1)

y_train = df[0:891][0]

X_test  = df[891:].drop([0], axis=1)



X_train.shape
# fix random seed for reproducibility

np.random.seed(7)

# create model

model = Sequential()

model.add(Dense(12, input_dim=15, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model

model.fit(X_train, y_train, epochs=200,validation_split=0.33, batch_size=10)

# evaluate the model

scores = model.evaluate(X_train,y_train)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
out_pred = model.predict(X_test)

print(out_pred)

out_final = (out_pred > 0.5).astype(int).reshape(X_test.shape[0])

output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': out_final})

print(output)

output.to_csv('ANN prediction.csv', index=False)