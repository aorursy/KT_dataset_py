import numpy as np 

import pandas as pd 

from pandas import Series, DataFrame
# load data from csv files as a DataFrame

train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# preview two line of the train dataset

train_df.head(2)
print("The shape of train dataset:")

print(train_df.shape)



#compre the shape of dateset with  the result of info() method, 

#we can learn that which column there are missing value ,then we'll process the missing value

# For example, there two missing values in the train-dateset's "Embarked" column

print("show the imformation about the train dataset:" )

train_df.info() 



print("----------------------------")



print("The shape of test dataset:")

print(test_df.shape)

print("show the imformation about the test dataset:" )

test_df.info()
# drop unnecessary columns, these columns won't be useful in analysis and prediction

train_df = train_df.drop(['PassengerId','Name','Ticket'], axis=1)

test_df    = test_df.drop(['Name','Ticket'], axis=1)
# use seaborn to visualize the data

import seaborn as sns 

import matplotlib.pyplot as plt

#fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5)

# plot the  Histogram for "Embarked" column

sns.countplot(x="Embarked",data=train_df)
# we can see the "S" are most common in the "Embarked" column

# only in train_df, fill the two missing values with the most occurred value, which is "S".

train_df["Embarked"] = train_df["Embarked"].fillna("S")



# plot

sns.factorplot('Embarked','Survived', data=train_df,size=4,aspect=3)



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)

# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)

sns.countplot(x='Embarked', data=train_df, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0], ax=axis2)



# group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc = train_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)



# Either to consider Embarked column in predictions,

# and remove "S" dummy variable, 

# and leave "C" & "Q", since they seem to have a good rate for Survival.



# OR, don't create dummy variables for Embarked column, just drop it, 

# because logically, Embarked doesn't seem to be useful in prediction.



embark_dummies_train  = pd.get_dummies(train_df['Embarked'])

embark_dummies_train.drop(['S'], axis=1, inplace=True)



embark_dummies_test  = pd.get_dummies(test_df['Embarked'])

embark_dummies_test.drop(['S'], axis=1, inplace=True)



train_df = train_df.join(embark_dummies_train)

test_df    = test_df.join(embark_dummies_test)



train_df.drop(['Embarked'], axis=1,inplace=True)

test_df.drop(['Embarked'], axis=1,inplace=True)
