import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

sns.set_style('whitegrid')



#Print you can execute arbitrary python code

train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
print(train_df.info())
print(test_df.info())
def changeFeatureDataType(df):

    

    # convert sex to numeric value

    df.loc[df['Sex'] == 'male','Sex'] = 0

    df.loc[df['Sex'] == 'female','Sex'] = 1

    

    # convert the Embarked values to numeric values s=0, c=1, q=2

    df.loc[df['Embarked']=='S','Embarked'] = 0

    df.loc[df['Embarked']=='C','Embarked'] = 1

    df.loc[df['Embarked']=='Q','Embarked'] = 2

    

    return df

       

def removeUnwantedfeatures(df):

    drop_columns = ['Ticket','Name','Cabin']

    df = df.drop(drop_columns, 1)

    return df

    

# transform and clean the dataset



def transform_features(df):

    df = changeFeatureDataType(df)

    df = removeUnwantedfeatures(df)

    return df



train_df = transform_features(train_df)

test_df = transform_features(test_df)



print(train_df.head())
print(train_df.isnull().sum())

print('-----------------------------')

print(test_df.isnull().sum())
print(test_df.loc[test_df['Fare'].isnull(),['Embarked','Pclass']])
#check all the similar Fare values of this combination and see if we can get a sorted conclusion

fare_distribution =  train_df.loc[(train_df.Embarked == 0) & (train_df.Pclass == 3), ['Fare']]

fare_distribution = fare_distribution['Fare'].value_counts().head(20)

fare_distribution = fare_distribution.reset_index()

fare_distribution.columns = ['Fare', 'Counts']

print(fare_distribution)
g = sns.lmplot('Fare', 'Counts',data=fare_distribution,fit_reg=False,hue='Fare',x_jitter=5.0,y_jitter=5.0,size=8,scatter_kws={"s": 100})

g.set(xlim=(0, None))

g.set(ylim=(0, None))

plt.title('Embarked = S and Pclass == 3')

plt.xlabel('Fare')

plt.ylabel('Counts')

plt.show()
#Lets put 8.0500 value in the missing fare 

test_df['Fare'] = test_df['Fare'].fillna(8.0500)

print(test_df.isnull().sum())
# check the Fare and Pclass of train_df 

print(train_df.loc[train_df['Embarked'].isnull(),['Fare','Pclass']])
print(train_df.loc[(train_df['Pclass'] == 1) & (train_df['Fare'] == 80.0),['Embarked']])
Embarked_distribution = train_df.loc[(train_df['Fare'] > 79.0) & (train_df['Fare'] < 81.0) & (train_df['Pclass'] == 1), ['Fare','Embarked']]

print(Embarked_distribution['Embarked'].value_counts())

# 1=c and 0=s
train_df['Embarked'] = train_df['Embarked'].fillna(1)

print(train_df.isnull().sum())
#combine both the datasets so that we can get a clear picture of impact of survival on Age feature

titanic_df = train_df.append(pd.DataFrame(data = test_df), ignore_index=True)



titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())



# for the sake of showing the plot for each and every age we will drop all the null values 

# remove the outlier age values from the Age feature

titanic_df['Age1'] = titanic_df.Age

titanic_df['Age1'] = titanic_df[titanic_df['Age1'] < 60]



#Impact visualization of Age on Survival through graph

fig = plt.figure(figsize=(13, 5))

average_age = titanic_df[["Age1", "Survived"]].groupby(['Age1'],as_index=False).mean()

average_age['Age1'] = average_age['Age1'].astype(int)

sns.barplot("Age1", "Survived",data=average_age)

plt.show()
fig = plt.figure(figsize=(13, 5))

alpha = 0.3



titanic_df[titanic_df.Survived==0].Age.value_counts().plot(kind='density', color='#6ACC65', label='Not Survived', alpha=alpha)

titanic_df[titanic_df.Survived==1].Age.value_counts().plot(kind='density', color='#FA2379', label='Survived', alpha=alpha)



plt.xlim(0,80)

plt.xlabel('Age')

plt.ylabel('Survival Count')

plt.title('Age Distribution')

plt.legend(loc ='best')

plt.grid()
fig = plt.figure(figsize=(13, 5))

alpha = 0.3



titanic_df[titanic_df.Survived==0].Sex.value_counts().plot(kind='bar', color='#00FFFF', label='Not Survived', alpha=alpha)

titanic_df[titanic_df.Survived==1].Sex.value_counts().plot(kind='bar', color='#6ACC65', label='Survived', alpha=alpha)



plt.xlabel('Sex')

plt.ylabel('Survival Count')

plt.title('Impact of sex on Survival')

plt.legend(loc ='best')

plt.grid()
sex_survived = pd.crosstab(train_df["Sex"],train_df["Survived"])

parch_survived = pd.crosstab(train_df["Parch"],train_df["Survived"])

pclass_survived = pd.crosstab(train_df["Pclass"],train_df["Survived"])



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(12,5))    

sns.barplot(train_df["Sex"], train_df["Survived"], palette="Set3" ,ax=axis1)

sns.barplot(train_df["Parch"], train_df["Survived"], palette="Set3", ax=axis2)



fig, (axis3,axis4) = plt.subplots(1,2,figsize=(12,5))  

sns.barplot(train_df["Parch"], train_df["Survived"], palette="Set3", ax=axis3)

sns.barplot(train_df["Embarked"], train_df["Survived"], palette="Set3", ax=axis4)



plt.xticks(rotation=90)
from sklearn.linear_model import LogisticRegression

from sklearn import cross_validation



imp_features = ["Pclass", "Sex", "Age", "Fare", "Embarked","SibSp", "Parch"]



model = LogisticRegression(random_state=1)

scores = cross_validation.cross_val_score(

    model,

    train_df[imp_features],

    train_df["Survived"],

    cv=3

)



print(scores.mean())
from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation



imp_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



model = RandomForestClassifier(

    random_state=1,

    n_estimators=150,

    min_samples_split=4,

    min_samples_leaf=2

)



scores = cross_validation.cross_val_score(

    model,

    train_df[imp_features],

    train_df["Survived"],

    cv=3

)



print(scores.mean())
def submission_result(model, train_df, test_df, predictors, filename):



    model.fit(train_df[predictors], train_df["Survived"])

    predictions = model.predict(test_df[predictors])



    submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": predictions

    })

    

    submission.to_csv(filename, index=False)

    

    

# call the submission_result function to submit the result

submission_result(model, train_df, test_df, imp_features, "titanic_result.csv")

    