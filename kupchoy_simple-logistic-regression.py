import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
def noAgetoMed(df):

    median=df.Age.median()

    df.Age=df.Age.fillna(median)

    return df



def embarked(df):

    df.Embarked=df.Embarked.fillna('S')

    df.Embarked=df.Embarked.map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    return df



def fare_bands(df):

    bins = ( 0, 10, 52, 512)

    group_int = [3,2,1]

    cat = pd.cut(df.Fare, bins,labels=group_int)

    df.Fare=cat

    return df



def create_alone(df):

    # Alone = 1, Not Alone = 0

    famsize=df.SibSp+df.Parch

    alone=~(famsize >0)

    df['Alone']=alone.astype(int)

    return df



def create_child(df):

    child=(df.Age<18).astype(int)

    df['Child']=child

    return df



def sex_map(df):

    sex_mapping = {"male": 0, "female": 1}

    df['Sex']=df['Sex'].map(sex_mapping).astype(int)

    return df



def drop_features(df):

    df=df.drop(['Fare', 'Cabin', 'Name','Ticket','Age'], axis=1)

    return df



def mod_data(df):

    df = noAgetoMed(df)

    df = embarked(df)

    df = fare_bands(df)

    df = create_alone(df)

    df = create_child(df)

    df = sex_map(df)

    df = drop_features(df)

    return df



train=mod_data(train)

test=mod_data(test)
train.head()
x_train=train.loc[:,'Pclass':'Child']

x_test=test.loc[:,'Pclass':'Child']

y_train=train.Survived
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



# Initialize the algorithm class

logr = LogisticRegression()



# Train the algorithm using all the training data

logr.fit(x_train,y_train)

# Make predictions using the test set

LogRpred = logr.predict(x_test)

#LogR_acc=round(accuracy_score(test['Survived'],LogRpred),2) 

#print('LogR_acc = ',LogR_acc)

#LogR_acc
from sklearn.feature_selection import SelectKBest, f_classif



# Perform feature selection

selector = SelectKBest(f_classif, k=5)

selector.fit(x_train, y_train)



# Get the raw p-values for each feature, and transform them from p-values into scores

scores = -np.log10(selector.pvalues_)



# Plot the scores  

plt.bar(range(x_train.shape[1]), scores)

plt.xticks(range(x_train.shape[1]), x_train.columns, rotation='vertical')

plt.show()
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": LogRpred

    })



#submission.to_csv("kaggleTitanic_6-13-17.csv", index=False)