import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
train_original = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
train = train_original.copy()
train.head()
for i in train[["Pclass","Sex","SibSp","Parch","Embarked"]]:

    print(train[i].value_counts())

    print("\n\n\n")
train.hist(figsize=(20,20))
train.describe(include="all")
(train.groupby('Embarked').mean())['Survived']#persentage of people how survived 

#based on embarked

#c has the most survival ratio
train.head()
train.groupby('Sex').sum()['Survived'].plot(kind="bar",stacked=True)#more survived female than males
train.groupby('Embarked').sum()['Survived'].plot(kind="bar",stacked=True)
sns.violinplot(x="Age",y="Sex",hue="Survived",data=train,split=True)
def comb(x):

    if(pd.isna(x)):

        return 'N';

    return x[0]





train['Cabin'] = train['Cabin'].apply(comb)

le_cabin = LabelEncoder()

hot_cabin = OneHotEncoder()

le_cabin.fit(train['Cabin'])

hot_cabin.fit(le_cabin.transform(train['Cabin']).reshape(-1,1))



age_scaler = StandardScaler()

age_scaler.fit(np.array(train['Age']).reshape(-1,1))



fare_scaler = StandardScaler()

fare_scaler.fit(np.array(train['Fare']).reshape(-1,1))



def drop_columns(df,col): # col = ["Name",'PassengerId',"Ticket"]

    df.drop(col,axis=1,inplace=True)







def cabin_2(df):

    df['Cabin'] = df['Cabin'].apply(comb)



def one_hot_encoder_cabin(df):

    i='Cabin'

    con = le_cabin.transform(df[i])

    hot_encoder = hot_cabin.transform(con.reshape(-1,1)).toarray()

    df_hot_encoded = pd.DataFrame(hot_encoder,columns = ['Cabin ('+j+')' for j in (le_cabin.classes_)])

    df = pd.concat([df,df_hot_encoded],axis=1)

    df.drop(i,axis=1,inplace=True)

    return df





def fill_nan_with_mostcommon(df):

    for i in df.columns:

        df[i].fillna(df[i].mode()[0],inplace=True)





def one_hot_encoder_embarked(df):

    m = {"S":1,"Q":2,"C":3}

    df['Embarked'] = df['Embarked'].map(m)

    hot = OneHotEncoder()

    hot_encoder = hot.fit_transform(np.array(df['Embarked']).reshape(-1,1) ).toarray()

    df_hot_encoded = pd.DataFrame(hot_encoder,columns = ['Embarked(S)','Embarked(Q)','Embarked(C)'] )

    df = pd.concat([df,df_hot_encoded],axis=1)

    df.drop("Embarked",axis=1,inplace=True)

    return df





    

def map_sex(df):

    m = {"male":1,"female":2}

    df['Sex'] = df['Sex'].map(m)

    

def one_hot_encoder_pclass(df):

    hot = OneHotEncoder()

    hot_encoder = hot.fit_transform(np.array(df['Pclass']).reshape(-1,1) ).toarray()

    df_hot_encoded = pd.DataFrame(hot_encoder,columns = ['Pclass(1)','Pclass(2)','Pclass(3)'] )

    df = pd.concat([df,df_hot_encoded],axis=1)

    df.drop("Pclass",axis=1,inplace=True)

    return df





def Age_Fare_Scaling(df):

    df['Age'] = age_scaler.transform(np.array(df['Age']).reshape(-1,1))

    df['Fare'] = fare_scaler.transform(np.array(df['Fare']).reshape(-1,1))



def change(df):

    col_to_drop = ["Name",'PassengerId',"Ticket"] 

    scaling = ['Age','Fare']

    

    drop_columns(df,col_to_drop)

    cabin_2(df)

    fill_nan_with_mostcommon(df)

    map_sex(df)

    df = one_hot_encoder_embarked(df)

    df = one_hot_encoder_cabin(df)

    df = one_hot_encoder_pclass(df)

    Age_Fare_Scaling(df)

    

    return df

    
train = change(train)
train.head()
testing = test.copy()

testing = change(testing)

testing.head()
test.head()
from sklearn.model_selection import train_test_split



def split_feature_labels(train,labels_name):

    x_train =  train.drop(labels_name, axis=1)

    y_train =  train[labels_name]

    return x_train,y_train



train_set,validation_set = train_test_split(train,test_size=0.2,random_state=42)



x_train,y_train = split_feature_labels(train_set,"Survived")

x_validation,y_validation = split_feature_labels(validation_set,"Survived")
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



param_grid = [

    {'n_estimators': [3, 10, 30,70,100,130], 'max_features': [0.5,1,2, 4, 6, 8,10]},

  ]



forest_reg = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='accuracy', return_train_score=True)

grid_search.fit(x_train, y_train)
print("Training Accuracy :",grid_search.score(x_train,y_train))

print("Validation Accuracy :",grid_search.score(x_validation,y_validation))
forest_reg = RandomForestClassifier(random_state=42,max_features=5,n_estimators=100)

forest_reg.fit(x_train, y_train)

print("Training Accuracy :",forest_reg.score(x_train,y_train))

print("Validation Accuracy :",forest_reg.score(x_validation,y_validation))

output = forest_reg.predict(testing)
out = pd.read_csv('../input/gender_submission.csv')
out.Survived = output
out.head()
out.to_csv('submission.csv', index=False)