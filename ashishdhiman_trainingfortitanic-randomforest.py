#imports

import numpy as np 

import pandas as pd

import sklearn

import math

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/train.csv')

df.head(5)
#Study differnent params 

df['Age'].describe()
def substrings_in_string(big_string, substrings):

    if big_string != None:

        for substring in substrings:

            if big_string.find(substring) != -1:

                return substring

    else: return np.nan

    

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title



title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer']



cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']



def normalize_age(row):

    age = row['Age']

    row['Age_Unknown']=0

    row['Age_1-5']=0

    row['Age_6-15']=0

    row['Age_16-25']=0

    row['Age_26-40']=0

    row['Age_>40']=0

    

    if math.isnan(age):

        row['Age_Unknown'] = 1

    elif 0.0 < age <= 5.0 :

        row['Age_1-5']=1

    elif 5.0 < age <= 15.0:

        row['Age_6-15']=1

    elif 15.0 < age <= 25.0:

        row['Age_16-25']=1

    elif 25.0 < age <= 40.0:

        row['Age_26-40']=1

    else:

        row['Age_>40']=1

    return row
 #Clean up the data

def cleanup_data(raw_df):

    raw_df['Cabin'] = raw_df['Cabin'].fillna('Unknown')

    raw_df['Title']= raw_df['Name'].map(lambda x: substrings_in_string(x, title_list))

    raw_df['Title']=raw_df.apply(replace_titles, axis=1)

    raw_df['Deck']=raw_df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

    

    raw_df['Family_Size']=raw_df['SibSp']+raw_df['Parch']

    

#     raw_df['Age_Class']=raw_df['Age']*raw_df['Pclass']

    

    raw_df['Embarked']=raw_df['Embarked'].fillna('Unknown')

    

    raw_df['Fare'] = raw_df['Fare'].fillna(0)

    raw_df['Fare_Per_Person']=raw_df['Fare']/(df['Family_Size']+1)

    raw_df = raw_df.apply(normalize_age,axis=1)



#     raw_df = raw_df.dropna(axis=0,how='any')

# "PassengerId",

    clean_df = raw_df.drop(["PassengerId","Name","SibSp","Parch","Ticket","Fare","Cabin","Title","Age"],axis=1)

    

    return clean_df



def split_labels(clean_df):

    y = clean_df["Survived"]

    X = clean_df.drop("Survived",axis=1)

    return X,y
 #Encode the labels

def create_label_encoders(X):

    le_sex = LabelEncoder()

    le_sex.fit(X["Sex"])

    

    le_emb = LabelEncoder()

    le_emb.fit(X["Embarked"])

    

    le_dek = LabelEncoder()

    le_dek.fit(X["Deck"])

    

    return {"sex":le_sex,"embarked":le_emb,"deck":le_dek}



#Transform the encoded data

def transorm_with_encoders(X,encoders):

    X["Sex"] = encoders['sex'].transform(X["Sex"])

    X["Embarked"] = encoders['embarked'].transform(X["Embarked"])

    X["Deck"] = encoders['deck'].transform(X["Deck"])

    return X
 #Method to create Random Forest Classifier

def create_RF_model(input_data):

    #clean input data

    clean_data = cleanup_data(input_data)

    X,y = split_labels(clean_data)

    

    #create encoders for data

    encoders_dict = create_label_encoders(X)

    X = transorm_with_encoders(X,encoders_dict)

    

    #split into train/dev set

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.20, random_state=5)

    

    #train the RF classifier

    classifier = RandomForestClassifier(max_depth=3,n_estimators=101)

    classifier.fit(X, y)

    

    #test accuracy with train/dev set

    acc_train = classifier.score(X_train, y_train)

    print('Training Set Accuracy : '+str(acc_train*100))

    acc_dev = classifier.score(X_dev, y_dev)

    print('Dev Set Accuracy : '+str(acc_dev*100))

    return classifier,encoders_dict



#Method to predict the results

def predict(classifier,encoders,data):

    X = cleanup_data(data)

    X = transorm_with_encoders(X,encoders)

    return classifier.predict(X)
model,encoders = create_RF_model(df)
test = pd.read_csv('../input/test.csv')

ids = test["PassengerId"]

prediction = predict(model,encoders,test)
result = pd.DataFrame.from_dict({"PassengerId":ids,"Survived":prediction.astype(np.int64)})

result.to_csv('survival_submission_v1.csv',header=["PassengerId","Survived"],index=False)