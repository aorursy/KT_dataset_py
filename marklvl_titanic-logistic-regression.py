import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")


titanic_df.info()
print('-'*50)
test_df.info()
# drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_df.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)
test_df.drop(['Name','Ticket'], axis=1, inplace=True)
# Checking for na values in datasets columns
print('Titanic dataset:')
print(titanic_df.isnull().sum())
print('-'*50)
print('Test dataset:')
print(test_df.isnull().sum())
print('-'*50)
#################
# Titanic dataset
#################
# Getting the proportion of Embarked within titanic dataframe
print("\nProportion of Embarked values in dataset: \n{}".format(titanic_df.Embarked.value_counts() / len(titanic_df)))

# Filling missed Embarked records with S
titanic_df.Embarked.fillna('S', inplace=True)

# Filling missing ages with random values with normal distribution of current ages
titanic_df.loc[titanic_df.Age.isnull(),'Age'] = np.random.randint(titanic_df.Age.mean() - titanic_df.Age.std(),
                                                                  titanic_df.Age.mean() + titanic_df.Age.std(),
                                                                  size=titanic_df.Age.isnull().sum())

# Checking for cabin columns
print("\nProportion of NAs in cabin column:{}".format(titanic_df.Cabin.isnull().sum() / len(titanic_df))) # 78% NAs, we can drop it
titanic_df.drop('Cabin', axis=1, inplace=True)

##############
# Test dataset
##############
test_df.fillna(test_df.Fare.median(), inplace=True)

# Filling missing ages with random values with normal distribution of current ages
test_df.loc[test_df.Age.isnull(),'Age'] = np.random.randint(test_df.Age.mean() - test_df.Age.std(),
                                                            test_df.Age.mean() + test_df.Age.std(),
                                                            size=test_df.Age.isnull().sum())

test_df.drop('Cabin', axis=1, inplace=True)
# Checking for na values in datasets columns
print('Titanic dataset:')
print(titanic_df.isnull().sum())
print('-'*50)
print('Test dataset:')
print(test_df.isnull().sum())
print('-'*50)
titanic_df.info()
titanic_df.SibSp.value_counts()
###########################
# Setting proper data types
###########################
titanic_df['Survived'] = titanic_df.Survived.astype('category')
titanic_df['Pclass'] = titanic_df.Pclass.astype('category')
titanic_df['Sex'] = titanic_df.Sex.astype('category')
titanic_df['Embarked'] = titanic_df.Embarked.astype('category')

test_df['Pclass'] = test_df.Pclass.astype('category')
test_df['Sex'] = test_df.Sex.astype('category')
test_df['Embarked'] = test_df.Embarked.astype('category')
# Constructing FamilySize and traveling Alone columns and removing Parch and SibSp
titanic_df['FamilySize'] =  titanic_df["Parch"] + titanic_df["SibSp"]
titanic_df.loc[titanic_df['FamilySize'] > 0, 'Alone'] = 0
titanic_df.loc[titanic_df['FamilySize'] == 0, 'Alone'] = 1
titanic_df['Alone'] = titanic_df.Alone.astype('category')
titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)

# Doing the same for test dataset
test_df['FamilySize'] =  test_df["Parch"] + test_df["SibSp"]
test_df.loc[test_df['FamilySize'] > 0, 'Alone'] = 0
test_df.loc[test_df['FamilySize'] == 0, 'Alone'] = 1
test_df['Alone'] = test_df.Alone.astype('category')
test_df = test_df.drop(['SibSp','Parch'], axis=1)
# Defining categorical variables encoder method
def fit_transform_ohe(df,col_name):
    """This function performs one hot encoding for the specified
column.
    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        col_name: the column to be one hot encoded
    Returns:
        tuple: label_encoder, one_hot_encoder, transformed column as pandas Series
    """
    # label encode the column
    le = preprocessing.LabelEncoder()
    le_labels = le.fit_transform(df[col_name])
    df[col_name+'_label'] = le_labels
    # one hot encoding
    ohe = preprocessing.OneHotEncoder()
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    return le,ohe,features_df

# given label encoder and one hot encoder objects, 
# encode attribute to ohe
def transform_ohe(df,le,ohe,col_name):
    """This function performs one hot encoding for the specified
        column using the specified encoder objects.

    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        le(Label Encoder): the label encoder object used to fit label encoding
        ohe(One Hot Encoder): the onen hot encoder object used to fit one hot encoding
        col_name: the column to be one hot encoded

    Returns:
        tuple: transformed column as pandas Series

    """
    # label encode
    col_labels = le.transform(df[col_name])
    df[col_name+'_label'] = col_labels
    
    # ohe 
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return features_df
X = titanic_df.iloc[:,1:]
y = titanic_df.iloc[:,0]

X_test = test_df.iloc[:,1:]
y_test = test_df.iloc[:,0]
# Encoding all the categorical features
cat_attr_list = ['Pclass','Sex',
                 'Embarked','Alone']
# though we have transformed all categoricals into their one-hot encodings, note that ordinal
# attributes such as hour, weekday, and so on do not require such encoding.
numeric_feature_cols = ['Age','Fare','FamilySize']
subset_cat_features =  ['Pclass','Sex','Embarked','Alone']

###############
# Train dataset
###############
encoded_attr_list = []
for col in cat_attr_list:
    return_obj = fit_transform_ohe(X,col)
    encoded_attr_list.append({'label_enc':return_obj[0],
                              'ohe_enc':return_obj[1],
                              'feature_df':return_obj[2],
                              'col_name':col})


feature_df_list  = [X[numeric_feature_cols]]
feature_df_list.extend([enc['feature_df'] \
                        for enc in encoded_attr_list \
                        if enc['col_name'] in subset_cat_features])

train_df_new = pd.concat(feature_df_list, axis=1)
print("Train dataset shape::{}".format(train_df_new.shape))
print(train_df_new.head())

##############
# Test dataset
##############
test_encoded_attr_list = []
for enc in encoded_attr_list:
    col_name = enc['col_name']
    le = enc['label_enc']
    ohe = enc['ohe_enc']
    test_encoded_attr_list.append({'feature_df':transform_ohe(X_test,
                                                              le,ohe,
                                                              col_name),
                                   'col_name':col_name})
    
    
test_feature_df_list = [X_test[numeric_feature_cols]]
test_feature_df_list.extend([enc['feature_df'] \
                             for enc in test_encoded_attr_list \
                             if enc['col_name'] in subset_cat_features])

test_df_new = pd.concat(test_feature_df_list, axis=1) 
print("Test dataset shape::{}".format(test_df_new.shape))
print(test_df_new.head())
# Constructing train dataset
X = train_df_new
#y= y.Survived

# Constructing test dataset
X_test = test_df_new
#y_test = y_test.Survived
print(X.shape,y.shape)
logreg = LogisticRegression()

logreg.fit(X,y)
print("R-Squared on train dataset={}".format(logreg.score(X,y)))

Y_pred = logreg.predict(X_test)

logreg.fit(X_test,y_test)   
print("R-Squared on test dataset={}".format(logreg.score(X_test,y_test)))
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)
