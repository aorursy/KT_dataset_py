import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import keras as kr
import tensorflow as tf 



%matplotlib inline

data_train = pd.read_csv("../input/titanic/train.csv")
data_test = pd.read_csv("../input/titanic/test.csv")
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

data_train = transform_features(data_train)
data_test = transform_features(data_test)
from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_test = encode_features(data_train, data_test)

from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']
num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)
#TensorFlow Data preprossesing


feature_columns = [tf.feature_column.numeric_column(key="Pclass"),
tf.feature_column.numeric_column(key="Sex"),
tf.feature_column.numeric_column(key="Age"),
tf.feature_column.numeric_column(key="SibSp"),
tf.feature_column.numeric_column(key="Parch"),
tf.feature_column.numeric_column(key="Fare"),
tf.feature_column.numeric_column(key="Cabin"),
tf.feature_column.numeric_column(key="Lname"),
tf.feature_column.numeric_column(key="NamePrefix")]


classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10,30, 10], n_classes=2)

train_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=60, shuffle=True, num_epochs=None)

classifier.train(input_fn=train_input_fn, steps=20000)

ids = data_test['PassengerId']











test_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, shuffle=True, num_epochs=1)
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

#Fixing the weird data structures 
X_pred = data_test.drop('PassengerId', axis=1)


pred_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_pred, shuffle=False, num_epochs=1)

pred = classifier.predict(input_fn=test_input_fn)









predictions = (classifier.predict(input_fn=pred_input_fn))

#output

predicted_classes = [p["classes"] for p in predictions]

#
print(predicted_classes)



test = pd.DataFrame(predicted_classes)
print(test)

testt = (test)
print(testt)

#output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predicted_classes })
#output.to_csv('Titanic-Predictions-CII-V1.csv', index=False)
#print(output)


