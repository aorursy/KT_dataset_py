import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import logistic as lr

from sklearn.cluster import k_means



# Load the input data we need

train_dataset = pd.read_csv('../input/train.csv')

test_dataset = pd.read_csv('../input/test.csv')

full_dataset = [train_dataset, test_dataset]

# See what the test dataset looks like

train_dataset.head()
train_dataset[['Sex', 'Survived']].groupby('Sex').mean()
train_dataset[['Pclass', 'Survived']].groupby('Pclass').mean()
train_dataset[['Sex', 'Pclass', 'Survived']].groupby(['Sex', 'Pclass']).mean()
train_dataset[['Age', 'Survived']].groupby(['Age']).mean()
sum(pd.isnull(train_dataset['Age']) == True)
train_dataset['Age'] = train_dataset.Age.fillna(-0.001)

test_dataset['Age'] = test_dataset.Age.fillna(-0.001)

train_dataset.Age.describe()

test_dataset.Age.describe()
train_dataset.Fare.describe()
test_dataset['Fare'] = test_dataset['Fare'].fillna(test_dataset['Fare'].mean())

test_dataset.Fare.describe()
test_dataset[['Age','Sex']].groupby('Sex').mean()
features = ['Fare', 'Age', 'Sex', 'Pclass']



train_data_encoded = train_dataset

test_data_encoded = test_dataset



combined_df = pd.concat([train_dataset[features], test_dataset[features]])

combined_df.head()
def encode_features(train_df, test_df):

    features = ['Fare', 'Age', 'Sex', 'Pclass']    

    combined_df = pd.concat([train_df[features], test_df[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(combined_df[feature])

        train_df[feature] = le.transform(train_df[feature])

        test_df[feature] = le.transform(test_df[feature])

    

    return train_df[features], test_df[features]



train_data_encoded, test_data_encoded = encode_features(train_dataset, test_dataset)

    

train_data_encoded.head()
y_train = train_dataset['Survived'].astype(int)

X_train = train_data_encoded

X_test  = test_data_encoded

logreg = lr.LogisticRegression()

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



logreg.score(X_train, y_train)
# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = pd.DataFrame(train_data_encoded.columns)

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



# preview

coeff_df
output = pd.DataFrame({

        "PassengerId": test_dataset["PassengerId"],

        "Survived": y_pred

    })

output.to_csv('c:\titanic.csv', index=False, )

import os

os.chdir('c:/')