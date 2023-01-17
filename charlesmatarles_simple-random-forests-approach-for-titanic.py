import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.ensemble import RandomForestClassifier



# Load Data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# Extract Title from Names:

data_dfs = [train_df, test_df]



for df in data_dfs:

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



for df in data_dfs:

    # Replace rare titles with 'Rare'

    df.Title = df.Title.replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    # Replace alt forms of titles:

    df.Title = df.Title.replace('Mlle', 'Miss')

    df.Title = df.Title.replace('Ms', 'Miss')

    df.Title = df.Title.replace('Mme', 'Mrs')

    # Fill in missing data with a placeholder.

    df.Title = df.Title.fillna('Missing')



print("What titles survived?")

print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())



# Convert titles to integer category

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5, "Missing": 0}

for df in data_dfs:

    df['Title'] = df['Title'].map(title_mapping)



# Clean Up data - drop Name column

for df in data_dfs:

    df = df.drop(['Name'], axis=1)
# Convert Sex to an integer category

for df in data_dfs:

    df.Sex = df.Sex.map( {'female': 1, 'male': 0} ).astype(int)



print("What gender survived?")

train_df[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean()
# Convert Embarked to an integer Category and fill in missing values

port_mode = train_df.Embarked.dropna().mode()[0]

print('Mode:', port_mode)



for df in data_dfs:

    # Replace NaNs with the mode:

    df.Embarked = df.Embarked.fillna(port_mode)

    # Convert to integer

    df.Embarked = df.Embarked.map({'S':0, 'C':1, 'Q':2})



print("Which port's passsenger's survived?")

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Missing values for fare:

median_fare = train_df.Fare.median()

print("Median Fare:", median_fare)

for df in data_dfs:

    df.Fare = df.Fare.fillna(median_fare)
# Fill missing ages with the median from the dataset.

for df in data_dfs:

    df_med_age = df.Age.median()

    df.Age = df.Age.fillna(df_med_age)
# Drop Ticket and Cabin

for df in data_dfs:

    df = df.drop(['Ticket', 'Cabin'], axis=1)
# Set up Training and testing data



feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']

pred_column = 'Survived'



X_train = train_df[feature_columns].copy()

Y_train = train_df[pred_column].copy()



# Set up and train classifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, Y_train)



rfc.score(X_train, Y_train)
# Look at Feature Importances

pd.DataFrame({"Feature": feature_columns, "Importance": rfc.feature_importances_})
X_test = test_df[feature_columns].copy()

Y_pred = rfc.predict(X_test)



submission = pd.DataFrame({

    "PassengerId": test_df["PassengerId"],

    "Survived": Y_pred

})



submission.to_csv('submission.csv', index=False)



submission.head(10)