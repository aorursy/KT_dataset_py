import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import xgboost

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



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

    

# Add age density column

for df in data_dfs:

    counts = df.Age.value_counts()

    df['AgeDensity'] = df.Age.map(counts)
deck_dict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'T':6, 'G':7, 'Unknown':8}



for df in data_dfs:

    df['Deck'] = df.Cabin.str.extract('([A-Z]){1}', expand=False)

    df.Deck = df.Deck.map(deck_dict)

    df = df.drop(['Cabin'], axis=1)



train_df['Deck'].value_counts()
for df in data_dfs:

    ticket_density = df.Ticket.value_counts()

    df['TicketDensity'] = df.Ticket.map(ticket_density)

    df = df.drop(['Ticket'], axis=1)

# Family Size Feature



for df in data_dfs:

    df['FamilySize'] = df.Parch + df.SibSp + 1

    

# Age*Class feature



for df in data_dfs:

    df['AgexPClass']=df.Age * df.Pclass



# Fare per person feature

for df in data_dfs:

    df['FarePerPerson'] = df.Fare / df.FamilySize

# Columns =

train_df.columns
# Set up Training and testing data



#Ignoring: 'FarePerPerson' 'SibSp', 'Parch',  'AgexPClass'



feature_columns = ['Pclass', 'Sex', 'Age',  'Fare',   'AgeDensity', 'Deck', 'TicketDensity', 'FamilySize', 'Title','Embarked']

pred_column = 'Survived'



X_data = train_df[feature_columns].copy()

Y_data = train_df[pred_column].copy()

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=1234)



# Set up and train classifier

# model = RandomForestClassifier(n_estimators=1000)





model = xgboost.XGBClassifier()

model.fit(X_train, y_train)



model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=10, verbose=True)



model.score(X_test, y_test)
# Look at Feature Importances

# pd.DataFrame({"Feature": feature_columns, "Importance": rfc.feature_importances_})

xgboost.plot_importance(model)
X_comp = test_df[feature_columns].copy()

Y_comp = model.predict(X_comp)



submission = pd.DataFrame({

    "PassengerId": test_df["PassengerId"],

    "Survived": Y_comp

})



submission.to_csv('submission.csv', index=False)



submission.head(10)