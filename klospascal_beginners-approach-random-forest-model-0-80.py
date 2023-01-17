import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

#Merging the Data Sets
data_df = df.append(df_test, sort=False) 
data_df

data_df.head()
print(data_df.columns[df.isnull().any()])

print(data_df[data_df["Age"].isnull() == True].count())
print(data_df[data_df["Cabin"].isnull() == True].count())
print(data_df[data_df["Embarked"].isnull() == True].count())
# Fill NaN
median_value=data_df['Age'].median()
data_df['Age']=data_df['Age'].fillna(median_value)
data_df['Age']=data_df["Age"].astype("int")

median_value_fare=data_df['Fare'].median()
data_df['Fare']=data_df['Fare'].fillna(median_value_fare)

data_df['Embarked']=data_df['Embarked'].fillna("S")
#Encoding Sex & Embarked
data_df["Sex"] = data_df["Sex"].replace("male", 0).replace("female", 1)
data_df["Embarked"]=data_df["Embarked"].replace("C", 0).replace("S", 1).replace("Q",2)

data_df.head()
#Pairplot
sns.pairplot(data_df, hue="Survived")
data_df.groupby(['Pclass'])['Survived'].mean()
data_df.groupby(['Sex'])['Survived'].mean()
data_df.groupby(['Embarked'])['Survived'].mean()
data_df.groupby(['SibSp'])['Survived'].mean()
data_df.groupby(['Parch'])['Survived'].mean()
#Make Groups
print(pd.qcut(data_df["Fare"], 4).unique())
print(pd.qcut(data_df["Age"], 4).unique())
data_df.loc[data_df["Fare"] <= 7.896, 'Fare_Grouped'] = 0
data_df.loc[(data_df["Fare"] > 7.896) & (data_df["Fare"] <= 14.454),  'Fare_Grouped'] = 1
data_df.loc[(data_df["Fare"] > 14.454) & (data_df["Fare"] <= 31.275),  'Fare_Grouped'] = 2
data_df.loc[(data_df["Fare"] > 31.275),  'Fare_Grouped'] = 3
data_df['Fare_Grouped']=data_df["Fare_Grouped"].astype("int")

data_df.loc[data_df["Age"] <= 16, 'Age_Grouped'] = 0
data_df.loc[(data_df["Age"] > 16) & (data_df["Age"] <= 28),  'Age_Grouped'] = 1
data_df.loc[(data_df["Age"] > 28) & (data_df["Age"] <= 35),  'Age_Grouped'] = 2
data_df.loc[(data_df["Age"] > 35) & (data_df["Age"] <= 64),  'Age_Grouped'] = 3
data_df.loc[(data_df["Age"] > 64),  'Age_Grouped'] = 4
data_df["Age_Grouped"] = data_df.Age.astype(int)

data_df.head()
data_df.groupby(['Fare'])['Survived'].mean()
data_df.groupby(['Age'])['Survived'].mean()
#Family Variable
data_df["Family"] = data_df["SibSp"] + data_df["Parch"] +1 
df = data_df.drop(["SibSp", "Parch"], axis=1)
data_df.head()
#Deck
data_df["Deck"] = data_df["Cabin"].str.slice(0,1)
data_df["Deck"] = data_df["Deck"].fillna("N")

#Room
data_df["Room"] = data_df["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
data_df["Room"] = data_df["Room"].fillna(data_df["Room"].mean())
data_df["Room"] = data_df.Room.astype(int)

#Create Dummies
data_df = pd.get_dummies(data_df, columns = ["Deck"])

data_df.head()
#Ticket Length

data_df['Ticket_Len'] = data_df['Ticket'].apply(lambda x: len(x))
data_df.groupby(['Ticket_Len'])['Survived'].mean()
#Ticket First Letter

data_df['Ticket_Lett'] = data_df['Ticket'].apply(lambda x: str(x)[0])
data_df.groupby(['Ticket_Lett'])['Survived'].mean()
#Ticket Letter Encoding
replacement = {
    'A': 0,
    'P': 1,
    'S': 0,
    '1': 1,
    '2': 0,
    'C': 0,
    '7': 0,
    'W': 0,
    '4': 0,
    'F': 1,
    'L': 0,
    '9': 1,
    '6': 0,
    '5': 0,
    '8': 0,   
    '3': 0,
}

data_df['Ticket_Lett'] = data_df['Ticket_Lett'].apply(lambda x: replacement.get(x))

data_df.head()
#Name Len
data_df["Name_Len"] = data_df['Name'].apply(lambda x: len(x))
data_df.groupby(['Name_Len'])['Survived'].mean()
#Extract Title from Name
for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.')


#Coding of Titles
replacement = {
    'Dr': 0,
    'Master': 1,
    'Miss': 2,
    'Ms': 2,
    'Mlle': 2,
    'Rev': 3,
    'Mme': 4,
    'Mrs': 4,
    'Lady': 4,
    'the Countess': 5,
    'Don': 5,
    'Dona': 5,
    'Jonkheer': 5,
    'Sir': 5,
    'Capt': 5,
    'Col': 5,
    'Major': 5,
    'Mr': 5,   
}

data_df['Title'] = data_df['Title'].apply(lambda x: replacement.get(x))

#Fill Missing Values
median_value=data_df['Title'].median()
data_df['Title']=data_df['Title'].fillna(median_value)

data_df["Title"] = data_df.Title.astype(int)

data_df.head()
#Family Group (from S.Xu's kernel)

data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

for _, grp_df in data_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))

data_df.head()
#Drop unused columns
data_df = data_df.drop(["Ticket", "Last_Name", "Name", "Cabin", "Age", "Fare"], axis=1)


# Split in TRAIN_DF and TEST_DF:
df = data_df[:891]
df_test= data_df[891:]
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

forrest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 40, 5)],
)

x = df.drop(["Survived", "PassengerId"], axis=1)
y = df["Survived"]

forrest = RandomForestClassifier()
forest_cv = GridSearchCV(estimator=forrest, param_grid=forrest_params, cv=5) 
forest_cv.fit(x, y)
print(forest_cv.best_score_)
print(forest_cv.best_estimator_)
testdat = df_test.drop(["Survived", "PassengerId"], axis=1)

predict = forest_cv.predict(testdat)

print(predict.astype(int))

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": predict.astype(int)
})

submission.to_csv("submision.csv", index=False)