import pandas as pd

import numpy as np



%matplotlib inline

from scipy.stats import norm

import scipy.stats as st

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

from sklearn.preprocessing.imputation import Imputer

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



#from sklearn.ensemble import RandomForestClassifier

#from sklearn.ensemble import GradientBoostingRegressor

#from sklearn.tree import DecisionTreeRegressor



titanic_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

warnings.filterwarnings(action='ignore', category=FutureWarning)



titanic_data.info()

titanic_data.describe()

titanic_data.columns

titanic_data.shape

# Columns with missing values

total = titanic_data.isnull().sum().sort_values(ascending=False)

percent = (titanic_data.isnull().sum()/titanic_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head()
# on Test_data

total = test_data.isnull().sum().sort_values(ascending=False)

percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head()



#test_data[test_data.isnull().any(axis=1)]
# Correlation Matrix (heatmap)

corrmat = pd.get_dummies(titanic_data, columns=['Sex']).corr()

f, ax = plt.subplots(figsize=(16, 9))

sns.heatmap(corrmat, vmax=.8, square=True, annot=True, cmap="RdBu_r") # altri valori: BuGn_r, BrBG
# SNS Graph: Men and women count

sns.countplot(titanic_data['Sex'])
data = titanic_data

# data = pd.get_dummies(titanic_data, columns=['Sex'])

# Age of people

plt.title("Age (Survived in orange)")

data['Age'].plot.hist(edgecolor='black', linewidth=0.5)

data[data.Survived == 1]['Age'].plot.hist(edgecolor='black', linewidth=0.5)

plt.xlabel("Age")

plt.ylabel("Persons")
# Men's age

data[(data.Sex == 'male')]['Age'].plot.hist(edgecolor='black', linewidth=0.5)

data[(data.Survived == 1) & (data.Sex == 'male')]['Age'].plot.hist(edgecolor='black', linewidth=0.5)
# Women's age

data[(data.Sex == 'female')]['Age'].plot.hist(edgecolor='black', linewidth=0.5)

data[(data.Survived == 1) & (data.Sex == 'female')]['Age'].plot.hist(edgecolor='black', linewidth=0.5)
# Correlation between Fare and Survived, Pclass as Hue

sns.barplot(x="Survived", y="Fare", hue="Pclass", data=data)
# Scatter plot Fare/Age, Survived as Hue, Pclass as filter

g = sns.FacetGrid(data, hue="Survived", col="Pclass", margin_titles=True,

                  palette={1:"blue", 0:"red"})

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
# Scatterplot Fare/Age, Survived ad Hue, Sex as filter

g = sns.FacetGrid(data, hue="Survived", col="Sex", margin_titles=True,

                palette={1:"blue", 0:"red"},hue_kws=dict(marker=["^", "v"]))

g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Survival by Gender , Age and Fare');
# Correlation with SibSp

sns.barplot(x="SibSp", y="Survived", data=data)
# Correlation with Parch

sns.barplot(x="Parch", y="Survived", data=data)
# with Embarked

sns.barplot(x="Embarked", y="Survived", data=data)
# Correlation with FamilySize, a new feature made from the sum of Parch and SibSp

data["FamilySize"] = data['Parch'] + data['SibSp'] + 1

sns.barplot(x="FamilySize", y="Survived", data=data)



# Another two new features: IsAlone and BigFamily (if FamilySize>=5 then BigFamily=1)

# No changes to the dataset, just testing

data["IsAlone"] = 0

data["BigFamily"] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

data.loc[data['FamilySize'] >= 5, 'BigFamily'] = 1
sns.barplot(x="IsAlone", y="Survived", data=data)
sns.barplot(x="BigFamily", y="Survived", data=data)
# Another new feature named Couple (not used)

data['Couple'] = 0

data.loc[data['FamilySize'] == 2, 'Couple'] = 1



sns.barplot(x="Couple", y="Survived", data=data)
# New feature: SharedTicket (not used, IsAlone is a better predictor, though it may need improving)

# Yet no changes in the original dataset, only testing

ticket = pd.DataFrame(data['Ticket'].sort_values()) # sort

ticket['SharedTicket'] = 0



ticket['SharedTicket'] = (ticket['Ticket'].eq(ticket['Ticket'].shift(1)) | ticket['Ticket'].eq(ticket['Ticket'].shift(-1)))

ticket['SharedTicket'] = ticket['SharedTicket'].astype(int)



data['SharedTicket'] = 0

data['SharedTicket'] = ticket['SharedTicket'] # joins by index, works correctly



data.loc[data['SharedTicket']==1, 'Ticket'].sort_values()



# List of travelers with a shared ticket (SharedTicket), first 20 rows

data.loc[data['SharedTicket']==1, ['Ticket', 'Name']].sort_values("Ticket").head(20)
# Correlation with Survived

sns.barplot(x="SharedTicket", y="Survived", data=data)
# TITLES AND AGE - Titles are captured and filled in a new column

def make_ages_from_titles(titanic_data):

    names = titanic_data.Name.str.split(",", expand=True)

    names = names[1].str.split(n=1, expand=True)

    titanic_data["Title"] = names[0]

    

    # Plotting

    # titanic_data['Title'].value_counts().plot.bar(rot=0, edgecolor='black', figsize=(15, 6), linewidth=0.5)

    # plt.xlabel("Titles")

    

    # Average age by title - Values to fill on Age NaNs

    #titanic_data[(titanic_data.Title == "Mr.")]['Age'].mean()

    #titanic_data[(titanic_data.Title == "Miss.")]['Age'].mean()

    #titanic_data[(titanic_data.Title == "Mrs.")]['Age'].mean()

    #titanic_data[(titanic_data.Title == "Dr.")]['Age'].mean()

    #titanic_data[(titanic_data.Title == "Master.")]['Age'].mean()



    df = titanic_data



    mean = round(titanic_data[(titanic_data.Title == "Mr.")]['Age'].mean(), 0)

    df.loc[df['Title'] == 'Mr.', 'Age'] = df.loc[df['Title'] == 'Mr.', 'Age'].fillna(mean)



    mean = round(titanic_data[(titanic_data.Title == "Miss.")]['Age'].mean(), 0)

    df.loc[df['Title'] == 'Miss.', 'Age'] = df.loc[df['Title'] == 'Miss.', 'Age'].fillna(mean)

    df.loc[df['Title'] == 'Ms.', 'Age'] = df.loc[df['Title'] == 'Ms.', 'Age'].fillna(mean)



    mean = round(titanic_data[(titanic_data.Title == "Mrs.")]['Age'].mean(), 0)

    df.loc[df['Title'] == 'Mrs.', 'Age'] = df.loc[df['Title'] == 'Mrs.', 'Age'].fillna(mean)



    mean = round(titanic_data[(titanic_data.Title == "Dr.")]['Age'].mean(), 0)

    df.loc[df['Title'] == 'Dr.', 'Age'] = df.loc[df['Title'] == 'Dr.', 'Age'].fillna(mean)



    mean = round(titanic_data[(titanic_data.Title == "Master.")]['Age'].mean(), 0)

    df.loc[df['Title'] == 'Master.', 'Age'] = df.loc[df['Title'] == 'Master.', 'Age'].fillna(mean)

    

    titanic_data['Title'] = titanic_data['Title'].replace(['Lady', 'Countess', 'Capt.', 'Col.',

                                'Don.', 'Dr.', 'Major', 'Rev.', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    

    title_mapping = {"Mr.": 1, "Miss.": 2, "Ms.": 2, "Mrs.": 3, "Master.": 4, "Rare": 5}

    titanic_data['Title'] = titanic_data['Title'].map(title_mapping)

    titanic_data['Title'] = titanic_data['Title'].fillna(0)



    titanic_data = df

    return titanic_data



# Run it in both data files

titanic_data = make_ages_from_titles(titanic_data)

test_data = make_ages_from_titles(test_data)

# CABIN and EMBARKED - Filling NaN values with Unknown (for now)



def cabin_embarked(titanic_data):

    df = titanic_data

    df[['Cabin']] = df[['Cabin']].fillna(value="Unknown")

    #df[df["Embarked"].isnull()] # to show the two Null rows

    df[['Embarked']] = df[['Embarked']].fillna(value="Unknown")

    titanic_data = df

    return titanic_data



titanic_data = cabin_embarked(titanic_data)

test_data = cabin_embarked(test_data)



# SHAREDTICKET - New feature - to move below



def shared_ticket(titanic_data):

    

    ticket = pd.DataFrame(titanic_data['Ticket'].sort_values()) # sort

    ticket['SharedTicket'] = 0



    ticket['SharedTicket'] = (ticket['Ticket'].eq(ticket['Ticket'].shift(1)) | ticket['Ticket'].eq(ticket['Ticket'].shift(-1)))

    ticket['SharedTicket'] = ticket['SharedTicket'].astype(int)



    titanic_data['SharedTicket'] = 0

    titanic_data['SharedTicket'] = ticket['SharedTicket']

    return titanic_data



titanic_data = shared_ticket(titanic_data)

test_data = shared_ticket(test_data)
titanic_data.sample(5)

test_data.sample(5)
def cabin_letters (titanic_data):

    

    df = titanic_data

    # Cabin numbers is equal to the number of str.split elements (number of the cabins)

    cabins = df.loc[df['Cabin'] != 'Unknown', 'Cabin'].str.split()

    df['Cabin_numbers'] = cabins.transform(lambda x: len(x)).astype(int)

    df['Cabin_numbers'] = df['Cabin_numbers'].fillna(value=1) # at least one cabin?



    # We need only the first letter

    titanic_data['Cabin_letter'] = titanic_data['Cabin'].str[0]



    titanic_data = df

    return titanic_data



titanic_data = cabin_letters(titanic_data)

test_data = cabin_letters(test_data)
titanic_data.head()

test_data.head()
# New Heatmap

corrmat = pd.get_dummies(titanic_data, columns=['Sex','Cabin_numbers','Cabin_letter',]).corr()

f, ax = plt.subplots(figsize=(35, 16))

sns.heatmap(corrmat, vmax=.8, square=True, annot=True, cmap="RdBu_r") # other values to try: BuGn_r, BrBG
def fill_unknown_cabins (titanic_data):



    df = titanic_data

    

    # V20: Tried to left them as Unknown/U but prediction did not improve

    # V21: Fixing the cabins letter improves prediction (0.79 from 0.77)

    

    df.loc[(df['Pclass']==1) & (df['Cabin_letter']=='U'), 'Cabin_letter'] = 'B'

    df.loc[(df['Pclass']==2) & (df['Cabin_letter']=='U'), 'Cabin_letter'] = 'D'

    df.loc[(df['Pclass']==3) & (df['Cabin_letter']=='U'), 'Cabin_letter'] = 'G'



    # Checking if everything ok

    df.Cabin_letter.value_counts()



    # Cabin with letter T

    df.loc[df['Cabin_letter']=='T']



    # It is a first class cabin so i set it as B

    df.loc[df['Cabin_letter']=='T', ['Cabin_letter','Cabin']] = 'B'



    titanic_data = df

    return titanic_data



titanic_data = fill_unknown_cabins(titanic_data)

test_data = fill_unknown_cabins(test_data)



def family_size_alone (titanic_data):

    titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1

    titanic_data['IsAlone'] = 0

    titanic_data['BigFamily'] = 0

    titanic_data['Couple'] = 0

    titanic_data.loc[titanic_data['FamilySize'] == 2, 'Couple'] = 1

    titanic_data.loc[titanic_data['FamilySize'] == 1, 'IsAlone'] = 1

    titanic_data.loc[titanic_data['FamilySize'] >= 5, 'BigFamily'] = 1

    

    # Dropping Parch, SibSp e FamilySize for IsAlone, decide later about BigFamily

    titanic_data = titanic_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

    return titanic_data



titanic_data = family_size_alone(titanic_data)

test_data = family_size_alone(test_data)
titanic_data.head()

test_data.head()
# Analysis only

df = titanic_data

df['AgeBand'] = pd.cut(df['Age'], 5)

df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
# Qcut

df['FareBand'] = pd.qcut(df['Fare'], 4)

df[['FareBand', 'Survived']].groupby(['FareBand'], 

                                               as_index=False).mean().sort_values(by='FareBand', ascending=True)
def Age_Fare_Band(titanic_data):

    

    # Group ages and fair in bars (bins), an example of feature scaling

    

    titanic_data['AgeBand'] = pd.cut(titanic_data['Age'], 5)

    titanic_data.loc[ titanic_data['Age'] <= 16, 'Age'] = 0

    titanic_data.loc[(titanic_data['Age'] > 16) & (titanic_data['Age'] <= 32), 'Age'] = 1

    titanic_data.loc[(titanic_data['Age'] > 32) & (titanic_data['Age'] <= 48), 'Age'] = 2

    titanic_data.loc[(titanic_data['Age'] > 48) & (titanic_data['Age'] <= 64), 'Age'] = 3

    titanic_data.loc[ titanic_data['Age'] > 64, 'Age'] = 4

    

    titanic_data['Fare'].fillna(titanic_data['Fare'].dropna().median(), inplace=True)

    titanic_data['FareBand'] = pd.qcut(titanic_data['Fare'], 4)

    titanic_data.loc[ titanic_data['Fare'] <= 7.91, 'Fare'] = 0

    titanic_data.loc[(titanic_data['Fare'] > 7.91) & (titanic_data['Fare'] <= 14.454), 'Fare'] = 1

    titanic_data.loc[(titanic_data['Fare'] > 14.454) & (titanic_data['Fare'] <= 31), 'Fare']   = 2

    titanic_data.loc[ titanic_data['Fare'] > 31, 'Fare'] = 3

    titanic_data['Fare'] = titanic_data['Fare'].astype(int)



    titanic_data = titanic_data.drop(['AgeBand'], axis=1)

    titanic_data = titanic_data.drop(['FareBand'], axis=1)

    return titanic_data



titanic_data = Age_Fare_Band(titanic_data)

test_data = Age_Fare_Band(test_data)
titanic_data.head()

test_data.head()
# Correction where two rows of Embarked are Unknown - V23

titanic_data.loc[titanic_data['Embarked'] == 'Unknown']

titanic_data['Embarked'] = titanic_data['Embarked'].replace('Unknown', 'U')



#titanic_data.iloc[61]

#titanic_data.iloc[829]
from sklearn import preprocessing



def mapping(titanic_data):

    # sex_map = {"female": 1, "male": 0} # si passa sex_map dentro map(), un altro modo di fare il mapping

    titanic_data['Sex'] = titanic_data['Sex'].map( {"female": 1, "male": 0} ).astype(int)

    

    # V23: deactivated because the prediction get worse (!), don't know why. LabelEconder below works better

    #titanic_data['Embarked'] = titanic_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, 'U': 3} ).astype(int)

    #titanic_data['Cabin_letter'] = titanic_data['Cabin_letter'].map( {'A': 6, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'U': 0} ).astype(int)

    return titanic_data



titanic_data = mapping(titanic_data)

test_data = mapping(test_data)



# V23: reactivated because it improves the model, need check why the manual mapping worsen it

def encode_features(titanic_data, test_data):

    features = ['Cabin_letter','Embarked']

    df_combined = pd.concat([titanic_data, test_data])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        titanic_data[feature] = le.transform(titanic_data[feature])

        test_data[feature] = le.transform(test_data[feature])

    return titanic_data, test_data

    

titanic_data, test_data  = encode_features(titanic_data, test_data)
# Heatmap

corrmat = titanic_data.corr()

f, ax = plt.subplots(figsize=(16, 9))

sns.heatmap(corrmat, vmax=.8, square=True, annot=True, cmap="RdBu_r")
titanic_data.head()

test_data.head()



# List of features, will be used on test_data too



features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'Cabin_letter', 'IsAlone']



X = titanic_data [features] #V27: features (0.8086)

y = titanic_data['Survived']
# Split into validation and training data

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.50, random_state=1)



# Parameters

n_est = 1000

learn = 0.10

max_dp = 3



from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score



# Modello: XGBClassifier

xgb_model = XGBClassifier (n_estimators=n_est, learning_rate=learn, max_depth=max_dp)

xgb_model.fit (train_X, train_y)

xgb_predictions = xgb_model.predict (test_X)



print("Accuracy: {0}".format(accuracy_score(test_y, xgb_predictions)))
# XGBRegressor Pipeline with full data and Cross-Validation #V27 = 0.8060

xgb_final = make_pipeline(XGBClassifier(n_estimators=n_est, 

                                        learning_rate=learn, 

                                        xgbclassifier__early_stopping_rounds=5, 

                                        xgbclassifier__eval_set=[(X, y)]))



# Cross-Validation

scores = cross_val_score(xgb_final, X, y, scoring='accuracy', cv=3)

print('XGB Pipeline Cross-Validation Accuracy: %2f' %scores.mean())

print(scores)



xgb_final.fit(X, y);
ids = test_data['PassengerId']

final_data = test_data [features]



predictions = xgb_final.predict(final_data);



output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)

output.head()