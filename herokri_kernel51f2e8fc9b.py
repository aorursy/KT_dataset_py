import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from scipy import signal

from collections import Counter

import re



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder



import kaggle

import copy
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



def get_family_name(name):

    #' ([A-Za-z]+)\.'

    name_search = re.search('([A-Za-z]+)\,', name)

    # If the name exists, extract and return it.

    if name_search:

        return name_search.group(1)

    return ""



def clean_data(df):

    df["Cabin"] = df["Cabin"].fillna("Not known") 

    df["HasCabin"] = 1

    df.loc[df["Cabin"] == "Not known", "HasCabin"] = 0

    

    df.loc[df["Sex"] == "male", "Sex"] = 0

    df.loc[df["Sex"] == "female", "Sex"] = 1

    

    df["Embarked"] = df["Embarked"].fillna("S")

    df.loc[df["Embarked"] == "S", "Embarked"] = 2

    df.loc[df["Embarked"] == "C", "Embarked"] = 0

    df.loc[df["Embarked"] == "Q", "Embarked"] = 1

    

    df["FamilySize"] = df["Parch"] + df["SibSp"] + 1 

    

    df["Alone"] = 0

    df.loc[df["FamilySize"] == 1, "Alone"] = 1

    

    df["BigFamily"] = 0

    df.loc[df["FamilySize"] >= 5, "BigFamily"] = 1

    

    #df["Fare"] = df["Fare"].fillna(df["Fare"].dropna().median())

    df.loc[(df["Pclass"] == 1) & (df['Fare'].isna()) , "Fare"] = 33.76

    df.loc[(df["Pclass"] == 2) & (df['Fare'].isna()) , "Fare"] = 13.0

    df.loc[(df["Pclass"] == 3) & (df['Fare'].isna()) , "Fare"] = 7.75

    

    df["NormFare"] = df["Fare"] / df["FamilySize"]

    

    #mean_p1 33.760400000000004

    #mean_p2 13.0

    #mean_p3 7.75

    

    df["Cheap"] = 0

    df.loc[(df["NormFare"] < 33.76) & (df["Pclass"] == 1), "Cheap"] = 1

    df.loc[(df["NormFare"] < 13.0) & (df["Pclass"] == 2), "Cheap"] = 1

    df.loc[(df["NormFare"] < 7.75) & (df["Pclass"] == 3), "Cheap"] = 1

    



    df['Title'] = df['Name'].apply(get_title)

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')



    

    df['Special'] = 1

    df.loc[(df["Title"] == "Miss") | (df["Title"] == "Mr") | (df["Title"] == "Mrs") , "Special"] = 0



    df['Elder'] = 0

    df.loc[(df["Title"] == "Master")

           | (df["Title"] == "Lady") | (df["Title"] == "Sir") | (df["Title"] == "Countess"), "Elder"] = 1

    

    df['TitleAge'] = 0

    for title in pd.unique(df['Title']):

        mean = df[df.Title == title]['Age'].dropna().median()

        df.loc[df['Title'] == title, 'TitleAge'] = mean

    

    df.loc[np.isnan(df["Age"]), 'Age'] = df["TitleAge"]

    

    df['FamilyName'] = df['Name'].apply(get_family_name)

    le = LabelEncoder()

    df['FamilyLabel'] = le.fit_transform(df['FamilyName'])

    

    df["Woman"] = 0

    df.loc[(df["Sex"] == 1), "Woman"] = 1

    

    df["Man"] = 0

    df.loc[(df["Sex"] == 0), "Man"] = 1

    

    df["Baby"] = 0

    df.loc[df["Age"] < 5, "Baby"] = 1

    df.loc[df["Baby"] == 1, "Woman"] = 0

    df.loc[df["Baby"] == 1, "Man"] = 0

    

    df["Kid"] = 0

    df.loc[(df["Age"] < 12) & (df["Baby"] == 0), "Kid"] = 1

    

    df["Teen"] = 0

    df.loc[(df["Age"] < 20) & (df["Kid"] == 0) & (df["Baby"] == 0), "Teen"] = 0

    

    df["OldMan"] = 0

    df.loc[(df["Age"] >= 60) & (df["Sex"] == 0), "OldMan"] = 1

    df.loc[df["OldMan"] == 1, "Man"] = 0

    

    df["OldWoman"] = 0

    df.loc[(df["Age"] >= 60) & (df["Sex"] == 1), "OldWoman"] = 1

    df.loc[df["OldWoman"] == 1, "Woman"] = 0

    

    df["CombinedAgeSex"] = 0

    df.loc[(df["Baby"] == 1), "CombinedAgeSex"] = 1

    df.loc[(df["Woman"] == 1), "CombinedAgeSex"] = 2

    df.loc[(df["Kid"] == 1), "CombinedAgeSex"] = 3

    df.loc[(df["Man"] == 1), "CombinedAgeSex"] = 4

    df.loc[(df["OldMan"] == 1), "CombinedAgeSex"] = 5

    

    return df

  



class kaggle_competition:

    

    def __init__(self, competition_name='titanic'):

        self.competition_name = competition_name

        self.submission_fn = 'submission.csv'

        if competition_name == '':

            train = pd.read_csv("../input/train.csv")

            test = pd.read_csv("../input/test.csv")

        else:

            kaggle.api.authenticate()

            kaggle.api.competition_download_cli(competition_name)

            train = pd.read_csv("train.csv")

            test = pd.read_csv("test.csv")



        # preview the data

        train.head()

        

        self.train_org = train

        self.test_org = test

        self.train = train

        self.test = test

        

    

    def clean(self):

        self.train = clean_data(self.train_org)

        print("Done with cleaning training data")

        if self.train.isnull().sum().sum() != 0:

            raise Exception('Still {} uncleaned values'.format(self.train.isnull().sum().sum()))

        self.test = clean_data(self.test_org)

        print("Done with cleaning test data")

        if self.test.isnull().sum().sum() != 0:

            raise Exception('Still {} uncleaned values'.format(self.test.isnull().sum().sum()))

        

    def selection(self):

        drop_set = ['PassengerId', 'Cabin', 'Ticket', 'Name', 'Parch', 'SibSp', 'Embarked', 'FamilySize',

                    'FamilyName', 'FamilyLabel', 'Man', 'Woman', 'Kid',

                    'Fare', 'Pclass', 'Title', 'TitleAge', 'OldWoman', 'Teen', 'CombinedAgeSex']

        

        self.train_ready = self.train.drop(drop_set, axis=1)

        self.test_ready = self.test.drop(drop_set, axis=1)

        

        self.target_ready = self.train.Survived

        self.train_ready = self.train_ready.drop(["Survived"], axis=1)

        

        

    def training(self):

        clf = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0, max_depth = 6)



        # Applying k-Fold Cross Validation

        accuracies = cross_val_score(estimator = clf, X=self.train_ready , y=self.target_ready , cv = 10)

        

        self.model = clf.fit(X=self.train_ready , y=self.target_ready)

        

        pd.Series(clf.feature_importances_, index=self.train_ready.columns).nlargest(12).plot(kind='barh')  

        

        print("Random Forest:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std())

        

        

    def predict_and_write(self):

        prediction = self.model.predict(X=self.test_ready)        

        pd.concat([self.test['PassengerId'],

                   pd.DataFrame(prediction, columns=['Survived'])], axis=1).to_csv(self.submission_fn, index=False)

        submission = pd.DataFrame({"PassengerId": self.test_ready["PassengerId"], "Survived": prediction})

        

        

    def submit(self):

        if not self.competition_name == "":

            kaggle.api.authenticate()

            kaggle.api.competition_submit_cli(self.submission_fn, "test", self.competition_name)

        

        

    def run(self):

        self.clean()

        self.selection()

        

        self.training()

        self.predict_and_write()

        



myc =  kaggle_competition(competition_name='')

myc.clean()
df = myc.train_org

df.head(5)
df = myc.train

print(pd.crosstab(df['Title'], df['Sex']))
df = myc.train_org

Titles = []

Survived = []

for title in pd.unique(df['Title']):

    Titles.append(title)

    Survived.append(df[df.Title == title]['Survived'].mean())

    

df_titles = pd.DataFrame([Survived, Titles])

df_titles.index = ['Survived', 'Titles']

df_titles.T.plot(kind='bar', x='Titles', y='Survived', title="Survived by Title")
df = myc.train

# Explore SibSp feature vs Survived

g = sns.catplot(x="Elder", y="Survived", data=df, kind="bar", height=6 , palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
df = myc.train_org

Titles = []

Age = []

for title in pd.unique(df['Title']):

    Titles.append(title)

    Age.append(df[df.Title == title]['Age'].mean())

    

df_titles = pd.DataFrame([Age, Titles])

df_titles.index = ['Age', 'Titles']

df_titles.T.plot(kind='bar', x='Titles', y='Age', title="Age by Title")
df = myc.train_org

# Explore SibSp feature vs Survived

g = sns.catplot(x="Embarked", y="Survived", data=df, kind="bar", height=6 , palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Pclass vs Survived by Sex

g = sns.catplot(x="Embarked", y="Survived", hue="Pclass", data=df,

                   height=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Pclass vs Survived by Sex

g = sns.catplot(x="Embarked", y="Sex", hue="Pclass", data=df,

                   height=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Female ratio")
df = myc.train_org

Survived = df[df.Survived == 1]['Sex'].value_counts()

Died = df[df.Survived == 0]['Sex'].value_counts()

df_sex = pd.DataFrame([Survived , Died])

df_sex.index = ['Survived', 'Died']

print(df_sex)

df_sex.plot(kind='bar',stacked=True, figsize=(5,3), title="Survived/Died by Sex")
df2 =  copy.deepcopy(df)

fac = 3

df2["Age"] = round(df2["Age"] / fac ) * fac

sns.lineplot(x="Age", y="Survived", hue="Sex", data=df2, ci='sd')

del df2
plt.figure(figsize=(15,8))

ax = sns.kdeplot(df["Fare"][df.Pclass == 1], color="darkturquoise", shade=True)

sns.kdeplot(df["Fare"][df.Pclass == 2], color="lightcoral", shade=True)

sns.kdeplot(df["Fare"][df.Pclass == 3], color="green", shade=True)

plt.legend(['1st Class', '2nd Class', '3rd Class'])

plt.title('Density Plot of Fare against classes')

ax.set(xlabel='Fare')

plt.xlim(-5, 100)

plt.show()
df["NormFare"] = df["Fare"] / df["FamilySize"]

plt.figure(figsize=(15,8))

ax = sns.kdeplot(df["NormFare"][df.Pclass == 1], color="darkturquoise", shade=True)

sns.kdeplot(df["NormFare"][df.Pclass == 2], color="lightcoral", shade=True)

sns.kdeplot(df["NormFare"][df.Pclass == 3], color="green", shade=True)

plt.legend(['1st Class', '2nd Class', '3rd Class'])

plt.title('Density Plot of Fare against classes')

ax.set(xlabel='Fare')

plt.xlim(-5, 100)

plt.show()
df = myc.train_org

# Explore Parch feature vs Survived

g  = sns.catplot(x="FamilySize", y="Survived", data=df, kind="bar", height=4, palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
df = myc.train

# Explore Parch feature vs Survived

g  = sns.catplot(x="HasCabin", y="Survived", data=df, kind="bar", height=4, palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
new =  kaggle_competition(competition_name='')

new.run()