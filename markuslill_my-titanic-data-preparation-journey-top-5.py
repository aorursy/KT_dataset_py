import pandas as pd



df_train = pd.read_csv("/kaggle/input/titanic/train.csv")



len(df_train[df_train.Survived == 0])/len(df_train)
df_train.info()
df_train.describe()
df_train.head()
sm = pd.plotting.scatter_matrix(df_train, figsize=(15,8))
df_train[df_train.SibSp == df_train.SibSp.max()]
fare_hist = df_train.Fare.hist(bins=20)
df_train[df_train.Fare == df_train.Fare.max()]
df_train.groupby('Pclass').Survived.mean()
age_hist = df_train.Age.hist(bins=20)
df_train.Sex.value_counts()
df_train.groupby('Sex').Survived.mean()
pd.crosstab(index=df_train['Sex'], columns=df_train['Pclass'], values=df_train.Survived, aggfunc='mean')
df_train = pd.get_dummies(df_train, columns = ["Sex"])

df_train.shape
df_train['Pclass Sex_female'] = df_train.Pclass * df_train.Sex_female
df_train['Title'] = [i[i.find(',') + 1: i.find('.')].strip() for i in df_train.Name]

age_bytitle_box = df_train.boxplot(column=['Age'], by=['Title'], figsize=(15,8))
df_train['Age'] = df_train['Age'].fillna(df_train.groupby('Title')['Age'].transform('mean'))

df_train.Age.isna().sum()
from sklearn import preprocessing



df_train.Age = preprocessing.scale(df_train.Age)

age_hist = df_train.Age.hist(bins=20)
pd.concat([df_train.groupby('Title').Survived.mean().sort_values(ascending=True),

           df_train.groupby('Title').Title.count()], 

          axis=1)
df_train['IsHonor'] = 0

honorable_titles = ["Capt", "Don", "Jonkheer", "Rev"]

df_train.loc[df_train['Title'].isin(honorable_titles), "IsHonor"] = 1

df_train[df_train.IsHonor == 1].IsHonor.sum()
df_train.loc[df_train.Cabin.isna(), 'Cabin'] = 0

df_train.loc[df_train.Cabin != 0, 'Cabin'] = 1

        

df_train['Embarked'] = df_train['Embarked'].fillna('S')

df_train = pd.get_dummies(df_train, columns = ["Embarked"])



df_train.info()
df_train['FamilySize'] = df_train.SibSp + df_train.Parch
child_age_margin = 11

mother_age_margin = 20



def is_row_mother(row):

    if (row.Sex_female == 1 and row.Age > mother_age_margin):

        return True

    return False



df_train['LastName'] = [i[0:i.find(',')] for i in df_train.Name]

df_train['MotherChildRelation'] = 0

for index, row in df_train.iterrows():

    if (row.Age < child_age_margin):

        for index2, row2 in df_train.iterrows():

            if (row.LastName == row2.LastName):

                if (is_row_mother(row2)):

                    if (row2.Survived == 1):

                        df_train.loc[index, 'MotherChildRelation'] = 1

                    else:

                        df_train.loc[index, 'MotherChildRelation'] = -1

                    if (row.Survived == 1):

                        df_train.loc[index2, 'MotherChildRelation'] = 1

                    else:

                        df_train.loc[index2, 'MotherChildRelation'] = -1
df_train["MotherChildRelation"].value_counts()
df_train["Ticket"].value_counts()
df_train[df_train["Ticket"] == "CA. 2343"].Survived
df_train['TicketProbability'] = 0

for index, row in df_train.iterrows():

    ticket = row.Ticket

    survived_array = []

    for index2, row2 in df_train.iterrows():

        if (index != index2 and row.Ticket == row2.Ticket):

            survived_array.append(row2.Survived)

    if (len(survived_array) > 0):

        df_train.loc[index, 'TicketProbability'] = sum(survived_array) / len(survived_array)

    else:

        df_train.loc[index, 'TicketProbability'] = df_train.Survived.mean()