import pandas as pd
import numpy as np

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
train_df.columns
train_df.head()
train_df.info()
print('_'*40)
test_df.info()
train_df.describe()
test_df.describe()
fig, axes = plt.subplots(1, 2)

train_df.Age.hist(bins = 30, ax=axes[0])
test_df.Age.hist(bins = 30, ax=axes[1])
fig, axes = plt.subplots(1, 2)

train_df.Fare.hist(bins = 30, ax=axes[0])
test_df.Fare.hist(bins = 30, ax=axes[1])
fig, axes = plt.subplots(1, 2)

train_df.Parch.hist(bins = 20, ax=axes[0])
test_df.Parch.hist(bins = 20, ax=axes[1])
fig, axes = plt.subplots(1, 2)

train_df.SibSp.hist(bins = 20, ax=axes[0])
test_df.SibSp.hist(bins = 20, ax=axes[1])
print("Percentage of survival in the train set: {}%".format(round(sum(train_df.Survived)/train_df.Survived.count(), 2)))
print("In train: ")
print(train_df.Sex.value_counts())
print('_'*40)
print("In test: ")
print(test_df.Sex.value_counts())

fig, axes = plt.subplots(1, 2)

train_df.Sex.value_counts().plot(kind = "bar", ax=axes[0])
test_df.Sex.value_counts().plot(kind = "bar", ax=axes[1])
print("In train: ")
print(train_df.Pclass.value_counts())
print('_'*40)
print("In test: ")
print(test_df.Pclass.value_counts())

fig, axes = plt.subplots(1, 2)

train_df.Pclass.value_counts().plot(kind = "bar", ax=axes[0])
test_df.Pclass.value_counts().plot(kind = "bar", ax=axes[1])
print("In train: ")
print(train_df.Embarked.value_counts())
print('_'*40)
print("In test: ")
print(test_df.Embarked.value_counts())

fig, axes = plt.subplots(1, 2)

train_df.Embarked.value_counts().plot(kind = "bar", ax=axes[0])
test_df.Embarked.value_counts().plot(kind = "bar", ax=axes[1])
train_df.corr()
test_df.corr()
train_df[['Pclass', 'Survived']].groupby(['Pclass'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], 
                                      as_index=True).mean().sort_values(by='Survived', 
                                                                         ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], 
                                        as_index=True).mean().sort_values(by='Survived', 
                                                                           ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], 
                                        as_index=True).mean().sort_values(by='Survived', 
                                                                           ascending=False)
train_df[["Embarked", "Survived"]].groupby(['Embarked'], 
                                        as_index=True).mean().sort_values(by='Survived', 
                                                                           ascending=False)
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=40)
for dataset in combine:
    df = dataset.groupby(['Sex', 'Pclass']).size().unstack(0)
    df['fem_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)
train_df[["Sex", "Pclass", "Survived"]].groupby(['Sex', 'Pclass'], 
                                        as_index=True).mean()
for dataset in combine:
    df = dataset.groupby(['Sex', 'Parch']).size().unstack(0).fillna(0)
    df['male_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['fem_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)
train_df[["Sex", "Parch", "Survived"]].groupby(['Sex', 'Parch'], 
                                        as_index=True).mean()
for dataset in combine:
    df = dataset.groupby(['Sex', 'SibSp']).size().unstack(0).fillna(0)
    df['male_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['fem_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)
train_df[["Sex", "SibSp", "Survived"]].groupby(['Sex', 'SibSp'], 
                                        as_index=True).mean()
grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid.map(plt.hist, 'Age', alpha=.5, bins=30)
grid.add_legend()
grid = sns.FacetGrid(train_df, col='Pclass', row='Sex', hue='Survived')
grid.map(plt.hist, 'Age', alpha=.5, bins=30)
grid.add_legend()
for dataset in combine:
    df = dataset.groupby(['Sex', 'Embarked']).size().unstack(0)
    df['male_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['fem_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)
train_df[["Embarked", "Sex", "Survived"]].groupby(['Sex', 'Embarked'], 
                                        as_index=True).mean()
for dataset in combine:
    df = dataset.groupby(['Pclass', 'Embarked']).size().unstack(0)
    df['first_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    df['second_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    df['third_perc'] = (df[df.columns[2]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    print(df)
    print("_"*40)
train_df[["Embarked", "Pclass", "Survived"]].groupby(['Embarked', 'Pclass'], 
                                        as_index=True).mean()
for dataset in combine:
    fil1 = (dataset.Cabin.isnull())
    fil2 = (dataset.Cabin.notnull())
    dataset.loc[fil1, 'Cabin'] = 0
    dataset.loc[fil2, 'Cabin'] = 1
    dataset.Cabin = pd.to_numeric(dataset['Cabin'])

print(train_df.Cabin.value_counts())
print("_"*40)
print(test_df.Cabin.value_counts())
train_df[['Cabin', 'Survived']].groupby(['Cabin'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
for dataset in combine:
    df = dataset.groupby(['Sex', 'Cabin']).size().unstack(0)
    df['fem_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['male_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)
train_df[["Cabin", "Sex", "Survived"]].groupby(['Sex', 'Cabin'], 
                                        as_index=True).mean()
for dataset in combine:
    df = dataset.groupby(['Cabin', 'Pclass']).size().unstack(0)
    df['miss_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)
train_df[["Cabin", "Pclass", "Survived"]].groupby(['Pclass', 'Cabin'], 
                                        as_index=True).mean()
train_df[train_df.Embarked.isnull()]
test_df[test_df.Fare.isnull()]
fil = ((train_df.Pclass == 1) & (train_df.SibSp == 0) & (train_df.Parch == 0) 
       & (train_df.Sex == 'female'))
mis = train_df[fil].Embarked.mode()
print(mis)
fil = train_df.Embarked.isnull()
train_df.loc[fil, 'Embarked'] = 'C' #I new it was C from a previous run
print("_"*40)
print(train_df.Embarked.value_counts(dropna = False))
fil = ((test_df.Pclass == 3) & (test_df.SibSp == 0) & (test_df.Parch == 0) 
       & (test_df.Cabin == 0) & (test_df.Sex == 'male') & (test_df.Embarked == 'S'))
mis = round(test_df[fil].Fare.median(), 4)
print(mis)
fil = test_df.Fare.isnull()
test_df.loc[fil, 'Fare'] = mis
print("_"*40)
print(test_df.Fare.isnull().value_counts(dropna = False))
for dataset in combine:
    dataset['MisAge'] = 0
    fil = (dataset.Age.isnull())
    dataset.loc[fil, 'MisAge'] = 1

print(train_df.MisAge.value_counts())
print("_"*40)
print(test_df.MisAge.value_counts())
print("_"*40)
print("_"*40)

train_df[['MisAge', 'Survived']].groupby(['MisAge'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
for dataset in combine:
    df = dataset.groupby(['MisAge', 'Sex']).size().unstack(0)
    df['miss_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['nomiss_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)
train_df[["Sex", "MisAge", "Survived"]].groupby(['Sex', 'MisAge'], 
                                        as_index=True).mean()
for dataset in combine:
    df = dataset.groupby(['MisAge', 'Pclass']).size().unstack(0)
    df['miss_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['nomiss_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)
train_df[["Pclass", "MisAge", "Survived"]].groupby(['Pclass', 'MisAge'], 
                                        as_index=True).mean()
train_df[["Cabin", "MisAge", "Survived"]].groupby(['Cabin', 'MisAge'], 
                                        as_index=True).mean()
fil = (train_df.Age.isnull())
print("By class:")
print(train_df[fil].Pclass.value_counts())
print("_"*40)
print(train_df[train_df.MisAge == 0].Pclass.value_counts())
print("_"*40)
print("_"*40)
print("By sex:")
print(train_df[fil].Sex.value_counts())
print("_"*40)
print(train_df[train_df.MisAge == 0].Sex.value_counts())
print("_"*40)
print("_"*40)
print("By parents and children:")
print(train_df[fil].Parch.value_counts())
print("_"*40)
print(train_df[train_df.MisAge == 0].Parch.value_counts())
print("_"*40)
print("_"*40)
print("By spouse and siblings:")
print(train_df[fil].SibSp.value_counts())
print("_"*40)
print(train_df[train_df.MisAge == 0].SibSp.value_counts())
fil = (test_df.Age.isnull())
print("By class:")
print(test_df[fil].Pclass.value_counts())
print("_"*40)
print(test_df[test_df.MisAge == 0].Pclass.value_counts())
print("_"*40)
print("_"*40)
print("By sex:")
print(test_df[fil].Sex.value_counts())
print("_"*40)
print(test_df[test_df.MisAge == 0].Sex.value_counts())
print("_"*40)
print("_"*40)
print("By parents and children:")
print(test_df[fil].Parch.value_counts())
print("_"*40)
print(test_df[test_df.MisAge == 0].Parch.value_counts())
print("_"*40)
print("_"*40)
print("By spouse and siblings:")
print(test_df[fil].SibSp.value_counts())
print("_"*40)
print(test_df[test_df.MisAge == 0].SibSp.value_counts())
# Extract the title from the name feature
for df in combine:
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(train_df['Title'], train_df['Sex'])
pd.crosstab(test_df['Title'], test_df['Sex'])
# handling the rare classes with class
for df in combine:
    df['Title'] = df['Title'].replace(['Mme', 'Countess','Dona'], 'Mrs')
    df['Title'] = df['Title'].replace(['Capt', 'Col','Don', 'Jonkheer', 'Rev', 
                                                 'Major', 'Sir'], 'Mr')
    df['Title'] = df['Title'].replace(['Mlle', 'Lady','Ms'], 'Miss')
    df.loc[(df.Sex == 'male') & (df.Title == 'Dr') , 'Title'] = 'Mr'
    df.loc[(df.Sex == 'female') & (df.Title == 'Dr') , 'Title'] =  'Mrs' 
    
pd.crosstab(train_df['Title'], train_df['Sex'])
pd.crosstab(test_df['Title'], test_df['Sex'])
fil = (test_df.Age.isnull())
print("By title:")
print(test_df[fil].Title.value_counts())
print("_"*40)
print(test_df[test_df.MisAge == 0].Title.value_counts())
print("_"*40)
print("_"*40)
fil = (train_df.Age.isnull())
print("By class:")
print(train_df[fil].Title.value_counts())
print("_"*40)
print(train_df[train_df.MisAge == 0].Title.value_counts())
np.random.seed(452) #reproducibility
for df in combine:
    titles = list(set(df.Title))
    classes = list(set(df.Pclass))
    for title in titles:
        for cl in classes:
            fil = (df.Title == title) & (df.Pclass == cl)
            med_age = df[fil].Age.dropna().median()
            var_age = med_age / 5
            mis_age = df[fil].MisAge.sum()
            df.loc[fil & (df.Age.isnull()), 'Age'] = np.random.randint(int(med_age - var_age - 1), 
                                                                       int(med_age + var_age), mis_age)
        
train_df.Age.describe()
test_df.Age.describe()
train_df.info()
print('_'*40)
test_df.info()
# Convert Sex to numerical
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'male':1 , 'female':2}).astype(int)

train_df.sample(5)
# Convert Embarked to numerical
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':1 , 'C':2, 'Q':3}).astype(int)

train_df.sample(5)
train_df[['Title', 'Survived']].groupby(['Title'], as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
train_df.Title.hist()
train_df.Title.value_counts()
for df in combine:
    df['Title'] = df['Title'].map({'Mr':1 , 'Mrs':2, 'Miss':3, 'Master':4}).astype(int)

train_df.sample(5)
for df in combine:
    df['IsAlone'] = 0
    fil = (df.SibSp == 0) & (df.Parch == 0)
    df.loc[fil, 'IsAlone'] = 1
    
print(train_df.IsAlone.value_counts())
print("_"*40)
print(test_df.IsAlone.value_counts())
#checking correlation with the target
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
for dataset in combine:
    df = dataset.groupby(['Sex', 'IsAlone']).size().unstack(0)
    df['fem_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['male_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)
train_df[["Sex", "IsAlone", "Survived"]].groupby(['IsAlone','Sex'], 
                                        as_index=True).mean()
for dataset in combine:
    df = dataset.groupby(['Pclass', 'IsAlone']).size().unstack(0)
    df['first_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    df['second_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    df['third_perc'] = (df[df.columns[2]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    print(df)
    print("_"*40)
train_df[["Pclass", "IsAlone", "Survived"]].groupby(['IsAlone', 'Pclass'], 
                                        as_index=True).mean()
for df in combine:
    df['IsKid'] = 0
    fil = (df.Age < 16)
    df.loc[fil, 'IsKid'] = 1
    
print(train_df.IsKid.value_counts())
print("_"*40)
print(test_df.IsKid.value_counts())
#checking correlation with the target
train_df[['IsKid', 'Survived']].groupby(['IsKid'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
g = sns.FacetGrid(train_df[train_df.Age > -1], hue='Survived')
g.map(plt.hist, 'Age', bins=30, alpha = 0.6)
g.add_legend()
bins = [0, 16, 32, 48, 81] #I just want to avoid the sparse class at 64-80

for df in combine:
    df['AgeBin'] = pd.cut(df['Age'], bins)
    #df['AgeBin'] = pd.to_numeric(df['AgeBin'])
    
print(train_df.AgeBin.value_counts())
print("_"*40)
print(test_df.AgeBin.value_counts())
#checking correlation with the target
train_df[['AgeBin', 'Survived']].groupby(['AgeBin'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
#same thing but with labels
bins = [0, 16, 32, 48, 81]
names = [0, 1, 2, 3]

for df in combine:
    df['AgeBin'] = pd.cut(df['Age'], bins, labels = names)
    df['AgeBin'] = pd.to_numeric(df['AgeBin'])
    
print(train_df.AgeBin.value_counts())
print("_"*40)
print(test_df.AgeBin.value_counts())
for df in combine:
    df['NumFam'] = df['SibSp'] + df['Parch'] + 1
    df['FarePP'] = df['Fare'] / df['NumFam']
    
fig, axes = plt.subplots(1, 2)

train_df.NumFam.hist(ax=axes[0])
test_df.NumFam.hist(ax=axes[1])
fig, axes = plt.subplots(1, 2)

train_df.FarePP.hist(ax=axes[0], bins=20)
test_df.FarePP.hist(ax=axes[1], bins=20)
for df in combine:
    df['FareCat'] = pd.qcut(df.FarePP, 4)
    
print(train_df.FareCat.value_counts())
print("_"*40)
print(test_df.FareCat.value_counts())
#same thing with labels
labels = [0, 1, 2, 3]

for df in combine:
    df['FareCat'] = pd.qcut(df.FarePP, 4, labels=labels)
    df['FareCat'] = pd.to_numeric(df['FareCat'])
    
print(train_df.FareCat.value_counts())
print("_"*40)
print(test_df.FareCat.value_counts())
train_df[['FareCat', 'Survived']].groupby(['FareCat'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
for dataset in combine:
    df = dataset.groupby(['Pclass', 'FareCat']).size().unstack(0).fillna(0)
    df['first_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    df['second_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    df['third_perc'] = (df[df.columns[2]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    print(df)
    print("_"*40)
for df in combine:
    df['FamSize'] = 0 #alone people
    df.loc[(df.NumFam > 1), 'FamSize'] = 1 #small families
    df.loc[(df.NumFam > 3), 'FamSize'] = 2 #medium families
    df.loc[(df.NumFam > 5), 'FamSize'] = 3 #big families

print(train_df.FamSize.value_counts())
print("_"*40)
print(test_df.FamSize.value_counts())
train_df[['FamSize', 'Survived']].groupby(['FamSize'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
train_df[["Pclass", "FamSize", "Survived"]].groupby(['FamSize', 'Pclass'], 
                                        as_index=True).mean()
train_df[["Sex", "FamSize", "Survived"]].groupby(['FamSize', 'Sex'], 
                                        as_index=True).mean()
for df in combine:
    df['Se_Cl'] = 0
    df.loc[((df.Sex == 1) & (df.Pclass == 1)) , 'Se_Cl'] = 1 #rich male
    df.loc[((df.Sex == 1) & (df.Pclass == 2)) , 'Se_Cl'] = 2 #avg male
    df.loc[((df.Sex == 1) & (df.Pclass == 3)) , 'Se_Cl'] = 3 #poor male
    df.loc[((df.Sex == 2) & (df.Pclass == 1)) , 'Se_Cl'] = 4 #rich female
    df.loc[((df.Sex == 2) & (df.Pclass == 2)) , 'Se_Cl'] = 5 #avg female
    df.loc[((df.Sex == 2) & (df.Pclass == 3)) , 'Se_Cl'] = 6 #poor female 
    
print(train_df.Se_Cl.value_counts())
print("_"*40)
print(test_df.Se_Cl.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['Se_Cl', 'Survived']].groupby(['Se_Cl'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
for df in combine:
    df['Cl_IA'] = 0
    df.loc[((df.IsAlone == 1) & (df.Pclass == 1)) , 'Cl_IA'] = 1 #rich alone
    df.loc[((df.IsAlone == 1) & (df.Pclass == 2)) , 'Cl_IA'] = 2 #avg alone
    df.loc[((df.IsAlone == 1) & (df.Pclass == 3)) , 'Cl_IA'] = 3 #poor alone
    df.loc[((df.IsAlone == 0) & (df.Pclass == 1)) , 'Cl_IA'] = 4 #rich with family
    df.loc[((df.IsAlone == 0) & (df.Pclass == 2)) , 'Cl_IA'] = 5 #avg with family
    df.loc[((df.IsAlone == 0) & (df.Pclass == 3)) , 'Cl_IA'] = 6 #poor with family 
    
    
print(train_df.Cl_IA.value_counts())
print("_"*40)
print(test_df.Cl_IA.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['Cl_IA', 'Survived']].groupby(['Cl_IA'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
for df in combine:
    df['Ca_Cl'] = 0
    df.loc[((df.Cabin == 0) & (df.Pclass == 1)) , 'Ca_Cl'] = 1 #rich no cabin
    df.loc[((df.Cabin == 0) & (df.Pclass == 2)) , 'Ca_Cl'] = 2 #avg no cabin
    df.loc[((df.Cabin == 0) & (df.Pclass == 3)) , 'Ca_Cl'] = 3 #poor no cabin
    df.loc[((df.Cabin == 1) & (df.Pclass == 1)) , 'Ca_Cl'] = 4 #rich with cabin
    df.loc[((df.Cabin == 1) & (df.Pclass == 2)) , 'Ca_Cl'] = 5 #avg with cabin
    df.loc[((df.Cabin == 1) & (df.Pclass == 3)) , 'Ca_Cl'] = 6 #poor with cabin
    
print(train_df.Ca_Cl.value_counts())
print("_"*40)
print(test_df.Ca_Cl.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['Ca_Cl', 'Survived']].groupby(['Ca_Cl'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
for df in combine:
    df['MA_Cl'] = 0
    df.loc[((df.MisAge == 0) & (df.Pclass == 1)) , 'MA_Cl'] = 1 #rich with age
    df.loc[((df.MisAge == 0) & (df.Pclass == 2)) , 'MA_Cl'] = 2 #avg with age
    df.loc[((df.MisAge == 0) & (df.Pclass == 3)) , 'MA_Cl'] = 3 #poor with age
    df.loc[((df.MisAge == 1) & (df.Pclass == 1)) , 'MA_Cl'] = 4 #rich without age
    df.loc[((df.MisAge == 1) & (df.Pclass == 2)) , 'MA_Cl'] = 5 #avg without age
    df.loc[((df.MisAge == 1) & (df.Pclass == 3)) , 'MA_Cl'] = 6 #poor without age
    
print(train_df.MA_Cl.value_counts())
print("_"*40)
print(test_df.MA_Cl.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['MA_Cl', 'Survived']].groupby(['MA_Cl'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
for df in combine:
    df['IK_Cl'] = 0
    df.loc[((df.IsKid == 0) & (df.Pclass == 1)) , 'IK_Cl'] = 1 #rich adult
    df.loc[((df.IsKid == 0) & (df.Pclass == 2)) , 'IK_Cl'] = 2 #avg adult
    df.loc[((df.IsKid == 0) & (df.Pclass == 3)) , 'IK_Cl'] = 3 #poor adult
    df.loc[((df.IsKid == 1) & (df.Pclass == 1)) , 'IK_Cl'] = 4 #rich kid
    df.loc[((df.IsKid == 1) & (df.Pclass == 2)) , 'IK_Cl'] = 5 #avg kid
    df.loc[((df.IsKid == 1) & (df.Pclass == 3)) , 'IK_Cl'] = 6 #poor kid
    
print(train_df.IK_Cl.value_counts())
print("_"*40)
print(test_df.IK_Cl.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['IK_Cl', 'Survived']].groupby(['IK_Cl'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
#for embarked and class I will just multiply them
for df in combine:
    df["Em_Cl"] = df["Embarked"] * df["Pclass"]

print(train_df.Em_Cl.value_counts())
print("_"*40)
print(test_df.Em_Cl.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['Em_Cl', 'Survived']].groupby(['Em_Cl'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
for df in combine:
    df['Se_Ca'] = 0
    df.loc[((df.Sex == 1) & (df.Cabin == 0)) , 'Se_Ca'] = 1 #male without cabin
    df.loc[((df.Sex == 1) & (df.Cabin == 1)) , 'Se_Ca'] = 2 #male with cabin
    df.loc[((df.Sex == 2) & (df.Cabin == 0)) , 'Se_Ca'] = 3 #female without cabin
    df.loc[((df.Sex == 2) & (df.Cabin == 1)) , 'Se_Ca'] = 4 #female with cabin
    
print(train_df.Se_Ca.value_counts())
print("_"*40)
print(test_df.Se_Ca.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['Se_Ca', 'Survived']].groupby(['Se_Ca'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
for df in combine:
    df['MA_Ca'] = 0
    df.loc[((df.MisAge == 0) & (df.Cabin == 0)) , 'MA_Ca'] = 1 #Age no Cabin
    df.loc[((df.MisAge == 0) & (df.Cabin == 1)) , 'MA_Ca'] = 2 #Age and Cabin
    df.loc[((df.MisAge == 1) & (df.Cabin == 0)) , 'MA_Ca'] = 3 #No Age no Cabin
    df.loc[((df.MisAge == 1) & (df.Cabin == 1)) , 'MA_Ca'] = 4 #No Age but Cabin
    
print(train_df.MA_Ca.value_counts())
print("_"*40)
print(test_df.MA_Ca.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['MA_Ca', 'Survived']].groupby(['MA_Ca'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
for df in combine:
    df['Se_IA'] = 0
    df.loc[((df.Sex == 1) & (df.IsAlone == 0)) , 'Se_IA'] = 1 #Male with family
    df.loc[((df.Sex == 1) & (df.IsAlone == 1)) , 'Se_IA'] = 2 #Male without family
    df.loc[((df.Sex == 2) & (df.IsAlone == 0)) , 'Se_IA'] = 3 #Female with family
    df.loc[((df.Sex == 2) & (df.IsAlone == 1)) , 'Se_IA'] = 4 #Female without family
    
print(train_df.Se_IA.value_counts())
print("_"*40)
print(test_df.Se_IA.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['Se_IA', 'Survived']].groupby(['Se_IA'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)
train_df.describe()
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")
train_df.columns
features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Title', 'AgeBin', 'MisAge', 'IsKid', 
            'FamSize', 'Se_Cl', 'Cl_IA', 'Se_Ca', 'MA_Ca', 'Se_IA']

y = train_df['Survived'].copy()
X = train_df[features].copy()
test = test_df[features].copy()

X.head()
y.head()
# classifier list
clf_list = [DecisionTreeClassifier(), 
            RandomForestClassifier(), 
            AdaBoostClassifier(), 
            GradientBoostingClassifier(), 
            XGBClassifier(),
            Perceptron(),
            LogisticRegression(), 
            SVC(), 
            LinearSVC(), 
            KNeighborsClassifier(), 
            GaussianNB(),
            SGDClassifier()
           ]
mdl = []
bias_acc = []
var_acc = []
bias_f1 = []
var_f1 = []
bias_auc = []
var_auc = []

acc_scorer = make_scorer(f1_score)

for clf in clf_list:
    model = clf.__class__.__name__
    res = cross_val_score(clf, X, y, scoring='accuracy', cv = 5)
    score = round(res.mean() * 100, 3)
    var = round(res.std(), 3)
    bias_acc.append(score)
    var_acc.append(var)
    res = cross_val_score(clf, X, y, scoring=acc_scorer, cv = 5)
    score = round(res.mean() * 100, 3)
    var = round(res.std(), 3)
    bias_f1.append(score)
    var_f1.append(var)
    res = cross_val_score(clf, X, y, scoring='roc_auc', cv = 5)
    score = round(res.mean() * 100, 3)
    var = round(res.std(), 3)
    bias_auc.append(score)
    var_auc.append(var)
    mdl.append(model)
    print(model)
    
#create a small df with the scores
robcon = pd.DataFrame({'Model': mdl, 'Bias_acc':bias_acc,'Variance_acc':var_acc, 
                       'Bias_f1':bias_f1,'Variance_f1':var_f1, 'Bias_auc':bias_auc,'Variance_auc':var_auc,
                      })
robcon = robcon[['Model','Bias_acc','Variance_acc', 'Bias_f1','Variance_f1','Bias_auc','Variance_auc' ]]
robcon
print("Best for accuracy")
print(robcon[['Model','Bias_acc']].sort_values(by= 'Bias_acc', ascending=False).head(6))
print("_"*40)
print("Best for f1")
print(robcon[['Model','Bias_f1']].sort_values(by= 'Bias_f1', ascending=False).head(6))
print("_"*40)
print("Best for roc_auc")
print(robcon[['Model','Bias_auc']].sort_values(by= 'Bias_auc', ascending=False).head(6))
print("_"*40)
print("Least variance for accuracy")
print(robcon[['Model','Variance_acc']].sort_values(by= 'Variance_acc').head(6))
print("_"*40)
print("Least variance for f1")
print(robcon[['Model','Variance_f1']].sort_values(by= 'Variance_f1').head(6))
print("_"*40)
print("Least variance for roc_auc")
print(robcon[['Model','Variance_auc']].sort_values(by= 'Variance_auc').head(6))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=895)
from sklearn.feature_selection import RFECV
X_train.columns
# feature selection
FeatSel_log = RFECV(LogisticRegression(), step = 1, scoring = 'roc_auc', cv = 10)
FeatSel_log.fit(X_train, y_train)

BestFeat_log = X_train.columns.values[FeatSel_log.get_support()]
BestFeat_log
# define the parameters grid
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
             'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
             'random_state' : [42]}

# create the grid
grid_log = GridSearchCV(LogisticRegression(), param_grid, cv = 10, scoring= 'roc_auc')

#training
%time grid_log.fit(X_train[BestFeat_log], y_train)

#let's see the best estimator
best_log = grid_log.best_estimator_
print(best_log)
print("_"*40)
#with its score
print(np.abs(grid_log.best_score_))
print("_"*40)
#accuracy on test
predictions = best_log.predict(X_test[BestFeat_log])
accuracy_score(y_true = y_test, y_pred = predictions)
# define the parameters grid with NORMAL
param_grid = {'C': np.arange(1,10),
             'tol': [0.0001, 0.001, 0.01, 0.1, 1],
             'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
             'random_state': [42]}

# create the grid
grid_SVC = GridSearchCV(SVC(), param_grid, cv = 10, scoring= 'roc_auc')

#training
%time grid_SVC.fit(X_train, y_train)

#let's see the best estimator
best_SVC = grid_SVC.best_estimator_
print(best_SVC)
print("_"*40)
#with its score
print(np.abs(grid_SVC.best_score_))
#accuracy on test
predictions = best_SVC.predict(X_test)
accuracy_score(y_true = y_test, y_pred = predictions)
# feature selection
FeatSel_ada = RFECV(AdaBoostClassifier(), step = 1, scoring = 'roc_auc', cv = 10)
FeatSel_ada.fit(X_train, y_train)

BestFeat_ada = X_train.columns.values[FeatSel_ada.get_support()]
BestFeat_ada
# define the parameters grid
param_grid = {'n_estimators': np.arange(50, 500, 50),
             'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 2],
             'algorithm': ['SAMME', 'SAMME.R'],
             'random_state': [42]}

# create the grid
grid_ada = GridSearchCV(AdaBoostClassifier(), param_grid, cv = 10, scoring= 'roc_auc')

#training
%time grid_ada.fit(X_train[BestFeat_ada], y_train)

#let's see the best estimator
best_ada = grid_ada.best_estimator_
print(best_ada)
print("_"*40)
#with its score
print(np.abs(grid_ada.best_score_))
#accuracy on test
predictions = best_ada.predict(X_test[BestFeat_ada])
accuracy_score(y_true = y_test, y_pred = predictions)
# feature selection
FeatSel_for = RFECV(RandomForestClassifier(), step = 1, scoring = 'roc_auc', cv = 10)
FeatSel_for.fit(X_train, y_train)

BestFeat_for = X_train.columns.values[FeatSel_for.get_support()]
BestFeat_for
# define the parameters grid
param_grid = {'n_estimators': np.arange(10, 100, 10),
             'max_depth': np.arange(2,20),
             'max_features' : ['auto', 'log2', None],
              'criterion' : ['gini', 'entropy'],
             'random_state' : [42]}

# create the grid
grid_forest = GridSearchCV(RandomForestClassifier(), param_grid, cv = 10, scoring= 'roc_auc')

#training
%time grid_forest.fit(X_train[BestFeat_for], y_train)

#let's see the best estimator
best_forest = grid_forest.best_estimator_
print(best_forest)
print("_"*40)
#with its score
print(np.abs(grid_forest.best_score_))
#accuracy on test
predictions = best_forest.predict(X_test[BestFeat_for])
accuracy_score(y_true = y_test, y_pred = predictions)
# feature selection
FeatSel_XGB = RFECV(XGBClassifier(), step = 1, scoring = 'roc_auc', cv = 10)
FeatSel_XGB.fit(X_train, y_train)

BestFeat_XGB = X_train.columns.values[FeatSel_XGB.get_support()]
BestFeat_XGB
# define the parameters grid
param_grid = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 2],
             'max_depth': np.arange(2,10),
             'n_estimators': np.arange(50, 500, 50),
             'random_state': [42]}

# create the grid
grid_XGB = GridSearchCV(XGBClassifier(), param_grid, cv = 10, scoring= 'roc_auc')

#training
%time grid_XGB.fit(X_train[BestFeat_XGB], y_train)

#let's see the best estimator
best_XGB = grid_XGB.best_estimator_
print(best_XGB)
print("_"*40)
#with its score
print(np.abs(grid_XGB.best_score_))
#accuracy on test
predictions = best_XGB.predict(X_test[BestFeat_XGB])
accuracy_score(y_true = y_test, y_pred = predictions)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": best_XGB.predict(test[BestFeat_XGB])
    })
submission.to_csv('submission.csv', index=False)