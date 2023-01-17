# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('/kaggle/input/mytitanic/train.csv')
data.columns
data.dtypes
data.head(10)
labels = data['Survived']
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score





def run_and_validate(numeric_data, labels):

    X_train, X_test, y_train, y_test = train_test_split(numeric_data, labels, test_size=0.33, random_state=42)



    rfc = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=2, min_samples_leaf=1, 

                                        min_weight_fraction_leaf=0.0, random_state=20)

    rfc.fit(X_train, y_train)

    X_train_preds_rfc = rfc.predict(X_train)

    X_test_preds_rfc = rfc.predict(X_test)



    # End

    print(accuracy_score(y_train, X_train_preds_rfc))

    print(accuracy_score(y_test, X_test_preds_rfc))

columns = ['Pclass', 'SibSp', 'Parch']

run_and_validate(data[columns], labels)
# Explore SibSp feature vs Survived

g = sns.factorplot(x="SibSp", y="Survived", data=data, kind="bar", size=6, palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Parch feature vs Survived

g  = sns.factorplot(x="Parch", y="Survived", data=data, kind="bar", size=6, palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")

g = sns.FacetGrid(data, col='Survived')

g = g.map(sns.distplot, "Age")
# Your code here



data['Age'] = data['Age'].fillna(data['Age'].median())



# End



columns = ['Pclass', 'SibSp', 'Parch', 'Age']

run_and_validate(data[columns], labels)
g = sns.FacetGrid(data, col='Survived')

g = g.map(sns.distplot, "Fare")

data["Fare"].isnull().sum()
data["Fare"]
# Your code here

data["Fare"][data["Fare"] > 200] = 200



# End





columns = ['Pclass', 'SibSp', 'Parch', 'Age', 'Fare']

run_and_validate(data[columns], labels)
# Explore Fare distribution 

g = sns.distplot(data["Fare"], color="m", label="Skewness : %.2f"%(data["Fare"].skew()))

g = g.legend(loc="best")
# Your code here

data["Fare"] = data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

# End



columns = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Fare']

run_and_validate(data[columns], labels)

g = sns.distplot(data["Fare"], color="b", label="Skewness : %.2f"%(data["Fare"].skew()))

g = g.legend(loc="best")
f,ax = plt.subplots(figsize=(12, 10))

sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt='.2f',ax=ax)
g = sns.barplot(x="Sex",y="Survived",data=data)

g = g.set_ylabel("Survival Probability")
data[["Sex","Survived"]].groupby('Sex').mean()
# Your code here



from sklearn.preprocessing import LabelEncoder

data['Sex'] = LabelEncoder().fit_transform(data['Sex'])



# End



columns = ['Pclass', 'SibSp', 'Parch', 'Age', 'Fare', 'Sex']

run_and_validate(data[columns], labels)

data["Embarked"]
# Your code here



data['Embarked'] = LabelEncoder().fit_transform(data["Embarked"].fillna("S"))



# End



columns = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Fare', 'Sex', 'Embarked']

run_and_validate(data[columns], labels)

data["Name"].head(10)
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in data["Name"]]

data["Title"] = dataset_title

data["Title"].head(10)
g = sns.countplot(x="Title",data=data)

g = plt.setp(g.get_xticklabels(), rotation=45) 
# Your code here



data["Title"] = data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

data["Title"] = data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

# End

data["Title"] = data["Title"].astype(int)
g = sns.countplot(data["Title"])



columns = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Fare', 'Sex', 'Embarked', 'Title']

run_and_validate(data[columns], labels)

data["Cabin"].isnull().sum()
# Your code here



data["Cabin"] = [i[0] if not pd.isnull(i) else 'X' for i in data['Cabin'] ]

# End
g = sns.countplot(data["Cabin"],order=['A','B','C','D','E','F','G','T','X'])
g = sns.factorplot(y="Survived",x="Cabin",data=data,kind="bar",order=['A','B','C','D','E','F','G','T','X'])

g = g.set_ylabels("Survival Probability")
# Your code here

data = pd.get_dummies(data, columns=["Cabin"], prefix="Cabin")

# End
cabins = [ 'Cabin_' + x for x in ['A','B','C','D','E','F','G','T','X']]

columns = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Fare', 'Sex', 'Embarked', 'Title'] + cabins

run_and_validate(data[columns], labels)
# Your code here

data["Fsize"] = data["SibSp"] + data["Parch"] + 1

# End
# Create new feature of family size

data['Single'] = data['Fsize'].map(lambda s: 1 if s == 1 else 0)

data['SmallF'] = data['Fsize'].map(lambda s: 1 if  s == 2  else 0)

data['MedF'] = data['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

data['LargeF'] = data['Fsize'].map(lambda s: 1 if s >= 5 else 0)



g = sns.factorplot(x="Single",y="Survived",data=data,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="SmallF",y="Survived",data=data,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="MedF",y="Survived",data=data,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="LargeF",y="Survived",data=data,kind="bar")

g = g.set_ylabels("Survival Probability")
data = pd.get_dummies(data, columns = ["Title"])

data = pd.get_dummies(data, columns = ["Embarked"], prefix="Em")



cabins_cols = [ 'Cabin_' + x for x in ['A','B','C','D','E','F','G','T','X']]

embarked_cols = [ 'Em_' + x for x in ['0', '1', '2']]

title_cols = [ 'Title_' + x for x in ['0', '1', '2', '3']]



columns = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Fare', 'Sex',

           'Single', 'SmallF', 'MedF', 'LargeF'] + cabins_cols + embarked_cols + title_cols

run_and_validate(data[columns], labels)



Ticket = []

for i in list(data.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix

    else:

        Ticket.append("X")

        

data["Ticket"] = Ticket

data["Ticket"].head()





data = pd.get_dummies(data, columns = ["Ticket"], prefix="T")

# Create categorical values for Pclass

data["Pclass"] = data["Pclass"].astype("category")

data = pd.get_dummies(data, columns = ["Pclass"],prefix="Pc")

# Drop useless variables 

data.drop(labels = ["PassengerId"], axis = 1, inplace = True)

data.head()
cabins_cols = [ 'Cabin_' + x for x in ['A','B','C','D','E','F','G','T','X']]

embarked_cols = [ 'Em_' + x for x in ['0', '1', '2']]

pc_cols = [ 'Pc_' + x for x in ['1', '2', '3']]

title_cols = [ 'Title_' + x for x in ['0', '1', '2', '3']]

tickets_cols = ['T_A4', 'T_A5','T_AS','T_C','T_CA','T_CASOTON','T_FC','T_FCC','T_Fa','T_LINE','T_PC','T_PP','T_PPP','T_SC', 'T_SCA4','T_SCAH','T_SCOW','T_SCPARIS','T_SCParis','T_SOC','T_SOP','T_SOPP','T_SOTONO2','T_SOTONOQ', 'T_SP', 'T_STONO', 'T_STONO2', 'T_SWPP', 'T_WC', 'T_WEP', 'T_X',]

columns = ['SibSp', 'Parch', 'Fare', 'Age', 'Fare', 'Sex',

           'Single', 'SmallF', 'MedF', 'LargeF'] + cabins_cols + embarked_cols + title_cols + tickets_cols + pc_cols

run_and_validate(data[columns], labels)

# Your code here

from sklearn.decomposition import PCA

pca = PCA(n_components=3, whiten=True)

data_3 = pca.fit_transform(data[columns])



# End
# Dim reduction

import plotly.graph_objs as go

import plotly.offline as py

py.init_notebook_mode(connected=True)



def interactive_3d_plot(data):

    dead = np.where(labels == 0)[0]

    surv = np.where(labels == 1)[0]

    scatt = go.Scatter3d(x=data[surv, 0], y=data[surv, 1], z=data[surv, 2], mode='markers', marker=dict(color='rgb(127, 127, 127)'))

    scatt2 = go.Scatter3d(x=data[dead, 0], y=data[dead, 1], z=data[dead, 2], mode='markers')

    #data = go.Data([scatt, scatt2])

    layout = go.Layout(title="3d projection of the created features")

    figure = go.Figure(data=[scatt, scatt2], layout=layout)

    py.iplot(figure)

    

interactive_3d_plot(data_3)
run_and_validate(data_3, labels)