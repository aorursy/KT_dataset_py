# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#step 0

import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RepeatedKFold

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.pylab as plb



#Step0.1

acesso_um = "/kaggle/input/titanic/train.csv"

acesso_dois = "/kaggle/input/titanic/test.csv"

train = pd.read_csv(acesso_um, sep= ",", encoding="UTF-8")

test = pd.read_csv(acesso_dois, sep= ",", encoding="UTF-8")

#Step0.1

train.head(10)
#Step0.1

test.head(10)

#step1

train.corr()
#step1.1 (1)

# we used isnull() to find null values and sum() to sum them

train.Sex.isnull().sum()

#step1.1 (2)

# this is an errorbar grafic and here we can see the data standard deviation or the error length

sns.barplot(x="Sex",y="Survived", data= train)

plt.show();

#step1.1 (3)

plb.hist(train.Sex);

#step1.2 (1)

train.Age.isnull().sum()

#step1.2 (2)

# create another variable into train and test dataset named "NewOrderAge" with the role to make the analisis easier



def catage(vage):

	if ((vage >= 0) and (vage <= 6)):

		return "Baby"

	elif ((vage > 6) and (vage <= 13)):

			return "Child"

	elif ((vage > 13) and (vage <= 19)):

		return "Teen"

	elif ((vage > 19) and (vage <= 61)):

		return "Adult"

	elif vage > 61:

		return "Older"

	else:

		return "NV"

#step1.2.1 (2)



train["NewOrderAge"] = train["Age"].map(catage)

test["NewOrderAge"] = test["Age"]. map(catage)

#step1.2.2 (2)

#just to verify if it's work's

train.head()

#step1.2.3 (2)

#just to verify if it's work's

test.head()

#step1.2 (3)

sns.barplot(x="NewOrderAge",y="Survived", data= train)

plt.show();

#step1.2 (4)

plb.hist(train.NewOrderAge);

#step1.3 (1)

#

train["Name_Contem_Miss"] = train["Name"].str.contains("Miss").astype(int)

train["Name_Contem_Mrs"] = train["Name"].str.contains("Mrs").astype(int)

train["Name_Contem_Master"] = train["Name"].str.contains("Master").astype(int)

train["Name_Contem_Col"] = train["Name"].str.contains("Col").astype(int)

train["Name_Contem_Mr"] = train["Name"].str.contains("Mr").astype(int)



test["Name_Contem_Miss"] = test["Name"].str.contains("Miss").astype(int)

test["Name_Contem_Mrs"] = test["Name"].str.contains("Mrs").astype(int)

test["Name_Contem_Master"] = test["Name"].str.contains("Master").astype(int)

test["Name_Contem_Col"] = test["Name"].str.contains("Col").astype(int)

test["Name_Contem_Mr"] = test["Name"].str.contains("Mr").astype(int)

#step1.3 (2)

#just verify



train.head()

#step1.3 (2)

#just verify



test.head()
#step1.4 (1)

train.Pclass.isnull().sum()
#step1.4 (2)

sns.barplot(x="Pclass",y="Survived", data= train)

plt.show();
#step1.4 (3)

plb.hist(train.Pclass);

plb.hist(train.Fare)

sns.barplot(x="Pclass",y="Embarked", data= train)

plt.show();

#step1.4 (1)

def Embarque_S(sl):

    if sl == "S":

        return 1

    else:

        return 0

    

def Embarque_C(cl):

    if cl == "C":

        return 1

    else:

        return 0



train["Embarked_S"] = train["Embarked"].map(Embarque_S)

train["Embarked_C"] = train["Embarked"].map(Embarque_C)



test["Embarked_S"] = test["Embarked"].map(Embarque_S)

test["Embarked_C"] = test["Embarked"].map(Embarque_C)

#step1.4 (2)

train.head()
#step1.4 (2)

test.head()
#Step2::: Prepare the data



def fsexqual(squal):

    if squal == "female":

        return 8

    else:

        return 2



def fembarkedqual(embqual):

    if embqual == "C":

        return 3

    elif embqual == "Q":

        return 2

    else:

        return 1

    

def fpclass(vpclass):

    if vpclass == 1:

        return 3

    elif vpclass == 2:

        return 2

    else:

        return 1



def fagequal(fagq):

	if fagq == "Baby":

		return 8

	elif fagq == "Teen":

		return 5

	elif fagq == "Adult":

		return 4

	elif fagq == "Child":

		return 6

	elif fagq == "NV":

		return 4

	else:

		return 4





modelo = LogisticRegression(random_state=0)  



variaveis = ["SexQual", "Age", "Pclass", "Fare", "Embarked_S",

             "Embarked_C", "Name_Contem_Miss", "Name_Contem_Mrs",

             "Name_Contem_Master", "Name_Contem_Col", "Name_Contem_Mr",

             "PclassQually", "EmbarkedQuall", "AgeQuall"]



train["SexQual"] = train["Sex"].map(fsexqual)

train["PclassQually"] = train["Pclass"].map(fpclass)

train["EmbarkedQuall"] = train["Embarked"].map(fembarkedqual)

train["AgeQuall"] = train["NewOrderAge"].map(fagequal)



test["SexQual"] = test["Sex"].map(fsexqual)

test["PclassQually"] = test["Pclass"].map(fpclass)

test["EmbarkedQuall"] = test["Embarked"].map(fembarkedqual)

test["AgeQuall"] = test["NewOrderAge"].map(fagequal)



x = train[variaveis].fillna(-1)

y = train["Survived"]

modelo.fit(x,y)



x_prev = test[variaveis]

x_prev = x_prev.fillna(-1)

p = modelo.predict(test[variaveis].fillna(-1))
#step2.1

#verify our dataset

x.head()

#step3 training data

np.random.seed(0)

x_treino, x_valid, y_treino, y_valid = train_test_split(x, y, test_size = 0.5)

modelo.fit(x_treino, y_treino)

q = modelo.predict(x_valid)



Acuracia = np.mean(y_valid == q)



#step3 Valid

values = []

kf = RepeatedKFold(n_splits=2, n_repeats=40, random_state=10)

for linhas_treino, linhas_valid in kf.split(x):

    print("Treino:", linhas_treino.shape[0])

    print("Valid:", linhas_valid.shape[0])

    x_treino, x_valid = x.iloc[linhas_treino], x.iloc[linhas_valid]

    y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]

    modelo = LogisticRegression(random_state=0)

    modelo.fit(x_treino, y_treino)

    s = modelo.predict(x_valid)

    acc = np.mean(y_valid == s)

    values.append(acc)

        

media_r = np.mean(values)



modelo = LogisticRegression(random_state=0)

modelo.fit(x,y)

r = modelo.predict(test[variaveis].fillna(-1))
print("Media dos Valores: ", media_r)

print()

print("Acc: ", Acuracia)

print()
plb.hist(values);

#step Final: create a .csv File and submit to kaggle

sub = pd.Series(r, index=test["PassengerId"], name="Survived")

#sub.to_csv("SubNtebookOne.csv", header=True)

sub.head(20)
