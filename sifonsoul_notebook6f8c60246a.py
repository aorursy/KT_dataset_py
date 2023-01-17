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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



# plotly library

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)





import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

#warnings.filterwarnings("ignore")

#warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

#warnings.filterwarnings(action='once')



from sklearn.utils.testing import ignore_warnings



from subprocess import check_output



def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn

print(check_output(["ls", "../input"]).decode("utf8"))
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_train.head()
df_test.head()
df_train.info()
df_test.info()
fig, ax = plt.subplots(figsize=(9,5))

sns.heatmap(df_train.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
fig, ax = plt.subplots(figsize=(9,5))

sns.heatmap(df_test.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
pd.isnull(df_test).sum()
cols = ['Survived', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']
ax=sns.countplot(x='Sex',hue='Survived',data=df_train)
nr_rows = 2

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        

        i = r*nr_cols+c       

        ax = axs[r][c]

        sns.countplot(df_train[cols[i]], hue=df_train["Survived"], ax=ax)

        ax.set_title(cols[i], fontsize=14, fontweight='bold')

        ax.legend(title="survived", loc='upper center') 

        

plt.tight_layout()   
bins = np.arange(0, 80, 5)

g = sns.FacetGrid(df_train, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)

g.map(sns.distplot, 'Age', kde=False, bins=bins, hist_kws=dict(alpha=0.6))

g.add_legend()  

plt.show()  
bins = np.arange(0, 550, 50)

g = sns.FacetGrid(df_train, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)

g.map(sns.distplot, 'Fare', kde=False, bins=bins, hist_kws=dict(alpha=0.6))

g.add_legend()  

plt.show()
sns.barplot(x='Pclass', y='Survived', data=df_train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Pclass")

plt.show()
sns.barplot(x="Sex", y="Survived", data=df_train)

plt.ylabel('Survival Sex')

plt.title('Survival as function of Sex')

plt.show()
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df_train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Pclass and Sex")

plt.show()
sns.barplot(x='Embarked', y='Survived', data=df_train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Embarked Port")

plt.show()
sns.barplot(x='Embarked', y='Survived', hue='Pclass', data=df_train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Embarked Port")

plt.show()
sns.countplot(x='Embarked', hue='Pclass', data=df_train)

plt.title("Count of Passengers as function of Embarked Port")

plt.show()
sns.boxplot(x='Embarked', y='Age', data=df_train)

plt.title("Age distribution as function of Embarked Port")

plt.show()
sns.boxplot(x='Embarked', y='Fare', data=df_train)

plt.title("Fare distribution as function of Embarked Port")

plt.show()
cm_surv = ["darkgrey" , "lightgreen"]
ax = sns.swarmplot(x='Pclass', y='Age', data=df_train)
_=sns.swarmplot(x='Pclass', y='Age', hue='Sex', data=data_train)

fig, ax = plt.subplots(figsize=(13,7))

sns.swarmplot(x='Pclass', y='Age', hue='Survived', split=True, data=df_train , palette=cm_surv, size=7, ax=ax)

plt.title('Survivals for Age and Pclass ')

plt.show()
cx = sns.violinplot(x='Pclass',y='Age',hue='Survived', split = True, data=df_train)
fig, ax = plt.subplots(figsize=(13,7))

sns.violinplot(x="Pclass", y="Age", hue='Survived', data=df_train, split=True, bw=0.05 , palette=cm_surv, ax=ax)

plt.title('Survivals for Age and Pclass ')

plt.show()
g = sns.factorplot(x="Pclass", y="Age", hue="Survived", col="Sex", data=df_train, kind="swarm", split=True, palette=cm_surv, size=7, aspect=.9, s=7)
for df in [df_train, df_test] :

    

    df['FamilySize'] = df['SibSp'] + df['Parch'] +1

    

    df['Alone']=0

    df.loc[(df.FamilySize==1),'Alone'] = 1

    

    df['NameLen'] = df.Name.apply(lambda x : len(x)) 

    df['NameLenBin']=np.nan

    for i in range(20,0,-1):

        df.loc[ df['NameLen'] <= i*5, 'NameLenBin'] = i

    

    

    df['Title']=0

    df['Title']=df.Name.str.extract(r'([A-Za-z]+)\.') #lets extract the Salutations

    df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],

                    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
print(df_train[['NameLen' , 'NameLenBin']].head(10))
grps_namelenbin_survrate = df_train.groupby(['NameLenBin'])['Survived'].mean().to_frame()

grps_namelenbin_survrate
plt.subplots(figsize=(10,6))

sns.barplot(x='NameLenBin' , y='Survived' , data = df_train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of NameLenBin")

plt.show()
ax = sns.violinplot(x='Pclass', y='NameLenBin', hue='Survived',  data=df_train)
fig, ax = plt.subplots(figsize=(9,7))

sns.violinplot(x="NameLenBin", y="Pclass", data=df_train, hue='Survived', split=True, 

               orient="h", bw=0.2 , palette=cm_surv, ax=ax)

plt.show()
g = sns.factorplot(x="NameLenBin", y="Survived", col="Sex", data=df_train, kind="bar", size=5, aspect=1.2)

grps_title_survrate = df_train.groupby(['Title'])['Survived'].mean().to_frame()

grps_title_survrate
plt.subplots(figsize=(10,6))

sns.barplot(x='Title', y='Survived', data=df_train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of Pclass")

plt.show()
plt.subplots(figsize=(10,6))

sns.barplot(x='FamilySize' , y='Survived' , data = df_train)

plt.ylabel("Survival Rate")

plt.title("Survival as function of FamilySize")

plt.show()
for df in [df_train, df_test]:



    # Title

    df['Title'] = df['Title'].fillna(df['Title'].mode().iloc[0])



    # Age: use Title to fill missing values

    df.loc[(df.Age.isnull())&(df.Title=='Mr'),'Age']= df.Age[df.Title=="Mr"].mean()

    df.loc[(df.Age.isnull())&(df.Title=='Mrs'),'Age']= df.Age[df.Title=="Mrs"].mean()

    df.loc[(df.Age.isnull())&(df.Title=='Master'),'Age']= df.Age[df.Title=="Master"].mean()

    df.loc[(df.Age.isnull())&(df.Title=='Miss'),'Age']= df.Age[df.Title=="Miss"].mean()

    df.loc[(df.Age.isnull())&(df.Title=='Other'),'Age']= df.Age[df.Title=="Other"].mean()

    df = df.drop('Name', axis=1)

# Embarked

df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode().iloc[0])

df_test['Embarked'] = df_test['Embarked'].fillna(df_test['Embarked'].mode().iloc[0])



# Fare

df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())

df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean())
for df in [df_train, df_test]:

    

    df['Age_bin']=np.nan

    for i in range(8,0,-1):

        df.loc[ df['Age'] <= i*10, 'Age_bin'] = i

        

    df['Fare_bin']=np.nan

    for i in range(12,0,-1):

        df.loc[ df['Fare'] <= i*50, 'Fare_bin'] = i        

    

    # convert Title to numerical

    df['Title'] = df['Title'].map( {'Other':0, 'Mr': 1, 'Master':2, 'Miss': 3, 'Mrs': 4 } )

    # fill na with maximum frequency mode

    df['Title'] = df['Title'].fillna(df['Title'].mode().iloc[0])

    df['Title'] = df['Title'].astype(int)
df_train_ml = df_train.copy()

df_test_ml = df_test.copy()



passenger_id = df_test_ml['PassengerId']
df_train_ml.info()
df_test_ml.info()
df_train_ml = pd.get_dummies(df_train_ml, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

df_test_ml = pd.get_dummies(df_test_ml, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)



df_train_ml.drop(['PassengerId','Name','Ticket', 'Cabin', 'Age', 'Fare_bin'],axis=1,inplace=True)

df_test_ml.drop(['PassengerId','Name','Ticket', 'Cabin', 'Age', 'Fare_bin'],axis=1,inplace=True)
df_train_ml.dropna(inplace=True)
df_train_ml.info()
for df in [df_train_ml, df_test_ml]:

    df.drop(['NameLen'], axis=1, inplace=True)



    df.drop(['SibSp'], axis=1, inplace=True)

    df.drop(['Parch'], axis=1, inplace=True)

    df.drop(['Alone'], axis=1, inplace=True)
df_train_ml.info()
df_test_ml.info()
df_test_ml.fillna(df_test_ml.mean(), inplace=True)

df_test_ml.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



# for df_train_ml

scaler.fit(df_train_ml.drop(['Survived'],axis=1))

scaled_features = scaler.transform(df_train_ml.drop(['Survived'],axis=1))

df_train_ml_sc = pd.DataFrame(scaled_features) # columns=df_train_ml.columns[1::])



# for df_test_ml

df_test_ml.fillna(df_test_ml.mean(), inplace=True)

#scaler.fit(df_test_ml)

scaled_features = scaler.transform(df_test_ml)

df_test_ml_sc = pd.DataFrame(scaled_features) # , columns=df_test_ml.columns)
df_test_ml_sc.head()