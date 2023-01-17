import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import warnings



%matplotlib inline

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv")
data.head(3)
print("Shape of Dataset is", data.shape)
data.describe()
data.info()
data.isnull().any()
# Missing values

def missing_values_table(df):

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_cols = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        mis_val_table_cols = mis_val_table_cols[mis_val_table_cols.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

        print ("Your selected dataframe has " + str(df.shape[1]))   

        print("There are " + str(mis_val_table_cols.shape[0])+" columns that have missing values.")

        return mis_val_table_cols
miss_values= missing_values_table(data)

miss_values.style.background_gradient(cmap='plasma')
data['species'].value_counts()

sns.countplot(data['species'],palette = "gist_ncar")
data['island'].value_counts()

sns.countplot(data['island'], palette = "cubehelix_r")
data['sex'].value_counts()
data['sex'].fillna(data['sex'].mode()[0],inplace=True)

data["sex"].replace({".": "FEMALE"}, inplace=True)

data['sex'].value_counts()
sns.countplot(data['sex'],palette="cubehelix")
data['culmen_length_mm'].groupby(data['sex']).mean()
data.groupby(['sex', 'species'])['culmen_length_mm'].median()
col_to_imput = ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g']

for item in col_to_imput:

    data[item].fillna(data[item].median(),inplace=True)
missing_values= missing_values_table(data)

missing_values.style.background_gradient(cmap='Reds')
sns.heatmap(data.isnull(), yticklabels= False)
sns.heatmap(data.corr(), annot = True, cmap="magma" )
sns.pairplot(data,hue='species')
sns.pairplot(data,hue='sex', palette="Dark2" )
fig,axes=plt.subplots(2,2,figsize=(10,10))

sns.boxplot(x=data.species,y=data.flipper_length_mm,hue = data.sex, ax=axes[0,0])

sns.boxplot(x=data.species,y=data.culmen_length_mm,hue = data.sex, ax=axes[0,1])

sns.boxplot(x=data.species,y=data.culmen_depth_mm,hue = data.sex, ax=axes[1,0])

sns.boxplot(x=data.species,y=data.body_mass_g,hue = data.sex, ax=axes[1,1])
#distribution plot

fig,axes=plt.subplots(2,2,figsize=(10,10))

sns.distplot(data.flipper_length_mm,ax=axes[0,0])

sns.distplot(data.culmen_length_mm,ax=axes[0,1])

sns.distplot(data.culmen_depth_mm,ax=axes[1,0])

sns.distplot(data.body_mass_g,ax=axes[1,1])
col_list = ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g']

col = 'species'

row = 'sex'

for i in col_list:

    grid = sns.FacetGrid(data, col=col, row=row, size=2.2, aspect=1.6)

    grid.map(plt.hist, i, alpha=.5, bins=20)

    grid.add_legend();
sns.FacetGrid(data = data,row = "island", col = "sex").map(plt.scatter ,"flipper_length_mm","body_mass_g").add_legend()
sns.FacetGrid(data = data,row = "island", col = "sex").map(plt.scatter ,'culmen_length_mm', 'culmen_depth_mm' ).add_legend()
from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
le = LabelEncoder() 

  

data['sex']= le.fit_transform(data['sex']) 

data['island']= le.fit_transform(data['island'])

data['species']= le.fit_transform(data['species'])
data.head()
#defining logistic regression model

def logreg(X,y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    lr = LogisticRegression()

    lr.fit(X_train,y_train)

    y_pred = lr.predict(X_val)

    print('Accuracy : ', accuracy_score(y_val, y_pred))

    print('F1 Score : ', f1_score(y_val, y_pred, average = 'weighted'))

    print('Precision : ', precision_score(y_val, y_pred, average = 'weighted'))

    print('Recall : ', recall_score(y_val, y_pred, average = 'weighted'))
#defining decision tree model

def DesTre(X,y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    dtc = DecisionTreeClassifier(criterion='entropy')

    dtc.fit(X_train,y_train)

    y_pred = dtc.predict(X_val)

    print('Accuracy : ', accuracy_score(y_val, y_pred))

    print('F1 Score : ', f1_score(y_val, y_pred, average = 'weighted'))

    print('Precision : ', precision_score(y_val, y_pred, average = 'weighted'))

    print('Recall : ', recall_score(y_val, y_pred, average = 'weighted'))
#defining random forest classifer model

def rfc(X,y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    rfc = RandomForestClassifier()

    rfc.fit(X_train,y_train)

    y_pred = rfc.predict(X_val)

    print('Accuracy : ', accuracy_score(y_val, y_pred))

    print('F1 Score : ', f1_score(y_val, y_pred, average = 'weighted'))

    print('Precision : ', precision_score(y_val, y_pred, average = 'weighted'))

    print('Recall : ', recall_score(y_val, y_pred, average = 'weighted'))
#defining k neighour model

def knn(X,y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=10,weights='distance',n_jobs=100)

    knn.fit(X_train,y_train)

    y_pred = knn.predict(X_val)

    print('Accuracy : ', accuracy_score(y_val, y_pred))

    print('F1 Score : ', f1_score(y_val, y_pred, average = 'weighted'))

    print('Precision : ', precision_score(y_val, y_pred, average = 'weighted'))

    print('Recall : ', recall_score(y_val, y_pred, average = 'weighted'))
X = data.drop('species', axis = 1)

y = data['species']
logreg(X,y)
DesTre(X,y)
rfc(X,y)
knn(X,y)
X_gender = data.drop('sex', axis = 1)

y_gender = data['sex']
logreg(X_gender,y_gender)
DesTre(X_gender,y_gender)
rfc(X_gender,y_gender)
knn(X_gender,y_gender)
X_island = data.drop('island', axis = 1)

y_island = data['island']
logreg(X_island,y_island)
DesTre(X_island,y_island)
rfc(X_island,y_island)
knn(X_island,y_island)