# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')

df.head()
df.shape
df.columns
df.info()
df_species_count = df.species.value_counts()

# df_species_perc = 100.0*df_species_count / len(df)

df_species_perc = 100.0*df_species_count / df_species_count.sum()



df_species_table = pd.concat([df_species_count, df_species_perc], axis=1)

df_species_table.columns = ['df_species_count','df_species_perc']



df_species_table
df_count = df.island.value_counts()

df_perc = 100.0*df_count / df_count.sum()



df_table = pd.concat([df_count, df_perc], axis=1)

df_table.columns = ['df_island_count','df_island_perc']



df_table
df_count = df.sex.value_counts(dropna=False)

df_perc = 100.0*df_count / df_count.sum()



df_table = pd.concat([df_count, df_perc], axis=1)

df_table.columns = ['df_sex_count','df_sex_perc']



df_table
df.isnull().sum()
df[df.culmen_length_mm.isna()]
df = df.drop(df[df.culmen_length_mm.isna()].index)
df.sex.unique()
df[df.sex=="."]
df = df.drop(df[df.sex=="."].index)
df.shape
df.isnull().sum()
df[df.sex.isna()]
df[df.sex=="MALE"].culmen_length_mm.mean() , df[df.sex=="FEMALE"].culmen_length_mm.mean()
df[df.sex=="MALE"].culmen_depth_mm.mean() , df[df.sex=="FEMALE"].culmen_depth_mm.mean()
df[df.sex=="MALE"].flipper_length_mm.mean() , df[df.sex=="FEMALE"].flipper_length_mm.mean()
df[df.sex=="MALE"].body_mass_g.mean() , df[df.sex=="FEMALE"].body_mass_g.mean()

# df[df.sex=="MALE"].body_mass_g.describe() , df[df.sex=="FEMALE"].body_mass_g.describe()
### actual - find distance from either means and then assign

round((4545.684523809524+ 3862.2727272727275)/2,2)
# df.loc[df['sex'] == 'FEMALE', 'body_mass_g'].mean()



df.loc[(df.sex.isna()) & (df.body_mass_g <= 4203.98) , "sex"] = "FEMALE"
df['sex'] = df['sex'].fillna("MALE")
df.sex.value_counts(dropna=False)
df.describe()
df.info()
plt.figure(figsize = (8,8))

sns.violinplot(x="species", y="culmen_length_mm", data=df)

plt.show()
plt.figure(figsize = (8,8))

sns.violinplot(x="species", y="culmen_depth_mm", data=df)

plt.show()
plt.figure(figsize = (8,8))

sns.violinplot(x="species", y="flipper_length_mm", data=df)

plt.show()
plt.figure(figsize = (8,8))

sns.violinplot(x="species", y="body_mass_g", data=df)

plt.show()
plt.figure(figsize = (15,15))

sns.pairplot(df, hue="species",diag_kind="kde")

plt.show()
# to be updated ...
le = preprocessing.LabelEncoder()
le.fit(df.species)

df.species = le.transform(df.species)
class_names = list(le.classes_)
# try not using drop_first for ISLAND variable

df = pd.get_dummies(df,columns=["island","sex"], drop_first=True)
df.shape
df.head(10)
y = df.species.values
X = df.drop(columns=["species"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# dt = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=42)

dt = DecisionTreeClassifier(criterion='entropy', random_state=42)

# dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)

accuracy = metrics.accuracy_score(y_train, y_pred_train)

print("Accuracy: {:.2f}".format(accuracy))

cm=confusion_matrix(y_train,y_pred_train)

print('Confusion Matrix: \n', cm)

print(classification_report(y_train, y_pred_train, target_names=class_names))
y_pred_test = dt.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred_test)

print("Accuracy: {:.2f}".format(accuracy))

cm=confusion_matrix(y_test,y_pred_test)

print('Confusion Matrix: \n', cm)

print(classification_report(y_test, y_pred_test, target_names=class_names))