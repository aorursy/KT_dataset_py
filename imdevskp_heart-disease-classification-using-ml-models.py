import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



# plt.style.use('default')

color_pallete = ['#2a2a2a', '#ff0000']

sns.set_palette(color_pallete, 2)

sns.set_style("whitegrid")
df = pd.read_csv('../input/heart.csv')

df.head()
# data.describe()
# df.info()
# df.sample(5)
df.isna().sum()
sns.countplot(x='target', data=df)

plt.show()
plt.figure(figsize=(10,10))

sns.heatmap(df.corr(), annot=True, fmt='.1f', cmap='RdBu', vmax=0.8, vmin=-0.8)

plt.show()
# plt.figure(figsize=(8, 8))

# sns.pairplot(df, hue="target")

# plt.plot()
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

dis_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']



for i in cat_cols:

    df[i] = df[i].astype('category')
for i in dis_cols:

    ax = sns.kdeplot(df[df['target']==1][i], shade=True, )

    ax = sns.kdeplot(df[df['target']==0][i], shade=True)

    ax.set_xlabel(i)

    plt.legend(['Absent', 'Present'])

    plt.show()
for i in cat_cols:

    sns.countplot(x=i, hue='target', data=df)

    ax.set_xlabel(i)

    plt.legend(['Absent', 'Present'])

    plt.show()
df['age_cat'] = pd.cut(df['age'], 

                       bins = [0, 40, 55, 90],

                       labels = ['young', 'mid', 'old'],

                       include_lowest=True)



ax = sns.countplot(x="age_cat", hue='target', data=df)

plt.plot()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



for i in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach' , 'exang', 'oldpeak', 'slope', 'ca', 'thal']:

    df[i] = df[i].astype('float64')

    df[i] =  sc.fit_transform(df[i].values.reshape(-1,1))

    

df.head()
df = pd.get_dummies(df, drop_first=True)

df.head()
df.columns
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',

       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'age_cat_mid',

       'age_cat_old']]

y = df['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
nbc = GaussianNB()

nbc.fit(X_train, y_train)

pred = nbc.predict(X_test)



print(accuracy_score(pred, y_test))

print(confusion_matrix(pred, y_test))
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(nbc, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist(), )
# decision tree

# tree visualization

# auc, roc plot

# heatmap - confusion matrix

# neural network