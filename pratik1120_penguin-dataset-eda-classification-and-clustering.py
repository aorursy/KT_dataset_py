import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px

from pandas_profiling import ProfileReport

from plotly.offline import iplot

!pip install joypy

import joypy

from sklearn.cluster import KMeans



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")



data = pd.read_csv('../input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')

study_data = pd.read_csv('../input/palmer-archipelago-antarctica-penguin-data/penguins_lter.csv')
# description



data.describe(include='all')
#Covariance



data.cov()
#correlation



data.corr()
data['species'].value_counts().plot(kind='barh')

plt.show()
#checking number of null values in the data

data.isnull().sum()
# Dropping the 2 rows with null values for all variables



data.drop(data[data['body_mass_g'].isnull()].index,axis=0, inplace=True)
#imputing the null values in sex with its mode



data['sex'] = data['sex'].fillna('MALE')
#dropping the 336th row due to its faulty value in sex variable



data.drop(data[data['sex']=='.'].index, inplace=True)
print('Culmen Length Distribution')

sns.violinplot(data=data, x="species", y="culmen_length_mm", size=8)

plt.show()
print('Culmen Depth Distribution')

sns.boxplot(data=data, x="species", y="culmen_depth_mm")

plt.show()
print('Flipper Length Distribution')

df = data.copy()

df["MALE_flipper"] = df.apply(lambda row: row["flipper_length_mm"] if row["sex"] == "MALE" else np.nan, axis = 1)

df["FEMALE_flipper"] = df.apply(lambda row: row["flipper_length_mm"] if row["sex"] == "FEMALE" else np.nan, axis = 1)

fig, axes = joypy.joyplot(df, 

                          column=['FEMALE_flipper', 'MALE_flipper'],

                          by = "species",

                          ylim = 'own',

                          figsize = (12,8), 

                          legend = True

                         )
print('Body Mass Distribution')

sns.FacetGrid(data, hue="species", height=6,).map(sns.kdeplot, "body_mass_g",shade=True).add_legend()

plt.show()
print('culmen_length vs culmen_depth')

sns.scatterplot(data=data, x='culmen_length_mm', y='culmen_depth_mm', hue='species')

plt.show()
print('culmen_length vs flipper_length')

sns.scatterplot(data=data, x='culmen_length_mm', y='flipper_length_mm', hue='species')

plt.show()
print('culmen_depth vs flipper_length')

sns.scatterplot(data=data, x='culmen_depth_mm', y='flipper_length_mm', hue='species')

plt.show()
print('culmen_depth vs body_mass')

sns.scatterplot(data=data, x='culmen_depth_mm', y='body_mass_g', hue='species')

plt.show()
print('culmen_length vs body_mass')

sns.scatterplot(data=data, x='culmen_length_mm', y='body_mass_g', hue='species')

plt.show()
print('flipper_length vs body_mass')

sns.scatterplot(data=data, x='flipper_length_mm', y='body_mass_g', hue='species')

plt.show()
print('Pairplot')

sns.pairplot(data=data[['species','culmen_length_mm','culmen_depth_mm','flipper_length_mm', 'body_mass_g']], hue="species", height=3, diag_kind="hist")

plt.show()
print('Which island consists of most Penguins?')

print('Answer: Biscoe')

df = data['island'].value_counts().reset_index()



fig = sns.barplot(data=df, x='island', y='index')

fig.set(xlabel='', ylabel='ISLANDS')

plt.show()
print('Which species have highest culmen_length?')

print('Answer: Chinstrap(male and female)')

df = data.loc[:,['species','culmen_length_mm','sex']]

df['mean_culmen_length'] = df.groupby(['species','sex'])['culmen_length_mm'].transform('mean')

df = df.drop('culmen_length_mm', axis=1).drop_duplicates()



sns.barplot(data=df, x='mean_culmen_length', y='species', hue='sex')

plt.show()
print('Which species have highest culmen_depth?')

print('Answer: Chinstrap(male and female)')

df = data.loc[:,['species','culmen_depth_mm','sex']]

df['mean_culmen_depth'] = df.groupby(['species','sex'])['culmen_depth_mm'].transform('mean')

df = df.drop('culmen_depth_mm', axis=1).drop_duplicates()



sns.barplot(data=df, x='mean_culmen_depth', y='species', hue='sex')

plt.show()
print('Which species have highest flipper_length?')

print('Answer: Gentoo(male and female)')

df = data.loc[:,['species','flipper_length_mm','sex']]

df['mean_flipper_length'] = df.groupby(['species','sex'])['flipper_length_mm'].transform('mean')

df = df.drop('flipper_length_mm', axis=1).drop_duplicates()



sns.barplot(data=df, x='mean_flipper_length', y='species', hue='sex')

plt.show()
print('Which species have highest body_mass?')

print('Answer: Gentoo(male and female) - Highly diverse values noticed')

df = data.loc[:,['species','body_mass_g','sex']]

df['mean_body_mass'] = df.groupby(['species','sex'])['body_mass_g'].transform('mean')

df = df.drop('body_mass_g', axis=1).drop_duplicates()



sns.barplot(data=df, x='mean_body_mass', y='species', hue='sex')

plt.show()
df = data.copy()

target = 'sex'

encode = ['species','island']



for col in encode:

    dummy = pd.get_dummies(df[col], prefix=col)

    df = pd.concat([df,dummy], axis=1)

    del df[col]
target_mapper = {'MALE':0, 'FEMALE':1}

def target_encode(val):

    return target_mapper[val]



df['sex'] = df['sex'].apply(target_encode)
#separating X and y



X = df.drop('sex', axis=1)

y = df['sex']
# scaling the data



from sklearn import preprocessing

X = preprocessing.scale(X)
#splitting the data



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=13)
# model fitting and prediction



from sklearn.linear_model import LogisticRegression



model = LogisticRegression().fit(X_train, y_train)

pred = model.predict(X_test)
# checking performance of model



from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score



print('CONFUSION MATRIX')

print(confusion_matrix(y_test, pred))
print('CLASSIFICATION REPORT\n')

print(classification_report(y_test, pred))
# ROC CURVE



print('ROC CURVE')

train_probs = model.predict_proba(X_train)

train_probs1 = train_probs[:, 1]

fpr0, tpr0, thresholds0 = roc_curve(y_train, train_probs1)



test_probs = model.predict_proba(X_test)

test_probs1 = test_probs[:, 1]

fpr1, tpr1, thresholds1 = roc_curve(y_test, test_probs1)



plt.plot(fpr0, tpr0, marker='.', label='train')

plt.plot(fpr1, tpr1, marker='.', label='validation')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()
df = data.copy()
print('CLUSTERING ON CULMEN LENGTH AND CULMEN DEPTH')

X = df[['culmen_length_mm','culmen_depth_mm']]



kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

y_kmeans = kmeans.predict(X)



plt.scatter(X.loc[:, 'culmen_length_mm'], X.loc[:, 'culmen_depth_mm'], c=y_kmeans, s=50, cmap='viridis')



centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.show()
print('CLUSTERING ON FLIPPER LENGTH AND CULMEN DEPTH')

X = df[['flipper_length_mm','culmen_depth_mm']]



kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

y_kmeans = kmeans.predict(X)



plt.scatter(X.loc[:, 'flipper_length_mm'], X.loc[:, 'culmen_depth_mm'], c=y_kmeans, s=50, cmap='viridis')



centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.show()
print('CLUSTERING ON FLIPPER LENGTH AND BODY MASS')

X = df[['flipper_length_mm','body_mass_g']]



kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

y_kmeans = kmeans.predict(X)



plt.scatter(X.loc[:, 'flipper_length_mm'], X.loc[:, 'body_mass_g'], c=y_kmeans, s=50, cmap='viridis')



centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.show()