import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/mental-health-in-tech-survey-dataset/Mental Health in Tech Survey dataset CSV.csv")

df
df.isnull().sum()
def ordering_age(age):

    if age>=0 and age<=100:

        return age

    else:

        return np.nan

df['Age'] = df['Age'].apply(ordering_age)
print(df['Age'].min())

print(df['Age'].max())
df.Gender.unique()
df['Gender'] = df['Gender'].str.lower()

male = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "cis male"]

trans = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]

female = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

df['Gender'] = df['Gender'].apply(lambda x:"Male" if x in male else x)

df['Gender'] = df['Gender'].apply(lambda x:"Female" if x in female else x)

df['Gender'] = df['Gender'].apply(lambda x:"Trans" if x in trans else x)

df.drop(df[df.Gender == 'p'].index, inplace=True)

df.drop(df[df.Gender == 'a little about you'].index, inplace=True)
#setting the style of background with grid view for clear understanding of data interpretation

plt.figure(figsize=(10,7))

sns.set_style("whitegrid")

sns.distplot(df['Age'].dropna())

plt.title("Distribution of Age")

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x = "Gender", hue="treatment", data=df)

plt.title("Mental health related to Gender", fontweight = "bold")

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x = "self_employed", hue="treatment", data=df)

plt.title("Mental health condition of self-employed people", fontweight = "bold")

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x = "family_history", hue="treatment", data=df)

plt.title("people undergone treatment related to family history", fontweight = "bold")

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x = "supervisor", hue="treatment", data=df)

plt.title("Supervisors undergoing treatment", fontweight = "bold")

plt.show()
top_5 = df['Country'].value_counts()[:5].to_frame()

plt.figure(figsize=(10,7))

sns.barplot(top_5.index,top_5['Country'])

plt.title('Top 5 Countries who contributed the survey',fontweight="bold")

plt.xlabel("Countries")

plt.ylabel("Count")

plt.show()

from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

for i in df.columns:

    df[i] = number.fit_transform(df[i].astype('str'))
corr=df.corr()['treatment']

corr[np.argsort(corr,axis=0)[::-1]]
features_correlation = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(features_correlation,vmax=1,square=True,annot=False)

plt.show()