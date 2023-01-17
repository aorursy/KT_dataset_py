# Importing Required Module for Data Preperation And Analysis

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



warnings.filterwarnings('ignore')

sns.set()

%matplotlib inline
# Reading CSV File 

data = pd.read_csv('../input/mushrooms.csv')

data.head() 
# Number of CLasses Counts

data['class'].value_counts()
# Updating Labels for odor

odor_dict = {"a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy", "f": "Foul", "m": "Musty", "n": "None", "p": "Pungent", "s": "Spicy"} 

data['odor'] = data['odor'].apply(lambda x:odor_dict[x])
# Getting all Poisonous Mushroom odor   

Poisonous_Odor = data[data['class']=='p']['odor'].value_counts()



# # Showing With Pie Chart Poisonous_Odor

ordor_pos = Poisonous_Odor.plot(kind='bar',figsize=(12,5),fontsize=12)

ordor_pos.set_title('Poisonous Mushroom with their Odor',fontsize=14)

ordor_pos.tick_params(labelrotation=0)
# Getting all Edible Mushroom odor

Edible_Odor = data[data['class']=='e']['odor'].value_counts()



# # Showing with Pie Chart Edible_Odor

ordor_ed = Edible_Odor.plot(kind='bar',figsize=(12,5),fontsize=12)

ordor_ed.set_title('Edible Mushroom with their Odor',fontsize=14)

ordor_ed.tick_params(labelrotation=0)
# Bruises Mushrooms :  

# t: Bruises:True

# f: No Bruises: False



Poisonous_Bruises = data[data['class']== 'p']['bruises'].value_counts()

Edible_Bruises = data[data['class']== 'e']['bruises'].value_counts()



Bruises = pd.DataFrame([Poisonous_Bruises,Edible_Bruises],

                       index=['Poisonous','Edible'])

Bruises.plot(kind = 'barh',stacked = True,figsize=(14,5),fontsize=12)

plt.title("Bruises Mushrooms",fontsize=14)

plt.legend(labels=["f = False","t = True"],fontsize=12)

plt.show()
# Updating name for habitat feature

habitate_dict = {"g": "Grasses","l": "Leaves","m": "Meadows", "p": "Paths", "u": "Urban", "w": "Waste", "d": "Woods"}

data['habitat'] = data['habitat'].apply(lambda x:habitate_dict[x])
# Analysing Habitat for Edible Mushroom:

Edible_habitate = data[data['class'] == 'e']['habitat'].value_counts()

habit_ed = Edible_habitate.plot(kind='bar',figsize=(12,5),fontsize=12)

habit_ed.set_title('Edible Mushroom with their Habitat',fontsize=14)

habit_ed.tick_params(labelrotation=0)
# Analysing Habitat for Poisonous Mushroom:

Poisonous_habitate = data[data['class'] == 'p']['habitat'].value_counts()

habit_pos = Poisonous_habitate.plot(kind='bar',figsize=(12,5))

habit_pos.set_title('Poisonous Mushroom with their Habitat',fontsize=14)

habit_pos.tick_params(labelrotation=0)

plt.show()
data.head(3)
# basic label encoding

from sklearn.preprocessing import LabelEncoder



df = data.copy()

label_encoder = LabelEncoder()

for column in df.columns:

    df[column] = label_encoder.fit_transform(df[column])

df.head()    
# Using Advanced Plotting Tool to plot correlation 

plt.figure(figsize=(15,15))

sns.heatmap(df.corr(), annot=True, linewidths=.5)

plt.show()
# Seaparating the labels and Features

Label = df['class']

Features = df.drop(['class'],axis=1)
# Spltting Data in Training and Testing Data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(Features,Label,random_state = 125)
# Importing Models

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier 
# Model List

models = []

models.append(('LogisticRegression',LogisticRegression()))

models.append(('GaussianNB',GaussianNB()))

models.append(('KNeighborsClassifier',KNeighborsClassifier()))

models.append(('SVC',SVC()))

models.append(('DecisionTreeClassifier',DecisionTreeClassifier()))

models.append(('RandomForestClassifier',RandomForestClassifier()))
from sklearn.model_selection import cross_val_score

acc = []

names = []

result = []



for name, model in models:

    # Cross Validation

    acc_of_model = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')

    # Accuracy of model

    acc.append(acc_of_model)

    # Name of model

    names.append(name)

    

    Out = "Model: %s: Accuracy: %f" % (name, acc_of_model.mean())

    result.append(acc_of_model)

    print(Out)