import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

!pip install plotly

import plotly.plotly as py

import plotly.tools as tls

import sklearn.metrics

import warnings

warnings.filterwarnings('ignore')
# This line loads the data into a dataframe

mentalhealth_df = pd.read_csv("../input/../input/survey.csv")

# This line prints out the first 5 rows of the data

mentalhealth_df.head()
genderset = set()

for i, row in mentalhealth_df.iterrows():

    if row['Gender'] not in genderset:

        genderset.add(row['Gender'])

        print(row['Gender'])
male = set(["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr", "cis man"])

female = set(["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"])

for i, row in mentalhealth_df.iterrows():

    if row['Gender'].lower() in male:

        mentalhealth_df.set_value(i, 'Gender', "Male")

    elif row['Gender'].lower() in female:

        mentalhealth_df.set_value(i, 'Gender', "Female")

    else:

        mentalhealth_df.set_value(i, 'Gender', "Transgender")

mentalhealth_df.head()
genderset = set()

for i, row in mentalhealth_df.iterrows():

    if row['Gender'] not in genderset:

        genderset.add(row['Gender'])

        print(row['Gender'])
numbersdf = pd.DataFrame()



def yn(i, row, x):

    if row[x] == "Yes":

        numbersdf.set_value(i, x, 1)

    else:

        numbersdf.set_value(i, x, 0)



def ydkn(i, row, x):

    if row[x] == "Yes":

        numbersdf.set_value(i, x, 2)

    elif row[x] == "Don't know":

        numbersdf.set_value(i, x, 1)

    else:

        numbersdf.set_value(i, x, 0)

        

def ymn(i, row, x):

    if row[x] == "Yes":

        numbersdf.set_value(i, x, 2)

    elif row[x] == "Maybe":

        numbersdf.set_value(i, x, 1)

    else:

        numbersdf.set_value(i, x, 0)



def ysotn(i, row, x):

    if row[x] == "Yes":

        numbersdf.set_value(i, x, 2)

    elif row[x] == "Some of them":

        numbersdf.set_value(i, x, 1)

    else:

        numbersdf.set_value(i, x, 0)

        

for i, row in mentalhealth_df.iterrows():

    numbersdf.set_value(i, 'Age', row['Age'])

    if row['Gender'] == "Male":

        numbersdf.set_value(i, 'Gender', 2)

    elif row['Gender'] == "Trans":

        numbersdf.set_value(i, 'Gender', 1)

    else:

        numbersdf.set_value(i, 'Gender', 0)

    yn(i, row, 'family_history')

    yn(i, row, 'treatment')

    if row['work_interfere'] == "Often":

        numbersdf.set_value(i, 'work_interfere', 3)

    elif row['work_interfere'] == "Sometimes":

        numbersdf.set_value(i, 'work_interfere', 2)

    elif row['work_interfere'] == "Rarely":

        numbersdf.set_value(i, 'work_interfere', 1)

    else:

        numbersdf.set_value(i, 'work_interfere', 0)

    if row['no_employees'] == "1-5":

        numbersdf.set_value(i, 'no_employees', 1)

    elif row['no_employees'] == "6-25":

        numbersdf.set_value(i, 'no_employees', 2)

    elif row['no_employees'] == "26-100":

        numbersdf.set_value(i, 'no_employees', 3)

    elif row['no_employees'] == "100-500":

        numbersdf.set_value(i, 'no_employees', 4)

    elif row['no_employees'] == "500-1000":

        numbersdf.set_value(i, 'no_employees', 5)

    elif row['no_employees'] == "More than 1000":

        numbersdf.set_value(i, 'no_employees', 6)

    else:

        numbersdf.set_value(i, 'no_employees', 0)

    yn(i, row, 'remote_work')

    yn(i, row, 'tech_company')

    ydkn(i, row, 'benefits')

    if row['care_options'] == "Yes":

        numbersdf.set_value(i, 'care_options', 2)

    elif row['care_options'] == "Not sure":

        numbersdf.set_value(i, 'care_options', 1)

    else:

        numbersdf.set_value(i, 'care_options', 0)

    ydkn(i, row, 'wellness_program')

    ydkn(i, row, 'seek_help')

    ydkn(i, row, 'anonymity')

    if row['leave'] == "Very Easy":

        numbersdf.set_value(i, 'leave', 4)

    elif row['leave'] == "Somewhat easy":

        numbersdf.set_value(i, 'leave', 3)

    elif row['leave'] == "Somewhat difficult":

        numbersdf.set_value(i, 'leave', 2)

    elif row['leave'] == "Very difficult":

        numbersdf.set_value(i, 'leave', 1)

    else:

        numbersdf.set_value(i, 'leave', 0)

    ymn(i, row, 'mental_health_consequence')

    ymn(i, row, 'phys_health_consequence')

    ysotn(i, row, 'coworkers')

    ysotn(i, row, 'supervisor')

    ymn(i, row, 'mental_health_interview')

    ymn(i, row, 'phys_health_interview')

    ydkn(i, row, 'mental_vs_physical')

    yn(i, row, 'obs_consequence')

numbersdf.head()
tech = 0

for i, row in numbersdf.iterrows():

    if row['tech_company'] == 1:

        tech += 1

print("Percentage of people that work for a tech company:", 100*tech/1259, "%")
plt.figure(figsize=(10, 10))



corr = numbersdf.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score



models = dict()

models["Logistic Regression:"] = LogisticRegression()

models["K-Nearest Neighbor:"] = KNeighborsClassifier(n_neighbors=10)

models["Random Forest:"] = RandomForestClassifier(n_estimators=10)

models["Decision Tree Classifier:"] = DecisionTreeClassifier(max_depth=10)

models["Gaussian NB:"] = GaussianNB()

models["Quadratic Discriminant Analysis:"] = QuadraticDiscriminantAnalysis()
X = numbersdf.drop(['treatment'], axis=1)

y = numbersdf['treatment']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=10)



ml_df = pd.DataFrame(columns=["Model", "Accuracy"])



def score(model):

    clf = model

    clf.fit(X_train, Y_train)

    return accuracy_score(Y_test, clf.predict(X_test))



i = 0

for name, model in models.items():

    accuracy = score(model)

    ml_df.set_value(i, "Model", name)

    ml_df.set_value(i, "Accuracy", accuracy*100)

    i += 1

    

ml_df
plt.bar(range(len(ml_df['Model'])), ml_df['Accuracy'], align='center')

plt.xticks(range(len(ml_df['Model'])), ml_df['Model'], rotation='vertical')

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.title('Accuracies of ML Models')

 

plt.show()