import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import warnings

import seaborn as sns

import pandas as pd

import numpy as np

from colorama import Fore, Back, Style 

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

import xgboost

from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import plotly.express as px

from statsmodels.formula.api import ols

import plotly.graph_objs as gobj

init_notebook_mode(connected=True)

warnings.filterwarnings("ignore")

import plotly.figure_factory as ff



%matplotlib inline



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
input_data = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

input_data.head()
hist_data =[input_data["age"].values]

group_labels = ['age'] 



fig = ff.create_distplot(hist_data, group_labels)

fig.update_layout(title_text='Plot of Age distribution')



fig.show()
fig = px.box(input_data, x='sex', y='age', points="all")

fig.update_layout(

    title_text="Spread(age v/s gender) - Male -> 1 Female -> 0")

fig.show()
male = input_data[input_data["sex"]==1]

female = input_data[input_data["sex"]==0]

male_survi = male[input_data["DEATH_EVENT"]==0]

male_not = male[input_data["DEATH_EVENT"]==1]

female_survi = female[input_data["DEATH_EVENT"]==0]

female_not = female[input_data["DEATH_EVENT"]==1]

labels = ['Male - Survived','Male - Not Survived', "Female -  Survived", "Female - Not Survived"]

values = [len(male[input_data["DEATH_EVENT"]==0]),len(male[input_data["DEATH_EVENT"]==1]),

         len(female[input_data["DEATH_EVENT"]==0]),len(female[input_data["DEATH_EVENT"]==1])]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Survival analysis using gender")

fig.show()
surv = input_data[input_data["DEATH_EVENT"]==0]["age"]

not_surv = input_data[input_data["DEATH_EVENT"]==1]["age"]

hist_data = [surv,not_surv]

group_labels = ['Survived', 'Not Survived']

fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)

fig.update_layout(

    title_text="Analysis in Age on Survival Status")

fig.show()
fig = px.violin(input_data, y="age", x="sex", color="DEATH_EVENT", box=True, points="all", hover_data=input_data.columns)

fig.update_layout(title_text="Analysis in Age and Gender on Survival Status")

fig.show()
fig = px.violin(input_data, y="age", x="smoking", color="DEATH_EVENT", box=True, points="all", hover_data=input_data.columns)

fig.update_layout(title_text="Analysis in Age and Smoking on Survival Status")

fig.show()
fig = px.violin(input_data, y="age", x="diabetes", color="DEATH_EVENT", box=True, points="all", hover_data=input_data.columns)

fig.update_layout(title_text="Analysis in Age and Diabetes on Survival Status")

fig.show()
fig = px.histogram(input_data, x="creatinine_phosphokinase", color="DEATH_EVENT", marginal="violin", hover_data=input_data.columns)

fig.show()
fig = px.histogram(input_data, x="ejection_fraction", color="DEATH_EVENT", marginal="violin", hover_data=input_data.columns)

fig.show()
fig = px.histogram(input_data, x="platelets", color="DEATH_EVENT", marginal="violin", hover_data=input_data.columns)

fig.show()
fig = px.histogram(input_data, x="serum_creatinine", color="DEATH_EVENT", marginal="violin", hover_data=input_data.columns)

fig.show()
fig = px.histogram(input_data, x="serum_sodium", color="DEATH_EVENT", marginal="violin",hover_data=input_data.columns)

fig.show()
fig = px.pie(input_data, values='diabetes',names='DEATH_EVENT', title='Diabetes Death Event Ratio')

fig.show()
fig = px.pie(input_data, values='anaemia',names='DEATH_EVENT', title='Anaemia Death Event Ration')

fig.show()
fig = px.pie(input_data, values='anaemia',names='DEATH_EVENT', title='Anaemia Death Event Ration')

fig.show()
fig = px.pie(input_data, values='high_blood_pressure',names='DEATH_EVENT', title='High Blood Pressure Death Event Ratio')

fig.show()
plt.figure(figsize=(10,10))

sns.heatmap(input_data.corr(), vmin=-1, cmap='coolwarm', annot=True);
Features = ['time','ejection_fraction','serum_creatinine','age']

x = input_data[Features]

y = input_data["DEATH_EVENT"]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2698)
clf = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)

clf.fit(x_train, y_train)

pred=clf.predict(x_test)

print(Fore.GREEN + "Accuracy of RandomForestClassifier is : ",clf.score(x_test,y_test))

cm = confusion_matrix(y_test, pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("Random Forest Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
gradientboost_clf = GradientBoostingClassifier(max_depth=2, random_state=4)

gradientboost_clf.fit(x_train,y_train)

gradientboost_pred = gradientboost_clf.predict(x_test)

print(Fore.GREEN + "Accuracy of Gradient Boosting is : ",gradientboost_clf.score(x_test,y_test))

cm = confusion_matrix(y_test, gradientboost_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("Gredient Boosting Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()