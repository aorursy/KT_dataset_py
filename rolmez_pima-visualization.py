import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import plotly
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
from plotnine import *
pima = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
pima.head()
pima.groupby('Outcome').mean().T
pima.describe().T
pima.hist(figsize = (10, 10));
trace0 = go.Box(
    name = 'Pregnancies',
    y = pima["Pregnancies"]
)

trace1 = go.Box(
    name = "Glucose",
    y = pima["Glucose"]
)

trace2 = go.Box(
    name = "BloodPressure",
    y = pima["BloodPressure"]
)

trace3 = go.Box(
    name = "SkinThickness",
    y = pima["SkinThickness"]
)

trace4 = go.Box(
    name = "Insulin",
    y = pima["Insulin"]
)

trace5 = go.Box(
    name = "DiabetesPedigreeFunction",
    y = pima["DiabetesPedigreeFunction"]
)

trace6 = go.Box(
    name = "Age",
    y = pima["Age"]
)

trace7 = go.Box(
    name = "Outcome",
    y = pima["Outcome"]
)

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]
plotly.offline.iplot(data)
plt.rcParams.update({'font.size': 25})
sns.set_context("paper")
pima["Over"] = [1 if x > 25 else 0 for x in pima.BMI]
plt.scatter(pima.Age, pima.Insulin, c = pima.Over, s = 389,
            alpha = 0.2, cmap = "viridis")
plt.colorbar();
plt.xlabel("Age")
plt.ylabel("Insulin") 
plt.title("Age and Insulin")
plt.show()
fig, ax = plt.subplots()
ax.scatter(pima.Age, pima.Insulin, c = pima.Over, cmap = "viridis")
ax.set_xlabel("Age")
ax.set_ylabel("Insulin")
ax.set_title("Age and Insulin")
plt.show()
fig, ax = plt.subplots()
ax.hist(pima.Age, label="Age", bins=10)
ax.set_xlabel("Age")
ax.set_ylabel("Number of Observations")
plt.show()
fig, ax = plt.subplots()
ax.bar(pima.Outcome, pima.Insulin)
ax.set_xlabel("Outcome")
ax.set_ylabel("Insulin")
plt.show()
sns.pairplot(pima[['Age', 'Pregnancies', 'Insulin', 'BMI', 'SkinThickness', 'Glucose']])
g = sns.catplot(x = "Age", aspect = 3, data = pima, kind = "count")
g.fig.suptitle("Age Counts", y = 1.1)
plt.show()
sns.scatterplot(x = "Age", y = "Insulin",data = pima, hue="Outcome")
plt.show()
sns.relplot(x = "Insulin", y = "Glucose", data = pima, kind="scatter", row="Outcome")
plt.show()
sns.set_palette("RdBu")
correlation = pima.corr()
sns.heatmap(correlation, annot=True)
plt.show()
ggplot(pima, aes(x = 'Age', y = 'Glucose', colour = 'Outcome')) + geom_point() + stat_smooth()
ggplot(pima, aes(x = 'Age', y = 'Glucose', colour = 'BloodPressure')) + geom_point() + stat_smooth() + facet_wrap('~Outcome')
ggplot(pima, aes(x = 'Age', y = 'Pregnancies')) + geom_point(aes(color = 'BMI')) + facet_wrap('~Outcome') + stat_smooth()
