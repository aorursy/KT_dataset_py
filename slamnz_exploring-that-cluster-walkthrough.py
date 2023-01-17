import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
#Instantiate dataset



data = pd.read_csv("../input/HR_comma_sep.csv")
#Global Variables



features = data.drop("left",1).columns

categorical_features = ["promotion_last_5years", "Work_accident", "sales","salary"]

numerical_features = [f for f in features if f not in categorical_features]

target = "left"



labels = {

    "satisfaction_level" : "Employee Satisfaction Level (0 - 1)",

    "last_evaluation" : "Evaluation of Employee Performance (0 - 1)",

    "number_project" : "Number of Projects Completed At Work",

    "average_montly_hours" : "Average Monthly Worked Hours",

    "time_spend_company" : "Employment Duration by Number of Years",

    "Work_accident" : "Had a Workplace Accident?",

    "left" : "Left or Stayed In The Company?",

    "promotion_last_5years" : "Has been promoted in the last 5 years?", 

    "sales" : "Department",

    "salary" : "Level of Salary"

}
from math import ceil



def add_new_features(data):

    data["total_tenure_by_hours"] = data["average_montly_hours"].multiply(12).multiply(data["time_spend_company"])

    data["hours_per_project"] = data["total_tenure_by_hours"].divide(data["number_project"])

    data["hours_per_day"] = data["average_montly_hours"].divide(24).apply(ceil)

    return ["total_tenure_by_hours","hours_per_project","hours_per_day"]



counts = ["number_project", "time_spend_company"]



add_new_features(data)

numerical_features += ["total_tenure_by_hours", "hours_per_project"]

counts += ["hours_per_day"]
data[categorical_features].head()
data[numerical_features].head()
from seaborn import regplot

from matplotlib.pyplot import show, subplot, figure, legend

from itertools import combinations
selected_features = [n for n in numerical_features if n not in counts]
ceil((len(selected_features) * 2 - 1)/2) * 100
from matplotlib.pyplot import title, tight_layout

index = 1

figure(figsize=(12.5,ceil((len(selected_features) * 2 - 1)/2)*7))

for i,j in combinations(selected_features, 2):

    subplot( ceil((len(selected_features) * 2 - 1)/2), 2, index)

    regplot(data=data[data[target] == 0], x = i, y = j, fit_reg=False, marker=".", color="silver", label="stayed")

    regplot(data=data[data[target] == 1], x = i, y = j, fit_reg=False, marker="+", color="red", label="left")

    title("Fig %s." % index)

    index += 1



#legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=2)

tight_layout()

show()
from seaborn import distplot

distplot(data[data["left"] == 1]["satisfaction_level"], kde=False, color="red")

show()
leavers = data[data[target] == 1]
from sklearn.cluster import KMeans



cluster_algorithm = KMeans(3)

cluster_algorithm.fit(leavers[selected_features[0:2]])
leavers = leavers.assign(ClusterGroup = cluster_algorithm.labels_)
from matplotlib.pyplot import xlabel, ylabel



index = 1

figure(figsize=(12.5,17))

for i,j in combinations(selected_features, 2):

    

    subplot(ceil((len(selected_features) * 2 - 1)/2), 2, index)

    regplot(data=leavers[leavers["ClusterGroup"] == 0], x = i, y = j, fit_reg=False, marker=".", color="blue", label="0")

    regplot(data=leavers[leavers["ClusterGroup"] == 1], x = i, y = j, fit_reg=False, marker="+", color="red", label="1")

    regplot(data=leavers[leavers["ClusterGroup"] == 2], x = i, y = j, fit_reg=False, marker="x", color="yellow", label="2")

    

    try: 

        xlabel(labels[i]) 

    except: 

        pass

    

    try: 

        ylabel(labels[j]) 

    except: 

        pass

    

    index += 1



legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=2)



show()
from seaborn import countplot
index = 1

discretes = categorical_features + counts

figure(figsize=(12.5,len(discretes)*6))

for c in discretes:

    subplot(ceil((len(discretes) * 2 - 1)/2), 2,index)

    ax = countplot(data=leavers, x=c, hue="ClusterGroup", palette=["b","r","yellow"])

    try:

        ax.set_xlabel(labels[c])

    except:

        pass

    if c == "sales": ax.set_xticklabels(ax.get_xticklabels(),rotation = 20)

    index += 1

show()
from seaborn import pointplot
for n in selected_features:

    index = 1

    figure(figsize=(12.5,len(discretes)*6))

    for c in discretes:

        subplot(ceil((len(discretes) * 2 - 1)/2), 2,index)

        ax = pointplot(data=leavers, y=n, x=c, hue="ClusterGroup", capsize=0.1, palette=["b","r","yellow"], markers="s")

        ax.legend().set_visible(False)

        if c == "sales": ax.set_xticklabels(ax.get_xticklabels(),rotation = 20)

        index += 1

    show()