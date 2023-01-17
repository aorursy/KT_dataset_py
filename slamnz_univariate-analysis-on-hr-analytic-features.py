import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
#Instantiate dataset



data = pd.read_csv("../input/HR_comma_sep.csv")
#Global Variables



features = data.drop("left",1).columns

categorical_features = ["promotion_last_5years", "Work_accident", "sales","salary"]

numerical_features = [f for f in features if f not in categorical_features]

count_features = ["number_project", "time_spend_company"]

target = ["left"]



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
data[categorical_features].head()
data[numerical_features].head()
from seaborn import countplot, set_style,despine, axes_style

from matplotlib.pyplot import show

from IPython.display import display

from pandas import DataFrame



def category_analysis(series):

    

    set_style("whitegrid")

    

    with axes_style({'axes.grid': False}):

        cp = countplot(series)

        cp.set_title(cp.get_xlabel())

        cp.set_xlabel("",visible=False)

        despine()

    

    show()

    display(DataFrame(series.value_counts().apply(lambda x: x / len(data) * 100).round(2)).T)
for category in categorical_features + target + count_features:

    category_analysis(data[category])
from seaborn import distplot, boxplot

from matplotlib.pyplot import subplot



def numeric_analysis(series):

    

    no_nulls = series.dropna()

    

    with axes_style({"axes.grid": False}):

        

        cell_1 = subplot(211)

        dp = distplot(no_nulls, kde=False)

        dp.set_xlabel("",visible=False)

        dp.set_yticklabels(dp.get_yticklabels(),visible=False)

        despine(left = True)



        cell_2 = subplot(212, sharex=cell_1)

        boxplot(no_nulls)

        despine(left=True)

    

    show()

    

    display(DataFrame(series.describe().round(2)).T)
for n in numerical_features:

    numeric_analysis(data[data[n].notnull()][n])
from math import ceil



def add_new_features(data):

    data["total_tenure_by_hours"] = data["average_montly_hours"].multiply(12).multiply(data["time_spend_company"])

    data["hours_per_project"] = data["total_tenure_by_hours"].divide(data["number_project"])

    data["hours_per_day"] = data["average_montly_hours"].divide(24).apply(ceil)

    return ["total_tenure_by_hours","hours_per_project","hours_per_day"]
new = add_new_features(data)
for n in new:

    numeric_analysis(data[n])
category_analysis(data["hours_per_day"])