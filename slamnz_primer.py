import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
#Instantiate dataset



data = pd.read_csv("../input/HR_comma_sep.csv")
#Global Variables



features = data.drop("left",1).columns

categorical_features = ["promotion_last_5years", "Work_accident", "sales","salary"]

numerical_features = [f for f in features if f not in categorical_features]

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