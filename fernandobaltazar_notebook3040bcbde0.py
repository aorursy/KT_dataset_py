import numpy as np

import pandas as pd

pd.set_option('display.max.columns', 100)

# to draw pictures in jupyter notebook

%matplotlib inline 

import matplotlib.pyplot as plt

import seaborn as sns

# we don't like warnings

# you can comment the following 2 lines if you'd like to

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/uci-adult/adult.data.csv')

data.head(3)
# men = (data["sex"] == "Male").sum()

# Which way are better?

num_each_sex = data.groupby("sex").sex.count()

num_each_sex
# average = data.groupby("sex").age.mean()

average_women = round(data[data.sex == "Female"].age.mean() , 2)

average_women
100 * (((data["native-country"] == "Germany").sum())/ data["native-country"].count())
mean_std_age = data.groupby("salary").age.agg(["mean","std"])

mean_std_age
plt.figure(figsize=(20,6))

cou = data[data["salary"] == ">50K"].education.value_counts()



distribution_bachelor = cou.loc[["Bachelors", "Masters", "Prof-school","Assoc-voc", "Doctorate", "Assoc-acdm"]].sum()

distribution_HS = cou.loc[["HS-grad", "Some-college", "10th", "11th", "7th-8th", "12th", "9th", "5th-6th", "1st-4th"]].sum()



salary_education = pd.Series([distribution_bachelor, distribution_HS], index=["More than HS", "At must HS"])



salary_education.plot.pie(figsize=(5, 5))
data.groupby(["race", "sex"]).age.describe()
marital_status_more_50k = data[data["salary"] == ">50K"]["marital-status"].value_counts()

married_more_50k = marital_status_more_50k.loc[["Married-civ-spouse","Married-spouse-absent","Married-AF-spouse"]].sum()



# marital_status_more_50k.sum() , married_more_50k



marital_status_50k_distribution = pd.Series([married_more_50k, (marital_status_more_50k.sum()-married_more_50k)], index=["Married" , "No married"])



marital_status_50k_distribution.plot.pie(figsize=(5,5))
max_hours = data["hours-per-week"].max()

many_people_max = (data["hours-per-week"] == max_hours).sum()

percent_50k = round(100 * (((data["hours-per-week"] == max_hours) & (data["salary"] == ">50K")).sum())/many_people_max , 2)

print("Max number of hours: ", max_hours , "\nPeople who works the max number of hours: ", many_people_max, 

      "\nPercent of people who works the max number of hours and earns a lot: ", percent_50k)
mean_all_countries = data.groupby(["native-country" , "salary"])["hours-per-week"].mean()

round(mean_all_countries,2)
round(mean_all_countries["Japan"],2)