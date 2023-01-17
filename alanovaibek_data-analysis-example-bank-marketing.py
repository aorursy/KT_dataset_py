# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import skew
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
%matplotlib inline
njobs = 4
randomState = 0
# Get data
data = pd.read_csv("../input/bank-additional-full.csv", sep = ",")
display(data.head())
print("Data dimensions : " + str(data.shape))
print("Data columns : " + str(data.columns))
# Remove dots from column names
data.columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 
                'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx',
                'cons_conf_idx', 'euribor3m', 'nr_employed', 'y']
# Quick look at the target variable
print("Target variable 'y' is yes or no -> classification problem")
print("Number of yes : " + str(data[data.y == "yes"].shape[0]))
print("Number of no : " + str(data[data.y == "no"].shape[0]))

# Encode it as integer for machine learning algorithms
data = data.replace({"y" : {"no" : 0, "yes" : 1}})
# age
data.age.hist(bins = 50)
print("NAs for age : " + str(data.age.isnull().values.sum()))
# job
print(data.job.value_counts())
print("NAs for job : " + str(data.job.isnull().values.sum()))
print("330 'unknown', impute most common value")
data.loc[data["job"] == "unknown", "job"] = "admin."
# marital
print(data.marital.value_counts())
print("NAs for marital : " + str(data.marital.isnull().values.sum()))
print("80 'unknown', impute most common value")
data.loc[data["marital"] == "unknown", "marital"] = "married"
# education
print(data.education.value_counts())
print("NAs for education : " + str(data.education.isnull().values.sum()))
print("1731 'unknown', impute most common value")
data.loc[data["education"] == "unknown", "education"] = "university.degree"
print("basic.4y : left school at 10 years old")
print("basic.6y : left school at 12 years old")
print("basic.9y : left school at 15 years old")
# default
print(data.default.value_counts())
print("NAs for default : " + str(data.default.isnull().values.sum()))
print("8597 'unknown'")
print("Only 3 'yes' -> we'll discard this variable, not enough information in it")
data = data.drop(["default"], axis = 1)
# housing
print(data.housing.value_counts())
print("NAs for housing : " + str(data.housing.isnull().values.sum()))
print("990 'unknown'. Since we have about same proportion of yes and no, let's impute NAs randomly")
data.loc[data["housing"] == "unknown", "housing"] = random.choice(["yes", "no"])
# month
print(data.month.value_counts())
print("NAs for month : " + str(data.month.isnull().values.sum()))
# day_of_week
print(data.day_of_week.value_counts())
print("NAs for day_of_week : " + str(data.day_of_week.isnull().values.sum()))
# duration
data.duration.hist(bins = 50)
print("NAs for duration : " + str(data.duration.isnull().values.sum()))
print("Important note:  this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the \
duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input \
should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive \
model.")
print("Removing it from data")
data = data.drop(["duration"], axis = 1)
# campaign
print(data.campaign.value_counts())
data.campaign.hist(bins = 50)
print("NAs for campaign : " + str(data.campaign.isnull().values.sum()))
# Encode some categorical features as ordered numbers when there is information in the order
data = data.replace({"education" : {"illiterate" : 1, "basic.4y" : 2, "basic.6y" : 3, "basic.9y" : 4, "high.school" : 5, 
                                    "professional.course" : 6, "university.degree" : 7}})
# Create new features
# 1* Simplifications of existing features
data["smplJob"] = data.job.replace({"retired" : 1, "student" : 1, "unemployed" : 1,
                                    "admin." : 2, "blue-collar" : 2, "entrepreneur" : 2, "housemaid" : 2, 
                                    "management" : 2, "self-employed" : 2, "services" : 2, "technician" : 2
                                   })# 1 : not working, 2 : working
data["smplMarital"] = data.marital.replace({"married" : 1, 
                                            "single" : 2, "divorced" : 2 
                                           })# 1 : married, 2 : not married                                                    
data["smplEducation"] = data.education.replace({1 : 1, 2 : 1, 3 : 1, 4 : 1,
                                                5 : 2, 
                                                6 : 3, 7 : 3
                                   })# 1 : noEduc, 2 : HS, 3 : superior
data["smplMonth"] = data.month.replace({"dec" : 1, "jan" : 1, "feb" : 1,
                                        "mar" : 2, "apr" : 2, "may" : 2, 
                                        "jun" : 3, "jul" : 3, "aug" : 3,
                                        "sep" : 4, "oct" : 4, "nov" : 4
                                   })# 1 : winter, 2 : spring, 3 : summer, 4 : fall
data["smplAge"] = data.age.copy()
data.loc[data["age"] < 30, "smplAge"] = 1
data.loc[(data["age"] >= 30) & (data["age"] < 40), "smplAge"] = 2
data.loc[(data["age"] >= 40) & (data["age"] < 50), "smplAge"] = 3
data.loc[(data["age"] >= 50) & (data["age"] < 60), "smplAge"] = 4
data.loc[data["age"] >= 60, "smplAge"] = 5
data["smplCampaign"] = data.campaign.copy()
data.loc[data["campaign"] == 1, "smplCampaign"] = 1
data.loc[data["campaign"] == 2, "smplCampaign"] = 2
data.loc[(data["campaign"] >= 3) & (data["campaign"] <= 5), "smplCampaign"] = 3
data.loc[data["campaign"] >= 6, "smplCampaign"] = 4
data["smplPdays"] = data.pdays.copy()
data.loc[data["pdays"] == 999, "smplPdays"] = 1 # never contacted
data.loc[data["pdays"] < 7, "smplPdays"] = 2 # contacted less than a week ago (5 working days)
data.loc[(data["pdays"] >= 7) & (data["pdays"] < 14), "smplPdays"] = 3 # contacted less than 2 weeks ago
data.loc[data["pdays"] >= 14, "smplPdays"] = 4
data["smplPrevious"] = data.previous.copy()
data.loc[data["previous"] == 0, "smplPrevious"] = 1
data.loc[data["previous"] == 1, "smplPrevious"] = 2
data.loc[data["previous"] >= 2, "smplPrevious"] = 3
data["smplEmp_var_rate"] = data.emp_var_rate.copy()
data.loc[data["emp_var_rate"] <= -1, "smplEmp_var_rate"] = 1
data.loc[(data["emp_var_rate"] > -1) & (data["emp_var_rate"] < 1), "smplEmp_var_rate"] = 2
data.loc[data["emp_var_rate"] >= 1, "smplEmp_var_rate"] = 3
data["smplEuribor3m"] = data.euribor3m.copy()
data.loc[data["euribor3m"] <= 3, "smplEuribor3m"] = 1
data.loc[data["euribor3m"] > 3, "smplEuribor3m"] = 2
data["smplNr_employed"] = data.nr_employed.copy()
data.loc[data["nr_employed"] <= 5050, "smplNr_employed"] = 1
data.loc[(data["nr_employed"] > 5050) & (data["nr_employed"] < 5150), "smplNr_employed"] = 2
data.loc[data["nr_employed"] >= 5150, "smplNr_employed"] = 3
# 2* Combinations of existing features
# Combo of age and education should be a good proxy for wealth
data["smplAge*education"] = data["smplAge"] * data["education"]
print(data.shape)
display(data.head())
# Find most important features relative to target
print("Find most important features relative to target")
corr = abs(data.corr())
corr.sort_values(["y"], ascending = False, inplace = True)
print(corr.y)
print("No strong predictor so far")