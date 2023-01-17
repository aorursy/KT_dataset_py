#Importing important Python libraries
import pandas as pd

import numpy as np

import seaborn as sns

import scipy as sp

from scipy.stats import f_oneway
telecom_data = pd.read_csv("../input/Telco-Customer-Churn.csv")
telecom_data.head()
#Bottom 5 rows of Dataset
telecom_data.tail()
#Data types of various features
telecom_data.dtypes
telecom_data.describe()
telecom_data.boxplot()
#Finding correlation of various features with one another
telecom_data.corr()
sp.stats.pearsonr(telecom_data['SeniorCitizen'],telecom_data['MonthlyCharges'])
sp.stats.pearsonr(telecom_data['tenure'],telecom_data['MonthlyCharges'])
plt.scatter(telecom_data['SeniorCitizen'],telecom_data['Churn'])
import matplotlib.pyplot as plt
sp.stats.pearsonr(telecom_data['tenure'],telecom_data['MonthlyCharges'])
import pandas as pd

import numpy as np

import seaborn as sns

import scipy as sp

from scipy.stats import f_oneway
telecom_data = pd.read_csv("../input/Telco-Customer-Churn.csv")
plt.scatter(telecom_data['SeniorCitizen'],telecom_data['Churn'])
plt.scatter(telecom_data['SeniorCitizen'],telecom_data['MonthlyCharges'])
plt.scatter(telecom_data['MonthlyCharges'],telecom_data['Churn'])
plt.scatter(telecom_data['tenure'],telecom_data['Churn'])
#Some scatter plots
sns.residplot(telecom_data['SeniorCitizen'],telecom_data['MonthlyCharges'])
sns.regplot(telecom_data['SeniorCitizen'],telecom_data['MonthlyCharges'])
sns.residplot(telecom_data['tenure'],telecom_data['MonthlyCharges'])
sns.regplot(telecom_data['tenure'],telecom_data['MonthlyCharges'])
#Residual and regression plots to find the strength of correlation between variables
#Positive slope of regression plot show HIGH corelation between SeniorCitizen and MonthlyCHarges and also between tenure and MonthlyCharges
#Suggesstion on what changes to make and how to learn more things to perform an EDA are welcomed.
#This was my first EDA that I have run till date. It might not be good and might not met conditions of a proper EDA on such a dataset. Your comments will help me learn and will support me.  