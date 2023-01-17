# Call Libraries

import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

import warnings

import os
warnings.filterwarnings("ignore") # To ignore errors- library=>warnings
# Accessing data folder

#os.chdir("E:\\Python\\Excercise\\Jupiter")

os.chdir("../input")

os.listdir()
# Accessing Data



AttrData = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv",header = 0)

AttrData.shape
# Checking data:

AttrData

pd.isnull(AttrData).values.any() # to check any null values
AttrData.size #Memory Occupied
AttrData.ndim #Number of dimensions

AttrData.columns.values
AttrData.nunique().sort_values()
AttrDataUniCol = AttrData.nunique().sort_values(ascending=False) #Sort Descending

AttrDataUniCol
AttrDatNumCol = AttrDataUniCol[AttrDataUniCol>900].index.values

AttrDatNumCol
# Slicing Non-numeric data for better understanding

AttrData["PercentSalaryHike"].unique()

pd.value_counts(AttrData["PercentSalaryHike"])
SalBins = [10,15,20,25]

AttrData["PercentSalaryHike_Cat"] = pd.cut(AttrData["PercentSalaryHike"],SalBins, labels = ['Low','Medium','High'])

pd.value_counts(AttrData["PercentSalaryHike_Cat"])
np.max(AttrData['DailyRate'])
np.min(AttrData['DailyRate'])
AttrData['DailyRateSlab'] = pd.cut(AttrData["DailyRate"],8, labels = ['D1','D2','D3','D4','D5','D6','D7','D8'])
pd.value_counts(AttrData['DailyRateSlab'])
pd.value_counts(AttrData['HourlyRate'])

np.max(AttrData['HourlyRate'])

np.min(AttrData['HourlyRate'])
AttrData['HourlyRateSlab'] = pd.cut(AttrData["HourlyRate"],4, labels = ['H1','H2','H3','H4'])
pd.value_counts(AttrData['HourlyRateSlab'])
np.max(AttrData['Age'])

np.min(AttrData['Age'])

pd.value_counts(AttrData['Age'])

AgeBins=[17,21,32,45,100]

AttrData['Age_Levels'] = pd.cut(AttrData["Age"],AgeBins, labels = ['Teen','Young','Middle-Age','Old'])

pd.value_counts(AttrData['Age_Levels'])
np.max(AttrData['TotalWorkingYears'])

np.min(AttrData['TotalWorkingYears'])

pd.value_counts(AttrData['TotalWorkingYears'])

WorkingYearBins = [-1,5,12,21,42]

AttrData['Working_Levels'] = pd.cut(AttrData["TotalWorkingYears"],WorkingYearBins, labels = ['Fresher','Junior','Senior','Super-Senior'])

pd.value_counts(AttrData['Working_Levels'])
np.max(AttrData['YearsAtCompany'])

np.min(AttrData['YearsAtCompany'])

pd.value_counts(AttrData['YearsAtCompany'])

LoyaltyIndex = [-1,5,10,25,40]

AttrData['Loyalty_Index'] = pd.cut(AttrData["YearsAtCompany"],LoyaltyIndex, labels = ['Opportunists','Loyalists','Extreme Loyalists','Kattappas'])

pd.value_counts(AttrData['Loyalty_Index'])
np.max(AttrData['DistanceFromHome'])

np.min(AttrData['DistanceFromHome'])

pd.value_counts(AttrData['DistanceFromHome'])

DistanceBins = [0,3,8,16,32]

AttrData['Distance_Level'] = pd.cut(AttrData["DistanceFromHome"],DistanceBins, labels = ['Near-by','Near','Mid','Far'])

pd.value_counts(AttrData['Distance_Level'])
np.max(AttrData['YearsInCurrentRole'])

np.min(AttrData['YearsInCurrentRole'])

pd.value_counts(AttrData['YearsInCurrentRole'])

CurrRoleBins = [-1,4,10,20]

AttrData['CurrentRole_Level'] = pd.cut(AttrData["YearsInCurrentRole"],CurrRoleBins, labels = ['New','Not New','Old'])

pd.value_counts(AttrData['CurrentRole_Level'])
np.max(AttrData['YearsWithCurrManager'])

np.min(AttrData['YearsWithCurrManager'])

pd.value_counts(AttrData['YearsWithCurrManager'])

AttrData['YrsWithManager_Level'] = pd.cut(AttrData["YearsWithCurrManager"],3, labels = ['New','Not New','Old'])

pd.value_counts(AttrData['YrsWithManager_Level'])
np.max(AttrData['YearsSinceLastPromotion'])

np.min(AttrData['YearsSinceLastPromotion'])

pd.value_counts(AttrData['YearsSinceLastPromotion'])

AttrData['YearsSinceLastPromotion_Lvl'] = pd.cut(AttrData["YearsSinceLastPromotion"],3, labels = ['Low','Medium','High'])

pd.value_counts(AttrData['YearsSinceLastPromotion_Lvl'])
AttrData.info()

pd.isnull(AttrData).values.any() # to check any null values

AttrData.head()
df = AttrData[["Gender","Attrition"]]

AttrData.groupby(["Gender","Attrition"]).size()

AttrData.groupby(["Gender","Attrition"]).size().unstack()
[pd.value_counts(df.Gender),pd.value_counts(df.Attrition)]

AttrData.groupby(["Gender","Attrition"]).sum()
sb.heatmap(AttrData.groupby(["Gender","Attrition"]).size().unstack(),annot=True,fmt='d')#fmt--> gives the exact string value

plt.show() 
#Creating definition :

def SRKHeatPlot(X):

    df = AttrData.groupby([X,"Attrition"]).size().unstack()

    sb.heatmap(df,annot=True,fmt = 'f') # float values

    plt.show()
colNames = ['DailyRate','HourlyRate','Age','TotalWorkingYears','YearsAtCompany','DistanceFromHome','YearsInCurrentRole','YearsWithCurrManager','YearsSinceLastPromotion','PercentSalaryHike','NumCompaniesWorked','JobRole','TrainingTimesLastYear','EducationField','JobLevel','Education','WorkLifeBalance','RelationshipSatisfaction','JobSatisfaction','JobInvolvement','EnvironmentSatisfaction','StockOptionLevel','BusinessTravel','Department','MaritalStatus','OverTime','PerformanceRating','Attrition','Gender','EmployeeCount','StandardHours','Over18']

ClassifiedColNames = ['DailyRateSlab','HourlyRateSlab','Age_Levels','Working_Levels','Loyalty_Index','Distance_Level','CurrentRole_Level','YrsWithManager_Level','YearsSinceLastPromotion_Lvl','PercentSalaryHike_Cat','NumCompaniesWorked','JobRole','TrainingTimesLastYear','EducationField','JobLevel','Education','WorkLifeBalance','RelationshipSatisfaction','JobSatisfaction','JobInvolvement','EnvironmentSatisfaction','StockOptionLevel','BusinessTravel','Department','MaritalStatus','OverTime','PerformanceRating','Attrition','Gender']



# Heat Plots to find the Attrition w.r.t various parameters :



for i in ClassifiedColNames:

    SRKHeatPlot(i)
# From the graphs it seems that below are the impacted parameters w.r.t attribution 'Yes':

# when Performance rating is 3

# People who does Overtime

# Marital Status - Single

# Who travel frequently

# whose joInvovement, job satisfaction, workife balance is 1

# For freshers and agelevels-teen and young
sb.jointplot(AttrData.Age,AttrData.MonthlyIncome, kind = "scatter")

plt.show()
sb.jointplot(AttrData.EmployeeNumber,AttrData.MonthlyIncome, kind = "hex")

plt.show()
sb.jointplot(AttrData.Age,AttrData.MonthlyIncome, kind = "hex")

plt.show()
sb.swarmplot(AttrData.Age,AttrData.MonthlyIncome)

plt.show()
sb.swarmplot(AttrData.EmployeeNumber,AttrData.MonthlyIncome)

plt.show()
sb.boxplot(AttrData.Attrition,AttrData.MonthlyIncome)

plt.show()
sb.boxplot(AttrData.Attrition,AttrData.TotalWorkingYears)

plt.show()