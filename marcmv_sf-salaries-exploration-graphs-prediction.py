#Import necessary packages

import os

import pandas as pd

import matplotlib.pyplot as plt

#use notebook visualization

%matplotlib inline 
df_salariesInput = pd.read_csv("../input/Salaries.csv",low_memory=False)
#Review what has been imported

df_salariesInput.head()
#Extract name value

df_salariesInput["Name"] = None if df_salariesInput["EmployeeName"].empty else df_salariesInput["EmployeeName"].str.split(expand = True)[0]



#Extract surname value

df_salariesInput["Surname"] = None if df_salariesInput["EmployeeName"].empty else df_salariesInput["EmployeeName"].str.split(expand = True)[1]
df_salariesInput.head()
df_salariesInput.drop(["Id", "Notes", "Agency"], axis = 1, inplace = True)

df_salariesInput.head()
df_salariesInput["Status"].value_counts()
df_salariesInput[["Year","TotalPay", "TotalPayBenefits"]].groupby("Year").mean()
df_salariesInput.describe()
df_salariesInput.info()
#Convert to float

df_salariesInput["BasePay"] = pd.to_numeric(df_salariesInput['BasePay'], errors='coerce').fillna(0)

df_salariesInput["OvertimePay"] = pd.to_numeric(df_salariesInput['OvertimePay'], errors='coerce').fillna(0)

df_salariesInput["OtherPay"] = pd.to_numeric(df_salariesInput['OtherPay'], errors='coerce').fillna(0)

df_salariesInput["Benefits"] = pd.to_numeric(df_salariesInput['Benefits'], errors='coerce').fillna(0)
df_salariesInput.describe()
df_salariesInput.head()
df_salariesInput["JobTitle"].value_counts().head(15)
def salaryEvolutionOf(jobTitle):

    variableJobTitleLowerCase = jobTitle.lower()

    df_subset = df_salariesInput

    df_subset["JobTitleLowercase"] = df_salariesInput["JobTitle"].str.lower()

    df_subset = df_subset.loc[df_subset["JobTitleLowercase"] == variableJobTitleLowerCase]

    if df_subset.empty: 

        print("Lower case title: ", variableJobTitleLowerCase)

        print("We have identified " , len(df_subset), " matches")

        return False

    plotTableMean = df_subset[["Year", "BasePay", "TotalPay", "TotalPayBenefits"]].groupby("Year").mean()

    plotTableMean.plot(kind="bar", title = "Mean")

    plotTableMedian = df_subset[["Year", "BasePay", "TotalPay", "TotalPayBenefits"]].groupby("Year").median()

    plotTableMedian.plot(kind="bar", title = "Median")

    print("Found " , len(df_subset), " matches for ", variableJobTitleLowerCase)

    return True
salaryEvolutionOf("Transit Operator")
df_salariesInput.hist(bins=50, figsize=(20,15))
#BoxPlot Data

y2011 = df_salariesInput.loc[df_salariesInput["Year"] == 2011]

year2012 = df_salariesInput[df_salariesInput.Year == 2012]

year2013 = df_salariesInput[df_salariesInput.Year == 2013]

year2014 = df_salariesInput[df_salariesInput.Year == 2014]





plt.figure(figsize=(10,5))

plt.boxplot([y2011["TotalPay"], year2012.TotalPay, year2013.TotalPay, year2014.TotalPay])

plt.ylim(0, 200000) #its the limit of y

plt.title('TotalPay - Boxplot')

plt.xticks([1, 2, 3, 4], ['2011', '2012', '2013', "2014"])

plt.tight_layout()
#Based on the name, the title and the year I want to know the expected total pay with benefits



#Convert string to numbers using LabelEncoder

from sklearn.preprocessing import LabelEncoder

encoderJobTitle = LabelEncoder()

df_salariesInput["JobTitleLowercase"] = encoderJobTitle.fit_transform(df_salariesInput["JobTitleLowercase"] )

df_salariesInput.head()
encoderName = LabelEncoder()

df_salariesInput["Name"] = encoderJobTitle.fit_transform(df_salariesInput["Name"] )

df_salariesInput.head()
#Split data set

from sklearn.model_selection import train_test_split

df_train_set, df_test_set = train_test_split(df_salariesInput, test_size = 0.2, random_state = 42)

print("Splitted sets: ",len(df_train_set), "train +", len(df_test_set), "test")

#Correlation

corr_matrix = df_train_set.corr()

corr_matrix
X_test = df_test_set[["JobTitleLowercase", "Year","Name"]]

X_train = df_train_set[["JobTitleLowercase", "Year","Name"]]

X_train.head()
y_test = df_test_set["TotalPayBenefits"]

y_train = df_train_set["TotalPayBenefits"]

y_train.head()
from sklearn.ensemble import RandomForestRegressor



clf = RandomForestRegressor()

clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)

pred_test = clf.predict(X_test)



accuracy_train = clf.score(X_train, y_train)

accuracy_test = clf.score(X_test, y_test)



print("Accuracy train -> ", accuracy_train)

print("Accuracy test -> ", accuracy_test)