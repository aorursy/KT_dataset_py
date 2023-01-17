

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plots

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os
print(os.listdir("../input"))



##
### TODO EDA. Exploration. Cleaning. Put in some graphs. Do this before 11/27
### TODO MODELING. Do this before 11/27
##Test/holdout data. Save10% of data to calculate metrics
##

# Any results you write to the current directory are saved as output.
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'] #1970

data = pd.read_csv("../input/housing.csv", delimiter=r"\s+", names=column_names)



data.isnull().sum()
print(data.head(20))
print(data.columns)
print("MEDV")
data.MEDV.describe()
total_amnt_data = 506
######## Functions for graphs

def crime_by_MedV():
    plt.xlabel("Crime")
    plt.ylabel("Median Value")
    plt.scatter(data.loc[:, "CRIM"], data.loc[:, "MEDV"])
    
def ZN_by_MedV():
    plt.xlabel("Zoning")
    plt.ylabel("Median Value")
    plt.scatter(data.loc[:, "ZN"], data.loc[:, "MEDV"])

def NOX_by_MedV():
    plt.xlabel("Nitric Oxides Concentration")
    plt.ylabel("Median Value")
    plt.scatter(data.loc[:,"NOX"], data.loc[:, "MEDV"])

def RM_by_MedV():
    plt.xlabel("AVG rooms per dwelling")
    plt.ylabel("Median Value")
    plt.scatter(data.loc[:,"RM"], data.loc[:, "MEDV"])
    
def DIS_by_MedV():
    plt.xlabel("AVG Distance to Work")
    plt.ylabel("Median Value")
    plt.scatter(data.loc[:,"DIS"], data.loc[:, "MEDV"])
    
def RAD_by_MedV():
    plt.xlabel("Accessibility to highways")
    plt.ylabel("Median Value")
    plt.scatter(data.loc[:,"RAD"], data.loc[:, "MEDV"])
    
def TAX_by_MedV():
    plt.xlabel("Tax rate per $10,000")
    plt.ylabel("Median Value")
    plt.scatter(data.loc[:,"TAX"], data.loc[:, "MEDV"])
    
def LSTAT_by_MedV():
    plt.xlabel("% of lower working class of POP")
    plt.ylabel("Median Value")
    plt.scatter(data.loc[:,"LSTAT"], data.loc[:, "MEDV"])

######
def indus_by_MedV():
    plt.scatter(data.loc[:, "INDUS"], data.loc[:, "MEDV"])
    plt.xlabel("Industrial")
    plt.ylabel("Median Value")
    plt.show()
######
def B_by_MedV():
    print("X, B /// Y, MEDV")
    #plt.yscale("Log")
    plt.xlabel("Black ratio by formula")
    plt.ylabel("Median Value")
    plt.scatter(data.loc[:, "B"], data.loc[:, "MEDV"])
    plt.show()

def B_by_crime():
    print("X, B /// Y, CRIM")
    plt.scatter(data.loc[:, "B"], data.loc[:, 'CRIM'])
    plt.show()
plt.scatter(data.loc[:, "CHAS"], data.loc[:, "MEDV"])
plt.xlabel("0 = not by a river    1= By a river")
plt.ylabel("Median Value")
plt.show()
LSTAT_by_MedV()
plt.show()
RM_by_MedV()
plt.show()
DIS_by_MedV()
plt.show()

CHAS_count_1 = np.count_nonzero(data.loc[:,"CHAS"] == 1)
print("amount of neighborhoods by a river: " + str(CHAS_count_1))

CHAS_count_0 = np.count_nonzero(data.loc[:,"CHAS"] == 0)
print("amount of neighborhoods not by a river: " + str(CHAS_count_0))
data.corr(method="pearson")

### Heat Map ###
import seaborn as sns
plt.figure(figsize = (15,10))
sns.heatmap(data.corr(method="pearson"), annot = True)

plt.xlabel("LSTAT")
plt.ylabel("INDUS")
plt.scatter(data.loc[:, "LSTAT"], data.loc[:, "INDUS"])
plt.show()
print(max(data.loc[:,"LSTAT"]))
print(max(data.loc[:,"INDUS"]))
print()

##### AREA 0 1 2 3 / amount,percentage

print(np.count_nonzero( (data.loc[:,"LSTAT"] < 18.985) & (data.loc[:,"INDUS"] < 13.768) ))
print(np.count_nonzero( (data.loc[:,"LSTAT"] < 18.985) & (data.loc[:,"INDUS"] < 13.768) ) / total_amnt_data)
print("Area 0 ^")
print()

print(np.count_nonzero( (data.loc[:,"LSTAT"] < 18.985) & (data.loc[:,"INDUS"] > 13.768) ))
print(np.count_nonzero( (data.loc[:,"LSTAT"] < 18.985) & (data.loc[:,"INDUS"] > 13.768) ) / total_amnt_data)
print("Area 1 ^")
print()

print(np.count_nonzero( (data.loc[:,"LSTAT"] > 18.985) & (data.loc[:,"INDUS"] > 13.768) ))
print(np.count_nonzero( (data.loc[:,"LSTAT"] > 18.985) & (data.loc[:,"INDUS"] > 13.768) ) / total_amnt_data)
print("Area 2 ^")
print()

print(np.count_nonzero( (data.loc[:,"LSTAT"] > 18.985) & (data.loc[:,"INDUS"] < 13.768) ))
print(np.count_nonzero( (data.loc[:,"LSTAT"] > 18.985) & (data.loc[:,"INDUS"] < 13.768) ) / total_amnt_data)
print("Area 3 ^")
print()

def count_percent_area(column_1, column_2):

    print(np.count_nonzero( (data.column_1 < max(data.column_1)/2) & (data.column_2 < max(data.column_2)/2) ))
    print(np.count_nonzero( (data.column_1 < max(data.column_1)/2) & (data.column_2 < max(data.column_2)/2)  )/ total_amnt_data)
    print("Area 0 ^")
    print()

    print(np.count_nonzero( (data.column_1 < max(data.column_1)/2) & (data.column_2 > max(data.column_2)/2) ))
    print(np.count_nonzero( (data.column_1 < max(data.column_1)/2) & (data.column_2 > max(data.column_2)/2) )/ total_amnt_data)
    print("Area 1 ^")
    print()

    print(np.count_nonzero( (data.column_1 > max(data.column_1)/2) & (data.column_2 > max(data.column_2)/2) ))
    print(np.count_nonzero( (data.column_1 > max(data.column_1)/2) & (data.column_2 > max(data.column_2)/2) )/ total_amnt_data)
    print("Area 2 ^")
    print()

    print(np.count_nonzero( (data.column_1 > max(data.column_1)/2) & (data.column_2 < max(data.column_2)/2) ))
    print(np.count_nonzero( (data.column_1 > max(data.column_1)/2) & (data.column_2 < max(data.column_2)/2) )/ total_amnt_data)
    print("Area 3 ^")
    print()

print(np.count_nonzero( (data.loc[:,"LSTAT"] <= max(data.loc[:,"LSTAT"])/2) & (data.loc[:,"MEDV"] <= max(data.loc[:,"MEDV"])/2) ))
print(np.count_nonzero( (data.loc[:,"LSTAT"] <= max(data.loc[:,"LSTAT"])/2) & (data.loc[:,"MEDV"] <= max(data.loc[:,"MEDV"])/2) ) / total_amnt_data)
print("Area 0 ^")
print()

print(np.count_nonzero( (data.loc[:,"LSTAT"] <= max(data.loc[:,"LSTAT"])/2) & (data.loc[:,"MEDV"] >= max(data.loc[:,"MEDV"])/2) ))
print(np.count_nonzero( (data.loc[:,"LSTAT"] <= max(data.loc[:,"LSTAT"])/2) & (data.loc[:,"MEDV"] >= max(data.loc[:,"MEDV"])/2)) / total_amnt_data)
print("Area 1 ^")
print()

print(np.count_nonzero( (data.loc[:,"LSTAT"] >= max(data.loc[:,"LSTAT"])/2) & (data.loc[:,"MEDV"] >= max(data.loc[:,"MEDV"])/2) ))
print(np.count_nonzero( (data.loc[:,"LSTAT"] >= max(data.loc[:,"LSTAT"])/2) & (data.loc[:,"MEDV"] >= max(data.loc[:,"MEDV"])/2)) / total_amnt_data)
print("Area 2 ^")
print()

print(np.count_nonzero( (data.loc[:,"LSTAT"] >= max(data.loc[:,"LSTAT"])/2) & (data.loc[:,"MEDV"] <= max(data.loc[:,"MEDV"])/2) ))
print(np.count_nonzero( (data.loc[:,"LSTAT"] >= max(data.loc[:,"LSTAT"])/2) & (data.loc[:,"MEDV"] <= max(data.loc[:,"MEDV"])/2)) / total_amnt_data)
print("Area 3 ^")
print()
print(np.count_nonzero( (data.loc[:,"RM"] <= max(data.loc[:,"RM"])/2) & (data.loc[:,"MEDV"] <= max(data.loc[:,"MEDV"])/2) ))
print(np.count_nonzero( (data.loc[:,"RM"] <= max(data.loc[:,"RM"])/2) & (data.loc[:,"MEDV"] <= max(data.loc[:,"MEDV"])/2) ) / total_amnt_data)
print("Area 0 ^")
print()

print(np.count_nonzero( (data.loc[:,"RM"] <= max(data.loc[:,"RM"])/2) & (data.loc[:,"MEDV"] >= max(data.loc[:,"MEDV"])/2) ))
print(np.count_nonzero( (data.loc[:,"RM"] <= max(data.loc[:,"RM"])/2) & (data.loc[:,"MEDV"] >= max(data.loc[:,"MEDV"])/2)) / total_amnt_data)
print("Area 1 ^")
print()

print(np.count_nonzero( (data.loc[:,"RM"] >= max(data.loc[:,"RM"])/2) & (data.loc[:,"MEDV"] >= max(data.loc[:,"MEDV"])/2) ))
print(np.count_nonzero( (data.loc[:,"RM"] >= max(data.loc[:,"RM"])/2) & (data.loc[:,"MEDV"] >= max(data.loc[:,"MEDV"])/2)) / total_amnt_data)
print("Area 2 ^")
print()

print(np.count_nonzero( (data.loc[:,"RM"] >= max(data.loc[:,"RM"])/2) & (data.loc[:,"MEDV"] <= max(data.loc[:,"MEDV"])/2) ))
print(np.count_nonzero( (data.loc[:,"RM"] >= max(data.loc[:,"RM"])/2) & (data.loc[:,"MEDV"] <= max(data.loc[:,"MEDV"])/2)) / total_amnt_data)
print("Area 3 ^")
print()
plt.ylabel("Frequency")
plt.xlabel("Median Value in $10,000")
plt.hist(data.loc[:,"MEDV"], bins = 15, normed = True, ec = "black")
plt.boxplot(data.loc[:,"MEDV"])