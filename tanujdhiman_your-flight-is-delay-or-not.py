import pandas as pd
import numpy as np
data_2019 = pd.read_csv('../input/flight-delay-prediction/Jan_2019_ontime.csv')
data_2020 = pd.read_csv('../input/flight-delay-prediction/Jan_2020_ontime.csv')
print("\t\t____________________Data of Jan 2019 Flight Delay_______________________")
print()
print(data_2019.head())
print("_________________________________________________________________________________________________________")
print()
print("\t\t____________________Data of Jan 2020 Flight Delay_______________________")
print()
print(data_2020.head())
print('\t\t____________________Infomation of Dataset Jan 2019____________________')
print()
print(data_2019.info())
print("_________________________________________________________________________________________________________")
print()
print('\t\t____________________Inforamtion of Dataset Jan 2020___________________')
print()
print(data_2020.info())
cols_2019 = data_2019.columns
cols_2020 = data_2020.columns

num_cols_2019 = data_2019._get_numeric_data().columns
num_cols_2020 = data_2020._get_numeric_data().columns

cat_cols_2019 = list(set(cols_2019) - set(num_cols_2019))
cat_cols_2020 = list(set(cols_2020) - set(num_cols_2020))
print("Number of Categorical variables in dataset of Jan 2019 :", len(cat_cols_2019))
print("Number of categorical variables in dataset of Jan 2020 :", len(cat_cols_2020))
print("Integer Variables of dataset 2019 is :", num_cols_2019)
print("Integer Variables of dataset 2020 is :", num_cols_2020)
print("Categorical Variables of dataset 2019 is :", cat_cols_2019)
print("Categorical Variables of dataset 2020 is :", cat_cols_2020)
print("\t\t________________________NULL Values of Dataset Jan 2019________________________")
print()
print(data_2019.isnull().sum())
print()
print("\t\t_______________________Null Values of Dataset Jan 2020________________________")
print()
print(data_2020.isnull().sum())
import matplotlib.pyplot as plt
import seaborn as sns
print("____________________Plotting of Dataset Jan 2019________________________")
data_2019.hist(figsize= (15, 14))
print("____________________Plotting of Dataset Jan 2020________________________")
data_2020.hist(figsize= (15, 14))
plt.figure(figsize = (18, 16))
sns.heatmap(data_2019.corr(), annot = True, cmap = 'coolwarm')
plt.show()
plt.figure(figsize = (18, 16))
sns.heatmap(data_2020.corr(), annot = True, cmap = 'coolwarm', center = 0)
plt.show()