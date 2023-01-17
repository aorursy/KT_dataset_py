%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mplt
data_train = '../input/liver_train.csv'
data_test =  '../input/liver_test.csv'
data = pd.read_csv(data_train)
data.head()
print(list(data['Result']).count(0))
print(list(data['Result']).count(1))
data.describe()
data.isna().sum()
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data.isna().sum()
data.drop(columns=['Id'], inplace=True)
data.describe()
data
result = data['Result']
features_raw = data.drop(columns=['Result'])
result.head()
features_raw.head()
clr = result.apply(lambda x : 'r' if x == 1 else 'b')
numerical = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alpha-1_Antitrypsin', 'Blood_Urea_Nitrogen', 'Alpha_Galactosidase', 'Globulin', 'Total_Protein', 'Alpha_Feto_Protein', 'Albumin']
plt.hist(data[numerical[1]])
plt.hist(data[numerical[2]])
plt.hist(data[numerical[3]])
plt.hist(data[numerical[4]])
plt.hist(data[numerical[5]])
plt.hist(data[numerical[6]])
plt.hist(data[numerical[7]])
plt.hist(data[numerical[8]])
plt.hist(data[numerical[9]])
skewed = ['Total_Bilirubin', 'Direct_Bilirubin', 'Alpha-1_Antitrypsin', 'Blood_Urea_Nitrogen', 'Alpha_Galactosidase', 'Alpha_Feto_Protein']
plt.hist(result)
























