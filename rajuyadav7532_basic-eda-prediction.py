#Import helping packages 
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import dicom
import os
import scipy.ndimage
base_path="../input/osic-pulmonary-fibrosis-progression"# basepath
os.listdir(base_path)#list of folder 
df=pd.read_csv(os.path.join(base_path,"train.csv"))
print("Columns of train ",df.columns.values)##columns values 
print(df['SmokingStatus'].unique()) # Uniques values SmokingStatus
print("*"*100)
print("Missing values in train ",df.isna().sum())
print("*"*100)
print("Missing values in train ",df.isnull().sum())
print("*"*100)
plt.figure(figsize=(15,8))
sns.countplot(x ='Sex',hue = "SmokingStatus" ,data = df)
plt.show()
plt.figure(figsize=(15,8))
sns.violinplot(y='Age', data=df)
plt.show()
plt.figure(figsize=(35,15))
sns.violinplot(y='FVC',x='Age',hue = "SmokingStatus",data = df)
plt.show()
df
