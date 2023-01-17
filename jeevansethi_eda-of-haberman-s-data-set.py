#import useful packages for EDA (Exploratory Data Analysis)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Reading Haberman's Survival data set file (.csv)
#read_csv() is used to read csv file
#using proper path as defined i.e '../input/haberman.csv'
path='../input/haberman.csv'
sample_data=pd.read_csv(path)
sample_data.columns=['AGE','SURGERY_YEAR','NODES_DETECTED','STATUS']
print(sample_data.head())
print(sample_data.tail())
print(sample_data.shape[0])
print(sample_data.shape[1])
#value_counts() is used to count distinct values
print(sample_data['STATUS'].value_counts())
sns.set_style('whitegrid')
sns.FacetGrid(sample_data,hue='STATUS',size=5).map(sns.distplot,'AGE').add_legend()
plt.show()
sns.set_style('whitegrid')
sns.FacetGrid(sample_data,hue='STATUS',size=5).map(sns.distplot,'SURGERY_YEAR').add_legend()
plt.show()
sns.set_style('whitegrid')
sns.FacetGrid(sample_data,hue='STATUS',size=5).map(sns.distplot,'NODES_DETECTED').add_legend()
plt.show()