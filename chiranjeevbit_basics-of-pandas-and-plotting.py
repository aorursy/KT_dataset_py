import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.pyplot as plote
from sklearn import preprocessing
from sklearn import cross_validation, metrics
from sklearn.model_selection import train_test_split
from pylab import plot, show, subplot, specgram, imshow, savefig
# Reading the dataset
train = pd.ExcelFile("../input/research_student (1) (1).xlsx")
dataset = train.parse('Sheet1', header=0)

dataset.head()
dataset = dataset.drop(dataset[dataset['GPA 2']>10].index)

dataset = dataset.dropna()
dataset.head()
target = dataset.CGPA
features = dataset.drop(['CGPA'], axis = 1)
f,ax = plote.subplots(figsize=(18, 18))
sns.heatmap(features.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
features = features.drop(['Rank'], axis=1)
features.describe()
features.iloc[:,0:14].describe()
features.iloc[:,14:].describe()
list1 = []
for item in features.columns:
    item = item.replace("[","")
    item = item.replace("]","")
    list1.append(item)
    
features.columns = list1
list1
#Combining features and CGPA(Target)
vis_dataset = pd.concat([features, target], axis = 1)
sns.distplot(target)
f, axes = plote.subplots(2, 4, figsize=(35, 15), sharex=True)
# CSE
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="CSE"].index)
d = pd.Series(d.CGPA, name="CSE")
sns.distplot(d, color="r", ax=axes[0,0])
# IT
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="IT"].index)
d = pd.Series(d.CGPA, name = "IT")
sns.distplot(d, color="b", ax=axes[0,1])
# ECE
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="ECE"].index)
d = pd.Series(d.CGPA, name = "ECE")
sns.distplot(d, color="g", ax=axes[0,2])
# EEE
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="EEE"].index)
d = pd.Series(d.CGPA, name = "EEE")
sns.distplot(d, color="y", ax=axes[0,3])
# CIVIL
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="CIVIL"].index)
d = pd.Series(d.CGPA, name = "CIVIL")
sns.distplot(d, color="m", ax=axes[1,0])
# Mech
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="MECH"].index)
d = pd.Series(d.CGPA, name = "MECH")
sns.distplot(d, color="r", ax=axes[1,1])
# Prod
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="PROD"].index)
d = pd.Series(d.CGPA, name = "PROD")
sns.distplot(d, color="g", ax=axes[1,2])
f, axes = plote.subplots(1, 2, figsize=(15, 7), sharex=True)
# Male
d = vis_dataset.drop(vis_dataset[vis_dataset["Gender"]!="Male"].index)
d = pd.Series(d.CGPA, name="Male")
sns.distplot(d, color="b", ax=axes[0])
# Fe
d = vis_dataset.drop(vis_dataset[vis_dataset["Gender"]!="Female"].index)
d = pd.Series(d.CGPA, name = "Female")
sns.distplot(d, color="y", ax=axes[1])
f, axes = plote.subplots(2, 2, figsize=(15, 10), sharex=True)
#General
d = vis_dataset.drop(vis_dataset[vis_dataset["Category"]!="GEN"].index)
d = pd.Series(d.CGPA, name="GEN")
sns.distplot(d, color="r", ax=axes[0,0])
# obc
d = vis_dataset.drop(vis_dataset[vis_dataset["Category"]!="OBC"].index)
d = pd.Series(d.CGPA, name = "OBC")
sns.distplot(d, color="b", ax=axes[0,1])
# sc
d = vis_dataset.drop(vis_dataset[vis_dataset["Category"]!="SC"].index)
d = pd.Series(d.CGPA, name = "SC")
sns.distplot(d, color="g", ax=axes[1,0])
# st
d = vis_dataset.drop(vis_dataset[vis_dataset["Category"]!="ST"].index)
d = pd.Series(d.CGPA, name = "ST")
sns.distplot(d, color="y", ax=axes[1,1])

sns.reset_orig()
sns.pairplot(features[['GPA 1','GPA 5','GPA 6','Normalized Rank','GPA 4','Gender']],hue='Gender',palette='inferno')
sns.pairplot(features[['GPA 2','Marks10th','GPA 3','Marks12th','Gender']],hue='Gender',palette='inferno')
vis_dataset.CGPA = vis_dataset.CGPA.apply(lambda x: int(x))
vis_dataset_WM = vis_dataset.drop(vis_dataset[vis_dataset['CGPA']>7].index)
vis_dataset_WM = vis_dataset_WM.drop(vis_dataset_WM[vis_dataset_WM['CGPA']<6].index)
optimal_features1 = ['GPA 1','GPA 2','GPA 3','GPA 4','GPA 5','GPA 6','CGPA']
optimal_features2 = ['Normalized Rank','Marks10th','Marks12th','CGPA']
optimal_features2 = ['Normalized Rank','Marks10th','Marks12th','CGPA']
data_wa = pd.melt(vis_dataset_WM[optimal_features1],id_vars="CGPA",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="CGPA", data=data_wa,split=True, inner="quart")
plt.xticks(rotation=45)
plt.title("Values of diff semester exams  ")
data_wa2 = pd.melt(vis_dataset_WM[optimal_features2],id_vars="CGPA",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="CGPA", data=data_wa2,split=True, inner="quart")
plt.xticks(rotation=90)
plt.title("Values of diff exams before coming to college  ")
vis_dataset_WM = vis_dataset.drop(vis_dataset[vis_dataset['CGPA']>8].index)
vis_dataset_WM = vis_dataset_WM.drop(vis_dataset_WM[vis_dataset_WM['CGPA']<7].index)
sns.reset_orig()
data_ag1 = pd.melt(vis_dataset_WM[optimal_features1],id_vars="CGPA",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="CGPA", data=data_ag1,split=True, inner="quart")
plt.xticks(rotation=90)
plt.title("Values of diff exams")
sns.reset_orig()
data_ag1 = pd.melt(vis_dataset_WM[optimal_features2],id_vars="CGPA",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="CGPA", data=data_ag1,split=True, inner="quart")
plt.xticks(rotation=90)
plt.title("Values of diff exams before coming to college  ")
plt.figure(figsize=(10,10))
sns.violinplot(x="Branch", y="CGPA", hue="Gender", data=vis_dataset,split=True, inner="quart", palette={"Male": "b", "Female": "y"})
plt.xticks(rotation=90)





































