import numpy as np
import pandas as pd
import seaborn as sns
import re
import os
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
from __future__ import division
%config InlineBackend.figure_format = 'svg'

print(os.listdir("../input"))

sp_data = pd.read_csv('../input/StudentsPerformance.csv')
sp_data.head()
sp_data.info()
# print (sp_data[1])
for column in (sp_data.columns.values):
    if (sp_data[column].dtype) != np.dtype('int'):
        print ("Unique values in '"+ column + "' column are ", end='')
        print (sp_data[column].unique())
print (sp_data['gender'].value_counts())
sns.countplot(x='gender', data=sp_data);
print (sp_data['race/ethnicity'].value_counts())
ax = sns.countplot(x='race/ethnicity', data=sp_data, order= ['group A', 'group B', 'group C', 'group D', 'group E'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
print (sp_data['parental level of education'].value_counts())
ax = sns.countplot(x='parental level of education', data=sp_data, order=['some high school', 'high school', 'associate\'s degree', 'some college',
                                                                         "bachelor's degree","master's degree"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
print (sp_data['lunch'].value_counts())
sns.countplot(x='lunch', data=sp_data);
sns.countplot(x='test preparation course', data=sp_data);
sns.barplot(x='gender',y='math score',data=sp_data);
sp_data[['math score','reading score','writing score']].describe()
sns.distplot(sp_data['math score'], bins=25, kde=False);
sns.distplot(sp_data['reading score'], bins=25, kde=False);
sns.distplot(sp_data['writing score'], bins=25, kde=False);
sp_data['gender'].loc[sp_data['gender'] == 'male'] = 1
sp_data['gender'].loc[sp_data['gender'] == 'female'] = 0
sp_data['race/ethnicity'].loc[sp_data['race/ethnicity'] == 'group A'] = 1
sp_data['race/ethnicity'].loc[sp_data['race/ethnicity'] == 'group B'] = 2
sp_data['race/ethnicity'].loc[sp_data['race/ethnicity'] == 'group C'] = 3
sp_data['race/ethnicity'].loc[sp_data['race/ethnicity'] == 'group D'] = 4
sp_data['race/ethnicity'].loc[sp_data['race/ethnicity'] == 'group E'] = 5
# sp_data['lunch'].loc[sp_data['lunch'] == 'standard'] = 1
# sp_data['lunch'].loc[sp_data['lunch'] == 'free/reduced'] = 0


# MData["test preparation course"]=MData["test preparation course"].replace({"none":0,"completed":1})

sp_data['lunch'] = sp_data['lunch'].replace({'standard':1,'free/reduced':0})
sp_data['test preparation course'].loc[sp_data['test preparation course'] == 'none'] = 0
sp_data['test preparation course'].loc[sp_data['test preparation course'] == 'completed'] = 1
sp_data['parental level of education'].loc[sp_data['parental level of education'] == 'some high school'] = 1
sp_data['parental level of education'].loc[sp_data['parental level of education'] == 'high school'] = 2
sp_data['parental level of education'].loc[sp_data['parental level of education'] == 'associate\'s degree'] = 3
sp_data['parental level of education'].loc[sp_data['parental level of education'] == 'some college'] = 4
sp_data['parental level of education'].loc[sp_data['parental level of education'] == 'bachelor\'s degree'] = 5
sp_data['parental level of education'].loc[sp_data['parental level of education'] == 'master\'s degree'] = 6
sp_data.head(10)
(sp_data[['race/ethnicity','math score', 'reading score', 'writing score']].corr())
ax = sns.heatmap(sp_data.corr(),cmap="Blues",annot=True,annot_kws={"size": 7.5},linewidths=.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right");
# plt.tight_layout()
sp_score_data = sp_data[['gender','math score','reading score','writing score']].groupby('gender',as_index=True).mean()
print ('averages: \n'+str(sp_score_data.head()))
fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="gender", y="math score", data=sp_data, ax=axs[0],showmeans=True);
sns.boxplot(x="gender", y="reading score", data=sp_data, ax=axs[1],showmeans=True);
sns.boxplot(x="gender", y="writing score", data=sp_data, ax=axs[2],showmeans=True);
sns.boxplot(x="gender", y="math score", data=sp_data, showmeans=True);
fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="lunch", y="math score", data=sp_data, ax=axs[0],showmeans=True);
sns.boxplot(x="lunch", y="reading score", data=sp_data, ax=axs[1],showmeans=True);
sns.boxplot(x="lunch", y="writing score", data=sp_data, ax=axs[2],showmeans=True);
fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="race/ethnicity", y="math score", data=sp_data, ax=axs[0],showmeans=True);
sns.boxplot(x="race/ethnicity", y="reading score", data=sp_data, ax=axs[1],showmeans=True);
sns.boxplot(x="race/ethnicity", y="writing score", data=sp_data, ax=axs[2],showmeans=True);
fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="parental level of education", y="math score", data=sp_data, ax=axs[0],showmeans=True);
sns.boxplot(x="parental level of education", y="reading score", data=sp_data, ax=axs[1],showmeans=True);
sns.boxplot(x="parental level of education", y="writing score", data=sp_data, ax=axs[2],showmeans=True);
fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="test preparation course", y="math score", data=sp_data, ax=axs[0],showmeans=True);
sns.boxplot(x="test preparation course", y="reading score", data=sp_data, ax=axs[1],showmeans=True);
sns.boxplot(x="test preparation course", y="writing score", data=sp_data, ax=axs[2],showmeans=True);
reduced_lunch = sp_data.loc[sp_data['lunch'] == 0]
fig, axs = plt.subplots(ncols=3,figsize=(12,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None);
sns.boxplot(x="test preparation course", y="math score", data=reduced_lunch, ax=axs[0],showmeans=True);
sns.boxplot(x="test preparation course", y="reading score", data=reduced_lunch, ax=axs[1],showmeans=True);
sns.boxplot(x="test preparation course", y="writing score", data=reduced_lunch, ax=axs[2],showmeans=True);