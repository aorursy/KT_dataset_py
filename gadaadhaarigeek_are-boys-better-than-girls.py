import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()
# sns.set_style(style="darkgrid")
# sns.despine()

import warnings
warnings.filterwarnings(action="ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list 
# the files in the input directory

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/StudentsPerformance.csv")
data.head()
data.columns = ["gender", "race_ethnicity", "parent_education", "lunch", "test_preparation", "maths_score", "reading_score", "writing_score"]
data.columns
data.info()
data.describe()
data.isnull().sum()
for column in data.columns:
    if column == "maths_score":
        break
    print(column.upper())
    print(data[column].value_counts())
    print("\n\n\n")
plt.figure(figsize=(20, 10))
sns.countplot(x="race_ethnicity", data=data, hue="gender", order=["group A", "group B", "group C", "group D", "group E"])
plt.figure(figsize=(20, 10))
sns.countplot(x="race_ethnicity", data=data, hue="parent_education", order=["group A", "group B", "group C", "group D", "group E"])
race = sorted(data["race_ethnicity"].value_counts().index)
p_education = sorted(data["parent_education"].value_counts().index)
df = {"race_ethnicity":[], "parent_education":[], "per":[]}
for col in race:
    d = data[data["race_ethnicity"] == col].shape[0]
    d = d*1.0
    for edu in p_education:
        n = data[(data["race_ethnicity"] == col) & (data["parent_education"] == edu)].shape[0]
        df["race_ethnicity"].append(col)
        df["parent_education"].append(edu)
        df["per"].append(round((n/d)*100, 2))

df = pd.DataFrame(data=df)
df.head(10)
plt.figure(figsize=(20, 10))
sns.barplot(x="race_ethnicity", y="per", data=df, hue="parent_education", order=["group A", "group B", "group C", "group D", "group E"])
race = sorted(data["race_ethnicity"].value_counts().index)
p_education = sorted(data["parent_education"].value_counts().index)
df = {"race_ethnicity":[], "parent_education":[], "per":[]}
for edu in p_education:
    d = data[data["parent_education"] == edu].shape[0]
    d = d*1.0
    for col in race:
        n = data[(data["race_ethnicity"] == col) & (data["parent_education"] == edu)].shape[0]
        df["race_ethnicity"].append(col)
        df["parent_education"].append(edu)
        df["per"].append(round((n/d)*100, 2))

df = pd.DataFrame(data=df)
df.head()
plt.figure(figsize=(20, 10))
sns.barplot(x="race_ethnicity", y="per", data=df, hue="parent_education", order=["group A", "group B", "group C", "group D", "group E"])
plt.figure(figsize=(20, 10))
sns.pointplot(x="race_ethnicity", y="per", data=df, hue="parent_education", order=["group A", "group B", "group C", "group D", "group E"])
plt.figure(figsize=(20, 10))
g = sns.countplot(x="parent_education", data=data, hue="race_ethnicity")
g.set_xticklabels(labels=g.get_xticklabels(), rotation=45)
new_df = get_format(x="parent_education", y="race_ethnicity", data=data)
new_df.head()
sns.countplot(x="lunch", data=data)
sns.countplot(x="lunch", data=data, hue="gender")
sns.countplot(x="lunch", data=data, hue="race_ethnicity")
plt.figure(figsize=(20,8))
sns.countplot(x="lunch", data=data, hue="parent_education")
plt.figure(figsize=(20,8))
sns.countplot(x="test_preparation", data=data, hue="race_ethnicity")
plt.figure(figsize=(20,8))
sns.countplot(x="test_preparation", data=data, hue="parent_education")
plt.figure(figsize=(15,8))
sns.boxplot(x="race_ethnicity", y="maths_score", data=data, hue="gender", palette="Set1")
plt.figure(figsize=(15,8))
sns.boxenplot(x="race_ethnicity", y="maths_score", data=data, hue="gender", palette="Set1")
plt.figure(figsize=(15,8))
sns.violinplot(x="race_ethnicity", y="maths_score", data=data, hue="gender", palette="Set1")
plt.figure(figsize=(15,6))
sns.violinplot(x="race_ethnicity", y="reading_score", data=data, hue="gender", palette="Set1")
plt.figure(figsize=(15,8))
sns.barplot(x="race_ethnicity", y="writing_score", data=data, hue="gender", palette="Set1")
plt.figure(figsize=(15,8))
sns.boxenplot(x="parent_education", y="maths_score", data=data, palette="Set1")
plt.figure(figsize=(20,20))
plt.subplot(2, 2, 1)
sns.boxplot(x="parent_education", y="maths_score", data=data, palette="Set1")
plt.xticks(rotation=30)
plt.subplot(2, 2, 2)
sns.boxplot(x="parent_education", y="reading_score", data=data, palette="Set1")
plt.xticks(rotation=30)
plt.subplot(2, 2, 3)
sns.boxplot(x="parent_education", y="writing_score", data=data, palette="Set1")
plt.xticks(rotation=30)
plt.figure(figsize=(20,20))
plt.subplot(2, 2, 1)
sns.barplot(x="parent_education", y="maths_score", data=data, palette="Set1", estimator=np.median)
plt.xticks(rotation=30)
plt.subplot(2, 2, 2)
sns.barplot(x="parent_education", y="reading_score", data=data, palette="Set1", estimator=np.median)
plt.xticks(rotation=30)
plt.subplot(2, 2, 3)
sns.barplot(x="parent_education", y="writing_score", data=data, palette="Set1", estimator=np.median)
plt.xticks(rotation=30)
binsize=15
plt.figure(figsize=(20,20))
plt.subplot(2, 2, 1)
sns.distplot(a=data["maths_score"], bins=binsize, hist=True)
plt.xticks(rotation=30)
plt.subplot(2, 2, 2)
sns.distplot(a=data["reading_score"], bins=binsize, hist=True)
plt.xticks(rotation=30)
plt.subplot(2, 2, 3)
sns.distplot(a=data["writing_score"], bins=binsize, hist=True)
plt.xticks(rotation=30)
# binsize=15
# plt.figure(figsize=(20,20))
# plt.subplot(2, 2, 1)
# sns.FacetGrid(data, hue="gender", size=5).map(sns.kdeplot, "maths_score").add_legend()
# plt.xticks(rotation=30)
# plt.subplot(2, 2, 2)
# sns.FacetGrid(data, hue="gender", size=5).map(sns.kdeplot, "reading_score").add_legend()
# plt.xticks(rotation=30)
# plt.subplot(2, 2, 3)
# sns.FacetGrid(data, hue="gender", size=5).map(sns.kdeplot, "writing_score").add_legend()
sns.FacetGrid(data, hue="gender", size=5).map(sns.kdeplot, "maths_score").add_legend()
sns.FacetGrid(data, hue="gender", size=5).map(sns.kdeplot, "reading_score").add_legend()
sns.FacetGrid(data, hue="gender", size=5).map(sns.kdeplot, "writing_score").add_legend()
plt.figure(figsize=(20,20))
plt.subplot(2, 2, 1)
sns.boxplot(x="gender", y="maths_score", data=data, palette="Set1")
plt.xticks(rotation=30)
plt.subplot(2, 2, 2)
sns.boxplot(x="gender", y="reading_score", data=data, palette="Set1")
plt.xticks(rotation=30)
plt.subplot(2, 2, 3)
sns.boxplot(x="gender", y="writing_score", data=data, palette="Set1")
plt.xticks(rotation=30)
plt.figure(figsize=(20,20))
plt.subplot(2, 2, 1)
sns.swarmplot(x="gender", y="maths_score", data=data, palette="Set1")
plt.xticks(rotation=30)
plt.subplot(2, 2, 2)
sns.swarmplot(x="gender", y="reading_score", data=data, palette="Set1")
plt.xticks(rotation=30)
plt.subplot(2, 2, 3)
sns.swarmplot(x="gender", y="writing_score", data=data, palette="Set1")
plt.xticks(rotation=30)
plt.figure(figsize=(20,10))
plt.subplot(1, 3, 1)
sns.boxplot(y="maths_score", data=data, palette="Set1")
plt.xticks(rotation=30)
plt.subplot(1, 3, 2)
sns.boxplot(y="reading_score", data=data, palette="Set1")
plt.xticks(rotation=30)
plt.subplot(1, 3, 3)
sns.boxplot(y="writing_score", data=data, palette="Set1")
plt.xticks(rotation=30)