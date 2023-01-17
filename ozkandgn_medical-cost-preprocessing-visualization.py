import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("/kaggle/input/insurance/insurance.csv")
data.sample(10)
data.info()
data.isna().any()
## Show categoric values

print(data["sex"].unique())

print(data["smoker"].unique())

print(data["region"].unique())
# sex value type change

#male = 1 -- famale = 0

data["sex"] = [1 if i=="male" else 0 for i in data["sex"]]
# smoker value type change

#yes = 1 -- no = 0

data["smoker"] = [1 if i=="yes" else 0 for i in data["smoker"]]
change_region = lambda i : 1 if i=="southwest" else (2 if i=="southeast" else (3 if i=="northwest" else 4))

data["region_num"] = data["region"].apply(change_region)
data["age"].describe()
## new age range feature

age_range = lambda i : "18-26" if i<27 else ("27-38" if i<39 else ("39-50" if i<51 else "51-64"))

age_range_num = lambda i : 1 if i<27 else (2 if i<39 else (3 if i<51 else 4))



data["age_range"] = data["age"].apply(age_range)

data["age_range_num"] = data["age"].apply(age_range_num)
data["bmi"].describe()
## new age range feature

bmi_range = lambda i : "16-26" if i<26.5 else ("27-30" if i<30.5 else ("31-34" if i<34.5 else "35-53"))

bmi_range_num = lambda i : 1 if i<26.5 else (2 if i<30.5 else (3 if i<34.5 else 4))



data["bmi_range"] = data["bmi"].apply(bmi_range)

data["bmi_range_num"] = data["bmi"].apply(bmi_range_num)
data.head()
# Outlier values viewed

plt.subplot(1,2,1)

data.boxplot(column="charges")
# Observe correlation for relationship between features

sns.heatmap(data.corr())
plot = sns.barplot(x="smoker",y="charges",data=data)

plot.set_xticklabels(["No","Yes"]);

#we can easly see here relation between smoker and charges
## relationship between sex-smoker and charges

plot = sns.barplot(x="smoker",y="charges",hue="sex",data=data)

# there isn't a spesific relationship
data["region"].value_counts()
## There isn't a relationship between regions

plt.pie(data["region"].value_counts(),labels=data["region"].value_counts().index,shadow=True);
plot = sns.barplot(x="age_range",y="charges",hue="bmi_range",data=data)

## bmi and age clearly affects to charge 
## generally bmi distribuion between 25-35

sns.jointplot("age", "bmi", kind="hex",data=data)
# Generally humans have 28-32 bmi

plt.figure(figsize=(12,5))

sns.barplot(x="age",y="bmi",data=data)
# Charge distribution 

plt.figure(figsize=(10,4));

sns.catplot(x="smoker", y="charges", hue="sex", data=data, kind="violin");

## Generally humans have 0 or 1 child

data[["children","age"]].groupby("children").count()
# Proportion of smokers by age

age_smoke_group_ratio = data[["age","smoker"]].groupby("age").count() / data[["age","smoker"]].groupby("age").sum()

plt.figure(figsize=(15,5))

sns.barplot(age_smoke_group_ratio.index,age_smoke_group_ratio["smoker"])