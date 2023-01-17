import os

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from matplotlib import style

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.preprocessing import StandardScaler

from matplotlib.ticker import MaxNLocator

from statsmodels.formula.api import ols
path="/kaggle/input/insurance/insurance.csv"
raw_data=pd.read_csv(path)

raw_data
raw_data.info()
raw_data_c=raw_data.drop(["sex", "smoker", "region"], axis=1).copy() #only continuous variable dataset will be used for plots
raw_data.describe()
raw_data.isnull().sum()
plt.figure(figsize=(14,8))

style.use("seaborn-dark-palette")

plt.subplot(2,2,1)

plt.hist(raw_data["age"])

plt.xlabel("Ages", fontsize=12)

plt.subplot(2,2,2)

plt.hist(raw_data["bmi"])

plt.xlabel("BMI", fontsize=12)

plt.subplot(2,2,3)

plt.hist(raw_data["charges"])

plt.xlabel("Charges", fontsize=12)

plt.subplot(2,2,4)

plt.hist(raw_data["region"])

plt.xlabel("Region", fontsize=12)
corr_mat=raw_data_c.corr()

corr_mat
plt.figure(figsize=(10,8))

corar=np.array(corr_mat.values)

sns.set(font_scale=1.5)

sns.heatmap(corr_mat, annot=corar,cmap="coolwarm_r")
raw_data.age.describe()
raw_data.loc[(raw_data.age>17) & (raw_data.age<=30), "age_cat"]="Young Adult"

raw_data.loc[(raw_data.age>30) & (raw_data.age<=59), "age_cat"]="Adult"

raw_data.loc[(raw_data.age>59), "age_cat"]="Old"

raw_data
labels=raw_data.age_cat.unique().tolist()

count=raw_data.age_cat.value_counts()

print(count)

count=count.values

style.use("ggplot")

plt.figure(figsize=(8,8))

explode=(0.1,0,0)

plt.pie(count, labels=labels,explode=explode, autopct="%1.1f%%", textprops={'fontsize': 20})
charge_avg_age=raw_data.groupby("age_cat")["charges"].mean()

labels_avg=charge_avg_age.keys()

charge_avg_age=charge_avg_age.tolist()



charge_sum_age=raw_data.groupby(["age_cat"])["charges"].sum()

labels_sum=charge_sum_age.keys()

charge_sum_age=charge_sum_age.tolist()



charge_std_age=raw_data.groupby(["age_cat"])["charges"].std()

labels_std=charge_std_age.keys()

charge_std_age=charge_std_age.tolist()





style.use("seaborn")

plt.figure(figsize=(16,10))

plt.subplot(2,2,1)

plt.bar(labels_avg, charge_avg_age, color="green")

plt.ylabel("Mean Charges", fontsize=16)

plt.subplot(2,2,2)

plt.bar(labels_sum, charge_sum_age, color="indigo")

plt.ylabel("Sum  (1e7)", fontsize=16)

plt.subplot(2,2,3)

plt.bar(labels_sum, charge_std_age, color="black")

plt.ylabel("Charges Standard Deviation", fontsize=16)
raw_data["log_charges"]=np.log(raw_data["charges"])

raw_data 
plt.figure(figsize=(16,6))



plt.subplot(1,2,1)

raw_data["charges"].hist()

plt.xlabel("Charges", fontsize=16)





plt.subplot(1,2,2)

raw_data["log_charges"].hist()

plt.xlabel("Log of Charges", fontsize=16)
plt.figure(figsize=(15,10))

sns.set(font_scale=1.5)

plt.subplot(1,2,1)

sns.swarmplot(raw_data["sex"], raw_data["charges"], palette ="seismic")

plt.subplot(1,2,2)

sns.boxenplot(raw_data["sex"], raw_data["log_charges"], palette ="seismic")
plt.figure(figsize=(14,8))

sns.set(font_scale=1.5)

sns.boxenplot(raw_data["sex"], raw_data["bmi"], palette ="seismic_r")
raw_data.loc[(raw_data.age<19), "bmi_cat"]="Underweight"

raw_data.loc[(raw_data.age>=19) & (raw_data.age<=25), "bmi_cat"]="Normal"

raw_data.loc[(raw_data.age>25) & (raw_data.age<=30), "bmi_cat"]="Overweight"

raw_data.loc[(raw_data.age>30), "bmi_cat"]="Obese"

raw_data
bmi_val=raw_data["bmi_cat"].value_counts()

bmi_val=bmi_val.tolist()

style.use("seaborn-dark-palette")

labels=raw_data["bmi_cat"].unique()

plt.figure(figsize=(12,5))

plt.bar(labels, bmi_val)

plt.ylabel("Count", fontsize=16)
bmi_avg_charge=raw_data.groupby("bmi_cat")["charges"].mean()

labels_a=bmi_avg_charge.keys()

bmi_avg_charge=bmi_avg_charge.tolist()



bmi_count_charge=raw_data.groupby("bmi_cat")["charges"].count()

labels_c=bmi_count_charge.keys()

bmi_count_charge=bmi_count_charge.tolist()





style.use("seaborn-dark-palette")

plt.figure(figsize=(16,5))

plt.subplot(1,2,1)

plt.bar(labels_a, bmi_avg_charge)

plt.ylabel("Mean Charges", fontsize=16)



plt.subplot(1,2,2)

plt.bar(labels_c, bmi_count_charge)

plt.ylabel("Count", fontsize=16)

plt.figure(figsize=(14,8))

sns.set(font_scale=1.5)

sns.swarmplot(raw_data["smoker"], raw_data["charges"],hue=raw_data["sex"], palette="winter")
plt.figure(figsize=(15,10))

style.use("ggplot")

ax=plt.subplot(2,1,1)

smk_bmi=raw_data.groupby(["smoker", "bmi_cat"])["charges"].mean().unstack()

print(smk_bmi)

smk_bmi.plot(ax=ax)



ax=plt.subplot(2,1,2)

smk_bmi=raw_data.groupby(["smoker", "bmi_cat"])["charges"].mean().plot(ax=ax)

ax.tick_params('x',labelrotation=45)
raw_data_c
std_scl=StandardScaler()

raw_data_std=std_scl.fit_transform(raw_data_c)

print("columns as age, bmi. children, charges")

print(raw_data_std)
bmi_charg_c=raw_data_std[:,[1,3]]

print(bmi_charg_c)

print(bmi_charg_c.shape)
wss=[]

sil=[]

for k in range(2,16):

    kmeans=KMeans(n_clusters=k, random_state=1).fit(bmi_charg_c)

    wss.append(kmeans.inertia_)

    labels=kmeans.labels_

    silhoutte=silhouette_score(bmi_charg_c, labels, metric = 'euclidean')

    sil.append(silhoutte)
k=range(2,16)

style.use("bmh")

fig,ax=plt.subplots(figsize=(14,6))

ax.set_facecolor("white")

ax.plot(k, wss, color="green")

ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))

ax.set_xlabel("No of clusters", fontsize=20)

ax.set_ylabel("WSS (With in Sum of squares)", fontsize=20)

ax2=ax.twinx()

ax2.plot(k, sil, color="blue")

ax2.set_ylabel("Silhouette scores", fontsize=20)

ax2.grid(True,color="silver")

plt.show()
k=3

kmeans=KMeans(n_clusters=k, random_state=1).fit(bmi_charg_c)

clusters=kmeans.labels_

centrids=kmeans.cluster_centers_

raw_data["clusters"]=clusters

raw_data
raw_data2=raw_data.sort_values(["clusters"]).copy()
for i in range(0,k+1):

    raw_data2["clusters"]=raw_data2["clusters"].replace(i, chr(i+65))

    

raw_data2
raw_data2["clusters"].unique()
x=raw_data2.iloc[:,[2,6]].values

print(x.shape)

y=kmeans.fit_predict(x)

print(y.shape)
plt.figure(figsize=(14,8))

style.use("Solarize_Light2")

plt.scatter(x[y==0,0], x[y==0,1], color="red", label="A")

plt.scatter(x[y==1,0], x[y==1,1], color="blue", label="B")

plt.scatter(x[y==2,0], x[y==2,1], color="green", label="C")



plt.xlabel("BMI", fontsize=16)

plt.ylabel("Charges", fontsize=16)

plt.title("Charges depends on BMI??", fontsize=18)
age_charg_c=raw_data_std[:,[0,3]]

print(age_charg_c)

print(age_charg_c.shape)
wss=[]

sil=[]

for k in range(2,16):

    kmeans=KMeans(n_clusters=k, random_state=1).fit(age_charg_c)

    wss.append(kmeans.inertia_)

    labels=kmeans.labels_

    silhoutte=silhouette_score(age_charg_c, labels, metric = 'euclidean')

    sil.append(silhoutte)
k=range(2,16)

style.use("bmh")

fig,ax=plt.subplots(figsize=(14,6))

ax.set_facecolor("white")

ax.plot(k, wss, color="green")

ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))

ax.set_xlabel("No of clusters", fontsize=20)

ax.set_ylabel("WSS (With in Sum of squares)", fontsize=20)

ax2=ax.twinx()

ax2.plot(k, sil, color="blue")

ax2.set_ylabel("Silhouette scores", fontsize=20)

ax2.grid(True,color="silver")

plt.show()
k=3

kmeans=KMeans(n_clusters=k, random_state=1).fit(age_charg_c)

clusters=kmeans.labels_

centrids=kmeans.cluster_centers_

raw_data["clusters"]=clusters

raw_data
raw_data2=raw_data.sort_values(["clusters"]).copy()
for i in range(0,k+1):

    raw_data2["clusters"]=raw_data2["clusters"].replace(i, chr(i+65))

    

raw_data2
x=raw_data2.iloc[:,[0,6]].values

print(x.shape)

y=kmeans.fit_predict(x)

print(y.shape)
plt.figure(figsize=(14,8))

style.use("Solarize_Light2")

plt.scatter(x[y==0,0], x[y==0,1], color="red", label="A")

plt.scatter(x[y==1,0], x[y==1,1], color="blue", label="B")

plt.scatter(x[y==2,0], x[y==2,1], color="green", label="C")



plt.xlabel("Age", fontsize=16)

plt.ylabel("Charges", fontsize=16)

plt.title("Charges depends on Age??", fontsize=18)
raw_data2["smoker"]=raw_data2["smoker"].replace(["yes", "no"],[1,0])
pval=ols("charges~bmi+age+children+smoker", data=raw_data).fit()

print(pval.summary())