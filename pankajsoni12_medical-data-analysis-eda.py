import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
df = pd.read_csv("../input/data.csv")
df.head(5)
df.drop(["id","zipcode"],axis=1,inplace=True)
df.head(1)
df.isnull().any()
plt.figure(figsize=(16,4))
sns.countplot(df.gender,palette="RdBu",hue=df.marital_status)
plt.figure(figsize=(16,4))
sns.countplot(df.disease)
plt.xticks(rotation="90")
plt.figure(figsize=(16,4))
sns.barplot(df.disease,df.available_vehicles)
plt.xticks(rotation="90")
plt.figure(figsize=(16,4))
sns.countplot(df.gender,palette="husl",hue=df.disease)
# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.employment_status,hue=df.disease,palette="rainbow")
# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.employment_status[df.employment_status=="student"],hue=df.disease)
# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.education)
# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.education,hue=df.employment_status)
# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.ancestry)
# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.disease)
plt.xticks(rotation=90)
# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.ancestry[df.ancestry=="Ireland"],hue=df.disease)
# print(df.head(1))
plt.figure(figsize=(16,4))
sns.countplot(df.ancestry,hue=df.disease)
plt.figure(figsize=(16,4))
sns.distplot(df.avg_commute)
plt.figure(figsize=(16,4))
sns.countplot(df.military_service,hue=df.disease)
plt.figure(figsize=(16,4))
sns.distplot(df.daily_internet_use)
plt.figure(figsize=(16,4))
sns.pairplot(df,hue="disease",palette="coolwarm")
plt.figure(figsize=(16,4))
sns.pairplot(df,hue="disease",palette="winter_r",diag_kind="hist")
plt.figure(figsize=(16,4))
sns.pairplot(df,hue="marital_status",palette="husl",diag_kind="hist",markers=["D","*"])
plt.figure(figsize=(16,4))
sns.pairplot(df,hue="education",palette="gist_earth_r",diag_kind="hist",markers=["D","*","^","<",">","."])
plt.figure(figsize=(16,4))
sns.pairplot(df,hue="gender",palette="cubehelix",diag_kind="hist",markers=[">","."])
plt.figure(figsize=(16,4))
sns.kdeplot(df.available_vehicles,df.daily_internet_use,cbar=True)
plt.figure(figsize=(16,4))
sns.countplot(df.gender,hue=df.ancestry)
net_m = df[df.gender=="male"].daily_internet_use.mean()
net_f = df[df.gender=="female"].daily_internet_use.mean()
print(net_m,net_f)
plt.figure(figsize=(16,4))
sns.barplot(["male","female"],[net_m,net_f])
plt.figure(figsize=(16,4))
sns.pointplot(df.gender,df.daily_internet_use,estimator=np.mean,markers=["*","<"],color="r",linestyles="--")
plt.figure(figsize=(16,4))
sns.lineplot(df.gender,df.daily_internet_use)
df.head(1)