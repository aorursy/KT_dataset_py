# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

df.head()
df.describe()
df.info()
df_group=df.groupby(["gender","status","specialisation","hsc_s"])[["hsc_p","degree_p","etest_p","salary"]].mean()

df_group
df_p=df[df["status"]=="Placed"]

mask_m=df["gender"]=="M"

mask_f=df["gender"]=="F"

df_np=df[df["status"]=="Not Placed"]

df_p=df[df["status"]=="Placed"]

df_p["gender"].value_counts()
plt.figure(figsize=(10,5))

plt.pie([100,48], explode=(0,0.08), labels=["MALE","FEMALE"], autopct='%1.2f%%',

        shadow=True, startangle=100,colors=["yellow","cyan"])

plt.title("MALE VS FEMALE GOT PLACED")
plt.figure(figsize=(10,5))

sns.countplot(x="gender",data=df_p,hue="specialisation")

plt.title("MALE VS FEMALE GOT PLACED ACCORDING TO SPECILISATION")
sns.countplot(x="gender",data=df_np,hue="specialisation")

plt.title("MALE VS FEMALE NOT PLACED")
sns.countplot(x="gender",data=df_p,hue="hsc_s")

plt.title("MALE VS FEMALE GOT PLACED")
sns.countplot(x="gender",data=df_p,hue="degree_t",color="red")

plt.title("MALE VS FEMALE GOT PLACED BASED ON BACHELOR DEGREE")
import plotly.express as px

grs = df.groupby(["gender"])[["salary"]].mean().reset_index()

fig = px.bar(grs[['gender', 'salary']].sort_values('salary', ascending=False), 

             y="salary", x="gender", color='gender', 

             log_y=True, template='ggplot2')

fig.show()
grgs = df.groupby(["gender","specialisation"])[["salary"]].mean().reset_index()

fig = px.bar(grgs, x="gender", y="salary", color='specialisation', barmode='stack',

             height=400)

fig.show()
plt.figure(figsize=(10,5))

sns.distplot(df['salary'], bins=50, hist=False)

plt.title("SALARY DISTRIBUTION")
plt.figure(figsize=(20,7))

sns.violinplot(x=df_p["gender"],y=df_p["salary"],hue=df_p["specialisation"],palette="Set2")
plt.figure(figsize=(20,5))

df_pm=df_p[mask_m]

df_pf=df_p[mask_f]

mask_g=["M","F"]

mask_spe=["Mkt&HR","Mkt&Fin"]

#f, axes = plt.subplots(1, 3, figsize=(18,5), sharex=True)

for j in range(len(mask_spe)):

    df_p_=df_p[df_p["specialisation"]==mask_spe[j]]

    for i in mask_g:

        df_p__=df_p_[df_p_["gender"]==i]

        sns.distplot(df_p__["salary"],hist=False,kde_kws = {'shade': True, 'linewidth': 3},label=(mask_spe[j],i))
gp = df.groupby(["gender","specialisation"])[["salary"]].mean().reset_index()



fig = px.treemap(gp, path=['gender','specialisation'], values='salary',

                  color='salary', hover_data=['specialisation'],

                  color_continuous_scale='rainbow')

fig.show()
grss = df.groupby(["hsc_b","hsc_s"])[["hsc_p"]].mean().reset_index()



fig = px.treemap(grss, path=['hsc_b','hsc_s'], values='hsc_p',

                  color='hsc_p', hover_data=['hsc_s'],

                  color_continuous_scale='rainbow')

fig.show()
grdsp = df.groupby(["degree_t"])[["degree_p"]].mean().reset_index()



fig = px.pie(grdsp,

             values="degree_p",

             names="degree_t",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
import plotly.express as px

fig = px.scatter_ternary(df, a="ssc_p", b="hsc_p",c="degree_p",color = "status")

fig.show()
sns.violinplot(x="status",y="degree_p",data=df,hue="gender")
sns.violinplot(x="status", y="etest_p", hue="gender", data=df)
df.tail()


sns.catplot(y="degree_p",x="degree_t",col="hsc_s",data=df)