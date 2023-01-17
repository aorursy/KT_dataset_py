import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
#Read the dataset from the path and see a preview of it

df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

df.head()
#Check the datatypes of the columns

df.dtypes
#Checking out for count of missing values in each column

df.isnull().sum()
#Checking the distribution of target variable

sns.countplot(df["target"]);
df.columns
fig, axes = plt.subplots(1,2,figsize = (15,5))



#Checking the distribution of this column

sns.distplot(df["age"], ax = axes[0]).set_title("Age distribution")



#Target wise old peak average

df.groupby(by = ["target"])["age"].mean().plot(kind = "bar", ax = axes[1], title="Target wise mean age");

#Creating bins for age

lstBins = [20,40,50,60,70,90]

df["ageGrp"] = pd.cut(df["age"], bins = lstBins, labels = ["Young", "Young2Old", "Old", "Senior", "Fragile"])
#Checking the distribution of age grp variable and its impact on heart disease

fig, axes = plt.subplots(1,3,figsize = (15,5))

sns.countplot(x = df["ageGrp"], ax = axes[0]).set_title("Age grp distribution")



#Impact of sex on heart disease

dfTemp = pd.crosstab(df["ageGrp"], df["target"])

dfTemp.plot(kind="bar", ax = axes[1], title="Age grp wise heart disease count");



#Calculating the odds of having heart disease for each age grp type

dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)

dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);
#Check the swarm plot of the age grp variable

sns.swarmplot(x="ageGrp", y="age",hue='target', data=df).set_title("Target(0/1) separation among age groups");
#Checking the distribution of sex variable and its impact on heart disease

fig, axes = plt.subplots(1,3,figsize = (15,5))

df["sex"].plot(kind="hist", ax = axes[0], title="sex distribution");



#Impact of sex on heart disease

dfTemp = pd.crosstab(df["sex"], df["target"])

dfTemp.plot(kind="bar", ax = axes[1], title="Sex wise heart disease count");



#Calculating the odds of having heart disease for each sex type

dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)

dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);
#Checking the distribution of chest pain variable and its impact on heart disease

fig, axes = plt.subplots(1,3,figsize = (15,5))

df["cp"].plot(kind="hist", ax = axes[0], title="Chest pain distribution");



#Impact of chest pain on heart disease

dfTemp = pd.crosstab(df["cp"], df["target"])

dfTemp.plot(kind="bar", ax = axes[1], title="Chest pain wise heart disease count");



#Calculating the odds of having heart disease for each chest pain type

dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)

dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);
fig, axes = plt.subplots(1,2,figsize = (15,5))



#Checking the distribution of this column

sns.distplot(df["trestbps"], ax = axes[0]).set_title("BP distribution")



#Target wise old peak average

df.groupby(by = ["target"])["trestbps"].mean().plot(kind = "bar", ax = axes[1], title="Target wise mean BP ");
#Binning the rest systolic BP

bpCatLst = [70,100,120,140,160,220]

df["bpGrp"] = pd.cut(df["trestbps"], bins = bpCatLst, labels = ["very low", "low", "normal","high","very high"])
#Checking the distribution of BP grp variable and its impact on heart disease

fig, axes = plt.subplots(1,3,figsize = (15,5))

sns.countplot(x = df["bpGrp"], ax = axes[0]).set_title("BP grp distribution")



#Impact of BP grp on heart disease

dfTemp = pd.crosstab(df["bpGrp"], df["target"])

dfTemp.plot(kind="bar", ax = axes[1], title="BP grp wise heart disease count");



#Calculating the odds of having heart disease for each BP grp type

dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)

dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);
fig, axes = plt.subplots(1,2,figsize = (15,5))



#Checking the distribution of this column

sns.distplot(df["chol"], ax = axes[0]).set_title("Cholestrol distribution")



#Target wise old peak average

df.groupby(by = ["target"])["chol"].mean().plot(kind = "bar", ax = axes[1], title="Target wise cholestrol mean");
#Binning the cholestrol levels

cholCatLst = [100,200,239,300,350,700]

df["cholGrp"] = pd.cut(df["chol"], bins = cholCatLst, labels = ["normal", "borderline high", "high","very high","risky high"])
#Checking the distribution of cholestrol grp variable and its impact on heart disease

fig, axes = plt.subplots(1,3,figsize = (15,5))

sns.countplot(x = df["cholGrp"], ax = axes[0]).set_title("CHOLESTROL group distribution")



#Impact of BP grp on heart disease

dfTemp = pd.crosstab(df["cholGrp"], df["target"])

dfTemp.plot(kind="bar", ax = axes[1], title="Cholestrol wise heart disease count");



#Calculating the odds of having heart disease for each  type

dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)

dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);
#Checking the distribution of fasting blood sugar variable and its impact on heart disease

fig, axes = plt.subplots(1,3,figsize = (15,5))

sns.countplot(x = df["fbs"], ax = axes[0]).set_title("Blood Sugar distribution")



#Impact of BP grp on heart disease

dfTemp = pd.crosstab(df["fbs"], df["target"])

dfTemp.plot(kind="bar", ax = axes[1], title="Blood Sugar wise heart disease count");



#Calculating the odds of having heart disease for each BS grp type

dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)

dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);
#Checking the distribution of rest ECG variable and its impact on heart disease

fig, axes = plt.subplots(1,3,figsize = (15,5))

sns.countplot(x = df["restecg"], ax = axes[0]).set_title("ECG distribution")



#Impact of BP grp on heart disease

dfTemp = pd.crosstab(df["restecg"], df["target"])

dfTemp.plot(kind="bar", ax = axes[1], title="ECG wise heart disease count");



#Calculating the odds of having heart disease for each type

dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)

dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);
#Checking the distribution of exang variable and its impact on heart disease

fig, axes = plt.subplots(1,3,figsize = (15,5))

sns.countplot(x = df["exang"], ax = axes[0]).set_title("Exercize induced angina distribution")



#Impact of BP grp on heart disease

dfTemp = pd.crosstab(df["exang"], df["target"])

dfTemp.plot(kind="bar", ax = axes[1], title="EXANG wise heart disease count");



#Calculating the odds of having heart disease for each type

dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)

dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);
fig, axes = plt.subplots(1,2,figsize = (15,5))



#Checking the distribution of this column

sns.distplot(df["oldpeak"], ax = axes[0]).set_title("Old peak distribution")



#Target wise old peak average

df.groupby(by = ["target"])["oldpeak"].mean().plot(kind = "bar", ax = axes[1], title="Target wise Old peak mean");
#Checking the distribution of slope variable and its impact on heart disease

fig, axes = plt.subplots(1,3,figsize = (15,5))

sns.countplot(x = df["slope"], ax = axes[0]).set_title("Slope distribution")



#Impact of BP grp on heart disease

dfTemp = pd.crosstab(df["slope"], df["target"])

dfTemp.plot(kind="bar", ax = axes[1], title="Slope wise heart disease count");



#Calculating the odds of having heart disease for each  type

dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)

dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);
#Checking the distribution of CA variable and its impact on heart disease

fig, axes = plt.subplots(1,3,figsize = (15,5))

sns.countplot(x = df["ca"], ax = axes[0]).set_title("Coloured arteries distribution")



#Impact of BP grp on heart disease

dfTemp = pd.crosstab(df["ca"], df["target"])

dfTemp.plot(kind="bar", ax = axes[1], title="Coloured arteries wise heart disease count");



#Calculating the odds of having heart disease for each  type

dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)

dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);
#Checking the distribution of variable and its impact on heart disease

fig, axes = plt.subplots(1,3,figsize = (15,5))

sns.countplot(x = df["thal"], ax = axes[0]).set_title("THAL distribution")



#Impact on heart disease

dfTemp = pd.crosstab(df["thal"], df["target"])

dfTemp.plot(kind="bar", ax = axes[1], title="THAL wise heart disease count");



#Calculating the odds of having heart disease for each type

dfTemp["Odds"] = round(dfTemp[1]/dfTemp[0],3)

dfTemp["Odds"].plot(kind="line", ax = axes[2], title ="Odds of having a heart disease", grid=True);