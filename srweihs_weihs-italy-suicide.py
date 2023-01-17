import seaborn as sns
import pandas as pd
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('Italy_Suicide.csv',na_values="NaN")
df.info()
df=df.drop_duplicates()
df.describe
df=df.dropna()
df
df.Age_Group = [x.strip('years') for x in df.Age_Group]
df.Age_Group = [x.strip("+") for x in df.Age_Group]
df["Age_Group"] = df.Age_Group.str.replace('-24','')
df["Age_Group"] = df.Age_Group.str.replace('+','')
df["Age_Group"] = df.Age_Group.str.replace('-34','')
df["Age_Group"] = df.Age_Group.str.replace('-14','')
df["Age_Group"] = df.Age_Group.str.replace('-74','')
df["Age_Group"] = df.Age_Group.str.replace('-54','')
df
sns.pairplot(df)
df.groupby('Year')['Suicide_no'].sum()
df.plot.scatter(x='Year',y='Suicide_no')
df.plot.scatter(x='Population',y='Suicide_no')
df.groupby('Gender')['Suicide_no'].sum()['female']
df.groupby('Gender')['Suicide_no'].sum()['male']
df.boxplot(by='Gender',column='Suicide_no')
sns.pairplot(df)
df.plot.scatter(x='Year',y='Suicide_no')
df.groupby('Gender')['Suicide_no'].sum()
df.plot.bar(x='Gender',y='Suicide_no')
df.boxplot(by='Year',column='Suicide_no',figsize=(12,8),rot=45)
df2 = df.groupby('Year')['Suicide_no'].sum()
df2.plot.bar()
