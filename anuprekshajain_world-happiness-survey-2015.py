import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline 
import matplotlib.pyplot as plt
df = pd.read_csv("../input/2015.csv")
df.head(5)
plt.bar(df['Region'],df['Happiness Score'],color='green')
plt.xticks(rotation=90)
plt.scatter(df['Happiness Score'],df['Economy (GDP per Capita)'],color='green')
plt.xlabel("Happiness Score")
plt.ylabel("Economy")
plt.scatter(df['Happiness Score'],df['Generosity'],color='red') 
plt.xlabel("Happiness Score")
plt.ylabel("Generosity")
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
g = sns.stripplot(x="Region", y="Happiness Rank", data=df, jitter=True)
plt.xticks(rotation=90)
selectCols=  ['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Region']
sns.pairplot(df[selectCols], hue='Region',size=2.5)
SoutheasternAsia=df[df.Region=='Southeastern Asia']
SouthernAsia=df[df.Region=='Southern Asia']
EasternAsia=df[df.Region=='Eastern Asia']
f, axes = plt.subplots(3, 2, figsize=(16, 16))
axes = axes.flatten()
compareCols = ['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)']
for i in range(len(compareCols)):
    col = compareCols[i]
    axi = axes[i]
    sns.distplot(SoutheasternAsia[col],color='blue' , label='SouthEastern', ax=axi,rug='True')
    sns.distplot(SouthernAsia[col],color='green', label='Southern',ax=axi,rug='True')
    sns.distplot(EasternAsia[col],color='red',label='Eastern',ax=axi,rug='True')
    axi.legend()