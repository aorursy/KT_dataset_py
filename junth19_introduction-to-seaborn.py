#Importing necessary modules

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style = 'darkgrid')
#Loading dataset

df = pd.read_csv("../input/tips.csv")
df.head()
df.describe()
df.info()
#Default Relational Plot is Scatter Plot

sns.relplot(x = 'total_bill', y = 'tip', data = df)
#Adding hue semantic

sns.relplot(x = 'total_bill', y = 'tip',data = df, hue = 'smoker')
#Adding different marker

sns.relplot(x = 'total_bill', y = 'tip', data = df, hue = 'smoker', style = 'smoker' )
#Changing hue colors

hue_colors = {"Yes": "black","No": "red"}

sns.relplot(x = 'total_bill', y = 'tip', data = df, hue = 'smoker', palette = hue_colors)
#Setting hue order

sns.scatterplot(x = 'total_bill', y = 'tip', data = df, hue = 'smoker', style = 'smoker', hue_order = ["Yes", "No"] )
sns.relplot(x="total_bill", y="tip", hue="size", data=df)
sns.relplot(x="total_bill", y="tip", hue="size", palette="ch:r=-.5,l=.75", data=df)
sns.relplot(x="total_bill", y="tip", hue="size",size = 'size', data=df)