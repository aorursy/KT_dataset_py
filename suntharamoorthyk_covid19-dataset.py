import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
test=pd.read_csv("../input/corona-virus-report/country_wise_latest.csv" )
test
test.info()
test.shape
test.describe()
sns.set_style('whitegrid')
sns.scatterplot(test['New cases'],test['New deaths'])
sns.boxplot(test['Deaths'],orient='v')
sns.countplot(test['WHO Region'])
sns.distplot(test['Confirmed'],color='blue',kde=False)

lis = ['Confirmed','Deaths','Recovered','Active','New cases','New recovered','New deaths',]
print(f"{'Attribute':18} {'Mean':10} {'Median':10} {'Mode':10} {'Std':<14} {'Variance':10}")
for col in lis:
    print(f"{col:18}{round(test[col].mean(),3):10}{round(test[col].median(),3):10}{round(test[col].mode(),3)[0]:10}{round(test[col].std(),3):14}{'':3} {round(test[col].var(),3):10}")  
test.columns.values
