!pip install squarify
import squarify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
transaction.head()
identity.head()
print(f"The dataset has about {transaction.shape[0]} rows and {transaction.shape[1]+identity.shape[1]-1} columns")
transaction.columns.values
identity.columns.values
lis = ['TransactionAmt','dist1','dist2','C1','C2','C3','C4','D1','D2','D3','D4','D5']
print(f"{'Attribute':18} {'Mean':10} {'Median':10} {'Mode':10} {'Std':<14} {'Variance':10}")
for col in lis:
    print(f"{col:18}{round(transaction[col].mean(),3):10}{round(transaction[col].median(),3):10}{round(transaction[col].mode(),3)[0]:10}{round(transaction[col].std(),3):14}{'':3} {round(transaction[col].var(),3):10}")  
lis = ['id_01','id_02','id_03','id_04']
print(f"{'Attribute':18} {'Mean':10} {'Median':10} {'Mode':10} {'Std':<14} {'Variance':10}")
for col in lis:
    print(f"{col:18}{round(identity[col].mean(),3):10}{round(identity[col].median(),3):10}{round(identity[col].mode(),3)[0]:10}{round(identity[col].std(),3):14}{'':3} {round(identity[col].var(),3):10}")  
sns.set_style('whitegrid')
sns.scatterplot(transaction['D2'],transaction['D1'])
sns.boxplot(transaction['D4'],orient='v')
sns.countplot(transaction['card4'])
sns.distplot(transaction['card1'],color='salmon',kde=False)
plt.figure(figsize=(10,7))
plt.title('TREE MAP OF CARD TYPE')

color=['red','#219FB0','#32A0CE','#1C51B0']

squarify.plot(sizes=transaction['card4'].value_counts().values,label=transaction['card4'].value_counts().index,color=color, alpha=0.6)
fig = px.parallel_coordinates(transaction[['card2','card3','card5']].sample(10000))
fig.show()

