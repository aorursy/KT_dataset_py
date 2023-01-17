#import the necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None
df = pd.read_csv('../input/fertilizers-by-product-fao/FertilizersProduct.csv', encoding = 'latin-1')
df.head()
df = df.loc[df.Area == 'India']
df.reset_index(inplace = True, drop = True) 
df.drop(columns = {'Area Code','Area','Year Code','Unit','Flag'}, inplace = True)
df.head()
plt.figure(figsize=(14,8))
sns.countplot(x='Year',data=df);
plt.title('Import/Export of fertilizer in India over the years')
imp_qty = df.loc[df['Element Code'] == 5610]
imp_qty.drop(columns = {'Item Code','Element Code','Year'},inplace = True)
imp_qty = imp_qty.groupby(['Item'])["Value"].sum().reset_index().sort_values("Value",ascending=False).reset_index(drop=True)
imp_qty
slices = imp_qty.head()['Value']
labels = labels= imp_qty.head()['Item']
explode = [0.1,0.05,0,0,0]
colors = ['#fa744f','#0779e4','#a8df65','#76ead7','#f6bed6']
plt.pie(slices, labels = labels,wedgeprops= {'edgecolor': 'black'},explode = explode, shadow= True, autopct = '%1.1f%%',
        colors = colors)
plt.title('Top Five Fertiliser Imports in India 2002-17')
plt.show()
item = imp_qty.head()['Item'].unique()
k = 0
ls = ['-','--','-.',':','-']
mk = ['.','*','x','o','>']
plt.figure(figsize=(12,8))
for i in item:
    x = df.loc[df['Item'] == i]
    x = x.loc[df['Element Code'] == 5610]
    year = x['Year']
    value = x['Value']
    plt.xlabel('Year')
    plt.ylabel('Import Quantity')
    plt.plot(year, value, label = i, color = colors[k], linestyle=ls[k],linewidth = 5-k, marker = mk[k])
    k=k+1
    plt.legend()
plt.title('Top Five Fertiliser Imports in India 2002-17(tonnes)')
imp_val = df.loc[df['Element Code'] == 5622]
imp_val.drop(columns = {'Item Code','Element Code','Year'},inplace = True)
imp_val = imp_val.groupby(['Item'])["Value"].sum().reset_index().sort_values("Value",ascending=False).reset_index(drop=True)
imp_val.head(5)
slices = imp_val.head()['Value']
labels = labels= imp_val.head()['Item']
explode = [0.1,0.05,0,0,0]
colors = ['#40bad5','#f6d743','#a8df65','#481380','#f6bed6']
plt.pie(slices, labels = labels,wedgeprops= {'edgecolor': 'black'},explode = explode, shadow= True, autopct = '%1.1f%%',
        colors = colors)
plt.title('Top Five Fertiliser Import Values in India 2002-17')
plt.show()
item = imp_val.head()['Item'].unique()
k = 0
ls = ['-','--','-.',':','-']
mk = ['.','*','x','o','>']
plt.figure(figsize=(12,8))
for i in item:
    x = df.loc[df['Item'] == i]
    x = x.loc[df['Element Code'] == 5622]
    year = x['Year']
    value = x['Value']
    plt.xlabel('Year')
    plt.ylabel('Import Value')
    plt.plot(year, value, label = i, color = colors[k], linestyle=ls[k],linewidth = 5-k, marker = mk[k])
    k=k+1
    plt.legend()
plt.title('Top Five Fertiliser Import Values in India 2002-17(in $1000 US)')
exp_qty = df.loc[df['Element Code'] == 5910]
exp_qty.drop(columns = {'Item Code','Element Code','Year'},inplace = True)
exp_qty = exp_qty.groupby(['Item'])["Value"].sum().reset_index().sort_values("Value",ascending=False).reset_index(drop=True)
exp_qty
slices = exp_qty.head(8)['Value']
labels = labels= exp_qty.head(8)['Item']
explode = [0.1,0.05,0,0,0,0,0,0]
colors = ['#092532','#3ca59d','#a8df65','#bac964','#79d70f']
plt.pie(slices, labels = labels,wedgeprops= {'edgecolor': 'black'}, explode = explode, shadow  =True)
plt.title('Top Eight Fertiliser Exports in India 2002-17')
plt.show()
item = exp_qty.head()['Item'].unique()
k = 0
ls = ['-','--','-.',':','-']
mk = ['.','*','x','o','>']
plt.figure(figsize=(12,8))
for i in item:
    x = df.loc[df['Item'] == i]
    x = x.loc[df['Element Code'] == 5910]
    year = x['Year']
    value = x['Value']
    plt.xlabel('Year')
    plt.ylabel('Export Quantity')
    plt.plot(year, value, label = i, color = colors[k], linestyle=ls[k],linewidth = 5-k, marker = mk[k])
    k=k+1
    plt.legend()
plt.title('Top Five Fertiliser Exports in India 2002-17(tonnes)')
exp_val = df.loc[df['Element Code'] == 5922]
exp_val.drop(columns = {'Item Code','Element Code','Year'},inplace = True)
exp_val = exp_val.groupby(['Item'])["Value"].sum().reset_index().sort_values("Value",ascending=False).reset_index(drop=True)
exp_val
slices = exp_val.head(10)['Value']
labels = labels= exp_val.head(10)['Item']
explode = [0.1,0.05,0,0,0,0,0,0,0,0]
plt.pie(slices, labels = labels,wedgeprops= {'edgecolor': 'black'}, shadow= True, explode = explode)
plt.title('Top Ten Fertiliser Export Values in India 2002-17')
plt.show()
item = exp_val.head()['Item'].unique()
k = 0
ls = ['-','--','-.',':','-']
mk = ['.','*','x','o','>']
colors = ['#111d5e','#5fdde5','#e79cc2','#b2ebf2','#79d70f']
plt.figure(figsize=(12,8))
for i in item:
    x = df.loc[df['Item'] == i]
    x = x.loc[df['Element Code'] == 5922]
    year = x['Year']
    value = x['Value']
    plt.xlabel('Year')
    plt.ylabel('Export Values')
    plt.plot(year, value, label = i, color = colors[k], linestyle=ls[k],linewidth = 5-k, marker = mk[k])
    k=k+1
    plt.legend()
plt.title('Top Five Fertiliser Export Values in India 2002-17(in $1000 US)')
prod = df.loc[df['Element Code'] == 5510]
prod.drop(columns = {'Item Code','Element Code','Year'},inplace = True)
prod = prod.groupby(['Item'])["Value"].sum().reset_index().sort_values("Value",ascending=False).reset_index(drop=True)
prod
slices = prod.head()['Value']
labels = labels= prod.head()['Item']
explode = [0.1,0,0,0,0]
colors = ['#b7472a','#fbfd8a','#726a95','#fee2b3','#0e9aa7']
plt.pie(slices, labels = labels,wedgeprops= {'edgecolor': 'black'},explode = explode, shadow= True, autopct = '%1.1f%%',
        colors = colors)
plt.title('Top Five Produced Fertilisers in India 2002-17')
plt.show()
item = prod.head()['Item'].unique()
k = 0
ls = ['-','--','-.',':','-']
mk = ['.','*','x','o','>']
plt.figure(figsize=(12,8))
for i in item:
    x = df.loc[df['Item'] == i]
    x = x.loc[df['Element Code'] == 5510]
    year = x['Year']
    value = x['Value']
    plt.xlabel('Year')
    plt.ylabel('Produced Quantity')
    plt.plot(year, value, label = i, color = colors[k], linestyle=ls[k],linewidth = 5-k, marker = mk[k])
    k=k+1
    plt.legend()
plt.title('Top Five Produced Fertilisers in India 2002-17(in tonnes)')
agri = df.loc[df['Element Code'] == 5157]
agri.drop(columns = {'Item Code','Element Code','Year'},inplace = True)
agri = agri.groupby(['Item'])["Value"].sum().reset_index().sort_values("Value",ascending=False).reset_index(drop=True)
agri
slices = agri.head(7)['Value']
labels = labels= agri.head(7)['Item']
explode = [0.1,0.05,0,0,0,0,0]
plt.pie(slices, labels = labels,wedgeprops= {'edgecolor': 'black'},shadow=True,explode = explode)
plt.title('Most Used Fertilisers in India 2002-17')
plt.show()
plt.figure(figsize=(12,8))
item = agri.head(7)['Item'].unique()
k = 0
ls = ['-','--','-.',':','-','--','-.']
mk = ['.','*','x','o','>','<','d']
colors = ['#111d5e','#43d8c9','#fee2b3','#ff9c71','#654062','#fe346e','#ffb2a7']
for i in item:
    x = df.loc[df['Item'] == i]
    x = x.loc[df['Element Code'] == 5157]
    year = x['Year']
    value = x['Value']
    plt.xlabel('Year')
    plt.ylabel('Agriculture Use')
    plt.plot(year, value, label = i, color = colors[k], linestyle=ls[k],linewidth = 7-k, marker = mk[k])
    k=k+1
    plt.legend()
plt.title('Most Used Fertilisers in India 2002-17(in tonnes)')
