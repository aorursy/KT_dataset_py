import pandas as pd
dataset=pd.read_csv('../input/honeyproduction.csv')
print(dataset.info())

print(dataset.describe())
import matplotlib.pyplot as plt
print(dataset['year'].unique())
l1=dataset['year'].unique()
l2=[]
for i in l1 :
    a=((dataset[dataset.year == i]).totalprod.sum())/len((dataset[dataset.year == i]))
    l2.append(a)

print(l2)
plt.plot(l1,l2)
plt.xlabel('Year')
plt.ylabel('Total_prod')
print(dataset['state'].unique())
print(dataset['state'].unique())
slist=dataset['state'].unique()
prodlist=[]
for i in slist:
    b=dataset[dataset.state==i].totalprod.sum()
    prodlist.append(b)
print(prodlist)   
# Get current size
fig_size = plt.rcParams["figure.figsize"]

# Prints: [8.0, 6.0]

fig_size[0] = 15
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
plt.bar(slist,prodlist)
print(prodlist.index(min(prodlist)))
print(prodlist.index(max(prodlist)))
print(" State with least production is ",slist[43])
print(" State with least production is ",slist[28])
import seaborn as sns
sns.set_style('whitegrid');
sns.pairplot(dataset,  size = 10)
plt.show()
list_of_each_year=[]
k=0
for i in slist:
    eachyearprod=[]
    for j in l1:
        c=dataset[dataset.state==i] 
        b=(c[c.year==j].totalprod.sum()) 
        eachyearprod.append(b) 
    list_of_each_year.append(eachyearprod) 
for i in list_of_each_year:
    plt.plot(l1,i) 
    plt.legend(slist)
l1=dataset['year'].unique()
yield_per_col=[]
for i in l1 :
    a=((dataset[dataset.year == i]).yieldpercol.sum())/len((dataset[dataset.year == i]))
    yield_per_col.append(a)

print(l2)
plt.plot(l1,yield_per_col)
plt.xlabel('year')
plt.ylabel('yield per col')
price_list=[]
for i in l1 :
    a=((dataset[dataset.year == i]).priceperlb.sum())/len((dataset[dataset.year == i]))
    price_list.append(a)
print(price_list)
plt.plot(l1,price_list)
plt.xlabel('Year')
plt.ylabel('Price per lb')

numcol_per_yr=[]
for i in l1 :
    a=((dataset[dataset.year == i]).numcol.sum())/len((dataset[dataset.year == i]))
    numcol_per_yr.append(a)
plt.plot(l1,numcol_per_yr)