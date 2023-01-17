import pandas as pd
df =pd.read_csv('../input/BreadBasket_DMS.csv')
df2 = df['Time'].str[0:5].str.split(':',n=1,expand=True)
df2[[0,1]] = df2[[0,1]].apply(pd.to_numeric)
df2[1] = df2[1]*10/6
highmask = df2[1] > 50
lowmask = df2[1] < 50
df2.loc[highmask, 1] = 50
df2.loc[lowmask, 1] = 0
df2[0] = (df2[0].astype(str)+'.'+df2[1].astype(str).str[0:1]).astype(float)
df = pd.concat([df,df2], axis=1, sort=False)
df.pop(1)
df.head()
timelist = df[0].unique().tolist()
#df[0].value_counts()
allitems = df['Item'].unique().tolist()

newdic = {}
for i in range(len(allitems)):
    filt = df['Item'].str.contains(allitems[i])
    newdf = df[filt]
    meandf = newdf['Time'].str[0:5].str.split(':',n=1,expand=True)
    meandf[[0,1]] = meandf[[0,1]].apply(pd.to_numeric)
    meandf[1] = meandf[1]*10/6
    meandf[0] = (meandf[0].astype(str)+'.'+meandf[1].astype(str).str[0:1]).astype(float)
    count = meandf[0].count()
    mean = meandf[0].mean()
    std = meandf[0].std()
    newdic[allitems[i]]=(mean,std,count)
Results = pd.DataFrame(newdic, index=('Mean Time', 'Time STD', 'Transaction Count')).T
print(Results.sort_values(by=['Transaction Count'], ascending=False).head())
orderresults = Results.sort_values(by=['Mean Time'], ascending=True)
orderlist = orderresults.index.values
top = Results.sort_values(by='Transaction Count', ascending=False)
toplist= top.index.values
toplist = toplist[:26]
toplist
orderedtop = []
p = 0
while p < len(orderlist):
    if orderlist[p] in toplist:
        orderedtop.append(orderlist[p])
    p+=1
orderedtop.remove('NONE')
orderedtopprice = {'Pastry':2,'Toast':1.50,'Medialuna':2,'Baguette':3,'Farm House':4,'Bread':1.50,'Coffee':1,'Scandinavian':2,'Jam':0.50,'Muffin':2,'Spanish Brunch':12,'Cookies':2,'Juice':2,'Scone':2,'Hot chocolate':3,'Fudge':5,'Tea':1.50,'Brownie':2,'Tiffin':1,'Sandwich':4.5,'Alfajores':3,'Cake':3,'Soup':3,'Truffles':3,'Coke':1}
finaldict = {}
df4 = pd.DataFrame()
dfmean = pd.DataFrame()
i = 0
while i < len(allitems):
    filt = df['Item'].str.contains(allitems[i])
    df3 = df[filt]
    newseries = pd.Series(df3[0].value_counts(),name=allitems[i])
    average = newseries.max()
    meanseries = newseries/average
    df4 = pd.concat([df4,newseries], axis=1, sort=False)
    dfmean = pd.concat([dfmean,meanseries],axis=1,sort=False)
    i+=1

    
df5 = df4.drop([1.0,7.0,7.5,19.0,18.5,19.5,20.0,21.5,22.0,22.5,20.5,23.0,23.5])
dfmean = dfmean.drop([1.0,7.0,7.5,19.0,18.5,19.5,20.0,21.5,22.0,22.5,20.5,23.0,23.5])
df6 = df5[orderedtop]
dfmean = dfmean[orderedtop]
import seaborn as sns
import matplotlib.pyplot as plt
dfmean = dfmean.fillna(0)
heatmap = dfmean.T
df6 = df6.fillna(0)
heatmap2 = df6.T
plt.subplots(figsize=(20,15))
sns.heatmap(heatmap, annot=False, fmt="g", cmap='viridis')

plt.subplots(figsize=(20,15))
sns.heatmap(heatmap2, annot=False, fmt="g", cmap='viridis')
df7 =df6
for i in range(len(orderedtop)):
    df7[orderedtop[i]] = df7[orderedtop[i]]*orderedtopprice[orderedtop[i]]
priceadjusted = df7.T
plt.subplots(figsize=(20,15))
sns.heatmap(priceadjusted, annot=False, fmt="g", cmap='viridis')


