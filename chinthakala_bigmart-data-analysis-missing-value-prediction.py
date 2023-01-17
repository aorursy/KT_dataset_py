import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train=pd.read_csv('/kaggle/input/big-mart-sales-data-train-data-set/Train.csv')
df_train.head()
df_train.info()
df_train.isnull().sum()
sns.heatmap(df_train.isnull(), yticklabels=False)
weight_grp=df_train.groupby(['Item_Type','Outlet_Location_Type'])
weight_grp=weight_grp['Item_Weight'].agg(['count','min','max','mean','median']).reset_index()
weight_grp.head()
def add_weight(cols):
    weight=cols[0]
    itype=cols[1]
    location=cols[2]
    
    if pd.isnull(weight):
        for items in weight_grp.itertuples():
            if (items[1]==itype) and (items[2]==location):
                return items[6]
    else:
        return weight
df_train['Item_Weight']=df_train[['Item_Weight','Item_Type','Outlet_Location_Type']].apply(add_weight, axis=1)
df_train['Item_Weight'].head(10)
df_train.info()
df_train.isnull().sum()
sns.heatmap(df_train.isnull(), yticklabels=False)
df_train.head()
g=sns.catplot(x='Outlet_Type', y='Item_Outlet_Sales', hue='Outlet_Size', data=df_train, kind='swarm')
plt.tight_layout()
g.set_xticklabels(rotation=45)
g2=sns.catplot(x='Outlet_Location_Type', y='Item_Outlet_Sales', hue='Outlet_Size', data=df_train, kind='violin', col='Outlet_Type')
plt.tight_layout()
g2.set_xticklabels(rotation=45)
osize_grp=df_train.groupby(['Outlet_Type','Outlet_Location_Type','Outlet_Size'])
osize_grp=osize_grp['Outlet_Size'].count()
osize_grp
df_train['Outlet_Size'].value_counts()
df_train.isnull().sum()
def add_osize(cols):
    osize=cols[0]
    olocation=cols[1]
    otype=cols[2]
    
    if pd.isnull(osize):
        if (otype=='Supermarket Type2') or (otype=='Supermarket Type3'):
            return 'Medium'
        elif otype=='Grocery Store':
            return 'Small'
        else:
            if olocation=='Tier 2':
                return 'Small'
            elif olocation=='Tier 3':
                return 'High'
            else:
                return 'Small'
    else:
        return osize
        
df_train['Outlet_Size']=df_train[['Outlet_Size','Outlet_Location_Type','Outlet_Type']].apply(add_osize, axis=1)
df_train['Outlet_Size'].head(10)
sns.heatmap(df_train.isnull(), yticklabels=False)
df_train['Outlet_Size'].value_counts()
df_train.corr()
visib_grp=df_train.groupby(['Item_Visibility','Item_Type'])
visib_grp=visib_grp[['Item_Outlet_Sales']].mean().reset_index()
visib_grp
g3=sns.relplot(hue='Item_Outlet_Sales', y='Item_Visibility', data=visib_grp, x='Item_Type')
plt.tight_layout()
g3.set_xticklabels(rotation=90)
visib_grp2=df_train.groupby(['Item_Visibility','Outlet_Location_Type','Outlet_Type'])
visib_grp2=visib_grp2[['Item_Outlet_Sales']].mean().reset_index()
visib_grp2
g4=sns.relplot(x='Item_Outlet_Sales', y='Item_Visibility', data=visib_grp2, col='Outlet_Type', hue='Outlet_Location_Type')
plt.tight_layout()
g4.set_xticklabels(rotation=90)
df_train.head()
fat_grp=df_train.groupby('Item_Fat_Content')
fat_grp=fat_grp[['Item_Outlet_Sales']].mean().reset_index()
fat_grp
def add_fat(cols):
    fat=cols[0]
    if (fat=='LF') or (fat=='low fat'):
        return 'Low Fat'
    elif fat == 'reg':
        return 'Regular'
    else:
        return fat
df_train['Item_Fat_Content']=df_train[['Item_Fat_Content']].apply(add_fat, axis=1)
df_train['Item_Fat_Content'].head(10)
fat_grpn=df_train.groupby('Item_Fat_Content')
fat_grpn=fat_grpn[['Item_Outlet_Sales']].agg(['mean','count','sum']).reset_index()
fat_grpn
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3)
fat_pie1=ax1.pie(fat_grpn[('Item_Outlet_Sales','mean')], labels=fat_grpn['Item_Fat_Content'],autopct='%1.1f%%',textprops={'color':"w"})
fat_pie2=ax2.pie(fat_grpn[('Item_Outlet_Sales','sum')], labels=fat_grpn['Item_Fat_Content'],autopct='%1.1f%%',textprops={'color':"w"})
fat_pie3=ax3.pie(fat_grpn[('Item_Outlet_Sales','count')], labels=fat_grpn['Item_Fat_Content'],autopct='%1.1f%%',textprops={'color':"w"})
ax1.set_title('Mean',bbox={'facecolor':'0.8', 'pad':1})
ax2.set_title('Total Sales',bbox={'facecolor':'0.8', 'pad':1})
ax3.set_title('Item Count',bbox={'facecolor':'0.8', 'pad':1})
plt.show()
fat_grpn=df_train.groupby(['Item_Fat_Content','Item_Type'])
fat_grpn=fat_grpn[['Item_Outlet_Sales']].mean().reset_index()
fat_grpn
g5=sns.catplot(x='Item_Type', y='Item_Outlet_Sales', data=fat_grpn, hue='Item_Fat_Content')
plt.tight_layout()
g5.set_xticklabels(rotation=90)
