#Retails store analysis
import numpy as np

import pandas as pd

data1=pd.DataFrame(np.random.randint(10,100,(10000,1)))

data1['Customer_id']=np.arange(1,10001)

data1['Store']=np.random.randint(1,15,10000)

data1['Sales']=np.random.randint(10,100,10000)

data1['Units']=np.random.randint(10,100,10000)

data1['Sales']=np.random.randint(5,16,10000)

data1['Gross profit']=data1['Sales']*.30

data1['Transactions']=np.random.randint(1,6,10000)

data1['HML']=np.random.randint(1,4,10000)

data1['Age']=np.random.randint(10,100,10000)

data1['Gender']=np.random.randint(0,2,10000)

data1['Baseket Value']=data1['Sales']/data1['Transactions']

data1['Baseket Size']=data1['Units']/data1['Transactions']

data1


data1.drop(0,axis=1,inplace=True)

data1

s=[]

for i in data1['Store']:

    s.append('Str' + str(i))

data1['Store']=s   



c=[]

for i in data1['Customer_id']:

    c.append('Cust_id '+str(i))

data1['Customer_id']=c   

#c
data1
data1.describe().T


data1.columns









data1.groupby('Gender').sum()[['Sales','Transactions','Units','Baseket Value','Baseket Size','Gross profit']]
k=[]

data1['Values']=np.random.randint(0,2,10000)

for i in data1['Values']:

        if i==1 :

            k.append('New Customer')

        else :

            k.append('Existing Customer')

data1['Values']=k
data1.head(20)



data1.groupby('Values').sum()[['Sales','Transactions','Units','Baseket Value','Baseket Size','Gross profit']]
 # New customer vs Existing customer



data1.groupby('Values').sum()[['Sales','Transactions','Units','Baseket Value','Baseket Size','Gross profit']]



data1.groupby('Values').sum()[['Sales','Transactions','Units','Baseket Value','Baseket Size','Gross profit']].describe().T

###


data1.groupby('Values').sum().T.describe().T 


# #Age	Sales	Units	Trasaction	Basket Value	Basket Size	Gross Profits

# 0-18						

# 18-24						

# 24-35						

# 35-50						

# 50+						

# 


x=[]

for i in (data1['Age']):

    if i >= 0 and i < 18:

        x.append('0-18')

    elif i >= 18 and i < 24:

        x.append('18-24')

    elif i >= 24 and i < 35:

        x.append('24-35')

    elif i >= 35 and i < 50:

        x.append('35-50')

    else:

        x.append('50')

data1['Age']=x
data1.groupby('Age').sum()[['Sales','Transactions','Units','Baseket Value','Baseket Size','Gross profit']]


data1.groupby('Age').sum()[['Sales','Transactions','Units','Baseket Value','Baseket Size','Gross profit']].describe().T





# In[28]:







data1.groupby(['Gender','Age']).sum().T
data1.groupby(['Gender','Age']).sum().T.describe().T




data1.groupby(['Gender','Age']).sum().T













data1.groupby(['Gender','Age']).sum()
data1.groupby(['Values','Age']).sum().T
#End******************************************************************************************