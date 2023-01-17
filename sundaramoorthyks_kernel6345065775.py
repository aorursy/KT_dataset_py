# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data= pd.read_excel("/kaggle/input/gowtham/Test Data.xlsx", sheet_name="Sheet1")
data.head()
data.iloc[:,1].unique() #only 5 products
len(data.iloc[:,2].unique()) # 32 dates
import seaborn as sns

sns.distplot(data.iloc[:,3]) # range between 10 -25 
sns.distplot(data.iloc[:,4]) # amount 2500 max
# create a new feature with the amount * qty and latest date

len(data.iloc[:,0].unique())
len(data.iloc[:,0]) # only 9 is unique customers
data.columns
data["f_1"]=data['Qty']*data["Amount"]
data.groupby("Product ")["f_1"].sum() # sum
p_1,p_2,p_3,p_4,p_5=[],[],[],[],[]

score=[]



for c in range(1,10):

    customer="C_"+str(c)

    print(customer)

    c_df=(data[data["Customer"]==customer])

    o_df=c_df.groupby("Product ")["f_1"].sum().reset_index()

   # print(o_df.head())

    p_1.append((o_df.iloc[0,1]/1705960)*100)

    p_2.append((o_df.iloc[1,1]/1551633)*100)

    p_3.append((o_df.iloc[2,1]/1554841)*100)

    p_4.append((o_df.iloc[3,1]/1461854)*100)

    p_5.append((o_df.iloc[4,1]/1468240)*100)

    score.append(p_1[-1]+p_2[-1]+p_3[-1]+p_4[-1]+p_5[-1])

    #print(score)

   

group=[]

for s in score:

    if s>=60:

        group.append(1)

    elif s<50:

        group.append(3)

    else:

        group.append(2)
group