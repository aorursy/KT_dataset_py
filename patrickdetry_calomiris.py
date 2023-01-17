import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.basemap import Basemap

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
calomiris = pd.read_csv('../input/CalomirisPritchett_data.csv',sep=",",na_values=[".", ""],dtype={'Name Child 7':str, 'Sex Child 7':str, 'Price':str})

calomiris = calomiris[calomiris['Reason for Omission'].isnull()]

#Price

calomiris['Price'] = calomiris['Price'].replace("233.33 1/3","233.33")

calomiris['Price'] = pd.to_numeric(calomiris['Price'])

#Notary

calomiris['notary'] = calomiris['Notary First Name']+[" "]+calomiris['Notary Last Name'].str.strip()

calomiris['notary'] = calomiris['notary'].str.replace("  ", " ").str.strip()

#Sellers

calomiris['sellers'] = calomiris['Sellers First Name']+[" "]+calomiris['Sellers Last Name'].str.strip()

calomiris['sellers'] = calomiris['sellers'].str.replace("  ", " ").str.strip()

#Buyers

calomiris['buyers'] = calomiris['Buyers First Name']+[" "]+calomiris['Buyers Last Name'].str.strip()

calomiris['buyers'] = calomiris['buyers'].str.replace("  ", " ").str.strip()

#Sales Date

calomiris['Sales Date']= calomiris['Sales Date'].str.replace("/","")

calomiris['Sales Date']= calomiris['Sales Date'].str.replace(" ","")

calomiris['Sales Date']=pd.to_datetime(calomiris['Sales Date'], format='%m%d%Y',errors='coerce')

#Age

calomiris['Age']= calomiris['Age'].fillna(calomiris['Age'].mean())



calomiris['Sex'] = calomiris['Sex'].str.replace(" ","")

calomiris = calomiris.drop (['ID number', 'Researcher'],axis =1)

calomiris = calomiris[:14713]

calomiris['Sex Child 1']=  calomiris['Sex Child 1'].astype('category')

calomiris['Sex Child 2']=  calomiris['Sex Child 2'].astype('category')

calomiris['Sex Child 3']=  calomiris['Sex Child 3'].astype('category')

calomiris['Sex Child 4']=  calomiris['Sex Child 4'].astype('category')

calomiris['Sex Child 5']=  calomiris['Sex Child 5'].astype('category')

calomiris['Sex Child 6']=  calomiris['Sex Child 6'].astype('category')

calomiris['Sex Child 7']=  calomiris['Sex Child 7'].astype('category')

calomiris['Sex Child 8']=  calomiris['Sex Child 8'].astype('category')

calomiris['total']=1

calomiris['total']=calomiris['total']+calomiris['Sex Child 1'].notnull()

calomiris['total']=calomiris['total']+calomiris['Sex Child 2'].notnull()

calomiris['total']=calomiris['total']+calomiris['Sex Child 3'].notnull()

calomiris['total']=calomiris['total']+calomiris['Sex Child 5'].notnull()

calomiris['total']=calomiris['total']+calomiris['Sex Child 6'].notnull()

calomiris['total']=calomiris['total']+calomiris['Sex Child 7'].notnull()

calomiris['total']=calomiris['total']+calomiris['Sex Child 8'].notnull()

calomiris['mean_price']=calomiris['Price']/calomiris['total']
print (calomiris.columns)

print(calomiris.describe())

calomiris.head(5)
calomale = calomiris[calomiris['Sex']=="M"]

calofemale = calomiris[calomiris['Sex']=="F"]

plt.figure(dpi=250)

plt.subplot(311)

plt.hist(calomale['Age'],bins=range(1,80,2),color='lightblue')

plt.title("men")

plt.xlabel("Age")

plt.subplot(312)

plt.hist(calofemale['Age'],bins=range(1,80,2),color='pink')

plt.title("Women")

plt.xlabel("Age")

plt.subplot(313)

sns.distplot(a=calomale['Age'], hist=False,color='lightblue',bins=range(1,80,2))

sns.distplot(a=calofemale['Age'], hist=False,color='pink',bins=range(1,80,2))
geography_buyer = pd.pivot_table(data=calomiris,index=['Buyers County of Origin'],values=['total'],aggfunc=np.sum)

geography_buyer.sort_values(by=['total'], ascending= False).head(10)
geography_seller = pd.pivot_table(data=calomiris,index=['Sellers County of Origin'],values=['total'],aggfunc=np.sum)

geography_seller.sort_values(by=['total'], ascending= False).head(10)
ts_sale= calomiris.pivot_table(index='Sales Date',values='total',aggfunc=np.sum)

ts_sale=ts_sale.resample('m').mean()

ts_sale.plot()

plt.show()
buyers = calomiris.pivot_table(index=['buyers'],values=['total','Price'],aggfunc=('sum'))

buyers.sort_values(ascending=False,by='total').head(10)

sellers = calomiris.pivot_table(index=['sellers'],values=['total','Price'],aggfunc=(np.sum),dropna=True)

sellers.columns

sellers.sort_values(ascending=False,by=['total']).head(10)
map = Basemap(projection='merc',lat_0=39,lon_0=98,resolution='c')

map.drawcoastlines(linewidth=0.25)

map.drawcountries(linewidth=0.25)

map.fillcontinents(color='coral',lake_color='aqua')