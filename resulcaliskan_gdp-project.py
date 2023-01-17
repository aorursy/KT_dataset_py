import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
#location=r"C:\Users\res\Desktop\projects\GDP\w_gdp.xls"
#location=r"http://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=csv"
df=pd.read_excel("../input/1960-2017-gdp/w_gdp.xls",
                 header=[3],index_col="Country Name")
df.head()
df.iloc[10:20]
df.info()
#As we see there are some non-null values, lets clean them:
df1=df.drop(df.iloc[:,:43],axis=1)
df1=df1.drop("2017",axis=1)
df1.head()
df1.info()
#There are still null values, lets get rid of them:
df2=df1.dropna(thresh=2)
df2=df2.interpolate(method='linear',axis=1)
df2.info()
df2=df2.fillna(method="bfill",axis=1)
df2.info()
df2.head(2)
df3=df2.abs()
df3=df2.astype(int)
df3[:3]
df3.columns
#len(df3.columns)
df3.index
df3.tail()
pd.set_option('display.precision', 3) # to adjust longness of the output number.
(df3.describe()/1000000).round()   # to round the long result numbers.
df3.describe()
# transpose the df3 and plot it.
df3.T.plot.line(figsize=(20,10))
df4=df3["2016"].sort_values(ascending=False) # To sort 2016
df4.head(20)
dfmean=(df3.mean(axis=1).sort_values(ascending=False))/1000000000
dfmean.head(20)
dfmean.iloc[:20].plot(kind="bar",figsize=(15,10),color="orange")
plt.show()
df4.iloc[0:20].T.plot.bar(figsize=(20,10),fontsize=18)
#plt.plot(dfmean,color="red")
plt.xlabel("COUNTRY NAME",fontsize=18)
plt.ylabel("10 X TRILLION DOLAR ($)",fontsize=18)
plt.title("COUNTRIES GROSS DOMESTIC PRODUCT (GDP)",fontsize=18)
plt.show()
df4.iloc[0:20].T.plot.barh(figsize=(20,10),fontsize=18, color="red")
plt.xlabel("10 X TRILLION DOLAR ($)",fontsize=18)
plt.ylabel("COUNTRY NAME",fontsize=18)
plt.title("COUNTRIES GROSS DOMESTIC PRODUCT (GDP)",fontsize=18)
plt.show()
#dfc=df.loc["Turkey","1960":"2016"]
dfc=pd.DataFrame(df.loc["Turkey"],index=df.columns[3:-1])
dfc=dfc.astype(int)
dfc.info()
(dfc.describe()/1000000).round()
# this is too small but just for the seeing the forest.
plt.plot(dfc, linestyle='dashed', linewidth=2,)
plt.show() 
dfc.plot.bar(figsize=(20,10))
plt.xlabel('year', fontsize=20)
plt.ylabel('gdp', fontsize=20)
plt.title('TURKEY GDP 1960-2016')
plt.show()
dfc.plot.barh(figsize=(20,10))
plt.xlabel('year', fontsize=20)
plt.ylabel('gdp', fontsize=20)
plt.title('TURKEY GDP 1960-2016')
plt.show()
growth=pd.read_excel(r"../input/gdp-growth/GDP_growth.xlsx")
growth.head()
growth
growth.info()
growth=growth.sort_values(by=[2017],ascending=False)
first20=growth[:20]
first20
first20.plot(kind="bar",figsize=(10,5))
plt.show()
 # nice coding..