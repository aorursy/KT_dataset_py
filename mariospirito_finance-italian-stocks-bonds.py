import os  #you can read the currently directory
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt
import numpy as np
import datetime
# IMPORT AND CLEANING THE DATA FROM EXCEL

df_PI = pd.read_excel('../input/italian-stocks-bonds/italian_stocks_bonds.xlsx', sheet_name='stocks', converters= {'Name': pd.to_datetime},)

type(df_PI)
                      
print(os.getcwd())
df_PI.info()
df_PI.columns=(["Date", "FTSEMIB Index", "FCA", "UCG", "ISP", "ENI", "LUX"])
df_PI.head(20)
df_PI.drop([0,1,2,3,4], inplace= True)
df_PI.head(20)
df_PI.info()
df_PI[["FTSEMIB Index", "FCA", "UCG", "ISP", "ENI", "LUX"]] = df_PI[["FTSEMIB Index", "FCA", "UCG", "ISP", "ENI", "LUX"]].astype(float)
df_PI
df_dataindex= df_PI.set_index(["Date"])
df_PI= df_dataindex
df_PI
df_PI.info()
# I am creating daily returns for the stock index and for the stocks
# I am assuming continous coumpounding
df_PI["FTSEMIB DAILY RET"]=np.log(df_PI["FTSEMIB Index"]/ df_PI["FTSEMIB Index"].shift(1))
df_PI["FCA DAILY RET"]=np.log(df_PI["FCA"]/ df_PI["FCA"].shift(1))
df_PI["UCG DAILY RET"]=np.log(df_PI["UCG"]/ df_PI["UCG"].shift(1))
df_PI["ISP DAILY RET"]=np.log(df_PI["ISP"]/ df_PI["ISP"].shift(1))
df_PI["ENI DAILY RET"]=np.log(df_PI["ENI"]/ df_PI["ENI"].shift(1))
df_PI["LUX DAILY RET"]=np.log(df_PI["LUX"]/ df_PI["LUX"].shift(1))

df_PI[["FTSEMIB DAILY RET","FCA DAILY RET", "UCG DAILY RET", "ISP DAILY RET","ENI DAILY RET", "LUX DAILY RET"]].tail()  
df_PI
# Plotting the return

df_PI["FTSEMIB DAILY RET"].cumsum().plot(subplots=True, figsize = (10,6), lw=4) #FTSEMIB_return
df_PI["FCA DAILY RET"].cumsum().plot(subplots=True, figsize = (10,6), color = 'red') #FCA_return
df_PI["UCG DAILY RET"].cumsum().plot(subplots=True, figsize = (10,6), color = 'green') #UCG_return
df_PI["ISP DAILY RET"].cumsum().plot(subplots=True, figsize = (10,6), color = 'purple') #ISP_return
df_PI["ENI DAILY RET"].cumsum().plot(subplots=True, figsize = (10,6), color = 'yellow') #ENI_return
df_PI["LUX DAILY RET"].cumsum().plot(subplots=True, figsize = (10,6), color = 'orange') #LUX_return
plt.xlabel('Date')
plt.ylabel('return')
plt.title('return of main indexes of FTSEMIB')
plt.legend()
# DataFrame with: SKEW and KURTOSIS
df_df = pd.DataFrame([(df_PI.iloc[:,0].skew(),
                       df_PI.iloc[:,1].skew(),
                       df_PI.iloc[:,2].skew(),
                       df_PI.iloc[:,3].skew(),
                       df_PI.iloc[:,4].skew(),
                       df_PI.iloc[:,5].skew()),
                      (df_PI.iloc[:,0].kurtosis(), 
                      df_PI.iloc[:,1].kurtosis(),
                      df_PI.iloc[:,2].kurtosis(),
                      df_PI.iloc[:,3].kurtosis(),
                      df_PI.iloc[:,4].kurtosis(),
                      df_PI.iloc[:,5].kurtosis())],
                     columns = ['FTSEMIB', 'FCA', 'UNICREDIT', 'ISP', 'ENI', 'LUX'], 
                     index = ['SKEW', 'KURTOSIS'])

df_df
#Plotting the distributions of returns

grid = plt.GridSpec(2,3, wspace = 0.4, hspace=0.3)
plt.figure(figsize=(10,6))

plt.subplot(grid[0,0]).hist(df_PI.iloc[:,6], color = 'black', bins=55)
plt.title('FTSEMIB')
plt.ylabel('frequency')

plt.subplot(grid[0,1]).hist(df_PI.iloc[:,7], color = 'blue', bins=55)
plt.title('FCA')
plt.ylabel('frequency')

plt.subplot(grid[0,2]).hist(df_PI.iloc[:,8], color = 'green', bins=55)
plt.title('UCG')
plt.ylabel('frequency')

plt.subplot(grid[1,0]).hist(df_PI.iloc[:,9], color = 'brown', bins=55)
plt.title('ISP')
plt.xlabel('Price')
plt.ylabel('frequency')

plt.subplot(grid[1,1]).hist(df_PI.iloc[:,10], color = 'red', bins=55)
plt.title('ENI')
plt.xlabel('Price')
plt.ylabel('frequency')

plt.subplot(grid[1,2]).hist(df_PI.iloc[:,11], color = 'yellow', bins=25)
plt.title('LUX')
plt.xlabel('Price')
plt.ylabel('frequency')
#how about the correletions...
axes = ['FTSEMIB', 'FCA', 'UCG', 'ISP', 'ENI', 'LUX']

harvest = np.array(round(df_PI.iloc[:,6:].corr(),2))

fig, ax = plt.subplots()
im = ax.imshow(harvest) #colors change based on the values

#set the tickets
ax.set_xticks(np.arange(len(axes)))
ax.set_yticks(np.arange(len(axes)))

#give label to the axes
ax.set_xticklabels(axes)
ax.set_yticklabels(axes)

#setting the labels
plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', rotation_mode = 'anchor')

#loop over data dimension and create text annotations
for i in range(len(axes)):
    for j in range(len(axes)):
        text = ax.text(j,i, harvest[i,j],
                      ha = 'center', va = 'center', color = 'w', size=13)
#mean of returns computed over a rolling window of 60 days
df_PI['FTSEMIB 60days'] = df_PI["FTSEMIB DAILY RET"].rolling(window=60,center=False).mean() # 60 days average
df_PI['FCA 60days'] = df_PI["FCA DAILY RET"].rolling(window=60,center=False).mean() # 60 days average
df_PI['UCG 60days'] = df_PI["UCG DAILY RET"].rolling(window=60,center=False).mean() # 60 days average
df_PI['ISP 60days'] = df_PI["ISP DAILY RET"].rolling(window=60,center=False).mean() # 60 days average
df_PI['ENI 60days'] = df_PI["ENI DAILY RET"].rolling(window=60,center=False).mean() # 60 days average
df_PI['LUX 60days'] = df_PI["LUX DAILY RET"].rolling(window=60,center=False).mean() # 60 days average

df_PI[['FTSEMIB 60days','FCA 60days','UCG 60days','ISP 60days','ENI 60days','LUX 60days']].plot(figsize=(10,5))
plt.xlabel("Date")
plt.ylabel("return")
plt.title("Returnd of indexes")
plt.legend()
#standard deviation computed over a rolling window of 60 days
df_PI['FTSEMIB vol'] = df_PI['FTSEMIB DAILY RET'].rolling(window=60,center=False).std() # 60 days volatility
df_PI['FCA vol'] = df_PI['FCA DAILY RET'].rolling(window=60,center=False).std() # 60 days volatility
df_PI['UCG vol'] = df_PI['UCG DAILY RET'].rolling(window=60,center=False).std() # 60 days volatility
df_PI['ISP vol'] = df_PI['ISP DAILY RET'].rolling(window=60,center=False).std() # 60 days volatility
df_PI['ENI vol'] = df_PI['ENI DAILY RET'].rolling(window=60,center=False).std() # 60 days volatility
df_PI['LUX vol'] = df_PI['LUX DAILY RET'].rolling(window=60,center=False).std() # 60 days volatility


df_PI[['FTSEMIB vol','FCA vol','UCG vol','ISP vol','ENI vol','LUX vol']].plot(subplots = True,style = 'b', 
                                                                              figsize=(10,5))


df_DR= df_PI[1:] [["FTSEMIB DAILY RET","FCA DAILY RET", "UCG DAILY RET", "ISP DAILY RET","ENI DAILY RET", "LUX DAILY RET"]]
df_DR
df_DR.plot(subplots=True,grid=True,style='b',figsize=(8,6))
#1st regression
import numpy as np
import statsmodels.formula.api as smf
mod = smf.ols( "Q('FCA DAILY RET') ~ Q('FTSEMIB DAILY RET')", df_DR).fit()
print(mod.summary())
print(mod.params) # print coefficients
#2th regression
import numpy as np
import statsmodels.formula.api as smf
mod = smf.ols( "Q('UCG DAILY RET') ~ Q('FTSEMIB DAILY RET')", df_DR).fit()
print(mod.summary())
print(mod.params) # print coefficients
#3th  regression
import numpy as np
import statsmodels.formula.api as smf
mod = smf.ols( "Q('ISP DAILY RET') ~ Q('FTSEMIB DAILY RET')", df_DR).fit()
print(mod.summary())
print(mod.params) # print coefficients
#4th reggression
import numpy as np
import statsmodels.formula.api as smf
mod = smf.ols( "Q('ENI DAILY RET') ~ Q('FTSEMIB DAILY RET')", df_DR).fit()
print(mod.summary())
print(mod.params) # print coefficients
#5th regression
import numpy as np
import statsmodels.formula.api as smf
mod = smf.ols( "Q('LUX DAILY RET') ~ Q('FTSEMIB DAILY RET')", df_DR).fit()
print(mod.summary())
print(mod.params) # print coefficients
del df_DR["FTSEMIB DAILY RET"]
df_DRmean=df_DR.mean()
df_DRmean
beta=[1.396233, 1.468434, 1.297887, 0.798773, 0.276803, ]

col=["red", "green", "blue", "yellow","orange"]
plt.scatter(beta,df_DRmean, color= col)

plt.xlabel("β of each stock")
plt.ylabel("Average daily stock returns")

plt.text(1.396233,-0.000100, "FCA")
plt.text(1.468434, -0.000716,  "UCG")
plt.text(1.297887,-0.000490,  "ISP")
plt.text(0.798773,-0.000058,  "ENI")
plt.text(0.276803,-0.000009, "LUX")

plt.grid(True)
plt.show()
df_bonds = pd.read_excel('../input/italian-stocks-bonds/italian_stocks_bonds.xlsx', sheet_name='bonds', converters= {'Name': pd.to_datetime})

type(df_bonds)
df_bonds.info()
df_bonds.columns=(["Date", "3M","1Y", "2Y", "5Y", "10Y"])
df_bonds.drop([0,1,2,3,4], inplace= True)
df_bonds[["3M","1Y", "2Y", "5Y", "10Y"]] = df_bonds[["3M","1Y", "2Y", "5Y", "10Y"]].astype(float)
df_bonds

df_index= df_bonds.set_index(["Date"])
df_bonds= df_index
df_bonds
df_bonds.info()
df_bmean= df_bonds.mean()
df_bmean
df_bonds.iloc[0,0:]
df_bonds.iloc[48,0:]
meanyield = df_bmean

plt.figure(figsize=(8,5))
df_bonds.iloc[0,0:].plot(style='r',lw=2)
df_bonds.iloc[48,0:].plot(style='b--',lw=2)
meanyield.plot(style='m-.',lw=2)
plt.axis ('tight')
plt.legend(['30/10/2015','31/10/2019','meanyield'])
plt.title('yield curve')

plt.ylabel('%')
import scipy.interpolate as spi
x = [3,12,24,60,120]
xx= np.linspace(3,120,25)

ipo = spi.splrep(x,meanyield,k=3)
iy = spi.splev(xx,ipo)
plt.figure(figsize=(10,6))
plt.plot(x,meanyield,'r.',label='data points',markersize=20)
plt.plot(xx,iy,'b--',label='cubic spline', lw=3)
plt.plot(x,meanyield,'k',label='linear', color = 'green', lw=2)
plt.axis ('tight')
plt.legend(['data points','cubic spline','linear'])
plt.title('Yield Curve')
plt.xlabel('months')
plt.ylabel('%')
df_bonds_copy = df_bonds.copy()
df_bonds_copy.iloc[:,1] = 100/(1+df_bonds.iloc[:,1]/100)**(3/12)
df_bonds_copy.iloc[:,1].mean()