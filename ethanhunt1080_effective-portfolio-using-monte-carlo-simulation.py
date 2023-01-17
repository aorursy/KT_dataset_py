#importing necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
#reading the dataset using pandas and storing it in a dictionary

raw_data = {}

raw_data['INFY'] = pd.read_csv('/kaggle/input/historical-datas/INFY.csv')

raw_data['SUNPHARMA'] = pd.read_csv('/kaggle/input/historical-datas/SUNPHARMA.csv')

raw_data['ITC'] = pd.read_csv('/kaggle/input/historical-datas/ITC.csv')

raw_data['COALINDIA'] = pd.read_csv('/kaggle/input/historical-datas/COALINDIA.csv')

raw_data['SBIN'] = pd.read_csv('/kaggle/input/historical-datas/SBIN.csv')
#visualising a dataset

raw_data['SBIN']
print(', '.join(raw_data['SBIN']['Series'].unique()))

datas = {}



for name,data in raw_data.items():

    datas[name] = raw_data[name][raw_data[name]['Series']=='EQ'].reset_index()
datas['SBIN']
CP_dict = {}



for name,data in datas.items():

    CP_dict[name] = datas[name]['Close Price']

    

CP_df = pd.DataFrame(CP_dict)

CP_df
daily_return = np.log(CP_df.pct_change() + 1).dropna()

#Here I am calculating log daily returns and removing the first empty column of each stock



#Now calculating mean of those daily return

daily_return_mean = np.array(daily_return.mean())



#Now assigning weights

#Since there are five stocks so each will have a weight of 0.2 (1/5)

weights = np.array([0.2,0.2,0.2,0.2,0.2])
#Calculating Portfolio Return

Port_return = np.sum(weights * daily_return_mean)



print('The Annual Return of Portfolio is {}%'.format((Port_return * 252)*100))

#Here 252 is multiplied for annual calculation i.e. 252 trading days in a year
#Calculating Portfolio Volatility

cov = daily_return.cov()

Port_Vol = np.sqrt(np.dot(weights.T,np.dot(cov,weights)))



print('The Annual Volatility of Portfolio is {}%'.format((Port_Vol * np.sqrt(252))*100))
#Declare the number of Portfolio to be generated

num_portfolio = 30000



#creating a empty list for storing returns,volatility,sharpe_ratio(return/volatility) and weightage of each stock in portfolio

results = np.zeros((3 + len(daily_return.columns),num_portfolio))







#Monte Carlo Simulation

for i in range(num_portfolio):

    

    weight = np.random.rand(len(daily_return.columns)) #Declaring random weights

    weight = weight/np.sum(weight) #So that sum of all weight will be equal to 1



    p_annual_return = np.sum(weight * daily_return_mean) * 252 #Annual Return

    p_annual_volatility = np.sqrt(np.dot(weight.T,np.dot(cov,weight))) * np.sqrt(252) #Annual Volatility

    

    #Storing the values in results list

    results[0,i] = p_annual_return

    results[1,i] = p_annual_volatility

    results[2,i] = results[0,i]/results[1,i]



    for j in range(len(weight)):

        results[j+3,i] =  weight[j]

        

        

        

#Making a dataframe for results list of all generated Portfolio

cols = ['Ann_Ret','Ann_Vol','Sharpe_Ratio']

for num in range(len(list(daily_return.columns))):

    cols.append(list(daily_return.columns)[num])



    

result_df = pd.DataFrame(results.T,columns=cols)



#Visulising the result dataframe

result_df
#locating 

#Portfolio 1 - Sharpe ratio is the highest (Return/Volatility)

#Portfolio 2 - Volatility is the lowest

            



#Portfolio 1

max_sharpe_ratio = result_df.iloc[result_df['Sharpe_Ratio'].idxmax()]



#Portfolio 2

volatility_lowest = result_df.iloc[result_df['Ann_Vol'].idxmin()]
#Plotting the simulation

plt.figure(figsize=(15,8))

plt.scatter(result_df['Ann_Vol'],result_df['Ann_Ret'],c =result_df['Sharpe_Ratio'],cmap='RdYlBu')

plt.colorbar()



plt.scatter(max_sharpe_ratio[1],max_sharpe_ratio[0],marker = (5,1,3),color='red',s=700) #Red - Portfolio 1

plt.scatter(volatility_lowest[1],volatility_lowest[0],marker = (5,1,3),color='green',s=700)#Green - Portfolio 2

            



plt.xlabel('Volatility',fontsize = 15)

plt.ylabel('Returns',fontsize = 15)

plt.show()
print('The Portfolio with maximum Return(or Sharpe_Ratio) is:')

print('All values in percentage')

print(round(max_sharpe_ratio * 100,2))
print('The Portfolio with least Volatility is:')

print('All values in percentage')

print(round(volatility_lowest * 100,2))