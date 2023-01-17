import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd # pandas is a python module, used to process and analyze data.
data = pd.read_csv("../input/RCOM.csv")
# creating a filter to contain 1 at rows in series contating 'EQ'

Series_filter = data['Series'] == 'EQ'
data_only_EQ = data[Series_filter]
data_only_EQ.reset_index(inplace = True, drop = True) # resetting the index value unless it will not be continuous due to dropped rows
data_only_EQ['Series'].unique() # checking unique values in 'Series' column 
data_only_EQ.head() # printing first tuples of data only containing EQ
data_only_EQ.tail() #printing last 5 tuples
data_only_EQ.describe() # using describe to find more information about the columns of data with only 'EQ' category
data_only_EQ['Date'] = pd.to_datetime(data_only_EQ['Date'])
# After this operation the 'Date' column will become the index



data_only_EQ.set_index('Date', inplace=True, drop = False) #to use the functionality of using last we need to make Date to be the index
price = 'Close Price'

sent  = "price for the last 90 days is "



print("Maximum " + sent, data_only_EQ[price].last('90D').max())

print("Minimum " + sent, data_only_EQ[price].last('90D').min())

print("Mean " + sent, data_only_EQ[price].last('90D').mean())
# Analyze the datatype of each column

data_only_EQ.info()
# print the largest and smallest value of Date to find the range of Data collected



print(data_only_EQ.index.min(), " to ", data_only_EQ.index.max())



print("NO. of days :",data_only_EQ.index.max()- data_only_EQ.index.min())
data_only_EQ = pd.concat([data_only_EQ, pd.DataFrame({'Transactions': data_only_EQ['Average Price']*data_only_EQ['Total Traded Quantity']})], axis = 1)
#calculate monthwise VWAP
month_wise_VWAP = data_only_EQ['Transactions'].groupby(data_only_EQ['Date'].dt.strftime('%B')).sum() / data_only_EQ['Total Traded Quantity'].groupby(data_only_EQ['Date'].dt.strftime('%B')).sum()
print(month_wise_VWAP)
#Calculate year wise VWAP
year_wise_VWAP = data_only_EQ['Transactions'].groupby(data_only_EQ['Date'].dt.strftime('%Y')).sum() / data_only_EQ['Total Traded Quantity'].groupby(data_only_EQ['Date'].dt.strftime('%Y')).sum()
print(year_wise_VWAP)
# Function to calculate the average price over the last N days of the stock price

def calculate_average_price_over_last_N(n, stmp = 'D'):

    

    return data_only_EQ['Close Price'].last(str(n) + stmp).mean()
calculate_average_price_over_last_N(2)
def calculate_profit_loss_over_last(n, stmp = 'D'):

    k = len(data_only_EQ)

    curr = data_only_EQ['Close Price'][k-1]

    n_days_old = data_only_EQ['Close Price'].last(str(n) + stmp)[0]

    

    return ((curr - n_days_old)/n_days_old)*100
calculate_profit_loss_over_last(90) # '+' sign indicate a profit and '-' sign indicate a loss
def calculate_the_average_price_and_profit_loss(num, stmp):

    

    print("Average Price over last " + str(num) + stmp+ " is", calculate_average_price_over_last_N(num, stmp))



    val = calculate_profit_loss_over_last(num, stmp)



    if val == 0.0:

        print("No profit and No loss")

    elif val > 0:

        print("Profit over last {} {} is {:.2f}%".format(num, stmp, val))

    else:

        print("Loss over last {} {} is {:.2f}%".format(num, stmp, val*-1))

    
# pass 'D' for days, 'W' for weeks, 'M' for months and 'Y' for 'Year' 

# calculate_the_average_price_and_profit_loss(3, 'Y')



# Average price and profit/loss over last 1 week

calculate_the_average_price_and_profit_loss(1, 'W') # Here is only one day in the last week

print()



# Average price and profit/loss over last 2 week

calculate_the_average_price_and_profit_loss(2, 'W')

print()



# Average price and profit/loss over last 1 month

calculate_the_average_price_and_profit_loss(1, 'M')

print()



# Average price and profit/loss over last 3 month

calculate_the_average_price_and_profit_loss(3, 'M')

print()



# Average price and profit/loss over last 6 month

calculate_the_average_price_and_profit_loss(6, 'M')

print()



# Average price and profit/loss over last 1 year

calculate_the_average_price_and_profit_loss(1, 'Y')
data_only_EQ.insert(data_only_EQ.shape[1], 'Day_Perc_Change', data_only_EQ['Close Price'].pct_change())
data_only_EQ = data_only_EQ.iloc[1:,:] # removing the first row containing the Nan value
data_only_EQ.insert(data_only_EQ.shape[1], 'Trend', 'Slight')
# using .loc to update the value in the column

data_only_EQ.loc[data_only_EQ['Day_Perc_Change'].between(-0.005, -0.01, inclusive = True), 'Trend'] = 'Slight negative'



data_only_EQ.loc[data_only_EQ['Day_Perc_Change'].between(0.005, 0.01, inclusive = True), 'Trend'] = 'Slight positive'

data_only_EQ.loc[data_only_EQ['Day_Perc_Change'].between(-0.03, -0.01, inclusive = False),'Trend'] = 'Negative'



data_only_EQ.loc[data_only_EQ['Day_Perc_Change'].between(0.01, 0.03, inclusive = False),'Trend'] = 'Positive'

data_only_EQ.loc[data_only_EQ['Day_Perc_Change'].between(-0.07, -0.03, inclusive = True),'Trend'] = 'Top losers'



data_only_EQ.loc[data_only_EQ['Day_Perc_Change'].between(0.03, 0.07, inclusive = True),'Trend'] = 'Top gainers'

data_only_EQ.loc[data_only_EQ['Day_Perc_Change']<-0.07,'Trend']= 'Bear drop'



data_only_EQ.loc[data_only_EQ['Day_Perc_Change']>0.07,'Trend'] = 'Bear run'
data_only_EQ['Trend'].unique() # unique values in the 'Trend' column
# The average of the 'Total Traded Quantity' for each types of 'Trend'

data_only_EQ.groupby('Trend')['Total Traded Quantity'].mean()
# The median values of the 'Total Traded Quantity' for each types of 'Trend'

data_only_EQ.groupby('Trend')['Total Traded Quantity'].median()
data_only_EQ.to_csv('week2_RCOM.csv') #saving the dataframe 'data_only_EQ' to csv file 'week2.csv'