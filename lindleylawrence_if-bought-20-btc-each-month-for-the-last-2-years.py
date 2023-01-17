# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(os.path.join(dirname, filename))
df
df.info()
# Change date to datetime object
df["Date"] = pd.to_datetime(df["Date"])
df

df.info()
# Only date and close price
df = df[["Date","Close"]]
df["Close"] = df["Close"].str.strip()
df
df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')


df
df = df.loc[df["Date"] >= "2018-09"]
df
# Get prices if bought bitcoin at for the last 2 years(24 months)
prices_bought_btc_at_list = []

for price in df["Close"]:
    prices_bought_btc_at_list.append(price)
    
prices_bought_btc_at_list


# Remove 's
prices_bought_btc_at_list_cleaned = []

for p in prices_bought_btc_at_list:
    p = p.replace(",","")
    p = float(p)
    prices_bought_btc_at_list_cleaned.append(p)

prices_bought_btc_at_list_cleaned

# Create new df
new_df = pd.DataFrame({"date": df["Date"] ,"price": prices_bought_btc_at_list_cleaned, "month" : df["Month"]})
new_df = new_df.sort_values(by=["date"],ascending=False)
new_df["current price"] = new_df["price"][0]


new_df
min_price = new_df.groupby(["month"])["price"].min()
mean_price = new_df.groupby(["month"])["price"].mean()
median_price = new_df.groupby(["month"])["price"].median()
max_price = new_df.groupby(["month"])["price"].max()

min_price_df = pd.DataFrame(min_price)
mean_price_df = pd.DataFrame(mean_price)
median_price_df = pd.DataFrame(median_price)
max_price_df = pd.DataFrame(max_price)

price_df = pd.DataFrame()
price_df["monthly_min"] = min_price_df["price"]
price_df["monthly_mean"] = mean_price_df["price"]
price_df["monthly_median"] = median_price_df["price"]
price_df["monthly_max"] = max_price_df["price"]

price_df["current_price"] = new_df["price"][0]

price_df

# Plot date and prices bought btc at
import matplotlib.pyplot as plt
import matplotlib as mpl



price_df.plot(figsize=(15,8))
price_df["current_price"].plot(color="g")
plt.xlabel("Date bought btc")

plt.xticks(rotation=45)
plt.ylabel("USD/btc price")
plt.title("If bought $20 btc each month for the last 2 years")

plt.style.use(['seaborn'])

# Calculate bitcoin for each month's purchase of 20 dollars
bitcoins_bought_at_min_list = []
bitcoins_bought_at_mean_list = []
bitcoins_bought_at_median_list = []
bitcoins_bought_at_max_list = []
for btcmin,btcmean,btcmed,btcmax in zip(price_df["monthly_min"],price_df["monthly_mean"],
                                        price_df["monthly_median"], price_df["monthly_max"]):
    minbtc_in_units = 20 / btcmin
    meanbtc_in_units = 20 / btcmean
    medbtc_in_units = 20 / btcmed
    maxbtc_in_units = 20 / btcmax
    
    bitcoins_bought_at_min_list.append(minbtc_in_units)
    bitcoins_bought_at_mean_list.append(meanbtc_in_units)
    bitcoins_bought_at_median_list.append(medbtc_in_units)
    bitcoins_bought_at_max_list.append(maxbtc_in_units)
    
btc_bought_df = pd.DataFrame()
btc_bought_df["month"] = price_df["monthly_min"]
btc_bought_df.drop("month",axis=1,inplace=True)
btc_bought_df["btc_at_min_price"] = bitcoins_bought_at_min_list
btc_bought_df["btc_at_mean_price"] = bitcoins_bought_at_mean_list
btc_bought_df["btc_at_median_price"] = bitcoins_bought_at_median_list
btc_bought_df["btc_at_max_price"] = bitcoins_bought_at_max_list

btc_bought_df
sums_of_btcs = btc_bought_df.sum()
sums_of_btcs
# Total btct to current price to give the total USD value
sums_of_btcs_val = btc_bought_df.sum() * 11758.28
sums_of_btcs_val
# Calculate return on investment
dollar_amount_spent = 20 * 24
current_bitcoin_price = 11758.28

btcs_at_min_price = sums_of_btcs[0]
btcs_at_mean_price = sums_of_btcs[1]
btcs_at_median_price = sums_of_btcs[2]
btcs_at_max_price = sums_of_btcs[3]

btc_at_min_price_value = sums_of_btcs_val[0]
btc_at_mean_price_value = sums_of_btcs_val[1]
btc_at_median_price_value = sums_of_btcs_val[2]
btc_at_max_price_value = sums_of_btcs_val[3]


return_on_investment_dollars_min = btc_at_min_price_value - dollar_amount_spent
return_on_investment_dollars_mean = btc_at_mean_price_value - dollar_amount_spent
return_on_investment_dollars_med = btc_at_median_price_value - dollar_amount_spent
return_on_investment_dollars_max = btc_at_max_price_value - dollar_amount_spent

return_on_investment_percentage_min = round(return_on_investment_dollars_min / dollar_amount_spent * 100,2)
return_on_investment_percentage_mean = round(return_on_investment_dollars_mean / dollar_amount_spent * 100,2)
return_on_investment_percentage_med = round(return_on_investment_dollars_med / dollar_amount_spent * 100,2)
return_on_investment_percentage_max = round(return_on_investment_dollars_max / dollar_amount_spent * 100,2)

print("===============================================================================================")
print("If bought bitcoin at the lowest price per month")
print("Amount spent", dollar_amount_spent)
print("Bitcoins accumulated", round(btcs_at_min_price,8))
print("Bitcoin value in dollars right now", round(btc_at_min_price_value,2))
print("Return on investment in dollars", round(return_on_investment_dollars_min,2))
print("Return on investment in percentage", return_on_investment_percentage_min, "%")


print("===============================================================================================")
print("If bought bitcoin at the average(mean) price per month")
print("Amount spent", dollar_amount_spent)
print("Bitcoins accumulated", round(btcs_at_mean_price,8))
print("Bitcoin value in dollars right now", round(btc_at_mean_price_value,2))
print("Return on investment in dollars", round(return_on_investment_dollars_mean,2))
print("Return on investment in percentage", return_on_investment_percentage_mean, "%")

print("===============================================================================================")
print("If bought bitcoin at the median price per month")

print("Amount spent", dollar_amount_spent)
print("Bitcoins accumulated", round(btcs_at_median_price,8))
print("Bitcoin value in dollars right now", round(btc_at_median_price_value,2))
print("Return on investment in dollars", round(return_on_investment_dollars_med,2))
print("Return on investment in percentage", return_on_investment_percentage_med, "%")

print("===============================================================================================")
print("If bought bitcoin at the max price per month")

print("Amount spent", dollar_amount_spent)
print("Bitcoins accumulated", round(btcs_at_max_price,8))
print("Bitcoin value in dollars right now", round(btc_at_max_price_value,2))
print("Return on investment in dollars", round(return_on_investment_dollars_max,2))
print("Return on investment in percentage", return_on_investment_percentage_max, "%")


