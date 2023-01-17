# We'll start by importing the pandas library
import pandas as pd
# Since we'll be generating numbers, we'll need to import the numpy library
import numpy as np
countries = pd.DataFrame({"Country":['France', 'Germany', 'Spain', 'Belgium',
                                  'Russia'], "GDP(2017)":[2500, 3600, 1300, 500, 1600],})
countries
pd.DataFrame(np.random.normal(2.5, 1, (10,3)), 
             columns= ["1st column", "2nd column", "3rd Column"])
dataset = pd.read_csv("../input/fortune1000.csv")
dataset.head(3)
# The head and tail methods give us the first and last n elements in our datasets, n=5 by default
dataset.info()
dataset.describe()
# let's for example check the type of this output
type(dataset.describe())
dataset.describe()['Profits']
dataset.describe()['Profits'].count()
dataset['Sector'].unique()
dataset['Sector'].nunique()
dataset['Sector'].value_counts().head()
dataset.set_index("Rank", inplace = True)
dataset.head(3)
dataset["Revenue"].head()
dataset["Profits"].nlargest(3)
dataset["Profits"].nsmallest(4)
dataset.groupby("Sector").describe().head()
type(dataset.groupby("Sector").describe())
# We can grab the single column we want
dataset.groupby("Sector").describe()["Profits"].head()
dataset.groupby("Sector").agg('mean').head()
dataset.groupby("Sector").agg({"Profits":['min','max'],"Revenue":['mean','median']}).head()
dataset["Location"].head()
dataset["Location"].apply(lambda loc:loc.split(',')[1]).head()
dataset["Location"].str.strip().str.split(',').str.get(1).head()
dataset["State"] = dataset["Location"].apply(lambda loc:loc.split(',')[1])
dataset.head()
dataset["State"].nunique()
dataset.groupby("State").agg({'Profits': ['min', 'max', 'mean']}).head()
def rate_profit(profit):
    if profit <0:
        return "Negative"
    elif (profit >0) & (profit <=3500):
        return "Average"
    else:
        return "High"
rate_profit(200)
dataset["Rating"] = dataset["Profits"].apply(rate_profit)
# Check the output
dataset.head(3)
dataset.groupby("Rating").agg('count')["Company"]
dataset["Rating"].map({'High':'AAA', 'Average':'BAA', 'Negative':'BBB'}).head()
dataset.head(3)
dataset[dataset["Rating"] == "High"].head()
dataset[dataset["Profits"].between(1500,3000)].head()
dataset[(dataset["State"].str.contains('CA'))& (dataset["Rating"] =="High")].head()
dataset[(dataset["Sector"] == "Health Care") &(dataset["Profits"]>3000)].head()
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12,6]
# Histogram of the Revenues
dataset["Revenue"].plot(kind = 'hist', bins = 100)
dataset["State"].value_counts().plot(kind = 'bar')
np.log(dataset[dataset["Profits"]>0]['Profits']).plot(kind = 'box')
plt.title("Logarithm of the Profit")
plt.ylabel("log")
dataset.hist(by = "Rating", column= "Revenue", bins = 50)
plt.show()
dataset.plot.scatter("Revenue", "Profits")
model_data = pd.get_dummies(dataset, columns= ["Rating"], drop_first= True)
model_data.head(3)
np.sum(model_data.isna())
# We can also find the percentage
(np.sum(model_data.isna())/len(model_data.index))*100
# Locate where are the missing values
model_data[model_data.isna().any(axis = 1)].head()
missing_index = model_data[model_data.isna().any(axis = 1)].index
model_data.fillna(method='ffill').head()
model_data.interpolate(inplace=True)
model_data.iloc[list(missing_index),].head()
