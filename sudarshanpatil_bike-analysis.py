import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as ws
ws.filterwarnings("ignore")
df  = pd.read_csv("/kaggle/input/best-buy-indian-bikes/Bikes Best Buy.csv")
df.head()
# Cleaning the names of the columns
def clean_name(name):
    return name.strip().lower().replace(" ", "_")

df.rename(columns = clean_name, inplace = True)
sns.set()
#Checking the Fuel type of bike 
sns.countplot(df["fuel"])
plt.show()
#Signifying the companies who manifatured the bikes
temp = df.company.value_counts().reset_index()
sns.barplot(y="index", x="company" , data =temp)
plt.show()
df.bike_name.unique().shape
df.describe()
# Viewing the distribution of the numeric coulumn
num_col = ['milage_(km/l)',
 'tank_size_(cc)',
 'price(inr)']
for i in num_col:
    sns.distplot(df[i])
    plt.show()
avg_per_catagaory_of_fuel =  df.groupby("fuel")["milage_(km/l)"].mean().reset_index()
avg_per_catagaory_of_fuel
