import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../input/craigslist-carstrucks-data/vehicles.csv")
df.head()
df.describe()
df.info()
# Display the missing values
plt.figure(figsize=(12,12))
plt.title("Missing values for each column")
sns.heatmap(df.isnull())
plt.show()
# Count the values by year avoiding value under 1900, 
df[df.year >= 1900].year.value_counts().sort_index().plot(lw = 4)
plt.title("Number of vehicles in the dataset by build year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()
df.condition.value_counts().plot.bar()
plt.title("Condition of the vehicles")
plt.show()
df.title_status.value_counts().plot.bar()
plt.title("Status of the vehicles")
plt.show()
idx1 = df[df["condition"] == "salvage"].index

for w in ["salvage","lien","missing","parts only"]:
    idx2 = df[df["title_status"] == w].index
    idx1 = idx1.union(idx2)
    
df.drop(idx1, axis = 0, inplace = True)
print(f"Maximum price: {df.price.max()} $\nMinimum price: {df.price.min()} $")
df = df[(df["price"] >= 100) & (df["price"] <= 100000)]
sns.boxplot(df.price)
plt.title("Repartition of the price after deleting price over 100.000$")
plt.show()
print(f"Higher year: {df.year.max()}\nLowest year: {df.year.min()}")
df = df[df.year.notnull()]
df["age"] = df.year.apply(lambda x: int(2020-x))
df = df[(df.age >= 0) & (df.age <= 30)]
sns.distplot(df.age, hist = False)
plt.title("Distribution of the age of the vehicles")
plt.show()
df.type.value_counts(dropna=False).plot(kind = "bar")
plt.title("Number of each type of vehicle:")
plt.show()
# Delete the NaN
df = df[df["type"].notnull()]

# Delete "other","bus" and "offroad"
for v in ["other","bus", "offroad"]:
    df = df[df["type"] != v]
cols = ["price","age", "odometer"]
sns.heatmap(df[cols].corr(), annot = True)
plt.title("Correlation:")
plt.show()
# Images of cars for the graphics
images = {'all':'https://i.imgur.com/1vNeS3S.png',
         'SUV':'https://i.imgur.com/hDAAIQ1.png', 
         'wagon':'https://i.imgur.com/AScvovW.png', 
         'sedan':'https://i.imgur.com/geFnoDw.png',
         'convertible':'https://i.imgur.com/OJyUNkl.png',
         'pickup':'https://i.imgur.com/RZI2aBP.png',
         'hatchback':'https://i.imgur.com/I6nKBgU.png',
         'truck':'https://i.imgur.com/d5ImbCK.png',
         'coupe':'https://i.imgur.com/zf6cHos.png',
         'van':'https://i.imgur.com/ly3Fg5V.png',
         'mini-van':'https://i.imgur.com/CfmLXIG.png'}

def display_price(df, age = (0,12), price = (100,100000), vehicle_type = "all", state = "all"):
    # Display the median price of vehicles depending on its type and its state.
    
    if state != "all":
        df = df[df["state"] == state]
    
    if vehicle_type != "all":
        df = df[df["type"] == vehicle_type]
        
    df = df[(df["age"] <= age[1]) & (df["age"] >= age[0])]
    
    df = df[(df["price"] >= price[0]) & (df["price"] <= price[1])]
    
    price_age = pd.pivot_table(df, values = "price", index = "age", aggfunc= np.median)
    price_age.columns = ["Median Price"]
    
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_axes([0,0,1,1])
    ax2 = fig.add_axes([0.6,0.47,.35,.35])
    
    ax.plot(price_age["Median Price"], lw = 5)
    
    ax2.imshow(plt.imread(images[vehicle_type]))
    ax2.set_title(f"Vehicle type: {vehicle_type}\nNumber of vehicles: {df.shape[0]}\nCountry: USA\nUS-State: {state}", fontsize = 15)
    ax2.axis('off')
    
    ax.set_title(f"Median price by age of the vehicles",fontsize=25)
    ax.set_ylim(0,price_age["Median Price"].max()+1000)
    ax.set_xlabel("Age", fontsize = 15)
    ax.set_ylabel("Median price in $", fontsize = 15)
    
    ax.tick_params(axis='both', which='major', labelsize=15) 

    plt.show()
display_price(df, vehicle_type="all")

for t in df.type.unique()[:3]:
    display_price(df, vehicle_type=t)
for t in df.type.unique()[3:6]:
    display_price(df, vehicle_type=t)
for t in df.type.unique()[6:9]:
    display_price(df, vehicle_type=t)
for t in df.type.unique()[9:]:
    display_price(df, vehicle_type=t)