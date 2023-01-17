# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualization
%matplotlib inline
import seaborn as sns #for visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# We dont use plt.show() if we use "% matplotlib inline"
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
store = pd.read_csv("../input/googleplaystore.csv",usecols = [0,1,2,3,4,5,6,7,8,9,10]) 

#I dont wanna use "Current Ver" and "Android Ver" so ı use "usecols" 
store.info()
store.head() #quick look at the csv from first index
store.tail() #quick look at the csv from last index
store.sample(6) #random 
store.columns
store.columns = store.columns.str.replace(" ","_") #adding "_" to columns which has space.
store.dtypes
store.Size = store.Size.replace("Varies with device",np.nan)
store.Size = store.Size.str.replace("M","000")
store.Size = store.Size.str.replace("k","")
#store.Size = store.Size.apply(lambda x: float(x.replace("k",""))/1000 if "k" in x else x)
#ı wanted use this but ı get error so ı added "000" to Megabyte

store.Size = store.Size.replace("1,000+",1000)

store.Installs = store.Installs.str.replace(",","")
store.Installs = store.Installs.apply(lambda x: x.strip("+"))
store.Installs = store.Installs.replace("Free",np.nan)

store.Price = store.Price.str.replace("$","")

store = store.drop(store.index[10472])

store[["Size","Installs","Reviews","Price"]] = store[["Size","Installs","Reviews","Price"]].astype("float")
store.Category = store.Category.astype("category")
store.Installs = pd.to_numeric(store.Installs)
store.Price = pd.to_numeric(store.Price)
store = store.drop_duplicates(subset = "App", keep = "first")
store.dtypes
store.corr()
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(store.corr(), annot = True, linewidths = .5, fmt = ".2f", ax=ax)
#I wanna seperate free and paid apps so I can analysis objectively
free = store[store.Type == "Free"]
paid = store[store.Type == "Paid"]
store.head()
#We can see clearly reviews decreasing as the price increases.
paid.plot(kind = "line",x = "Price", y = "Reviews", color = "r", linestyle = ":", alpha = .5, 
          grid = True, linewidth = 1, figsize = (12,6))
plt.xlabel("Price")
plt.ylabel("Reviews")
plt.title("Paid App-Reviews")

# We can comprasion free and paid app reviews
free.Reviews.plot(kind = "line", color = "g", linestyle = ":", alpha = .7, 
          grid = True, linewidth = 1, figsize = (12,6), label = "Free")
paid.Reviews.plot(kind = "line", color = "r", linestyle = "-.", alpha = 1, 
          grid = True, linewidth = 1, figsize = (12,6), label = "Paid")
plt.legend()
plt.xlabel("İndex")
plt.ylabel("Reviews")
plt.title("Free-Paid")

#We can see this plot nearly almost cheap applications has  high rating.
paid.plot(kind = "scatter", x = "Rating", y = "Price", alpha = .5, color = "b")
plt.xlabel("Rating")
plt.ylabel("Price")

store.Rating.plot(kind = "hist", bins = 50, figsize = (12,6))
plt.xlabel("Rating")
plt.title("Rating of Distribution")

store.Category.value_counts().plot(kind='barh',figsize= (12,8))

store.Content_Rating.value_counts().plot(kind="bar")
plt.yscale("log")
def expensive(count=5):
    exp = store.sort_values(by=["Price"],ascending = False).head(count)
    return exp
expensive(10)
#expensive applications
exp = paid["Price"]>100
paid[exp]
x = paid[(paid.Price>100) & (paid.Rating>4)]
x
for index,value in paid[["App"]].head().iterrows():  #iterrows gives us index and value. İt is like "enumerate" in list
    print(index,":",value)
paid_head = paid.head()
paid_head
melted = pd.melt(frame = paid_head, id_vars = "App",value_vars = ["Installs","Price"])
melted
melted.pivot(index = "App", columns = "variable", values = "value")
#Adding from row axis = 0
f1 = free.head()
f2 = free.tail()
conc = pd.concat([f1,f2],axis = 0, ignore_index = True)
conc
#Adding from column axis = 1
app = store.App.head()
rate = store.Rating.head()
conc = pd.concat([app,rate], axis = 1)
conc
store.isnull().sum() #detect missing values
store.Size.value_counts(dropna=False) #1227 NaN value
store1 = store.copy()
store1.Size.dropna(inplace = True) #dropped nan values
assert store1.Size.notnull().all() #checking with assert . return nothing because we drop nan values
store1.Size.fillna(0,inplace = True) #filling nan values with "0"
paid1 = paid.copy()
paid1["total_money"] = paid1.Installs*paid1.Price
paid1.head()
paid.head(75).describe()
paid.head(75).boxplot(column = "Price", by = "Rating", figsize = (20,6))
store.plot(kind = "hist", y = "Rating", bins = 50,range = (0,5), normed = True, figsize = (12,6))
fig,axes = plt.subplots(nrows = 2, ncols = 1)
store.plot(kind = "hist", y = "Rating", bins =50, range = (0,5), normed = True, ax = axes[0])
store.plot(kind = "hist", y = "Rating", bins =50, range = (0,5), normed = True, ax = axes[1], cumulative = True)
plt.savefig("graph.png")
store1 = store.copy()
store1["Last_Updated"] = pd.to_datetime(store1.Last_Updated)
store1.head()
store1 = store1.set_index("Last_Updated")
store1.head()
#we can select according to our date index
store1.loc["2018-01-01":"2018-01-02"] #this show us which app updated in 2days
store1.resample("A").mean() #this is getting an average to store for year(A) or month(M)
store1.resample("M").mean()
# As you can see there are a lot of nan because store1 does not include all months
store1.resample("M").first().interpolate("linear")
store1.resample("M").first().interpolate("linear")
store1.resample("M").mean().interpolate("linear")
store2 = store.copy()
store2.head()
store2 = store2.set_index(["Category","Type"])
#store2.sample(50)
paid.pivot(index = "App", columns = "Content_Rating", values = "Price")
store.head()
store.info()
store.index = range(1,9660 ,1) #our index is starting with "0" I wanna chage with "1"
store.head()
store1 = store.copy()
store1 = store1.set_index(["Category","Type"])
store1.head(50)
store2 = store.copy()
store2.groupby("Type").mean()
store2.groupby("Type").Price.max()
store2.groupby("Type")[["Installs","Rating"]].max()
