import os

import pandas as pd

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/windows-store/msft.csv")
df.describe()
import matplotlib.pyplot as plt

from matplotlib import gridspec

import seaborn as sns

df.corr() #Alright, from here we can generally say this dataset cannot be used for ML.

#Cool. Let's start our plots!

def low_rating(rating):

  return "low rating" if rating < 4 else "high rating"

df["rating level"] = df["Rating"].apply(lambda x: low_rating(x))

sns.pairplot(df,hue="rating level", palette="husl")
df.isnull().sum()
df[df["Name"].isnull()] 
#Here we go. We can definitely drop this one.

df.drop(index=5321,axis=0,inplace=True)
# Normalize the price and change them to US dollars. (currency for July 27th)

df["Price"]=df["Price"].str.replace(",", '')

df["Price"]=df["Price"].str.replace("₹", '')

# Convert the Price into float

def price_normalization(price):

 return 0 if price=="Free" else (float(price)*0.013) #Indian Rupee to US Dollar

df["Price"]=df["Price"].apply(lambda x: price_normalization(x))
#Set up the sorts for seaborn plot.

rating_count_sort = df.groupby(["Rating"]).count().reset_index().sort_values("Name", ascending=False)

rating_count_sort = rating_count_sort[["Rating","Name"]]



category_count_sort = df.groupby(["Category"]).count().reset_index().sort_values("Name", ascending=False)

category_count_sort = category_count_sort[["Category","Name"]]
plt.figure(figsize=(18,6))

gs = gridspec.GridSpec(1, 2, width_ratios=[2,3])

ax1 = plt.subplot(gs[0])

ax2 = plt.subplot(gs[1]) 



rating_count_plot = sns.countplot(y="Rating",data=df,order=rating_count_sort["Rating"]\

                                  ,palette="Blues_d",ax=ax1)

rating_count_plot.set(xlabel="Count of Values", ylabel="Rating")



category_count_plot = sns.countplot(y="Category",data=df,order=category_count_sort["Category"]\

                                    ,palette="Blues_d",ax=ax2)

category_count_plot.set_yticklabels(category_count_plot.get_yticklabels(), rotation=40, ha="right")

category_count_plot.set(xlabel="Count of Values", ylabel="Category")
# Now lets see the hist distribution of rating.

sns.distplot(df["Rating"], bins=10)

plt.xlim(1,5)
# I did not draw a hist for price because most of prices are "Free"

# Now we can create a group for price range. 

def group_price(price):

  if price==0:

    return "FREE"

  elif price>0 and price<=1:

    return "Below $1"

  elif price>1 and price<=3:

    return "$1-$3"

  elif price>3 and price<=5:

    return "$3-$5"

  else:

    return "Above $5"

df["Price Group"] = df["Price"].apply(lambda x: group_price(x))  
df.groupby(["Price Group"]).count()
#df[df["Price"]!=0]["Price"].mean() # The average price for non-free product is $4.8

#df[df["Price"]!=0]["Price"].median() # The average price for non-free product is $3.4
# Lets see the average rating for different price group.

avg_rating_by_price = df.groupby(["Price Group"]).mean().reset_index()#.sort_values("Name", ascending=False)

avg_rating_by_price = avg_rating_by_price[["Price Group","Rating"]] 

sns.barplot(x="Price Group", y="Rating", data=avg_rating_by_price, capsize=.2,palette="Set3",\

            order=["FREE","Below $1","$1-$3","$3-$5","Above $5"])

# It does not show very significant pattern. We don't have enough data for paid product as well. 
# Is there a correlation between number of people rated and the ratings?

# Kind of already knew there won't be, but lets PLOT.

plt.figure(figsize=(12,6))

sns.jointplot("Rating","No of people Rated", data=df,

                  kind="hex",color="darkslateblue")

# No significant pattern shows here.