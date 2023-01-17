import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline



#set the style 

sns.set_style('whitegrid')

plt.rcParams['figure.figsize'] = 14,7

plt.style.use("seaborn")
#load the data

zomato_data = pd.read_csv("../input/zomato.csv")
zomato_data.head()
#shape of the dataset

zomato_data.shape
zomato_data.columns
zomato_data.info()
#get the datatypes of the columns

zomato_data.dtypes
#count of data types

zomato_data.get_dtype_counts()
#basic stats

zomato_data.describe() #only for votes
#check for missing values



pd.DataFrame(round(zomato_data.isnull().sum()/zomato_data.shape[0] * 100,3), columns = ["Missing"])
#check for any duplicate values

zomato_data.duplicated().sum()
#cleaning the column names

zomato_data.columns
zomato_data.rename(columns={"approx_cost(for two people)": "cost_two", "listed_in(type)":"service_type", "listed_in(city)":"serve_to"},

                   inplace = True)
#dropping the url and address column - because they are not very useful in data analysis

zomato_data.drop(["url", "address",  "phone"], axis = 1, inplace = True)

zomato_data.head()
#Manipulating the rate column - rate is read as object, but for analysis we need that to be present in numerical format.



zomato_data.rate.unique()
#removing the "/5" in the rate column

zomato_data.rate = zomato_data.rate.astype('str')

zomato_data.rate = zomato_data.rate.apply(lambda x: x.replace('/5','').strip())
#rate column contains 'NEW' and '-' replacing those with nan and drop those fields without any rating

# Replace "NEW" & "-" to np.nan

zomato_data.rate.replace(('NEW','-'),np.nan,inplace =True)
#dropping the observations where rate and cost_two is null

zomato_data.dropna(subset = ["rate", "cost_two"], inplace = True)

#Converting Rate Column datetype to float

zomato_data.rate = zomato_data.rate.astype('float')
#online_order and book_table are given as 'Yes' and 'No'. Converting these two True and False for better manipulation.

zomato_data.online_order.replace(('Yes','No'),(True,False),inplace =True)

zomato_data.book_table.replace(('Yes','No'),(True,False),inplace =True)
#converting the cost_two variable to float.

zomato_data.cost_two = zomato_data.cost_two.apply(lambda x: int(x.replace(',','')))
#converting to int

zomato_data.cost_two = zomato_data.cost_two.astype('int')
zomato_data.head()
#lets plot the distribution of votes

plt.rcParams['figure.figsize'] = 14,7

sns.distplot(zomato_data["votes"], kde=False,bins=5,color="y")

plt.title("Distribution of votes")

plt.ylabel("Count")

plt.show()
#plot the count of rating.

plt.rcParams['figure.figsize'] = 14,7

sns.countplot(zomato_data["rate"], palette="Set1")

plt.title("Count plot of rate variable")

plt.show()
#lets check if there is any relationship between rate and votes



plt.scatter(zomato_data["rate"], zomato_data["votes"], marker='+',color="purple",cmap = "viridis")

plt.xlabel("rating")

plt.ylabel("votes")

plt.title("Scatter plot between rate and votes")

plt.show()
sns.jointplot(x = "rate", y = "votes", data = zomato_data, height=8, ratio=4, color="g")

plt.show()
#similarly lets plot the relationship between rate and cost_two



sns.jointplot(x = "rate", y = "cost_two", data = zomato_data, height=8, ratio=4, kind = "kde", space=0, color="g")

plt.show()
sns.heatmap(zomato_data.corr(), annot = True, cmap = "viridis",linecolor='white',linewidths=1)

plt.show()
plt.rcParams['figure.figsize'] = 14,7

zomato_data.location.value_counts().nlargest(10).plot(kind = "barh")

plt.title("Number of restaurants by location")

plt.xlabel("Count")

plt.show()
plt.rcParams['figure.figsize'] = 14,7

zomato_data.serve_to.value_counts().nlargest(10).plot(kind = "barh")

plt.title("Number of restaurants listed in a particular location")

plt.xlabel("Count")

plt.show()
plt.rcParams['figure.figsize'] = 14,7

sns.countplot(zomato_data["online_order"], palette = "Set2")

plt.show()
#lets check if restaurants listed online offer delivery or not.

plt.rcParams['figure.figsize'] = 14,7

sns.countplot(zomato_data["online_order"], palette = "Set2", hue = zomato_data["service_type"])

plt.show()
#checking whether online_order impacts rating of the restaurant

sns.countplot(hue = zomato_data["online_order"], palette = "Set1", x = zomato_data["rate"])

plt.title("Distribution of restaurant rating over online order facility")

plt.show()
#rating vs booking table

sns.countplot(hue = zomato_data["book_table"], palette = "Set2", x = zomato_data["rate"])

plt.title("Distribution of restaurant rating over booking table facility")

plt.show()
#Use catplot() to combine a countplot() and a FacetGrid. This allows grouping within additional categorical variables

g = sns.catplot(x="book_table", hue="service_type", col="online_order", data=zomato_data, kind="count")
#check the restaurant service type



zomato_data.service_type.value_counts().plot(kind = "pie", autopct='%.1f%%')

plt.show()
#ratings vs service type

sns.boxplot(x="service_type", y="rate", data = zomato_data)

plt.show()
#lets plot swarmplot and violin plot together better understanding of rating vs service type



sns.violinplot(x = "service_type", y = "rate",data = zomato_data,palette="rainbow")

plt.show()
plt.rcParams['figure.figsize'] = 14,7

plt.subplot(1,2,1)

zomato_data.name.value_counts().head().plot(kind = "barh", color = sns.color_palette("hls", 5))

plt.xlabel("Number of restaurants")

plt.title("Biggest Restaurant Chain (Top 5)")



plt.subplot(1,2,2)

zomato_data[zomato_data['rate']>=4.5]['name'].value_counts().nlargest(5).plot(kind = "barh", color = sns.color_palette("Paired"))

plt.xlabel("Number of restaurants")

plt.title("Best Restaurant Chain (Top 5) - Rating More than 4.5")

plt.tight_layout()
plt.rcParams['figure.figsize'] = 14,7

zomato_data.rest_type.value_counts().nlargest(10).plot(kind = "barh", color = sns.color_palette("hls", 10))

plt.xlabel("Count")

plt.title("Top Restaurant Type (Top 10)")

plt.show()