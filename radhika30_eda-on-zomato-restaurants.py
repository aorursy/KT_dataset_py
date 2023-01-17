#importing all required libraries for analysis 
import pandas as pd
import numpy as np

#importing data using pandas
data = pd.read_csv("../input/zomato.csv", encoding = "ISO-8859-1")
country = pd.read_excel("../input/Country-Code.xlsx")
data.head()

country.head()   
data = pd.merge(data, country, on = "Country Code") #merging the two data frames to access country name
data.head(3)
entries = pd.value_counts(data.Country) #shows frequency distribution of data according to countries
print(entries) #since India has the highest data, we shall look into only India's data for further analysis
India_data = data[data.Country=="India"]
India_data.head()
city_entries = pd.value_counts(India_data.City)
print(city_entries)
#since the data is skewed to National Capital Region (New Delhi, Gurgaon, Noida and Faridabad), we shall be analysing data of NCR

NCR = ["New Delhi", "Noida", "Gurgaon", "Faridabad"]
our_data = India_data[India_data.City.isin(NCR)] #filtering data for cities passed in NCR list

cities = pd.value_counts(our_data.City)
print(cities) #re-checking work
# importing additional libraries for data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.rcParams["figure.figsize"] = 15,4
import warnings
warnings.filterwarnings("ignore")

our_data.columns
vis1 = sns.countplot(x = our_data["Aggregate rating"]).set_title("No. of restaurants with respective Zomato rating") #countplot to show spread of restaurant ratings on Zomato
#since over 2,000 restaurants have 0 ratings, I shall remove them from our analysis

#dropping rows with 0 rating
NCR_data = our_data.ix[our_data["Aggregate rating"]!= 0]
vis2 = sns.countplot(x = NCR_data["Aggregate rating"]).set_title("Rating of NCR Restaurants") #looks like normal distribution to me!
#splitting cuisines into primary and secondary
NCR_data["Primary Cuisine"], NCR_data["Secondary Cuisine"] = NCR_data["Cuisines"].str.split(',',1).str
NCR_data.head()
count_cuisine = pd.value_counts(NCR_data["Primary Cuisine"])
top_cuisines = pd.Series(count_cuisine).sort_values(ascending=False)
vis3 = sns.barplot(x= top_cuisines[:10].index, y = top_cuisines[:10].values, palette = "BuGn_r").set_title("Popularity of cuisines")
#North Indian is the most popular cuisine followed by Chinese 
#top 10 places in NCR for North Indian cuisine restaurants
NI = NCR_data[NCR_data["Primary Cuisine"] == "North Indian"]
NI.head()
count_loc = pd.value_counts(NI["Locality"])
top_loc = pd.Series(count_loc).sort_values(ascending = False)
vis4 = sns.barplot(x = top_loc[:10].index, y = top_loc[:10].values, palette = "coolwarm").set_title("Distribution of North Indian cuisine restuarants in NCR")
#Connaught Place has the highest number of North Indian (NI) cuisines
#average rating of NI cuisine restaurants in Connaught Place
rest_CP = NI[NI["Locality"] == "Connaught Place"]
vis5 = sns.countplot(x = rest_CP["Aggregate rating"], palette = "inferno").set_title("Rating of North Indian cuisine restaurants in Connaught Place")
#surprisingly CP has very few restaurants with rating 4 and above
#rating distribution for NI cuisine in NCR
vis6 = sns.countplot(x = NI["Aggregate rating"], palette = "cubehelix").set_title("Rating of North Indian cuisine restaurants in NCR")
rating_4 = pd.value_counts(NI["Aggregate rating"] >= 4)
print(rating_4)

#Despite being the most popular cuisine, NI has highest number of resturants falling between 3.0 t0 3.2 Zomato rating
# Only 109 NI restaurants have rating 4 and above in NCR, out of 1,097 restaurants

NCR_data.columns

vis7 = sns.countplot(x = NCR_data["Has Table booking"]).set_title("Restaurants with table booking")
vis8 = sns.countplot(x = NCR_data["Has Online delivery"]).set_title("Restaurants which deliver on Zomato")
rest_del = NCR_data[NCR_data["Has Online delivery"]== "Yes"]
vis9 = sns.distplot(rest_del["Aggregate rating"], color = "Blue", bins = 20).set_title("Rating of restaurants which deliver online")
rest_del.describe()

rest_delna = NCR_data[NCR_data["Has Online delivery"]== "No"]
vis10 = sns.distplot(rest_delna["Aggregate rating"], color = "Black", bins = 20).set_title("Rating of restaurants which dont have online delivery")
rest_delna.describe()

plt.rcParams["figure.figsize"] = 10,10
vis11 = sns.boxplot(x = NCR_data["Average Cost for two"], 
                    y= NCR_data["Has Online delivery"],palette = "Accent")
# % of Fine Dine (FD) Restaurant 
a = rest_delna["Aggregate rating"].shape[0]
b = rest_delna[rest_delna["Aggregate rating"] >=4].shape[0]
c = round((b/a)*100,2)
print(c)
#closer look into restaurants with rating 4 and above
goodrating = NCR_data[NCR_data["Aggregate rating"] >=4]
plt.rcParams["figure.figsize"] = 8,6
vis12 = sns.boxplot(x = goodrating["Average Cost for two"], palette = "inferno").set_title("Cost for Rating 4 & above restaurants")
NCR_data["avg_cost"] = pd.cut(NCR_data["Average Cost for two"],bins = [0, 200, 500, 1000,  2500, 5000, 7500, 10000],
                                  labels = ["0", "<=200", "<=500", "<=1,000","<=2500", "<=5000", "<=7500",])

vis13 = sns.boxplot(x = NCR_data["avg_cost"], y = NCR_data["Aggregate rating"]).set_title("Zomato ratings v/s Cost")
vis14 = sns.countplot(x = NCR_data["Rating text"]).set_title("Rating text")

