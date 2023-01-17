#import libraries and list the files in the input directory

import os

print(os.listdir("../input")) #list the docs

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plot graphs

import seaborn as sns #plotting interactive graphs

import random #to use in random choices

from datetime import datetime as dt #for date conversion

%matplotlib inline 

#to include graphs inline within the frontends next to code
df = pd.read_csv("../input/googleplaystore.csv") #load googleplaystore data in a dataframe(df)

print(df.head()) #have a look at the first 5 columns
print(df.shape) #gives number of rows and columns
print("Name of the columns are:", "\n", df.columns) #look at the column names
print("Dataframe has following datatypes:", "\n", df.dtypes) #look at the datatypes in the dataframe

print("Number of null values per column:", "\n", df.isnull().sum()) #look at # of null values per column
#look at the App name column if they have unique values

print("Statistics of App Column:", df["App"].describe().to_dict())



#drop NA values of the df:

df.dropna(axis=0, how='any', inplace=True)

print("After the clean-up:",df.shape)
#look at the distribution of app ratings in a distplot

sns.set(font_scale=1.5, style = "whitegrid") #set the font size and background

f, axes = plt.subplots(figsize=(8,6), sharex=True) #set the plotsize



g = sns.distplot(df.Rating, hist=False, color="g", kde_kws={"shade": True})

plt.ylabel("Frequency")

plt.title("Distribution of App Ratings")
plt.figure(figsize=(8,6))

plt.hist(df.Rating, range=(1,5), bins=16)

plt.axvline(x=df.Rating.mean(), linewidth=4, color='g', label="mean")

plt.axvline(x=df.Rating.median(), linewidth=4, color='r', label="median")

plt.xlabel("App Ratings")

plt.ylabel("Count")

plt.title("Histogram of App Ratings")

plt.legend(["mean", "median"])

plt.show()
#look at the ratings per app category in a sequential order



#prepare data and sort

new_df = df.groupby("Category").Rating.mean().reset_index() #average ratings per category

sorted_df = new_df.sort_values(by=["Rating"], ascending=True)  #sort by rating in descending order



sns.set(font_scale=1.5, style="whitegrid")

f, axes = plt.subplots(figsize=(15, 6), sharex=True)



#since we have categorical data in the x axis, we will look data with a barplot

ax = sns.barplot(x="Category", y="Rating", data=sorted_df, palette="Blues") 

for item in ax.get_xticklabels():

    item.set_rotation(90) #rotate every xticklabels for readability

ax.set(ylim=(3.5,5)) #zoom in y axes since values are distributed around 4

plt.xlabel("Average Rating")

plt.title("Average Rating per Category", size=20)



#look at the number of apps per category

f, axes = plt.subplots(figsize=(15, 6), sharex=True)



#since we are going to count # of apps per each category we are going to use barplot

ax = sns.countplot(x="Category", data=df, palette="Set3")

plt.ylabel("Number of Apps")

plt.title("Number of Apps per Category", size=20)

for item in ax.get_xticklabels():

    item.set_rotation(90)
#load data into numpy arrays

business = list(df[df.Category == "BUSINESS"].Rating)

dating = list(df[df.Category == "DATING"].Rating)



# 1) samples should be normally distributed: two samples resembles normal distribution

sns.kdeplot(business)

sns.kdeplot(dating)

plt.title("Rating distributions")



# 2) 2 populations standard deviations must be equal: standard deviations are equal

business_array = np.asarray(business)

dating_array = np.asarray(dating)



print("Standard deviation of business app ratings:", business_array.std())

print("Standard deviation of dating app ratings:", dating_array.std())



# 3) two distributions are already independent from each other
from scipy.stats import ttest_ind #import statistics library to run the tests

#confidence interval: 95%

#setting confidence interval sets our alpha (treshold value) = 1-0.95 = 0.05



#Null Hypothesis: Difference in the mean rating of Business and Dating apps are due to a random chance

#Alternative Hypothesis: Mean rating of Business and Dating apps are significantly different



#p-value = when it is assumed that our null hypothesis is correct, p value gives us the probability of

#getting a sample with the results we assumed.



#run the 2 sample test:



_, pvalue = ttest_ind(business, dating)

if pvalue <= 0.05:

    print("Reject Null Hypothesis")

else:

    print("Accept Null Hypothesis") 
#remember that # of reviews are in object dtype

df.Reviews = df.Reviews.apply(lambda x: int(x)) #convert object into int



#look at the total reviews per install category

total_reviews = df.groupby('Installs').Reviews.sum().reset_index()

sorted_total_reviews = total_reviews.sort_values(by='Reviews', ascending=False).reset_index(drop=True)

print(sorted_total_reviews.head(5))
#look at the distribution of the reviews per top-review install category

#since this a distribution per categorical data boxplot will be plotted

g = sns.catplot(x="Installs",

                y="Reviews",

                data=df[(df.Installs == "1,000,000,000+") | (df.Installs == "500,000,000+") | 

                        (df.Installs == "100,000,000+")], 

                kind="box", height = 8 ,palette = "Set2")

plt.ticklabel_format(style='plain', axis='y')

#g.set_yticklabels(["0","10M", "20M", "30M", "40M", "50M", "60M", "70M", "80M"])

plt.title("Distribution of Reviews for Popular Apps", size=20)
print(df.App[df.Reviews>60000000].unique())
#we are going to look the ratio of installs to review per app



df['Int_installs'] = df.Installs.replace(to_replace = ['\,','\+'], value=['',''], regex=True) 

#create new column, remove special characters for integer conversion



df.Int_installs = df.Int_installs.astype('int64') 

#change data type from string to integer



df['Review_to_Install_Ratio'] = df.Reviews / df.Int_installs
#then plot the distribution per each popular install category

f, axes = plt.subplots(1, 3, figsize=(35, 10), sharex=True) #set the plotsize, divide plot into 3 columns



g1 = sns.kdeplot(df.Review_to_Install_Ratio[df.Installs == "1,000,000,000+"], shade=True, ax=axes[0], color="blue")

g1.title.set_text("Distriution of Reviews per Download in 1 Billion Installed Apps")



g2 = sns.kdeplot(df.Review_to_Install_Ratio[df.Installs == "500,000,000+"], shade=True, ax=axes[1], color="green")

g2.title.set_text("Distriution of Reviews per Download in 500 Million Installed Apps")



g3 = sns.kdeplot(df.Review_to_Install_Ratio[df.Installs == "100,000,000+"], shade=True, ax=axes[2],color="red")

g3.title.set_text("Distriution of Reviews per Download in 100 Million Installed Apps")
#plot a swarmplot since there are multiple categories (Installs and Type)

sns.set(font_scale=1.5, style="whitegrid")

fig, ax = plt.subplots(figsize=(30,20))

ax = sns.swarmplot(x="Installs", y="Rating", data=df, hue="Type", palette="Set2", dodge=True)

for item in ax.get_xticklabels():

    item.set_rotation(45)

plt.title("Ratings per Type and Install Category", size=20)
# plot a pie chart

labels = df.Type.unique() #set labels

sizes = [len(df[df.Type == "Free"]), len(df[df.Type == "Paid"])] #count the number of free and paid apps

explode = (0, 0.2) #emphasize "Paid" apps



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90, colors=["palegreen","orangered"]) #plot pie chart

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show() #render plot

#we are going to plot multiple linear regressions to answer this question

#linear regression looks for the correlation between continuous variables

#price was string so change it to integer



df['Decimal_price'] = df.Price.replace(to_replace = '\$', value='', regex=True)

#add new column and remove special characters



df.Decimal_price = df.Decimal_price.astype('float')

#change to float



fig, ax = plt.subplots(figsize=(14,8))

sns.regplot(x="Decimal_price", y="Rating", data=df)

plt.title("Price VS Rating", size=20)
#look at the price distribution to determine price bins

g2 = sns.distplot(df.Decimal_price, hist=False, color="orange", kde_kws={"shade": True})

plt.axvline(x=df.Decimal_price.mean(), c="g", linewidth=1)

plt.title("Price Distribution")

print("Mean price in the dataframe is:", df.Decimal_price.mean())
#price shows a bimodal distribution around 0 and 400 with a right-skew

#we are going to group price data into 10 categories

#I am going to create more categories around 0

def categorize(x):

    if x==0:

        return 0

    if (x > 0) & (x < 0.5):

        return 1

    if (x >= 0.5) & (x < 1):

        return 2

    if (x>=1) &  (x < 2):

        return 3

    if (x>=2) &  (x < 3):

        return 4

    if (x>=3) &  (x < 5):

        return 5

    if (x>=5) &  (x < 10):

        return 6

    if (x>=10) &  (x < 25):

        return 7

    if (x>=25) &  (x < 100):

        return 8

    else:

        return 9
#change price data into categorical data

#plot lmplot since we have changed the data into categorical data

df["Categorical_price"] = df.Decimal_price.apply(categorize)

sns.lmplot(x="Categorical_price", y="Rating", data=df, height=8.27, aspect=14.1/8.27)

plt.title("Categorical Price VS Rating")
#look at the last updated column

print(df['Last Updated'].head())



#change the date column to a date format from object type

df["Update_date"] = df['Last Updated'].apply(lambda x: dt.strptime(x, '%B %d, %Y').date()) 



#fetch update year from date

df["Update_year"] = df["Update_date"].apply(lambda x: x.strftime('%Y')).astype('int64') 



fig, ax = plt.subplots(figsize=(14,8))

sns.regplot(x="Update_year", y="Rating", data=df)

plt.title("Update Year VS Rating")
print(len(df.Genres.unique()))

print(df.Genres.unique())

#looks like we have genres and its sub genre seperated by semi colons, format is:

#main_genre; sub_genre



print(df["Content Rating"].unique())

#we have 5 content ratings and one not categorized:

#unrated
#divide genre columns and clean Content Rating

df["Main_genre"] = df.Genres.apply(lambda x: x.split(";")[0])

df["Sub_genre"] = df.Genres.apply(lambda x: x.split(";")[1] if x.find(";")>0 else "NA")

df["Content Rating"] =df["Content Rating"].replace(to_replace = 'Unrated', value='Everyone', regex=True)
#we are going to look at the data with the stacked bars with pandas dataframe

#create count view per main_genre and content rating

df_by_main_genre = df.groupby(["Main_genre", "Content Rating"]).count().reset_index().sort_values(

    by=["App"], ascending=False).reset_index()



#select relevant columns

df_by_main_genre= df_by_main_genre[["Main_genre", "Content Rating", "App"]]



#reshape data to plot stacked bars

df_pivoted = df_by_main_genre.pivot(columns="Content Rating", index="Main_genre", values="App")

colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

df_pivoted.loc[:,df["Content Rating"].unique()].plot.bar(stacked=True, color=colors, figsize=(20,10))

plt.ylabel("Count")

plt.title("Genre and Content Rating")
#remember that size column was string object type

print(df.Size.unique())

#sizes differs between KBs and MBs and we have one object column "varies with device"
def convert_to_MB(df_column):

    if df_column == "Varies with device":

        result = np.NaN

    elif "k" in df_column:

        result = float(df_column.split("k")[0])*0.001 

    elif "M" in df_column:

        result = float(df_column.split("M")[0]) 

    return result
# "varies with device" column will be replaced with NA values for now

# convert every size to MB and float number

#fill NA sizes wtih average size



df["Size_in_MB"]=df.Size.apply(convert_to_MB)

df["Size_in_MB"].fillna(value=df.Size_in_MB.mean(), inplace=True)



#plot boxplot bacause we are interested in the distribution of each app per main genre

fig, ax = plt.subplots(figsize=(20,16))

sns.boxenplot(x='Main_genre', y='Size_in_MB', data=df, palette="Set2")

for item in ax.get_xticklabels():

    item.set_rotation(90)

plt.title("Distribution of App Sizes per Genre", size=20)
