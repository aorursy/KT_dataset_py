import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as plt

pd.set_option("display.max_rows",None)

pd.set_option("display.max_columns",None)
import os

print(os.listdir("../input"))
data = pd.read_csv('../input/womens-shoes-prices/7210_1.csv')
data.head(2)
data2 = pd.read_csv('../input/womens-shoes-prices/7210_1.csv')
data2.head(2)
data3 = pd.read_csv('../input/womens-shoes-prices/7210_1.csv')
data3.head(2)
shoes=data3
shoes.columns
shoes.shape
shoes.isnull().sum()
shoes.drop(["Unnamed: 48","Unnamed: 49","Unnamed: 50","Unnamed: 51","vin","websiteIDs","weight","prices.warranty","prices.returnPolicy","prices.flavor","flavors","count","prices.source","isbn","prices.count","asins"],axis=1,inplace=True)
shoes.shape
shoes.columns
shoes.isnull().sum()
shoes.head()
# finding the mean of price columns grouped by brands



shoes_avg=shoes.groupby(by="brand")["prices.amountMin","prices.amountMax"].mean()

shoes_avg.head()
shoes_avg.dtypes
# this will show all the rows of our dataset 

pd.set_option("display.max_rows",None)
shoes_avg.head()
shoes_avg["Average_Price"]=(shoes_avg["prices.amountMin"]+shoes_avg["prices.amountMax"])/2
shoes_avg.head()
# sorting the values so as to find the top ten brands with highest average price
shoes_avg.sort_values(by="Average_Price",ascending=False).head(10)
# this will help us to check the skewness of out price distribution and as we can see our dataset is right skewed



sns.distplot(shoes_avg["Average_Price"],kde=True)
sns.boxplot(shoes_avg["Average_Price"])
shoes_avg.describe()    # this will identify the mean of Average price column along with other details
# lets plot to show the brandwise no of shoes that have their average price above mean average price
sns.countplot(shoes_avg["Average_Price"]>95.62)    
shoes_avg.head()
shoes.head()
shoes.columns
shoes["prices.isSale"].value_counts()    
shoes["Average_Price"]=(shoes["prices.amountMin"]+shoes["prices.amountMax"])/2
shoes.head()
# lets found out the price gap between the lowest and highest prices grouped by brand
shoes_avg.head()
shoes_avg.columns
shoes_avg["Price Gap"]=shoes_avg["prices.amountMax"]-shoes_avg["prices.amountMin"]
# These are the top 10 brands that have the largest price gaps between their highest and lowest prices
shoes_avg.sort_values(by="Price Gap",ascending=False).head(10)
# comparing highest 10 entries of Price columns



shoes_avg.sort_values(by="Price Gap",ascending=False).head(10).plot(kind="bar",stacked=True)
shoes.head()
# applying Annova test to check if brand value has impact on Average Price on our data
import scipy.stats as stats

import statsmodels.api as sms

import statsmodels.formula.api as statsmodels

from statsmodels.formula.api import ols

shoes_ol=shoes[["brand","Average_Price"]]
shoes_ols=shoes_ol.sort_values(by="brand")

shoes_ols.head()
n = 3

dfn = n-1



dfd = shoes_ols["brand"].shape[0]-2

dfd
stats.f.ppf(1-0.05, dfn=dfn, dfd=dfd)
model_b = ols("Average_Price~brand",data=shoes).fit()
print(sms.stats.anova_lm(model_b))
# now we know our null hyp. was H0 : the brand of shoes has no impact on the average price

# likewise our alternte hyp. was H1 : the brand of shoes has impact on the average price 



# according to the pvalue we can say that that we reject our null hypothesis and hence brand has impact on the average price of our shoes
shoes["brand"].value_counts().shape
model_g = ols("Average_Price~colors",data=shoes).fit()
print(sms.stats.anova_lm(model_g))
shoes[["brand","dateAdded","Average_Price"]].head()
# now lets see how we can fill the null values of brand columns



# here's the idea , we can extract the brand name string from prices.sourceURLs but in order to do so we have to create a local data frame and then use split function
shoes["prices.sourceURLs"][12]
shoes[["brand","prices.sourceURLs"]].head()
shoes_strip=shoes[["brand","prices.sourceURLs"]]   # local dataframe
shoes_strip["Stripcol"]=shoes["prices.sourceURLs"].str.split("-").str[0]     # splitting the URL at "-" 
shoes_strip.head(15)
shoes_strip["Stripcol"]=shoes_strip["Stripcol"].str.split("/").str[-1]     # again splitting the URL ar "/" and -1 indicates we start from right
shoes_strip.head(50)
shoes_strip[shoes_strip["Stripcol"]=="SNEED"].count()
for i in range(0,len(shoes_strip["Stripcol"])):

    if shoes_strip["Stripcol"][i]=="SNEED" :

        shoes_strip["brand"][i]="SNEED"



        
shoes_strip["brand"].isnull().sum()
# grouping the brand=null dataframe on the basis of brand 

shoes_strip[shoes_strip["brand"].isnull()].groupby(by="Stripcol").first()
for i in range(0,len(shoes_strip["Stripcol"])):

    if shoes_strip["Stripcol"][i]=="Nine" :

        shoes_strip["brand"][i]="Nine West"
for i in range(0,len(shoes_strip["Stripcol"])):

    if shoes_strip["Stripcol"][i]=="Mirak" :

        shoes_strip["brand"][i]="Mirak Montana"
shoes_strip["brand"].isnull().sum()
shoes_strip["brand"].fillna(100,inplace=True)    # now in order to replace "NaN" null value with our brand name lets first fill the null value with a number so that it can be used in conditional loops
shoes_strip.head()
x=len(shoes_strip["brand"])     # replacing the 100 with our required brand name
for i in range(0,x):

    for j in shoes_strip["Stripcol"]:

        if shoes_strip["brand"][i]==100:

            shoes_strip["brand"][i]=j

        break

        

            
shoes_strip["brand"].isnull().sum()
shoes_strip.head()
shoes.head()
shoes.brand=shoes_strip.brand
shoes.brand.head()
shoes.brand.isnull().sum()
shoes.colors.isnull().sum()
model_g = ols("Average_Price~brand",data=shoes).fit()
print(sms.stats.anova_lm(model_g))     # without sorting
shoes_sort=shoes.sort_values(by="brand")
model_g=ols("Average_Price~brand",data=shoes_sort).fit()   # after sorting
print(sms.stats.anova_lm(model_g))    
# lets check the corelation between numeric columns

shoes_sort.corr()
# Word Cloud of brand names



import os

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import numpy as np



list=[]

for i in shoes["brand"] :

    list.append(i)

slist = str(list)



wordcloud = WordCloud(width=400, height=200).generate(slist)

plt.figure(figsize=(12,12))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()

shoes_scatter=shoes_avg.sort_values(by="Average_Price",ascending=False).head(5)
shoes_scatter
# Top 10 Brands with highest Average Price

list1=[]

for i in shoes_scatter.index :

    list1.append(i)

slist1 = str(list1)



wordcloud = WordCloud(width=480, height=480, max_font_size=40, min_font_size=10).generate(slist1)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()



# Much work is still remaining