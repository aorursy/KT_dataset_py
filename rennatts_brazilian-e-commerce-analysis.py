#Let´s import the main libraries

import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
# Importing datasets

customers = pd.read_csv("../input/brazilian-ecommerce/olist_customers_dataset.csv")

geo= pd.read_csv("../input/brazilian-ecommerce/olist_geolocation_dataset.csv")

order_items= pd.read_csv("../input/brazilian-ecommerce/olist_order_items_dataset.csv")

order_pay= pd.read_csv("../input/brazilian-ecommerce/olist_order_payments_dataset.csv")

order_reviews= pd.read_csv("../input/brazilian-ecommerce/olist_order_reviews_dataset.csv")

orders= pd.read_csv("../input/brazilian-ecommerce/olist_orders_dataset.csv")

products= pd.read_csv("../input/brazilian-ecommerce/olist_products_dataset.csv")

sellers= pd.read_csv("../input/brazilian-ecommerce/olist_sellers_dataset.csv")

prod_categ= pd.read_csv("../input/brazilian-ecommerce/product_category_name_translation.csv")
#Data pre-processing



#merge dataset that have "order_id" column

Order_tot= pd.merge(orders, order_items, on='order_id')



#evaluate the dataset

Order_tot.info()

#the columns we need for now have the same amout of line, so let´s not delete anything now







#separate date and time of oder purchase timestamp

variable_split= Order_tot['order_purchase_timestamp'].str.split(' ')



#create a new colum with only the date

Order_tot['order_purchase_date']=variable_split.str.get(0)





#separete the date into day, month, year

#separate date and time, for this i need the date without time specifications

variable_split= Order_tot['order_purchase_date'].str.split('-')





Order_tot['year']=variable_split.str.get(0)

Order_tot['month']=variable_split.str.get(1)

Order_tot['day']=variable_split.str.get(2)



#graph with the total sales- price- over time

#set time 

Order_tot["order_purchase_date"]= pd.to_datetime(Order_tot["order_purchase_date"])



#plot the graph

sales_by_time= Order_tot.groupby(["order_purchase_date"])["price"].sum().plot()



#graph with the freight over time, let´s see if the freight mean changed over time

Order_tot.groupby(["order_purchase_date"])["freight_value"].mean().plot()

#Now let´s see which month presented the highest total sales considering the price



Order_tot.groupby(["month"])["price"].sum().plot(kind='bar', figsize=(10,5))
#Let´s see if the freight values changes over time

Order_tot.groupby(["month"])["freight_value"].mean().plot(kind='bar', figsize=(10,5))
#SEPARATE DATE AND TIME FOR THE 4 COLUMNS IN LOG

variable_split= Order_tot["order_approved_at"].str.split(' ')



Order_tot['approved_date']=variable_split.str.get(0)







variable_split= Order_tot["order_delivered_carrier_date"].str.split(' ')



Order_tot['order_carrier_date']=variable_split.str.get(0)







variable_split= Order_tot["order_delivered_customer_date"].str.split(' ')



Order_tot['order_delivered_date']=variable_split.str.get(0)







variable_split= Order_tot["order_estimated_delivery_date"].str.split(' ')



Order_tot['est_delivery_date']=variable_split.str.get(0)
#Transformar all the new columns into datatime

Order_tot["approved_date"]= pd.to_datetime(Order_tot["approved_date"])



Order_tot["order_carrier_date"]= pd.to_datetime(Order_tot["order_carrier_date"])



Order_tot["order_delivered_date"]= pd.to_datetime(Order_tot["order_delivered_date"])



Order_tot["est_delivery_date"]= pd.to_datetime(Order_tot["est_delivery_date"])
#calculate the logistics gap between the time columns

Order_tot["gap1"]= Order_tot["order_carrier_date"] - Order_tot["approved_date"]





Order_tot["gap2"]= Order_tot["order_delivered_date"] - Order_tot["order_carrier_date"]





Order_tot["gap3"]= Order_tot["est_delivery_date"] - Order_tot["order_delivered_date"]



print(Order_tot["gap1"])

print(Order_tot["gap2"])

print(Order_tot["gap3"])
#Most orders take 1 day to be delivered to the carrier

Order_tot.gap1.value_counts().nlargest(20).plot(kind='bar',figsize=(12,5))
#Most of orders take 6 to 7 days to go from the carrier to be delivered to the customer

Order_tot.gap2.value_counts().nlargest(20).plot(kind='bar',figsize=(12,5))



#The gap from the delivery prediction to the actual date of delivery varies from 13 to 15 days.

Order_tot.gap3.value_counts().nlargest(20).plot(kind='bar',figsize=(12,5))
Order_tot.info()
#merge Order_tot with order_reviews

df= pd.merge(Order_tot, order_reviews, on="order_id")
df.info()
#Drop the missing values

df= df.dropna()
#Transform the gap1, gap2, gap3 datatype to int

df["gap1"]= df["gap1"].astype('timedelta64[D]').astype(int)

df["gap2"]= df["gap2"].astype('timedelta64[D]').astype(int)

df["gap3"]= df["gap3"].astype('timedelta64[D]').astype(int)
#review score according to gap1(time from order approved to delivered to carrier)

df.groupby(["review_score"]).gap1.mean().plot(kind='bar',figsize=(8,5))
#review score according to gap2(time from received by the carrier to delivered to final customer)

df.groupby(["review_score"]).gap2.mean().plot(kind='bar',figsize=(8,5))
#review score according to gap3(difference between the predicted deliver time and the actual delivered time)

df.groupby(["review_score"]).gap3.mean().plot(kind='bar',figsize=(8,5))
print(df.head())
#SCATTER REVIEW SCORE X GAP3

import matplotlib.pyplot as plt



x = df.iloc[:, -5].values

y = df.iloc[:, -7].values



plt.scatter(x,y, label='skitscat', color='k', s=25, marker="o")



plt.xlabel('Review scores')

plt.ylabel('Gap3')

plt.title('REVIEW SCORE X GAP3')

plt.legend()

plt.show()
#SCATTER REVIEW SCORE X GAP2

import matplotlib.pyplot as plt



x = df.iloc[:, -5].values

y = df.iloc[:, -8].values



plt.scatter(x,y, label='skitscat', color='k', s=25, marker="o")



plt.xlabel('Review scores')

plt.ylabel('Gap2')

plt.title('REVIEW SCORE X GAP2')

plt.legend()

plt.show()
tot1= pd.merge(orders, order_items, on='order_id')

tot2= pd.merge(tot1, sellers, on='seller_id' )

tot3= pd.merge(tot2, customers, on='customer_id')

tot4= pd.merge(tot3, order_reviews, on='order_id')

tot5= pd.merge(tot4, products, on='product_id')

tot6= pd.merge(tot5, sellers, on='seller_id')
#Count the number of origem states and destination states 

tot6.groupby(['seller_state_x', 'customer_state']).order_item_id.value_counts().nlargest(20).plot(kind='bar',figsize=(12,5))
#see the smallest review scores accoring to origen and destinatin states

tot6.groupby(['seller_state_x', 'customer_state']).review_score.mean().nsmallest(30).plot(kind='bar',figsize=(8,5))
#sellers with the best review score by state

tot6.groupby(['seller_state_x']).review_score.mean().nlargest(30).plot(kind='bar',figsize=(8,5))
#sellers with the worst review scores by state

tot6.groupby(['seller_state_x']).review_score.mean().nsmallest(30).plot(kind='bar',figsize=(8,5))
#product categories with the most sales units

tot6.product_category_name.value_counts().nlargest(20).plot(kind='bar',figsize=(8,5))

#product category with the best review_score

tot6.groupby(["product_category_name"])["review_score"].mean().nlargest(30).plot(kind='bar',figsize=(8,5))
#product category with the worst review_score

tot6.groupby(["product_category_name"])["review_score"].mean().nsmallest(30).plot(kind='bar',figsize=(8,5))

#unite gap values from Order_tot into tot6

tot6= pd.merge(df, tot6, on="order_id" )
#wich origen states and destination states have the biggest gap2(delivery time)?

tot6.groupby(["seller_state_x", "customer_state"])["gap2"].sum().nlargest(30).plot(kind='bar',figsize=(8,5))

#wich origen states and destination states have the smallest gap2(delivery time)?

tot6.groupby(["seller_state_x", "customer_state"])["gap2"].mean().nsmallest(30).plot(kind='bar',figsize=(8,5))

#sales per  seller_city

tot6.groupby(["seller_city_x"])["price_x"].sum().nlargest(30).plot(kind='bar',figsize=(8,5))

#sales per state

tot6.groupby(["seller_state_x"])["price_x"].sum().nlargest(30).plot(kind='bar',figsize=(8,5))

#HIGHEST FREIGHT VALUE STATE ORIGIN TO STATE DESTINATION

freigh_per_states= tot6.groupby(["seller_state_x", "customer_state"])["freight_value_x"].mean().nlargest(30).plot(kind='bar', figsize=(8,5))
nlp= order_reviews.loc[:, ["order_id", "review_score", "review_comment_message"]]

nlp.info()



nlp= nlp.dropna()



nlp= nlp.reset_index(drop=True)



nlp.info()
print(nlp.head())
# Cleaning the texts

import re

import nltk

nltk.download('stopwords')

#nltk.download('floresta')

from nltk.corpus import stopwords

from string import punctuation

#from nltk.corpus import floresta

from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0, 41753):

    review = re.sub('[^a-zA-Z]', ' ', nlp["review_comment_message"][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('portuguese'))]

    review = ' '.join(review)

    corpus.append(review)

from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize



#transform corpus into str data type

corpus2= str(corpus)



sentences = corpus2

palavras = word_tokenize(corpus2.lower())



from nltk.corpus import stopwords

from string import punctuation



stopwords = set(stopwords.words('portuguese') + list(punctuation))

palavras_sem_stopwords = [palavra for palavra in palavras if palavra not in stopwords]





#Let´s see the most occuring words in the review messages

from nltk.probability import FreqDist



frequencia = FreqDist(palavras_sem_stopwords)



from collections import Counter 



# Pass the split_it list to instance of Counter class. 

Counter = Counter(palavras_sem_stopwords) 





# most_common() produces k frequently encountered 

# input values and their respective counts. 

most_occur = Counter.most_common(30) 

  

print(most_occur) 
#word cloud 

# Start with loading all necessary libraries

import numpy as np

import pandas as pd

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt





import warnings

warnings.filterwarnings("ignore")





#join all the reviews lines in one single line

text= str(corpus)





# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()