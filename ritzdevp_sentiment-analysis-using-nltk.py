#Importing libraries

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

from sklearn.utils import shuffle

import nltk
data_file = "../input/Amazon_Unlocked_Mobile.csv"

#reading csv file

data = pd.read_csv( data_file)
data.head() 
product_name = []

for item in data["Product Name"]:

    if (item in product_name):

        continue

    else:

        product_name.append(item)

        
len(product_name) # 4410 phones
data["Brand Name"]

brands = []

for item in data["Brand Name"]:

    if (item in brands):

        continue

    else:

        brands.append(item)
len(brands) 
data_df = pd.DataFrame(data) #converting the data into a pandas dataframe.
data_df.head()
data_df = shuffle(data_df) #Shuffle Data 
data_df[:10]
#dropped rows having NaN values

data_df = data_df.dropna()
# General Description of data_df

data_df.describe() 
info = pd.pivot_table(data_df,index=['Brand Name'],values=['Rating', 'Review Votes'],

               columns=[],aggfunc=[np.sum, np.mean],fill_value=0)

info = info.sort_values(by=('sum', 'Rating'), ascending = False)



info.head(10)

import matplotlib.pyplot as plt

ylabel = data_df["Price"]

plt.ylabel("Price")

plt.xlabel("Rating")

xlabel = data_df["Rating"]

plt.scatter(xlabel, ylabel, alpha=0.1)

plt.show()
ylabel2 = data_df["Price"]

plt.ylabel("Price")

xlabel2 = data_df["Review Votes"]

plt.xlabel("Review Votes")

plt.scatter(xlabel2, ylabel2, alpha=0.1)

plt.show()
ylabel3 = data_df["Rating"]

plt.ylabel("Rating")

xlabel3 = data_df["Review Votes"]

plt.xlabel("Review Votes")

plt.scatter(xlabel3, ylabel3, alpha=0.1)

plt.show()
corr_matrix = data_df.corr()

corr_matrix["Rating"].sort_values(ascending = False)
corr_matrix = data_df.corr()

corr_matrix["Price"].sort_values(ascending = False)
all_reviews = data_df["Reviews"]

all_reviews.head()
#reset_index

data_df = data_df.reset_index(drop=True)
data_df.head()
all_reviews = data_df['Reviews']

all_sent_values = []

all_sentiments = []

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiment_value(paragraph):

    analyser = SentimentIntensityAnalyzer()

    result = analyser.polarity_scores(paragraph)

    score = result['compound']

    return round(score,1)
sample = data_df['Reviews'][1231]

print(sample)

print('Sentiment: ')

print(sentiment_value(sample))
sample1 = data_df['Reviews'][99314]

print(sample1)

print('Sentiment: ')

print(sentiment_value(sample1))
sample2 = data_df['Reviews'][9001]

print(sample2)

print('Sentiment: ')

print(sentiment_value(sample2))
for i in range(0,20000):

    all_sent_values.append(sentiment_value(all_reviews[i])) # 8 minutes for calculation 
len(all_sent_values)
#Sentiment Analysis on first 20,000 rows

temp_data = data_df[0:20000]
temp_data.shape
SENTIMENT_VALUE = []

SENTIMENT = []

for i in range(0,20000):

    sent = all_sent_values[i]

    if (sent<=1 and sent>=0.5):

        SENTIMENT.append('V.Positive')

        SENTIMENT_VALUE.append(5)

    elif (sent<0.5 and sent>0):

        SENTIMENT.append('Positive')

        SENTIMENT_VALUE.append(4)

    elif (sent==0):

        SENTIMENT.append('Neutral')

        SENTIMENT_VALUE.append(3)

    elif (sent<0 and sent>=-0.5):

        SENTIMENT.append('Negative')

        SENTIMENT_VALUE.append(2)

    else:

        SENTIMENT.append('V.Negative')

        SENTIMENT_VALUE.append(1)

        

        
#update to temp_data
temp_data['SENTIMENT_VALUE'] = SENTIMENT_VALUE

temp_data['SENTIMENT'] = SENTIMENT
temp_data.head()
#find accuracy

counter = 0

for i in range(0,20000):

    if (abs(temp_data['Rating'][i]-temp_data['SENTIMENT_VALUE'][i])>1):

        counter += 1

    
counter
accuracy = (temp_data.shape[0]-counter)/temp_data.shape[0]
percent_accuracy = accuracy*100
percent_accuracy
temp_data.head()
xaxis = []

for i in range(0,20000):

    xaxis.append(i)



ylabel_new_1 = all_sent_values[:20000]



xlabel = xaxis

plt.figure(figsize=(9,9))

plt.xlabel('ReviewIndex')

plt.ylabel('SentimentValue(-1 to 1)')

plt.plot(xlabel, ylabel_new_1, 'ro',  alpha=0.04)



plt.title('Scatter Intensity Plot of Sentiments')

plt.show()
product_name_20k = []

for item in temp_data["Product Name"]:

    if (item in product_name_20k):

        continue

    else:

        product_name_20k.append(item)
len(product_name_20k)
brands_temp = []

for item in temp_data["Brand Name"]:

    if (item in brands_temp):

        continue

    else:

        brands_temp.append(item)
len(brands_temp)
testing2 = pd.pivot_table(temp_data,index=['Brand Name'],values=['Rating', 'Review Votes','SENTIMENT_VALUE'],

               columns=[],aggfunc=[np.sum, np.mean],fill_value=0)

testing2 = testing2.sort_values(by=('sum', 'Rating'), ascending = False)

testing2.head(10)

testing3 = pd.pivot_table(temp_data,index=['Product Name'],values=['Rating', 'Review Votes','SENTIMENT_VALUE'],

               columns=[],aggfunc=[np.sum, np.mean],fill_value=0)

testing3 = testing3.sort_values(by=('sum', 'Rating'), ascending = False)

testing3.head(10)
import pylab



names = testing2.index[:10]

y = testing2['sum', 'SENTIMENT_VALUE'][:10]

y2 = testing2['sum', 'Rating'][:10]







pylab.figure(figsize=(15,7))

x = range(10)

pylab.subplot(2,1,1)

pylab.xticks(x, names)

pylab.ylabel('Summed Values')

pylab.title('Total Sum Values')

pylab.plot(x,y,"r-",x,y2,'b-')

pylab.legend(['SentimentValue', 'Rating'])



y_new = testing2['mean', 'SENTIMENT_VALUE'][:10]

y2_new = testing2['mean', 'Rating'][:10]







pylab.figure(figsize=(15,7))





pylab.subplot(2,1,2)

pylab.xticks(x, names)

pylab.ylabel('Mean Values')

pylab.title('Mean Values')

pylab.plot(x,y_new,"r-",x,y2_new,'b-')

pylab.legend(['SentimentValue', 'Rating'])





pylab.show()
samsung = []

blu = []

apple = []

lg = []

nokia = []







for i in range(0,20000):

    score = all_sent_values[i]

    brand = temp_data['Brand Name'][i]

    if (brand == 'Samsung'):

        samsung.append(score)

    elif (brand == 'BLU'):

        blu.append(score)

    elif (brand == 'Apple'):

        apple.append(score)

    elif (brand == 'LG'):

        lg.append(score)

    elif (brand == 'Nokia'):

        nokia.append(score)

    else:

        continue
list_of_brands = [samsung, blu, apple,lg,nokia]

name_of_brands = ['Samsung', 'BLU', 'Apple', 'LG', 'Nokia']
def plot_brand(brand, name):

    pylab.figure(figsize=(20,3))

    x = range(0,800)

    

    #pylab.xticks(x)

    pylab.ylabel('Sentiment')

    pylab.title(name)

    #pylab.plot(x,brand,"ro", alpha = 0.2)

    pylab.plot(x, brand[:800], color='#4A148C', linestyle='none', marker='o',ms=9, alpha = 0.4)

    

    pylab.show()
for i in range(0,len(list_of_brands)):

    plot_brand(list_of_brands[i],name_of_brands[i])