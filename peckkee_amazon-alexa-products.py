import pandas as pd
df = pd.read_csv('../input/amazon_alexa.tsv',sep='\t')
# Looking at the data 
df.head()
# Understanding the different type of variations of Amazon Alexa
df['variation'].unique()

# Knowing the range of dates
last_date, first_date = df['date'].unique()[0] , df['date'].unique()[-1] 
print(f"The reviews of these Amazon products with Alexa were from: {first_date} to {last_date}")
# Verifying that there is no null values in all the columns.
df.columns.isnull()
req_cols = ['variation','date','verified_reviews','rating']
new_df = df[req_cols]
#Total 6 different types of Amazon Products, all using Alexa.

#Amazon Echo 1st Generation
echo_1G = ['Black', 'White']
cond1 = new_df['variation'].isin(echo_1G)
df1 = new_df[cond1]
new_df1 = df1.groupby('rating').size().reset_index(name='count')

# #Amazon Echo 2nd Generation
echo_2G = ['Charcoal Fabric ', 'Walnut Finish ', 'Heather Gray Fabric ','Sandstone Fabric ', 'Oak Finish ']
cond2 = new_df['variation'].isin(echo_2G)
df2 = new_df[cond2]
new_df2 = df2.groupby('rating').size().reset_index(name='count')

# #Amazon Echo Dot 3rd Generation
echo_dot = ['Black  Dot','White  Dot']
cond3 = new_df['variation'].isin(echo_dot)
df3 = new_df[cond3]
new_df3 = df3.groupby('rating').size().reset_index(name='count')

# #Amazon Echo Spot 
echo_spot = ['Black  Spot', 'White  Spot']
cond4 = new_df['variation'].isin(echo_spot)
df4 = new_df[cond4]
new_df4 = df4.groupby('rating').size().reset_index(name='count')

# #Amazon Fire TV Stick
fire_tv = ['Configuration: Fire TV Stick']
cond5 = new_df['variation'].isin(fire_tv)
df5 = new_df[cond5]
new_df5 = df5.groupby('rating').size().reset_index(name='count')

# #Amazon Echo Show
echo_show = ['Black  Show', 'White  Show']
cond6 = new_df['variation'].isin(echo_show)
df6 = new_df[cond6]
new_df6 = df6.groupby('rating').size().reset_index(name='count')

# Ratings given to 1G over the time frame 
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
new_df1.plot(kind='bar',x='rating',y='count',ax=ax)
fig.suptitle('Histogram of count of each rating given to Amazon Echo 1st Generation')
ax.set_xlabel('Ratings')
ax.set_ylabel('Total number for each rating')
ax.legend(['Count of each rating'])
# Ratings given to 2G over the time frame 
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
new_df2.plot(kind='bar',x='rating',y='count',ax=ax)
fig.suptitle('Histogram of count of each rating given to Amazon Echo 2nd Generation')
ax.set_xlabel('Ratings')
ax.set_ylabel('Total number for each rating')
ax.legend(['Count of each rating'])
# Ratings given to Dot over the time frame 
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
new_df3.plot(kind='bar',x='rating',y='count',ax=ax)
fig.suptitle('Histogram of count of each rating given to Amazon Echo Dot')
ax.set_xlabel('Ratings')
ax.set_ylabel('Total number for each rating')
ax.legend(['Count of each rating'])
# Ratings given to Spot over the time frame 
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
new_df4.plot(kind='bar',x='rating',y='count',ax=ax)
fig.suptitle('Histogram of count of each rating given to Amazon Echo Spot')
ax.set_xlabel('Ratings')
ax.set_ylabel('Total number for each rating')
ax.legend(['Count of each rating'])
# Ratings given to Fire TV Stick  over the time frame 
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
new_df5.plot(kind='bar',x='rating',y='count',ax=ax)
fig.suptitle('Histogram of count of each rating given to Amazon Fire TV Stick')
ax.set_xlabel('Ratings')
ax.set_ylabel('Total number for each rating')
ax.legend(['Count of each rating'])
# Ratings given to Show over the time frame 
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
new_df6.plot(kind='bar',x='rating',y='count',ax=ax)
fig.suptitle('Histogram of count of each rating given to Amazon Echo Show')
ax.set_xlabel('Ratings')
ax.set_ylabel('Total number for each rating')
ax.legend(['Count of each rating'])
# Performing Sentiment Analysis
#Clean

import re

def clean(x):
    result = re.sub(r"[()?!.#&’'😄👍😏😥❤⭐❤️😳😎😑;😊⏰🎶😍💋,]", '', x)
    result = result.strip()
    result = result.lower()
    
    return result

cleaned = df['verified_reviews'].apply(clean)
#Tokenise
ls = cleaned.tolist()
tokenised = [x.split() for x in ls]
#Stemming
import nltk
from nltk.stem import PorterStemmer

Stemmer = PorterStemmer()

stem = []

for comment in tokenised:
    stemmed = [Stemmer.stem(token) for token in comment] 
    stem.append(stemmed)

#Remove stopwords
import nltk
from nltk.corpus import stopwords
STOP_WORDS = stopwords.words('english')

removed = []

for comment in stem:
    remove = [word for word in comment if word not in STOP_WORDS]
    joined = ' '.join(remove)
    removed.append(joined)
    

df['Cleaned'] = removed
df.head()
#Training

#1.Required columns from Dataframe

Sentences = df['Cleaned']
Ratings = df['rating']


###Train-test split

from sklearn.model_selection import train_test_split
Sen_train,Sen_test,Rate_train,Rate_test = train_test_split(Sentences, Ratings, test_size=0.25)


#2. Creating a vectoriser

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

#3. Utilize the vectorizer to train your vocab and bag of words

bag_of_words = vectorizer.fit_transform(Sen_train)


#4. Train the classifier with bag_of_words with the ratings

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(bag_of_words,Rate_train)

#Testing with the 0.25%

test_bag_of_words = vectorizer.transform(Sen_test)
Rate_predict = classifier.predict(test_bag_of_words).tolist()
Rate_tested = Rate_test.tolist()
#Assuming Rating >= 3 is Good and vice versa
#Good == Positive

True_positive = 0
True_negative = 0
False_positive = 0
False_negative = 0

for number in range(0,len(Rate_predict)):
    if (Rate_predict[number] == Rate_tested[number]) & (Rate_predict[number] >= 3):
        True_positive += 1
    elif (Rate_predict[number] == Rate_tested[number]) & (Rate_predict[number] <= 2):
        True_negative += 1    
    elif (Rate_predict[number] != Rate_tested[number]) & (Rate_predict[number] >= 3) & (Rate_tested[number] <= 2):
        False_positive += 1
    elif (Rate_predict[number] != Rate_tested[number]) & (Rate_predict[number] <= 2) & (Rate_tested[number] >= 2):
        False_negative += 1
        
#So exactly how accurate is my model?
accuracy = format((True_positive + True_negative ) / (True_positive+True_negative+False_negative+False_positive),'.2f')
print("Accuracy of my model is: {}".format(accuracy))