import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
Kaggle=1

if Kaggle==0:

    df_alexa1 =pd.read_csv("amazon_alexa.tsv",sep="\t")

else:

    df_alexa1 = pd.read_csv("../input/amazon_alexa.tsv",sep="\t")
df_alexa1.head()
df_alexa1.tail()
df_alexa1.info() #Obtaining general information about the dataset we are using and checking there is no missing information with non-null
df_alexa1.describe() #displaying other relevant information from the dataset, the number of elemnts on it the values and percentages of the feedback, to know 1 as positive feedback and 0 as negative feedbak. Highlighting the mean on rating since that's the overall opinion of costumers about the product.
df_alexa1.keys() #Obtaining the headers of the specific information provided from the dataset
df_alexa1['verified_reviews'].tail() #Obtaining the reviews from real customers to later on analyse what are their most common opinions on the product
df_alexa = df_alexa1[~df_alexa1.variation.str.contains('Fire')]
df_alexa.describe()
df_alexa = df_alexa.reset_index(drop=True) #This way we reindex the whole dataframe from 0 to 2799 as for all 2800 elements on the dataset.
df_alexa['date'] = pd.to_datetime(df_alexa['date'], errors='coerce') #Enable access to datetime programming features
weekday_ratings = df_alexa['date'].dt.weekday_name.value_counts()

weekday_ratings = weekday_ratings.sort_index()

sns.barplot(weekday_ratings.index, weekday_ratings.values, order=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])

plt.xticks(rotation='vertical')

plt.ylabel('Count')



plt.show()
positive = df_alexa[df_alexa['feedback'] == 1] # From "df_alexa" visualise the positive reviews

positive
positive['feedback'].count() #providing the number of positive reviews
negative = df_alexa[df_alexa['feedback'] == 0]  # From "df_alexa" visualise the negative reviews

negative
negative['feedback'].count() #providing the number of negative reviews
sns.countplot(df_alexa['feedback'], label = 'count') #visualizing how many of those reviews were positive (1) and how many negative (0). In accordance with our expectations.
sns.countplot(df_alexa['rating'], label = 'count') #visualizing the satisfaction of the clients though the 5 stars rating
five_star = df_alexa[df_alexa['rating'] == 5]  

four_star = df_alexa[df_alexa['rating'] == 4]

three_star = df_alexa[df_alexa['rating'] == 3]

two_star = df_alexa[df_alexa['rating'] == 2]

one_star = df_alexa[df_alexa['rating'] == 1]



df_ratings = pd.DataFrame({'rating' : ['5', '4', '3', '2', '1'],

                           'count' : [five_star['rating'].count(), four_star['rating'].count(),

                                      three_star['rating'].count(), two_star['rating'].count(),

                                      one_star['rating'].count()]})

df_ratings
print('5 Star Rating = {0} %'.format((2004/2800)*100))

print('4 Star Rating = {0} %'.format((421/2800)*100))

print('3 Star Rating = {0} %'.format((146/2800)*100))

print('2 Star Rating = {0} %'.format((81/2800)*100))

print('1 Star Rating = {0} %'.format((148/2800)*100))
plt.figure(figsize = (30,10))



sns.barplot(x = 'variation', y = 'rating', data = df_alexa, palette = 'deep')
plt.figure(figsize = (30,10))



df_alexa.variation.value_counts().plot(kind='bar', fontsize=18)
df_alexa['verified_reviews'].iloc[:5]  #All reviews from real customers on the Amazon echo devices (excluding the already deleted Fire TV stick information) / Only the first 5 elements shown to gain space on desktop (displaying all of them would take longer for the program to upload and would add no useful information) also, improving the comparison with other parts of the code.  
for i in df_alexa['verified_reviews'].iloc[:5]: #A more elegant way to visualize the reviews  / Only the first 5 elements shown to gain space on desktop (displaying all of them would take longer for the program to upload and would add no useful information) also, improving the comparison with other parts of the code.  

    print(i, '\n')
words = df_alexa['verified_reviews'].tolist() #dataframe selected transformed into different strings

words[:5]  # Only the first 5 elements shown to gain space on desktop (displaying all of them would take longer for the program to upload and would add no useful information) also, improving the comparison with other parts of the code.  
words_as_one_string = ' '.join(words) #dataframe selected transformed into one single string

words_as_one_string[:500] # Only a certain number of character shown to gain space on desktop (displaying all of them would take longer for the program to upload and would add no useful information) also,  improving the comparison with other parts of the code.  
from wordcloud import WordCloud
plt.figure(figsize = (15,15))

plt.imshow(WordCloud().generate(words_as_one_string), interpolation = 'bilinear') #interpolation = 'bilinear' increases the quality of the image

plt.title("Averall words used on all reviews", fontsize=40)
df_bad_alexa = df_alexa[df_alexa['feedback'] == 0] #Creating a dataset with the bad reviews on it only.



bad_words = df_bad_alexa['verified_reviews'].tolist() #dataframe selected transformed into different strings

bad_words_as_one_string = ' '.join(bad_words) #dataframe selected transformed into one single string

plt.figure(figsize = (15,15))

plt.imshow(WordCloud().generate(bad_words_as_one_string), interpolation = 'bilinear')



plt.title("Averall words used on negative reviews", fontsize=40)
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()

alexa_CountVectorizer = vectorizer.fit_transform(df_alexa['verified_reviews'])
alexa_CountVectorizer.shape #The reviews are now structured in a structured array.
print(vectorizer.get_feature_names()) #every single word that has been mentioned on the reviews.
print(alexa_CountVectorizer.toarray())
word_count_array = alexa_CountVectorizer.toarray()
word_count_array[0,:] #Obtain the first review (first row and all columns)
plt.plot(word_count_array[0,:]) #All points are at 0 but three peaks, corresponding to the three words used on this 1st review 

df_alexa['verified_reviews'][0] #displaying the first review on screen to check the 3 words used by this user. 
plt.plot(word_count_array[3,:]) #Some points are now at 0 but some peaks appear now, corresponding to the total number of words used on this 4th review 

df_alexa['verified_reviews'][3] #displaying the 4th review on screen to check the words used by this user. 
plt.plot(word_count_array[13,:]) #All points are at 0 but 1 peaks, corresponding to the three words used on this 3rd review since it is the same word. 

df_alexa['verified_reviews'][13] #displaying the 3rd review on screen to check the 3 words used by this user, as  they are the same word, only one peak is shown. 
df_alexa['length'] = df_alexa['verified_reviews'].apply(len)

df_alexa.head() #This process gives a value to the total number of charachters used in the description.
df_alexa['length'].hist(bins = 100) 

plt.xlabel('Number of characters used')

plt.ylabel('Users')
min_char = df_alexa['length'].min()

df_alexa[df_alexa['length'] == min_char] ['verified_reviews'].iloc[0]
max_char = df_alexa['length'].max()

df_alexa[df_alexa['length'] == max_char] ['verified_reviews'].iloc[0]