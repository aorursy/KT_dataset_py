# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# storing and analysis

import numpy as np

import pandas as pd

import re



# visualization

import matplotlib.pyplot as plt

import warnings

import nltk

import string

import seaborn as sns



#import text classification modules

import os

from nltk.tokenize import WordPunctTokenizer

from bs4 import BeautifulSoup

from nltk.corpus import stopwords



from nltk.stem.porter import * 

from wordcloud import WordCloud

import spacy

from spacy import displacy

from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.feature_extraction.text import TfidfVectorizer



# import train/test split module

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split



# import scoring metrice

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# suppress cell warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#Load the training set and load the test set from kaggle

train_df = pd.read_csv('../input/climate-change-belief-analysis/train.csv') #Load train dataset

test_df = pd.read_csv('../input/climate-change-belief-analysis/test.csv') #Load test dataset
#display first 5 entries of the train data

train_df.head()
#Display the first 5 entries of the test data

test_df.head()
#Print out the Shape of the training data and the testing data

print('Shape of Train Dataset:',train_df.shape)

print('Shape of Test Dataset:',test_df.shape)
#Use the value_counts() method to displace the count of each sentiment in the training dataset

train_df['sentiment'].value_counts()
#Use the isnull() method to check for null values in training data

#.sum() method evaluates the total of each column of null values

train_df.isnull().sum()
#Combining both train and test data set before data cleaning as tweets in both the data set is unstructured

data = train_df.append(test_df, ignore_index=True) #Combine the two datasets and assing the variable 'data'
#User Defined function to clean unwanted text patterns from all tweets

# input - text to clean,pattern to replace

def remove_pattern(input_txt, pattern):

    """Removes unwanted text patterns from the tweets.

    

    Parameters

    ----------

    input_txt: string

                Original tweet string from the dataset

    

    pattern: regular expression

                pattern of text

    

    Returns

    -------

    input_txt: string

        returns the same input string without the given pattern as a result of the regular expresion on the input_text

    """

    r = re.findall(pattern, input_txt) #create an instance of the regular expression and assign the variable r

    for i in r:#Loop through the instance

        input_txt = re.sub(i, '', input_txt) #Remove the twitter handle from the dataset

        

    return input_txt
# remove twitter handles from the combined dataset (@user)

data['tidy_message'] = np.vectorize(remove_pattern)(data['message'], "@[\w]*")

#Removing RT from tweets and converting to lowercase

data['tidy_message'] = data['tidy_message'].str.replace('RT :',' ') #removing rt's

data['tidy_message'] = data['tidy_message'].apply(lambda x: x.lower())#Coverting all the tweets to lowercase since they are easier to work with
data.head()
#Initialise a Short for dictionary and assign it the variable 'short_word_dict'

short_word_dict = {

"121": "one to one",

"a/s/l": "age, sex, location",

"adn": "any day now",

"afaik": "as far as I know",

"afk": "away from keyboard",

"aight": "alright",

"alol": "actually laughing out loud",

"b4": "before",

"b4n": "bye for now",

"bak": "back at the keyboard",

"bf": "boyfriend",

"bff": "best friends forever",

"bfn": "bye for now",

"bg": "big grin",

"bta": "but then again",

"btw": "by the way",

"cid": "crying in disgrace",

"cnp": "continued in my next post",

"cp": "chat post",

"cu": "see you",

"cul": "see you later",

"cul8r": "see you later",

"cya": "bye",

"cyo": "see you online",

"dbau": "doing business as usual",

"fud": "fear, uncertainty, and doubt",

"fwiw": "for what it's worth",

"fyi": "for your information",

"g": "grin",

"g2g": "got to go",

"ga": "go ahead",

"gal": "get a life",

"gf": "girlfriend",

"gfn": "gone for now",

"gmbo": "giggling my butt off",

"gmta": "great minds think alike",

"h8": "hate",

"hagn": "have a good night",

"hdop": "help delete online predators",

"hhis": "hanging head in shame",

"iac": "in any case",

"ianal": "I am not a lawyer",

"ic": "I see",

"idk": "I don't know",

"imao": "in my arrogant opinion",

"imnsho": "in my not so humble opinion",

"imo": "in my opinion",

"iow": "in other words",

"ipn": "I’m posting naked",

"irl": "in real life",

"jk": "just kidding",

"l8r": "later",

"ld": "later, dude",

"ldr": "long distance relationship",

"llta": "lots and lots of thunderous applause",

"lmao": "laugh my ass off",

"lmirl": "let's meet in real life",

"lol": "laugh out loud",

"ltr": "longterm relationship",

"lulab": "love you like a brother",

"lulas": "love you like a sister",

"luv": "love",

"m/f": "male or female",

"m8": "mate",

"milf": "mother I would like to fuck",

"oll": "online love",

"omg": "oh my god",

"otoh": "on the other hand",

"pir": "parent in room",

"ppl": "people",

"r": "are",

"rofl": "roll on the floor laughing",

"rpg": "role playing games",

"ru": "are you",

"shid": "slaps head in disgust",

"somy": "sick of me yet",

"sot": "short of time",

"thanx": "thanks",

"thx": "thanks",

"ttyl": "talk to you later",

"u": "you",

"ur": "you are",

"uw": "you’re welcome",

"wb": "welcome back",

"wfm": "works for me",

"wibni": "wouldn't it be nice if",

"wtf": "what the fuck",

"wtg": "way to go",

"wtgp": "want to go private",

"ym": "young man",

"gr8": "great",

"8yo":"eight year old"

}
#Function used to lookup shortwords from the dictionary

def lookup_dict(text, dictionary):

    """Performs a lookup of the short word and returns the full meaning.

    

    Parameters

    ----------

    text: string

        Original tweet string from the dataset

    dictionary: dictionary

        dictionary with the short words as keys and the full meaning as the values

    

    Returns

    -------

    text: string

        Returns a new tweet string with the full meaning of the word as a result of replacement of the short word from the dictionary

    """

    for word in text.split(): #split the text into a list and loop through the list for find the words

        if word.lower() in dictionary: #Lower the words and see if they are in the dictionary

            if word.lower() in text.split(): #lower the words and see if they are in the text list

                text = text.replace(word, dictionary[word.lower()]) #replace the word in the text split with the values in the dictionary

    return text

#Perform lookup of short words and return the string with the full meaning

data['tidy_message'] = data['tidy_message'].apply(lambda x: lookup_dict(x,short_word_dict))# Use the short word 
#Initiatise a dictionary of apostrophe words and assign the variable 'contractions'

contractions = {

"ain't": "am not / are not",

"aren't": "are not / am not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he had / he would",

"he'd've": "he would have",

"he'll": "he shall / he will",

"he'll've": "he shall have / he will have",

"he's": "he has / he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how has / how is",

"i'd": "I had / I would",

"i'd've": "I would have",

"i'll": "I shall / I will",

"i'll've": "I shall have / I will have",

"i'm": "I am",

"i've": "I have",

"isn't": "is not",

"it'd": "it would",

"it'd've": "it would have",

"it'll": "it shall / it will",

"it'll've": "it shall have / it will have",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she had / she would",

"she'd've": "she would have",

"she'll": "she shall / she will",

"she'll've": "she shall have / she will have",

"she's": "she has / she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so as / so is",

"that'd": "that would / that had",

"that'd've": "that would have",

"that's": "that has / that is",

"there'd": "there had / there would",

"there'd've": "there would have",

"there's": "there has / there is",

"they'd": "they had / they would",

"they'd've": "they would have",

"they'll": "they shall / they will",

"they'll've": "they shall have / they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we had / we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what shall / what will",

"what'll've": "what shall have / what will have",

"what're": "what are",

"what's": "what has / what is",

"what've": "what have",

"when's": "when has / when is",

"when've": "when have",

"where'd": "where did",

"where's": "where has / where is",

"where've": "where have",

"who'll": "who shall / who will",

"who'll've": "who shall have / who will have",

"who's": "who has / who is",

"who've": "who have",

"why's": "why has / why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you had / you would",

"you'd've": "you would have",

"you'll": "you shall / you will",

"you'll've": "you shall have / you will have",

"you're": "you are",

"you've": "you have"

}
#Perform lookup of apostrophe words and return the string with the full meaning

data['tidy_message'] = data['tidy_message'].apply(lambda x: lookup_dict(x,contractions))# Use the contraction dictionary to replace the aprostophe words in data with the full meaning

data.head()
# Lookup which emojis exist in the tweets and save them as a list

tweets_text = data.tidy_message.str.cat() #Concatenate the tweet string and assign the variable 'tweets_text'

emos = set(re.findall(r" ([xX:;][-']?.) ",tweets_text)) #Create a regular expression that finds the emojis in the tweets

emos_count = [] #Initialise an empty list 'emo_count'

for emo in emos: #Loop through emoji

    emos_count.append((tweets_text.count(emo), emo)) #Add emojis found into the emo_count list

sorted(emos_count,reverse=True) #Sort the list in reverse order
#Initialise a emoji list and assign it to emoticon_dict

emoticon_dict={

'XD':'happy',

';)':'happy',

':-)':'happy',

';-)':'happy',

':P':'happy', 

':)':'happy',

'x ':'happy',

':(':'sad',

':/':'sad'

}
#Lookup the emojis in the tweets and replace them with the meaning behind the tweet

data['tidy_message'] = data['tidy_message'].apply(lambda x: lookup_dict(x,emoticon_dict))# Use emoticon_dict to replace decode emojis in the dataset
#Function to remove the remaining emojis in the tweets

def remove_emoji(message):

    """Performs a lookup of the short word and returns the full meaning.

    

    Parameters

    ----------

    message: string

        Original tweet string from the dataset



    Returns

    -------

    emoji_pattern.sub(r'', message): string

        Returns a new tweet which has removed unnecesary emojis as a result of the regular expression 

    """

    emoji_pattern = re.compile("["                   # Create a regular expression

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', message)
#Perform lookup of emojis on the dataset and replace the tweet

data['tidy_message'] = data['tidy_message'].apply(lambda x: remove_emoji(x))#Use the remove_emoji function to replace tweets in the dataset
# remove special characters, numbers, punctuations

data['tidy_message'] = data['tidy_message'].str.replace("[^a-zA-Z#]", " ")

data.head(10)
#remove short words of less than 3 letters in length

data['tidy_message'] = data['tidy_message'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))#Remove words that are shorter than 3 characters
data.head()
#Use tokenization to the words into a list of tokens 

tokenized_tweet = data['tidy_message'].apply(lambda x: x.split()) #Tokenize the dataset

tokenized_tweet.head()
#Use PorterStemmer() to strip suffixes from the words

from nltk.stem.porter import * #Import * from nltk.stem.porter

stemmer = PorterStemmer() #Initialize an instance of PorterStemmer and assign it a variable 'stemmer'



tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) #Stem the dataset

tokenized_tweet.head()
#We will now bring the tokens back together

for i in range(len(tokenized_tweet)): #Lopp through token list of stems

    tokenized_tweet[i] = ' '.join(tokenized_tweet[i]) #Join the list into a string of characters



data['tidy_message'] = tokenized_tweet #Assign the string of words to the dataset in the tid_message column
#Split the dataset back to the training set and the testing set

train = data[:len(train_df)] #Split to the lenght of original training set

test = data[len(train_df):]  #Split to the length of original testing set
#Use the .shape to see the length of the train dataset with the amound of features

train.shape
#Use the value_counts to see the values of the individual classes

train['sentiment'].value_counts() 
#Create a barplot for the train dataset classes

News = train['sentiment'].value_counts()[2] #Take values of class 2 and assign the variable 'News'

Pro= train['sentiment'].value_counts()[1]   #Take values of class 1 and assign the variabe 'Pro'

Neutral=train['sentiment'].value_counts()[0]#Take the values of class 0 and assign the variable 'Neutral'

Anti=train['sentiment'].value_counts()[-1]  #Take the values of class -1 and assing the variable 'Anti'



sns.barplot(['News ','Pro','Neutral','Anti'],[News,Pro,Neutral,Anti]) #Use seaborn barplot and add a list of classes

plt.xlabel('Tweet Classification') #X-label of the data

plt.ylabel('Count of Tweets')      #Y_label of the data

plt.title('Dataset labels distribution') #Give the data a title 'Dataset lables distribution'

plt.show() #Display the dataset



print('No of Tweets labelled as News:',News) #Print out all the classes values

print('No of Tweets labelled as Pro:',Pro)

print('No of Tweets labelled as Neutral:',Neutral)

print('No of Tweets labelled as Anti:',Anti)



print('Data is unbalanced and may need to be resampled in order to balance it with only',round(((News/(News+Pro+Neutral+Anti))*100),2),'% news tweets,',

      round(((Pro/(Pro+News+Neutral+Anti))*100),2),'% (pro)positive tweets,',round(((Neutral/(Neutral+Pro+News+Anti))*100),2),'% neutral tweets and',

      round(((Anti/(Anti+Neutral+News+Pro))*100),2),'% (anti)negative tweets') #Print out the percentages that each class represents in the dataset


#Check the Distribution of Length of Tweets in train and Test Dataset

tweetLengthTrain = train['message'].str.len() #Compute the length of the elements in the training set

tweetLengthTest = test['message'].str.len()   #Compute the length of the elements in the test set

flatui = ["#15ff00", "#ff0033"]#Color scheme

sns.set_palette(flatui)

plt.hist(tweetLengthTrain,bins=20,label='Train_Tweet') #Plot the histogram of training set

plt.hist(tweetLengthTest,bins=20,label='Test_Tweet')   #Plot the histogram of the test set

plt.legend()

plt.show()
#create a new length value column that contains the lengths of the messages

train['message_length'] = train['message'].apply(len)
#Create a violinplot of the dataset

plt.figure(figsize=(8,5)) #Set the figsize to 8 and 5 respectively

plt.title('Sentiments vs. Length of tweets') #Add the title of the violin plot

sns.violinplot(x='sentiment', y='message_length', data=train,scale='count') #Add the dimentions of the violin plot

plt.ylabel("Length of the tweets") #Y_lable of the plot

plt.xlabel("Sentiment Class") #X_label of the plot
#Use seaborn to create a boxplot using the message length and the sentiment

sns.boxplot(x='sentiment',y='message_length',data=train,palette='rainbow')
#Use groupby in order to numerically display what the boxplot is trying to show to the user

train['message_length'].groupby(train['sentiment']).describe()
#Create Multiple histograms for all classes

plt.figure(figsize=(10,6)) #Add figure dimentions

nbins = np.arange(0,210) #Add the necesarry amount of bins

plt.hist(train[train['sentiment']==2.0]['message_length'],bins=nbins, color="blue") #Histogram for news class

plt.hist(train[train['sentiment']==1.0]['message_length'],bins=nbins, color="red") #Histogram for positive class

plt.hist(train[train['sentiment']==0.0]['message_length'],bins=nbins, color="yellow") #Histogram for neutraL class

plt.hist(train[train['sentiment']==-1.0]['message_length'],bins=nbins) #Hisogram for negative class

plt.legend(('2-News','1-Pro','0-Neutral','1-Anti')) #Create a legend to displace each class color

#plt.ylabel("Count of messages")

plt.xlabel("Message Length")

plt.show()
#Create strings for each class

positive_words =' '.join([text for text in data['tidy_message'][data['sentiment'] == 1]]) #Words in the positve class

negative_words = ' '.join([text for text in data['tidy_message'][data['sentiment'] == -1]]) #Words in negative class

normal_words =' '.join([text for text in data['tidy_message'][data['sentiment'] == 0]]) #Words in the neutral class

news_words =' '.join([text for text in data['tidy_message'][data['sentiment'] == 2]]) #Words in the news class
#Create a user defined function to display a word cloud for each class

def word_cloud(class_words):

    """Generates a word cloud visualization of words in the different class.

    

    Parameters

    ----------

    class_words: string

        Words in each class



    Returns

    -------

    plt.show(): wordcloud visualisation

        Returns a visualisation of all the words that come from the class_words string

    """

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(class_words)

    plt.figure(figsize=(10, 7))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.title("Most Common positive words")

    plt.axis('off')

    return plt.show()

#Visualise all words from the positive class

word_cloud(positive_words)
#Visualise all words from the negative class class

word_cloud(negative_words)
#Visualise all words from the neutral class

word_cloud(normal_words)
#Visualise all words from the news class

word_cloud(news_words)
#Create a function to collect hashtags

def hashtag_extract(x):

    """Generate a list of all the hastags in different tweets.

    

    Parameters

    ----------

    x: string

        Original tweet from the training dataset



    Returns

    -------

    hashtags : list

        Returns a list of all hashtags in x

    """

    hashtags = [] #Initialize an empty list

    for i in x:   #Loop over the words in the tweet

        ht = re.findall(r"#(\w+)", i) #Create a regular expression to get the hashtags in a tweet

        hashtags.append(ht) #Add all those hashtags to the empty hashtag list



    return hashtags
# extracting hashtags from the news

HT_news = hashtag_extract(data['tidy_message'][data['sentiment'] == 2])

# extracting hashtags from positive sentiments

HT_positive = hashtag_extract(data['tidy_message'][data['sentiment'] == 1])

# extract hashtags from neutral sentiments

HT_normal = hashtag_extract(data['tidy_message'][data['sentiment'] == 0])

# extracting hashtags from negative sentiments

HT_negative = hashtag_extract(data['tidy_message'][data['sentiment'] == -1])



# unnesting list of all sentiments

HT_news = sum(HT_news,[])

HT_positive = sum(HT_positive,[])

HT_normal = sum(HT_normal,[])

HT_negative = sum(HT_negative,[])
#Create a function that visualises the barplot distribution of the hashtags

def bar_dist(x):

    """Generate a barplot of the top appearing hashtags in a class.

    

    Parameters

    ----------

    x: list

        List of all hashtag values in a class without the #



    Returns

    -------

    plt.show() : matplotlib barplot

        Returns a barplot of x

    """

    a = nltk.FreqDist(x) #Create a count of the hashtags in the list

    d = pd.DataFrame({'Hashtag': list(a.keys()), #Create a dataframe with the values of the counts of a

                  'Count': list(a.values())})  

    d = d.nlargest(columns="Count", n = 10) # selecting top 10 most frequent hashtags  

    plt.figure(figsize=(16,5))

    ax = sns.barplot(data=d, x= "Hashtag", y = "Count") #Initisalise seaborn barplot

    ax.set(ylabel = 'Count') #Set lables

    return plt.show()
#Display barplot of the News hastags

bar_dist(HT_news)
#Display barplot of the positive hastags

bar_dist(HT_positive)
#Display barplot of the neutral hastags

bar_dist(HT_normal)
#Display barplot of the neutral hastags

bar_dist(HT_negative)
#Splitting features and target variables

X = train['tidy_message'] #X is the features of the cleaned tweets

y = train['sentiment']    #Y is the target variable which is the train sentiment



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) #Splitting train set into training and testing data

#Print out the shape of the training set and the testing set

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# import and call the TFidfVectorizer 

from sklearn.feature_extraction.text import TfidfVectorizer #Import TFidfVectorizer from sklearn

tfidf = TfidfVectorizer() #Call the TFidfVectorizer and assign it to the tfidf variable

#import CountVectorizer and call it

from sklearn.feature_extraction.text import CountVectorizer #Import CountVectorizer from sklearn



cf= CountVectorizer() #Call the CountVectorizer and assing it to the variable 'cf'
#Import metrics from sklearn

from sklearn import metrics
# create a pipeline and fit it with a Logistic Regression

from sklearn.linear_model import LogisticRegression #Import Logistic Regression from sklearn



model = LogisticRegression(multi_class='ovr') #Call the Logistic Regression model and assign it to the variable 'model'



clf = Pipeline([('tfidf', tfidf), ('clf', model)]) #Create a pipeline with the TF-IDF Vectorizer with the logistic model





clf.fit(X_train, y_train) #Fit the training data to the pipeline



y_pred= clf.predict(X_test) #Make predictions and assign the predictions to y_pred



print('accuracy %s' % accuracy_score(y_pred, y_test)) #Print the accuracy

print('f1_score %s' % metrics.f1_score(y_test,y_pred,average='weighted')) #Print the weighted f1 score

print(classification_report(y_test, y_pred)) #Classification
## create a pipeline and fit it with a Linear Support Vector Classifier

from sklearn.svm import LinearSVC #Import LinearSVC from sklearn 



classifier = LinearSVC() #Call LinearSVC and assign the variable 'classifier'



clf = Pipeline([('tfidf', tfidf), ('clf', classifier)]) #Create a pipeline with the tdidf



clf.fit(X_train, y_train) #Fit the model

y_pred = clf.predict(X_test) #Make predictions and assign the variable 'y_pred'



print('accuracy %s' % accuracy_score(y_pred, y_test)) #Print the accuracy

print('f1_score %s' % metrics.f1_score(y_test,y_pred,average='weighted')) #Print the f1-score

print(classification_report(y_test, y_pred)) #Print the classification report
## create a pipeline and fit it with a  Support Vector Classifier

from sklearn.svm import SVC #Import SVC from sklearn 



classifier = SVC(kernel='rbf') #Call the SVC with the kernel='rbf' parameter



clf = Pipeline([('tfidf', tfidf), ('clf', classifier)]) #Add the SVC model to the pipeline



clf.fit(X_train, y_train) #Fit the training data

y_pred = clf.predict(X_test) #Make predictions to the test set and assign the variable 'y_pred'



print('accuracy %s' % accuracy_score(y_pred, y_test)) #Print the accuracy

print('f1_score %s' % metrics.f1_score(y_test,y_pred,average='weighted')) #Print the f1 score

print(classification_report(y_test, y_pred)) #Print out the classification
#Create a pipeline and predict the test sentiment using logistic regression

from sklearn.linear_model import LogisticRegression #Import Logistic Regression from the sklearn

model = LogisticRegression(multi_class='ovr' ) #Call logistic regression and assign it to the 'model' variable



text_lr= Pipeline([('cf', cf),('clf',model)]) #Create a pipeline of the bag of words and the logistic regression



text_lr.fit(X_train, y_train) #Fit the training data to the pipeline



y_pred = text_lr.predict(X_test) #Make a prediction of the test set



print('accuracy %s' % accuracy_score(y_pred, y_test)) #Print the accuracy

print('f1_score %s' % metrics.f1_score(y_test,y_pred,average='weighted')) #Print the f1-score

print(classification_report(y_test, y_pred)) #Print out the classification report
#Create a pipeline and make predictions of the bag of words using linearSVC

from sklearn.svm import LinearSVC #Import LinearSVC from the sklearn





clf= Pipeline([('cf', cf),('clf',  LinearSVC())]) #Create a pipeline with the bag or words features and the linearSVC



clf.fit(X_train, y_train) #Fit the training data to the pipeline



y_pred = clf.predict(X_test) #Make predictions with the test data



print('accuracy %s' % accuracy_score(y_pred, y_test)) #Print out the accuracy

print('f1_score %s' % metrics.f1_score(y_test,y_pred,average='weighted')) #Print out the f1 score

print(classification_report(y_test, y_pred)) #Print out the classification report
## create a pipeline and fit it with a  Support Vector Classifier

from sklearn.svm import SVC #Import SVC from sklearn 



classifier = SVC(kernel='rbf') #Call the SVC with the kernel='rbf' parameter



clf = Pipeline([('cf', cf), ('clf', classifier)]) #Add the SVC model to the pipeline



clf.fit(X_train, y_train) #Fit the training data

y_pred = clf.predict(X_test) #Make predictions to the test set and assign the variable 'y_pred'



print('accuracy %s' % accuracy_score(y_pred, y_test)) #Print the accuracy

print('f1_score %s' % metrics.f1_score(y_test,y_pred,average='weighted')) #Print the f1 score

print(classification_report(y_test, y_pred)) #Print out the classification
#Create a barplot for the train dataset classes

News = train['sentiment'].value_counts()[2] #Take values of class 2 and assign the variable 'News'

Pro= train['sentiment'].value_counts()[1]   #Take values of class 1 and assign the variabe 'Pro'

Neutral=train['sentiment'].value_counts()[0]#Take the values of class 0 and assign the variable 'Neutral'

Anti=train['sentiment'].value_counts()[-1]  #Take the values of class -1 and assing the variable 'Anti'



sns.barplot(['News ','Pro','Neutral','Anti'],[News,Pro,Neutral,Anti]) #Use seaborn barplot and add a list of classes

plt.xlabel('Tweet Classification') #X-label of the data

plt.ylabel('Count of Tweets')      #Y_label of the data

plt.title('Dataset labels distribution') #Give the data a title 'Dataset lables distribution'

plt.show() #Display the dataset
#Import the resampling module

from sklearn.utils import resample
#Downsample and upsample train dataset

df_majority = train[train.sentiment==1] #Create a new dataframe of the majority pro class

df_minority = train[train.sentiment==0] #Create a new dataframefor the minority neutral class

df_minority1 = train[train.sentiment==2] #Create a dataframe for the news class

df_minority2 = train[train.sentiment==-1]#Create a dataframe for the anti class



# Downsample majority class

df_majority_downsampled = resample(df_majority, 

                                 replace=False,    # sample without replacement

                                 n_samples=5000,     # Using a benchmark of 3640

                                 random_state=123) # reproducible results

#Upsampling the least minority class

df_minority_up = resample(df_minority, 

                        replace=True,    # sample without replacement

                        n_samples=5000,     # to match the second majority class

                        random_state=123) # reproducible results



df_minority_up1 = resample(df_minority1, 

                        replace=True,    # sample without replacement

                        n_samples=5000,     # to match the second majority class

                        random_state=123) # reproducible results



df_minority_up2 = resample(df_minority2, 

                        replace=True,    # sample without replacement

                        n_samples=5000,     # to match the second majority class

                        random_state=123) # reproducible results



# Combine minority class with downsampled majority class

df_resampled = pd.concat([df_majority_downsampled,df_minority_up,df_minority_up1, df_minority_up2])

 

# Display new class counts

df_resampled.sentiment.value_counts()

X = df_resampled['message']

y = df_resampled['sentiment']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# create a pipeline and fit it with a Logistic Regression

from sklearn.linear_model import LogisticRegression #Import logistic regression model from sklearn



model = LogisticRegression(C=50,multi_class='ovr') #Call logistic regression model and assign variable 'model'



clf_sam = Pipeline([('tfidf', tfidf), ('clf', model)]) #Create a pipeline with the logistic model and tf-idf vectorizer





clf_sam.fit(X_train, y_train) #Fit the training set



y_pred= clf_sam.predict(X_test) #Fit the test set



print('accuracy %s' % accuracy_score(y_pred, y_test)) #Print accuracy

print('f1_score %s' % metrics.f1_score(y_test,y_pred,average='weighted')) #Print f1 score

print(classification_report(y_test, y_pred)) #Print classification report
# create a pipeline and fit it with a Logistic Regression

from sklearn.linear_model import LogisticRegression #Import logistic regression model from sklearn



model = LogisticRegression(C=100,multi_class='ovr') #Call logistic regression model and assign variable 'model'



clf_sam1 = Pipeline([('cf', cf), ('clf', model)]) #Create a pipeline with the logistic model and bag-of-words





clf_sam1.fit(X_train, y_train) #Fit the training set



y_pred= clf_sam1.predict(X_test) #Fit the test set



print('accuracy %s' % accuracy_score(y_pred, y_test)) #Print accuracy

print('f1_score %s' % metrics.f1_score(y_test,y_pred,average='weighted')) #Print f1 score

print(classification_report(y_test, y_pred)) #Print classification report
## create a pipeline and fit it with a Linear Support Vector Classifier

from sklearn.svm import LinearSVC #Import LinearSVC from sklearn 



classifier = LinearSVC() #Call LinearSVC and assign the variable 'classifier'



clf = Pipeline([('tfidf', tfidf), ('clf', classifier)]) #Create a pipeline with the tdidf



clf.fit(X_train, y_train) #Fit the model

y_pred = clf.predict(X_test) #Make predictions and assign the variable 'y_pred'



print('accuracy %s' % accuracy_score(y_pred, y_test)) #Print the accuracy

print('f1_score %s' % metrics.f1_score(y_test,y_pred,average='weighted')) #Print the f1-score

print(classification_report(y_test, y_pred)) #Print the classification report
#Create a pipeline and make predictions of the bag of words using linearSVC

from sklearn.svm import LinearSVC #Import LinearSVC from the sklearn





clf= Pipeline([('cf', cf),('clf',  LinearSVC())]) #Create a pipeline with the bag or words features and the linearSVC



clf.fit(X_train, y_train) #Fit the training data to the pipeline



y_pred = clf.predict(X_test) #Make predictions with the test data



print('accuracy %s' % accuracy_score(y_pred, y_test)) #Print out the accuracy

print('f1_score %s' % metrics.f1_score(y_test,y_pred,average='weighted')) #Print out the f1 score

print(classification_report(y_test, y_pred)) #Print out the classification report
## create a pipeline and fit it with a  Support Vector Classifier

from sklearn.svm import SVC #Import SVC from sklearn 



classifier = SVC(kernel='rbf') #Call the SVC with the kernel='rbf' parameter



clf_rbf = Pipeline([('tfidf', tfidf), ('clf', classifier)]) #Add the SVC model to the pipeline



clf_rbf.fit(X_train, y_train) #Fit the training data

y_pred = clf_rbf.predict(X_test) #Make predictions to the test set and assign the variable 'y_pred'



print('accuracy %s' % accuracy_score(y_pred, y_test)) #Print the accuracy

print('f1_score %s' % metrics.f1_score(y_test,y_pred,average='weighted')) #Print the f1 score

print(classification_report(y_test, y_pred)) #Print out the classification
## create a pipeline and fit it with a  Support Vector Classifier

from sklearn.svm import SVC #Import SVC from sklearn 



classifier = SVC(kernel='rbf') #Call the SVC with the kernel='rbf' parameter



clf_rbfc = Pipeline([('cf', cf), ('clf', classifier)]) #Add the SVC model to the pipeline



clf_rbfc.fit(X_train, y_train) #Fit the training data

y_pred = clf_rbfc.predict(X_test) #Make predictions to the test set and assign the variable 'y_pred'



print('accuracy %s' % accuracy_score(y_pred, y_test)) #Print the accuracy

print('f1_score %s' % metrics.f1_score(y_test,y_pred,average='weighted')) #Print the f1 score

print(classification_report(y_test, y_pred)) #Print out the classification
test_x = test['message'] #Take test messages and assign variable test_x
y_pred = clf_rbf.predict(test_x) #Make rbf predictions and assign to t_pred
test['sentiment'] = y_pred #Add preditions to test data by create a new column 'sentiment'
test['sentiment'] = test['sentiment'].astype(int) #Change the datatype of the submission
test[['tweetid', 'sentiment']].to_csv('model_final.csv', index=False)#Extract twitter ID as sentiments to submit to kaggle