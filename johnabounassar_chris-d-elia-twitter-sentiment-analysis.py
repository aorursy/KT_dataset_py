#Importing necessary libraries

#Used for navigating files
import os

#Used for handling raw data
import numpy as np
import pandas as pd

#Used for visualizations
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#Used for scraping tweets about Chris D'Elia
#import GetOldTweets3 as got

#Used for handling dates/datetime objects
import time
from datetime import datetime, timedelta

#Used for assessing model performance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#Used for saving model
import pickle

#Used for preprocessing/modelling of text data
import spacy
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Import chris's follower count since 2016
followers_2016 = pd.read_csv('/kaggle/input/chris-delia-twitter-info/Chris_Followers_From_2016.csv')


#Plot the historical follower count
plt.figure(figsize=(22,10))
sns.lineplot(x='Date', y='Followers', data=followers_2016, markers=True, linewidth=3, marker='o', markersize=10)
plt.title("Chris D'Elia Follower Count From Oct. 2016", size = 30, fontweight='bold')
plt.xlabel('Date', size=20, fontweight='bold')
plt.ylabel('Followers', size=20, fontweight='bold')
plt.xticks(rotation=65, size=18, horizontalalignment='right')
plt.yticks(size=20)
plt.axvspan(xmin=44, xmax=45, color='red', alpha=0.5)
plt.show()
#Make a function to extract tweets using GOT3
def download_tweets(SinceDate, UntilDate, Query, num_tweets, wait_time):
    '''
    Downloads tweets within a certain time period and returns a dataframe of the tweets.
    Note that to avoid extracting too many tweets and getting an error, less than 10k tweets should be scraped every 10 minutes.
    
    Sincedate and Untildate should be strings. String of date, formatted: 'yyyy-mm-dd'. 
    Query = string
    num_tweets = integer value for number of tweets to be extracted per day
    wait_time = integer value for number of seconds to wait before extracting tweets for the next day
    '''
    #Convert the strings into datetime objects (for tracking time)
    start = datetime.strptime(SinceDate, '%Y-%m-%d')
    finish = datetime.strptime(UntilDate, '%Y-%m-%d')
      
    #Create storage place for tweets - dataframe
    df = pd.DataFrame(columns=['date', 'username','tweet', 'favorites', 'retweets', 'hashtags'])
    
    while start < finish:       
        #Setup criteria for tweet extraction - 7500 tweets per day
        tweets_crit = got.manager.TweetCriteria().setQuerySearch(Query).setSince(start.strftime('%Y-%m-%d'))\
                                                 .setUntil((start + timedelta(days=1)).strftime('%Y-%m-%d'))\
                                                 .setMaxTweets(num_tweets)
        
        #Query/download the tweets
        daily_tweets = got.manager.TweetManager().getTweets(tweets_crit)
        
        #Concatenate the tweets to the storage df
        tweets = [[tweet.date, tweet.username, tweet.text, tweet.favorites, tweet.retweets, tweet.hashtags] for \
                                                                                            tweet in daily_tweets]
        df_tweets = pd.DataFrame.from_records(tweets, columns=['date', 'username','tweet', 'favorites', 'retweets', 'hashtags'])
        df = pd.concat([df, df_tweets], axis = 0)
        
        #Puts program to sleep - prevents error of too many requests at one time
        time.sleep(wait_time)
        
        #Update sincedate
        start = start + timedelta(days=1)
    
    #Return the df of tweets
    return df

#downloaded_tweets = download_tweets("2020-06-12","2020-07-27", "Chris DElia", 7500, 600)
#downloaded_tweets.to_csv('summer_2020_tweets.csv')
#Read in the tweets, reset index and drop entries where there is no tweet
tweets = pd.read_csv('/kaggle/input/chris-delia-twitter-info/summer_2020_tweets.csv', index_col='Unnamed: 0')

#Create a function to clean tweets from the extraction function
def clean_tweets(dataframe):
    """
    Input is a dataframe from the tweet scraping function, it returns a cleaned version of the dataframe.
    """
    
    #Fill missing values for hashtags - replace null values with string 'None'
    dataframe['hashtags'].fillna('None', inplace = True)
    
    #Remove datapoints where there is a missing value for tweet - it will be useless for the sentiment anlaysis
    dataframe.dropna(inplace = True)
    
    #Convert datetime to string and extract the date in format (YYYY-MM-DD)
    dataframe['date'] = dataframe['date'].apply(lambda date: date.split(' ')[0])
    dataframe.reset_index(drop=True,inplace=True)
    
    return dataframe


#Create a function to clean tweets from the extraction function
def clean_tweets_new(dataframe):
    """
    Input is a dataframe from the tweet scraping function, it returns a cleaned version of the dataframe.
    """
    
    #Fill missing values for hashtags - replace null values with string 'None'
    dataframe['hashtags'].fillna('None', inplace = True)
    
    #Remove datapoints where there is a missing value for tweet - it will be useless for the sentiment anlaysis
    dataframe.dropna(inplace = True)
    
    #Convert datetime to string and extract the date in format (YYYY-MM-DD)
    dataframe['date'] = dataframe['date'].apply(lambda date: date.strftime('%Y-%m-%d').split(' ')[0])
    dataframe.reset_index(drop=True,inplace=True)
    
    return dataframe

#Use function on extracted tweets
#tweets = clean_tweets(tweets)

#Setup data for the spaCy model - TextCategorizer
tweets_array = tweets['tweet'].values

#show example of df
tweets.head(3)
#Reading in sentiment140 tweets
sent140_filepath = '/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv'
sent140_cols= ["target", "ids", "date", "flag", "user", "text"]
sent140_encoding = "ISO-8859-1"
sentiment140 = pd.read_csv(sent140_filepath, encoding=sent140_encoding, names=sent140_cols)

#Drop unneccessary data and convert labels into 'positive' or 'negative'
sentiment140.drop(['ids', 'date', 'flag', 'user'], axis=1, inplace=True)
sentiment140['target'].replace({0:'negative',4:'positive'}, inplace=True)
#Create blank spacy model that will be trained on sentiment140 data
nlp = spacy.blank('en')

#Create pipeline in spaCy (spaCy pipes) to create preprocessing/transforming object for tokens
text_cat = nlp.create_pipe("textcat", config={"exclusive_classes":True})

#Add the pipeline to the spacy model
nlp.add_pipe(text_cat)

#Add labels (target categories) to the classifier
text_cat.add_label('negative')
text_cat.add_label('positive')
#Define inputs and targets
inputs = sentiment140['text']
targets = sentiment140['target']

#split the training set into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.20, random_state=42)
print(X_train.shape, X_val.shape)

#Convert dataframes into numpy arrays - used for training/predictions
X_train = X_train.values
X_val = X_val.values
y_train = y_train.values
#Import library/objects used for making the function
import random
from spacy.util import minibatch

#Making a function to train the model. 
#Arguments will be: x_train, y_train, optimizer, number of epochs
def train(x_train, y_train, optimizer, num_epochs):
    """
    Function to train the spaCy model, takes four arguements defined in the following manner.
    
    x_train = numpy array of texts/documents
    y_train = numpy array of labels (positive/negative) for corresponding texts/documents
    optimizer = spaCy object used to initialize weights for the CNN.
    num_epochs = Integer defining number of epochs used to train the model.
    """
    #Define the random seed so the results are reproducable
    random.seed(42)
    spacy.util.fix_random_seed(42)
    
    #Setup training data for nlp.update(). Each label is a dictionary as described above.
    train_labels = [{'cats':{'negative':label=='negative', 'positive':label=='positive'}} for label in y_train]
    train_data = list(zip(x_train, train_labels))
    
    #Define a dictionary to store losses for each epoch
    losses={}
    for epoch in range(num_epochs):     
        #Create batch generator 
        batches = minibatch(train_data, size=1024)

        #Iterate through minibatches
        for batch in batches:
            #Each batch is a list of (text, label) tuples.
            #But we need to send separate lists for texts and labels to update().
            texts, labels = zip(*batch)
            nlp.update(texts, labels, drop=0.1, sgd=optimizer, losses=losses)
        print(losses)

#Defining arguments to enter into the function
optimizer = nlp.begin_training() #Initialize the model weights randomly
epochs = 10

#Call the function with the defined parameters to train the nlp model
#train(X_train, y_train, optimizer, epochs)
#Optional: save the model

#Use the open() function to open a file. Set the file mode to 'wb' to open the file for writing in binary mode. 
#Wrap it in a with statement to ensure the file is closed automatically when you’re done with it.
#The dump() function in the pickle module takes a serializable Python data structure, serializes it into a binary, 
#Python-specific format using the latest version of the pickle protocol, and saves it to an open file.


#with open('spaCy_sent140_model_v3', 'wb') as file:
    #pickle.dump(nlp, file)
#Loading in the model

#The pickle module uses a binary data format,so you should always open pickle files in binary mode.
#The pickle.load() function takes a stream object, reads the serialized data from the stream, 
#creates a new Python object, recreates the serialized data in the new Python object, and returns the new Python object.
with open('spaCy_sent140_model_v2', 'rb') as file:
    nlp = pickle.load(file)
    
#Make a function to predict sentiment of a tweet
import random
def pred_sent(pred_data):  
    """
    Function used to predict sentiment of text documents, returns a pandas dataframe of predictions.
    
    pred_data = A pandas series/dataframe or numpy array of texts/documents
    """  
    #Tokenize the documents/tweets
    tokenized_data = [nlp.tokenizer(tweet) for tweet in pred_data]
    
    #Use the textcategorizer object from the trained model to make predictions
    textcat = nlp.get_pipe('textcat')
    
    #Use textcat to get the score for each tweet's label
    scores,_ = textcat.predict(tokenized_data)
    
    #Get model predictions from the score
    predicted_labels = scores.argmax(axis=1)
    
    #Pick a random tweet from the df
    number = random.randrange(1,pred_data.shape[0])
    
    #Show an example of what the score looks like for each label
    print(f'Random tweet:  \n{pred_data[number]}\n')
    print('Example of scores for the random tweet:\nnegative    positive')
    print(scores[number])
    print('The tweets label is: ', textcat.labels[predicted_labels[number]])
    
    #Convert the predicted labels from integers to negative and positive labelsto match the format of the validation labels
    predicted_labels = pd.DataFrame(predicted_labels)
    predicted_labels.replace({0:'negative', 1:'positive'}, inplace=True)
    
    #return the predicted labels to assess the models performance
    return predicted_labels
#Make predictions for the sentiment140 validation set
val_preds = pred_sent(X_val)

#Compare validation predictions to the labels
print('\n', confusion_matrix(val_preds,y_val))
print(classification_report(val_preds,y_val))
#Predict the sentiment of the tweets mentioning Chris D'Elia
tweets['Sentiment'] = pred_sent(tweets_array)

#Save the results
#tweets.to_csv('tweets_v2.csv')
#Import tweets and sentiment
tweets = pd.read_csv('tweets_v2.csv', index_col='Unnamed: 0')

#Plot the sentiment distribution
sns.countplot(tweets['Sentiment'])
plt.title('Sentiment Distribution - 2020 Tweets', size=15, fontweight='bold')
plt.show()

#Positive tweets
pos_2020 = tweets[tweets['Sentiment']=='positive']['Sentiment'].count()

#Negative tweets
neg_2020 = tweets[tweets['Sentiment']=='negative']['Sentiment'].count()

#Print the ratio of negative to positive tweets
print(f'Ratio of negative:positive tweets \n{neg/neg} : {(pos_2020/neg_2020).round(2)}')
#Download and save the tweets
#downloaded_tweets = download_tweets("2019-06-01","2019-08-01", "Chris DElia", 1000, 60)
#downloaded_tweets.to_csv('summer_2019_tweets.csv')

#Import tweets and sentiment
tweets_2019 = pd.read_csv('/kaggle/input/chris-delia-twitter-info/summer_2019_tweets.csv', index_col='Unnamed: 0')

#Clean the dataframe
tweets_2019 = clean_tweets(tweets_2019)

#Setup data for the spaCy model - TextCategorizer
tweets_2019_array = tweets_2019['tweet'].values

#Implement the sentiment function to get labels
tweets_2019['Sentiment'] = pred_sent(tweets_2019_array)

#Create the subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))

#Plot the sentiment distribution
sns.countplot(tweets_2019['Sentiment'], ax=axes[0])
axes[0].set_title('Sentiment Distribution - 2019 Tweets', size=15, fontweight='bold')

#Plot the sentiment distribution
sns.countplot(tweets['Sentiment'], ax=axes[1])
axes[1].set_title('Sentiment Distribution - 2020 Tweets', size=15, fontweight='bold')
plt.show()

#Positive tweets
pos_2019 = tweets_2019[tweets_2019['Sentiment']=='positive']['Sentiment'].count()

#Negative tweets
neg_2019 = tweets_2019[tweets_2019['Sentiment']=='negative']['Sentiment'].count()

#Print the ratio of negative to positive tweets
print(f'Ratio of negative:positive tweets for Summer 2019 \n{neg_2019/neg_2019} : {(pos_2019/neg_2019).round(2)}')
print(f'\nRatio of negative:positive tweets for Summer 2020 \n1 : {(pos_2020/neg_2020).round(2)}')
#Import tweets and sentiment
tweets = pd.read_csv('tweets_v2.csv', index_col='Unnamed: 0')

#Import chris's follower count over that month
followers = pd.read_csv('/kaggle/input/chris-delia-twitter-info/2020_followers.csv', index_col='Unnamed: 0')

#Create subplots
fig, axes = plt.subplots(ncols=1,nrows=2, figsize=(22,18), constrained_layout=True)

#Define Xticklabels & Yticklabels for sentiment distribution
x_ticks = tweets['date'].unique()
y_ticks_sent = [y for y in range(0,4001,500)]

#Define Yticklabels for follower count
y_ticks_followers = [y for y in range(1_110_000, 1_170_000, 10000)]

#Plot Chris's follower count over the dates 2020-06-12 to 2020-07-27 
sns.lineplot(x='date', y='followers', data=followers, ax=axes[0], markers=True, linewidth=4, marker='o', markersize=14)
axes[0].set_title("Chris's Follower Count 2020: June & July", size=30, fontweight="bold")
axes[0].set_xlabel('Date', size=20, fontweight="bold")
axes[0].set_ylabel('Follower Count', size=20, fontweight="bold")
axes[0].set_xticklabels(x_ticks, rotation=65, size=20, horizontalalignment='right')
axes[0].set_yticklabels(y_ticks_followers,size=20)
axes[0].annotate('Date of accusation: June 16th', xy=(4,1168178.0), xycoords='data', xytext=(20, 1160000), size=20,
                 arrowprops={'width':4.0, 'headwidth':6, 'headlength':6, 'facecolor':'peru',
                            'connectionstyle':'angle3'},
                 bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="peru", lw=4))
axes[0].annotate("Chris's Response: June 17th", xy=(5,1161441.0), xycoords='data', xytext=(20, 1150000), size=20,
                 arrowprops={'width':4.0, 'headwidth':6, 'headlength':6, 'facecolor':'peru',
                           'connectionstyle':'angle3'},
                 bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="peru", lw=4))

#Plot the sentiment of each tweet during that same period
sns.countplot(x='date', data=tweets, hue='Sentiment', ax=axes[1])
axes[1].set_title('Tweet Sentiment Distribution 2020: June & July', size=30, fontweight="bold")
axes[1].set_xlabel('Date', size=20, fontweight="bold")
axes[1].set_ylabel('Count', size=20, fontweight="bold")
axes[1].set_xticklabels(x_ticks, rotation=65, size=20, horizontalalignment='right')
axes[1].set_yticklabels(y_ticks_sent, size=20)
axes[1].legend(fontsize=20)

plt.show()
#First make a function to organize the tweets and follower counts into a single df 
#This df will consist of all the tweets/RTs/Favs/follower difference in a day
def tweet_follower_merge(tweet_df, follower_df):
    """
    This function takes in two dataframes and merges them for OLS analysis.
    One df should consist of the scraped tweets, the df should be cleaned and the tweet sentiment should be included.
    The second df should consist of the follower count and follower difference for each day.
    """
    #Define the number of negative tweets for each day
    num_neg_tweets = pd.DataFrame(tweet_df[tweet_df['Sentiment']=='negative'].groupby('date')['tweet'].count())
    num_neg_tweets.reset_index(inplace=True)

    #Define the number of negative RT's and Favorites associated with the tweets
    agg_tweet_data = pd.DataFrame(tweet_df[tweet_df['Sentiment']=='negative'].groupby('date').sum())
    agg_tweet_data.reset_index(inplace=True)

    #Use dates to merge (SQL join) df's to followers.
    follower_df = pd.merge(left=follower_df, right=num_neg_tweets, how='inner', on='date')
    follower_df = pd.merge(left=follower_df, right=agg_tweet_data, how='inner', on='date')

    #Fix column headers
    follower_df.columns = ['date', 'followers', 'follower_loss', 'num_neg_tweets', 'neg_favs', 'neg_RTs']
    
    #Make follower_loss into positive numbers
    follower_df['follower_loss'] = abs(follower_df['follower_loss'])
    
    #Return the merged dataframes
    return follower_df


#Drop dates before June 16th, they are not relevant to analysis
followers.drop(index=followers.index[0:4], axis=0, inplace=True)
followers.reset_index(drop=True, inplace=True)

#Create the new df with tweets and follower counts
ols_df = tweet_follower_merge(tweets, followers)

#Get descriptive statistics of data
ols_df.describe()
#Check correlation of data
ols_df.corr()
#Plot features as a function of the target distribution
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20,12))

#axes[0][0] - Number of followers lost
sns.distplot(ols_df['follower_loss'], ax=axes[0][0], bins=50)
axes[0][0].set_title('Follower Loss (target)', size = 15, fontweight='bold')

#axes[0][1] - Number of negative tweets
sns.distplot(ols_df['num_neg_tweets'], ax=axes[0][1], bins=50)
axes[0][1].set_title('Negative Tweets', size = 15, fontweight='bold')

#axes[1][0] - Number of Favorites
sns.distplot(ols_df['neg_favs'], ax=axes[1][0], bins=50)
axes[1][0].set_title('Neg. Favorites', size = 15, fontweight='bold')

#axes[1][1] - Number of Retweets
sns.distplot(ols_df['neg_RTs'], ax=axes[1][1], bins=50)
axes[1][1].set_title('Neg. Retweets', size = 15, fontweight='bold')

plt.show()
#Drop the accusation day and the two days after
ols_df = ols_df[ols_df['follower_loss'] < 6000]
ols_df.reset_index(drop=True, inplace=True)

#Plot features as a function of the target distribution
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20,12))

#axes[0][0] - Number of followers lost
sns.distplot(ols_df['follower_loss'], ax=axes[0][0], bins=50)
axes[0][0].set_title('Follower Loss (target)', size = 15, fontweight='bold')

#axes[0][1] - Number of negative tweets
sns.distplot(ols_df['num_neg_tweets'], ax=axes[0][1], bins=50)
axes[0][1].set_title('Negative Tweets', size = 15, fontweight='bold')

#axes[1][0] - Number of Favorites
sns.distplot(ols_df['neg_favs'], ax=axes[1][0], bins=50)
axes[1][0].set_title('Neg. Favorites', size = 15, fontweight='bold')

#axes[1][1] - Number of Retweets
sns.distplot(ols_df['neg_RTs'], ax=axes[1][1], bins=50)
axes[1][1].set_title('Neg. Retweets', size = 15, fontweight='bold')

plt.show()
#Logarithmically transform follower loss
ols_df['log_follower_loss'] = np.log(ols_df['follower_loss'])

#Create list of features 
features = ['num_neg_tweets','neg_favs','neg_RTs']

#Plot features as a function of the target distribution
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(20,5))

#Plot each feature against the target to assess linearity
for i,feature in enumerate(features):
    sns.scatterplot(x=feature, y='follower_loss', data=ols_df, ax=axes[i])
    axes[i].set_title(f'Follower Loss vs. {feature}', size=14, fontweight='bold')
    
plt.show()
#Create function to perform logarithmic transformations
def log_transform(df, list_1, list_2):
    """
    Function to assess datapoints in a df for logarithmic transformation and removes unsuitable datapoints.
    Creates new columns that are logarithmic transformations of the columns entered.
    
    df = dataframe that contains columns
    list_1 = List of column titles that you want to logarithmically transform
    list_2 = List of new column titles for logarithmic transformations
    """
    #Create storage place for index
    index_storage = []
    
    #Iterate through rows and store index that is unsuitable for log transformation
    for index,row in df.iterrows():
        for feature in list_1:
            if row[feature] <= 0:
                index_storage.append(index)
    
    #Drop the rows that have unsuitable datapoints
    for i in index_storage:
        df.drop(df.index[i], axis=0, inplace=True)
    
    #Create log transformations of columns
    for i in range(len(list_1)):
        df[list_2[i]] = np.log(df[list_1[i]])
        
    #Reset the df index
    df.reset_index(drop=True, inplace=True)
      
    #Return df
    return df

#Create list for log_transform
features = ['num_neg_tweets','neg_favs','neg_RTs']
log_features = ['log_neg_tweets','log_favs','log_RTs']

#Make log transformations
ols_df = log_transform(ols_df, features, log_features)

#Plot features as a function of the target distribution
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(20,5))

#Create list of features for plot
plot = ['log_neg_tweets', 'log_favs','log_RTs']


#Plot each feature against the target to assess linearity
for i,feature in enumerate(plot):
    sns.scatterplot(x=feature, y='follower_loss', data=ols_df, ax=axes[i])
    axes[i].set_title(f'Follower Loss vs. {feature}', size=14, fontweight='bold')
    
plt.show()
#Make log transformations
ols_df = log_transform(ols_df, ['follower_loss'], ['log_follower_loss'])
ols_df.reset_index(drop=True, inplace=True)

#Plot features as a function of the target distribution
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(20,5))

#Plot each feature against the target to assess linearity
for i,feature in enumerate(plot):
    sns.scatterplot(x=feature, y='log_follower_loss', data=ols_df, ax=axes[i])
    axes[i].set_title(f'log(Follower Loss) vs. {feature}', size=14, fontweight='bold')
    
plt.show()
ols_df = ols_df[ols_df['log_favs'] < 7]
ols_df.reset_index(drop=True, inplace=True)

#Plot features as a function of the target distribution
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(20,5))

#Plot each feature against the target to assess linearity
for i,feature in enumerate(plot):
    sns.scatterplot(x=feature, y='log_follower_loss', data=ols_df, ax=axes[i])
    axes[i].set_title(f'log(follower Loss) vs. {feature}', size=14, fontweight='bold')
    
plt.show()
#Import the statsmodels VIF calculator
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Define the variables being tested for VIF
variables = ols_df[plot]

#Make a storage place for the VIF values
vif = pd.DataFrame()

#Here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
vif
#Import the statsmodels VIF calculator
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Define the variables being tested for VIF
new_variables = ols_df[['log_neg_tweets', 'log_RTs']]

#Make a storage place for the VIF values
vif = pd.DataFrame()

#Here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(new_variables.values, i) for i in range(new_variables.shape[1])]
vif["Features"] = new_variables.columns
vif
#### Import the statsmodels api and standard scaler
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Define inputs & Targets
x = ols_df[['log_neg_tweets', 'log_RTs']]
y = ols_df['log_follower_loss']

#Create the standard scaler object and use it to transform the features
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x))
x.columns = ['log_neg_tweets', 'log_RTs']

#Add an intercept to the features/ind. vars. 
X = sm.add_constant(x)

#Implement train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

#Fit the data to the simple OLS linear model
lin_reg = sm.OLS(y_train, X_train).fit()

#Printout statsmodels summary of OLS model
lin_reg.summary()
#Define inputs & Targets
x = ols_df['log_neg_tweets']
y = ols_df['log_follower_loss']

#Create the standard scaler object and use it to transform the features
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x.values.reshape(-1,1)))
x.columns = ['log_neg_tweets']

#Add an intercept to the features/ind. vars. 
X = sm.add_constant(x)

#Implement train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

#Fit the data to the simple OLS linear model
lin_reg = sm.OLS(y, X).fit()

#Printout statsmodels summary of OLS model
lin_reg.summary()
#Import Regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Make predictions
predictions = lin_reg.predict(X_test)

#Exponentially transform target predictions and y_test values to make data easily understandable 
#y_test = np.exp(y_test)
#predictions = np.exp(predictions)

#Use MSE to measure error values
RMSE = np.sqrt(mean_squared_error(y_test, predictions))
MAE = mean_absolute_error(y_test, predictions)
print(f'RMSE = {RMSE.round(1)}')
print(f'MAE = {MAE.round(1)}')

#Store predictions
prediction_df = pd.DataFrame({'predictions':predictions, 'target':y_test})

#Add errors to df
prediction_df['error'] = prediction_df['predictions'] - prediction_df['target']
#Checking residual plot  --> autocorrelation, no identifiable relationship b/w residuals.
sns.residplot(x=X_test['log_neg_tweets'], y=prediction_df['error'])
#Group the users by the number of tweets and use this to create 
user_tweets = pd.DataFrame(tweets.groupby('username').count()['tweet'])
user_tweets.reset_index(inplace=True)
user_tweets.columns = ['username', 'count']

#Sort the number of times people tweet into categories using a new function
def sort_tweet_count(column):
    if column == 1:
        return '1 tweet'
    elif 1 < column < 6:
        return '2 - 5 tweets'
    else:
        return '6 Tweets or more'

#Create categories for the number of tweets
user_tweets['count_category'] = user_tweets['count'].apply(sort_tweet_count)
plt.figure(figsize=(10,8))
sns.countplot(x='count_category', data=user_tweets)
plt.xlabel('Users Tweet Counts', size=15, fontweight='bold')
plt.ylabel('# of Users', size=15, fontweight='bold')
#Find usernames corresponding to each category
user_cat1 = list(user_tweets[user_tweets['count_category'] == '1 tweet']['username'])
user_cat2 = list(user_tweets[user_tweets['count_category'] == '2 - 5 tweets']['username'])
user_cat3 = list(user_tweets[user_tweets['count_category'] == '6 Tweets or more']['username'])

#Create plots
fig,axes = plt.subplots(nrows=1, ncols=3, figsize=(20,7))

#Plot each category
sns.countplot(x='Sentiment', data=tweets[tweets['username'].isin(user_cat1)], ax=axes[0])
axes[0].set_title('Sent. Dist. for User Categories: 1 Tweet', size=15, fontweight='bold')
axes[0].set_ylabel('Tweet Count', size = 13)

sns.countplot(x='Sentiment', data=tweets[tweets['username'].isin(user_cat2)], ax=axes[1])
axes[1].set_title('Sent. Dist. for User Categories: 2 - 5 Tweets', size=15, fontweight='bold')
axes[1].set_ylabel('Tweet Count', size = 13)

sns.countplot(x='Sentiment', data=tweets[tweets['username'].isin(user_cat3)], ax=axes[2])
axes[2].set_title('Sent. Dist. for User Categories: 6+ Tweets', size=15, fontweight='bold')
axes[2].set_ylabel('Tweet Count', size = 13)

#Defining the negative and positive tweets for each category
cat_neg1 = tweets[(tweets['username'].isin(user_cat1)) & (tweets['Sentiment'] == 'negative')]['Sentiment'].count()
cat_pos1 = tweets[(tweets['username'].isin(user_cat1)) & (tweets['Sentiment'] == 'positive')]['Sentiment'].count()
cat_neg2 = tweets[(tweets['username'].isin(user_cat2)) & (tweets['Sentiment'] == 'negative')]['Sentiment'].count()
cat_pos2 = tweets[(tweets['username'].isin(user_cat2)) & (tweets['Sentiment'] == 'positive')]['Sentiment'].count()
cat_neg3 = tweets[(tweets['username'].isin(user_cat3)) & (tweets['Sentiment'] == 'negative')]['Sentiment'].count()
cat_pos3 = tweets[(tweets['username'].isin(user_cat3)) & (tweets['Sentiment'] == 'positive')]['Sentiment'].count()

print('negative:positive sentiment ratio for... \n1 tweet: 1 : {0}'.format((cat_pos1/cat_neg1).round(2)))
print('2-5 tweets: 1 : {0}'.format((cat_pos2/cat_neg2).round(2)))
print('6+ tweets: 1 : {0}'.format((cat_pos3/cat_neg3).round(2)))
#Defining the top 10 hashtags used in the tweets
top_hashtags = list(tweets['hashtags'].value_counts()[1:11].index)

fig,axes = plt.subplots(ncols=1, nrows=2, figsize=(10,15))

sns.countplot(x='hashtags', data=tweets[tweets['hashtags'] == 'None'], hue='Sentiment', ax=axes[0])
axes[0].set_title('Sent.Dist. of Tweets with no Hashtags', size=15, fontweight='bold')
axes[0].set_xlabel('Hashtags', size=12, fontweight='bold')
axes[0].set_ylabel('Number of tweets', size=12, fontweight='bold')

sns.countplot(y='hashtags', data=tweets[tweets['hashtags'].isin(top_hashtags)], hue='Sentiment', ax=axes[1])
axes[1].set_ylabel('Hashtags', size=12, fontweight='bold')
axes[1].set_xlabel('# of Tweets', size=12, fontweight='bold')
axes[1].set_title('Top 10 Hashtags Sent. Dist.', size=15, fontweight='bold')