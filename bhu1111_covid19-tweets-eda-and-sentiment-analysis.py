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
import pandas as pd



df = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')
df.head()
df.tail()
df.describe()
print('Shape of tweets dataframe : {}'.format(df.shape))
df.info()




import seaborn as sns

import matplotlib.pyplot as plt



def return_missing_values(data_frame):

    missing_values = data_frame.isnull().sum()/len(data_frame)

    missing_values = missing_values[missing_values>0]

    missing_values.sort_values(inplace=True)

    return missing_values



def plot_missing_values(data_frame):

    missing_values = return_missing_values(data_frame)

    missing_values = missing_values.to_frame()

    missing_values.columns = ['count']

    missing_values.index.names = ['Name']

    missing_values['Name'] = missing_values.index

    sns.set(style='whitegrid', color_codes=True)

    sns.barplot(x='Name', y='count', data=missing_values)

    plt.xticks(rotation=90)

    plt.show()

     




return_missing_values(df)
plot_missing_values(df)
# heatmap representation of missing values



# plasma,visdir



sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='ocean')
def return_unique_values(data_frame):

    unique_dataframe = pd.DataFrame()

    unique_dataframe['Features'] = data_frame.columns

    uniques = []

    for col in data_frame.columns:

        u = data_frame[col].nunique()

        uniques.append(u)

    unique_dataframe['Uniques'] = uniques

    return unique_dataframe
udf = return_unique_values(df)

print(udf)
f, ax = plt.subplots(1,1, figsize=(10,5))#plt.figure(figsize=(10, 5))



sns.barplot(x=udf['Features'], y=udf['Uniques'], alpha=0.8)

plt.title('Bar plot for #unique values in each column')

plt.ylabel('#Unique values', fontsize=12)

plt.xlabel('Features', fontsize=12)

plt.xticks(rotation=90)

plt.show()
def plot_frequency_charts(df, feature, title, pallete):

    freq_df = pd.DataFrame()

    freq_df[feature] = df[feature]

    

    f, ax = plt.subplots(1,1, figsize=(16,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette=pallete)

    g.set_title("Number and percentage of {}".format(title))



    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 



    plt.title('Frequency of {} tweeting about Corona'.format(feature))

    plt.ylabel('Frequency', fontsize=12)

    plt.xlabel(title, fontsize=12)

    plt.xticks(rotation=90)

    plt.show()

    
plot_frequency_charts(df, 'user_name', 'User Names','Wistia')
plot_frequency_charts(df, 'user_location', 'User Locations', 'BuGn_r')
plot_frequency_charts(df, 'source','Source', 'vlag')
from string import punctuation

from nltk.corpus import stopwords

print(stopwords.words('english')[10:15])



def punctuation_stopwords_removal(sms):

    # filters charecter-by-charecter : ['h', 'e', 'e', 'l', 'o', 'o', ' ', 'm', 'y', ' ', 'n', 'a', 'm', 'e', ' ', 'i', 's', ' ', 'p', 'u', 'r', 'v', 'a']

    remove_punctuation = [ch for ch in sms if ch not in punctuation]

    # convert them back to sentences and split into words

    remove_punctuation = "".join(remove_punctuation).split()

    filtered_sms = [word.lower() for word in remove_punctuation if word.lower() not in stopwords.words('english')]

    return filtered_sms
df.head()
from collections import Counter



def draw_bar_graph_for_text_visualization(df, location):

    tweets_from_loc = df.loc[df.user_location==location]

    tweets_from_loc.loc[:, 'text'] = tweets_from_loc['text'].apply(punctuation_stopwords_removal)

    loc_tweets_curated = tweets_from_loc['text'].tolist()

    loc_tweet_list = []

    for sublist in loc_tweets_curated:

        for word in sublist:

            loc_tweet_list.append(word)

    loc_tweet_count = Counter(loc_tweet_list)

    loc_top_30_words = pd.DataFrame(loc_tweet_count.most_common(50), columns=['word', 'count'])

    fig, ax = plt.subplots(figsize=(16, 6))

    sns.barplot(x='word', y='count', 

                data=loc_top_30_words, ax=ax)

    plt.title("Top 50 Prevelant Words in {}".format(location))

    plt.xticks(rotation='vertical');

    
from wordcloud import WordCloud, STOPWORDS







def draw_word_cloud(df, location, title):

    loc_df = df.loc[df.user_location==location]

    loc_df.loc[:, 'text'] = loc_df['text'].apply(punctuation_stopwords_removal)

    word_cloud = WordCloud(

                    background_color='white',

                    stopwords=set(STOPWORDS),

                    max_words=50,

                    max_font_size=40,

                    scale=5,

                    random_state=1).generate(str(loc_df['text']))

    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    fig.suptitle(title, fontsize=20)

    fig.subplots_adjust(top=2.3)

    plt.imshow(word_cloud)

    plt.show()

    
draw_bar_graph_for_text_visualization(df, 'India')
draw_word_cloud(df, 'India', 'Word Cloud for top 50 prevelant words in India')
draw_bar_graph_for_text_visualization(df, 'United Kingdom')
draw_word_cloud(df, 'United Kingdom', 'Word Cloud for top 50 prevelant words in United Kingdom')
draw_bar_graph_for_text_visualization(df, 'Canada')
draw_word_cloud(df, 'Canada', 'Word Cloud for top 50 prevelant words in Canada')
draw_bar_graph_for_text_visualization(df, 'South Africa')
draw_word_cloud(df, 'South Africa', 'Word Cloud for top 50 prevelant words in South Africa')
draw_bar_graph_for_text_visualization(df, 'Switzerland')
draw_word_cloud(df, 'Switzerland', 'Word Cloud for top 50 prevelant words in Switzerland')
draw_bar_graph_for_text_visualization(df, 'London')
draw_word_cloud(df, 'London', 'Word Cloud for top 50 prevelant words in London')
sentiment_df = pd.read_csv('/kaggle/input/twitterdata/finalSentimentdata2.csv')
sentiment_df.head()
sentiment_df.columns
sentiment_df['sentiment'].nunique
sentiment_df.loc[:, 'text'] = sentiment_df['text'].apply(punctuation_stopwords_removal)
reviews_split = []

for i, j in sentiment_df.iterrows():

    reviews_split.append(j['text'])

words = []

for review in reviews_split:

    for word in review:

        words.append(word)

print(words[:20])
from collections import Counter



counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word:ii for ii, word in enumerate(vocab, 1)}
encoded_reviews = []

for review in reviews_split:

    encoded_reviews.append([vocab_to_int[word] for word in review])

print(len(vocab_to_int))

print(encoded_reviews[:10])
labels_to_int = []

for i, j in sentiment_df.iterrows():

    if j['sentiment']=='joy':

        labels_to_int.append(1)

    else:

        labels_to_int.append(0)

    
reviews_len = Counter([len(x) for x in encoded_reviews])

print(max(reviews_len))
print(len(encoded_reviews))
non_zero_idx = [ii for ii, review in enumerate(encoded_reviews) if len(encoded_reviews)!=0]

encoded_reviews = [encoded_reviews[ii] for ii in non_zero_idx]

encoded_labels = np.array([labels_to_int[ii] for ii in non_zero_idx])
print(len(encoded_reviews))

print(len(encoded_labels))
def pad_features(reviews_int, seq_length):

    features = np.zeros((len(reviews_int), seq_length), dtype=int)

    for i, row in enumerate(reviews_int):

        if len(row)!=0:

            features[i, -len(row):] = np.array(row)[:seq_length]

    return features
seq_length = 50

padded_features= pad_features(encoded_reviews, seq_length)

print(padded_features[:2])

split_frac = 0.8

split_idx = int(len(padded_features)*split_frac)



training_x, remaining_x = padded_features[:split_idx], padded_features[split_idx:]

training_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]



test_idx = int(len(remaining_x)*0.5)

val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]

val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

import torch

from torch.utils.data import TensorDataset, DataLoader
# torch.from_numpy creates a tensor data from n-d array

train_data = TensorDataset(torch.from_numpy(training_x), torch.from_numpy(training_y))

test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))



batch_size = 1



train_loader = DataLoader(train_data, batch_size=batch_size)

test_loader = DataLoader(test_data, batch_size=batch_size)

valid_loader = DataLoader(valid_data, batch_size=batch_size)
gpu_available = torch.cuda.is_available



if gpu_available:

    print('Training on GPU')

else:

    print('GPU not available')
import torch.nn as nn



class CovidTweetSentimentAnalysis(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2):

        super(CovidTweetSentimentAnalysis, self).__init__()

        self.output_size = output_size

        self.n_layers = n_layers

        self.hidden_dim = hidden_dim

        

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_dim, output_size)

        self.sig = nn.Sigmoid()

    

    def forward(self, x, hidden):

        # x : batch_size * seq_length * features

        batch_size = x.size(0)

        x = x.long()

        embeds = self.embedding_layer(x)

        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)

        out = self.fc(out)

        sig_out = self.sig(out)

        

        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1]

        

        return sig_out, hidden

    

    def init_hidden(self, batch_size):

        # initialize weights for lstm layer

        weights = next(self.parameters()).data

        

        if gpu_available:

            hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),

                     weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())

        else:

            hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_(),

                     weights.new(self.n_layers, batch_size, self.hidden_dim).zero())

        return hidden
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens

output_size = 1 # either happy or sad

embedding_dim = 400

hidden_dim = 256

n_layers = 2
net = CovidTweetSentimentAnalysis(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)
lr = 0.001

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
epochs = 4

count = 0

print_every = 100

clip = 5 

if gpu_available:

    net.cuda()



net.train()

for e in range(epochs):

    # initialize lstm's hidden layer 

    h = net.init_hidden(batch_size)

    for inputs, labels in train_loader:

        count += 1

        if gpu_available:

            inputs, labels = inputs.cuda(), labels.cuda()

        h = tuple([each.data for each in h])

        

        # training process

        net.zero_grad()

        outputs, h = net(inputs, h)

        loss = criterion(outputs.squeeze(), labels.float())

        loss.backward()

        nn.utils.clip_grad_norm(net.parameters(), clip)

        optimizer.step()

        

        # print average training losses

        if count % print_every == 0:

            val_h = net.init_hidden(batch_size)

            val_losses = []

            net.eval()

            for inputs, labels in valid_loader:

                val_h = tuple([each.data for each in val_h])

                if gpu_available:

                    inputs, labels = inputs.cuda(), labels.cuda()

            outputs, val_h = net(inputs, val_h)

            val_loss = criterion(outputs.squeeze(), labels.float())

            val_losses.append(val_loss.item())

        

            net.train()

            print("Epoch: {}/{}...".format(e+1, epochs),

                  "Step: {}...".format(count),

                  "Loss: {:.6f}...".format(loss.item()),

                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
test_losses = []

num_correct = 0



h = net.init_hidden(batch_size)

net.eval()



for inputs, labels in test_loader:

    h = tuple([each.data for each in h])

    if gpu_available:

        inputs, labels = inputs.cuda(), labels.cuda()

    

    outputs, h = net(inputs, h)

    test_loss = criterion(outputs.squeeze(), labels.float())

    test_losses.append(test_loss.item())

    pred = torch.round(outputs.squeeze())

    correct_tensor = pred.eq(labels.float().view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not gpu_available else np.squeeze(correct_tensor.cpu().numpy())

    num_correct += np.sum(correct)



# printing average statistics

print("Test loss: {:.3f}".format(np.mean(test_losses)))

    

# accuracy over all test data

test_acc = num_correct/len(test_loader.dataset)

print("Test accuracy: {:.3f}".format(test_acc))
from string import punctuation



def tokenize_covid_tweet(tweet):

    test_ints = []

    test_ints.append([vocab_to_int[word] for word in tweet])

    return test_ints
def predict_covid_sentiment(net, test_tweet, seq_length=50):

    print('Original Sentence :')

    print(test_tweet)

    

    print('\nAfter removing punctuations and stop-words :')

    test_tweet = punctuation_stopwords_removal(test_tweet)

    print(test_tweet)

    

    print('\nAfter converting pre-processed tweet to tokens :')

    tokenized_tweet = tokenize_covid_tweet(test_tweet)

    print(tokenized_tweet)

    

    print('\nAfter padding the tokens into fixed sequence lengths :')

    padded_tweet = pad_features(tokenized_tweet, 50)

    print(padded_tweet)

    

    feature_tensor = torch.from_numpy(padded_tweet)

    batch_size = feature_tensor.size(0)

    

    if gpu_available:

        feature_tensor = feature_tensor.cuda()

    

    h = net.init_hidden(batch_size)

    output, h = net(feature_tensor, h)

    

    predicted_sentiment = torch.round(output.squeeze())

    print('\n==========Predicted Sentiment==========\n')

    if predicted_sentiment == 1:

        print('Happy')

    else:

        print('Sad')

    print('\n==========Predicted Sentiment==========\n')

test_sad_tweet = 'It is very sad to see the corona pandemic increasing at such an alarming rate'

predict_covid_sentiment(net, test_sad_tweet)
test_happy_tweet = 'It is amazing to see that New Zealand reaches 100 days without Covid transmission!'

predict_covid_sentiment(net, test_happy_tweet)