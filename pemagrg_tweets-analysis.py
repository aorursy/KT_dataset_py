# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import re
import string
import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from plotly.subplots import make_subplots
# Load libraries
!pip install -U tensorflow==1.15.2
import tensorflow
print(tensorflow.__version__) # make sure the version of tensorflow
import numpy as np # for scientific computing
import pandas as pd # for data analysis
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization
import missingno as msno # for missing data visualization
import collections
import nltk
import codecs
import string
import re
from tqdm import tqdm
from collections import defaultdict
from collections import Counter 
from keras.initializers import Constant
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SpatialDropout1D, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dropout
plt.style.use('ggplot')
np.random.seed(42) # set the random seeds

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop=set(stopwords.words('english'))
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
train_df.head()
fig=make_subplots(1,2,subplot_titles=('Train set','Test set'))
x=train_df.sentiment.value_counts()
fig.add_trace(go.Bar(x=x.index,y=x.values,marker_color=['blue','green','red'],name='train'),row=1,col=1)
x=test_df.sentiment.value_counts()
fig.add_trace(go.Bar(x=x.index,y=x.values,marker_color=['blue','green','red'],name='test'),row=1,col=2)
temp = train_df.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)

temp.style.background_gradient(cmap='Purples')
plt.figure(figsize=(12,6))
sns.countplot(x='sentiment',data=train_df)
fig = go.Figure(go.Funnelarea(
    text =temp.sentiment,
    values = temp.text,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def remove_stopword(x):
    return [y for y in x if y not in stopwords.words('english')]

train_df['text'] = train_df['text'].apply(lambda x:clean_text(x))
train_df['selected_text'] = train_df['selected_text'].apply(lambda x:clean_text(x))

train_df['temp_list'] = train_df['selected_text'].apply(lambda x:str(x).split())
train_df['temp_list'] = train_df['temp_list'].apply(lambda x:remove_stopword(x))

top = Counter([item for sublist in train_df['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')

fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')
fig.show()
fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Text', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()
#MosT common positive/neutral/neg words


def plot_by_sentiment(sentiment,title):
    top = Counter([item for sublist in sentiment['temp_list'] for item in sublist])
    temp_sentiment = pd.DataFrame(top.most_common(20))
    temp_sentiment.columns = ['Common_words','count']
    temp_sentiment.style.background_gradient(cmap='Greens')

    fig = px.bar(temp_sentiment, x="count", y="Common_words", title=title, orientation='h', 
                 width=700, height=700,color='Common_words')
    fig.show()

Positive_sent = train_df[train_df['sentiment']=='positive']
Negative_sent = train_df[train_df['sentiment']=='negative']
Neutral_sent = train_df[train_df['sentiment']=='neutral']
plot_by_sentiment(Positive_sent,title="Most Commmon Positive Words")
plot_by_sentiment(Negative_sent,title="Most Commmon Negative Words")
def words_unique(sentiment,numwords,raw_words):
    '''
    Input:
        segment - Segment category (ex. 'Neutral');
        numwords - how many specific words do you want to see in the final result; 
        raw_words - list  for item in train_data[train_data.segments == segments]['temp_list1']:
    Output: 
        dataframe giving information about the name of the specific ingredient and how many times it occurs in the chosen cuisine (in descending order based on their counts)..

    '''
    allother = []
    for item in train_df[train_df.sentiment != sentiment]['temp_list']:
        for word in item:
            allother .append(word)
    allother  = list(set(allother ))
    
    specificnonly = [x for x in raw_text if x not in allother]
    
    mycounter = Counter()
    
    for item in train_df[train_df.sentiment == sentiment]['temp_list']:
        for word in item:
            mycounter[word] += 1
    keep = list(specificnonly)
    
    for word in list(mycounter):
        if word not in keep:
            del mycounter[word]
    
    Unique_words = pd.DataFrame(mycounter.most_common(numwords), columns = ['words','count'])
    
    return Unique_words

raw_text = [word for word_list in train_df['temp_list'] for word in word_list]
Unique_Positive= words_unique('positive', 20, raw_text)
print("The top 20 unique words in Positive Tweets are:")
Unique_Positive.style.background_gradient(cmap='Greens')
fig = px.treemap(Unique_Positive, path=['words'], values='count',title='Tree Of Unique Positive Words')
fig.show()
from palettable.colorbrewer.qualitative import Pastel1_7
plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.pie(Unique_Positive['count'], labels=Unique_Positive.words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Unique Positive Words')
plt.show()


"""
https://www.kaggle.com/aladdint/sentiment-analysis-using-ltsm-cnn
"""
import re
from matplotlib import pyplot as plt
import numpy as np
import tensorflow
import numpy as np # for scientific computing
import pandas as pd # for data analysis
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization
import missingno as msno # for missing data visualization
import collections
import nltk
import codecs
import string
import re
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
from keras.initializers import Constant
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SpatialDropout1D, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dropout

plt.style.use('ggplot')
np.random.seed(42) # set the random seeds

train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
train = train[train['text'] != '']
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
test = test[test['text'] != '']

"""
Index(['textID', 'text', 'selected_text', 'sentiment'], dtype='object')
"""
# Remove URL
url = re.compile(r'https?://\S+|www\.\S+')
train['text'] = train['text'].apply((lambda x: url.sub(r'', str(x))))
test['text'] = test['text'].apply((lambda x: url.sub(r'', x)))
# Remove HTML
html = re.compile(r'<.*?>')
train['text'] = train['text'].apply((lambda x: html.sub(r'', x)))
test['text'] = test['text'].apply((lambda x: html.sub(r'', x)))
# Remove the words which contain the number
train['text'] = train['text'].apply((lambda x: re.sub('\w*\d\w*', '', x)))
test['text'] = test['text'].apply((lambda x: re.sub('\w*\d\w*', '', x)))

# Remove stop words
# train['text'] = train['text'].apply(lambda x: [item for item in x if item not in stopwords.words('english')])
# test['text'] = test['text'].apply(lambda x: [item for item in x if item not in stopwords.words('english')])

# Make the tweet lower letters
train['text'] = train['text'].apply(lambda x: x.lower())
test['text'] = test['text'].apply(lambda x: x.lower())
# Remove punctuation
table = str.maketrans('', '', string.punctuation)
train['text'] = train['text'].apply((lambda x: x.translate(table)))
test['text'] = test['text'].apply((lambda x: x.translate(table)))


# Let's count the max length of tweet in both train and test data for turning tweets into seaquences
max_len = 0
for i in train['text']:
    split_i = i.split()
    if len(split_i) > max_len:
        max_len = len(split_i)

for j in test['text']:
    split_j = j.split()
    if len(split_j) > max_len:
        max_len = len(split_j)

print('Max length of tweets :', max_len)
# Convert the tweets into the sequences in train and test data

max_fatures = 300000  # the number of words to be used for the input of embedding layer
tokenizer = Tokenizer(num_words=max_fatures, split=' ')  # Create the instance of Tokenizer
tokenizer.fit_on_texts(train['text'].values)
train_converted = tokenizer.texts_to_sequences(train['text'].values)
test = tokenizer.texts_to_sequences(test['text'].values)
train_converted = pad_sequences(train_converted,
                                maxlen=max_len)  # Turning the vectors of train data into sequences
test = pad_sequences(test, maxlen=max_len)  # Turning the vectors of test data into sequences

target_converted = pd.get_dummies(train['sentiment']).values # One-hot expression
# Make sure that the shape of train and test data are same
X_train, X_test, Y_train, Y_test = train_test_split(train_converted, target_converted, test_size = 0.1, random_state = 42)

# Use half of the test data for validation during training
validation_size = 50000
# validation_size = 500
X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

print('The shape of train data :', X_train.shape)
print('The shape of labels of train data :', Y_train.shape)
print('The shape of test data :', X_test.shape)
print('The shape of test label data :', Y_test.shape)
# Parameters
embed_dim = 1024 # The size of the vector space where words will be embedded
lstm_out = 196 # The output size of lstm layer
batch_size = 1024
EPOCHS = 2

# Create the LSTM model
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = train_converted.shape[1]))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(lstm_out, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(3,activation='softmax')) #3 cuz we hav 3 classes
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['acc'])
print(model.summary()) # Show the summary of the model

history = model.fit(X_train, Y_train, epochs = EPOCHS, batch_size=batch_size,
                    validation_data=(X_validate, Y_validate), verbose = 2)

# Plot the result of trained model
train_acc = history.history['acc']
test_acc = history.history['val_acc']
x = np.arange(len(train_acc))
plt.plot(x, train_acc, label = 'train accuracy')
plt.plot(x, test_acc, label = 'test accuracy')
plt.title('Train and validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend()

# Let's compute the loss and accuracy of the trained model
score, acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("The loss of this model: %.2f" % (score))
print("The accuracy of this model: %.2f" % (acc))

#Prediction and Submission¶



