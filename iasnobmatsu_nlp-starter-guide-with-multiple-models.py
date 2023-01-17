import re 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score,train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,SpatialDropout1D,Embedding
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#import disaster tweets dataframes
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')
df_test = pd.read_csv('../input/nlp-getting-started/test.csv')

#take a look at dataset sizes
print('train shape:', df_train.shape)
print('test shape:',df_test.shape)
#take a look at the train dataset
pd.set_option('display.max_colwidth', 300) #set width of columns to display full tweet
df_train.head()
#take a look at the test dataset
df_test.head()
target_count=df_train['target'].value_counts(dropna=False) #count target
plt.figure(figsize=(5,5)) #set figure size
plt.pie([target_count[0], target_count[1]],labels=['not disaster', 'disaster'], shadow=False)#pie chart
# percent of location appearing grouped by target = 1/0
location_count_1=pd.DataFrame(df_train[df_train['target']==1]['location'].value_counts(dropna=False)) #find only disaster tweets
location_count_1 = location_count_1.reset_index() #reformate
location_count_1.columns=['location','count'] #rename headers
location_count_1['percent']=location_count_1['count']/location_count_1['count'].sum() #percentage

location_count_0=pd.DataFrame(df_train[df_train['target']==0]['location'].value_counts(dropna=False)) #only non-disaster
location_count_0 = location_count_0.reset_index() #reformat
location_count_0.columns=['location','count'] #headers
location_count_0['percent']=location_count_0['count']/location_count_0['count'].sum() #percentage

#make separate bar charts for taget =1/0
fig,a =  plt.subplots(2,1,figsize=(15,10)) #make 2 subplots 
fig.tight_layout(pad=12) #padding between subplots
print('number of different locations (disaster):', location_count_1.shape[0]) 
sns.barplot(x='location',y='count', data=location_count_1[:20], palette='Spectral', ax=a[0]) # barplot for top 20 most common 
a[0].set_title('target=1') 
a[0].tick_params(labelrotation=45)

print('number of different locations (non disaster):', location_count_0.shape[0])
sns.barplot(x='location',y='count', data=location_count_0[:20], palette='Spectral', ax=a[1]) # barplot for top 20 most common 
a[1].set_title('target=0')
a[1].tick_params(labelrotation=45)

print(location_count_0.head(1))
print(location_count_1.head(1))
# percent of keywords appearing grouped by target = 1/0
key_count_1=pd.DataFrame(df_train[df_train['target']==1]['keyword'].value_counts(dropna=False)) # only disaster
key_count_1 = key_count_1.reset_index()
key_count_1.columns=['keyword','count']
key_count_1['percent']=key_count_1['count']/key_count_1['count'].sum()

key_count_0=pd.DataFrame(df_train[df_train['target']==0]['keyword'].value_counts(dropna=False)) #only non disaster
key_count_0 = key_count_0.reset_index()
key_count_0.columns=['keyword','count']
key_count_0['percent']=key_count_0['count']/key_count_0['count'].sum()


#make separate bar charts for taget =1/0
fig,a =  plt.subplots(2,1,figsize=(15,10)) #make 2 subplots for target=1, target=0
fig.tight_layout(pad=12)
print('number of different keywords (disaster):', key_count_1.shape[0])
sns.barplot(x='keyword',y='count', data=key_count_1[:20], palette='Spectral', ax=a[0]) # barplot for top 20 most common 
a[0].set_title('target=1')
a[0].tick_params(labelrotation=45)

print('number of different keywords (non disaster):', key_count_0.shape[0])
sns.barplot(x='keyword',y='count', data=key_count_0[:20], palette='Spectral', ax=a[1]) # barplot for top 20 most common 
a[1].set_title('target=0')
a[1].tick_params(labelrotation=45)
# make a new variable for number of characters
df_train['characters']=df_train['text'].str.len()

# split dataset by target
char_1=df_train[df_train['target']==0]['characters']
char_0=df_train[df_train['target']==1]['characters']

# t test
clengtht, clengthp=scipy.stats.ttest_ind(char_1, char_0)
print('T Test number of characters by target t={:.2f},p={:.2f}'.format(clengtht, clengthp))

#histograms
fig,a =  plt.subplots(1,3,figsize=(15,5)) #make 3 subplots for target=1, target=0, and complete sample
sns.distplot(char_1,ax=a[0], color='purple')
a[0].set_title('target = 1')
sns.distplot(char_0,ax=a[1], color='blue')
a[1].set_title('target = 0')
sns.distplot(df_train['characters'],ax=a[2], color='green')
a[2].set_title('target = 0 and target = 1')
# make a new variable for number of words
df_train['words']=df_train['text'].apply(lambda x: len(str(x).split())) #split by space to turn tweet into words

# split dataset by target
w_1=df_train[df_train['target']==0]['words']
w_0=df_train[df_train['target']==1]['words']

# t test
wlengtht, wlengthp=scipy.stats.ttest_ind(w_1, w_0)
print('T Test number of words by target t={:.2f},p={:.2f}'.format(wlengtht, wlengthp))

#histograms
fig,a =  plt.subplots(1,3,figsize=(15,5)) #make 3 subplots for target=1, target=0, and complete sample
sns.distplot(w_1,ax=a[0], color='purple')
a[0].set_title('target = 1')
sns.distplot(w_0,ax=a[1], color='blue')
a[1].set_title('target = 0')
sns.distplot(df_train['words'],ax=a[2], color='green')
a[2].set_title('target = 0 and target = 1')
# make a new variable for number of characters
df_train['wlen']=df_train['text'].apply(lambda x: sum([len(a) for a in str(x).split()])/len(str(x).split()))
#split by space to turn tweet into words, use list comprehension to get total char length, divide by word list length

# split dataset by target
wl_1=df_train[df_train['target']==0]['wlen']
wl_0=df_train[df_train['target']==1]['wlen']

# t test
wllengtht, wllengthp=scipy.stats.ttest_ind(wl_1, wl_0)
print('T Test number of words by target t={:.2f},p={:.2f}'.format(wllengtht, wllengthp))

#histograms
fig,a =  plt.subplots(1,3,figsize=(15,5)) #make 3 subplots for target=1, target=0, and complete sample
sns.distplot(wl_1,ax=a[0], color='purple')
a[0].set_title('target = 1')
sns.distplot(wl_0,ax=a[1], color='blue')
a[1].set_title('target = 0')
sns.distplot(df_train['wlen'],ax=a[2], color='green')
a[2].set_title('target = 0 and target = 1')
# putting all texts across rows together as a big string variable 
alltextdisaster=' '.join(set([text for text in df_train[df_train['target']==1]['text']])) # disaster
alltextnondisaster=' '.join(set([text for text in df_train[df_train['target']==0]['text']])) # non disaster

# build word clouds 
wc1 = WordCloud(background_color="white", max_words=200, width=1000, height=800).generate(alltextdisaster)
wc2 = WordCloud(background_color="white", max_words=200, width=1000, height=800).generate(alltextnondisaster)

# plotting word clouds
fig,a =  plt.subplots(1,2,figsize=(20,10))
a[0].imshow(wc1, interpolation='bilinear')
a[0].axis("off")
a[0].set_title('disaster tweet word cloud')

a[1].imshow(wc2, interpolation='bilinear')
a[1].axis("off")
a[1].set_title('nondisaster tweet word cloud')
plt.show()
def removePunctuation(text):
    return "".join([c for c in text if c not in string.punctuation])
print('remove punctuation:', removePunctuation("It's me!!!! :/"))

def removeNumber(text):
    return "".join([c for c in text if not c.isdigit()])
print('remove numbers:', removeNumber("123 abc"))

def removeHTML(text):
    return re.sub(r'<.*?>','', text) # match <tag> minimally
print('remove HTML tags:', removeHTML("<h1>heading</h1><p attribute=''>tag"))

def removeURL(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text) # match url patterns
print('remove url:', removeURL("url https://www.kaggle.com kaggle"))

def removeEmoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE) # compiling all emojis as a reg ex expression
    return emoji_pattern.sub(r'', text)
print('remove emoji:', removeEmoji('Sad😔'))

def lowerCase(text): 
    return text.lower()
print('lower case:', lowerCase('crazy NoiSy Town!'))

def removeStopwords(text):
    return ' '.join([word for word in text.split() if word not in stopwords.words('english')])
print('remove stop words:', removeStopwords('I am a cup of tea'))


Pstemmer=PorterStemmer()
def stemText(text):
    return ' '.join([Pstemmer.stem(token) for token in text.split()])
print('stem Text:', stemText('Word clouds are visualizations of words in which the sizes of words reflect the relative importance of words'))


# put all the above cleaning functions into one function
def cleanTextData(text):
    text=lowerCase(text)
    text=removePunctuation(text)
    text=removeURL(text)
    text=removeEmoji(text)
    text=removeNumber(text)
    text=removeHTML(text)
    text=removeStopwords(text)
    text=stemText(text)
    return text
print('clean:', cleanTextData('Word clouds are visualizations of words in which the sizes of words reflect the relative importance of words <a>link https://www.kaggle.com<a/> ttps://www.kaggle.com 321😔'))

#clean train and test
df_train['cleaned_text']=df_train['text'].apply(lambda x: cleanTextData(x))
df_test['cleaned_text']=df_test['text'].apply(lambda x: cleanTextData(x))
df_train.head(10)
train_vectors=TfidfVectorizer().fit_transform(df_train['cleaned_text'])
test_vectors=TfidfVectorizer().fit_transform(df_test['cleaned_text'])
y=df_train['target']
X=train_vectors
#Multinomial NB
multinomialnb_classifier = MultinomialNB()
print('cv f1 scores:',cross_val_score(multinomialnb_classifier,X, y,scoring='f1', cv=10)) # 10 folds cross validation
# Confusion Matrix Visualization
mnb_pred=cross_val_predict(multinomialnb_classifier, X, y,cv=10)
multinomialnb_classifier_cm=confusion_matrix(mnb_pred,y)
print('correct 0: {}, correct 1: {}, incorrect: {}'.format(multinomialnb_classifier_cm[0][0],multinomialnb_classifier_cm[1][1],multinomialnb_classifier_cm[1][0]+multinomialnb_classifier_cm[0][1]))
sns.heatmap(multinomialnb_classifier_cm, cmap='PuBu')

#GaussianNB
gnb_classifier = GaussianNB()
X_gnb=X.toarray() #converting X to dense, required by GaussianNB
print('cv f1 scores:',cross_val_score(gnb_classifier,X_gnb, y,scoring='f1', cv=10))
# Confusion Matrix Visualization
gnb_pred=cross_val_predict(gnb_classifier, X_gnb, y,cv=10)
gnb_classifier_cm=confusion_matrix(gnb_pred,y)
print('correct 0: {}, correct 1: {}, incorrect: {}'.format(gnb_classifier_cm[0][0],gnb_classifier_cm[1][1],gnb_classifier_cm[1][0]+gnb_classifier_cm[0][1]))
sns.heatmap(gnb_classifier_cm, cmap='PuBu')

logisticreg_classifier = LogisticRegression()
print('cv f1 scores:',cross_val_score(logisticreg_classifier,X, y,scoring='f1', cv=10))
# Confusion Matrix Visualization
lr_pred=cross_val_predict(logisticreg_classifier, X, y,cv=10)
logisticreg_classifier_cm=confusion_matrix(lr_pred,y)
print('correct 0: {}, correct 1: {}, incorrect: {}'.format(logisticreg_classifier_cm[0][0],logisticreg_classifier_cm[1][1],logisticreg_classifier_cm[1][0]+logisticreg_classifier_cm[0][1]))
sns.heatmap(logisticreg_classifier_cm, cmap='PuBu')

# multinomialnb_classifier.fit(X_train, y_train)
# multinomialnb_classifier_pred = multinomialnb_classifier.predict(X_test)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train['cleaned_text'].values)
X = tokenizer.texts_to_sequences(df_train['cleaned_text'].values)
X = pad_sequences(X)


model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 128 ,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(256, return_sequences=True))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test,y_test))