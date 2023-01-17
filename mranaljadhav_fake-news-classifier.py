from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
df = pd.read_csv("../input/fake-news/train.csv")
df.head()
df.info()
print('Number of Rows : ', df.shape[0])
print('Number of Columns : ', df.shape[1])
df.isna().sum()
df = df.dropna()
fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(17, 5), dpi=100)
length=df[df["label"]==1]['title'].str.len()
ax1.hist(length,bins = 20,color='skyblue')
ax1.set_title('Fake News')
length=df[df["label"]==0]['title'].str.len()
ax2.hist(length, bins = 20)
ax2.set_title('Real News')
fig.suptitle('Characters in title')
plt.show()
fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(17, 5), dpi=100)
length=df[df["label"]==1]['text'].str.len()
ax1.hist(length,bins = 20,color='skyblue')
ax1.set_title('Fake News')
length=df[df["label"]==0]['text'].str.len()
ax2.hist(length, bins = 20)
ax2.set_title('Real News')
fig.suptitle('Characters in text')
plt.show()
fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(17, 5), dpi=100)
num=df[df["label"]==1]['title'].str.split().map(lambda x: len(x))
ax1.hist(num,bins = 20,color='skyblue')
ax1.set_title('Fake News')
num=df[df["label"]==0]['title'].str.split().map(lambda x: len(x))
ax2.hist(num,bins = 20)
ax2.set_title('Real News')
fig.suptitle('Words in title')
plt.show()
fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(17, 5), dpi=100)
num=df[df["label"]==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(num,bins = 20,color='skyblue')
ax1.set_title('Fake News')
num=df[df["label"]==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(num,bins = 20)
ax2.set_title('Real News')
fig.suptitle('Words in text')
plt.show()
print('Number of 0 (Not Fake) : ', df["label"].value_counts()[0])
print('Number of 1 (Fake) : ', df["label"].value_counts()[1])
label = df["label"].value_counts()
sns.barplot(label.index, label, color="salmon")
plt.title('Target Count', fontsize=14)
data = df.copy()
# here we reset the index because we had drop NaN values from df
data.reset_index(inplace=True)
X = data['text']# Independent Variable
y = data['label'] # Target or Dependent Variable
# Dataset Preprocessing
lemmatizer = WordNetLemmatizer()
def text_cleaning(text):
    text = re.sub("[^a-zA-Z]", " ", text) # removing punctuation
    text = text.lower() # text to lowercase
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if not word in stopwords.words('english')]
    return ' '.join(text)    
data['text'] = data['text'].apply(text_cleaning)
plt.figure(figsize = (20,20)) # Text that is not Fake(0)
wc = WordCloud(width = 1600 , height = 800 , max_words = 3000).generate(" ".join(data[data.label == 0].text))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20)) # Text that is Fake(1)
wc = WordCloud(width = 1600 , height = 800 , max_words = 3000).generate(" ".join(data[data.label == 1].text))
plt.imshow(wc , interpolation = 'bilinear')
# Vocabulary size
# here using one_hot method, here the word of sentence converted into particular index
voc_size=5000
oneHot = [ one_hot(word,voc_size) for word in data.text ]
# here we used padding because here sentences are not of equal length,
# so we used pad_sequence to make the sentence of equal length .
sent_length = 20
padding = pad_sequences(oneHot, padding='pre', maxlen=sent_length)
padding[0]
vector_feature = 40
seq = Sequential()
seq.add(Embedding(voc_size, vector_feature, input_length = sent_length))
seq.add(Dropout(0.3))
seq.add(LSTM(100))
seq.add(Dropout(0.3))
seq.add(Dense(1, activation = 'sigmoid'))
seq.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
seq.summary()
x_final = np.array(padding)
y_final = np.array(y)
print(x_final.shape)
print(y_final.shape)
# here we had split data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size = 0.2, random_state = 42)
seq.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=10,batch_size=64)
pred = seq.predict_classes(x_test)
cfm = confusion_matrix(pred, y_test)
plt.figure(figsize = (7,5))
sns.heatmap(cfm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')
plt.xlabel("Predicted")
plt.ylabel("Actual")
print(classification_report(pred, y_test))