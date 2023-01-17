import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import keras

from warnings import filterwarnings

filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print("Added shopping_cart.png for wordcloud")
data = pd.read_csv('/kaggle/input/multiclass-classification-data-for-turkish-tc32/ticaret-yorum.csv')

pd.set_option('max_colwidth', 500)

data.head(5)
data.describe()
data.info()
data.category.value_counts()
count = 0

cinemaximum4aydir = "Cinemaximum 4 Aydır Bilet Paralarını Bir Türlü İade Etmiyor,"

for text in data.text:

    if cinemaximum4aydir in text[:len(cinemaximum4aydir)]:

        count += 1

print(count)
exampleArray = np.array([[1,1],[1,2],[4,5]])

exampleFrame = pd.DataFrame(exampleArray,columns=["ex1","ex2"])

exampleFrame
ex1 = exampleFrame.drop_duplicates(subset="ex1",keep="first")

print("Without ignore_index")

print(ex1)

ex2 = exampleFrame.drop_duplicates(subset="ex1",keep="first",ignore_index=True)

print("With ignore_index")

print(ex2)
data.text.duplicated(keep="first").value_counts()
data.drop_duplicates(subset="text",keep="first",inplace=True,ignore_index=True)

data.describe()
import plotly.graph_objects as go

from plotly.offline import iplot, init_notebook_mode

import plotly.express as px

import plotly.io as pio



init_notebook_mode(True)



fig = px.bar(x=data.category.value_counts().index,y=data.category.value_counts(),color=data.category.value_counts().index,text=data.category.value_counts())

fig.update_traces(hovertemplate="Category:'%{x}' Counted: %{y}")

fig.update_layout(title={"text":"Category Counts","x":0.5,"font":{"size":35}},xaxis={"title":"Category","showgrid":False},yaxis={"title":"Value","showgrid":False},plot_bgcolor="white",width=800,height=500,showlegend=False)

iplot(fig)
fig1 = px.pie(data,values=data.category.value_counts(),names=data.category.value_counts().index)

fig1.update_traces(textposition='auto', textinfo='percent+label',marker={"line":{"width":1}},hoverinfo='label+percent',hole=0.4)

fig1.update_layout(annotations=[{"text":"Percentages","showarrow":False,"font_size":17}])

iplot(fig1)
import re



wordList = list()

for i in range(len(data)):

    temp = data.text[i].split()

    for k in temp:

        k = re.sub("[^a-zA-ZğĞüÜşŞıİöÖçÇ]","",k)

        if k != "":

            wordList.append(k)
from collections import Counter



wordCount = Counter(wordList)

countedWordDict = dict(wordCount)

sortedWordDict = sorted(countedWordDict.items(),key = lambda x : x[1],reverse=True)



print("Most Used 20 Words")

for word,counted in sortedWordDict[0:20]:

    print("{} : {}".format(word,counted))
for i in data["text"][7:10]:

    if "oku" in i:

        print(i)

        print("*"*20)
def dontReadMore(text):

    temptext = text.split(".")

    if "Devamını" in temptext[-1]:

        text = temptext[:-1]

    return "".join(text)



data["text"] = data["text"].apply(dontReadMore)
for i in data["text"][200:500]:

    if "oku" in i:

        print(i)

        print("*"*20)
wordList = list()

for i in range(len(data)):

    temp = data.text[i].split()

    for k in temp:

        k = re.sub("[^a-zA-ZğĞüÜşŞıİöÖçÇ]","",k)

        if k != "":

            wordList.append(k)

wordCount = Counter(wordList)

countedWordDict = dict(wordCount)

sortedWordDict = sorted(countedWordDict.items(),key = lambda x : x[1],reverse=True)

print("REAL Most Used 20 Words")

for word,counted in sortedWordDict[0:20]:

    print("{} : {}".format(word,counted))
num = 75 # For using most used 75 words

list1 = list()

list2 = list()

for i in range(num):

    list1.append(wordCount.most_common(num)[i][0])

    list2.append(wordCount.most_common(num)[i][1])
fig2 = px.bar(x=list1,y=list2,color=list2,hover_name=list1,hover_data={'Word':list1,"Count":list2})

fig2.update_traces(hovertemplate="Word:'%{x}' Value: %{y}")

fig2.update_layout(title={"text":"Word Values","x":0.5,"font":{"size":30}},xaxis={"title":"Words","showgrid":False},yaxis={"title":"Value","showgrid":False},plot_bgcolor="white")

fig2.show()
from PIL import Image



shopping_cart = np.array(Image.open("/kaggle/input/shopping-cart/shopping_cart.png"))

plt.imshow(shopping_cart)
from wordcloud import WordCloud

from nltk.corpus import stopwords



def grey_color_func(word, font_size, position,orientation,random_state=None, **kwargs):

    return("hsl(0,0%%, %d%%)" % np.random.randint(50,55))



stopwordCloud = set(stopwords.words("turkish"))



wordcloud = WordCloud(stopwords=stopwordCloud,max_words=1000,background_color="white",min_font_size=3,mask=shopping_cart).generate_from_frequencies(countedWordDict)

wordcloud.recolor(color_func = grey_color_func)

plt.figure(figsize=[13,10])

plt.axis("off")

plt.title("Word Cloud",fontsize=20)

plt.imshow(wordcloud)

plt.show()
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk import word_tokenize

import time



ps = PorterStemmer()

stopwordSet = set(stopwords.words('turkish'))



t = time.time()



def leadMyWord(text):

    text = re.sub('[^a-zA-ZğĞüÜşŞıİöÖçÇ]'," ",text)

    text = text.lower()

    text = word_tokenize(text,language='turkish')

    text = [word for word in text if not word in stopwordSet]

    text = " ".join(text)

    return text   



textList = data.text.apply(leadMyWord)

textList = list(textList)



print("Before")

print(data["text"][2])

print("After")

print(textList[2])

print("Time Passed")

print(time.time()-t)
#preparing y



from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical



le = LabelEncoder()

labelEncode = le.fit_transform(data["category"])

print("LabelEncode")

print(labelEncode)

categorical_y = to_categorical(labelEncode)

print("To_Categorical")

print(categorical_y)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split



#preparing x for ANN

tfidv = TfidfVectorizer(max_features=20001)

x = tfidv.fit_transform(textList)

x.sort_indices()



x_train,x_test,y_train,y_test = train_test_split(x,categorical_y,test_size=0.33,random_state=42)
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam

from keras.utils import plot_model

from keras.losses import categorical_crossentropy



def build_ann_model():

    model = Sequential()

    

    model.add(Dense(units=1024,activation="relu",input_dim=x_train.shape[1]))

    model.add(Dense(units=512,activation="relu"))

    model.add(Dense(units=256,activation="relu"))

    model.add(Dense(units=y_train.shape[1],activation="softmax"))

    

    optimizer = Adam(lr=0.000015,beta_1=0.9,beta_2=0.999)

    

    model.compile(optimizer=optimizer,metrics=["accuracy"],loss=categorical_crossentropy)

    return model
ann_model = build_ann_model()

plot_model(ann_model,show_shapes=True)
ann_history = ann_model.fit(x_train,y_train,epochs=10,batch_size=256,shuffle=True)

ypred = ann_model.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns



ann_accuracy = accuracy_score(y_test.argmax(axis=-1),ypred.argmax(axis=-1))

#print("ANN Accuracy:",ann_accuracy)

ann_cn = confusion_matrix(y_test.argmax(axis=-1),ypred.argmax(axis=-1))

plt.subplots(figsize=(18,14))

sns.heatmap(ann_cn,annot=True,fmt="1d",cbar=False,xticklabels=le.classes_,yticklabels=le.classes_)

plt.title("ANN Accuracy: {}".format(ann_accuracy),fontsize=50)

plt.xlabel("Predicted",fontsize=15)

plt.ylabel("Actual",fontsize=15)

plt.show()
fig3, axe1 = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

axe1[0].plot(ann_history.history["accuracy"],label="accuracy",color="blue")

axe1[1].plot(ann_history.history["loss"],label="loss",color="red")

axe1[0].title.set_text("ANN Accuracy")

axe1[1].title.set_text("ANN Loss")

axe1[0].set_xlabel("Epoch")

axe1[1].set_xlabel("Epoch")

axe1[0].set_ylabel("Rate")

plt.show()
from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences



#preparing x for CNN

MAX_FEATURES = 20001



onehot_corpus = []

for text in textList:

    onehot_corpus.append(one_hot(text,MAX_FEATURES))

    

maxTextLen = 0

for text in textList:

    word_token=word_tokenize(text)

    if(maxTextLen < len(word_token)):

        maxTextLen = len(word_token)

        

print("Max number of words : ",maxTextLen)



padded_corpus=pad_sequences(onehot_corpus,maxlen=maxTextLen,padding='post')

x_train2,x_test2,y_train2,y_test2 = train_test_split(padded_corpus,categorical_y,test_size=0.33,random_state=42)
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten

def build_cnn_model():

    model = Sequential()

    

    model.add(Embedding(MAX_FEATURES, 100, input_length=maxTextLen))





    model.add(Conv1D(64, 2, padding='same', activation='relu'))

    model.add(MaxPooling1D(2))

    #model.add(MaxPooling1D(2))

    

    model.add(Flatten())

    

    model.add(Dense(units=1024,activation="relu"))

    model.add(Dense(units=512,activation="relu"))

    

    model.add(Dense(units=y_train2.shape[1],activation="softmax"))

    

    optimizer = Adam(lr=0.000055,beta_1=0.9,beta_2=0.999)

    

    model.compile(optimizer=optimizer,metrics=["accuracy"],loss=categorical_crossentropy)

    return model
cnn_model = build_cnn_model()

plot_model(cnn_model,show_shapes=True)
cnn_history = cnn_model.fit(x_train2,y_train2,epochs=10,batch_size=1280,shuffle=True)

ypred2 = cnn_model.predict(x_test2)
cnn_accuracy = accuracy_score(y_test2.argmax(axis=-1),ypred2.argmax(axis=-1))

#print("CNN Accuracy:",cnn_accuracy)

cnn_cn = confusion_matrix(y_test2.argmax(axis=-1),ypred2.argmax(axis=-1))

plt.subplots(figsize=(18,14))

sns.heatmap(cnn_cn,annot=True,fmt="1d",cbar=False,xticklabels=le.classes_,yticklabels=le.classes_)

plt.title("CNN Accuracy: {}".format(cnn_accuracy),fontsize=50)

plt.xlabel("Predicted",fontsize=15)

plt.ylabel("Actual",fontsize=15)

plt.show()
fig3, axe1 = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

axe1[0].plot(cnn_history.history["accuracy"],label="accuracy",color="blue")

axe1[1].plot(cnn_history.history["loss"],label="loss",color="red")

axe1[0].title.set_text("CNN Accuracy")

axe1[1].title.set_text("CNN Loss")

axe1[0].set_xlabel("Epoch")

axe1[1].set_xlabel("Epoch")

axe1[0].set_ylabel("Rate")

plt.show()
def ann_predict(text):

    puretext = leadMyWord(text)

    vector = tfidv.transform([puretext])

    vector.sort_indices()

    predicted = ann_model.predict(vector)

    predicted_category = predicted.argmax(axis=1)

    return le.classes_[predicted_category]

def cnn_predict(text):

    puretext = leadMyWord(text)

    onehottext = one_hot(puretext,MAX_FEATURES)

    text_pad = pad_sequences([onehottext],maxlen=maxTextLen,padding='post')

    predicted = cnn_model.predict(text_pad)

    predicted_category = predicted.argmax(axis=1)

    return le.classes_[predicted_category]

    

for _ in range(10):

    randint = np.random.randint(len(data))

    text = data.text[randint]  

    print("  Text")

    print("-"*8)

    print(text)

    print("-"*20)

    print("Actual Category: {}".format(data.category[randint]))

    print("ANN Predicted Category: {}".format(ann_predict(text)[0]))

    print("CNN Predicted Category: {}".format(cnn_predict(text)[0]))

    print("*"*50)

    
#Let me try it too

def predict_print(text):

    print("  Text")

    print("-"*8)

    print(text)

    print("-"*20)

    print("ANN Predicted Category: {}".format(ann_predict(text)[0]))

    print("CNN Predicted Category: {}".format(cnn_predict(text)[0]))

    print("*"*50)

myText = "Yemeğin içinden kıl çıktı, gitmenizi önermiyorum." # hair came out of the dish, I don't suggest you go

predict_print(myText)

myText = "Tuş bozuk." # Key Broken

predict_print(myText)