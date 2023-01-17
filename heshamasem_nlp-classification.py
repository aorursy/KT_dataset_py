import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid")

from nltk.corpus import stopwords

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

import collections

import spacy

nlp = spacy.load('en_core_web_sm')

ResponseData = pd.read_csv("../input/deepnlp/Sheet_1.csv",encoding='latin-1')

ResumeData = pd.read_csv("../input/deepnlp/Sheet_2.csv",encoding='latin-1')
ResponseData.head()
ResponseData.drop(['response_id','Unnamed: 3', 'Unnamed: 4','Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'],axis=1, inplace=True)

ResponseData.head()
ResponseData.shape
ResponseData.info()
sns.countplot(x='class', data=ResponseData ,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))
def cloud(text):

    plt.figure(figsize=(15,15))

    plt.imshow(WordCloud(background_color="white",stopwords=set(stopwords.words('english')))

               .generate(" ".join([i for i in text.str.lower()])))

    plt.axis("off")

    plt.title("Response could words")
cloud(ResponseData[ResponseData['class']=='flagged']['response_text'])
cloud(ResponseData[ResponseData['class']=='not_flagged']['response_text'])
def CommonWords(text , kk=10) : 



    all_words = []



    for i in range(text.shape[0]) : 

        this_phrase = list(text)[i]

        for word in this_phrase.split() : 

            all_words.append(word)



    print(f'Total words are {len(all_words)} words')   

    print('')



    common_words = collections.Counter(all_words).most_common()

    k=0

    word_list =[]

    for word, i in common_words : 

        if not word.lower() in  nlp.Defaults.stop_words :

            print(f'The word is   {word}   repeated   {i}  times')

            word_list.append(word)

            k+=1

        if k==kk : 

            break

            

    return word_list
words1 = CommonWords(ResponseData[ResponseData['class']=='not_flagged']['response_text'],5)
words2 = CommonWords(ResponseData[ResponseData['class']=='flagged']['response_text'],5)
filtered_words = words1+words2

filtered_words
def RemoveWords(data , feature , new_feature, words_list ) : 

    new_column = []

    for i in range(data.shape[0]) : 

        this_phrase = data[feature][i]

        new_phrase = []

        for word in this_phrase.split() : 

            if not word.lower() in words_list : 

                new_phrase.append(word)

        new_column.append(' '.join(new_phrase))

    

    data.insert(data.shape[1],new_feature,new_column)
RemoveWords(ResponseData , 'response_text' , 'filtered_text' , filtered_words)
ResponseData.head()
cloud(ResponseData[ResponseData['class']=='flagged']['filtered_text'])
cloud(ResponseData[ResponseData['class']=='not_flagged']['filtered_text'])
enc  = LabelEncoder()

enc.fit(ResponseData['class'])

ResponseData['class'] = enc.transform(ResponseData['class'])
ResponseData.head()
X = ResponseData['filtered_text']

y = ResponseData['class']
X.shape
y.shape
VecModel = TfidfVectorizer()

X = VecModel.fit_transform(X)



print(f'The new shape for X is {X.shape}')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=402)
print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='gini',max_depth=10,random_state=33) 

DecisionTreeClassifierModel.fit(X_train, y_train)
print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, y_train))

print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, y_test))

print('DecisionTreeClassifierModel Classes are : ' , DecisionTreeClassifierModel.classes_)
y_pred = DecisionTreeClassifierModel.predict(X_test)

y_pred_prob = DecisionTreeClassifierModel.predict_proba(X_test)

print('Predicted Value for DecisionTreeClassifierModel is : ' , y_pred[:10])

print('Prediction Probabilities Value for DecisionTreeClassifierModel is : ' , y_pred_prob[:10])
phrase = ['I went to my friend to talk about normal issues']

enc.inverse_transform(DecisionTreeClassifierModel.predict(VecModel.transform(phrase)))
phrase = ['I know a Friend was thinking about suicide']

enc.inverse_transform(DecisionTreeClassifierModel.predict(VecModel.transform(phrase)))
ResumeData.head()
ResumeData.shape
ResumeData.info()
sns.countplot(x='class', data=ResumeData ,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))
cloud(ResumeData[ResumeData['class']=='flagged']['resume_text'])
cloud(ResumeData[ResumeData['class']=='not_flagged']['resume_text'])
words1 = CommonWords(ResumeData[ResumeData['class']=='flagged']['resume_text'],10)
words2 = CommonWords(ResumeData[ResumeData['class']=='not_flagged']['resume_text'],10)
filtered_words = words1+words2

filtered_words
RemoveWords(ResumeData , 'resume_text' , 'filtered_text' , filtered_words)

ResumeData.head()
cloud(ResumeData[ResumeData['class']=='flagged']['filtered_text'])
cloud(ResumeData[ResumeData['class']=='not_flagged']['filtered_text'])
enc.fit(ResumeData['class'])

ResumeData['class'] = enc.transform(ResumeData['class'])

ResumeData.head()
X = ResumeData['filtered_text']

y = ResumeData['class']
X.shape
y.shape
X = VecModel.fit_transform(X)



print(f'The new shape for X is {X.shape}')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=102)

print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
SVCModel = SVC(kernel= 'linear',# it can be also linear,poly,sigmoid,precomputed

               max_iter=10000,C=10,gamma='auto')

SVCModel.fit(X_train, y_train)
print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))

print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))
y_pred = SVCModel.predict(X_test)

print('Predicted Value for SVCModel is : ' , y_pred[:10])