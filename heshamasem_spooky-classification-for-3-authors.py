import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid")

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

from wordcloud import WordCloud

import collections

import spacy

nlp = spacy.load('en_core_web_sm')

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectPercentile , chi2

from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv('../input/spooky-author-identification/train/train.csv')  

print(f'Data Shape is {data.shape}')

data.head()
def show_details() : 

    global data

    for col in data.columns : 

        print(f'for feature : {col}')

        print(f'Number of Nulls is   {data[col].isna().sum()}')

        print(f'Number of Unique values is   {len(data[col].unique())}')

        print(f'random Value {data[col][0]}')

        print(f'random Value {data[col][10]}')

        print(f'random Value {data[col][20]}')

        print('--------------------------')



def CountWords(text) :  

    

    all_words = []



    for i in range(text.shape[0]) : 

        this_phrase = list(text)[i]

        for word in this_phrase.split() : 

            all_words.append(word)



    print(f'Total words are {len(all_words)} words')   

    print('')

    print(f'Total unique words are {len(set(all_words))} words')   

    

def CommonWords(text ,show = True , kk=10) : 

    all_words = []



    for i in range(text.shape[0]) : 

        this_phrase = list(text)[i]

        for word in this_phrase.split() : 

            all_words.append(word)

    common_words = collections.Counter(all_words).most_common()

    k=0

    word_list =[]

    for word, i in common_words : 

        if not word.lower() in  nlp.Defaults.stop_words :

            if show : 

                print(f'The word is   {word}   repeated   {i}  times')

            word_list.append(word)

            k+=1

        if k==kk : 

            break

            

    return word_list



def SelectedData(feature , value , operation, selected_feature ):

    global data

    if operation==0 : 

        result = data[data[feature]==value][selected_feature]

    elif operation==1 : 

        result = data[data[feature] > value][selected_feature]

    elif operation==2 : 

        result = data[data[feature]< value][selected_feature]

    

    return result 



def LowerCase(feature , newfeature) : 

    global data

    def ApplyLower(text) : 

        return text.lower()

    data[newfeature] = data[feature].apply(ApplyLower)

    

def Drop(feature) :

    global data

    data.drop([feature],axis=1, inplace=True)

    data.head()

def Unique(feature) : 

    global data

    print(f'Number of unique vaure are {len(list(data[feature].unique()))} which are : \n {list(data[feature].unique())}')

    

def Encoder(feature , new_feature, drop = False) : 

    global data

    enc  = LabelEncoder()

    enc.fit(data[feature])

    data[new_feature] = enc.transform(data[feature])

    if drop == True : 

        data.drop([feature],axis=1, inplace=True)

        

def MakeCloud(text , title = 'Word Clouds' , w = 15 , h = 15):

    plt.figure(figsize=(w,h))

    plt.imshow(WordCloud(background_color="white",stopwords=set(stopwords.words('english')))

               .generate(" ".join([i for i in text.str.lower()])))

    plt.axis("off")

    plt.title(title)

    plt.show()

def BPlot(feature_1,feature_2) :

    global data

    sns.barplot(x=feature_1, y=feature_2 , data=data)

    

def CPlot(feature) : 

    global data

    sns.countplot(x=feature, data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))

Drop('id')
LowerCase('text' , 'lower text')

Drop('text')
data.head(20)
show_details()
Unique('author')
Encoder('author' , 'author code')

data.head()
CountWords(data['lower text'])
BPlot(data['author'].value_counts().index , data['author'].value_counts().values )
AllCommon = CommonWords(data['lower text'])
MakeCloud(data['lower text'] , 'All Words')
ECommon = CommonWords(SelectedData('author','EAP',0,'lower text'))
MakeCloud(SelectedData('author','EAP',0,'lower text') , 'EAP Words')
HCommon = CommonWords(SelectedData('author','HPL',0,'lower text'))
MakeCloud(SelectedData('author','HPL',0,'lower text') , 'HPL Words')
MCommon = CommonWords(SelectedData('author','MWS',0,'lower text'))
MakeCloud(SelectedData('author','MWS',0,'lower text') , 'MWS Words')
data['number of words'] = data['lower text'].apply(lambda x : len(x.split()))

print('mean words for EAP is  ' , SelectedData('author', 'EAP' , 0 , 'number of words').mean())  

print('mean words for HPL is  ' , SelectedData('author', 'HPL' , 0 , 'number of words').mean())  

print('mean words for MWS is  ' , SelectedData('author', 'MWS' , 0 , 'number of words').mean())  
data['number of chars'] = data['lower text'].apply(lambda x : len(x))

print('mean chars for EAP is  ' , SelectedData('author', 'EAP' , 0 , 'number of chars').mean())  

print('mean chars for HPL is  ' , SelectedData('author', 'HPL' , 0 , 'number of chars').mean())  

print('mean chars for MWS is  ' , SelectedData('author', 'MWS' , 0 , 'number of chars').mean())  
data['number of punctuations'] = data['lower text'].apply(lambda x : len([k for k in  x if k in r'.,;:!?|\#$%^&*/']))

print('mean punctuations for EAP is  ' , SelectedData('author', 'EAP' , 0 , 'number of punctuations').mean())  

print('mean punctuations for HPL is  ' , SelectedData('author', 'HPL' , 0 , 'number of punctuations').mean())  

print('mean punctuations for MWS is  ' , SelectedData('author', 'MWS' , 0 , 'number of punctuations').mean())  
data['number of stop'] = data['lower text'].apply(lambda x : len([k for k in  x if k in nlp.Defaults.stop_words ]))

print('mean stop for EAP is  ' , SelectedData('author', 'EAP' , 0 , 'number of stop').mean())  

print('mean stop for HPL is  ' , SelectedData('author', 'HPL' , 0 , 'number of stop').mean())  

print('mean stop for MWS is  ' , SelectedData('author', 'MWS' , 0 , 'number of stop').mean())  
data.head()
X = data['lower text']

y = data['author code']
VecModel = TfidfVectorizer()

XVec = VecModel.fit_transform(X)



print(f'The new shape for X is {XVec.shape}')
FeatureSelection = SelectPercentile(score_func = chi2, percentile=50)

X_data = FeatureSelection.fit_transform(XVec, y)



print('X Shape is ' , X_data.shape)
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.33, random_state=44, shuffle =True)



print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
MultinomialNBModel = MultinomialNB(alpha=0.05)

MultinomialNBModel.fit(X_train, y_train)



print('MultinomialNBModel Train Score is : ' , MultinomialNBModel.score(X_train, y_train))

print('MultinomialNBModel Test Score is : ' , MultinomialNBModel.score(X_test, y_test))
data = pd.read_csv('../input/spooky-author-identification/test/test.csv')  

print(f'Test data Shape is {data.shape}')

data.head()
LowerCase('text' , 'lower text')

Drop('text')

data.head()
X = data['lower text']
XVec = VecModel.transform(X)

print(f'The new shape for X is {XVec.shape}')
X_data = FeatureSelection.transform(XVec)

print('X Shape is ' , X_data.shape)
y_pred = MultinomialNBModel.predict(X_data)

y_pred_prob = MultinomialNBModel.predict_proba(X_data)

print('Predicted Value for MultinomialNBModel is : ' , y_pred[:10])

print('Prediction Probabilities Value for MultinomialNBModel is : ' , y_pred_prob[:10])
data = pd.read_csv('../input/spooky-author-identification/sample_submission/sample_submission.csv')  

print(f'Test data Shape is {data.shape}')

data.head()
idd = data['id']

FinalResults = pd.DataFrame(y_pred_prob  ,columns= ['EAP','HPL','MWS'])

FinalResults.insert(0,'id',idd)
FinalResults.head()
FinalResults.to_csv("sample_submission.csv",index=False)