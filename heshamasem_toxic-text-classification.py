import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid")

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from nltk.corpus import stopwords

from wordcloud import WordCloud

import collections

import spacy

nlp = spacy.load('en_core_web_sm')

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectPercentile , f_classif 

data = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv' )  

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv' )  
data = data[:10000]
def unique(feature) : 

    global data

    print(f'Number of unique vaure are {len(list(data[feature].unique()))} which are : \n {list(data[feature].unique())}')



def count_nulls() : 

    global data

    for col in data.columns : 

        if not data[col].isna().sum() == 0 : 

            print(f'Column   {col}    got   {data[col].isna().sum()} nulls  ,  Percentage : {round(100*data[col].isna().sum()/data.shape[0])} %')



def cplot(feature) : 

    global data

    sns.countplot(x=feature, data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))



def encoder(feature , new_feature, drop = True) : 

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





def SelectedData(data , feature , value , operation, selected_feature ):

    if operation==0 : 

        result = data[data[feature]==value][selected_feature]

    elif operation==1 : 

        result = data[data[feature] > value][selected_feature]

    elif operation==2 : 

        result = data[data[feature]< value][selected_feature]

    

    return result 







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

    



    

def CountWords(text) :  

    

    all_words = []



    for i in range(text.shape[0]) : 

        this_phrase = list(text)[i]

        for word in this_phrase.split() : 

            all_words.append(word)



    print(f'Total words are {len(all_words)} words')   

    print('')

    print(f'Total unique words are {len(set(all_words))} words')   



def SlicedData(feature_list, dropna = False) : 

    global data

    if dropna :

        return data.loc[:, feature_list ].dropna()

    else : 

        return data.loc[:, feature_list ]
data.shape
data.head()
data.describe()
SlicedData(['obscene','identity_attack', 'insult' , 'thread']).describe()
SlicedData(['asian','atheist', 'bisexual' , 'black']).describe()
data['target'].min() , data['target'].max()
data['target sector'] = round(data['target']*3)
unique('target sector')
cplot('target sector')
temp_data = data[data['target sector'] > 0]['target sector']
plt.figure(figsize=(10,10))

plt.pie(temp_data.value_counts(),labels=list(temp_data.value_counts().index),autopct ='%1.2f%%',labeldistance = 1.1)

plt.show()
count_nulls()
data['comments']  =  data['comment_text'].str.lower()
SlicedData(['comment_text' , 'comments']).head(20)
CountWords(data['comments'])
common = CommonWords(data['comments'])
RemoveWords(data , 'comments' , 'filtered comments', common)

SlicedData(['comments' , 'filtered comments']).head(20)
MakeCloud(data['filtered comments'])
def showclouds(n) : 

    this_list = ['asian', 'atheist', 'bisexual','black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',

                 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability','jewish', 'latino', 'male', 'muslim',

                 'other_disability','other_gender', 'other_race_or_ethnicity', 'other_religion','other_sexual_orientation', 

                 'physical_disability','psychiatric_or_mental_illness', 'transgender', 'white' ]



    for item in this_list[n*3:(n*3)+3] : 

        this_data =  SelectedData(data ,item , 0.1 , 1 , 'filtered comments')

        print(f'for item    {item}')

        print(f'Number of selected rows {this_data.shape[0]}')

        print('common words : ')

        _ = CommonWords(this_data)

        if this_data.shape[0] >0 : 

            MakeCloud(this_data , str(f'Word Cloud for {item}'), 8 ,8)

        print('--------------------------')
showclouds(0)
showclouds(1)
showclouds(2)
showclouds(3)
showclouds(4)
showclouds(5)
showclouds(6)
showclouds(7)
X = data['filtered comments']

y = data['target sector']
X.head(10)
y.head(10)
X.isnull().sum() , y.isnull().sum()
VecModel = TfidfVectorizer()

X = VecModel.fit_transform(X)

print(f'The new shape for X is {X.shape}')
FeatureSelection = SelectPercentile(score_func = f_classif, percentile=1)

X = FeatureSelection.fit_transform(X, y)
print('X Shape is ' , X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)



print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
GBCModel = GradientBoostingClassifier(n_estimators=500,max_depth=5,random_state=33) 

GBCModel.fit(X_train, y_train)





print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))

print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))

test.head()
test['comments']  =  test['comment_text'].str.lower()

test.head()
X_test = test['comments']
X_test.shape
X_test = VecModel.transform(X_test)
X_test.shape
X_test = FeatureSelection.transform(X_test)
X_test.shape
y_pred = GBCModel.predict(X_test)

y_pred_prob = GBCModel.predict_proba(X_test)

print('Predicted Value for GBCModel is : ' , y_pred[:10])

print('Prediction Probabilities Value for GBCModel is : ' , y_pred_prob[:10])