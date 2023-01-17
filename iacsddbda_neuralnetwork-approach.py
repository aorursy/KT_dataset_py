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
import numpy as np

import pandas as pd

import seaborn as sns

import os

from imblearn.over_sampling import SMOTE





from sklearn.metrics import confusion_matrix

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from gensim.models import Word2Vec
train=pd.read_csv('/kaggle/input/donorschoose-application-screening/train.csv',nrows=70000)

test=pd.read_csv('/kaggle/input/donorschoose-application-screening/test.csv')

resources=pd.read_csv('/kaggle/input/donorschoose-application-screening/resources.csv')

submissions=pd.read_csv('/kaggle/input/donorschoose-application-screening/sample_submission.csv')
#removing three null values as i will be using teacher prefix

train=train.loc[train.iloc[:,3].notnull()]

print(train.shape)

train.drop(['project_essay_3','project_essay_4'],inplace=True,axis=1)

test.drop(['project_essay_3','project_essay_4'],inplace=True,axis=1)

print(test.shape)
train=pd.merge(train,resources.groupby('id').agg({'quantity':'sum','price':'sum'}),on='id',how='left')

test=pd.merge(test,resources.groupby('id').agg({'quantity':'sum','price':'sum'}),on='id',how='left')

train.dropna(inplace=True)

print("shape of train data:",train.shape)

print("*"*50)

print("shape of test data",test.shape)
pd.value_counts(train.project_is_approved)
y=train.project_is_approved

train.drop(['project_is_approved'],inplace=True,axis=1)
train.columns
x_train, x_test, y_train, y_test=train_test_split(train,y,test_size=0.2,stratify=y ,random_state=42)

del train

print(test.shape)

pd.value_counts(y_train)
#extracting only dates from datetime as datetime would lead to lots of unique data points

x_train.project_submitted_datetime=pd.to_datetime(x_train.project_submitted_datetime).dt.date

x_train.rename(columns={'project_submitted_datetime':'project_submitted_date'})

print("training done!")



x_test.project_submitted_datetime=pd.to_datetime(x_test.project_submitted_datetime).dt.date

x_test.rename(columns={'project_submitted_datetime':'project_submitted_date'})

print("testing done!")
x_train.shape
#subjectlist

def clean_categories(dataframe,column):

    my_list=list(column)

    sublist=[]

    for i in my_list:

        temp=""

        for j in i.split(','):

            #print(j)

            j=j.replace(" ",'')

            temp+=j.strip()+" "

            temp=temp.replace('&','_')

        sublist.append(temp)

    dataframe['clean_categories']=sublist

    dataframe.drop(['project_subject_categories'],inplace=True,axis=1)

    return sublist





ta=clean_categories(x_train,x_train.project_subject_categories)

my_subs=[j for i in ta for j in i.rstrip().split(" ")]

subject_dict=dict( Counter(my_subs))



clean_categories( x_test,x_test.project_subject_categories)

clean_categories( test,test.project_subject_categories)





print('done!')





from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(vocabulary=subject_dict.keys() ,lowercase=False, binary=True)

categories_one_hot_train = vectorizer.fit_transform(x_train.clean_categories.values)

categories_one_hot_test = vectorizer.transform(x_test.clean_categories.values)

categories_test_data=vectorizer.transform(test.clean_categories.values)

print(vectorizer.get_feature_names())

print("Shape of matrix after one hot encodig ",categories_one_hot_train.shape)

categories_one_hot_train.toarray()[1]

#subject subcategories

def clean_subcategories(dataframe,column):

    my_list=list(column)

    subj_sub_list=[]

    for i in my_list:

        temp=""

        for j in i.split(','):

            #print(j)

            j=j.replace(" ",'')

            temp+=j.strip()+" "

            temp=temp.replace('&','_')

        subj_sub_list.append(temp)

    dataframe['clean_subcategories']=subj_sub_list

    dataframe.drop(['project_subject_subcategories'],inplace=True,axis=1)

    return subj_sub_list



    



subj_sub_list=clean_subcategories(x_train,x_train.project_subject_subcategories)

my_subs=[j for i in subj_sub_list for j in i.rstrip().split(" ")]

subject_sub_dict=dict( Counter(my_subs))



clean_subcategories( x_test,x_test.project_subject_subcategories)

clean_subcategories( test,test.project_subject_subcategories)

print('done!')



from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(vocabulary=subject_sub_dict.keys() ,lowercase=False, binary=True)

subcategories_one_hot_train = vectorizer.fit_transform(x_train.clean_subcategories.values)

subcategories_one_hot_test = vectorizer.transform(x_test.clean_subcategories.values)

subcategories_test_data = vectorizer.transform(test.clean_subcategories.values)

print(vectorizer.get_feature_names())

x_train.iloc[1,3]
#encoding schoolstate for all three datasets

from sklearn.preprocessing import OneHotEncoder

encoder=OneHotEncoder(handle_unknown='ignore')

onehot_schoolstate_train=encoder.fit_transform(x_train.iloc[:,3].values.reshape(-1,1))

onehot_schoolstate_test=encoder.transform(x_test.iloc[:,3].values.reshape(-1,1))

onehot_schoolstate_actual_test=encoder.transform(test.iloc[:,3].values.reshape(-1,1))
#grade prefix sort of a label encoding

x_train['project_grade_encoded']=x_train.project_grade_category.map({'Grades PreK-2':0, 'Grades 6-8':2, 'Grades 3-5':1,'Grades 9-12':3})

x_test['project_grade_encoded']=x_test.project_grade_category.map({'Grades PreK-2':0, 'Grades 6-8':2, 'Grades 3-5':1,'Grades 9-12':3})

test['project_grade_encoded']=x_test.project_grade_category.map({'Grades PreK-2':0, 'Grades 6-8':2, 'Grades 3-5':1,'Grades 9-12':3})



#label_grade=encoder.transform(train.project_grade_category)



#label_grade.classes_
import re





def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase

# we are removing the words from the stop words list: 'no', 'nor', 'not'

stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"]



from tqdm import tqdm

def get_essay(train):

    train['essay']=train["project_essay_1"].map(str) + train["project_essay_2"].map(str) 

# tqdm is for printing the status bar

def preprocessing_essay(column):

    preprocessed_essays = []

    for sentance in tqdm(column.values):

        sent = decontracted(sentance)

        sent = sent.replace('\\r', ' ')

        sent = sent.replace('\\"', ' ')

        sent = sent.replace('\\n', ' ')

        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)

        preprocessed_essays.append(sent.lower().strip())

    return preprocessed_essays

get_essay(x_train)

preprocessed_essays_train=preprocessing_essay(x_train.essay)



get_essay(x_test)

preprocessed_essays_test=preprocessing_essay(x_test.essay)



get_essay(test)

preprocessed_essays_actual_test=preprocessing_essay(test.essay)

 



print('done!')

from sklearn.feature_extraction.text import TfidfVectorizer

#vectorizer = TfidfVectorizer(max_features=5000)

#text_tfidf_train = vectorizer.fit_transform(preprocessed_essays_train)

#text_tfidf_test = vectorizer.transform(preprocessed_essays_test)

#print("Shape of matrix after one hot encodig ",text_tfidf_cv.shape)   
corpus=[]

for i in tqdm(preprocessed_essays_train):

    corpus.append(i.split())

corpus



import gensim



w2v_essay=Word2Vec(corpus,size=300,workers=3)

#w2v_essay=gensim.models.Word2Vec.load("w2v_essay")

w2v_essay.save("w2v_essay")

del corpus
avg_w2v_vectors_essay_train = []; # the avg-w2v for each sentence/review is stored in this list

for sentence in tqdm(preprocessed_essays_train): # for each review/sentence

    vector = np.zeros(300) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sentence.split(): # for each word in a review/sentence

        if word in w2v_essay.wv.vocab.keys():

            vector += w2v_essay[word]

            cnt_words += 1

    if cnt_words != 0:

        vector /= cnt_words

    avg_w2v_vectors_essay_train.append(vector)

    



    

    #Average w2vec for test data

avg_w2v_vectors_essay_test = []; # the avg-w2v for each sentence/review is stored in this list

for sentence in tqdm(preprocessed_essays_test): # for each review/sentence

    vector = np.zeros(300) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sentence.split(): # for each word in a review/sentence

        if word in w2v_essay.wv.vocab.keys():

            vector += w2v_essay[word]

            cnt_words += 1

    if cnt_words != 0:

        vector /= cnt_words

    avg_w2v_vectors_essay_test.append(vector)

    

avg_w2v_vectors_essay_actual_test = []; # the avg-w2v for each sentence/review is stored in this list

for sentence in tqdm(preprocessed_essays_actual_test): # for each review/sentence

    vector = np.zeros(300) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sentence.split(): # for each word in a review/sentence

        if word in w2v_essay.wv.vocab.keys():

            vector += w2v_essay[word]

            cnt_words += 1

    if cnt_words != 0:

        vector /= cnt_words

    avg_w2v_vectors_essay_actual_test.append(vector)



#processing project title with TF_IDF

def get_title(column):

    preprocessed_title = []

    for sentance in tqdm(column.values):

        sent = decontracted(sentance)

        sent = sent.replace('\\r', ' ')

        sent = sent.replace('\\"', ' ')

        sent = sent.replace('\\n', ' ')

        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)

        preprocessed_title.append(sent.lower().strip())

    return preprocessed_title



preprocessed_title_train=get_title(x_train.project_title)



preprocessed_title_test=get_title(x_test.project_title)



preprocessed_title_actual_test=get_title(test.project_title)





#from sklearn.feature_extraction.text import TfidfVectorizer

#vectorizer = TfidfVectorizer(max_features=2000)

#title_tfidf_train = vectorizer.fit_transform(preprocessed_title_train)

#title_tfidf_test = vectorizer.transform(preprocessed_title_test)



corpus=[]

for i in preprocessed_title_train:

    corpus.append(i.split())

corpus





#w2v_title=gensim.models.Word2Vec.load("w2v_title")

w2v_title=Word2Vec(corpus,size=300,workers=3)

w2v_title.save("w2v_title")

del corpus
avg_w2v_vectors_title_train = []; # the avg-w2v for each sentence/review is stored in this list

for sentence in tqdm(preprocessed_title_train): # for each review/sentence

    vector = np.zeros(300) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sentence.split(): # for each word in a review/sentence

        if word in w2v_title.wv.vocab.keys():

            vector += w2v_title[word]

            cnt_words += 1

    if cnt_words != 0:

        vector /= cnt_words

    avg_w2v_vectors_title_train.append(vector)

    

    #Average w2vec for test data

avg_w2v_vectors_title_test = []; # the avg-w2v for each sentence/review is stored in this list

for sentence in tqdm(preprocessed_title_test): # for each review/sentence

    vector = np.zeros(300) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sentence.split(): # for each word in a review/sentence

        if word in w2v_title.wv.vocab.keys():

            vector += w2v_title[word]

            cnt_words += 1

    if cnt_words != 0:

        vector /= cnt_words

    avg_w2v_vectors_title_test.append(vector)



avg_w2v_vectors_title_actual_test = []; # the avg-w2v for each sentence/review is stored in this list

for sentence in tqdm(preprocessed_title_actual_test): # for each review/sentence

    vector = np.zeros(300) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sentence.split(): # for each word in a review/sentence

        if word in w2v_title.wv.vocab.keys():

            vector += w2v_title[word]

            cnt_words += 1

    if cnt_words != 0:

        vector /= cnt_words

    avg_w2v_vectors_title_actual_test.append(vector)



#scaling the numerical variables: price  

from sklearn.preprocessing import StandardScaler

price=StandardScaler()

price_train=price.fit_transform(x_train['price'].values.reshape(-1,1))

price_test=price.transform(x_test['price'].values.reshape(-1,1))

price_actual_test=price.transform(test['price'].values.reshape(-1,1))

print("done!")
#scaling the numerical variables: price  

from sklearn.preprocessing import StandardScaler

price=StandardScaler()

previously_posted_projects_train=price.fit_transform(x_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))

previously_posted_projects_test=price.transform(x_test['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))

previously_posted_projects_actual_test=price.transform(test['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))



print("done!")
#scaling the numerical variables: price  

from sklearn.preprocessing import StandardScaler

price=StandardScaler()

quantity_train=price.fit_transform(x_train['quantity'].values.reshape(-1,1))

quantity_test=price.transform(x_test['quantity'].values.reshape(-1,1))

quantity_actual_test=price.transform(test['quantity'].values.reshape(-1,1))



print("done!")
x_train.project_grade_encoded.shape

print(quantity_actual_test.shape)

print(previously_posted_projects_actual_test.shape)

print(price_actual_test.shape)

print(subcategories_test_data.shape)

print(categories_test_data.shape)

print(onehot_schoolstate_actual_test.shape)

#print(onehot_teacherprefix_actual_test.shape)

print(np.asarray( avg_w2v_vectors_essay_actual_test).shape)

print(np.asarray(avg_w2v_vectors_essay_actual_test).shape)

print(pd.DataFrame(test.project_grade_encoded).shape)
#since all the variables are done now it's time to stack them Average word2vec

from scipy.sparse import hstack

print(quantity_train.shape)

print(previously_posted_projects_train.shape)

print(price_train.shape)

print(subcategories_one_hot_train.shape)

print(categories_one_hot_train.shape)

print(onehot_schoolstate_train.shape)

#print(onehot_teacherprefix_train.shape)

print(np.asarray( avg_w2v_vectors_essay_train).shape)

print(np.asarray(avg_w2v_vectors_essay_train).shape)

print(pd.DataFrame(x_train.project_grade_encoded).shape)

x_tr_w2v=hstack((pd.DataFrame(x_train.project_grade_encoded),np.asarray(avg_w2v_vectors_essay_train),np.asarray(avg_w2v_vectors_title_train),quantity_train,previously_posted_projects_train,price_train,subcategories_one_hot_train,categories_one_hot_train,onehot_schoolstate_train)).tocsr()

x_te_w2v=hstack((pd.DataFrame(x_test.project_grade_encoded),np.asarray(avg_w2v_vectors_essay_test),np.asarray(avg_w2v_vectors_title_test),quantity_test,previously_posted_projects_test,price_test,subcategories_one_hot_test,categories_one_hot_test,onehot_schoolstate_test)).tocsr()

x_real_test=hstack((pd.DataFrame(test.project_grade_encoded),np.asarray(avg_w2v_vectors_essay_actual_test),np.asarray(avg_w2v_vectors_title_actual_test),quantity_actual_test,previously_posted_projects_actual_test,price_actual_test,subcategories_test_data,categories_test_data,onehot_schoolstate_actual_test)).tocsr()
print(x_tr_w2v.shape)

print(x_te_w2v.shape)

print(x_real_test.shape)
pd.value_counts(y_train)
y_train1=pd.DataFrame(y_train)
import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Activation,Dropout

from keras.layers.core import Dense

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy



n_inputs = x_tr_w2v.shape[1]



model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, )),

    Dropout(.5),



    Dense(256, activation='relu'),

    Dropout(.5),

   

    Dense(1, activation='sigmoid')

])

model.summary()

model.compile(Adam(lr=0.001),loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_tr_w2v, y_train,validation_split=0.1 ,batch_size=256,class_weight={0:3.2,1:.86}, epochs=3,verbose=2)
from sklearn.metrics import confusion_matrix

import numpy as np

#y_pred = regressor.predict(x_cv)

cv_pred=np.argmax(model.predict_proba(x_te_w2v),axis=1)

print("Train confusion matrix")

print(confusion_matrix(model.predict_classes(x_te_w2v),y_test))
from sklearn.metrics import classification_report

cv_pred=np.argmax(model.predict(x_te_w2v),axis=1)

print("Train confusion matrix")

print(classification_report(model.predict_classes(x_te_w2v),y_test))


pd.value_counts(y_test)

model.predict_proba(x_te_w2v)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver="sag", max_iter=200)

lr.fit(x_tr_w2v,y_train)
from sklearn.metrics import classification_report

cv_pred=np.argmax(model.predict_proba(x_te_w2v),axis=1)

print("Train confusion matrix")

print(classification_report(y_test,cv_pred))
preds=model.predict_proba(x_real_test)
preds

#pd.DataFrame(model.predict_proba(x_real_test)[:, 1],columns=["project_is_approved"],index=test.id)
preds
subm = pd.DataFrame()

subm['id'] = test.id

subm['project_is_approved'] = preds

subm.to_csv('submission.csv', index=False)
pd.value_counts(subm.project_is_approved)