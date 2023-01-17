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
# importing the Necssary modules
import matplotlib.pyplot as plt
import seaborn as sbn
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn import metrics
import warnings
import prettytable
from wordcloud import WordCloud
from sklearn.preprocessing import Normalizer

warnings.filterwarnings('ignore')
stemer=SnowballStemmer('english')
%%time
train_data=pd.read_csv("/kaggle/input/facebook-recruiting-iii-keyword-extraction/Train.zip",nrows=1000000)
# Initial Description about the data 
print("The Number of Question have in the dataset :",train_data.shape[0])
print("The Number of Columns have in the Dataset  :",train_data.shape[1])
print(train_data.columns)
# droping the Id Column from the Dataset
train_data.drop(columns=["Id"],axis=1,inplace=True)
%%time
# number of duplicate values in the dataset is :
print("The Number of Duplicate values in the Dataset are :",train_data.duplicated().sum())
# Droping the Duplicates in the dataset
train_data.drop_duplicates(inplace=True)
print("The Number of Remains after Droping the Duplicated values from the Dataset :",train_data.shape)
# Checking the Missing Values in the Dataset
print(train_data.isnull().sum())
# Here only contains single Null values so it better to remove that row instead repacing the null values.
train_data.dropna(inplace=True)
# The Shape of the Dataset after Removing the rows which Contains Null vales
train_data.shape
print("-"*50+"TITLE"+"-"*50)
print(train_data["Title"][30])
print("-"*50+"BODY"+"-"*50)
print(train_data["Body"][30])
print("-"*50+"TAGS"+"-"*50)
print(train_data["Tags"][30])
#Counting the Tags for each Question or Row in the Dataset
train_data["Count_tags"]=train_data.Tags.apply(lambda x:len(str(x).split())) # counting for each Query

# Distibution of Tags 
sbn.countplot(train_data.Count_tags)
plt.title("Number of Tags associated with the quarie")
plt.xlabel("Number of Tags")
plt.ylabel("Number of Quries")
# To get the frequent tags by the Countvectorizer
# GETTING THE Most Frequent Tags in the corpus.
tags_vector=CountVectorizer(tokenizer = lambda x: x.split(),binary=True)
x_tags=tags_vector.fit_transform(train_data.Tags)
count=x_tags.sum(axis=0)
freq_tags=pd.DataFrame()
freq_tags["tags"]=tags_vector.get_feature_names()
freq_tags["count"]=count.tolist()[0]
freq_tags.sort_values(by='count',ascending=False,inplace=True)
freq_tags.reset_index(inplace=True)
freq_tags.drop(columns="index",inplace=True)
print("----------------------The Top 10 Tags in the Corpus ---------------")
print(freq_tags["tags"][:10].values)
print("--------------------Top 10 Least Frequent tags in the Corpus--------------")
print(freq_tags.tail(10)["tags"].values)
# frequency plotting of the Tags 
sbn.set_style("whitegrid")
sbn.lineplot(data=freq_tags,x=freq_tags.index,y="count")
plt.title("Occurence of Tags in corpus",color="blue")
sbn.set_style("whitegrid")
sbn.lineplot(data=freq_tags.iloc[:1000],x=freq_tags.index[:1000],y="count")
plt.title("Occurence of Tags in corpus",color="blue")
# lets plot the Top 100 Tags in the Corpus
plt.figure(figsize=(10,7))
sbn.set_style("whitegrid")
sbn.lineplot(data=freq_tags.iloc[:100],x=freq_tags.index[:100],y="count")
a=[0,20,40,60,80,100]
sbn.scatterplot(a,freq_tags.iloc[a,1],hue=freq_tags.iloc[a,0])
plt.title("Occurence of Tags in corpus",color="blue")
# creating the Dictonary with the frequent words in the dataset
t=freq_tags.tags.to_list()
c=freq_tags["count"].to_list()
tags_dic={}
for i,j in zip(t,c):
    tags_dic[i]=j

# Create and generate a word cloud image:
wordcloud = WordCloud().generate_from_frequencies(tags_dic)
plt.figure(figsize=(8,8))
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
%%time
char_title_pre=train_data.Title.apply(lambda x:len(x))
char_body_pre=train_data.Body.apply(lambda x:len(x))
code_body_pre=train_data.Body.apply(lambda x:len(re.findall(r"<code>",x)))
words_title_pre=train_data.Title.apply(lambda x:len(str(x).split()))
# Defining Some Extra StopWords seems to be Not useful
extra=['could',"would","iis","sometimes","sometime","puts","put","get","gets","help","please","need",\
       "like","know","thank","thanks","madam","sir","hii","doubt","doubts","www","com"]
li=stopwords.words("english")
li=li+extra
def Preprocesser(doc):
    body=[]
    for text in doc:
        text=re.sub(r"href.*","",text)          #removing the href ie. removing the hyper links
        text=re.sub('<code>(.*?)</code>', '', text, flags=re.MULTILINE|re.DOTALL) # removing the code segments
        text=re.sub('<.*?>', ' ', str(text.encode('utf-8')))           #removing the Html Tags in the Text
        text=re.sub(r"[^a-zA-Z]+"," ",text)  ## removing numbers and most of Puncutuations in the Text.
        text=text.lower()                    ## converting from upper case to lower case
        body.append(" ".join([k for k in text.split() if((len(k)>2 or k=="c") and k not in li )]))
        
    return body    
%%time
pre_body=Preprocesser(train_data.Body)      ## Preprocessing the Body Columns
pre_text=Preprocesser(train_data.Title)     ## Preprocessing the Title Columns
## Replacing the a Title and Body with the Preprocessed Title and Body respectively.
train_data["Title"]=pre_text
train_data["Body"]=pre_body
pre_data=train_data    # Creating a Reference for the train_data_100k 
words_title_post=train_data.Title.apply(lambda x:len(str(x).split()))
words_body_post=train_data.Body.apply(lambda x:len(str(x).split()))
train_data["char_title_pre"]=(char_title_pre-min(char_title_pre))/(max(char_title_pre)-min(char_title_pre))
train_data["char_body_pre"]=(char_body_pre-min(char_body_pre))/(max(char_body_pre)-min(char_body_pre))
train_data["code_body_pre"]=(code_body_pre-min(code_body_pre))/(max(code_body_pre)-min(code_body_pre))
train_data["words_title_pre"]=(words_title_pre-min(words_title_pre))/(max(words_title_pre)-min(words_title_pre))
train_data["words_title_post"]=(words_title_post-min(words_title_post))/(max(words_title_post)-min(words_title_post))
train_data["words_body_post"]=(words_body_post-min(words_body_post))/(max(words_body_post)-min(words_body_post))
train_data.head()
## seperate the tag columns and the droping the Tags column from the dataset
y_tagss=train_data.Tags
train_data.drop(columns="Tags",axis=1,inplace=True)
## reference to the pre_data
pre_data=train_data 
%%time
# Converting the Tags columns in to the Mulit label Classification
# initializing the Count vectorizer
tag_vect=CountVectorizer(binary=True,tokenizer=lambda x:str(x).split(),max_features=500)
vec_tag=tag_vect.fit_transform(y_tagss)
## split the Training dataset and validataion dataset
x_train,x_val,y_train,y_val=train_test_split(pre_data.iloc[:500000,],vec_tag[:500000],test_size=0.2)
print("The shape of the Training Dataset :",x_train.shape,y_train.shape)
print("The shape of the validation Dataset :",x_val.shape,y_val.shape)
tit_bod_train=[i+" "+j for i,j in zip(x_train.Title,x_train.Body)] ## combining the both the title and Body
tit_bod_val=[i+" "+j for i,j in zip(x_val.Title,x_val.Body)]  # combining the title and Body for the Validation
%%time
feat_vec=TfidfVectorizer(tokenizer=lambda x:x.split(),max_features=100000,ngram_range=(1,1))
feat_vec.fit(tit_bod_train)
train_feat=feat_vec.transform(tit_bod_train)
val_feat=feat_vec.transform(tit_bod_val)
# Concatenate the Title_Body and Derived Features 
train_feat=hstack((train_feat,x_train.iloc[:,2:].values))
val_feat=hstack((val_feat,x_val.iloc[:,2:].values))
print("The Shape of the Training Dataset :",train_feat.shape)
print("The Shape of the Validation Dataset :",val_feat.shape)
%%time
### Using the Log Loss (Linear MOdels --> Logistic regression)

classifier = OneVsRestClassifier(SGDClassifier(penalty='l2',loss="log",alpha=0.000001), n_jobs=-1)
classifier.fit(train_feat, y_train)
val_pre = classifier.predict(val_feat)

print("accuracy :",metrics.accuracy_score(y_val,val_pre))
print("macro f1 score :",metrics.f1_score(y_val, val_pre, average = 'macro'))
print("micro f1 scoore :",metrics.f1_score(y_val, val_pre, average = 'micro'))


%%time
## Hinge Loss ---> Linear SVM Classifier 
classifier = OneVsRestClassifier(SGDClassifier(penalty='l2',loss="hinge",alpha=0.000001), n_jobs=-1)
classifier.fit(train_feat, y_train)
val_pre = classifier.predict(val_feat)

##VALIDATION ACCURACY
print("accuracy :",metrics.accuracy_score(y_val,val_pre))
print("macro f1 score :",metrics.f1_score(y_val, val_pre, average = 'macro'))
print("micro f1 scoore :",metrics.f1_score(y_val, val_pre, average = 'micro'))


## TRAINING ACCURACY
val_pre = classifier.predict(train_feat)

print("accuracy :",metrics.accuracy_score(y_train,val_pre))
print("macro f1 score :",metrics.f1_score(y_train, val_pre, average = 'macro'))
print("micro f1 scoore :",metrics.f1_score(y_train, val_pre, average = 'micro'))


%%time
## ‘modified_huber Loss’---> Problastic Models 
classifier = OneVsRestClassifier(SGDClassifier(penalty='l2',loss='modified_huber',alpha=0.0000001), n_jobs=-1)
classifier.fit(train_feat, y_train)
val_pre = classifier.predict(val_feat)

print("accuracy :",metrics.accuracy_score(y_val,val_pre))
print("macro f1 score :",metrics.f1_score(y_val, val_pre, average = 'macro'))
print("micro f1 scoore :",metrics.f1_score(y_val, val_pre, average = 'micro'))

