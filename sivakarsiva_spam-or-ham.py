import matplotlib.pyplot as plt

import seaborn as sns

import re

import string

import nltk

nltk.download('stopwords')

import pandas as pd

stopwords= nltk.corpus.stopwords.words('english')

from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")



import numpy as np

from matplotlib import pyplot

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
msgs=pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding="ISO-8859-1",index_col=0)  
msgs
pd.set_option('display.max_colwidth', 0) 
msgs
msgs=msgs.drop(['Unnamed: 2', 'Unnamed: 3','Unnamed: 3','Unnamed: 4'], axis=1)


msgs.reset_index(level=0, inplace=True)

msgs.shape

msgs
msgs
msgs.columns =['LABEL','SMS']

msgs.head()
data=msgs.copy()
data.groupby('LABEL').describe()
plt.figure(figsize=(5, 5))

sns.set(style="darkgrid")

count_balance=pd.value_counts(data["LABEL"], sort= True)

sns.barplot(x=count_balance.index, y=count_balance)

plt.title('LABEL_COUNT')
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")



def CLEAN(message):

    lower_text=''.join([word.lower() for word in message]) # LOWERING

    tokens=re.split('\W+',lower_text) # TOKENIZING THE TEXT

    CLEANed_text =' '.join([stemmer.stem(x) for x in tokens if x not in stopwords]) # STEMMING AND REMOVING STOPWORDS    

    return CLEANed_text



def count_PUNCT (text):  # COUNTING PUNCTUATION

    #punt=["!","$","£","€", "free","winner"]

    count = sum([1 for x in text if x in string.punctuation])

    return count
data["CLEAN"]=data["SMS"].apply(lambda x: CLEAN(x))
data['LENGTH'] = data['CLEAN'].apply(lambda x : len(x))
### VISULIZING THE LENGTH OF THE PREPROCESSED TEXT COMPARED IN EACH OF THE CATEGORIES["HAM ,SPAM"]

pyplot.figure(figsize=(15,6))



bins = np.linspace(0,300,num=20)

sns.distplot(data[data['LABEL']=='spam']['LENGTH'],bins,label='spam')

sns.distplot(data[data['LABEL']=='ham']['LENGTH'],bins,label ='ham')

pyplot.legend(loc ='upper right')
data['NUMBERS']=data['SMS'].apply(lambda x : len(re.findall('\d{1,16}',x)))
### VISULIZING THE SERIE OF NUMBERS PRESENT IN EACH CATEGORY["HAM ,SPAM"]

pyplot.figure(figsize=(15,6))

bins = np.linspace(0,15,num=15)

sns.distplot(data[data['LABEL']=='spam']['NUMBERS'],bins,label='spam')

sns.distplot(data[data['LABEL']=='ham']['NUMBERS'],bins,label ='ham')

pyplot.legend(loc ='upper right')
data['CAPITAL_TEXT']=data['SMS'].apply(lambda x : len(re.findall('[A-Z$]+',x)))
# VISULIZING THE NUMBERS OF TIME A CAPITAL LETTERS APPEAR IN EACH CATEGORY IN A SINGLE MESSAGE["HAM ,SPAM"]

pyplot.figure(figsize=(15,6))

bins = np.linspace(0,40,num=20)

sns.distplot(data[data['LABEL']=='spam']['CAPITAL_TEXT'],bins,label='spam')

sns.distplot(data[data['LABEL']=='ham']['CAPITAL_TEXT'],bins,label ='ham')

pyplot.legend(loc ='upper right')
data['PUNCT'] = data['SMS'].apply(lambda x : count_PUNCT(x))
### VISULIZING THE PUNCTUATION APPEARING IN EACH CATEGORY IN A SINGLE MESSAGE["HAM ,SPAM"]

pyplot.figure(figsize=(15,6))



bins = np.linspace(0,20,num=20)

pyplot.hist(data[data['LABEL']=='spam']['PUNCT'],bins,alpha=0.5,label='spam',normed=True)

pyplot.hist(data[data['LABEL']=='ham']['PUNCT'],bins,alpha =0.5,label ='ham', normed=True)

pyplot.legend(loc ='upper right')
data
from collections import Counter

count1 = Counter(" ".join(data[data['LABEL']=='ham']["CLEAN"]).split()).most_common(20)

data1 = pd.DataFrame.from_dict(count1)

data1.columns=["ham_most_words","counts"]



count2 = Counter(" ".join(data[data['LABEL']=='spam']["CLEAN"]).split()).most_common(20)

data2 = pd.DataFrame.from_dict(count2)

data2.columns=["spam_most_words","counts"]
plt.figure(figsize=(12, 9))

sns.set(style="darkgrid")

sns.barplot(x=data1["ham_most_words"], y=data1["counts"])

plt.title('Ham_Most_Common_words')
plt.figure(figsize=(12, 9))

sns.barplot(x=data2["spam_most_words"], y=data2["counts"])

plt.title('Spam_Most_Common_words')
data.T
Extra_features = pd.concat([data['LENGTH'],data['NUMBERS'],data['PUNCT'],data['CAPITAL_TEXT']],axis=1)
Extra_features.head()
import pandas as pd

from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(Extra_features)

Extra_features_scaled = pd.DataFrame(x_scaled)
Extra_features_scaled.head(5)
from sklearn.feature_extraction.text import TfidfVectorizer



vect = TfidfVectorizer()

vector_output = vect.fit_transform(data['CLEAN'])

X_data=pd.DataFrame(vector_output.toarray())
# Combining vectorized filtered data["CLEAN"] and combined_features!!!! 

combined_features = pd.concat([Extra_features_scaled,X_data],axis=1)
combined_features.head()
#Importing Model libraries!!!

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score

from sklearn import feature_extraction, model_selection, naive_bayes, metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support as score
Svc = SVC(kernel='rbf', gamma=1.0)

Kn = KNeighborsClassifier(n_neighbors=49)

Dt = DecisionTreeClassifier(min_samples_split=7, random_state=111)

Lg = LogisticRegression(solver='liblinear', penalty='l1')

Rf = RandomForestClassifier(n_estimators=50,random_state=111)

Ad = AdaBoostClassifier(n_estimators=62, random_state=111)

Bc = BaggingClassifier(n_estimators=9, random_state=111)
X_train, X_test, y_train, y_test = train_test_split(X_data,data['LABEL'],test_size=0.3, random_state=111)
classifiers = {'Support_Vector' : Svc,

               'K-Neighbours' : Kn,

               'Decision_Trees': Dt,

               'Logistic_Regression': Lg,

               'Random_Forest': Rf,

               'AdaBoost': Ad,

               'Bagging_Classifier': Bc}
def train_model(model, feature_train, LABELs_train):    

    model.fit(feature_train, LABELs_train)
def model_prediction(model, features):

    return (model.predict(features))
f1_score_prediction1 = []



for k,v in classifiers.items():

    

    print("Classifier: {}".format(k))

    

    ### Training and Predicting 

    train_model(v, X_train, y_train)

    pred = model_prediction(v,X_test)

    

    ### Confusion Matrix 

    m_confusion_test = metrics.confusion_matrix(y_test, pred)

    print(m_confusion_test)

    

    ### Precision, recall, Fscore 

    precision,recall,fscore,support =score(y_test,pred,pos_label='spam', average ='binary')

    f1_score_prediction1.append((k, [fscore]))

    print('Precision : {} / Recall : {} / fscore : {}'.format(round(precision,3),round(recall,3),round(fscore,3)))

    print('\n')
df1 = pd.DataFrame.from_dict(dict(f1_score_prediction1,orient='index',columns=['Fscore']))

df1
X_train, X_test, y_train, y_test = train_test_split(combined_features,data['LABEL'], test_size=0.3, random_state=111)
f1_score_prediction2 = []



for k,v in classifiers.items():

    

    print("Classifier:{}".format(k))

    

    ### Training and Predicting LABELs ###

    train_model(v, X_train, y_train)

    pred = model_prediction(v,X_test)

    

    ### Confusion Matrix ###

    m_confusion_test = metrics.confusion_matrix(y_test, pred)

    print(m_confusion_test)

    

    ### Precision, recall, Fscore ###

    precision,recall,fscore,support =score(y_test,pred,pos_label='spam', average ='binary')

    f1_score_prediction2.append((k, [fscore]))

    print('Precision : {} / Recall : {} / fscore : {}'.format(round(precision,3),round(recall,3),round(fscore,3)))
df2 = pd.DataFrame.from_dict(dict(f1_score_prediction2,orient='index',columns=['Fscore2']))

df_final = pd.concat([df1,df2],axis=1)

df_final
df_final.plot(kind='bar', ylim=(0.55,1.0), figsize=(11,11), align='center')

#plt.xticks(df_final.index)

plt.title('Distribution by Classifiers')