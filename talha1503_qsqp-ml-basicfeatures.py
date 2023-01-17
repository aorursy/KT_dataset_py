import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, STOPWORDS 
import operator
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.decomposition import TruncatedSVD
data = pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip')
data.head()
print("Length of the dataset is {}".format(len(data)))
data.describe()
data.info()
positive_count = len(data[data['is_duplicate']==1])
negative_count = len(data[data['is_duplicate']==0])

print("total {} positive samples  ".format(positive_count))
print("total {} negative samples  ".format(negative_count))
fig = plt.figure(1,(10,10))
labels = ['Positive','Negative']
plt.pie([positive_count,negative_count],labels= labels)
plt.show()
data.drop(['id','qid1','qid2'],inplace=True,axis = 1)
data.tail()
data = data.sample(frac=1).reset_index(drop=True)
data.head()
train = data[:100000]
validation = data[100000:150000]
train.head()
def generate_word_cloud(data,max_words=100):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(max_words=max_words,stopwords=stopwords).generate(str(data))
    fig = plt.figure(1,(15,10))
    plt.axis("off")
    plt.imshow(wordcloud)
    plt.show()
generate_word_cloud(train['question1']+train['question2'],max_words=400)
def plot_length_feature(text,top_k_words = 10):
    length_dict = {}
    for i in range(len(text)):
        split_words = str(text[i]).split()
        for j in range(len(split_words)):
            if split_words[j] not in set(stopwords.words('english')):
                if not length_dict.get(len(split_words[j])):
                    length_dict[len(split_words[j])]=1
                else:
                    length_dict[len(split_words[j])]+=1
    
    length_dict = sorted(length_dict.items(),key=operator.itemgetter(1),reverse=True)
    lengths = [length for length,frequency in length_dict][:top_k_words]
    freq = [frequency for length,frequency in length_dict][:top_k_words]
    
    
    plt.figure(1,(9,7))
    plt.title("Length v/S Frequency for top {} lengths for words".format(top_k_words))
    plt.xlabel('Lengths')
    plt.ylabel('Frequency')
    plt.bar(lengths,freq,align='center', alpha=0.5)
    plt.show()
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = text.replace('“','')
    text = text.replace('”','')
    text = re.sub("hasn’t","has not",text)
    text = re.sub("can’t","can not",text)
    text = re.sub("wouldn’t","would not",text)
    text = re.sub("couldn’t","could not",text)
    text = re.sub("won’t","will not",text)
    text = re.sub("isn’t","is not",text)
    text = re.sub("i’ll","i will",text)
    text = re.sub("he’ll","he will",text)
    text = re.sub("she’ll","she will",text)
    text = re.sub("i’m","i am",text)
    text = re.sub("you’ll","you will",text)
    text = re.sub("hadn’t","had not",text)
    text = re.sub("don’t","do not",text)
    text = re.sub("here’s","here is",text)
    text = re.sub("where’s","where is",text)
    text = re.sub("that’s","that is",text)
    text = re.sub("it’s","it is",text)
    text = re.sub("he’s","he is",text)
    text = re.sub("she’s","she is",text)
    text = re.sub("what’s","what is",text)
    text = re.sub("i’ve","i have",text)
    text = re.sub("they’re","they are",text)
    text = re.sub("you’re","you are",text)
    text = re.sub("we’d","we would",text)
    text = re.sub("i’d","i would",text)
    text = re.sub(r'[^A-Za-z]',' ',text)
    text = text.split()
    text = [word for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text
def clean_data(data):
    for index,row in data.iterrows():
        data.at[index,'question1'] = clean_text(row['question1'])
        data.at[index,'question2'] = clean_text(row['question2'])
        if index%10000==0:
            print(index)
    return data 
train = clean_data(train)
validation = clean_data(validation)
train.head()
train['question_comb'] = train['question1'] +" "+train['question2']
validation['question_comb'] = validation['question1'] +" "+validation['question2']
train.head()
validation.head()
question_list = train['question_comb'].values
valid_question_list = validation['question_comb'].values

vectorizer = TfidfVectorizer(max_features = 10000,stop_words = set(stopwords.words('english')))
x_train = vectorizer.fit_transform(question_list)
x_test = vectorizer.transform(valid_question_list)
y_train = train['is_duplicate'] 
y_test = validation['is_duplicate'] 
logistic_model = LogisticRegression()
logistic_model.fit(x_train,y_train)
y_predicted = logistic_model.predict(x_test)
print(accuracy_score(y_test,y_predicted))
xgbclassifier = XGBClassifier()
xgbclassifier.fit(x_train,y_train)
y_predicted = xgbclassifier.predict(x_test)
print(accuracy_score(y_test,y_predicted))
svd = TruncatedSVD(n_components = 1000)
x_train_svd = svd.fit_transform(x_train)
x_valid_svd = svd.transform(x_test)
logistic_model = LogisticRegression()
logistic_model.fit(x_train_svd,y_train)
y_predicted = logistic_model.predict(x_valid_svd)
print(accuracy_score(y_test,y_predicted))
xgbclassifier = XGBClassifier()
xgbclassifier.fit(x_train_svd,y_train)
y_predicted = xgbclassifier.predict(x_valid_svd)
print(accuracy_score(y_test,y_predicted))