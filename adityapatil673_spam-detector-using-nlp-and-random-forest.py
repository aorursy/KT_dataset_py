import pandas as pd
import numpy as np

data = pd.read_csv("../input/spam.csv",encoding='latin-1')
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

data.columns =['label','body']
data.head()
pd.set_option('display.max_colwidth', 0) 
print("sample messages from human")
print(data[data['label']=='ham']['body'].head(15))
print("sample messages which are spam")
print(data[data['label']=='spam']['body'].head(15))

import string
import nltk
#ps = nltk.PorterStemmer()
stopwords= nltk.corpus.stopwords.words('english')

def clean(sentence):
    s = "".join(x for x in sentence if x not in string.punctuation)
    temp = s.lower().split(' ')
    temp2 = [x for x in temp if x not in stopwords]
    return temp2
clean("hell peOople  are hOOow ! AAare ! you. enough.. are")
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(analyzer=clean)
vector_output = vect.fit_transform(data['body'])

print(vect.get_feature_names()[0:100])
# these numbers are the columns

print (vector_output [0:10])
pd.DataFrame(vector_output.toarray())
import re
data['len'] = data['body'].apply(lambda x : len(x) - x.count(" "))
# METHOD 2  : data['len'] = data['body'].apply(lambda x : len(re.split('\s+',x)))
#print(data['body'][1]+"  - "+str(data['len'][1]))
test = "Hello people this is my contact 999999999 222 888888888 20000002222"
len(re.findall('\d{7,}',test))
# for finding numbers with digits 4,5,6,7 we will write \d{4,7}
# for finding numbers with digits 7,8,9,10 .... and many more . We will write \d{7,}
data['long_number'] = data['body'].apply(lambda x : len(re.findall('\d{7,}',x)))
data['short_number'] = data['body'].apply(lambda x : len(re.findall('\d{4,6}',x)))

#data[data['label']=='spam']
#a=data.iloc[8,1]
import string
def count_punct (text):
    count = sum([1 for x in text if x in string.punctuation])
    pp = round(100*count/(len(text)-text.count(" ")),3)
    return pp

data['punct'] = data['body'].apply(lambda x : count_punct(x))

testlink = "hello buddwwy http how com are you.co ww ww."

def  website (text):
    if (len(re.findall('www|http|com|\.co',text))>0):
        return 1
    else:
        return 0

#pd.set_option('display.max_colwidth', 0) 
#pd.DataFrame(data[data['label']=='spam']['body'])
print(website(testlink))
data['website'] = data['body'].apply(lambda x : website(x))
#pd.DataFrame(data[data['label']=='spam'])
# how do they look like ? 
#1 len
from matplotlib import pyplot
%matplotlib inline
pyplot.figure(figsize=(15,6))

bins = np.linspace(0,200,num=40)
pyplot.hist(data[data['label']=='spam']['len'],bins,alpha=0.5,label='spam',normed=True)
pyplot.hist(data[data['label']=='ham']['len'],bins,alpha =0.5,label ='ham', normed=True)
pyplot.legend(loc ='upper left')
pyplot.show()
# punctuation 
pyplot.figure(figsize=(15,6))
i=4
bins = np.linspace(0,40**(1/i),num=40)
pyplot.hist(data[data['label']=='spam']['punct']**(1/i),bins,normed=True,label ='spam',alpha=0.5)
pyplot.hist(data[data['label']=='ham']['punct']**(1/i),bins, normed = True, label='ham',alpha=0.5)
pyplot.show

#using box cox transformation to see if the data reveal distinction

# Numbers

pyplot.figure(figsize=(6,6))
pyplot.pie(data[data['label']=='spam']['long_number'].value_counts(),labels=['0','1','2','3'], 
           colors=['#5f675c','#197632','#6cdfdc','blue'],)
pyplot.title("Spam - long numbers")
pyplot.show()


pyplot.figure(figsize=(6,6))
pyplot.pie(data[data['label']=='ham']['long_number'].value_counts(),labels=['0','1'], 
           colors=['#5f675c','#197632'],)
pyplot.title("Ham - long numbers")
pyplot.show()


# short Numbers
green_pallete = ['#5f675c','#3db161','#66cdaa','#bee687','#6cdfdc','#d7d7ff','#ffdb00','white']

spam_x = data[data['label']=='spam']['short_number'].value_counts()
spam_x.sort_index(inplace=True)
pyplot.figure(figsize=(8,8))
pyplot.pie(spam_x,labels=spam_x.index,startangle=0,colors=green_pallete)
pyplot.title("Spam - short numbers")
pyplot.show()

ham_x = data[data['label']=='ham']['short_number'].value_counts()
ham_x.sort_index(inplace=True)
pyplot.figure(figsize=(8,8))
pyplot.pie(ham_x,labels=ham_x.index, colors=green_pallete)
pyplot.title("Ham - short numbers")
pyplot.show()
x_features = pd.concat([data['len'],data['long_number'],data['short_number'],data['punct'],data['website'],pd.DataFrame(vector_output.toarray())],axis=1)
#,pd.DataFrame(vector_output.toarray())
#,data['long_number'],data['short_number']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score

x_train, x_test, y_train, y_test = train_test_split(x_features,data['label'])
rf = RandomForestClassifier(n_estimators=100,max_depth=None,n_jobs=-1)
rf_model = rf.fit(x_train,y_train)
sorted(zip(rf_model.feature_importances_,x_train.columns),reverse=True)[0:20]
y_pred=rf_model.predict(x_test)
precision,recall,fscore,support =score(y_test,y_pred,pos_label='spam', average ='binary')
print('Precision : {} / Recall : {} / fscore : {} / Accuracy: {}'.format(round(precision,3),round(recall,3),round(fscore,3),round((y_pred==y_test).sum()/len(y_test),3)))
