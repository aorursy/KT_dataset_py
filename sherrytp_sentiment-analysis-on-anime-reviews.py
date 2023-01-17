import os 

import sys 

import re



import scipy

import numpy as np

import pandas as pd

import jieba.analyse

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties



# import sklearn modules 

import sklearn.metrics as skm

import sklearn.model_selection

import sklearn.preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix as skm_conf_mat

from collections import Counter

from collections import defaultdict
datas = pd.read_csv("../input/bilibilib_gongzuoxibao.csv", sep = ",")
colnames = datas.columns

print(colnames) # author, score, disliked, likes, liked, ctime, score.1, content, last_ex_index, cursor, date
datas.shape
datas.head()
datas['score'].value_counts()
x = list(sorted(datas['score'].unique()))

y = list(datas['score'].value_counts())[::-1]

plt.bar(x,y, color='orange')

plt.xlabel('Score')

plt.ylabel('')

plt.title('Rating Frequencies')

plt.show()
#%% Content Analysis 

texts = ';'.join(datas['content'].tolist())

cut_text = " ".join(jieba.cut(texts))

# TF_IDF

keywords = jieba.analyse.extract_tags(cut_text, topK=100, withWeight=True, allowPOS=('a','e','n','nr','ns'))

text_cloud = dict(keywords)

###pd.DataFrame(keywords).to_excel('TF_IDF关键词前100.xlsx')
# Remove all punctuation and expression marks 

temp =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"

cut_text = re.sub(pattern = temp, repl = "", string = cut_text)
del datas['ctime']

del datas['cursor']

del datas['liked']

del datas['disliked']

del datas['likes']

del datas['last_ep_index']

pd.isnull(datas).astype(int).aggregate(sum, axis = 0)
perfect = datas[datas.score == 10]

imperfect = datas[datas.score != 10]

perfect_sample = perfect.sample(n = 1583, random_state = 1 )

new_data = pd.concat([perfect_sample, imperfect], axis = 0)



features = new_data['content']

labels = new_data['score']
rTrain, rTest, y_train, y_test = train_test_split(features, labels, test_size = 0.3, random_state=42)

# let's understand up a bit the data

## print out the shapes of  resultant feature data

print("\t\t\tFeature Shapes:")

print("Train set: \t\t{}".format(rTrain.shape), 

      #"\nValidation set: \t{}".format(rValidation.shape),

      "\nTest set: \t\t{}".format(rTest.shape))
texts = '\n'.join(rTrain.tolist())

#cut_text = jieba.lcut(texts)

cut_text = "".join(jieba.cut(texts))

cut_text = re.sub(pattern = temp, repl = "", string = cut_text)



keyword = jieba.analyse.extract_tags(cut_text, topK=100, allowPOS=('a','e','n','nr','ns'))  # list

cut_text = cut_text.split('\n')

keyword
cutlist = []



for i in range(0, len(cut_text)):

    cut_dic = defaultdict(int) 

    comment = cut_text[i]

    comment_cut = jieba.lcut(comment)

    for word in comment_cut: # word freq for every comment 

        if word in keyword:

            cut_dic[word] += 1  

    order = sorted(cut_dic.items(),key = lambda x:x[1],reverse = True) # word freq in descending order

    #print(order)

 

    myresult = "" 

    for j in range(0,len(order)): 

        result = order[j][0]+ "-" + str(order[j][1])

        myresult = myresult + " " + result  

    cutlist.append(myresult)

#print(cutlist)
word_freqs = []

for raw in cutlist:

    word_freq = {}

    for word_freq_raw in raw.split():

        index = word_freq_raw.find('-')

        word = word_freq_raw[:index]

        freq = int(word_freq_raw[index + 1])

        word_freq[word] = freq

    word_freqs.append(word_freq)

    

matrix = []

for word_freq in word_freqs:

    row = []

    for word in keyword:

        if word in word_freq:

            row.append(word_freq[word])

        else:

            row.append(0)

    matrix.append(row)

#print(matrix)

matrix = np.array(matrix)
grade1 = np.array([0.1

,0

,0

,0.7

,0.8

,0.1

,0

,0.3

,0

,0

,0

,0

,0.6

,0.1

,-1

,0

,0

,1

,0

,0

,0

,0.5

,-0.3

,-0.1

,0.8

,0

,0.4

,0

,0

,0

,0.6

,0.6

,0.8

,0

,0.6

,0.4

,0.6

,1

,0

,-0.7

,0

,0.9

,0

,-0.2

,0

,0

,0

,0

,0

,0.7

,0

,1

,0

,0

,0

,0

,-0.2

,0

,0

,0.6

,0.1

,0

,0.6

,0.3

,0

,0.7

,0.7

,0

,0

,0

,0

,0

,0

,0

,0

,0.4

,0

,0.6

,0

,1

,0.6

,0

,0

,1

,0.4

,0.2

,-1

,0.8

,-1

,0

,1

,0

,0.9

,0.7

,-0.3

,0

,0.2

,0

,0

,0])
X = np.array(matrix) * grade1
# import Logistic model

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV



clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y_train)

clf.score(X, y_train)
np.unique(clf.predict(X))
#Import Library of Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()



gnb.fit(X, y_train)

gnb.score(X,y_train)
np.unique(gnb.predict(X))
from sklearn.ensemble import RandomForestClassifier as RFClass

model_rf = RFClass(n_estimators = 100, max_depth=5, random_state=2019)

model_rf.fit(X, y_train)

model_rf.score(X, y_train)
np.unique(gnb.predict(X))
texts = '\n'.join(rTest.tolist())

#cut_text = jieba.lcut(texts)

cut_text = "".join(jieba.cut(texts))

cut_text = re.sub(pattern = temp, repl = "", string = cut_text)



keyword = jieba.analyse.extract_tags(cut_text, topK=100, allowPOS=('a','e','n','nr','ns'))  # list

cut_text = cut_text.split('\n')

keyword
cutlist = []



for i in range(0, len(cut_text)):

    cut_dic = defaultdict(int) 

    comment = cut_text[i]

    comment_cut = jieba.lcut(comment)

    for word in comment_cut: # word freq for every comment 

        if word in keyword:

            cut_dic[word] += 1  

    order = sorted(cut_dic.items(),key = lambda x:x[1],reverse = True) # word freq in descending order

    #print(order)

 

    myresult = "" 

    for j in range(0,len(order)): 

        result = order[j][0]+ "-" + str(order[j][1])

        myresult = myresult + " " + result  

    cutlist.append(myresult)

#print(cutlist)
word_freqs = []

for raw in cutlist:

    word_freq = {}

    for word_freq_raw in raw.split():

        index = word_freq_raw.find('-')

        word = word_freq_raw[:index]

        freq = int(word_freq_raw[index + 1])

        word_freq[word] = freq

    word_freqs.append(word_freq)

    

matrix = []

for word_freq in word_freqs:

    row = []

    for word in keyword:

        if word in word_freq:

            row.append(word_freq[word])

        else:

            row.append(0)

    matrix.append(row)

#print(matrix)

matrix = np.array(matrix)
grade2 = np.array([0.1

,0

,0

,0.7

,0.3

,0

,0

,0.8

,0.5

,0

,0.1

,0.1

,0

,0

,1

,-1

,0

,0

,0

,0.4

,0

,0.6

,0

,0.6

,0

,0

,1

,0

,0.8

,-0.1

,0

,0

,0.4

,0

,0

,0

,0.6

,0.6

,-0.4

,0

,0

,0

,0

,0

,0.4

,1

,-0.6

,0

,-0.7

,0.9

,-1

,0.4

,0.1

,-0.2

,-0.3

,0.6

,0

,0.2

,0

,0

,0

,0

,0.2

,0

,0.6

,0

,0.5

,-1

,0

,0

,0.9

,0

,0

,-0.6

,0.1

,0

,0.4

,-0.8

,0

,0

,-0.3

,0

,0.7

,0.5

,0

,0.8

,0

,0

,0

,0

,-0.2

,0.6

,0.5

,0.7

,0

,0

,0.8

,0.5

,0.7

,-0.4])
xTest = np.array(matrix) * grade2

xTest.shape
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)

    plt.show()



np.set_printoptions(precision = 2)
clf_proba = clf.predict_proba(xTest)   # predict probability 

clf_pred = clf.predict(xTest)   # prediction result

clf.score(xTest, y_test)
clf_cm = skm_conf_mat(y_test, clf_pred)

plot_confusion_matrix(clf_cm, classes = list(sorted(y_train.unique())), title = 'Confusion Matrix')
clfcv = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(X, y_train)

clfcv.score(X, y_train)
clfcv_proba = clfcv.predict_proba(xTest)

clfcv_pred = clfcv.predict(xTest)

clfcv.score(xTest, y_test)
clfcv_cm = skm_conf_mat(y_test, clf_pred)

plot_confusion_matrix(clfcv_cm, classes = list(sorted(datas['score'].unique())), title = 'Confusion Matrix')
rf_proba = model_rf.predict_proba(xTest)

rf_pred = model_rf.predict(xTest)

model_rf.score(xTest, y_test)
# Tree Plot

from graphviz import Source

from sklearn import tree as treemodule

Source(treemodule.export_graphviz(

        model_rf.estimators_[1]

        , out_file=None

        , filled = True

        , proportion = True #@@ try False and understand the differences

        )

)
rf_cm = skm_conf_mat(y_test, rf_pred)

plot_confusion_matrix(rf_cm, classes = list(sorted(datas['score'].unique())), title = 'Confusion Matrix')
rf_pred = pd.DataFrame(rf_pred)

rf_pred.to_csv("Predictions on Ratings.csv")
#score = (new_data.score == 2)|(new_data.score == 6)

new_data.loc[new_data.score == 6, 'score'] = 4

new_data.loc[new_data.score == 2, 'score'] = 4
features = new_data['content']

labels = new_data['score']



new_data['score'].value_counts()
rTrain, rTest, y_train, y_test = train_test_split(features, labels, test_size = 0.3, random_state=42)

# let's understand up a bit the data

## print out the shapes of  resultant feature data

print("\t\t\tFeature Shapes:")

print("Train set: \t\t{}".format(rTrain.shape), 

      #"\nValidation set: \t{}".format(rValidation.shape),

      "\nTest set: \t\t{}".format(rTest.shape))
texts = '\n'.join(rTrain.tolist())

#cut_text = jieba.lcut(texts)

cut_text = "".join(jieba.cut(texts))

cut_text = re.sub(pattern = temp, repl = "", string = cut_text)



keyword = jieba.analyse.extract_tags(cut_text, topK=100, allowPOS=('a','e','n','nr','ns'))  # list

cut_text = cut_text.split('\n')

keyword
cutlist = []



for i in range(0, len(cut_text)):

    cut_dic = defaultdict(int) 

    comment = cut_text[i]

    comment_cut = jieba.lcut(comment)

    for word in comment_cut: # word freq for every comment 

        if word in keyword:

            cut_dic[word] += 1  

    order = sorted(cut_dic.items(),key = lambda x:x[1],reverse = True) # word freq in descending order

    #print(order)

 

    myresult = "" 

    for j in range(0,len(order)): 

        result = order[j][0]+ "-" + str(order[j][1])

        myresult = myresult + " " + result  

    cutlist.append(myresult)

#print(cutlist)
word_freqs = []

for raw in cutlist:

    word_freq = {}

    for word_freq_raw in raw.split():

        index = word_freq_raw.find('-')

        word = word_freq_raw[:index]

        freq = int(word_freq_raw[index + 1])

        word_freq[word] = freq

    word_freqs.append(word_freq)

    

matrix = []

for word_freq in word_freqs:

    row = []

    for word in keyword:

        if word in word_freq:

            row.append(word_freq[word])

        else:

            row.append(0)

    matrix.append(row)

#print(matrix)

matrix = np.array(matrix)
X = np.array(matrix) * grade1
texts = '\n'.join(rTest.tolist())

#cut_text = jieba.lcut(texts)

cut_text = "".join(jieba.cut(texts))

cut_text = re.sub(pattern = temp, repl = "", string = cut_text)



keyword = jieba.analyse.extract_tags(cut_text, topK=100, allowPOS=('a','e','n','nr','ns'))  # list

cut_text = cut_text.split('\n')

keyword
cutlist = []



for i in range(0, len(cut_text)):

    cut_dic = defaultdict(int) 

    comment = cut_text[i]

    comment_cut = jieba.lcut(comment)

    for word in comment_cut: # word freq for every comment 

        if word in keyword:

            cut_dic[word] += 1  

    order = sorted(cut_dic.items(),key = lambda x:x[1],reverse = True) # word freq in descending order

    #print(order)

 

    myresult = "" 

    for j in range(0,len(order)): 

        result = order[j][0]+ "-" + str(order[j][1])

        myresult = myresult + " " + result  

    cutlist.append(myresult)

#print(cutlist)
word_freqs = []

for raw in cutlist:

    word_freq = {}

    for word_freq_raw in raw.split():

        index = word_freq_raw.find('-')

        word = word_freq_raw[:index]

        freq = int(word_freq_raw[index + 1])

        word_freq[word] = freq

    word_freqs.append(word_freq)

    

matrix = []

for word_freq in word_freqs:

    row = []

    for word in keyword:

        if word in word_freq:

            row.append(word_freq[word])

        else:

            row.append(0)

    matrix.append(row)

#print(matrix)

matrix = np.array(matrix)
xTest = np.array(matrix) * grade2

xTest.shape
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, y_train)

clf.score(xTest, y_test)
clfcv = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(X, y_train)

clfcv.score(xTest, y_test)
gnb.fit(X, y_train)

gnb.score(xTest, y_test)
model_rf.fit(X, y_train)

print(model_rf.score(X, y_train))

print(model_rf.score(xTest, y_test))