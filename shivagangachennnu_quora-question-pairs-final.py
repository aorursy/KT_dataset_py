#importing libraries numpy,pandas,mathplotlib for extracting, modifying and visualizing the data



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#loading input train file to train

train = pd.read_csv("../input/train.csv")

#loading iput test file to test

test = pd.read_csv("../input/test.csv")

#printing the top

train.head()
test.head()
train.info()
test.info()
train_duplicate_mean = train['is_duplicate'].mean()

print ("mean of train data is_duplicate column",train_duplicate_mean)
pt = train.groupby('is_duplicate')['id'].count()

pt.plot.bar()


question_id_1 = train['qid1'].tolist()

question_id_2 = train['qid2'].tolist()

question_id = pd.Series(question_id_1+question_id_2)

plt.figure(figsize=(15,6))

plt.hist(question_id.value_counts(), bins= 30)

plt.yscale('log', nonposy='clip')
from nltk.corpus import stopwords as st

stopwords_set = set(st.words("english"))



def word_dict(sentence):

    question_words_dict = {}

    for word in sentence.lower().split():

        if word not in stopwords_set:

            question_words_dict[word] = 1

    return question_words_dict

def common_words_percentage(entry):

    question_1_words = word_dict(str(entry['question1']))

    question_2_words = word_dict(str(entry['question2']))

     

    if len(question_1_words) == 0 or len(question_2_words) == 0:

        return 0

    shared_in_q1 = [word for word in question_1_words.keys() if word in question_2_words]

    feature_Ratio = ( 2*len(shared_in_q1) )/(len(question_1_words)+len(question_2_words))

    return feature_Ratio
def tfidf_weights(entry):

    question_1_words = word_dict(str(entry['question1']))

    question_2_words = word_dict(str(entry['question2']))

    if len(question_1_words) == 0 or len(question_2_words) == 0:

        return 0

    

    common_wts_1 = [weights.get(w, 0) for w in question_1_words.keys() if w in question_2_words]  

    common_wts_2 = [weights.get(w, 0) for w in question_2_words.keys() if w in question_2_words]

    common_wts = common_wts_1 + common_wts_2

    whole_wts = [weights.get(w, 0) for w in question_1_words] + [weights.get(w, 0) for w in question_2_words]

    

    feature_tfidf = np.sum(common_wts) / np.sum(whole_wts)

    return feature_tfidf
list_of_questions = (train['question1'].str.lower().astype('U').tolist() + train['question2'].str.lower().astype('U').tolist())



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df = 50,max_features = 3000000,ngram_range = (1,10))

X = vectorizer.fit_transform(list_of_questions)

idf = vectorizer.idf_

weights = (dict(zip(vectorizer.get_feature_names(), idf)))
train = train.dropna(axis=0, how='any')

test = test.dropna(axis=0, how='any')

X_TrainData = pd.DataFrame()

X_TestData = pd.DataFrame()

X_TrainData['common_word_percent'] = train.apply(common_words_percentage, axis=1, raw=True)

X_TrainData['feature_ifidf'] = train.apply(tfidf_weights, axis = 1, raw = True)

Y_TrainData = train['is_duplicate'].values

X_TestData['common_word_percent'] = test.apply(common_words_percentage, axis = 1, raw = True)

X_TestData['feature_ifidf'] = test.apply(tfidf_weights, axis = 1, raw = True)
import nltk

def jaccard_similarity_coefficient(row):

    if (type(row['question1']) is str) and (type(row['question2']) is str):

        words_1 = row['question1'].lower().split()

        words_2 = row['question2'].lower().split()

    else:

        words_1 = nltk.word_tokenize(str(row['question1']))

        words_2 = nltk.word_tokenize(str(row['question2']))

   

    joint_words = set(words_1).union(set(words_2))

    intersection_words = set(words_1).intersection(set(words_2))

    return len(intersection_words)/len(joint_words)
X_TrainData['Jacard_Distance'] = train.apply(jaccard_similarity_coefficient, axis = 1, raw = True)

X_TestData['Jacard_Distance'] = test.apply(jaccard_similarity_coefficient, axis = 1, raw = True)


from sklearn.metrics.pairwise import cosine_similarity as cs

import re, math

from collections import Counter



WORD = re.compile(r'\w+')

def _cosine_similarity(vector_1, vector_2):

     intersection = set(vector_1.keys()) & set(vector_2.keys())

     numerator = sum([vector_1[x] * vector_2[x] for x in intersection])



     sum1 = sum([vector_1[x]**2 for x in vector_1.keys()])

     sum2 = sum([vector_2[x]**2 for x in vector_2.keys()])

     denominator = math.sqrt(sum1) * math.sqrt(sum2)



     if not denominator:

        return 0.0

     else:

        return float(numerator) / denominator



def sentence_transform(sentence):

     words = WORD.findall(sentence)

     return Counter(words)



def cosine_sim(row):

    vector1 = sentence_transform(str(row['question1']))

    vector2 = sentence_transform(str(row['question2']))

    sim = _cosine_similarity(vector1,vector2)

    return sim



X_TrainData['cosine_sim'] = train.apply(cosine_sim,axis = 1,raw = True )

X_TestData['cosine_sim'] = test.apply(cosine_sim,axis = 1,raw = True )


X_TrainData2 = X_TrainData

Y_TrainData2 = Y_TrainData

#Y_TrainData = train['is_duplicate'].values


#X_TrainData
#X_TrainData = X_TrainData.dropna(axis=0, how='any')

rows = ~X_TrainData.isnull().any(axis=1)



X_TrainData = X_TrainData[rows]

Y_TrainData = Y_TrainData[rows]





from sklearn.cross_validation import train_test_split



#X_TrainData = my_imputer.fit_transform(X_TrainData)

#Y_TrainData = my_imputer.fit_transform(Y_TrainData)





X_TrainData, X_ValidData, Y_TrainData, Y_ValidData = train_test_split(X_TrainData, Y_TrainData, test_size=0.20, random_state=4242)


#2345790 rows × 4 columns

#data_without_missing_values = X_TestData.dropna(axis=1)



import xgboost as xgb



xg_TrainData = xgb.DMatrix(X_TrainData, label=Y_TrainData)

xg_ValidData = xgb.DMatrix(X_ValidData, label=Y_ValidData)



watchlist = [(xg_TrainData, 'train'), (xg_ValidData, 'valid')]



bst = xgb.train({'objective':'binary:logistic','eval_metric':'logloss','eta':0.02,'max_depth' :5}, xg_TrainData, 500, watchlist, early_stopping_rounds=50, verbose_eval=10)
from sklearn.preprocessing import Imputer

my_imputer = Imputer()

data_with_imputed_values = my_imputer.fit_transform(X_TestData)



import sklearn

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn import pipeline

classifier = LogisticRegression();

classifier.set_params(C = 10, max_iter = 10)

lr_model = classifier.fit(X_TrainData,

                          Y_TrainData)

Y_TestData_lr = lr_model.predict(data_with_imputed_values)

Y_TestData_lr

from sklearn import svm



clf = svm.SVC()

clf.set_params(C = 1, kernel = "rbf")

svm_clf = clf.fit(X_TrainData,Y_TrainData)

svm_predict = svm_clf.predict(data_with_imputed_values)



"""testData = pd.DataFrame(columns = ['test_id','question1','question2'])

testData = testData.append([{'test_id':1,'question1':'how are you','question2':'where are you'}])

print (testData)

X_TestData1 = pd.DataFrame()

X_TestData1['cosine_sim'] = testData.apply(cosine_sim,axis = 1,raw = True )

X_TestData1['Jacard_Distance'] = testData.apply(jaccard_similarity_coefficient, axis = 1, raw = True)

X_TestData1['common_word_percent'] = testData.apply(common_words_percentage, axis = 1, raw = True)

X_TestData1['feature_ifidf'] = testData.apply(tfidf_weights, axis = 1, raw = True)"""



xg_TestData = xgb.DMatrix(X_TestData)

xg_ValidData = xgb.DMatrix(X_ValidData)



Predict_TestData = bst.predict(xg_TestData)

Predict_ValidData = bst.predict(xg_ValidData)



Predict_TestData
from sklearn.metrics import precision_recall_curve, auc, roc_curve

fpr, tpr, _ = roc_curve(Y_ValidData, Predict_ValidData)

roc_area = auc(fpr, tpr)

plt.plot(fpr, tpr, lw=1)

np.round(roc_area, 10)
precison, recall, _ = precision_recall_curve(Y_ValidData, Predict_ValidData)

plt.figure(figsize=(10,5))



plt.plot(recall, precison)

plt.xlabel('Recall')

plt.ylabel('Precision')

auc(recall, precison)
result = pd.DataFrame()

result['test_id'] = test['test_id']

result['is_duplicate'] = Predict_TestData

result.to_csv('result.csv', index=False)
Predict_TestData