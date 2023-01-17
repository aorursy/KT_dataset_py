import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



#NLTK

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer





from sklearn.model_selection import GridSearchCV

from scipy.stats import uniform



from sklearn.metrics import f1_score, auc, precision_recall_curve, average_precision_score

from sklearn.metrics import auc

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split,TimeSeriesSplit

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer





from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import auc

from sklearn.metrics import confusion_matrix
# importing the data set 

data = pd.read_csv("../input/Reviews.csv")

data.head()
data = data[data["Score"] != 3]

print(data.shape)
def new_score(y):

    

    if y < 3:

        return 0

    else:

        return 1

    

       

data['new_score'] = data['Score'].map(new_score)



data.drop('Score',axis=1,inplace=True)

values = data["new_score"].value_counts()

plt.bar(("Positive","Negative"),(values[1],values[0]))

plt.show()
values
neg_rev= data[data["new_score"] == 0][:25000]

pos_rev = data[data["new_score"] ==1][:25000]
data_50 = pd.concat((neg_rev,pos_rev),axis=0)
data_50.shape
data_50 = data_50.sort_values("ProductId")
# This gives the rows which are duplicated on the bases if text

data_50[data_50.duplicated("Text")]
# Lets look into a row 

data[data["UserId"] == "A253F3QA4WHGXU"]

data_50=data_50.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
data_50=data_50[data_50.HelpfulnessNumerator<=data_50.HelpfulnessDenominator]
# Creating a backup

raw = data_50
tokenizer = RegexpTokenizer("[a-zA-Z@]+") # We only want words in text as punctuation and numbers are not helpful

en_stopwords = set(stopwords.words("english"))

ss = SnowballStemmer("english")





stop = stopwords.words('english') 



excluding = ['against','not','don', "don't",'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",

             'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 

             'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",'shouldn', "shouldn't", 'wasn',

             "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

stop = [words for words in stop if words not in excluding]
def clean_up(sentence):

    

    

    sentence  = tokenizer.tokenize(sentence) # Conerting in regualr expression

    sentence = [ss.stem(w) for w in sentence if w not in stop  ]  # Stemming and removing stop words

    return " ".join(sentence) # returning the sentence in the form of a string
data_50["Clean_Text"] = data_50["Text"].apply(clean_up)
print("Text === ",data_50["Text"][1])

print("*"*100)

print("Clean_Text === ",data_50["Clean_Text"][1])
X = data_50["Clean_Text"]

y = data_50["new_score"]
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.3,random_state=2)
print("Train Data Size: ",X_train.shape)

print("Test Data Size: ",X_test.shape)



print("Train Data Size: ",Y_train.shape)

print("Test Data Size: ",Y_test.shape)
count_vect = CountVectorizer(ngram_range=(1,2)) #in scikit-learn



X_train_bow = count_vect.fit_transform(X_train) #

X_test_bow = count_vect.transform(X_test)#

print("the type of count vectorizer ",type(X_train_bow))

print("the shape of out text BOW vectorizer ",X_train_bow.get_shape())

print("the number of unique words ", X_train_bow.get_shape()[1])
from sklearn import preprocessing

X_train_bow = preprocessing.normalize(X_train_bow)

X_test_bow = preprocessing.normalize(X_test_bow)
tscv = TimeSeriesSplit(n_splits=10)
tuned_parameters = { 'C': [10**-4, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**4],

              'penalty':['l1','l2']}

model_precision = RandomizedSearchCV(LogisticRegression(n_jobs = -1), tuned_parameters, cv = tscv,

                     scoring = "precision", n_jobs = -1)

model_precision.fit(X_train_bow, Y_train)



print("Best C and penalty",model_precision.best_params_)

print("precision on train data",model_precision.best_score_*100)
clf_bow = LogisticRegression(C = 10000,penalty="l2")

clf_bow.fit(X_train_bow,Y_train)

pred_train =clf_bow.predict(X_train_bow)

pred_test = clf_bow.predict(X_test_bow)







pred_train_proba = clf_bow.predict_proba(X_train_bow)

pred_test_proba = clf_bow.predict_proba(X_test_bow)


cm_bow = confusion_matrix(Y_test, pred_test)



import seaborn as sns

class_label = ["negative", "positive"]

df_cm = pd.DataFrame(cm_bow, index = class_label, columns = class_label)

sns.heatmap(df_cm, annot = True, fmt = "d")

plt.title("Confusiion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
print('TNR for Test = ',(cm_bow[0][0])/(cm_bow[0][0] + cm_bow[1][0]) )

print('FPR for Test = ',(cm_bow[1][0])/(cm_bow[1][0]+cm_bow[0][0]))
data_test = data[70000:80000]

data_test["Clean_review"] = data_test["Text"].apply(clean_up)

X_test_bow_1 = count_vect.transform(data_test["Clean_review"])

y1 = data_test["new_score"]

X_test_bow_1 = preprocessing.normalize(X_test_bow_1)
pred_test_1 = clf_bow.predict(X_test_bow_1)

pred_test_1_proba = clf_bow.predict_proba(X_test_bow_1)
print("roc_auc_score -- ",roc_auc_score(y1,pred_test_1))

print("Precision--",precision_score(y1,pred_test_1))

print("Recall--",recall_score(y1,pred_test_1))
cm = confusion_matrix(y1, pred_test_1)

cm
print('TNR for Test = ',(cm[0][0])/(cm[0][0] + cm[1][0]) )

print('FPR for Test = ',cm[1][0]/(cm[1][0]+cm[0][0]) )
fpr, tpr, threshold  = roc_curve(Y_train, pred_train_proba[:,1])

fpr1, tpr1, threshold1 = roc_curve(Y_test, pred_test_proba[:,1])

fpr2, tpr2, threshold2 =roc_curve(y1,pred_test_1_proba[:,1])



plt.plot(fpr,tpr, label = 'Train AUC = ' + str(auc(fpr,tpr)))

plt.plot(fpr1,tpr1, label = 'Test AUC = '+ str(auc(fpr1,tpr1)))

plt.plot(fpr2,tpr2, label = "Imbalanced AUC = "+ str(auc(fpr2,tpr2)))





plt.ylim(0,1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.grid(True)

plt.title("ROC Curve for Train and Test Data\n")

plt.xlabel("FPR")

plt.ylabel("TPR")

plt.show()
vectorizer = TfidfVectorizer(ngram_range=(1,2))

X_train_Tfidf = vectorizer.fit_transform(X_train)

X_test_Tfidf = vectorizer.transform(X_test)
tuned_parameters = { 'C': [10**-4, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**4],

              'penalty':['l1','l2']}

model_precision = RandomizedSearchCV(LogisticRegression(n_jobs = -1), tuned_parameters, cv = tscv,

                     scoring = "precision", n_jobs = -1)

model_precision.fit(X_train_Tfidf, Y_train)



print("Best C and penalty",model_precision.best_params_)

print("precision on train data",model_precision.best_score_*100)
clf_tf = LogisticRegression(C = 100,penalty="l2")

clf_tf.fit(X_train_Tfidf,Y_train)

pred_test_tf = clf_tf.predict(X_test_Tfidf)

pred_train_prob_tf = clf_tf.predict_proba(X_train_bow)

pred_test_prob_tf = clf_tf.predict_proba(X_test_bow)


cm_tf = confusion_matrix(Y_test, pred_test_tf)



import seaborn as sns

class_label = ["negative", "positive"]

df_cm = pd.DataFrame(cm_tf, index = class_label, columns = class_label)

sns.heatmap(df_cm, annot = True, fmt = "d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
print('TNR for Test = ',(cm_tf[0][0])/(cm_tf[0][0] + cm_tf[1][0]) )

print('FPR for Test = ',cm_tf[1][0]/(cm_tf[1][0]+cm_tf[0][0]) )
print("Scores on Balanced Test data  ")

print("roc_auc_score -- ",roc_auc_score(Y_test,pred_test_tf))

print("Precision--",precision_score(Y_test,pred_test_tf))

print("Recall--",recall_score(Y_test,pred_test_tf))
X_test_Tfidf_1 = vectorizer.transform(data_test["Clean_review"])

pred_Tfidf =clf_tf.predict(X_test_Tfidf_1 )

pred_Tfidf_proba = clf_tf.predict_proba(X_test_Tfidf_1)
print("Scores on Imbalanced Test data  ")

print("roc_auc_score -- ",roc_auc_score(y1,pred_Tfidf))

print("Precision--",precision_score(y1,pred_Tfidf))

print("Recall--",recall_score(y1,pred_Tfidf))
cm = confusion_matrix(y1, pred_test_1)

cm

print('TNR for Test = ',(cm[0][0])/(cm[0][0] + cm[1][0]))

print('FPR for Test = ',cm[1][0]/(cm[1][0]+cm[0][0]))
from sklearn.metrics import roc_curve
fpr, tpr, threshold  = roc_curve(Y_train, pred_train_prob_tf[:,1])

fpr1, tpr1, threshold1 = roc_curve(Y_test, pred_test_prob_tf[:,1])

fpr2, tpr2, threshold2 =roc_curve(y1, pred_Tfidf_proba[:,1])



plt.plot(fpr,tpr, label = 'Train AUC = ' + str(auc(fpr,tpr)))

plt.plot(fpr1,tpr1, label = 'Test AUC = '+ str(auc(fpr1,tpr1)))

plt.plot(fpr2,tpr2, label = "Imbalanced AUC = "+ str(auc(fpr2,tpr2)))





plt.ylim(0,1)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.grid(True)

plt.title("ROC Curve for Train and Test Data\n")

plt.xlabel("FPR")

plt.ylabel("TPR")

plt.show()
# Please write all the code with proper documentation0



LR = clf_tf

LR.fit(X_train_bow,Y_train)

weight1 = LR.coef_ # weight vector



# Getting new data set by addind a small noise

new_train = X_train_bow.astype(float)

new_train.data += np.random.uniform(-0.0001,0.0001,1 )



# Fitting the model again on new data

LR = clf_tf

LR.fit(new_train,Y_train)

weight2 = LR.coef_



# Adding small esilon to weight vector to avoid division by 0

weight1 += 10**-6

weight2 += 10**-6



percentage_change_vector = abs( (weight1-weight2) / (weight1) )*100





t = range(0,101,10)

for i in t:

    print(i, "th percentile : ",np.percentile(percentage_change_vector,i))



plt.plot(t,np.percentile(percentage_change_vector,t) )
diff = (abs(weight1 - weight2)/weight1) * 100

q = diff[np.where(diff > 30)].size

print("Percentage of features which did not change by more than 30% is :",(weight1.size - q)/weight1.size*100)
LR = clf_tf

LR.fit(X_train_bow,Y_train)

feat_log = LR.coef_



vectorizer = TfidfVectorizer(ngram_range=(1,2))

p = vectorizer.fit_transform(X_train)

p = pd.DataFrame(feat_log.T,columns=['+ve'])

p['feature'] = vectorizer.get_feature_names()

q = p.sort_values(by = '+ve',kind = 'quicksort',ascending= False)

print("Top 10  important features of positive class", np.array(q['feature'][:10]))
print("Top 10  important features of negative class",np.array(q.tail(10)['feature']))