import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import sklearn as sklearn
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
%config InlineBackend.figure_format = 'retina'

data = pd.read_csv("spam.csv",encoding='latin-1')
#Drop column and name change
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})

from sklearn.preprocessing import LabelEncoder
le = sklearn.preprocessing.LabelEncoder()
le.fit(data["label"])
data["label"] = le.transform(data["label"])    #change the labels to 0 and 1 
from sklearn.model_selection import train_test_split
train,test = train_test_split(data, test_size = 0.3, random_state = 42,
                                               shuffle=True, stratify= data["label"] )   #select balanced sample
X_train = train["text"]
X_test = test["text"]
y_train = train["label"]
y_test = test["label"]
#Separate the training set to "ham" and "spam" 
train_ham = train.loc[train.label == 0]
train_spam = train.loc[train.label == 1]
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
tokenizer = RegexpTokenizer(r'\w+') #tokenize words while removing punctuations
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() #to combine words of same lemma 
hamword = []
for i in train_ham.text:
    words = i.lower()
    words = tokenizer.tokenize(words)
    for j in words:
        if j not in stopwords.words("english"):
            if not j.isdigit():
                j = lemmatizer.lemmatize(j)
                hamword.append(j)
            
spamword = []
for i in train_spam.text:
    words = i.lower()
    words = tokenizer.tokenize(words)
    for j in words:
        if j not in stopwords.words("english"):
            if not j.isdigit():
                j = lemmatizer.lemmatize(j)
                spamword.append(j)      
from collections import Counter
Counter(hamword).most_common(10)
Counter(spamword).most_common(10)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',lowercase=True,use_idf=True)
Xtrain = vectorizer.fit_transform(X_train)
Xtest = vectorizer.transform(X_test) #use the fitted vectorizer to transform X_test 
print(Xtrain.shape,Xtest.shape)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,auc
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(Xtrain,y_train)
NB_pred = NB.predict(Xtest)
NB_pred_proba = NB.predict_proba(Xtest)
print ("prediciton Accuracy : %f" % accuracy_score(y_test, NB_pred))
#Plot the ROC curve to examine the performance of the model 
import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_roc_curve(y_test, NB_pred_proba)
plt.show()
print ("AUC Score : %f" % sklearn.metrics.roc_auc_score(y_test, NB_pred_proba[:,1]))
#Confusion Matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plt.figure()
plot_confusion_matrix(confusion_matrix(y_test,NB_pred), classes=['0', '1'], normalize=False,
                      title='Normalized confusion matrix')
plt.show()
#Classification table
def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):
    lines = cr.split('\n')
    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        t = line.split()
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)
    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)
    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
print(classification_report(y_test, NB_pred, labels=['0', '1']))
plot_classification_report(classification_report(y_test, NB_pred, labels=['0', '1']))
#class ratio:
print("ham vs spam =",Counter(y_train)[0]/Counter(y_train)[1])
#define a function for performance metrics
def model_eval(model,Xtest,y_test):
    pred = model.predict(Xtest)
    if not str(model)[:3] == "SGD":
        pred_proba = model.predict_proba(Xtest)
        pred_proba_c1 = pred_proba[:,1]
        print ("AUC Score : %f" % sklearn.metrics.roc_auc_score(y_test, pred_proba_c1))
    print ("prediciton Accuracy : %f" % accuracy_score(y_test, pred))
    print ("Confusion_matrix : ")
    print (confusion_matrix(y_test,pred))
    print ("classification report : ")
    print (classification_report(y_test, pred, labels=['0', '1']))
from sklearn.linear_model import LogisticRegressionCV
#"liblinear" is suitable for small dataset, use L1(Lasso) regularization to reduce dimensions, adjust weights
LRcv = LogisticRegressionCV(solver="liblinear",penalty = "l1",class_weight ="balanced")  
LRcv.fit(Xtrain,y_train)
model_eval(LRcv,Xtest,y_test)
#random forest works well when number of features is huge.
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators =100, max_features = "sqrt",bootstrap = True, oob_score=True,verbose=0,
                            class_weight = "balanced",random_state=42,max_depth = 40)
RF.fit(Xtrain,y_train)
print("RF.oob_score : %f" % RF.oob_score_)
model_eval(RF,Xtest,y_test)
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(n_estimators=100, max_features = "sqrt", learning_rate=0.25,
     max_depth=100, subsample= 0.8, random_state=42)
#create sample_weights array
sample_weights = [0.15 if x == 0 else 0.85 for x in y_train]
GBC.fit(Xtrain,y_train,sample_weight = sample_weights)
model_eval(GBC,Xtest,y_test)
from sklearn.linear_model import SGDClassifier 
#using SVM with loss="hinge" will automatically deal with the imbalance in the dataset
SGD = SGDClassifier(loss="hinge", penalty="l2", alpha=0.0001,           
                    l1_ratio=0.15, fit_intercept=True, 
                    shuffle=True, learning_rate="optimal", n_iter= np.ceil(10**6 / Xtrain.shape[1])) 
SGD.fit(Xtrain,y_train)
model_eval(SGD,Xtest,y_test)
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
svd = TruncatedSVD(n_components=100, random_state=42)   #dimension 100 as recommended by sklearn documentation
lsa = make_pipeline(svd, Normalizer(copy=False))
Xtrain_lsa = lsa.fit_transform(Xtrain)
Xtest_lsa = lsa.transform(Xtest)
print(svd.explained_variance_ratio_.sum())
#use Gradient Boosting on the transformed data
GBC_lsa = GradientBoostingClassifier(n_estimators=100, max_features = "sqrt", learning_rate=0.25,
     max_depth=20, subsample= 0.8, random_state=42)
#create sample_weights array
sample_weights = [0.15 if x == 0 else 0.85 for x in y_train]
GBC_lsa.fit(Xtrain_lsa,y_train,sample_weight = sample_weights)
model_eval(GBC_lsa,Xtest_lsa,y_test)
from sklearn.model_selection import GridSearchCV
#for the first parameter, we try to look for the best n_estimators under learning_rate = 0.1
param_test1 = {'n_estimators':range(50,151,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
                                   min_samples_leaf=10,max_depth=100,max_features='sqrt', 
                                    subsample=0.8,random_state=42), 
                       param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(Xtrain,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#We then use the best estimated n_estimators(130) and search for the best max_depth
param_test2 = {'max_depth':range(15,51,5)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=130, 
                                    min_samples_leaf=10, max_features='sqrt', 
                                        subsample=0.8, random_state=42), 
                   param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(Xtrain,y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
#min_samples_split and min_samples_leaf since these two parameters are related
param_test3 = {'min_samples_split':range(100,301,50), 'min_samples_leaf':range(3,24,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=130,
                                    max_depth=35,max_features='sqrt', 
                                        subsample=0.8, random_state=42), 
                       param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(Xtrain,y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
#max_features
param_test4 = {'max_features':range(40,131,10)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=130,
                                    max_depth=35, min_samples_leaf =3, min_samples_split =150, 
                                            subsample=0.8, random_state=42), 
                       param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(Xtrain,y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
#subsample
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=130,
                                    max_depth=35, min_samples_leaf =3, min_samples_split =150, 
                                                max_features=40, random_state=42), 
                       param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
gsearch5.fit(Xtrain,y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
GBC2 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=260,max_depth=35, min_samples_leaf =3, 
               min_samples_split =150, max_features=40, subsample=0.7, random_state=42)
sample_weights = [0.15 if x == 0 else 0.85 for x in y_train]
GBC2.fit(Xtrain,y_train,sample_weight = sample_weights)
model_eval(GBC2,Xtest,y_test)
importances = GBC2.feature_importances_
std = np.std([GBC2.feature_importances_ for tree in GBC2.estimators_],axis=0)
#top 10 indices:
indices = np.argsort(importances)[::-1][0:10]
feature_names = vectorizer.get_feature_names()
print ("top10words : ")
for i in range(10):
    print (indices[i],feature_names[indices[i]])
plt.figure()
plt.title("Feature importances")
plt.bar(range(10), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(10), indices)
plt.xlim([-1, 10])
plt.show()