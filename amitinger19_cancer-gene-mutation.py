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
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold 
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

from mlxtend.classifier import StackingClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


data_variants_tr = pd.read_csv('../input/cancer-data/training_variants')
print('shape of variants data:',data_variants_tr.shape)
data_variants_tr.head()
# note the seprator in this file
data_text =pd.read_csv("../input/cancer-data/training_text",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
print('Number of data points : ', data_text.shape[0])
print('Number of features : ', data_text.shape[1])
print('Features : ', data_text.columns.values)
data_text.head()
data_text['TEXT'][1]
import nltk
nltk.download('stopwords')
# loading stop words from nltk library
stop_words = set(stopwords.words('english'))


def nlp_preprocessing(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        # replace every special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
        # replace multiple spaces with single space
        total_text = re.sub('\s+',' ', total_text)
        # converting all the chars into lower-case.
        total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from the data
            if not word in stop_words:
                string += word + " "
        
        data_text[column][index] = string
data_text['TEXT']
#text processing stage.
start_time = time.clock()
for index, row in data_text.iterrows():
    if type(row['TEXT']) is str:
        nlp_preprocessing(row['TEXT'], index, 'TEXT')
    else:
        print("there is no text description for id:",index)
print('Time took for preprocessing the text :',time.clock() - start_time, "seconds")
result = pd.merge(data_variants_tr,data_text,on='ID',how='left')
result.head()
result.drop('ID',axis=1,inplace=True)
result.shape
result[result.isnull().any(axis=1)]
#result = pd.merge(data_variants_tr,data_text,on='ID',how='left')
result.loc[result['TEXT'].isnull(),'TEXT'] = result['Gene'] +' '+result['Variation']
result[result['Gene']=='FANCA']

result.Gene      = result.Gene.str.replace('\s+', '_')
result.Variation = result.Variation.str.replace('\s+', '_')
result_x = result.drop('Class',axis=1)
result_y = result['Class']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(result,result_y,stratify=result_y,test_size=0.2)
X_tr,X_cv,Y_tr,Y_cv = train_test_split(X_train,Y_train,stratify=Y_train,test_size=0.2)

print('Train data shape: ',X_tr.shape)
print('Train class shape: ',Y_tr.shape)
print('Test data shape: ',X_test.shape)
print('Test class shape: ',Y_test.shape)
print('CV data shape: ',X_cv.shape)
print('CV class shape: ',Y_cv.shape)
len(X_cv['Gene'].unique())
plt.figure(figsize=(15,10))
sns.countplot(Y_tr)
plt.title('Class distribution of train data',fontsize=30)
plt.xlabel('Class',fontsize=30)
plt.ylabel('count',fontsize=30)
plt.show()
for i in range(1,10):
 print('share of class {0} is {1} with {2} datapoints.'.format(i,100*(Y_tr[Y_tr==i].count())/len(Y_tr),Y_tr[Y_tr==i].count()))
plt.figure(figsize=(15,10))
sns.countplot(Y_test)
plt.title('Class distribution of test data',fontsize=30)
plt.xlabel('Class',fontsize=30)
plt.ylabel('count',fontsize=30)
plt.show()
for i in range(1,10):
 print('share of class {0} is {1} with {2} datapoints.'.format(i,100*(Y_test[Y_test==i].count())/len(Y_test),Y_test[Y_test==i].count()))
plt.figure(figsize=(15,10))
sns.countplot(Y_cv)
plt.title('Class distribution of CV data',fontsize=30)
plt.xlabel('Class',fontsize=30)
plt.ylabel('count',fontsize=30)
plt.show()
for i in range(1,10):
 print('share of class {0} is {1} with {2} datapoints.'.format(i,100*(Y_cv[Y_cv==i].count())/len(Y_cv),Y_cv[Y_cv==i].count()))
# This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j
    
    A =(((C.T)/(C.sum(axis=1))).T)
    #divid each element of the confusion matrix with the sum of elements in that column
    
    # C = [[1, 2],
    #     [3, 4]]
    # C.T = [[1, 3],
    #        [2, 4]]
    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =1) = [[3, 7]]
    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
    #                           [2/3, 4/7]]

    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
    #                           [3/7, 4/7]]
    # sum of row elements = 1
    
    B =(C/C.sum(axis=0))
    #divid each element of the confusion matrix with the sum of elements in that row
    # C = [[1, 2],
    #     [3, 4]]
    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =0) = [[4, 6]]
    # (C/C.sum(axis=0)) = [[1/4, 2/6],
    #                      [3/4, 4/6]] 
    
    labels = [1,2,3,4,5,6,7,8,9]
    # representing A in heatmap format
    print("-"*20, "Confusion matrix", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    
    # representing B in heatmap format
    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()



    # we need to generate 9 numbers and the sum of numbers should be 1
# one solution is to genarate 9 numbers and divide each of the numbers by their sum
# ref: https://stackoverflow.com/a/18662466/4084039
test_data_len = X_test.shape[0]
cv_data_len = X_cv.shape[0]

# we create a output array that has exactly same size as the CV data
cv_predicted_y = np.zeros((cv_data_len,9))
for i in range(cv_data_len):
    rand_probs = np.random.rand(1,9)
    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log loss on Cross Validation Data using Random Model",log_loss(Y_cv,cv_predicted_y, eps=1e-15))


# Test-Set error.
#we create a output array that has exactly same as the test data
test_predicted_y = np.zeros((test_data_len,9))
for i in range(test_data_len):
    rand_probs = np.random.rand(1,9)
    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log loss on Test Data using Random Model",log_loss(Y_test,test_predicted_y, eps=1e-15))

predicted_y =np.argmax(test_predicted_y, axis=1)
plot_confusion_matrix(Y_test, predicted_y+1)
def res_encoding(cat_feature,tar_class):
  genes = list(set(cat_feature))
  vec_list = []
  #print(genes)
  df = pd.DataFrame({'cat_feature':cat_feature,'tar_class':tar_class})
  #print('line 2')
  #print(df)
  dic_feature_prob_given_target = {}
  for i in genes:
    dic_feature_prob_given_target[i]=[]
    for j in range(1,10):
         #print('inside 2nd for loop') 
         #print(df[(df.cat_feature==i) & (df.tar_class==j)].count() / (df[df.cat_feature==i].count()))
         dic_feature_prob_given_target[i].append(((df['cat_feature'][(df['cat_feature']==i) & (df['tar_class']==j)].count())+10) / ((df['cat_feature'][df['cat_feature']==i].count())+90))
         #print('below dict')
  for i in cat_feature:
    vec_list.append(dic_feature_prob_given_target[i])       
  return np.array(vec_list)




def one_hot_encoding(feature_tr,feature):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    one_hot_encoder = CountVectorizer()
    one_hot_encoder.fit(feature_tr)
    one_hot_vec = one_hot_encoder.transform(feature).toarray()
    scaler.fit(one_hot_vec)
    return scaler.transform(one_hot_vec)


def tfidf_vectorizer(feature_tr,feature):
    from sklearn.feature_extraction.text import TfidfVectorizer 
    tfidf = TfidfVectorizer()
    tfidf.fit_transform(feature_tr)
    vec= tfidf.transform(feature).toarray()
    #print(tfidf.get_feature_names())
    return vec
len(result['Gene'].unique())
gene_one_hot_tr = one_hot_encoding(X_tr['Gene'],X_tr['Gene'])
gene_response_tr = res_encoding(X_tr['Gene'],Y_tr)
gene_tfidf_tr = tfidf_vectorizer(X_tr['Gene'],X_tr['Gene'])

gene_one_hot_cv = one_hot_encoding(X_tr['Gene'],X_cv['Gene'])
gene_response_cv = res_encoding(X_cv['Gene'],Y_cv)
gene_tfidf_cv = tfidf_vectorizer(X_tr['Gene'],X_cv['Gene'])

gene_one_hot_test = one_hot_encoding(X_tr['Gene'],X_test['Gene'])
gene_response_test = res_encoding(X_test['Gene'],Y_test)
gene_tfidf_test = tfidf_vectorizer(X_tr['Gene'],X_test['Gene'])
gene_response_test.shape
uniq_val = result['Gene'].value_counts()
plt.figure(figsize=(15,10))
plt.plot(uniq_val.values / uniq_val.sum(),color='blue',linewidth=3)
plt.title('percentage share of genes',fontsize=30)
plt.grid(b=True)
plt.xlabel('indices of genes',fontsize=20)
plt.ylabel('% share',fontsize=20)
plt.annotate('total unique genes= '+str(len(uniq_val)),xy=(180,0.075),fontsize=20,color='red' )
plt.show()

cs = np.cumsum(uniq_val.values / uniq_val.sum())
plt.figure(figsize=(15,10))
plt.plot(cs,color='blue',linewidth=3)
plt.title('cummulative sum',fontsize=30)
plt.grid(b=True)
plt.xlabel('indices of genes',fontsize=20)
plt.ylabel('% share',fontsize=20)
plt.annotate('total unique genes= '+str(len(uniq_val)),xy=(180,0.075),fontsize=20,color='red' )
plt.show()
def performance(vector_tr,vector_test,vector_cv,Y_tr,Y_test,Y_cv,alpha):
  error_cv=[]
  error_tr=[]
  error_test = []
  for i in alpha:
      clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
      clf.fit(vector_tr, Y_tr)
      sig_clf = CalibratedClassifierCV(clf, method="sigmoid") # we want our predicted value to be a probability for the interpretability hence we are using CalibratedClassifierCV
      sig_clf.fit(vector_tr,Y_tr)
      predict_y_tr = sig_clf.predict_proba(vector_tr)
      predict_y = sig_clf.predict_proba(vector_cv)
      predict_y_test = sig_clf.predict_proba(vector_test)
      error_cv.append(log_loss(Y_cv, predict_y, labels=clf.classes_, eps=1e-15))
      error_tr.append(log_loss(Y_tr, predict_y_tr, labels=clf.classes_, eps=1e-15))
      error_test.append(log_loss(Y_test, predict_y_test, labels=clf.classes_, eps=1e-15))  
  plt.figure(figsize=(15,10))
  plt.plot(error_cv,color='blue',linewidth=3)
  plt.plot(error_tr,color='grey',linewidth=3)
  plt.plot(error_test,color='red',linewidth=3)
  plt.title('performance checker',fontsize=30)
  plt.grid(b=True)
  plt.legend(['CV','Train','test'])
  plt.xlabel('Hyperparameter value',fontsize=20)
  plt.ylabel('log-loss',fontsize=20)
  #for i,j in zip(alpha,error_cv):
      #plt.annotate(str(round(j,2)),xy=(i,j),fontsize=20,color='grey' )
  plt.show()
  for i in (range(len(alpha))):
    print('Log loss is train =  {0}, test = {1} and cv = {2} for alpha value {3}'.format(error_tr[i],error_test[i],error_cv[i],alpha[i] )) 
 



print('*'*50,'One Hot Encoding','*'*50)
alpha = [10 ** x for x in range(-5, 1)]
performance(gene_one_hot_tr,gene_one_hot_test,gene_one_hot_cv,Y_tr,Y_test,Y_cv,alpha) 
print('*'*50,'Response Encoding','*'*50)
performance(gene_response_tr,gene_response_test,gene_response_cv,Y_tr,Y_test,Y_cv,alpha) 
print('*'*50,'TFIDF Encoding','*'*50)
performance(gene_tfidf_tr,gene_tfidf_test,gene_tfidf_cv,Y_tr,Y_test,Y_cv,alpha) 

var_one_hot_tr = one_hot_encoding(X_tr['Variation'],X_tr['Variation'])
var_response_tr = res_encoding(X_tr['Variation'],Y_tr)
var_tfidf_tr = tfidf_vectorizer(X_tr['Variation'],X_tr['Variation'])

var_one_hot_cv = one_hot_encoding(X_tr['Variation'],X_cv['Variation'])
var_response_cv = res_encoding(X_cv['Variation'],Y_cv)
var_tfidf_cv = tfidf_vectorizer(X_tr['Variation'],X_cv['Variation'])

var_one_hot_test = one_hot_encoding(X_tr['Variation'],X_test['Variation'])
var_response_test = res_encoding(X_test['Variation'],Y_test)
var_tfidf_test = tfidf_vectorizer(X_tr['Variation'],X_test['Variation'])
uniq_val_var = result['Variation'].value_counts()
plt.figure(figsize=(15,10))
plt.plot(uniq_val_var.values / uniq_val_var.sum(),color='blue',linewidth=3)
plt.title('percentage share of Variation',fontsize=30)
plt.grid(b=True)
plt.xlabel('indices of Variation',fontsize=20)
plt.ylabel('% share',fontsize=20)
plt.annotate('total unique variation= '+str(len(uniq_val_var)),xy=(2000,0.025),fontsize=20,color='red' )
plt.show()
cs = np.cumsum(uniq_val_var.values / uniq_val_var.sum())
plt.figure(figsize=(15,10))
plt.plot(cs,color='blue',linewidth=3)
plt.title('cummulative sum',fontsize=30)
plt.grid(b=True)
plt.xlabel('indices of variation',fontsize=20)
plt.ylabel('% share',fontsize=20)
plt.annotate('total unique variation= '+str(len(uniq_val_var)),xy=(180,0.075),fontsize=20,color='red' )
plt.show()
print('*'*50,'One Hot Encoding','*'*50)
alpha = [10 ** x for x in range(-5, 1)]
performance(var_one_hot_tr,var_one_hot_test,var_one_hot_cv,Y_tr,Y_test,Y_cv,alpha) 
print('*'*50,'Response Encoding','*'*50)
performance(var_response_tr,var_response_test,var_response_cv,Y_tr,Y_test,Y_cv,alpha) 
print('*'*50,'TFIDF Encoding','*'*50)
performance(var_tfidf_tr,var_tfidf_test,var_tfidf_cv,Y_tr,Y_test,Y_cv,alpha) 
text_one_hot_tr = one_hot_encoding(X_tr['TEXT'],X_tr['TEXT'])
text_response_tr = res_encoding(X_tr['TEXT'],Y_tr)
text_tfidf_tr = tfidf_vectorizer(X_tr['TEXT'],X_tr['TEXT'])

text_one_hot_cv = one_hot_encoding(X_tr['TEXT'],X_cv['TEXT'])
text_response_cv = res_encoding(X_cv['TEXT'],Y_cv)
text_tfidf_cv = tfidf_vectorizer(X_tr['TEXT'],X_cv['TEXT'])

text_one_hot_test = one_hot_encoding(X_tr['TEXT'],X_test['TEXT'])
text_response_test = res_encoding(X_test['TEXT'],Y_test)
text_tfidf_test = tfidf_vectorizer(X_tr['TEXT'],X_test['TEXT'])
uniq_val_text = result['TEXT'].value_counts()
plt.figure(figsize=(15,10))
plt.plot(uniq_val_text.values / uniq_val_text.sum(),color='blue',linewidth=3)
plt.title('percentage share of TEXT',fontsize=30)
plt.grid(b=True)
plt.xlabel('indices of TEXT',fontsize=20)
plt.ylabel('% share',fontsize=20)
plt.annotate('total unique TEXT= '+str(len(uniq_val_text)),xy=(1200,0.015),fontsize=20,color='red' )
plt.show()

cs = np.cumsum(uniq_val_text.values / uniq_val_text.sum())
plt.figure(figsize=(15,10))
plt.plot(cs,color='blue',linewidth=3)
plt.title('cummulative sum',fontsize=30)
plt.grid(b=True)
plt.xlabel('indices of TEXT',fontsize=20)
plt.ylabel('% share',fontsize=20)
plt.annotate('total unique TEXT= '+str(len(uniq_val_text  )),xy=(1200,0.015),fontsize=20,color='red' )
plt.show()
print('*'*50,'One Hot Encoding','*'*50)
alpha = [10 ** x for x in range(-5, 1)]
performance(text_one_hot_tr,text_one_hot_test,text_one_hot_cv,Y_tr,Y_test,Y_cv,alpha) 
print('*'*50,'Response Encoding','*'*50)
performance(text_response_tr,text_response_test,text_response_cv,Y_tr,Y_test,Y_cv,alpha) 
print('*'*50,'TFIDF Encoding','*'*50)
performance(text_tfidf_tr,text_tfidf_test,text_tfidf_cv,Y_tr,Y_test,Y_cv,alpha) 
data_tr = np.hstack((gene_one_hot_tr,var_response_tr,text_response_tr ))
data_test = np.hstack((gene_one_hot_test,var_response_test,text_response_test ))
data_cv = np.hstack((gene_one_hot_cv,var_response_cv,text_response_cv ))
print('train shape:',data_tr.shape)
print('test shape:',data_test.shape)
print('cv shape:',data_cv.shape)
alpha = [10 ** x for x in range(-6, 3)] # this is the  lambda value we need to find out by cross validation 
def performance_model(vector_tr,vector_cv,Y_tr,Y_cv,alpha):
  error_cv=[]
  error_tr=[]
  for i in alpha:
      clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)
      clf.fit(vector_tr, Y_tr)
      sig_clf = CalibratedClassifierCV(clf, method="sigmoid") # we want our predicted value to be a probability for the interpretability hence we are using CalibratedClassifierCV
      sig_clf.fit(vector_tr,Y_tr)
      predict_y_tr = sig_clf.predict_proba(vector_tr)
      predict_y = sig_clf.predict_proba(vector_cv)
      error_cv.append(log_loss(Y_cv, predict_y, labels=clf.classes_, eps=1e-15))
      error_tr.append(log_loss(Y_tr, predict_y_tr, labels=clf.classes_, eps=1e-15)) 
  plt.figure(figsize=(15,10))
  plt.plot(error_cv,color='blue',linewidth=3)
  plt.plot(error_tr,color='grey',linewidth=3)
  plt.title('performance checker',fontsize=30)
  plt.grid(b=True)
  plt.legend(['CV','Train'])
  plt.xlabel('Hyperparameter value',fontsize=20)
  plt.ylabel('log-loss',fontsize=20)
  #for i,j in zip(alpha,error_cv):
      #plt.annotate(str(round(j,2)),xy=(i,j),fontsize=20,color='grey' )
  plt.show()
  for i in (range(len(alpha))):
    print('Log loss is train =  {0} and cv = {1} for alpha value {2}'.format(error_tr[i],error_cv[i],alpha[i] )) 
 
performance_model(data_tr,data_cv,Y_tr,Y_cv,alpha)

# Now that we know our best model on whole dataset, we will train our model with best alpha value

def log_loss_and_confusion_matrix(data_tr, Y_tr,data_test, Y_test, model):
    model.fit(data_tr, Y_tr)
    sig_clf = CalibratedClassifierCV(model, method="sigmoid")
    sig_clf.fit(data_tr, Y_tr)
    pred_y = sig_clf.predict(data_test)

    # for calculating log_loss we willl provide the array of probabilities belongs to each class
    print("Log loss :",log_loss(Y_test, sig_clf.predict_proba(data_test)))
    # calculating the number of data points that are misclassified
    print("Number of mis-classified points :", np.count_nonzero((pred_y- Y_test))/Y_test.shape[0])
    plot_confusion_matrix(Y_test, pred_y)
    return pred_y 
from sklearn.metrics import log_loss
model = SGDClassifier( class_weight='balanced',loss='log',penalty='l2',alpha=0.001)
log_loss_and_confusion_matrix(data_tr, Y_tr,data_test, Y_test, model)
def performance_model_hinge(vector_tr,vector_cv,Y_tr,Y_cv,alpha):
  error_cv=[]
  error_tr=[]
  for i in alpha:
      clf = SGDClassifier(alpha=i, penalty='l2', loss='hinge', random_state=42)
      clf.fit(vector_tr, Y_tr)
      sig_clf = CalibratedClassifierCV(clf, method="sigmoid") # we want our predicted value to be a probability for the interpretability hence we are using CalibratedClassifierCV
      sig_clf.fit(vector_tr,Y_tr)
      predict_y_tr = sig_clf.predict_proba(vector_tr)
      predict_y = sig_clf.predict_proba(vector_cv)
      error_cv.append(log_loss(Y_cv, predict_y, labels=clf.classes_, eps=1e-15))
      error_tr.append(log_loss(Y_tr, predict_y_tr, labels=clf.classes_, eps=1e-15)) 
  plt.figure(figsize=(15,10))
  plt.plot(error_cv,color='blue',linewidth=3)
  plt.plot(error_tr,color='grey',linewidth=3)
  plt.title('performance checker',fontsize=30)
  plt.grid(b=True)
  plt.legend(['CV','Train'])
  plt.xlabel('Hyperparameter value',fontsize=20)
  plt.ylabel('log-loss',fontsize=20)
  #for i,j in zip(alpha,error_cv):
      #plt.annotate(str(round(j,2)),xy=(i,j),fontsize=20,color='grey' )
  plt.show()
  for i in (range(len(alpha))):
    print('Log loss is train =  {0} and cv = {1} for alpha value {2}'.format(error_tr[i],error_cv[i],alpha[i] )) 
 
performance_model_hinge(data_tr,data_cv,Y_tr,Y_cv,alpha)
model = SGDClassifier(loss='hinge',penalty='l2',alpha=100)
log_loss_and_confusion_matrix(data_tr, Y_tr,data_test, Y_test, model)
k = [1,3,5,7,9,11,13,15,17]
def performance_knn(vector_tr,vector_cv,Y_tr,Y_cv,alpha):
  from sklearn.neighbors import KNeighborsClassifier
  error_cv=[]
  error_tr=[]
  for i in alpha:
      clf = KNeighborsClassifier(n_neighbors=i)
      clf.fit(vector_tr, Y_tr)
      sig_clf = CalibratedClassifierCV(clf, method="sigmoid") # we want our predicted value to be a probability for the interpretability hence we are using CalibratedClassifierCV
      sig_clf.fit(vector_tr,Y_tr)
      predict_y_tr = sig_clf.predict_proba(vector_tr)
      predict_y = sig_clf.predict_proba(vector_cv)
      error_cv.append(log_loss(Y_cv, predict_y, labels=clf.classes_, eps=1e-15))
      error_tr.append(log_loss(Y_tr, predict_y_tr, labels=clf.classes_, eps=1e-15)) 
  plt.figure(figsize=(15,10))
  plt.plot(error_cv,color='blue',linewidth=3)
  plt.plot(error_tr,color='grey',linewidth=3)
  plt.title('performance checker',fontsize=30)
  plt.grid(b=True)
  plt.legend(['CV','Train'])
  plt.xlabel('Hyperparameter value',fontsize=20)
  plt.ylabel('log-loss',fontsize=20)
  #for i,j in zip(alpha,error_cv):
      #plt.annotate(str(round(j,2)),xy=(i,j),fontsize=20,color='grey' )
  plt.show()
  for i in (range(len(alpha))):
    print('Log loss is train =  {0} and cv = {1} for alpha value {2}'.format(error_tr[i],error_cv[i],alpha[i] )) 
performance_knn(data_tr,data_cv,Y_tr,Y_cv,k)
model = KNeighborsClassifier(n_neighbors=17)
log_loss_and_confusion_matrix(data_tr, Y_tr,data_test, Y_test, model)
def performance_rf(vector_tr,vector_cv,Y_tr,Y_cv,alpha):
  from sklearn.ensemble import RandomForestClassifier
  for l in [5,10]:
    error_cv=[]
    error_tr=[]
    for i in alpha:
        clf = RandomForestClassifier(n_estimators=i,criterion='gini',max_depth=l)
        clf.fit(vector_tr, Y_tr)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid") # we want our predicted value to be a probability for the interpretability hence we are using CalibratedClassifierCV
        sig_clf.fit(vector_tr,Y_tr)
        predict_y_tr = sig_clf.predict_proba(vector_tr)
        predict_y = sig_clf.predict_proba(vector_cv)
        error_cv.append(log_loss(Y_cv, predict_y, labels=clf.classes_, eps=1e-15))
        error_tr.append(log_loss(Y_tr, predict_y_tr, labels=clf.classes_, eps=1e-15)) 
    plt.figure(figsize=(15,10))
    plt.plot(error_cv,color='blue',linewidth=3)
    plt.plot(error_tr,color='grey',linewidth=3)
    plt.title('performance checker max_depth='+str(l),fontsize=30)
    plt.grid(b=True)
    plt.legend(['CV','Train'])
    plt.xlabel('Hyperparameter value',fontsize=20)
    plt.ylabel('log-loss',fontsize=20)
    #for i,j in zip(alpha,error_cv):
        #plt.annotate(str(round(j,2)),xy=(i,j),fontsize=20,color='grey' )
    plt.show()
    for i in (range(len(alpha))):
      print('Log loss is train =  {0} and cv = {1} for alpha value {2}'.format(error_tr[i],error_cv[i],alpha[i] )) 
alpha = [100,200,500,1000,2000]
performance_rf(data_tr,data_cv,Y_tr,Y_cv,alpha)
model = RandomForestClassifier(n_estimators=200,criterion='gini',max_depth=5)
prediction_test=log_loss_and_confusion_matrix(data_tr, Y_tr,data_test, Y_test, model)



