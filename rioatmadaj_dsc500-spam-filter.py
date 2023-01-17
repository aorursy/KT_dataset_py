from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_curve, auc 

import pandas as pd 

from scipy.sparse.csr import csr_matrix 

import numpy as np

from textblob import TextBlob

from pandas.core.series import Series 

%matplotlib inline

import matplotlib.pyplot as plt 

import seaborn as sns 

import matplotlib

plt.rcParams['figure.figsize'] = (10,20)  

plt.rcParams['font.size'] = 16

plt.style.use('ggplot')

 

# Misc

from typing import List, Dict 

import re 

from nltk import pos_tag, word_tokenize
def spam_engine(clf,text: str) -> str:

    """

    This function will classify if the given input is a spam or ham.

    :clf    : given the classifier

    :text   : 

    :return : a string of classifed text

    """

    if not clf and text: 

        raise AttributeError("Classifier and Text are required to run the engine.")

        

    return list( map(lambda x: 'HAM' if x == 1 else 'SPAM' , mlp.predict( tf_idf.transform([text])).tolist() ) )[0]
def get_freq(text: str) -> List:

    """

    This function will extract the most common word used in a spam email. 

    :text : given the spam text

    :return : The words that associated with spam message

    """

    if not text: 

        raise AttributeError('text is a required parameter.')

        

    vect = CountVectorizer() 

    word = vect.fit_transform( [text] ).toarray().sum(axis=0)

    

    freq: Dict = { pos_tag(word_tokenize(k) )[0]:v[0] for k,v in pd.DataFrame( word.reshape(1, len(word)) , columns=vect.get_feature_names() ).to_dict().items() }     

    

    return ' '.join([ word[0] for word in freq if 'NN' in word[-1] ])

    
def clf_model(clf_funct, X_train_dtm, X_test_dtm, y_train, **kwargs):

    """

    This is a generic function to classify a given ML model

    :X_train: given a vector of independent training data 

    :X_test: given a vector of independent test data

    :y_train: given the training dependent variable data

    :**kwargs : given the ML required arguments 

    """

 

    return clf_funct(kwargs).fit(X_train_dtm, y_train)
def get_clf_proba(**pred_proba) -> Dict:

    """

    This function will return the prediction proba

    :pred_proba: It's a kwargs that contains key: classfier name, value: prediction probabilities

    """

    return pred_proba
def get_accuracy_score(y_pred_class: Series, y_test: Series) -> str: 

    """

    This function will return the prediction accuracy of the given set of prediction class and test vector

    :y_pred_class : given the prediction class vector

    :y_test: given the test class vector 

    """

    return f"{accuracy_score(y_pred_class, y_test) * 100:.3f} %" 
# Data preparation: Encoding issue  

raw_spam: List[str] = open('../input/sms-spam-collection-dataset/spam.csv','r', encoding='latin').read().split('\n') 

spam_raw_data: List[tuple] = list( map(lambda spam: (spam.split(',')[0], spam.split(',')[1:]) , raw_spam[1:] ) )
spam = pd.DataFrame.from_dict(spam_raw_data)

spam.columns = ['classfication','text']

spam.index = np.arange(1,len(spam) + 1 )
spam.shape 
spam['text'] = spam['text'].apply(lambda x: re.sub( ',,', '', ' '.join(x)) )

spam['classfication'] = spam['classfication'].apply(lambda x: re.sub('\"','',x) )
# Set the image size and font here

plt.rcParams['figure.figsize'] = (25,10)  

plt.rcParams['font.size'] = 14
spam['classfication'].value_counts()
spam['classfication'].value_counts().plot(kind='bar')

plt.title("Distribution of Spams") # Lot of ham
# HAM: 1 , SPAM: 0 

spam['map'] = np.where(spam['classfication'] == 'ham' , 1 , 0)  
X = spam['text']

y = spam['map']



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=123)

# tokenize 

tf_idf = TfidfVectorizer(ngram_range=(1,4), lowercase=True, stop_words='english')

X_train_dtm = tf_idf.fit_transform(X_train)

X_test_dtm = tf_idf.transform(X_test)
tokens: List[str] = tf_idf.get_feature_names() 
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,10) , random_state=123)

mlp.fit(X_train_dtm,y_train)

mlp_pred = mlp.predict(X_test_dtm)

mlp_proba = mlp.predict_proba(X_test_dtm)
print(f"Accuracy: { round( accuracy_score(mlp_pred, y_test) * 100, 3 ) } ")
spam[spam['map'] == 0 ]['text']
# is the word Chance associated with spam or ham 

# Todo: extract word frequency in spam 

spam[ spam['text'].apply(lambda x: True if 'chance' in x.lower() else False) ]['classfication'].value_counts().plot(kind='bar')

plt.title('DSC500: Sampling the word chane')

plt.yticks(range(1,30, 5))
spam_engine( mlp, 'Choose the right credit card for you - we made it easy' ) 

spam_engine(mlp, 'Last chance: Get 50%')
extracted_values: List[Dict] = spam[ spam['text'].apply(lambda x: True if 'chance' in x.lower() else False) ]['text'].apply(get_freq).tolist() 
vect = CountVectorizer()

# Spam filter: the following words 

feature_names: List[str] = pd.DataFrame( {'spam_words': extracted_values}, index=range(1,len(extracted_values) + 1))['spam_words'].tolist()

vect.fit_transform(feature_names) 
spam_words = pd.DataFrame( vect.transform(feature_names).toarray().sum(axis=0).reshape(1,184) , columns=vect.get_feature_names()).transpose() 

spam_words.columns = ['spam_words']

spam_words.sort_values(by='spam_words',ascending=False,inplace=True)
spam_words.head(50).plot(kind='bar')

plt.title('DSC500: Words associated with spam')
spam[ spam['text'].apply(lambda x: True if 'chance' in x.lower() else False) ]['text'].apply(get_freq).tolist() 
# Support Vector Machine 

svm_clf = SVC(probability=True) 

svm_clf.fit(X_train_dtm, y_train)

svm_pred = svm_clf.predict(X_test_dtm)

svm_proba = svm_clf.predict_proba(X_test_dtm)
nb_multi = MultinomialNB() 

nb_multi.fit(X_train_dtm, y_train)

nb_multi_pred = nb_multi.predict(X_test_dtm)

nb_multi_proba = nb_multi.predict_proba(X_test_dtm)
knn = KNeighborsClassifier(n_neighbors=10) # With 10 Neighbors 

knn.fit(X_train_dtm, y_train)

knn_pred = knn.predict(X_test_dtm)

knn_proba = knn.predict_proba(X_test_dtm)
pred_proba: Dict = get_clf_proba(svm=(svm_pred,svm_proba[:,1]) , mlp=(mlp_pred,mlp_proba[:,1])  , knn=(knn_pred,knn_proba[:,1]), nb_multi=(nb_multi_pred,nb_multi_proba[:,1]) )
colors: List[str] = ['red', 'blue', 'green', 'purple']

counter: int = 0

    

for clf, data in pred_proba.items(): 

    clf_pred, clf_proba = data 

    fpr, tpr, thresholds = roc_curve(y_test,clf_proba)

    roc_auc = auc(fpr,tpr)

    plt.plot(fpr, tpr, colors[counter], label=f"[+] Model: {clf}\n[+] AUC: {round(roc_auc, 4)} \n[+]Prediction Accuracy: {get_accuracy_score(clf_pred,y_test)}")

    counter += 1



plt.title("DSC500: Spam filter classfier performance")

plt.xlim([-0.005,1.0])

plt.ylim([0.0,1.0])

plt.xlabel("False Postive Rate")

plt.ylabel("True Positve Rate")

plt.legend(loc="upper right")

plt.show() 