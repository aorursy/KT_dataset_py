!pip install scikit-learn==0.23
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')
!pip install multi-imbalance
df=pd.read_excel('../input/hasoc-hindi-lokesh/Hindi.xlsx')

df.head()
import nltk
import re
nltk.download('wordnet')
from nltk.tokenize import WordPunctTokenizer 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
tok = WordPunctTokenizer()

lemmatizer = WordNetLemmatizer() 

nltk.download('stopwords')
from nltk.corpus import stopwords
stopword = stopwords.words('english')
from sklearn.feature_extraction.text import TfidfVectorizer


def tweet_cleaner(text):
   
    wc=[]
    newString = str(text)                 #encode to ascii
    newString=re.sub(r'@[A-Za-z0-9]+','',newString)                                #removing user mentions
    letters_only = re.sub("[^a-zA-Z]", " ", newString)                             #Fetching out only ascii characters
    letters_onl = re.sub('(www|http)\S+', '', letters_only)                        #removing links
    lower_case = letters_onl.lower()                                               #converting everything to lowercase
    words = tok.tokenize(lower_case)                                               #tokenize and join together to remove white spaces between words
    rs = [word for word in words if word not in stopword]                           #remove stopwords
    long_words=[]
    for i in rs:
      if len(i)>3:                                                 #removing short words
        long_words.append(lemmatizer.lemmatize(i))                 #converting words to lemma
    return (" ".join(long_words)).strip()      
cleaned_tweets = []
for t in df.Hindi:
  cleaned_tweets.append(tweet_cleaner(t))
df['cleaned_tweets']=cleaned_tweets #creating new dataframe
cleaned_tweets[:5]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_tweets'], df['task1'], test_size=0.33, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tvec = TfidfVectorizer(stop_words='english') #max_features=Max no. of words to be considered, min-df=Min frequency of word 
tfidf = tvec.fit_transform(X_train)
tfidf.shape
#dev set

vecc2=TfidfVectorizer(vocabulary=tvec.get_feature_names())
tfidf3 = vecc2.fit_transform(X_test)
print(tfidf3.shape)  
type(tfidf)
from multi_imbalance.utils.plot import plot_cardinality_and_2d_data
plot_cardinality_and_2d_data(tfidf.toarray(), np.array(df.task1), 'Plotting')
from multi_imbalance.resampling.soup import SOUP
3+1
mdo = SOUP(maj_int_min={
        'maj': ['NOT'],
        'min': ['HOF']
    })
X_train_res, y_train_res = mdo.fit_resample(tfidf.toarray(), np.array(y_train))
len(X_train_res)
import numpy
numpy.save('geekfile', X_train_res) 
numpy.save('geekfiley', y_train_res) 
from multi_imbalance.utils.plot import plot_visual_comparision_datasets
plot_visual_comparision_datasets(tfidf.toarray(), y_train, X_train_res, y_train_res, 'Raw Data', 'After processing')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


lr = LogisticRegression(solver='lbfgs')
lr_fit =lr.fit(X_train_res, y_train_res)

prediction = lr_fit.predict(tfidf3)
#prediction = label_encoder.inverse_transform(prediction)
#print(classification_report(df1.category,prediction))
len(prediction)
from sklearn.metrics import classification_report
print(classification_report(prediction,y_test))
import joblib
joblib_file = "joblib_LR_Model1.pkl"  
joblib.dump(lr_fit, joblib_file)
joblib_LR_model = joblib.load('./joblib_LR_Model1.pkl')


joblib_LR_model

from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train_res, y_train_res) 

from sklearn.metrics import accuracy_score
accuracy_score(y_train_res, gnb.predict(X_train_res))
gnb_predictions = gnb.predict(tfidf3.toarray()) 
print(classification_report(gnb_predictions,y_test))
#gnb_predictions = label_encoder.inverse_transform(gnb_predictions)
#print(classification_report(df1.category,gnb_predictions))
from sklearn.model_selection import GridSearchCV
from sklearn import svm
parameters = {'kernel':('linear', 'rbf')}
svr = svm.SVC(verbose=2)
SVM = GridSearchCV(svr, parameters, verbose=2)

SVM.fit(X_train_res, y_train_res)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(tfidf3.toarray())
print(classification_report(predictions_SVM,y_test))

from sklearn.ensemble import RandomForestClassifier 
rf=RandomForestClassifier(n_estimators=50, criterion='entropy')
rf.fit(X_train_res, y_train_res)

y_pred = rf.predict(tfidf3.toarray())
print(classification_report(y_test, y_pred))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


lr = LogisticRegression(solver='liblinear')
lr_fit =lr.fit(X_train_res, y_train_res)
predictionlr = lr_fit.predict(tfidf3.toarray())
#print(classification_report(y_train,predictionlr))
print(classification_report(y_test,predictionlr))

