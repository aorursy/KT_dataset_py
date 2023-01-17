import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df.head()
df.shape
df.isnull().sum()
df.target.value_counts()
print('Number of unique keywords : ', df.keyword.nunique())
df.keyword.value_counts()
print('Number of unique locations :' , df.location.nunique())
df.drop('location',axis = 1 , inplace=  True)
df.dropna(inplace=True)
df.shape
df.isnull().sum()
df.reset_index(drop=True, inplace=True)
sns.countplot(data = df, x='target')
import nltk
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
def Lower (text):
  return text.lower()
def Tokenisation (text):
  return nltk.word_tokenize(text)
# Let's test it :
test = Tokenisation('Hello there. How! are you ? this super notebook is about nlp')
#Create a list of stopwords 
Stpwrd_List=stopwords.words('english')
def StopWordsAlphaText(tokenized_text):
  filtred_text=[]
  for word in tokenized_text:
  #strip punctuation
    word = word.strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    #check if the word starts with an alphabet
    val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
    #ignore if it is a stop word or val is none
    if((word not in Stpwrd_List) and (val is not None)):
      filtred_text.append(word)
  return filtred_text
StopWordsAlphaText(test)
from nltk.corpus import wordnet
tag_dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return tag_dict.get(tag, wordnet.NOUN)
lemmatizer = WordNetLemmatizer()
def Lemmetizer(tokens):
  lemmetized_text=[]
  for word in tokens:
    word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
    lemmetized_text.append(word)
  return lemmetized_text
PosStem = PorterStemmer ()
def Stemmer (tokens):
  stemmed_text=[]
  for word in tokens :
    word = PosStem.stem(word)
    stemmed_text.append(word)
  return stemmed_text
df.text = df.text.apply(Lower)
df.text = df.text.apply(Tokenisation)
df.text = df.text.apply(StopWordsAlphaText)
df.text = df.text.apply(Lemmetizer)
df.head()
df.head()
real=""
fake=""
for index,row in df.iterrows():
  text = " ".join(row["text"])
  if(row["target"]==1):
    real=real+" "+text
  else:
    fake=fake+" "+text
#Create a real_tweets wordcloud
wordcloud_real=WordCloud(max_font_size=100, max_words=100, background_color="white").generate(real)
#Create a real_tweets wordcloud
wordcloud_fake=WordCloud(max_font_size=100, max_words=100, background_color="white").generate(fake)
plt.figure(figsize=(15,15))
plt.imshow(wordcloud_real, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud of positive Reviews")
plt.show()
plt.figure(figsize=(15,15))
plt.imshow(wordcloud_fake, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud of positive Reviews")
plt.show()
df.text = df.text.apply(lambda x: " ".join(x))
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
X = df.drop(['target','id'],axis = 1)
y = df.target
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 42)
count_vect = CountVectorizer()
#Let's see what it does exactly
Example = [ 'this is the first line.',
           'this line is the second line.',
           'and this is the third.',
           'is this the fifth line?']
Vect_Cv = count_vect.fit_transform(Example)
count_vect.get_feature_names()
Vect_Cv
Vect_Cv.toarray()
Tfidf = TfidfTransformer()
Vect_Tfidf = Tfidf.fit_transform(Vect_Cv)
Vect_Tfidf
Vect_Tfidf.toarray()
from sklearn.feature_extraction.text import TfidfVectorizer
tfvect = TfidfVectorizer()
tfvect.fit_transform(Example).toarray()
text_transformer = Pipeline([
    ('CountVectorizer', CountVectorizer()),
    ('TfidfTransformer',TfidfTransformer())
          ])
Preprocessing = ColumnTransformer([
    ("text features", text_transformer, 'text'),
    ('categorical_features', OrdinalEncoder(),['keyword'])
                                   ])
X_train = Preprocessing.fit_transform(x_train)
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
import time
names = ["Gaussian Naive Bayes", "Multinomial Naive Bayes","Bernoulli Naive Bayes" , "Random Forest Classifier"]
classifiers = [GaussianNB(),
               MultinomialNB(),
               BernoulliNB(),
               RandomForestClassifier()
               ]
for name, clf in zip(names, classifiers):
  #Cross validation prediction, and we measure fitting time 
  start = time.time()
  preds = cross_val_predict(clf,X_train.toarray(),y_train,cv=3)
  end = time.time()
  #Metrics
  acc = accuracy_score(y_train,preds)
  precision = precision_score(y_train,preds)
  recall = recall_score(y_train,preds)
  f1 = f1_score(y_train,preds)
  cm = confusion_matrix(y_train,preds)
  #Printing results
  print (name, 'Accuracy  :  ', "%.2f" %(acc*100),'%', ', Precision',"%.3f" %precision, 'Recall :' , "%.3f" %recall ,'F1-Score : ',"%.3f" %f1)
  print('The confusion Matrix : ' )
  print(cm)
  #Now we check how long did it take
  print('Time used :', "%.3f" %(end - start), 'seconds')
  print(' *-----------------------------------------------------------------------------------------------------*')
from sklearn.model_selection import GridSearchCV
Grid_par = [
            {'alpha' : [0,0.5,1,1.5], 'fit_prior' : [True, False]}
            ]
model = BernoulliNB()
GridSearch = GridSearchCV(estimator= model , param_grid=Grid_par, cv = 5,
                         scoring='accuracy', return_train_score=True)
GridSearch.fit(X_train,y_train)
results = GridSearch.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print ("%.3f" %(mean_score*100),'% | Parameters : ',params) 
GridSearch.best_estimator_
final_model = GridSearch.best_estimator_
Final_Pipeline = Pipeline([
     ('Preprocessing', Preprocessing),
     ('clf', final_model)
])
Final_Pipeline.fit(x_train,y_train)
Preds = Final_Pipeline.predict(x_test)
print('Final model accuracy : ', "%.3f" %(accuracy_score(y_test,Preds)*100), '%')