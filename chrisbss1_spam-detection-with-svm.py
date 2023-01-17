import pandas as pd



df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')

df.head()
# Removing of empty columns



data = df[[i for i in df.columns if i not in ['Unnamed: 2','Unnamed: 3','Unnamed: 4']]]



# Renamming of columns



data = data.rename(columns={'v1': 'spam', 'v2': 'text'})



# Label encoding of text status



data.spam.replace(['spam', 'ham'], [1, 0], inplace=True)





data.head()
import matplotlib.pyplot as plt



spam= len(data[data['spam']==0])

ham= len(data[data['spam']==1])



y= [spam,ham]

x= ['Spam','Ham']



plt.xlabel('')

plt.ylabel('Number of people')

plt.bar(x,y,width=0.20)

plt.show()
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

nltk.download('punkt')



data_token = data



data_token['text'] = data['text'].apply(word_tokenize)
data_token.head()
import numpy as np

from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

  

lemmatizer = WordNetLemmatizer()





lem_list = []



for text_token in data_token['text']:

  text_lem = [lemmatizer.lemmatize(i) for i in text_token]

  lem_list.append(text_lem)

  

data_lemma = data_token



data_lemma['text'] = lem_list
from sklearn.feature_extraction.text import TfidfVectorizer



TfidfVectorizer = TfidfVectorizer(stop_words='english')





text_lem = [''.join(i) for i in data_lemma['text']]



data_vect = data_lemma



data_vect['text'] = text_lem



data_vect = TfidfVectorizer.fit_transform(data_vect['text'])



from sklearn.model_selection import train_test_split





xtrain, xtest, ytrain, ytest = train_test_split(data_vect, data['spam'], train_size=0.7, random_state=1)
from sklearn.svm import SVC



model = SVC(kernel = 'rbf', C = 1000,gamma=10, probability=True, random_state=1)

model.fit(xtrain, ytrain)
from sklearn import metrics



acc_train =model.score(xtrain,ytrain)

acc_test = model.score(xtest,ytest)





fpr, tpr, _ = metrics.roc_curve(np.array(ytrain), model.predict_proba(xtrain)[:,1])

auc_train = metrics.auc(fpr,tpr)



fpr, tpr, _ = metrics.roc_curve(np.array(ytest), model.predict_proba(xtest)[:,1])

auc_test = metrics.auc(fpr,tpr)



results = pd.DataFrame(np.array([[acc_train,acc_test],[auc_train,auc_test]]), columns = ["Train sample", "Test sample"], index =["Accuracy","AUC"])



results