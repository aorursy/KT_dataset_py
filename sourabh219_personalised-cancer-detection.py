import pandas as pd #Importing all the needed libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
data=pd.read_csv('../input/perosonalisedcancerdetection/training_variants') #Importing the data
text_data=pd.read_csv('../input/perosonalisedcancerdetection/training_text',sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
test_data=pd.read_csv('../input/perosonalisedcancerdetection/stage2_test_variants.csv') #Importing the data
test_text_data=pd.read_csv('../input/perosonalisedcancerdetection/stage2_test_text.csv',sep="\|\|",engine="python",names=["ID","TEXT"])
test_data=test_data.join(test_text_data,lsuffix='_')
data.shape #data has 3321 data points
data.columns  #has following features where class is dependent variable
text_data.shape #Text ahs 2 features one is id and other is actual text.
data=data.join(text_data,lsuffix='_') #Joining the data.
test_data.shape
data[data['Variation']=='Truncating Mutations']['Class'].value_counts() #Just a insight that specific type of variation belongs to which group.
data.Class.value_counts() #Total classes 1-9.(9 classes)
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
enc1=OneHotEncoder() #importing the encoders
#all_Genes=list(data['Gene']+test_data['Gene'])
Gene_vocab=pd.concat([test_data['Gene'],data['Gene']],ignore_index=True)
enc.fit(Gene_vocab.values.reshape(-1,1))
Variation_vocab=pd.concat([test_data['Variation'],data['Variation']],ignore_index=True)
enc1.fit(Variation_vocab.values.reshape(-1,1))
test_data.shape
encoded=enc.transform(data['Gene'].values.reshape(-1,1))
add=data.shape[1]
for i in range(0,encoded.shape[1]):
    data[add] = pd.arrays.SparseArray(encoded[:, i].toarray().ravel())
    add+=1 #Converting the Gene feature as one hot encoded
encoded=enc1.transform(data['Variation'].values.reshape(-1,1))
add=data.shape[1]
for i in range(0,encoded.shape[1]):
    data[add] = pd.arrays.SparseArray(encoded[:, i].toarray().ravel())
    add+=1  #Converting the Gene feature as one hot encoded
def preprocess(x): #Basic pre-processing the text feature
        x=str(x).lower()
        x=x.replace(',000,000','m').replace(',000','k').replace("'","`").replace("won't",'will not').replace("can't","can not").replace('cannot','can not').replace("n't",'not').replace("what's","what is").replace("it's","it is").replace("'ve","have").replace("i'm","i am").replace("'re","are").replace("he's","he is").replace("she's","she is ").replace("'s","own").replace("%"," precent").replace("₹"," rupeee").replace("$","dollar").replace("€","euro").replace("'ll","will")
        porter = PorterStemmer()
        pattern = re.compile('\W')
    
        if type(x) == type(''):
            x = re.sub(pattern, ' ', x)
        if type(x) == type(''):
          x = porter.stem(x)
          example1 = BeautifulSoup(x)
          x = example1.get_text()
        return str(x)
data['text']=data['TEXT'].apply(preprocess) #applying the text pre-processing to each feature.
data.drop(columns=['Gene','Variation','ID','ID_'],inplace=True) #Dropping the already encoded features.
import gensim
# Load Google's pre-trained Word2Vec model.
w2v_model_our_corpus = gensim.models.KeyedVectors.load_word2vec_format(r'../input/word2vec-google/GoogleNews-vectors-negative300.bin', binary=True)  
def sent_vectorizer(sent, w2v_model_our_corpus): #Get the sentence vectors
    sent_vec = np.zeros(300)
    numw = 0
    for w in sent.split():
        try:
            sent_vec = np.add(sent_vec, w2v_model_our_corpus[w])
            numw+=1
        except:
            pass
    return sent_vec/ np.sqrt(sent_vec.dot(sent_vec))
encoded=enc.transform(test_data['Gene'].values.reshape(-1,1))
add=test_data.shape[1]
for i in range(0,encoded.shape[1]):
    test_data[add] = pd.arrays.SparseArray(encoded[:, i].toarray().ravel())
    add+=1 #Converting the Gene feature as one hot encoded
encoded=enc1.transform(test_data['Variation'].values.reshape(-1,1))
add=test_data.shape[1]
for i in range(0,encoded.shape[1]):
    test_data[add] = pd.arrays.SparseArray(encoded[:, i].toarray().ravel())
    add+=1  #Converting the Gene feature as one hot encoded
test_data['text']=test_data['TEXT'].apply(preprocess)
test_data.drop(columns=['Gene','Variation','ID','ID_'],axis=1,inplace=True) #Dropping the already encoded features.
from tqdm import tqdm #Converting the sentences into vectors. 
V=[]
for sentence in tqdm(data['text']):
    V.append(sent_vectorizer(sentence, w2v_model_our_corpus))
V_test=[]
for sentence in tqdm(test_data['text']):
    V_test.append(sent_vectorizer(sentence, w2v_model_our_corpus))
V_test=np.array(V_test)
y_train=data['Class']
data.drop(columns=['TEXT','text','Class'],axis=1,inplace=True) #dropping not needed features from train set
test_data.drop(columns=['TEXT','text'],axis=1,inplace=True) #dropping not needed features from test set
from scipy.sparse import hstack #Stacking one-hot encoded Gene and variation feature with the sentence vectors whihc we got from 
X_train=hstack([data,V]) #Pre-trained Word2Vec model. 
X_test=hstack([test_data,V_test])
from catboost import CatBoostClassifier  #Importing the classifier.
lr=CatBoostClassifier()
from sklearn.calibration import CalibratedClassifierCV
cr=CalibratedClassifierCV(lr)
cr.fit(X=X_train,y=y_train)
predsCalibrated=cr.predict_proba(X_test)
ids=np.arange(1,987)
ids=pd.Series(ids)
class1=pd.Series(predsCalibrated[:,0])
class2=pd.Series(predsCalibrated[:,1])
class3=pd.Series(predsCalibrated[:,2])
class4=pd.Series(predsCalibrated[:,3])
class5=pd.Series(predsCalibrated[:,4])
class6=pd.Series(predsCalibrated[:,5])
class7=pd.Series(predsCalibrated[:,6])
class8=pd.Series(predsCalibrated[:,7])
class9=pd.Series(predsCalibrated[:,8])
preds=pd.DataFrame(data={'ID':ids,'class1':class1,'class2':class2,'class3':class3,'class4':class4,'class5':class5,'class6':class6,'class7':class7,'class8':class8,'class9':class9})
preds.to_csv('Submissions.csv')