import pandas as pd

from keras.datasets import imdb

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD
import numpy as np

# save np.load

np_load_old = np.load



# modify the default parameters of np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# call load_data with allow_pickle implicitly set to true

(X_train,Y_train),(X_test,Y_test)=imdb.load_data(num_words=50000)
word_index = imdb.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def getReview(index):

    decoded_review = ' '.join([reverse_word_index.get(i - 3,'') for i in index])

    tokeniser=RegexpTokenizer('\w+')

    token=tokeniser.tokenize(decoded_review.lower())

    stopWordList=stopwords.words('english')

    list_refined=[word for word in token if word not in stopWordList]

    wordList=[WordNetLemmatizer().lemmatize(word,pos='v') for word in list_refined]

    return " ".join(wordList)
df_train=pd.DataFrame(columns=['review','target'])

df_train['review']=[getReview(x) for x in X_train]

df_train['target']=Y_train

tf_fit=TfidfVectorizer().fit(df_train['review'])

tf_train=tf_fit.transform(df_train['review'])
df_test=pd.DataFrame(columns=['review','target'])

df_test['review']=[getReview(x) for x in X_test]

df_test['target']=Y_test

tf_test=tf_fit.transform(df_test['review'])
svd=TruncatedSVD(n_components=1000,algorithm='randomized')

svd_fit=svd.fit(tf_train)

svd_train=svd_fit.transform(tf_train)

svd_test=svd_fit.transform(tf_test)
from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.optimizers import RMSprop,SGD

from keras.losses import binary_crossentropy

from keras.metrics import binary_accuracy

from keras.regularizers import l2,l1
network=Sequential()

network.add(Dense(256,activation='relu',input_shape=(svd_train.shape[1],)))

network.add(Dropout(0.01))

network.add(Dense(512,activation='relu'))



network.add(Dense(512,activation='relu'))



network.add(Dense(1,activation='sigmoid'))



network.compile(loss='binary_crossentropy',optimizer=SGD(lr=0.005,momentum=0.9),metrics=['binary_accuracy'])

network.fit(svd_train,Y_train,epochs=55,batch_size=256)
network.evaluate(svd_test,Y_test)