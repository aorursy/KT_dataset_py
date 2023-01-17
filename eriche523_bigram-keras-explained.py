import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import keras 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras import layers

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.wrappers.scikit_learn import KerasClassifier

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Dense, Dropout, Embedding, Flatten

from keras.models import Model

from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report





print(pd.__version__)
# read in csv file directly from Keras public IMBD dataset



df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv", names=['comment', 'label'], header=0, encoding='utf-8')

df=pd.DataFrame(df)
df
# df=df1.sample(n=50000, random_state=23)

# Take a sample if necessary; for faster training



#check data format

print(df.iloc[0])



#check the labels in the dataset

df.label.value_counts()



#converts label into integer values

df['label'] = df.label.astype('category').cat.codes



#prints out dataframe

df
#creates a new column for total words in each row, because we want to know the matrix dimenison.

df['total_words'] = df['comment'].str.count(' ') + 1



#prints the dataframe at the index of it's longest review

print(df.loc[df.total_words.idxmax()])



#prints the length of the longest view in the dataset

print("\nThe longest comment is " + str(df['total_words'].max()) + " words.\n")
# plot word frequency



plt.figure(figsize=(15, 9))

plt.hist([(df['total_words'])],bins=100,color = "blue")

plt.xlabel('Number of word in a review')

plt.ylabel('Frequency')

plt.title('Review Words Count Distribution')

plt.show()
# counts the number of classes

# label the target feature



num_class = len(np.unique(df.label.values))

y = df['label'].values

print("\nThere are a total of " + str(num_class) + " classes.")



# evenly distributed dataset

df.groupby('label').count()
# import CountVectorizer to find common words used in the dataset

# the default CountVectorizer is unigram, which is what is needed in this situation to count the frequency of each words in the dataset



from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()



# transform and vectorize the comments in a sparse matrix. The dimenion length is the number of instances, the width of the matrix is the

# number of words in total corpse. The words that are not in the each review is padded with 0, hence the sparse matrix.



vect_texts = vectorizer.fit_transform(list(df['comment']))

all_ngrams = vectorizer.get_feature_names()



# display top min(50,len(all_ngrams) of the most frequent words

num_ngrams = min(50, len(all_ngrams))



# count the number of words in the total corpse

all_counts = vect_texts.sum(axis=0).tolist()[0]



# loop the words(features) with counts using zip function

all_ngrams, all_counts = zip(*[(n, c) for c, n in sorted(zip(all_counts, all_ngrams), reverse=True)])

ngrams = all_ngrams[:num_ngrams]

counts = all_counts[:num_ngrams]



idx = np.arange(num_ngrams)



# Let's now plot a frequency distribution plot of the most seen words in the corpus.

plt.figure(figsize=(15, 15))

plt.bar(idx, counts, width=0.8, color = "blue")

plt.xlabel('N-grams')

plt.ylabel('Frequencies')

plt.title('Frequency distribution of ngrams')

plt.xticks(idx, ngrams, rotation=45)

plt.show()

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest, chi2,f_classif, mutual_info_classif



# import a list of stopwords, when applied the stopwords in the corpose will not be tokenized, they will be skipped

stopwords=stopwords.words('english')



# min_df=2, any words that occurs less than 2 times in the total corpse will not be tokenized



vectorizer = TfidfVectorizer(min_df=3, binary=True, analyzer='word',ngram_range= (1,2), stop_words=stopwords)

df_bigram = vectorizer.fit_transform(df['comment'])

# There a few ways of selecting the number of words to be included in the training set

# you can use Chi2, f_classif

# specify the k features to be included in the training

# the k features (the width of the matrix) will correspond to the number of unique words and/or phrases (if using bigram or trigram etc)

# to be included in the the training set

# Chi2, f_classif and mutual_info_classif with all the same k-value will produce different sets of words to be included in the training dataset based on their calculation.

# another way of putting it is that the dimensions are the same, but the sparseness or the number of features selected to be a part of the 1800 features will be different

# personally I prefer chi2 (it's a measurement of feature dependency to the target, the higher the value the more relevant) as it produces a more dense matrix than f_classif



k=26000



selector = SelectKBest(chi2, k=min(k, df_bigram.shape[1]))

selector.fit(df_bigram, df.label)

transformed_texts = selector.transform(df_bigram).astype('float32')
# there are 3.2 million elements (non zero elemets) in the matrix

# they are words and phrases that are tokenized with idf weighting, this directly impacts training accuracy and speed

transformed_texts
# keep the matrix compressed in this case, you will get memorry error if you are trying to print the array. Kaggle only allocates 8gb of ram

transformed_texts=transformed_texts.toarray()
#train test split

X_train, X_test, y_train, y_test = train_test_split(transformed_texts, y, test_size=0.3)
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.layers import LSTM

from keras import optimizers



#builds input shape



# This a fully connected network with number of total parameters = 1,668,354   

# 2 hidden layers, with a droput of 0.5

# relu is the activation function, sigmoid also works well in this case since there are not too many layers to diminish the learn rate

# output function is sigmoid which is the same if you use softmax in a binary situation

# training on entropy, RMS and accuracy



max_features = min(k, df_bigram.shape[1])



model = Sequential()

model.add(Dense(64, input_dim=max_features, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='sigmoid'))





model.compile(loss='binary_crossentropy',

              optimizer='RMSprop',

              metrics=['acc'])





# prints out summary of model

model.summary()



# saves the model weights

# the train/validation/test split is 26500/8750/15000, it's essential to hold out a decent chunck of unseen data

filepath="weights-simple.hdf5"

checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 

          shuffle=True, epochs=15, callbacks=[checkpointer])
#plot model

df_result = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc'], 'validation_accuracy': history.history['val_acc']})

g = sns.pointplot(x="epochs", y="accuracy", data=df_result, fit_reg=False)

g = sns.pointplot(x="epochs", y="validation_accuracy", data=df_result, fit_reg=False, color='green')
#get prediction accuarcy for testing dataset 15000 samples

predicted = model.predict(X_test)

predicted_best = np.argmax(predicted, axis=1)

print (accuracy_score(predicted_best, y_test))

predicted=pd.DataFrame(data=predicted)
print (classification_report(predicted_best, y_test))