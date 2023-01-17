import numpy as np 
import pandas as pd 
import nltk
import re,string,unicodedata
import seaborn as sns
import gensim
import sklearn

from pandas import Series
from wordcloud import WordCloud,STOPWORDS
from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from gensim.models import word2vec, Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, BayesianRidge
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import ensemble
# get review and rating columns
review_path = 'bgg-13m-reviews.csv'

data = pd.read_csv(review_path, usecols=[2,3])
data.head()
# remove null comment
def remove_nan(data):
    data['comment']=data['comment'].fillna('null')
    data = data[~data['comment'].isin(['null'])]
    data = data.reset_index(drop=True)
    return data
data = remove_nan(data)
data.head()
data.describe()
#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')
#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
#Apply function on review column
data['comment']=data['comment'].apply(remove_between_square_brackets)
#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
data['comment']=data['comment'].apply(remove_special_characters)
#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
data['comment']=data['comment'].apply(simple_stemmer)
#set stopwords to english
stop=set(stopwords.words('english'))
print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
#Apply function on review column
data['comment']=data['comment'].apply(remove_stopwords)
data = remove_nan(data)
data.to_csv('data_after_remove_st.csv', header=False, index=False, encoding = 'utf-8')
columns = ['rate', 'comment']
data = pd.read_csv('data_after_remove_st.csv',names = columns)
data['comment'] = data.comment.str.lower()
data['document_sentences'] = data.comment.str.split('.') 
# data['tokenized_sentences'] = data['document_sentences']
data['tokenized_sentences'] = list(map(lambda sentences:list(map(nltk.word_tokenize, sentences)),data.document_sentences))  
data['tokenized_sentences'] = list(map(lambda sentences: list(filter(lambda lst: lst, sentences)), data.tokenized_sentences))
data.head()
# data.to_csv("data_after_pre.csv",sep=',',index=False, encoding = 'utf-8')
# data = pd.read_csv('data_after_pre.csv')
# Take the top 10k after random ordering
data = data.reindex(np.random.permutation(data.index))[:100000]
# split the data into training data and test data.
train, test, y_train, y_test = train_test_split(data, data['rate'], test_size=.2)
type(train.tokenized_sentences[993001])
#Collecting a vocabulary
voc = []
for sentence in train.tokenized_sentences:
    voc.extend(sentence)
#     print(sentence)

print("Number of sentences: {}.".format(len(voc)))
print("Number of rows: {}.".format(len(train)))
voc[:10]
# word2vector
num_features = 300    
min_word_count = 3     # Frequency < 3 will not be count in.
num_workers = 16       
context = 8           
downsampling = 1e-3   

# Initialize and train the model
W2Vmodel = Word2Vec(sentences=voc, sg=1, hs=0, workers=num_workers, size=num_features, min_count=min_word_count, window=context,
                    sample=downsampling, negative=5, iter=6)
model_voc = set(W2Vmodel.wv.vocab.keys()) 
print(len(model_voc))
# model save
W2Vmodel.save("Word2Vec2")
# model load
W2Vmodel = Word2Vec.load('Word2Vec2')
def sentence_vectors(model, sentence):
    #Collecting all words in the text
#     print(sentence)
    sent_vector = np.zeros(model.vector_size, dtype="float32")
    if sentence == [[]] or sentence == []  :
        return sent_vector
    words=np.concatenate(sentence)
#     words = sentence
    #Collecting words that are known to the model
    model_voc = set(model.wv.vocab.keys()) 
#     print(len(model_voc))

    # Use a counter variable for number of words in a text
    nwords = 0
    # Sum up all words vectors that are know to the model
    for word in words:
        if word in model_voc: 
            sent_vector += model[word]
            nwords += 1.

    # Now get the average
    if nwords > 0:
        sent_vector /= nwords
    return sent_vector
train['sentence_vectors'] = list(map(lambda sen_group:
                                      sentence_vectors(W2Vmodel, sen_group),
                                      train.tokenized_sentences))
test['sentence_vectors'] = list(map(lambda sen_group:
                                    sentence_vectors(W2Vmodel, sen_group), 
                                    test.tokenized_sentences))
def vectors_to_feats(df, ndim):
    index=[]
    for i in range(ndim):
        df[f'w2v_{i}'] = df['sentence_vectors'].apply(lambda x: x[i])
        index.append(f'w2v_{i}')
    return df[index]
X_train = vectors_to_feats(train, 300)
X_test = vectors_to_feats(test, 300)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
train.to_csv('train_w2v_100k.csv')
test.to_csv('test_w2v_100k.csv')
train = pd.read_csv('train_w2v_1000k.csv').drop(columns = 'Unnamed: 0')
test = pd.read_csv('test_w2v_1000k.csv').drop(columns = 'Unnamed: 0')
X_train = train.drop(columns = 'rate')
X_test = test.drop(columns = 'rate')
y_train = train.rate
y_test = test.rate
X_test
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
lr_y_predict=model_lr.predict(X_test)
y_test = np.array(y_test)
# (RMSE)
rmse = np.sqrt(mean_squared_error(y_test,lr_y_predict))

# (MAE)
mae = mean_absolute_error(y_test, lr_y_predict)

print('linear_regression_rmse = ', rmse)
print('linear_regression_mae = ', mae)
joblib.dump(model_lr, 'save/model_lr.pkl')

# model_lr = joblib.load('save/model_lr_1000k.pkl')
model_svm = SVR()
model_svm.fit(X_train, y_train)
svm_y_predict=model_svm.predict(X_test)
# (RMSE)
rmse = np.sqrt(mean_squared_error(y_test,svm_y_predict))

# (MAE)
mae = mean_absolute_error(y_test, svm_y_predict)

print('svm_rmse = ', rmse)
print('svm_mae = ', mae)
joblib.dump(model_lr, 'save/model_svm.pkl')

model_bayes_ridge = BayesianRidge()
model_bayes_ridge.fit(X_train, y_train)
bayes_y_predict = model_bayes_ridge.predict(X_test)
# (RMSE)
rmse = np.sqrt(mean_squared_error(y_test,bayes_y_predict))

# (MAE)
mae = mean_absolute_error(y_test, bayes_y_predict)

print('BayesianRidge_rmse = ', rmse)
print('BayesianRidge_mae = ', mae)
joblib.dump(model_bayes_ridge, 'save/model_bayes.pkl')

model_random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=20)
model_random_forest_regressor.fit(X_train, y_train)
random_forest_y_predict = model_random_forest_regressor.predict(X_test)
# (RMSE)
rmse = np.sqrt(mean_squared_error(y_test,random_forest_y_predict))

# (MAE)
mae = mean_absolute_error(y_test, random_forest_y_predict)

print('BayesianRidge_rmse = ', rmse)
print('BayesianRidge_mae = ', mae)
joblib.dump(model_random_forest_regressor, 'save/model_random_forest.pkl')

def predict(text):
    model_lr = joblib.load('save/model_lr.pkl')
    model_svm = joblib.load('save/model_svm.pkl')
    model_random_forest_regressor = joblib.load('save/model_random_forest.pkl')
    model_bayes_ridge = joblib.load('save/model_bayes.pkl')
    data = {'comment': Series(text)}
    data = pd.DataFrame(data)
    print(data)
    data['comment'] = data['comment'].apply(remove_between_square_brackets)
    data['comment'] = data['comment'].apply(remove_special_characters)
    data['comment'] = data['comment'].apply(simple_stemmer)
    data['comment'] = data['comment'].apply(remove_stopwords)

    data['comment'] = data.comment.str.lower()
    data['document_sentences'] = data.comment.str.split('.')
    data['tokenized_sentences'] = data['document_sentences']
    data['tokenized_sentences'] = list(
        map(lambda sentences: list(map(nltk.word_tokenize, sentences)), data.document_sentences))
    data['tokenized_sentences'] = list(
        map(lambda sentences: list(filter(lambda lst: lst, sentences)), data.tokenized_sentences))
    print(data)
    # sentence = data['tokenized_sentences'][0]
    W2Vmodel = Word2Vec.load("Word2Vec2")

    data['sentence_vectors'] = list(map(lambda sen_group:
                                        sentence_vectors(W2Vmodel, sen_group),
                                        data.tokenized_sentences))
    text = vectors_to_feats(data, 300)
    print(text)
    lr_y_predict = model_lr.predict(text)
    svm_y_predict = model_svm.predict(text)
    bayes_y_predict = model_bayes_ridge.predict(text)
    random_forest_y_predict = model_random_forest_regressor.predict(text)

    return lr_y_predict, svm_y_predict, random_forest_y_predict, bayes_y_predict

print(predict(["This is a great game.  I've even got a number of non game players enjoying it.  Fast to learn and always changing.",
        "This is a great game.  I've even got a number of non game players enjoying it.  Fast to learn and always changing."]))