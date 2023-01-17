!pip install --upgrade gensim
import warnings

warnings.filterwarnings('ignore')



from keras.models import Sequential, Model

from keras.layers import Dense, LSTM,Dropout, BatchNormalization, UpSampling2D, Conv2D, Input, MaxPooling2D, Conv1D

from keras.layers import MaxPooling1D, Activation, Subtract, Multiply, Lambda, GlobalMaxPooling1D, GlobalMaxPooling2D

from keras.utils import plot_model

from keras import backend as K

import gensim as gensim

from gensim.models.word2vec import Word2Vec

from gensim.models.fasttext import FastText

from collections import defaultdict

import multiprocessing

from keras.models import load_model

import numpy as np

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau

from sklearn.metrics import mean_absolute_error, mean_squared_error

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

import time



"import"
train_csv = pd.read_csv('../input/home-depot-product-search/train.csv',delimiter = ",", encoding="latin-1")

descriptions_csv = pd.read_csv('../input/home-depot-product-search/product_descriptions.csv')

attributes_csv = pd.read_csv('../input/home-depot-product-search/attributes.csv')

solution_csv = pd.read_csv('../input/home-depot-product-search/solution.csv')

test_csv = pd.read_csv('../input/home-depot-product-search/test.csv', encoding="latin-1")

solution_csv = pd.read_csv('../input/home-depot-product-search/solution.csv')

'load data'
train_csv
descriptions_csv
attributes_csv
solution_csv
test_csv
solution_csv
solution_csv[solution_csv.relevance != -1]
train_desc_data = pd.merge (train_csv,descriptions_csv,on="product_uid")

train_desc_data
test_desc_data = pd.merge(test_csv,descriptions_csv, on="product_uid")

test_desc_data
### Collect all text and pre-process

### we refer to each row of data as another entry in our corpus

### names and serach terms aren't tokenized 



sentences = []

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",

             "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",

             "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",

             "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",

             "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",

             "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",

             "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",

             "will", "just", "don", "should", "now"]



def row_tokenize (row):

  all_row = row.replace(',',' ').replace('.',' ').replace("("," ").replace(")"," ").replace('"',' ').replace("'",' ')

  return [word for word in all_row.lower().split(' ') if (word not in stop_words) and (word != " ") and (word != "")]



for _,row in train_desc_data.iterrows():

  to_append = []

  to_append.append(row.search_term.lower())

  to_append.append(row.product_title)

  all_row = row.search_term +" "+row.product_title +" "+  row.product_description

  to_append = to_append + row_tokenize(all_row)

  sentences.append(to_append)



len(sentences), type(sentences[0]), type(sentences[0][0]), sentences[0]
### create word embeddings

# gensim_model = Word2Vec(sentences, min_count=1, size=50, workers=multiprocessing.cpu_count(), window=20, sg = 1) # sg 0 -> cobow, 1 -> skip gram

# gensim_model = FastText(sentences, min_count=1, size=50, workers=multiprocessing.cpu_count(), window=20, sg = 1) # sg 0 -> cobow, 1 -> skip gram

# gensim_model.save('gensim_embedding_fast_text.model')

# gensim_model = Word2Vec.load('gensim_embedding.model')

gensim_model = FastText.load('../input/fast-text-embeddings/gensim_embedding_fast_text.model')
gensim_model['nails'].shape
print(gensim_model.similarity('Simpson Strong-Tie 12-Gauge Angle','angle bracket'))

print(gensim_model.similarity('Simpson Strong-Tie 12-Gauge Angle','24 whtie storage cabinet'))

print(gensim_model.similarity('Simpson Strong-Tie 12-Gauge Angle','angles'))

print(gensim_model.similarity('angles','angle'))
train,val = train_test_split(train_desc_data,train_size=0.8, random_state=42)

train.shape, val.shape
## convert train data to expected embedding

serach_line_length = 25

product_line_length = 650



#### split to train and test



train_data, validation_data= train_test_split(train_desc_data,train_size=0.8, random_state=42)





def gen_generator(batch_size,dataset):

  y_train = []

  x_train_product = []

  x_train_search = []

  while True:

    for _,row in dataset.iterrows():

      y_train.append(row.relevance)



      x_search_row = [gensim_model[row.search_term.lower()]] + [gensim_model[token] for token in row_tokenize(row.search_term)]

      ## pad x_search_row to be size of serach_line_length and each element is array size of 50

      if len(x_search_row) > serach_line_length:

        x_serach_row = x_search_row[:serach_line_length]

      else:

        for _ in range(serach_line_length - len(x_search_row)):

          x_search_row.append(np.zeros(50))

      x_train_search.append(x_search_row)



      all_product = row.product_title + " " + row.product_description

      x_product_row = [gensim_model[row.product_title]] + [gensim_model[token] for token in row_tokenize(all_product)]

      ## pad x_product_row to be size of serach_line_length and each element is array size of 50

      if len(x_product_row) > product_line_length:

        x_product_row = x_product_row[:product_line_length]

      else:

        for _ in range(product_line_length - len(x_product_row)):

          x_product_row.append(np.zeros(50))

      x_train_product.append(x_product_row)

        

      if (len(y_train) == batch_size):

        yield [np.array(x_train_search),np.array(x_train_product)],np.array(y_train)

        y_train = []

        x_train_product = []

        x_train_search = []

        





def train_generator(batch_size=64):

  gen = gen_generator(batch_size,train_data)

  while True:

    yield next(gen)



def validation_generator(batch_size = 64):

  gen = gen_generator(batch_size,validation_data)

  while True:

    yield next(gen)





def test_generator(batch_size = 1):

  x_test_product = []

  x_test_search = []

  while True:

    for idx,row in test_desc_data.iterrows():

      #### skip any ignored results

      if solution_csv.relevance[idx] == -1:

        continue



      search_row = [gensim_model[row.search_term.lower()]] + [gensim_model[token] for token in row_tokenize(row.search_term)]

      if len(search_row) > serach_line_length:

        serach_row = search_row[:serach_line_length]

      else:

        for _ in range(serach_line_length - len(search_row)):

          search_row.append(np.zeros(50))

      x_test_search.append(search_row)



      prodcut_full_desc =  row.product_title + " " + row.product_description

      product = [gensim_model[row.product_title]] + [gensim_model[token] for token in row_tokenize(prodcut_full_desc)]

      if len(product) > product_line_length:

        product = product[:product_line_length]

      else:

        for _ in range(product_line_length - len(product)):

          product.append(np.zeros(50))

      x_test_product.append(product)



      if len(x_test_product) == batch_size :

        yield [np.array(x_test_search),np.array(x_test_product)]

        x_test_search = []

        x_test_product = []



"generator"
g_train = train_generator()

g_val = validation_generator()

test_gen = test_generator()



for idx in range(10):

  x = next(g_train)

  if x[0][0].shape != (64,25,50) or x[0][1].shape != (64,product_line_length,50) or x[1].shape != (64,):

    print ("error train ", idx)

    print (len(x), len(x[0]),len(x[1]), x[0][0].shape, x[0][1].shape, x[1].shape)

    print ("\n")

    break

  x = next(g_val)

  if x[0][0].shape != (64,25,50) or x[0][1].shape != (64,product_line_length,50) or x[1].shape != (64,):

    print ("error val ", idx)

    print (len(x), len(x[0]),len(x[1]), x[0][0].shape, x[0][1].shape, x[1].shape)

    print ("\n")

    break

  x = next (test_gen)

  if x[0].shape != (1,25,50) or x[1].shape != (1,product_line_length,50):

    print ("error test ", idx)

    print(len(x),  x[0].shape, x[1].shape)

    print ("\n")

    break



x = next(g_train)

print(len(x), len(x[0]),len(x[1]), x[0][0].shape, x[0][1].shape, x[1].shape)

x = next(g_val)

print(len(x), len(x[0]),len(x[1]), x[0][0].shape, x[0][1].shape, x[1].shape)

x = next(test_gen)

print(len(x),  x[0].shape, x[1].shape)
def siamese_model(inp_dim = (128,1),dim = 64,dim_max= 100):

  inp = Input(inp_dim)

  x = Conv1D(dim,10,activation='relu')(inp)

  x = GlobalMaxPooling1D()(x)

  x = Activation('relu')(x)

  x = Dense(128, activation="relu")(x)

  return Model(inputs = inp, outputs = x,name = "siamese_model")



model = siamese_model()

model.summary()

plot_model(model)



def get_full_model():

  inp_product = Input((product_line_length,50))

  inp_search = Input((serach_line_length,50))

  lstm_product = LSTM(128)(inp_product)

  lstm_search = LSTM(128)(inp_search)

  reshape_layer = Lambda(lambda tensor: tensor[...,np.newaxis],name="reshape_lstm_output")

  reshape_product = reshape_layer(lstm_product)

  reshape_search = reshape_layer(lstm_search)

    

  model = siamese_model()



  encoded_l = model(reshape_product)

  encoded_r = model(reshape_search)

  diff = Subtract(name="subtract_layer")([encoded_l, encoded_r])

  ## calculate euclidian distance

  mult = Multiply()([diff,diff])

  add =  Lambda(lambda tensors: K.sum(tensors,axis=1,keepdims=True))(mult)

  norm =  Lambda(lambda tensors: K.sqrt(tensors), name="norm")(add)



  return Model(inputs=[inp_search,inp_product],outputs=norm, name="lstm_siamese_network")



full_model = get_full_model()

full_model.summary()

plot_model(full_model)
def set_callbacks(description='run1',patience=13,tb_base_logdir='./logs/'):

    cp = ModelCheckpoint('full_gensim_checkpoints.h5',save_best_only=True)

    # modelCheckpoint = ModelCheckpoint('best_model.hdf5', save_best_only = True)

    es = EarlyStopping(patience=patience,monitor='val_loss')

    rlop = ReduceLROnPlateau(patience=4,monitor = 'val_loss')   

    # tb = TensorBoard(log_dir='{}{}'.format(tb_base_logdir,description))

    #cycl = CyclicLR(max_lr=0.03,step_size=5000)

    cb = [cp,es,rlop]

    return cb
def rmse(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true)))
batch_size = 64

full_model.compile(optimizer = 'adam', loss ='mae', metrics=[rmse])

history = full_model.fit(train_generator(batch_size), epochs = 30, batch_size=None, steps_per_epoch= (59253 //(batch_size))  ,

               validation_data = validation_generator(), validation_steps= (14814//(batch_size)),

               callbacks=set_callbacks() )



full_model.save('full_model_gensim.h5')
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Mean Absolute Error')

plt.ylabel('MAE')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history.history['rmse'])

plt.plot(history.history['val_rmse'])

plt.title('model RMSE')

plt.ylabel('RMSE')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')
full_model = load_model('../input/modelweights/full_gensim_checkpoints.h5', {'rmse': rmse})

"loaded"
def get_score(amount,generator,results):

    steps=32

    predictions = full_model.predict(generator(steps),steps=amount//steps)

    return mean_squared_error(results[:predictions.shape[0]],predictions, squared=False), mean_absolute_error(results[:predictions.shape[0]],predictions)
test_solutions = solution_csv.relevance[solution_csv['relevance'] != -1]

train_solutions = train.relevance

validation_solutions =val.relevance 



test_solutions.shape, train_solutions.shape, validation_solutions.shape
score_test = get_score(112067,test_generator,test_solutions)

print(1)

score_train = get_score(59253,train_generator,train_solutions)

print(2)

score_validation = get_score(14814,validation_generator,validation_solutions)



print("Siamese Model Score on Test:")

print("RMSE: ", score_test[0], " MAE: ", score_test[1])



print("Siamese Model Score on Training Set:")

print("RMSE: ", score_train[0], " MAE: ", score_train[1])



print("Siamese Model Score on Validation Set:")

print("RMSE: ", score_validation[0], " MAE: ", score_validation[1])
feature_extractor_model = Model(inputs=full_model.inputs,outputs=full_model.get_layer("subtract_layer").output)

feature_extractor_model.summary()

plot_model(feature_extractor_model)
def create_train_features():

    train_features = []

    batch_size = 32

    train_gen = train_generator(batch_size)

    for i in range(59253 //(batch_size)):

        preds = feature_extractor_model.predict(next(train_gen))

        if (len(train_features) == 0):

            train_features = preds

        else:

            train_features = np.concatenate((train_features,preds))



    return train_features



train_features = create_train_features()

train_features.shape
def create_validation_features():

    validation_features = []



    batch_size = 32

    validation_gen = validation_generator(batch_size)

    for i in range(14814 //(batch_size)):

        preds = feature_extractor_model.predict(next(validation_gen))

        if (len(validation_features) == 0):

            validation_features = preds

        else:

            validation_features = np.concatenate((validation_features,preds))

    return validation_features



validation_features = create_validation_features()

validation_features.shape
def create_test_features():

    test_features = []

    batch_size = 32

    test_gen = test_generator(batch_size)

    for i in range(112067 //(batch_size)):

        preds = feature_extractor_model.predict(next(test_gen))

        if (len(test_features) == 0):

            test_features = preds

        else:

            test_features = np.concatenate((test_features,preds))



    return test_features





test_features = create_test_features()

test_features.shape
def get_ml_scores(model):

    train_preds = model.predict(train_features)

    validation_preds = model.predict(validation_features)

    test_preds = model.predict(test_features)

    

    scores = []

    scores.append((mean_squared_error(train_solutions[:train_features.shape[0]],train_preds,squared=False ), mean_absolute_error(train_solutions[:train_features.shape[0]],train_preds)))

    scores.append((mean_squared_error(validation_solutions[:validation_features.shape[0]],validation_preds, squared=False), mean_absolute_error(validation_solutions[:validation_features.shape[0]],validation_preds)))

    scores.append((mean_squared_error(test_solutions[:test_features.shape[0]],test_preds, squared=False), mean_absolute_error(test_solutions[:test_features.shape[0]],test_preds)))



    return scores

    
sgd = SGDRegressor()

start = time.time()

sgd.fit(train_features,train_solutions[:train_features.shape[0]])

end = time.time()

print("time for training: ",(end-start))
scores = get_ml_scores(sgd)



print("SGD Score on Test:")

print("RMSE: ", scores[2][0], "MAE: ",  scores[2][1])



print("SGD Score on Training Set:")

print("RMSE: ", scores[0][0], "MAE: ",  scores[0][1])



print("SGD Score on Validation Set:")

print("RMSE: ", scores[1][0], "MAE: ",  scores[1][1])
xgboost = GradientBoostingRegressor(random_state=42)

start = time.time()

xgboost.fit(train_features,train_solutions[:train_features.shape[0]])

end = time.time()

print("time for training: ",(end-start))
scores = get_ml_scores(xgboost)



print("xgboost Score on Test:")

print("RMSE: ", scores[2][0], "MAE: ",  scores[2][1])



print("xgboost Score on Training Set:")

print("RMSE: ", scores[0][0], "MAE: ",  scores[0][1])



print("xgboost Score on Validation Set:")

print("RMSE: ", scores[1][0], "MAE: ",  scores[1][1])
xgbr = XGBRegressor()

start = time.time()

xgbr.fit(train_features, train_solutions[:train_features.shape[0]])

end = time.time()

print("time for training: ",(end-start))
scores = get_ml_scores(sgd)



print("XGB Regressor Score on Test:")

print("RMSE: ", scores[2][0], "MAE: ",  scores[2][1])



print("XGB Regressor Score on Training Set:")

print("RMSE: ", scores[0][0], "MAE: ",  scores[0][1])



print("XGB Regressor Score on Validation Set:")

print("RMSE: ", scores[1][0], "MAE: ",  scores[1][1])