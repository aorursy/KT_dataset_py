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
df= pd.read_csv('/kaggle/input/wine-reviews/winemag-data-130k-v2.csv')
### Data Analysis

df.shape
df.info()
##### Country that have NaN values

df[df['country'].isna()]
df.describe()
df.points.nunique()
df.price.nunique()
##### Taster Points and Price

df.points.hist(bins=10)
# We have some outliers
df.price.hist(bins=100)
import seaborn as sns
sns.boxplot(df['price'])
##### Groupby country

df_country=df[['country','winery']].groupby('country').count().sort_values('winery', ascending=False)
df_country_1= df_country[df_country['winery']>100]

df_country_1.plot(kind='bar')
##### Winery

df.winery.nunique()
df_winery= df[['winery','variety']].groupby('winery').count().sort_values('variety', ascending= False)
df_winery_1= df_winery[df_winery['variety']>1]
df_winery_1.head()
df_winery_1.boxplot()
df_winery_1.describe()
### Data Cleaning
##### Removing outliers using z-score that is >3
df.head()
df.info()
df.sample(5)
df[df['region_1']==df['region_2']].shape
##### NA Values in the dataset
df.isna().sum()
from scipy import stats
from scipy.stats import zscore
import numpy as np
df.replace('NaN',np.NaN,inplace=True)
##### Drop price with NA values

df1= df[df.price.notna()]
df1['price_zscore']=(df1['price']-np.mean(df1['price']))/np.std(df1['price'])
df1
threshold = 3
df_filtered=df1[df1['price_zscore'] < threshold]
df_filtered.describe()
pd.DataFrame(df_filtered.price).boxplot()
(df1.isnull().sum() / len(df1))*100

df1.isna().sum()

df1.shape

df1.info()

df1
### Feature Engineering

df1.reset_index(inplace=True)

df1.head()

df1.points.describe()

df1[df1.points==100]

df1.loc[6585]
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(pd.DataFrame(df1['points']))

df1['points_oh']= pd.DataFrame(x_scaled)

df1.points_oh.describe()

df1.isna().sum()

df1
##### Obtain the year from the tile

import re
yearSearch=[]
# search for 19xx and 20xx numeric year
for value in df1['title']:    
    regexresult= re.search(r'19\d{2}|20\d{2}', value)
    if regexresult:
        yearSearch.append(regexresult.group())
    else:
        yearSearch.append(None)
        
df1['Year']= yearSearch

print('We have extracted %d years from the wine titles and %d did not have a year.'% (len(df1[df1['Year'].notna()]), len(df1[df1['Year'].isna()].index)))

df1['Year'].describe()
yearSearch
df1.dropna(subset=['Year'], inplace=True)

df1.shape

##### Convert Year into datetime data

import datetime as dt

df1['Year'] = pd.to_datetime(df1['Year']).dt.year

df1.fillna('Missing', inplace=True)

df1.info()

df1.isna().sum()
df1.reset_index(inplace=True)

df1.drop(['index','level_0'], axis=1, inplace=True)
df1.Year.nunique()

df1

import matplotlib.pyplot as plt
plt.figure(figsize=(30,8))
ax= sns.barplot(x= df1.Year, y=df1.points)
plt.title("Average Points vs. Year", fontsize=14)
ax.set(ylim=(80,100))

import matplotlib.pyplot as plt
plt.figure(figsize=(30,8))
ax= sns.barplot(x= df1.Year, y=df1.points_oh)
plt.title("Average Normalized Points vs. Year", fontsize=14)
ax.set(ylim=(0,1))
##### Price vs Points

sns.set(font_scale=1)
sns.jointplot(x= df1.price, y=df1.points).set_axis_labels(xlabel='Price', ylabel='Points')
# ax.set(ylim=(0,1))
##### Price vs Points

sns.set(font_scale=1)
sns.jointplot(x= df1.price, y=df1.points).set_axis_labels(xlabel='Price', ylabel='Points')
# ax.set(ylim=(0,1))
price_filtered_df=df1[df1.price.between(0,100)]

ax= sns.jointplot(x= price_filtered_df.price, y=price_filtered_df.points).set_axis_labels(xlabel='Price', ylabel='Points')
plt.figure(figsize=(50,10))
sns.set(font_scale=1)
sns.barplot(x= df1.country, y=df1.points_oh)
df_country_1.plot(kind='bar', figsize=(14,9), title='Number of reviews among countries')
df1.variety.nunique()

df1[['variety','description']].groupby('variety').count().sort_values('description', ascending=False).head(20)
df1[['variety','description']].groupby('variety').count().sort_values('description', ascending=False).head(10).plot(kind='bar', figsize=(20,9))
plt.title('Top 10 Wine Variety', fontsize=15)
plt.tick_params(axis='x', labelrotation=0)
plt.show()
from wordcloud import WordCloud,STOPWORDS
plt.figure(figsize=(15,8))
plt.title('Word Cloud for Description')
wc= WordCloud(max_words=1000, max_font_size=40, background_color='black', stopwords= STOPWORDS,
             colormap='Set1')
wc.generate(' '.join(df1['description']))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
df1.drop('Unnamed: 0', axis=1, inplace=True)
df1.head()
df1.description.to_frame()
import tensorflow as tf
dfWineClassifier= df1[['description','variety']]
dfWineClassifier
Cutoff_rate= df1[['variety','description']].groupby('variety').count().sort_values('description', ascending=False).description.iloc[19]
wine_freq= df1['variety'].apply(lambda s: str(s)).explode().value_counts().sort_values(ascending=False)

wine_freq
# create a list for those that is lower than cutoff rate
Rare_wine= list(wine_freq[wine_freq< Cutoff_rate].index)
Rare_wine
dfWineClassifier['variety']= dfWineClassifier.variety.apply(lambda x: str(x) if x not in Rare_wine else 'Others')

label_words= list(wine_freq[wine_freq>= Cutoff_rate].index)
label_words.append('Others')


num_labels = len(label_words)
print("\n"  + str(num_labels) + " different categories.")
for i in range(1,5):
    print(dfWineClassifier['variety'].iloc[i])
    print(dfWineClassifier['description'].iloc[i])
    print()
# Length of each review
NUM_WORDS= 4000
SEQ_LEN = 64

#create tokenizer for our data
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS, oov_token='<UNK>', lower=True)
tokenizer.fit_on_texts(dfWineClassifier['description'])

#convert text data to numerical indexes
wine_seqs=tokenizer.texts_to_sequences(dfWineClassifier['description'])

#pad data up to SEQ_LEN (note that we truncate if there are more than SEQ_LEN tokens)
wine_seqs=tf.keras.preprocessing.sequence.pad_sequences(wine_seqs, maxlen=SEQ_LEN, padding='post')

print(wine_seqs)
# Creating a mapping from unique words to indices
# char2idx = {u:i for i, u in enumerate(label_words)}
# print(char2idx)
# idx2char = np.array(label_words)
# print(idx2char)

# print(str(len(idx2char)) + ' wine styles.')

wine_labels=pd.DataFrame({'variety': dfWineClassifier['variety']})
# wine_labels=wine_labels.replace({'variety' : char2idx})
wine_labels=wine_labels.replace(' ', '_', regex=True)

wine_labels_list = []
for item in wine_labels['variety']:
    wine_labels_list.append(str(item))

label_tokenizer = tf.keras.preprocessing.text.Tokenizer(split=' ', filters='!"#$%&()*+,./:;<=>?@[\\]^`{|}~\t\n')
label_tokenizer.fit_on_texts(wine_labels_list)

print(len(label_words))
print(label_tokenizer.word_index)

wine_label_seq = np.array(label_tokenizer.texts_to_sequences(wine_labels_list))
wine_label_seq.shape
tokenizer ## fits on description
label_tokenizer  ## fits on wine labels list
wine_seqs  ## in one hot encoding, padded with max words- 128
wine_label_seq  ## label your popular wines
df_training= pd.DataFrame(wine_seqs)
df_training['Y']= wine_label_seq
df_training
reverse_word_index= dict([value, key] for (key, value) in  tokenizer.word_index.items())

def decode_article(text):
    return '.'.join([reverse_word_index.get(i,'?') for i in text])

reverse_label_index= dict([value, key] for (key, value) in label_tokenizer.word_index.items())

def decode_label(text):
    return'.'.join([reverse_label_index.get(i,'?') for i in text])
test_entry= 5

print(decode_article(wine_seqs[test_entry]))

print(wine_seqs[test_entry])
print(decode_label(wine_label_seq[test_entry]))

print(wine_label_seq[test_entry])
import sklearn as sk
from sklearn.model_selection import train_test_split
wine_seqs.shape
wine_label_seq.shape
X_train, X_test, Y_train, Y_test= train_test_split(wine_seqs, wine_label_seq, test_size=0.1)
X_train
Y_train
len(X_train)
len(X_test)
# from tqdm import tqdm
# embedding_vector = {}
# f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r', encoding='utf-8')
# for line in tqdm(f):
#     value = line.split(' ')
#     word = value[0]
#     coef = np.asarray([float(val) for val in value[1:]])
#     embedding_vector[word] = coef
# f.close()

# print('Found %s word vectors'% len(embedding_vector))
# len(embedding_vector)
vocab_size= len(tokenizer.word_index)+1
vocab_size
# embedding_matrix = np.zeros((vocab_size,300))
# for word,i in tqdm(tokenizer.word_index.items()):
#     embedding_value = embedding_vector.get(word)
#     if embedding_value is not None:
#         embedding_matrix[i] = embedding_value
import xgboost as xgb
import tensorflow as tf
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_validate, GridSearchCV   #Additional scklearn functions
from sklearn import metrics
import pickle
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['Y'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='merror', early_stopping_rounds=early_stopping_rounds, verbose_eval=10)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Y'],eval_metric='merror')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
#     dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Y'].values, dtrain_predictions))
#   print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Y'], dtrain_predprob))
                    
    alg.save_model('0001.model')
    alg.dump_model('dump.raw.txt')
    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
# #Choose all predictors except target & IDcols
# predictors = [x for x in df_training.columns if x not in ['Y']]
# xgb1 = XGBClassifier(
#  learning_rate =0.05,
#  n_estimators=1000,
#  max_depth=10,
#  min_child_weight=2,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'multi:softmax',
#  tree_method='gpu_hist',
#  num_class=len(label_words)+1,
#  seed=27)
# modelfit(xgb1, df_training, predictors)
# param_test1 = {
#  'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }
# gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
#  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'multi:softmax', seed=27, tree_method='gpu_hist'), 
#  param_grid = param_test1, scoring='f1_weighted',iid=False, cv=5)
# gsearch1.fit(df_training[predictors],df_training['Y'])
# gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_
# gsearch1.best_score_
# gsearch1.best_params_
# pd.DataFrame(gsearch1.cv_results_)
# predictors = [x for x in df_training.columns if x not in ['Y']]
# param_test2 = {
#  'max_depth':[8,9,10],
#  'min_child_weight':[1,2]
# }
# gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
#  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'multi:softmax', seed=27, tree_method='gpu_hist'), 
#  param_grid = param_test2, scoring='f1_weighted',n_jobs=4, cv=5)
# gsearch2.fit(df_training[predictors],df_training['Y'])
# gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_
# pd.DataFrame(gsearch2.cv_results_)
import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Input, Activation, Dropout
from keras.optimizers import Adam

EMBEDDING_SIZE = 256
EMBEDDING_SIZE_2 = 64
EMBEDDING_SIZE_3 = (num_labels+1)
BATCH_SIZE = 512  # This can really speed things up
EPOCHS = 100
LR = 0.00001  # Keep it small when transfer learning

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    model = tf.keras.Sequential([

        # Add an Embedding layer expecting input vocab of a given size, and output embedding dimension of fized size we set at the top
        # filter, kernel size
        tf.keras.layers.Embedding(NUM_WORDS, EMBEDDING_SIZE),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(EMBEDDING_SIZE)),
    #     tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(128, 5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)), 
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        # use ReLU in place of tanh function since they are very good alternatives of each other.
        tf.keras.layers.Dense(EMBEDDING_SIZE_2, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),

        # Add a Dense layer with additional units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        tf.keras.layers.Dense(EMBEDDING_SIZE_3, activation='softmax')
    ])

model.summary()
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=LR),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Directory where the checkpoints will be saved
# checkpoint_dir = './checkpoints/classifer_training_checkpoints'
# # Name of the checkpoint files
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# checkpoint_callback= tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_prefix,
#     monitor='accuracy',
#     save_best_only=True,
#     mode='auto',
#     save_weights_only=True)

history= model.fit(X_train, Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                  validation_data=(X_test, Y_test))
#                   callbacks=[checkpoint_callback])

loss, accuracy = model.evaluate(X_test, Y_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
# tf.train.latest_checkpoint(checkpoint_dir)

# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

# model.build(tf.TensorShape([1, None]))

# model.summary()
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
dfWineClassifier.head()
dfWineClassifier.description[0]
new_review = ['a white wine that owes much of its popularity to winemakers in Bordeaux and the Loire Valley in France']
encoded_sample_pred_text = tokenizer.texts_to_sequences(new_review)
# Some models need padding, some don't - depends on the embedding layer.
encoded_sample_pred_text = tf.keras.preprocessing.sequence.pad_sequences(encoded_sample_pred_text, maxlen=SEQ_LEN, padding="post")
predictions = model.predict(encoded_sample_pred_text)

for n in reversed((np.argsort(predictions))[0]):
    predicted_id = [n]
    print("Guess: %s \n Probability: %f" %(decode_label(predicted_id).replace('_', ' '), 100*predictions[0][predicted_id][0]) + '%')
