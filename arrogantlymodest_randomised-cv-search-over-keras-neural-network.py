import sklearn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.tokenize import RegexpTokenizer
import os
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.wrappers.scikit_learn import KerasClassifier
import numpy
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.layers import Dropout
# load in data
df = pd.read_json('../input/train.json')
y = pd.get_dummies(df['cuisine'])
# prepare cleaning products
my_tokenizer = RegexpTokenizer(r'\w+')
my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
#from nltk.corpus import stopwords

def clean_string(x, bigrams=True):
    tokens = my_tokenizer.tokenize(x)
    tokens = [t.lower() for t in tokens]
    tokens = [word_rooter(t) for t in tokens if t not in my_stopwords]
    if bigrams:
        tokens = tokens + [tokens[i]+'_'+tokens[i+1] for i in range(len(tokens)-1)]
    return ' '.join(tokens)

def clean_ingredients_list(x, bigrams=True):
    return ' '.join([clean_string(s, bigrams=bigrams) for s in x])
    
    from sklearn.model_selection import train_test_split
# clean data for neural network
tf = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=50)
X = tf.fit_transform(df.loc[:,'ingredients'].apply(clean_ingredients_list, bigrams=False))
def create_model( nl1=1, nl2=1,  nl3=1, 
                 nn1=1000, nn2=500, nn3 = 200, lr=0.01, decay=0., l1=0.01, l2=0.01,
                act = 'relu', dropout=0, input_shape=1000, output_shape=20):
    '''This is a model generating function so that we can search over neural net 
    parameters and architecture'''
    
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999,  decay=decay)
    reg = keras.regularizers.l1_l2(l1=l1, l2=l2)
                                                     
    model = Sequential()
    
    # for the firt layer we need to specify the input dimensions
    first=True
    
    for i in range(nl1):
        if first:
            model.add(Dense(nn1, input_dim=input_shape, activation=act, kernel_regularizer=reg))
            first=False
        else: 
            model.add(Dense(nn1, activation=act, kernel_regularizer=reg))
        if dropout!=0:
            model.add(Dropout(dropout))
            
    for i in range(nl2):
        if first:
            model.add(Dense(nn2, input_dim=input_shape, activation=act, kernel_regularizer=reg))
            first=False
        else: 
            model.add(Dense(nn2, activation=act, kernel_regularizer=reg))
        if dropout!=0:
            model.add(Dropout(dropout))
            
    for i in range(nl3):
        if first:
            model.add(Dense(nn3, input_dim=input_shape, activation=act, kernel_regularizer=reg))
            first=False
        else: 
            model.add(Dense(nn3, activation=act, kernel_regularizer=reg))
        if dropout!=0:
            model.add(Dropout(dropout))
            
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'],)
    return model
# model class to use in the scikit random search CV 
model = KerasClassifier(build_fn=create_model, epochs=6, batch_size=20, verbose=1)
# learning algorithm parameters
lr=[1e-2, 1e-3, 1e-4]
decay=[1e-6,1e-9,0]

# activation
activation=['relu', 'sigmoid']

# numbers of layers
nl1 = [0,1,2,3]
nl2 = [0,1,2,3]
nl3 = [0,1,2,3]

# neurons in each layer
nn1=[300,700,1400, 2100,]
nn2=[100,400,800]
nn3=[50,150,300]

# dropout and regularisation
dropout = [0, 0.1, 0.2, 0.3]
l1 = [0, 0.01, 0.003, 0.001,0.0001]
l2 = [0, 0.01, 0.003, 0.001,0.0001]

# dictionary summary
param_grid = dict(
                    nl1=nl1, nl2=nl2, nl3=nl3, nn1=nn1, nn2=nn2, nn3=nn3,
                    act=activation, l1=l1, l2=l2, lr=lr, decay=decay, dropout=dropout, 
                    input_shape=[X.shape[1]], output_shape = [y.shape[1]],
                 )
grid = RandomizedSearchCV(estimator=model, cv=KFold(3), param_distributions=param_grid, 
                          verbose=20,  n_iter=10, n_jobs=1)
grid_result = grid.fit(X.toarray(), y)
cv_results_df = pd.DataFrame(grid_result.cv_results_)
cv_results_df.to_csv('gridsearch.csv')
cv_results_df
print(grid_result.best_params_)
best_model = grid_result.best_estimator_
df_holdout = pd.read_json('../input/test.json')
df_submission = df_holdout[['id']]
X_holdout = tf.transform(df_holdout.loc[:,'ingredients'].apply(clean_ingredients_list, bigrams=False))

# Predictions 
y_pred_holdout_proba = best_model.predict_proba(X_holdout)
y_pred_holdout = (y_pred_holdout_proba == y_pred_holdout_proba.max(axis=1)[:,np.newaxis])*1
y_pred_string = np.sum(y_pred_holdout*y.columns.values, axis=1)

df_submission['cuisine']=y_pred_string
df_submission.to_csv('neural_net_base_submission.csv', index=False)