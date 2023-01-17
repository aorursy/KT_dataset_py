# loading libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import lightgbm as lgbm
import re
from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape,test.shape
# visualizacion repida de datos
train.tail(10)
test.tail(10)
# taking comment from train and test and making a single dataframe
# Tomar comentarios del set de entrenamiento y prueba y hacer un único marco de datos.
train_ = train['comment_text']
test_ = test['comment_text']

alldata = pd.concat([train_, test_], axis=0)

alldata = pd.DataFrame(alldata)
# imputando a los valores perdidos
alldata.comment_text.fillna('blllllllllllllllllllllllaaaaaaaaaaaaaaaahhhhhhhh...!!!', inplace=True)
# function to clean the comment
# adapted from a kaggle kernal, can't find its link now

# función para limpiar el comentario
# adaptado de un núcleo de Kaggle, https://www.kaggle.com/currie32/the-importance-of-cleaning-text
def cleanData(text):
    txt = str(text)
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"Find", "find", text) 

    return txt
# definimos un CountVectorizer como matriz de confusion
countvec = CountVectorizer(max_features = 1500, ngram_range=(1, 2))
# Ejecutamos el metodo de limpieza de datos
alldata['comment_text'] = alldata['comment_text'].map(lambda x: cleanData(x))
# transformar los datos de texto utilizando CountVectorizer
countvecdata = countvec.fit_transform(alldata['comment_text'])
# convertir los datos a una matriz
countvec_df = pd.DataFrame(countvecdata.todense()) 
# añadiendo encabezados de columna
countvec_df.columns = ['col' + str(x) for x in countvec_df.columns]
# Cortar los datos para entrenar y probar.
countvec_df_train = countvecdata[:len(train_)] 
countvec_df_test = countvecdata[len(train_):]
# convertimos en float32
countvec_df_train_ = countvec_df_train.astype('float32')
countvec_df_test_ = countvec_df_test.astype('float32')
# haciendo lista y marcador de posición
col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

preds = np.zeros((test.shape[0], len(col)))
# parametros para el clasificador LightGBMClassifier
params = {
    'objective' :'binary',
    'learning_rate' : 0.02,
    'num_leaves' : 76,
    'feature_fraction': 0.64, 
    'bagging_fraction': 0.8, 
    'bagging_freq':1,
    'boosting_type' : 'gbdt',
    'metric': 'binary_logloss'
}
# hacemos predicción para cada columna
# adaptado de kaggle kernel https://www.kaggle.com/yekenot/toxic-regression/code

for i, j in enumerate(col):
    print('columna de ajuste : '+j)
    # creamos los set de entrenamiento y validacion
    X_train, X_valid, Y_train, Y_valid = train_test_split(countvec_df_train_,  train[j], random_state=7, test_size=0.33)
    
    # modelamos el dataset lgbm para entrenamiento y validacion
    d_train = lgbm.Dataset(X_train, Y_train)
    d_valid = lgbm.Dataset(X_valid, Y_valid)
    
    # entrenamiento con parada temprana
    bst = lgbm.train(params, d_train, 5000, valid_sets=[d_valid], verbose_eval=50, early_stopping_rounds=100)
    
    # Realizamos la prediccion para cada columna
    print('prediciendo para :' +j)
    preds[:,i] = bst.predict(countvec_df_test_)

print('Entrenamiento terminado')
# visualizamos  los resultados!!
subm = pd.read_csv('../input/sample_submission.csv')
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.head(5)
submission.to_csv('submission_001.csv', index=False)