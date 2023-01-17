!pip install nlpaug  python-dotenv tensorflow_addons
import os
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
import json
from statistics import *
import nltk
import numpy as np
import pandas as pd
from transformers import *
import tokenizers
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from sklearn.metrics import log_loss
from nlpaug.util import Action
import tensorflow_addons as tfa
import random
import torch
 
def seed_all(seed_value):
    

    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars

    if torch.cuda.is_available():
        
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

seed_all(1)
train= pd.read_csv("../input/zindi-aldc/Train_df.csv").astype(str).fillna("")
test = pd.read_csv("../input/zindi-aldc/Test_df.csv").astype(str).fillna("")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)
def to_class(text):
    if text == 'Drugs': return np.array([0, 0, 0, 1])
    if text == 'Suicide': return np.array([0, 0, 1, 0])
    if text == 'Alcohol': return np.array([0, 1, 0, 0])
    if text == 'Depression': return np.array([1, 0, 0, 0])
    
def transform(array):
    Ynew = np.zeros((array.shape[0],4))
    for idx in range(array.shape[0]):
        Ynew[idx] = to_class(array[idx])
    return Ynew
 
from transformers import *
 
# text augmentation
def roberta_augment_text(text):
    aug = naw.ContextualWordEmbsAug(model_path ='roberta-base',
                                  action ="insert")

    return aug.augment(text)

def bert_augment_text(text):
    aug =naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert")
    return aug.augment(text)

def distillbert_augment_text(text):
    aug = naw.ContextualWordEmbsAug(
    model_path='distilbert-base-uncased', action="substitute")
    return aug.augment(text)
 
def synonym_augment(text):
    aug =naw.SynonymAug(aug_src='wordnet')
    return aug.augment(text)
drugs_df = train[train['label'] == "Drugs"]

y = pd.concat([train, drugs_df,drugs_df,drugs_df,drugs_df,drugs_df,drugs_df ])

y.shape
suicide_df = train[train['label'] == "Suicide"]

y = pd.concat([y, suicide_df,suicide_df,suicide_df,suicide_df,suicide_df])

y.shape
alcohol_df = train[train['label'] == "Alcohol"]

y = pd.concat([y, alcohol_df, alcohol_df, alcohol_df])

y.shape
train = y

train.sample(5)
train['roberta_text'] = train['text'].apply(lambda x : roberta_augment_text(x))
#train['bert_text'] = train['text'].apply(lambda x : bert_augment_text(x))
#train['distillbert_text'] = train['text'].apply(lambda x : distillbert_augment_text(x))

test['roberta_text'] = test['text'].apply(lambda x : roberta_augment_text(x))
#test['bert_text'] = test['text'].apply(lambda x : bert_augment_text(x))
#test['distillbert_text'] = test['text'].apply(lambda x : distillbert_augment_text(x))
'''
import nltk
nltk.download('averaged_perceptron_tagger')
xtrain['synonym_text'] = xtrain['text'].apply(lambda x : synonym_augment(x))
xtest['synonym_text'] = xtest['text'].apply(lambda x : synonym_augment(x))
test['synonym_text'] = test['text'].apply(lambda x : synonym_augment(x))
'''

df = list(train.text.values) + list(train.roberta_text.values) 
#from gensim.utils import simple_preprocess
import numpy as np
 
maxlen = 64
 
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
 
xtrain_ids = np.array(tokenizer.batch_encode_plus(df,
                       add_special_tokens =True,
                      return_attention_mask =False,
                        return_token_type_ids = False,
                        truncation=True,
                        max_length= maxlen, 
                        
                        pad_to_max_length=True)['input_ids'])


xtrain_masks = np.array(tokenizer.batch_encode_plus(df,
                  add_special_tokens =True,
                  return_attention_mask =True,
                  return_token_type_ids = False,
                  truncation=True,
                  max_length= maxlen, 
                  pad_to_max_length=True)['attention_mask']
)
xtest_ids = np.array(tokenizer.batch_encode_plus(test.text.values, 
                                                  add_special_tokens =True,
                      return_attention_mask =False,
                      return_token_type_ids = False,
                      truncation=True,
                      max_length= maxlen, 
                      
                      pad_to_max_length=True)['input_ids'])

xtest_masks =np.array(tokenizer.batch_encode_plus(test.text.values, 
                                                   add_special_tokens =True,
                      return_attention_mask =True,
                      return_token_type_ids = False,
                      truncation=True,
                      max_length= maxlen, 
                      
                      pad_to_max_length=True)['attention_mask'])


xtest_ids_r = np.array(tokenizer.batch_encode_plus(test.roberta_text.values, 
                                                    add_special_tokens =True,
                      return_attention_mask =False,
                      return_token_type_ids = False,
                      truncation=True,
                      max_length= maxlen, 
                      pad_to_max_length=True)['input_ids'])

xtest_masks_r =np.array(tokenizer.batch_encode_plus(test.roberta_text.values, 
                                                     add_special_tokens =True,
                      return_attention_mask =True,
                      return_token_type_ids = False,
                      truncation=True,
                      max_length= maxlen, 
                      
                      pad_to_max_length=True)['attention_mask'])

Y = np.hstack([train.label.values, train.label.values])
Y.shape, xtrain_ids.shape
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import Adam
from tensorflow import int32
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
def label_smoothing(y_true,y_pred):
    
    return tf.keras.losses.categorical_crossentropy(y_true,y_pred,label_smoothing=0.1)
import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import Adam
from tensorflow import int32
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model, to_categorical
 
def get_model(): 
    
    input_ids = Input((maxlen,), dtype=int32)
    input_masks = Input((maxlen,),dtype=int32)
    #input_stats = Input((2,), dtype=int32)
  
    roberta = TFRobertaModel.from_pretrained("roberta-base")
    x,cls   = roberta([input_ids, input_masks])
    drop1 = Dropout(0.1, name="dropout1")(x)

    x = GlobalMaxPooling1D()(drop1)
  

    hidden = Dropout(0.1, name="dropout2")(x)
    
    flat = Flatten()(hidden)
 
    classification = Dense(4, activation="softmax",
                           kernel_initializer=glorot_normal(seed=1),
                           bias_initializer=glorot_normal(seed=1),
                           name="classification")(flat)
                           
    model = Model(inputs = [input_ids,input_masks], outputs=classification)
    #plot_model(model)
    optim = tfa.optimizers.RectifiedAdam(learning_rate=3e-5,min_lr=1e-6,total_steps=6000)
    model.compile(loss= "categorical_crossentropy", optimizer=optim, metrics=['accuracy',tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    return model
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_loss(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch,  history.history['loss'],
               color=colors[n], label='Train '+label)
    plt.semilogy(history.epoch,  history.history['val_loss'],
          color=colors[n], label='Val '+label,
          linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler,History

history = History()
 
es = EarlyStopping(monitor ="val_loss", mode="min", verbose= 1, patience = 3)
 
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state =1).split(xtrain_ids,Y)
predictions = list()

predictions_r =list()

err = list()
n = 0
VER ='v0'
for i, (train_idx, test_idx) in enumerate(kfold):
    
 
    #features_trainfold = features[train_idx]
    #features_validfold = features[test_idx]
 
    ytrain_fold = transform(Y[train_idx])  #smooth_labels(transform(Y[train_idx]), 0.1)
    yvalid_fold = transform(Y[test_idx])  #smooth_labels(transform(Y[test_idx]), 0.1)
 
    model = get_model()
    model.summary()
    sv = ModelCheckpoint(
      '%s-roberta-%i.h5'%(VER,i), monitor='val_loss', verbose=1, save_best_only=True,
      save_weights_only=True, mode='auto', save_freq='epoch')
    
    #n_steps=(xtrain_ids.shape[0]+len(train_idx))//32
 
    h = model.fit([xtrain_ids[train_idx],xtrain_masks[train_idx]], 
            ytrain_fold, 
            batch_size= 32, 
            epochs=30, 
            callbacks =[sv],
            validation_data=([xtrain_ids[test_idx],xtrain_masks[test_idx]],yvalid_fold)
            )
    #print('Loss plot')
    n += 1
    plot_loss(h, "Fold"+str(n), n)
    print('Loading model...')
    model.load_weights('%s-roberta-%i.h5'%(VER,i))
 
    preds = model.predict([xtest_ids,xtest_masks], batch_size=32, verbose=1)
    preds_r = model.predict([xtest_ids_r,xtest_masks_r], batch_size=32, verbose=1)

    predictions.append(preds)
    predictions_r.append(preds_r)
test_preds = np.average(predictions, axis = 0)
test_preds_r = np.average(predictions_r, axis = 0)
preds = (test_preds + test_preds_r)/2
preds.shape
ss = pd.read_csv("../input/zindi-aldc/SampleSubmission_df.csv")
ss['Depression'] = preds[:,0]
ss['Alcohol'] = preds[:,1]
ss['Suicide'] = preds[:,2]
ss['Drugs']  = preds[:,3]

ss.to_csv("submission.csv", index = False)
ss.sample(10)