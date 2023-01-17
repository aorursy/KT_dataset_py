import pandas as pd
import numpy as np
import gc
train=pd.read_table('../input/train.tsv')
test=pd.read_table('../input/test.tsv')
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
train=train.fillna('missing')
test=test.fillna('missing')
allData=train.append(test).reset_index(drop=True)
del train
del test

gc.collect()
le=LabelEncoder()
allData['c_lv3']=le.fit_transform(allData.category_name.ravel())
allData['c_brand']=le.fit_transform(allData.brand_name.str.lower())
allData['c_condition']=le.fit_transform(allData.item_condition_id)
allData['c_shipping']=le.fit_transform(allData.shipping)
tokenizer=Tokenizer()
tokenizer.fit_on_texts(allData.name.str.lower())
allData['c_names']=pd.Series(tokenizer.texts_to_sequences(allData.name.str.lower()))
tokenizer.fit_on_texts(allData.item_description.str.lower())
allData['c_descriptions']=pd.Series(tokenizer.texts_to_sequences(allData.item_description.str.lower()))
max_desc=allData['c_descriptions'].apply(lambda x: len(x)).max()
max_name=allData['c_names'].apply(lambda x: len(x)).max()
dummy_name=allData['c_names'].apply(lambda x: max(x) if len(x)>=1 else 0).max()+1
dummy_desc=allData['c_descriptions'].apply(lambda x: max(x) if len(x)>=1 else 0).max()+1
def pad(sequence,maxlen,dummy_word):
    lsequence=list(sequence)
    if len(lsequence)>maxlen:
        return sequence[:maxlen]
    while len(lsequence)<maxlen:
        lsequence.append(dummy_word)
    return np.array(lsequence)
allData['c_names']=allData['c_names'].apply(lambda x: pad(x,max_name,dummy_name))
allData['c_descriptions']=allData['c_descriptions'].apply(lambda x: pad(x,60,dummy_desc))
tr_data=allData[(allData.price.as_matrix()>=1.0) & (np.isnan(allData.test_id.as_matrix()))]
names=np.array(list(tr_data.c_names))
descs=np.array(list(tr_data.c_descriptions))
category3=tr_data.c_lv3.ravel()
brand=tr_data.c_brand.ravel()
condition=tr_data.c_condition.ravel()
shipping=tr_data.c_shipping.ravel()
labels=tr_data.price.ravel()
max_c_lv3=allData.c_lv3.max()
max_brand=allData.c_brand.max()
max_condition=allData.c_condition.max()
max_shipping=allData.c_shipping.max()
te_data=allData[np.isnan(allData.train_id.as_matrix())]
names_te=np.array(list(te_data.c_names))
descs_te=np.array(list(te_data.c_descriptions))
category3_te=te_data.c_lv3.ravel()
brand_te=te_data.c_brand.ravel()
condition_te=te_data.c_condition.ravel()
shipping_te=te_data.c_shipping.ravel()
#s_labels=scaler.fit_transform(labels.reshape(-1,1))
s_labels=np.log(labels+1.).reshape(-1,1)
X={'name':names,
   'descriptions':descs,
   #'category_level_1':category1,
   #'category_level_2':category2,
   'category_level_3':category3,
   'brand':brand,
   'condition':condition,
   'shipping':shipping}
X_te={'name':names_te,
      'descriptions':descs_te,
      #'category_level_1':category1,
      #'category_level_2':category2,
      'category_level_3':category3_te,
      'brand':brand_te,
      'condition':condition_te,
      'shipping':shipping_te}
del names
del descs
del category3
del brand
del condition
del shipping
del labels

del names_te
del descs_te
del category3_te
del brand_te
del condition_te
del shipping_te

del allData

gc.collect()
import keras.layers as kl
import keras.models as km
import keras.backend as K
import keras
def schedule(e):
    if e<2:
        return 0.0013
    elif e==2:
        return 0.0012
    else:
        return 0.0011
#Inputs
NN_names=kl.Input(shape=[X['name'].shape[1]],name='name')
NN_descs=kl.Input(shape=[X['descriptions'].shape[1]],name='descriptions')
#NN_cat1=kl.Input(shape=[1],name='category_level_1')
#NN_cat2=kl.Input(shape=[1],name='category_level_2')
NN_cat3=kl.Input(shape=[1],name='category_level_3')
NN_brand=kl.Input(shape=[1],name='brand')
NN_condition=kl.Input(shape=[1],name='condition')
NN_shipping=kl.Input(shape=[1],name='shipping')

#Embeddings
NN_emb_name=kl.Embedding(dummy_name+1, 20)(NN_names)
NN_emb_desc=kl.Embedding(dummy_desc+1, 30)(NN_descs)
#NN_emb_cat1=kl.Embedding(max_c_lv1+1, 3)(NN_cat1)
#NN_emb_cat2=kl.Embedding(max_c_lv2+1, 5)(NN_cat2)
NN_emb_cat3=kl.Embedding(max_c_lv3+1, 8)(NN_cat3)
NN_emb_brand=kl.Embedding(max_brand+1, 5)(NN_brand)

#LSTM Layer
NN_lstm_name=kl.LSTM(8)(NN_emb_name)
NN_lstm_desc=kl.LSTM(20)(NN_emb_desc)

#Main layer, joins all data
NN_main=kl.concatenate([#kl.Flatten() (NN_emb_cat1),
#                        kl.Flatten() (NN_emb_cat2),
                        kl.Flatten() (NN_emb_cat3),
                        kl.Flatten() (NN_emb_brand),
                        NN_condition,
                        NN_shipping,
                        NN_lstm_name,
                        NN_lstm_desc])

#Add a dropout layer before two dense layers to process the whole picture
NN_main=kl.Dropout(.1) (kl.Dense(128,activation='relu') (NN_main))
NN_main=kl.Dropout(.1) (kl.Dense(64,activation='relu') (NN_main))

#output
NN_output=kl.Dense(1,activation='linear') (NN_main)

model=km.Model([NN_names,NN_descs,NN_cat3,NN_brand,NN_condition,NN_shipping],NN_output)
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=0.0013,decay=0.0), metrics=["mae"])
model.summary()
history=model.fit(X,s_labels,epochs=5,batch_size=15000,validation_split=0.0,callbacks=[keras.callbacks.LearningRateScheduler(schedule)])
pd.DataFrame({'test_id':te_data.test_id.as_matrix().astype(int),
              'price':(np.exp(model.predict(X_te))-1.).reshape(-1)}).to_csv('submissions.csv',
                                                                             index=False,
                                                                             header=True,columns=['test_id','price'])
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_palette('nipy_spectral')
sns.distplot(s_labels)
sns.distplot(np.log(pd.read_csv('submissions.csv').price.as_matrix()+1))
plt.show()
