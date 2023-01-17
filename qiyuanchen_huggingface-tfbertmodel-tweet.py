!pip install transformers

import transformers
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
import pandas as pd

train = pd.read_csv('../input/fake-news/fake_news.csv',encoding='utf-8')
import numpy as np

import tensorflow as tf 
def bert_encode(data,maximum_length) :

  input_ids = []

  attention_masks = []

  



  for i in range(len(data.text_x)):

      encoded = tokenizer.encode_plus(

        

        data.text_x[i],

        add_special_tokens=True,

        max_length=maximum_length,

        pad_to_max_length=True,

        

        return_attention_mask=True,

        

      )

      

      input_ids.append(encoded['input_ids'])

      attention_masks.append(encoded['attention_mask'])

  return np.array(input_ids),np.array(attention_masks)
%%time

train_input_ids,train_attention_masks = bert_encode(train,256)
train_input_ids
!pip install xgboost
from xgboost.sklearn import XGBClassifier

clf = XGBClassifier(

silent=0 ,#設定成1則沒有執行資訊輸出，最好是設定為0.是否在執行升級時列印訊息。

#nthread=4,# cpu 執行緒數 預設最大

tree_method='gpu_hist',

learning_rate= 0.3, # 如同學習率

min_child_weight=1, 

# 這個引數預設是 1，是每個葉子裡面 h 的和至少是多少，對正負樣本不均衡時的 0-1 分類而言

#，假設 h 在 0.01 附近，min_child_weight 為 1 意味著葉子節點中最少需要包含 100 個樣本。

#這個引數非常影響結果，控制葉子節點中二階導的和的最小值，該引數值越小，越容易 overfitting。

max_depth=20, # 構建樹的深度，越大越容易過擬合

gamma=0,  # 樹的葉子節點上作進一步分割槽所需的最小損失減少,越大越保守，一般0.1、0.2這樣子。

subsample=0.9, # 隨機取樣訓練樣本 訓練例項的子取樣比

max_delta_step=0,#最大增量步長，我們允許每個樹的權重估計。

colsample_bytree=1, # 生成樹時進行的列取樣 

reg_lambda=1,  # 控制模型複雜度的權重值的L2正則化項引數，引數越大，模型越不容易過擬合。

#reg_alpha=0, # L1 正則項引數

#scale_pos_weight=1, #如果取值大於0的話，在類別樣本不平衡的情況下有助於快速收斂。平衡正負權重

objective= 'binary:logistic', #多分類的問題 指定學習任務和相應的學習目標

#num_class=10, # 類別數，多分類與 multisoftmax 並用

n_estimators=700, #樹的個數

seed=1000 #隨機種子

#eval_metric= 'auc'

)
clf
from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_input_ids, np.array(train.Fake), test_size=0.1, random_state=0)##test_size測試集合所佔比例

clf.fit(X_train,y_train,eval_metric='auc')

#設定驗證集合 verbose=False不列印過程

clf.fit(X_train, y_train,eval_metric='auc',verbose=True)

#獲取驗證集合結果

y_true, y_pred = y_test, clf.predict(X_test)

print("Accuracy:",accuracy_score(y_true, y_pred))

!pip install transformers

import transformers
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
vocab = tokenizer.vocab

print("字典大小：", len(vocab))
import pandas as pd

train = pd.read_csv('../input/fake-news/fake_news.csv',encoding='utf-8')
train
import numpy as np

import tensorflow as tf 
def bert_encode(data,maximum_length) :

  input_ids = []

  attention_masks = []

  



  for i in range(len(data.text_x)):

      encoded = tokenizer.encode_plus(

        

        data.text_x[i],

        add_special_tokens=True,

        max_length=maximum_length,

        pad_to_max_length=True,

        

        return_attention_mask=True,

        

      )

      

      input_ids.append(encoded['input_ids'])

      attention_masks.append(encoded['attention_mask'])

  return np.array(input_ids),np.array(attention_masks)
%%time

train_input_ids,train_attention_masks = bert_encode(train,256)
train_input_ids
train_attention_masks
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

def create_model(bert_model):

  input_ids = tf.keras.Input(shape=(256,),dtype='int32')

  attention_masks = tf.keras.Input(shape=(256,),dtype='int32')

  

  output = bert_model([input_ids,attention_masks])

  output = output[1]

  output = tf.keras.layers.Dense(32,activation='relu')(output)

  output = tf.keras.layers.Dropout(0.2)(output)



  output = tf.keras.layers.Dense(1,activation='sigmoid')(output)

  model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)

  model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])

  return model
from transformers import TFBertModel

bert_model = TFBertModel.from_pretrained('bert-base-chinese')
model = create_model(bert_model)

model.summary()
np.array(train.Fake)
history = model.fit([train_input_ids,train_attention_masks],np.array(train.Fake),validation_split=0.1, epochs=2,batch_size=10)
model.save_weights('fk_model_weights.h5')
test=train[:1]

test
test_input_ids
test_attention_masks
test.text_x	
test_input_ids,test_attention_masks = bert_encode(test,256)

result = model.predict([test_input_ids,test_attention_masks])

result
result = np.round(result).astype(int)

result
import pandas as pd

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train
test
import matplotlib.pyplot as plt

plt.title('Train Data')

plt.xlabel('Target Distribution')

plt.ylabel('Samples')

plt.hist(train.target)

plt.show()
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)#使用bert-large-uncased
vocab = tokenizer.vocab

print("字典大小：", len(vocab))
import numpy as np

import tensorflow as tf 
def bert_encode(data,maximum_length) :

  input_ids = []

  attention_masks = []

  



  for i in range(len(data.text)):

      encoded = tokenizer.encode_plus(

        

        data.text[i],

        add_special_tokens=True,

        max_length=maximum_length,

        pad_to_max_length=True,

        

        return_attention_mask=True,

        

      )

      

      input_ids.append(encoded['input_ids'])

      attention_masks.append(encoded['attention_mask'])

  return np.array(input_ids),np.array(attention_masks)
train_input_ids,train_attention_masks = bert_encode(train,60)

test_input_ids,test_attention_masks = bert_encode(test,60)
train_input_ids
len(train_input_ids)
train_attention_masks
len(train_attention_masks)
test_input_ids
len(test_input_ids)
test_attention_masks
len(test_attention_masks)
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

def create_model(bert_model):

  input_ids = tf.keras.Input(shape=(60,),dtype='int32')

  attention_masks = tf.keras.Input(shape=(60,),dtype='int32')

  

  output = bert_model([input_ids,attention_masks])

  output = output[1]

  output = tf.keras.layers.Dense(32,activation='relu')(output)

  output = tf.keras.layers.Dropout(0.2)(output)



  output = tf.keras.layers.Dense(1,activation='sigmoid')(output)

  model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)

  model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])

  return model



from transformers import TFBertModel

bert_model = TFBertModel.from_pretrained('bert-large-uncased')
model = create_model(bert_model)

model.summary()
history = model.fit([train_input_ids,train_attention_masks],train.target,validation_split=0.2, epochs=2,batch_size=10)
result = model.predict([test_input_ids,test_attention_masks])

result = np.round(result).astype(int)
test_input_ids
test_attention_masks
result
model
model.save_weights('my_model_weights.h5')
'''%%time

# Prediction by BERT model with my tuning

model.load_weights('my_model_weights.h5')'''
result = pd.DataFrame(result)

submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

output = pd.DataFrame({'id':submission.id,'target':result[0]})

output.to_csv('submission.csv',index=False)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()