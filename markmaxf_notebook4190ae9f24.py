# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# read in input data 

test_df=pd.read_csv('../input/test.tsv',sep='\t',quoting=3)

train_df=pd.read_csv('../input/train.tsv',sep='\t')

train_df.head()
train_df.columns
test_df.head()
train_df.isnull().any()
df_train=train_df

df_test=test_df
df_train.category_name.fillna(value='missing',inplace=True)

df_train.brand_name.fillna(value='missing',inplace=True)

df_train.item_description.fillna(value='missing',inplace=True)



df_test.category_name.fillna(value='missing',inplace=True)

df_test.brand_name.fillna(value='missing',inplace=True)

df_test.item_description.fillna(value='missing',inplace=True)
df_train.head()
# split catrgory data 

#cat1,cat2,cat3=[],[],[]

#for i in range(df_train.shape[0]):

#    k=df_train['category_name'][i].split('/')

#    cat1.append(k[0])

#    cat2.append(k[1])

#    cat3.append(k[2])

def split_txt(text):

    if text=='missing':

        return ['missing']*3

    else:

        return text.split('/')
df_train['cat1'],df_train['cat2'],df_train['cat3']=zip(*df_train.category_name.apply(lambda x: split_txt(x)))

df_test['cat1'],df_test['cat2'],df_test['cat3']=zip(*df_test.category_name.apply(lambda x: split_txt(x)))
# the category_name could be subdived into three feature. while the name and item_description is 

# bag of words which contains the import info about the item. 
df_train.head()
import matplotlib.pyplot as plt 

%matplotlib inline 

plt.figure(figsize=(16,9))

x=df_train.cat1.value_counts().reset_index(0)

plt.barh(x.index,x.cat1)

plt.xlabel('number of item')

plt.ylabel('Top 20 of cat 1')

plt.yticks(x.index,x['index'])
# check cat 2

plt.figure(figsize=(16,9))

x=df_train.cat2.value_counts().reset_index(0)[:20]

plt.barh(x.index,x.cat2,)

plt.xlabel('number of item')

plt.ylabel('Top 20 of cat 2')

plt.yticks(x.index,x['index'])
#
plt.figure(figsize=(16,9))

x=df_train.cat3.value_counts().reset_index(0)[:20]

plt.barh(x.index,x.cat3)

plt.xlabel('number of item')

plt.ylabel('Top 20 of cat 1')

plt.yticks(x.index,x['index'])
#since name and item description is bag of word. so using rnn to model 

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.text import text_to_word_sequence
Token=Tokenizer()

text_all=np.hstack([df_train.item_description.str.lower(),df_train.name.str.lower()])

Token.fit_on_texts(text_all)



df_train['name_sequence']=Token.texts_to_sequences(df_train.name)

df_train['item_des_sequence']=Token.texts_to_sequences(df_train.item_description)



df_test['name_sequence']=Token.texts_to_sequences(df_test.name)

df_test['item_des_sequence']=Token.texts_to_sequences(df_test.item_description)
# after convert text into sequence, build training model 

#using rnn wo bad of words

df_train.columns
from keras.models import Model

from keras.layers import Input,Embedding, Dense, Dropout,Recurrent,Flatten,concatenate,GRU
# data processing, Label words for cat1,cat2,cat3 

# need to stack train and test for fit model otherwise, something wrong on embedding model 



from sklearn.preprocessing import LabelEncoder

Le_cat1=LabelEncoder()

Le_cat2=LabelEncoder()

Le_cat3=LabelEncoder()

Le_brand=LabelEncoder()



raw_cat1=np.hstack([np.array(df_train.cat1),np.array(df_test.cat1)])

raw_cat2=np.hstack([np.array(df_train.cat2),np.array(df_test.cat2)])

raw_cat3=np.hstack([np.array(df_train.cat3),np.array(df_test.cat3)])

raw_brand=np.hstack([np.array(df_train.brand_name),np.array(df_test.brand_name)])



Le_cat1.fit(raw_cat1)

Le_cat2.fit(raw_cat2)

Le_cat3.fit(raw_cat3)

Le_brand.fit(raw_brand)



df_train.cat1=Le_cat1.transform(df_train.cat1)

df_train.cat2=Le_cat2.transform(df_train.cat2)

df_train.cat3=Le_cat3.transform(df_train.cat3)

df_train.brand_name=Le_brand.transform(df_train.brand_name)



df_test.cat1=Le_cat1.transform(df_test.cat1)

df_test.cat2=Le_cat2.transform(df_test.cat2)

df_test.cat3=Le_cat3.transform(df_test.cat3)

df_test.brand_name=Le_brand.transform(df_test.brand_name)
max_name_seq = 10

max_item_desc = 75

max_text = np.max([np.max(df_train.name_sequence.max())

                   , np.max(df_test.name_sequence.max())

                  , np.max(df_train.item_des_sequence.max())

                  , np.max(df_test.item_des_sequence.max())])+5

max_cat1 = np.max([df_train.cat1.max(), df_test.cat1.max()])+1

max_cat2 = np.max([df_train.cat2.max(), df_test.cat2.max()])+1

max_cat3 = np.max([df_train.cat3.max(), df_test.cat3.max()])+1

max_brand = np.max([df_train.brand_name.max(), df_test.brand_name.max()])+1





# extract data for train and test 

from keras.preprocessing.sequence import pad_sequences

def extract_data(d_in):

    d_out={

        'name':pad_sequences(d_in.name_sequence,maxlen=max_name_seq),

        'item_desc':pad_sequences(d_in.item_des_sequence,maxlen=max_item_desc),

        'brand_name':np.array(d_in.brand_name).reshape(-1,1),

        'cat1':np.array(d_in.cat1).reshape(-1,1),

        'cat2':np.array(d_in.cat2).reshape(-1,1),

        'cat3':np.array(d_in.cat3).reshape(-1,1),

        'item_condition':np.array(d_in.item_condition_id).reshape(-1,1),

        'shipping':np.array(d_in.shipping).reshape(-1,1)

    }

    return d_out



from sklearn.model_selection import train_test_split

x_train,x_valid=train_test_split(df_train,test_size=0.2,random_state=1)

train_keras=extract_data(x_train)

valid_keras=extract_data(x_valid)

test_kreas=extract_data(df_test)
max_text
# build model 

name=Input(shape=[train_keras['name'].shape[1]],name='name')

item_desc=Input(shape=[train_keras['item_desc'].shape[1]],name='item_desc')

brand_name=Input(shape=[train_keras['brand_name'].shape[1]],name='brand_name')

cat1=Input(shape=[train_keras['cat1'].shape[1]],name='cat1')

cat2=Input(shape=[train_keras['cat2'].shape[1]],name='cat2')

cat3=Input(shape=[train_keras['cat3'].shape[1]],name='cat3')

item_condition=Input(shape=[train_keras['item_condition'].shape[1]],name='item_condition')

ship=Input(shape=[train_keras['shipping'].shape[1]],name='shipping')

# Embedding: convert spart matrix to dense matrix 

Embed_name=Embedding(max_text,output_dim=30)(name)

Embed_desc=Embedding(max_text,output_dim=30)(item_desc)

Embed_brand=Embedding(max_brand,output_dim=10)(brand_name)

Embed_cat1=Embedding(max_cat1,output_dim=10)(cat1)

Embed_cat2=Embedding(max_cat2,output_dim=20)(cat2)

Embed_cat3=Embedding(max_cat3,output_dim=20)(cat3)
rnn_name=GRU(24)(Embed_name)

rnn_desc=GRU(24)(Embed_desc)
model=concatenate([rnn_name,rnn_desc,

                  Flatten()(Embed_brand),Flatten()(Embed_cat1),Flatten()(Embed_cat2),

                  Flatten()(Embed_cat3),item_condition,ship])
model1=Dropout(rate=0.1)(Dense(128)(model))

model2=Dropout(rate=0.1)(Dense(16)(model1))



output=Dense(1,activation='linear')(model2)
model_keras=Model([name,item_desc,brand_name,cat1,

                  cat2,cat3,item_condition,ship],output)
model_keras.compile(optimizer='adam',loss=['mse'],metrics=['mae'])
model_keras.summary(0)
from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()

y=sc_x.fit_transform(np.log(x_train.price.reshape(-1,1)+1))

y_valid=sc_x.transform(np.log(x_valid.price.reshape(-1,1)+1))
model_keras.fit(train_keras,y,batch_size=20000,epochs=5)
#prediction=model_keras.predict(valid_keras)
#predict_price=sc_x.inverse_transform(prediction)
#error=np.abs(predict_price-np.array(x_valid.price).reshape(-1,1))
predict_2=model_keras.predict(test_kreas)
predict_2_price=np.exp(sc_x.inverse_transform(predict_2))-1
write_csv=pd.DataFrame({'test_id':test_df.test_id,

                        'price':predict_2_price.reshape(predict_2_price.shape[0])})
write_csv2=pd.DataFrame({'test_id':write_csv.test_id,

                         'price':write_csv.price})
sam=pd.read_csv('../input/sample_submission.csv')
write_csv2.head()
#write_csv2.to_csv('../input/result.csv',index=False)
write_csv2.to_csv('sample_submission.csv',index=False)
#plt.barh(x.index,np.array(x))
#
#
#
#
#
#
#['item_condition'].shape