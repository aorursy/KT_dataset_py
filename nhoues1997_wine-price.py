import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns





import os

import gc

import joblib







from sklearn import metrics, linear_model

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn.preprocessing import StandardScaler

from tqdm.notebook import tqdm







import tensorflow as tf

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics, preprocessing

from tensorflow.keras import layers

from tensorflow.keras import optimizers

from tensorflow.keras.models import Model, load_model

from tensorflow.keras import callbacks

from tensorflow.keras import backend as K

from tensorflow.keras import utils



from sklearn import model_selection

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_error

from math import sqrt



import warnings

warnings.filterwarnings("ignore")

train = pd.read_csv('../input/datacept-wine-prices-prediction/trainfinal.csv')

test = pd.read_csv('../input/datacept-wine-prices-prediction/test.csv')
train.head(2)
test['price'] = -1 

all_data = pd.concat([train,test],ignore_index =True)
all_data.shape
(train[train['price']<250]['price']).hist(bins=50)
def missing_values(data,number=20) : 

  total = data.isnull().sum().sort_values(ascending=False)

  percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

  missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



  print(missing_data.head(number))

missing_values(all_data)
all_data = all_data.drop(['taster_twitter_handle','taster_name','region_2'],1)
#Nlp 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.naive_bayes import MultinomialNB

from nltk import word_tokenize

from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')

stop_words = stopwords.words('english')
# Always start with these features. They work (almost) everytime!

tfv = TfidfVectorizer(min_df=3,  max_features=64, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')



# Fitting TF-IDF to both training and test sets (semi-supervised learning)

tfv.fit(all_data.description)
all_data_description = tfv.transform(all_data.description)

all_data_description = pd.DataFrame(data=all_data_description.todense(),columns=tfv.get_feature_names())

all_data_description.head()
description_features = all_data_description.columns.tolist()
all_data =pd.concat([all_data,all_data_description],axis=1)
# Always start with these features. They work (almost) everytime!

tfv = TfidfVectorizer(min_df=3,  max_features=32, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')



# Fitting TF-IDF to both training and test sets (semi-supervised learning)

tfv.fit(all_data.title.fillna('other'))
all_data_title= tfv.transform(all_data.title.fillna('other'))

columns_name = [ x+'_title' for x in tfv.get_feature_names()]

all_data_title = pd.DataFrame(data=all_data_title.todense(),columns=columns_name)

all_data_title.head()
title_features = all_data_title.columns.tolist()
all_data = pd.concat([all_data,all_data_title],axis=1)
# Always start with these features. They work (almost) everytime!

tfv = TfidfVectorizer(min_df=3,  max_features=32, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')



# Fitting TF-IDF to both training and test sets (semi-supervised learning)

tfv.fit(all_data.designation.fillna('and'))
all_data_des= tfv.transform(all_data.title.fillna('other'))

columns_name = [ x+'_des' for x in tfv.get_feature_names()]

all_data_des = pd.DataFrame(data=all_data_des.todense(),columns=columns_name)

all_data_des.head()
des_features = all_data_des.columns.tolist()
all_data = pd.concat([all_data,all_data_des],axis=1)
# useful function 

def plot_cat(all_data,var,target='price') : 

  data = pd.concat([all_data[target], all_data[var]], axis=1)

  f, ax = plt.subplots(figsize=(30, 8))

  fig = sns.boxplot(x=var, y=target, data=data)

  fig.axis(ymin=0, ymax=100);

def cat_summary(all_data,var) : 

  new = all_data[var].value_counts() 

  print('there are ',len(new.values),'differnet value of ',var)

  res = pd.DataFrame(data=new.values,

                  index=new.index.to_list(),

                  columns=['number'])

  return res  
plot_cat(all_data,'country')
embed_cols=[i for i in all_data.select_dtypes(include=['object'])]

print('Categorical Features and Cardinality')

for i in embed_cols:

    print(i,all_data[i].nunique())
#converting data to list format to match the network structure

def preproc(X_train, X_val, X_test):



    input_list_train = dict()

    input_list_val = dict()

    input_list_test = dict()

    

    #the cols to be embedded: rescaling to range [0, # values)

    for c in embed_cols:

        cat_emb_name= c.replace(" ", "")+'_Embedding'

        raw_vals = X_train[c].unique()

        val_map = {}

        for i in range(len(raw_vals)):

            val_map[raw_vals[i]] = i       

        input_list_train[cat_emb_name]=X_train[c].map(val_map).values

        input_list_val[cat_emb_name]=X_val[c].map(val_map).fillna(0).values

        input_list_test[cat_emb_name]=X_test[c].map(val_map).fillna(0).values

    

    input_list_train['points']=X_train['points'].values

    input_list_val['points']=X_val['points'].values

    input_list_test['points']=X_test['points'].values



    input_list_train['description']=X_train[description_features].values

    input_list_val['description']=X_val[description_features].values

    input_list_test['description']=X_test[description_features].values



    input_list_train['title']=X_train[title_features].values

    input_list_val['title']=X_val[title_features].values

    input_list_test['title']=X_test[title_features].values



    input_list_train['desgination']=X_train[des_features].values

    input_list_val['desgination']=X_val[des_features].values

    input_list_test['desgination']=X_test[des_features].values

    

    return input_list_train, input_list_val, input_list_test


train = all_data[all_data['price']!=-1].reset_index(drop=True) 

test = all_data[all_data['price']==-1].reset_index(drop=True) 

embed_cols=['country','province','region_1','variety','winery']
for categorical_var in embed_cols:

    

    cat_emb_name= categorical_var.replace(" ", "")+'_Embedding'

  

    no_of_unique_cat  = train[categorical_var].nunique()

    embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 64))

  

    print('Categorica Variable:', categorical_var,

        'Unique Categories:', no_of_unique_cat,

        'Embedding Size:', embedding_size)
# Proper Naming of Categorical Features for Labelling NN Layers

for categorical_var in embed_cols :

    

    input_name= 'Input_' + categorical_var.replace(" ", "")

    print(input_name)
cat_features =['country','province','region_1','variety','winery']
def create_model(data, cat_cols  ):    

  input_models=[]

  output_embeddings=[]



  for categorical_var in cat_cols :

      

      #Name of the categorical variable that will be used in the Keras Embedding layer

      cat_emb_name= categorical_var.replace(" ", "")+'_Embedding'

    

      # Define the embedding_size

      no_of_unique_cat  = data[categorical_var].nunique() +1

      embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 24 ))

    

      #One Embedding Layer for each categorical variable

      input_model = layers.Input(shape=(1,),name=cat_emb_name)

      output_model = layers.Embedding(no_of_unique_cat, embedding_size, name=cat_emb_name+'emblayer')(input_model)

      output_model = layers.Reshape(target_shape=(embedding_size,))(output_model)    

    

      #Appending all the categorical inputs

      input_models.append(input_model)

    

      #Appending all the embeddings

      output_embeddings.append(output_model)

    

  #Other non-categorical data columns (numerical). 

  #I define single another network for the other columns and add them to our models list.





  input_numeric = layers.Input(shape=(1,),name='points')

  embedding_numeric = layers.Dense(16, kernel_initializer="uniform")(input_numeric) 

  input_models.append(input_numeric)

  output_embeddings.append(embedding_numeric)



  #description NN

  input_numeric = layers.Input(shape=(len(description_features),),name='description')



  embedding_numeric = layers.Dense(512, kernel_initializer="uniform")(input_numeric) 

  embedding_numeric = layers.Activation('relu')(embedding_numeric)

  embedding_numeric= layers.Dropout(0.6)(embedding_numeric)

  embedding_numeric = layers.Dense(256, kernel_initializer="uniform")(embedding_numeric) 

  embedding_numeric = layers.Activation('relu')(embedding_numeric)

  embedding_numeric= layers.Dropout(0.4)(embedding_numeric)



  input_models.append(input_numeric)

  output_embeddings.append(embedding_numeric)



  # Title NN

  input_numeric = layers.Input(shape=(len(title_features),),name='title')

  embedding_numeric = layers.Dense(32, kernel_initializer="uniform")(input_numeric) 

  embedding_numeric = layers.Activation('relu')(embedding_numeric)

  embedding_numeric= layers.Dropout(0.6)(embedding_numeric)

  input_models.append(input_numeric)

  output_embeddings.append(embedding_numeric)



  # desgination NN

  input_numeric = layers.Input(shape=(len(des_features),),name='desgination')

  embedding_numeric = layers.Dense(32, kernel_initializer="uniform")(input_numeric)

  embedding_numeric = layers.Activation('relu')(embedding_numeric)

  embedding_numeric= layers.Dropout(0.4)(embedding_numeric) 

  

  input_models.append(input_numeric)

  output_embeddings.append(embedding_numeric)



  #At the end we concatenate altogther and add other Dense layers

  output = layers.Concatenate()(output_embeddings)





  output = layers.Dense(1024, kernel_initializer="uniform")(output)

  output = layers.Activation('relu')(output)

  output= layers.Dropout(0.6)(output)

  output = layers.Dense(512, kernel_initializer="uniform")(output)

  output = layers.Activation('relu')(output)

  output= layers.Dropout(0.4)(output)

  output = layers.Dense(256, kernel_initializer="uniform")(output)

  output = layers.Activation('relu')(output)

  output= layers.Dropout(0.2)(output)

  output = layers.Dense(1)(output)



  model = Model(inputs=input_models, outputs=output)

  return model 
model = create_model(train , cat_features )

model.summary()
from sklearn.preprocessing import StandardScaler

scalar=StandardScaler()

scalar.fit(train['points'].values.reshape(-1, 1))

train['points']=scalar.transform(train['points'].values.reshape(-1, 1)) 

test['points']=scalar.transform(test['points'].values.reshape(-1, 1)) 
# try 

def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def rmse(predictions, targets): 

  return sqrt(mean_squared_error(predictions, targets))
X_train,X_vaild =model_selection.train_test_split(train)
y_train,y_valid = X_train.price,X_vaild.price

X_train,X_vaild,_ = preproc(X_train,X_vaild,train)

EPOCHS = 10 

BATCH_SIZE =1024

AUTO = tf.data.experimental.AUTOTUNE

train[description_features].shape
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_train, y_train))

    .repeat() 

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_vaild, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)
model = create_model(train, cat_features)

es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,

                                 verbose=5, baseline=None, restore_best_weights=True)

rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,

                                      patience=3, min_lr=1e-6, mode='max', verbose=1)

model.compile(optimizer = Adam(lr=5e-5), loss = 'mean_squared_error', metrics =[root_mean_squared_error])

n_steps = sum( [x.shape[0] for x in X_train.values()] ) // BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS

)



              


# summarize history for accuracy

plt.plot(train_history.history['loss'])

plt.plot(train_history.history['val_loss'])

plt.title('loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper right')

plt.show()

valid_fold_preds = model.predict(X_vaild)

print(rmse(y_valid.values, valid_fold_preds  ))
kf = model_selection.KFold(n_splits=10)



test_preds = np.zeros((len(test)))

score = []

counter = 0 

for fold, (train_index, test_index) in enumerate(kf.split(X=train)):

    counter = counter + 1 

    X_train, X_valid = train.iloc[train_index, :], train.iloc[test_index, :]

    y_train, y_valid = X_train['price'].values, X_valid['price'].values

    X_train ,X_vaild,X_test= preproc(X_train,X_valid,test)

   

    train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_train, y_train))

    .repeat() 

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

    )



    valid_dataset = (

        tf.data.Dataset

        .from_tensor_slices((X_vaild, y_valid))

        .batch(BATCH_SIZE)

        .cache()

        .prefetch(AUTO)

      )

    test_dataset = (

      tf.data.Dataset

    .from_tensor_slices(X_test)

    .batch(BATCH_SIZE)

    )

    model = create_model(train, cat_features)

    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,

                                 verbose=5, baseline=None, restore_best_weights=True)

    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,

                                          patience=3, min_lr=1e-6, mode='max', verbose=1)

    model.compile(optimizer = Adam(lr=5e-5), loss = 'mean_squared_error', metrics =[root_mean_squared_error])

    n_steps = sum( [x.shape[0] for x in X_train.values()] ) // BATCH_SIZE

    train_history = model.fit(

        train_dataset,

        steps_per_epoch=n_steps,

        validation_data=valid_dataset,

        epochs=10

    )



    valid_fold_preds = model.predict(valid_dataset, verbose=1)

    print(f'fold {fold} loss = ' ,  rmse(y_valid, valid_fold_preds  ))     

    score.append( rmse(y_valid, valid_fold_preds ))

    test_fold_preds = model.predict(test_dataset, verbose=1)

    test_preds += test_fold_preds.ravel()



    K.clear_session()
test_preds /= counter

test_ids = test.id.values

print("Saving submission file")

submission = pd.DataFrame.from_dict({

    'id': test_ids,

    'price': test_preds

})

submission.to_csv("submission.csv", index=False)
submission['price'].hist()
import torch 

from torch import nn,optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

from sklearn import preprocessing  

import transformers 

from tqdm.notebook import tqdm

import pandas as pd

import numpy as np

from sklearn import model_selection

from sklearn import metrics

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup
train = pd.read_csv('trainfinal.csv')
class simple_linear(nn.Module) : 

  def __init__(self) : 

    super(simple_linear,self).__init__()

    self.linear1 = nn.Linear(1,128)

    self.linear = nn.Linear(128,1) 

  def forward(self,data) :

    points = (data['points'].view(-1,1)).to(device,dtype=torch.float)  

    x=F.relu(self.linear1(points))

    out = self.linear(x)

    return out 
from sklearn.preprocessing import StandardScaler

scalar=StandardScaler()

scalar.fit(train['points'].values.reshape(-1, 1))

train['points']=scalar.transform(train['points'].values.reshape(-1, 1)) 

class data_set : 

  def __init__(self,df) : 

    self.points = df.points 

    self.price = df.price 

  def __len__(self) : 

    return(len(self.price)) 

  def __getitem__(self,index) : 

    return {

        'points' : torch.tensor(self.points[index]) , 

        'price'  : torch.tensor(self.price[index])

    }
categorical_features = ['country','province','region_1','variety','winery']
for f in categorical_features : 

  label_encoder = preprocessing.LabelEncoder()

  label_encoder.fit(train[f].astype('str'))

  train[f] = label_encoder.transform(train[f].astype('str').fillna('-1'))
class EmbDataSet() : 

  def __init__(self,df,cat_features) : 

    self.df = df

    self.categorical  = cat_features 

  def __len__(self) : 

    return len(self.df)

  def __getitem__(self,item) : 

    out = dict()

    for i in self.categorical : 

      out[i] = torch.tensor( self.df[i].values[item] , dtype=torch.long )

    

    out['points'] = torch.tensor(self.df['points'].values[item], dtype=torch.float )



    out['price'] = torch.tensor(self.df['price'].values[item],dtype=float ) 

    return out 

    
def get_emb_dim(df,categorical):

  output=[]

  for categorical_var in categorical:

      

      cat_emb_name= categorical_var.replace(" ", "")+'_Embedding'

    

      no_of_unique_cat  = train[categorical_var].nunique()

      embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 24))

      output.append((no_of_unique_cat,embedding_size))    

      print('Categorica Variable:', categorical_var,

          'Unique Categories:', no_of_unique_cat,

          'Embedding Size:', embedding_size)

  return output
emb_size = get_emb_dim(train,categorical_features)
class Embedding_model(nn.Module) : 

  def __init__(self,cat,emb_size) :

    super(Embedding_model,self).__init__()

    self.cat =cat 

    self.emb_size = emb_size 

    outputs_cat = nn.ModuleList()

    for inp , emb  in emb_size :

      embedding_layer = nn.Embedding(inp+2,emb)

                                   

      outputs_cat.append(embedding_layer)

    self.outputs_cat = outputs_cat 



    n_emb = sum([e[1] for e in self.emb_size])

    self.num = nn.Sequential( nn.Linear(1,128),

                              nn.Dropout(0.4) 

                              )

    self.embedding = nn.Sequential( nn.Linear(n_emb,384),

                                    nn.Dropout(0.4)

                                    )

    

    self.fc = nn.Sequential(  



                            

                              nn.Linear(512,256),

                              nn.Dropout(0.3),

                              nn.ReLU(),

                              nn.Linear(256,1)

    )



        

  def forward(self,data)  : 

    outputs_emb = [] 

    for i in range(len(self.cat)) : 

      inputs = data[self.cat[i]].to(device,dtype=torch.long) 

      out = self.outputs_cat[i](inputs)

      outputs_emb.append(out) 

    

    x_cat = torch.cat(outputs_emb,dim= 1)

    x_cat = self.embedding(x_cat)



    inputs = (data['points'].view(-1,1)).to(device,dtype=torch.float)

    inputs = self.num(inputs)

    

    x_all = torch.cat([inputs,x_cat],dim=1) 

    x_final = self.fc(x_all)



    return x_final
class BertBaseUncased(nn.Module) :

    def __init__(self) : 

        super(BertBaseUncased,self).__init__() 

        self.bert = transformers.BertModel.from_pretrained(BERT_PATH) 

        self.bert_drop = nn.Dropout(0.4) 

        self.out = nn.Linear(768,1) 

    def forward(self,d) : 

        ids = d["ids"]

        token_type_ids = d["token_type_ids"]

        mask = d["mask"]



        ids = ids.to(device, dtype=torch.long)

        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        out1,out2 = self.bert( 

            ids , 

            attention_mask = mask , 

            token_type_ids = token_type_ids 

        )

        bo = self.bert_drop(out2) 

        output = self.out(bo) 

        return output 
def loss_fn(outputs, targets):

    return nn.MSELoss()(outputs, targets.view(-1, 1))
def train_fn(data_loader, model, optimizer, scheduler):

  model.train()

  tr_loss = 0 

  counter = 0 

  for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):

    targets = d["price"]

    targets = targets.to(device, dtype=torch.float)

    optimizer.zero_grad()

    outputs = model(d)



    loss = loss_fn(outputs, targets)

    tr_loss += loss.item()

    counter +=1 

    loss.backward()

    optimizer.step()

  return tr_loss/counter
def eval_fn(data_loader, model):

  model.eval()

  fin_loss = 0

  counter = 0

  with torch.no_grad():

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):

      targets = d["price"]

      targets = targets.to(device, dtype=torch.float)

      outputs = model(d)

      loss = loss_fn(outputs, targets)

      fin_loss +=loss.item()

      counter += 1

    return fin_loss/counter 
df_train, df_valid = model_selection.train_test_split(

        train,

        test_size=0.2,

        random_state=42,

    )

DEVICE =torch.device("cuda")

device = torch.device("cuda")

def run(model,EPOCHS):

    

    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=TRAIN_BATCH_SIZE,

        num_workers=4

    )

    

    

    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=VALID_BATCH_SIZE,

        num_workers=1

    )



    device = torch.device("cuda")

    

    

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]



    num_train_steps = int(len(train_data_loader)) * EPOCHS

    optimizer = AdamW(optimizer_parameters, lr=1e-3)

    scheduler = get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=0,

        num_training_steps=num_train_steps

    )





    model = nn.DataParallel(model)



    train_loss =  []

    val_loss = []

    for epoch in range(EPOCHS):

       

        tr_loss=train_fn(train_data_loader, model, optimizer, scheduler)

        train_loss.append(tr_loss)

        print(f" train_loss  = {np.sqrt(tr_loss)}")



        

        val = eval_fn(valid_data_loader, model)

        val_loss.append(val)

        print(f" val_loss  = {np.sqrt(val)}")



        scheduler.step()

    return val_loss,train_loss
TRAIN_BATCH_SIZE =128

VALID_BATCH_SIZE = 64
train_dataset = EmbDataSet(

        df_train,categorical_features

    )



valid_dataset = EmbDataSet(

        df_valid,

        categorical_features



    )
model = Embedding_model(categorical_features,emb_size)

model.to(device)

getattr(tqdm, '_instances', {}).clear()

val_loss,tr_loss = run(model,30)


# summarize history for accuracy

plt.plot(tr_loss)

plt.plot(val_loss)

plt.title('loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper right')

plt.show()
# summarize history for accuracy

plt.plot(np.sqrt(tr_loss))

plt.plot(np.sqrt(val_loss))

plt.title('mean_squared_error')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper right')

plt.show()
class Final_Data_set : 

    def __init__(self,df,cat_features) : 

        self.df = df

        self.categorical  = cat_features 

        self.description = df['description'].values

        self.tokenizer = TOKENZIER 

        self.max_len = MAX_Len 

    def __len__(self) : 

        return len(self.description) 

    def __getitem__(self, item):

      out = dict()

        

      # bert input  



      text = str(self.description[item])

      text = " ".join(text.split())



      inputs = self.tokenizer.encode_plus(

              text,

              None,

              add_special_tokens=True,

              max_length=self.max_len

          )



      ids = inputs["input_ids"]

      mask = inputs["attention_mask"]

      token_type_ids = inputs["token_type_ids"]



      padding_length = self.max_len - len(ids)

      ids = ids + ([0] * padding_length)

      mask = mask + ([0] * padding_length)

      token_type_ids = token_type_ids + ([0] * padding_length)



      out['ids']= torch.tensor(ids, dtype=torch.long)

      out['mask']= torch.tensor(mask, dtype=torch.long)

      out['token_type_ids'] = torch.tensor(token_type_ids, dtype=torch.long)

      

      # other inputs 

      for i in self.categorical : 

        out[i] = torch.tensor( self.df[i].values[item] , dtype=torch.long )

    

      out['points'] = torch.tensor(self.df['points'].values[item], dtype=torch.float )

      out['price'] = torch.tensor(self.df['price'].values[item],dtype=float ) 



      return out
class Final_Model(nn.Module) : 

  def __init__(self,cat,emb_size) :

    super(Final_Model,self).__init__()





    # Embeddings layers 

    self.cat =cat 

    self.emb_size = emb_size 

    outputs_cat = nn.ModuleList()

    for inp , emb  in emb_size :

      embedding_layer = nn.Embedding(inp+2,emb)

                                   

      outputs_cat.append(embedding_layer)

    

    self.outputs_cat = outputs_cat 

    n_emb = sum([e[1] for e in self.emb_size])

    self.embedding = nn.Sequential( nn.Linear(n_emb,384),

                                    nn.Dropout(0.4)

                                    )

    

    #Numerical layers 

    self.num = nn.Sequential( nn.Linear(1,128),

                              nn.Dropout(0.4) 

                              )

    #BERT input

    self.bert = transformers.BertModel.from_pretrained(BERT_PATH) 

    self.bert_drop = nn.Dropout(0.4) 

    self.bert_out = nn.Linear(768,512) 

    



    #putting it all together

    self.fc = nn.Sequential(  



                              nn.Linear(1024,512),

                              nn.Dropout(0.4),

                              nn.ReLU(),

                              nn.Linear(512,256),

                              nn.Dropout(0.3),

                              nn.ReLU(),

                              nn.Linear(256,1)

    )



        

  def forward(self,data)  : 

    

    # Categorical features 

    outputs_emb = [] 

    for i in range(len(self.cat)) : 

      inputs = data[self.cat[i]].to(device,dtype=torch.long) 

      out = self.outputs_cat[i](inputs)

      outputs_emb.append(out) 

    x_cat = torch.cat(outputs_emb,dim= 1)

    x_cat = self.embedding(x_cat)



    #numrique features

    inputs = (data['points'].view(-1,1)).to(device,dtype=torch.float)

    inputs = self.num(inputs)

    

    #description input

    ids = data["ids"].to(device, dtype=torch.long)

    token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)

    mask = data["mask"].to(device, dtype=torch.long)

    out1,out2 = self.bert( 

                             ids , 

                             attention_mask = mask , 

                             token_type_ids = token_type_ids 

                         )

    bo = self.bert_drop(out2) 

    output = self.bert_out(bo) 



    #putting it all together 

    x_all = torch.cat([output,inputs,x_cat],dim=1) 

    x_final = self.fc(x_all)



    return x_final
MAX_Len = 128 

TRAIN_BATCH_SIZE =96

VALID_BATCH_SIZE = 32

BERT_PATH = 'bert-base-uncased'

TOKENZIER = transformers.BertTokenizer.from_pretrained(BERT_PATH ,do_lower_case = True )
train_dataset = Final_Data_set(

        df_train,categorical_features

    )



valid_dataset = Final_Data_set(

        df_valid,

        categorical_features



    )
model = Final_Model(categorical_features,emb_size)

model.to(device)

getattr(tqdm, '_instances', {}).clear()

#val_loss,tr_loss = run(model,30)