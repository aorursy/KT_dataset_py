%load_ext tensorboard
%tensorboard --logdir logs


!pip install transformers
!pip install simpletransformers
!pip install wandb
import sys
sys.path.append('../../')
#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
!wandb login

!pip install datasets
from datasets import load_dataset
dataset = load_dataset('imdb')
dataset_train = load_dataset('imdb',split="train")
dataset_test = load_dataset('imdb',split="test")
dataset_train[0]
df_train=pd.DataFrame(dataset_train)
df_train
train_df=pd.DataFrame(df_train['text'])
train_df['label']=df_train['label']
train_df
df_valid=pd.DataFrame(dataset_test)
valid_df=pd.DataFrame(df_valid['text'])
valid_df['label']=df_valid['label']
args = {
    'fp16':False,
    'wandb_project': 'bert-large24-imdb',
    'num_train_epochs':5,
    'overwrite_output_dir':True,
    'learning_rate': 1e-5,
    'config':{'num_hidden_layers':24}
    
    }

model = ClassificationModel('bert', 'bert-large-cased', use_cuda=True,args=args) 

model.config
x=model.get_named_parameters()
model

model.train_model(train_df, output_dir='bert-large24-imdb')

result, model_outputs, wrong_predictions = model.eval_model(valid_df)

model.config
(result['tp']+result['tn'])/(result['tp']+result['tn']+result['fp']+result['fn'])
model.predict(['The movie was really good'])

valid_df
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred=model.predict(valid_df['text'])
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

yx=list(y_pred)
len(yx[0])
y_pred=np.array(yx[0])
y_pred
type(y_pred)
len(y_pred)
conf=confusion_matrix(valid_df['label'],y_pred)
conf
plt.figure(figsize=(10,10))
sns.heatmap(conf, annot=True,fmt='.0f')
accuracy_score(valid_df['label'],y_pred)
model.save_model('/content/bert-large24-imdb/trainedmodel',model=model.model)