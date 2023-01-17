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
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df_train.head()
! pip uninstall torch torchvision -y

! pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

!pip install -U transformers

!pip install -U simpletransformers  
from simpletransformers.classification import ClassificationModel

import pandas as pd

import logging





logging.basicConfig(level=logging.INFO)

transformers_logger = logging.getLogger("transformers")

transformers_logger.setLevel(logging.WARNING)
# splitting the data into training and test dataset

X = df_train['text']

y = df_train['target']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_df = pd.DataFrame(X_train)

train_df['target'] = y_train



eval_df = pd.DataFrame(X_test)

eval_df['target'] = y_test
train_df.shape, eval_df.shape
from simpletransformers.classification import ClassificationModel

import pandas as pd

import logging

import sklearn





logging.basicConfig(level=logging.INFO)

transformers_logger = logging.getLogger("transformers")

transformers_logger.setLevel(logging.WARNING)

'''

args = {

   'output_dir': 'outputs/',

   'cache_dir': 'cache/',

   'fp16': True,

   'fp16_opt_level': 'O1',

   'max_seq_length': 256,

   'train_batch_size': 8,

   'eval_batch_size': 8,

   'gradient_accumulation_steps': 1,

   'num_train_epochs': 3,

   'weight_decay': 0,

   'learning_rate': 4e-5,

   'adam_epsilon': 1e-8,

   'warmup_ratio': 0.06,

   'warmup_steps': 0,

   'max_grad_norm': 1.0,

   'logging_steps': 50,

   'evaluate_during_training': False,

   'save_steps': 2000,

   'eval_all_checkpoints': True,

   'use_tensorboard': True,

   'overwrite_output_dir': True,

   'reprocess_input_data': False,

}



'''



# Create a ClassificationModel

model = ClassificationModel('xlnet', 'xlnet-base-cased', args={'num_train_epochs':2, 'train_batch_size':16, 'max_seq_length':128}) # You can set class weights by using the optional weight argument



# Train the model

model.train_model(train_df)



# Evaluate the model

result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
result
predictions, raw_outputs = model.predict(df_test.text.tolist())
sample_sub=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_sub['target'] = predictions



sample_sub.to_csv("submission_01092020_xlnet_base.csv", index=False)