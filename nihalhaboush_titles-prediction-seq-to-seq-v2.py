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
! pip uninstall torch torchvision -y
! pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!pip install -U transformers
!pip install -U simpletransformers 
df = pd.read_json('../input/arxiv-papers-2010-2020/arXiv_title_abstract_20200809_2011_2020.json')

df
df.abstract
text = ""

for i,r in df.iterrows():

    text += r.abstract + '\n'
text[:1000]
len(text)
len(text.split())
df = df.sample(1000)
df
df.drop(['year'],axis=1,inplace=True)

df.info()
df
import logging

logging.basicConfig(level=logging.INFO)

transformers_logger = logging.getLogger("transformers")

transformers_logger.setLevel(logging.WARNING)
df.columns = ['target_text', 'input_text']

df = df.dropna()

df
val_df = df.sample(frac=0.1, random_state=1007)

train_df = df.drop(val_df.index)

test_df = train_df.sample(frac=0.1, random_state=1007)

train_df.drop(test_df.index, inplace=True)
train_df
val_df
test_df
train_df.shape, val_df.shape
import logging

import pandas as pd

from simpletransformers.t5 import T5Model



logging.basicConfig(level=logging.INFO)

transformers_logger = logging.getLogger("transformers")

transformers_logger.setLevel(logging.WARNING)



train_df['prefix'] = "summarize"

val_df['prefix'] = "summarize"



# "max_seq_length": 512,

model_args = {

    "reprocess_input_data": True,

    "overwrite_output_dir": True,

    "train_batch_size": 8,

    "num_train_epochs": 2,

}



# Create T5 Model

model = T5Model("t5-small", args=model_args, use_cuda=False)



# Train T5 Model on new task

model.train_model(train_df)
test_df['prefix'] = "summarize"
# Evaluate T5 Model on new task

results = model.eval_model(test_df)



# Predict with trained T5 model

#print(model.predict(["convert: four"]))

results
random_num = 1

actual_title = test_df.iloc[random_num]['target_text']

actual_abstract = ["summarize: "+test_df.iloc[random_num]['input_text']]

predicted_title = model.predict(actual_abstract)



print(f'Actual Title: {actual_title}')

print(f'Predicted Title: {predicted_title}')

print(f'Actual Abstract: {actual_abstract}')
random_num = 5

actual_title = test_df.iloc[random_num]['target_text']

actual_abstract = ["summarize: "+test_df.iloc[random_num]['input_text']]

predicted_title = model.predict(actual_abstract)



print(f'Actual Title: {actual_title}')

print(f'Predicted Title: {predicted_title}')

print(f'Actual Abstract: {actual_abstract}')