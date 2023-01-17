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
papers = pd.DataFrame({

    'title': df.title,

    'abstract': df.abstract

})

papers.head()

papers = papers.sample(10000)

df = papers

df
# Simpletransformers implementation of T5 model expects a data to be a dataframe with 3 columns: <prefix>, <input_text>, <target_text>



# <prefix>: A string indicating the task to perform. (E.g. "question", "stsb")

# <input_text>: The input text sequence (we will use Paper's abstract as input_text )

# <target_text: The target sequence (we will use Paper's title as output_text )
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
# your code here 

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

    "train_batch_size": 10,

    "num_train_epochs": 3,

}



# Create T5 Model

model = T5Model("t5-small", args=model_args, use_cuda=True)



# Train T5 Model on new task

model.train_model(train_df)
test_df['prefix'] = "summarize"
# your code here 

# Evaluate T5 Model on new task

results = model.eval_model(test_df)



# Predict with trained T5 model

#print(model.predict(["convert: four"]))

results
# your code here 

random_num = 3

actual_title = test_df.iloc[random_num]['target_text']

actual_abstract = ["summarize: "+test_df.iloc[random_num]['input_text']]

predicted_title = model.predict(actual_abstract)



print(f'Actual Title: {actual_title}')

print(f'Predicted Title: {predicted_title}')

print(f'Actual Abstract: {actual_abstract}')
random_num = 7

actual_title = test_df.iloc[random_num]['target_text']

actual_abstract = ["summarize: "+test_df.iloc[random_num]['input_text']]

predicted_title = model.predict(actual_abstract)



print(f'Actual Title: {actual_title}')

print(f'Predicted Title: {predicted_title}')

print(f'Actual Abstract: {actual_abstract}')