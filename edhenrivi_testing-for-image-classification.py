# Prepare ludwig

!pip install ludwig

!python -m spacy download en


# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ludwig import LudwigModel # For ludwig



from tqdm import tqdm_notebook



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os, io

import requests

import tempfile, shutil



# Any results you write to the current directory are saved as output.
print(os.listdir("../input"))

print(os.listdir("../input/histopathologic-cancer-detection"))
train_file = '../input/histcancer-wpaths3/1_train_labels.csv'

train_df = pd.read_csv(train_file)

train_df.head()
test_file = '../input/histcancer-wpaths3/1_sample_submission.csv'

test_df = pd.read_csv(test_file)

test_df.head()
train_df.columns
test_df.columns
model_definition = {

    "input_features": [{

            "name": "id",

            "type": "image",

            "encoder": "stacked_cnn"

        }

    ], 

    "output_features": [{

            "name": "label", 

            "type": "category"

        }

     ],

    "training": {"epochs": 10}

}

ludwig_model = LudwigModel(model_definition,

                           logging_level=0)
train_stats = ludwig_model.train(data_csv=train_file,

                                skip_save_model=True,

                                skip_save_progress=True,

                                skip_save_log=True,

                                skip_save_processed_input=True,

                                logging_level=0)
train_stats
dir(ludwig_model)
for i in dir(ludwig_model):

    print(i)

    print(dir(i))
predictions = ludwig_model.test(data_csv=test_file,

                                   #data_df=test_df,

                                   return_type='dict',

                                  logging_level=0)
model.close()
for i in predictions:

    print(i)