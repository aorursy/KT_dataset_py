# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install ludwig
from ludwig import LudwigModel

import yaml

import logging
UCI_yaml = """

input_features:

    -

        name: var_0

        type: numerical

    -

        name: var_1

        type: numerical

    -

        name: var_2

        type: numerical

    -

        name: var_3

        type: numerical    

    -

        name: var_4

        type: numerical

    -

        name: var_5

        type: numerical    

    -

        name: var_6

        type: numerical

    -

        name: var_7

        type: numerical    

    -

        name: var_8

        type: numerical

    -

        name: var_9

        type: numerical    

    -

        name: var_10

        type: numerical

    -

        name: var_11

        type: numerical

    -

        name: var_12

        type: numerical

    -

        name: var_13

        type: numerical

    -

        name: var_14

        type: numerical

    -

        name: var_15

        type: numerical    

    -

        name: var_16

        type: numerical

    -

        name: var_17

        type: numerical

        

output_features:

    -

        name: target

        type: binary

"""



# train a model

model_definition = yaml.load(UCI_yaml)

print(yaml.dump(model_definition))

model = LudwigModel(model_definition)

training_dataframe = pd.read_csv('../input/train.csv')

print("training...")

train_stats = model.train(training_dataframe, logging_level=logging.INFO)

print("finished training.\n")
# obtain predictions

test_dataframe = pd.read_csv('../input/test.csv')

print("predicting...")

predictions = model.predict(test_dataframe, logging_level=logging.INFO)

print("finised predicting\n")

model.close()

print(predictions.head())