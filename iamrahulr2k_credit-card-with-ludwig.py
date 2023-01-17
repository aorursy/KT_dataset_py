# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import confusion_matrix



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

        name: X1

        type: numerical

    -

        name: X2

        type: numerical

    -

        name: X3

        type: numerical

    -

        name: X4

        type: numerical    

    -

        name: X5

        type: numerical

    -

        name: X6

        type: numerical    

    -

        name: X7

        type: numerical

    -

        name: X8

        type: numerical    

    -

        name: X9

        type: numerical

    -

        name: X10

        type: numerical    

    -

        name: X11

        type: numerical

    -

        name: X12

        type: numerical

    -

        name: X13

        type: numerical

    -

        name: X14

        type: numerical

    -

        name: X15

        type: numerical

    -

        name: X16

        type: numerical    

    -

        name: X17

        type: numerical

    -

        name: X18

        type: numerical    

    -

        name: X19

        type: numerical

    -

        name: X20

        type: numerical    

    -

        name: X21

        type: numerical

    -

        name: X22

        type: numerical    

    -

        name: X23

        type: numerical





output_features:

    -

        name: Y

        type: binary

"""



# train a model

model_definition = yaml.load(UCI_yaml)

print(yaml.dump(model_definition))

model = LudwigModel(model_definition)

training_dataframe = pd.read_excel('../input/NEWDATAFILE.xls')

print("training...")

train_stats = model.train(training_dataframe, logging_level=logging.INFO)

print("finished training.\n")
# obtain predictions

test_dataframe = pd.read_excel('../input/NEWTESTFILE.xls')

print("predicting...")

predictions = model.predict(test_dataframe, logging_level=logging.INFO)

print("finised predicting\n")

model.close()

print(predictions.head())
from sklearn.metrics import confusion_matrix

print (confusion_matrix(test_dataframe['Y'], predictions['Y_predictions']))


