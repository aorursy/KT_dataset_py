import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import plotly.graph_objs as go
import plotly.plotly as py
import os
import matplotlib.pyplot as pl
files = os.listdir("../input")
print(files)

# Any results you write to the current directory are saved as output.
COLUMNS = ['Country', 'Happiness.Rank', 'Happiness.Score', 'Whisker.high',
       'Whisker.low', 'Economy..GDP.per.Capita.', 'Family',
       'Health..Life.Expectancy.', 'Freedom', 'Generosity',
       'Trust..Government.Corruption.', 'Dystopia.Residual']
FEATURES = ['Whisker.high',
       'Whisker.low', 'Economy..GDP.per.Capita.', 'Family',
       'Health..Life.Expectancy.', 'Freedom', 'Generosity',
       'Trust..Government.Corruption.', 'Dystopia.Residual']
LABEL = 'Happiness.Score'
for i,f in enumerate(files):
    files[i] = pd.read_csv('../input/'+f,skipinitialspace=True,
                         skiprows=1, names=COLUMNS)

files[0].head()    
training_set = files[2] #2017
training_set.columns
training_set.describe()
training_set.isna().sum().plot('bar');
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
tf.logging.set_verbosity(tf.logging.INFO)

def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)

test_set = pd.read_csv("../input/2017.csv", skipinitialspace=True, skiprows=100, names=COLUMNS)
test_set.info()
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                    hidden_units=[20, 10])
# Train
regressor.train(input_fn=get_input_fn(training_set), steps=1000)

# Evaluate loss over one epoch of test_set.
ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
ev
prediction_set = pd.read_csv("../input/2017.csv", skipinitialspace=True, skiprows=130, names=COLUMNS)
p = prediction_set[FEATURES].head()
p
test_in = tf.estimator.inputs.pandas_input_fn(p, shuffle=False)
preds = regressor.predict(input_fn=test_in)
for i,pr in enumerate(preds):
    print(f'''{prediction_set['Country'][i]} has Happiness Score:- {round(float(pr['predictions'][0]),3)}''')
training_set[129:134][COLUMNS[:3]]