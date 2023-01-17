# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import h2o

from h2o.automl import H2OAutoML

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

h2o.init()



# Any results you write to the current directory are saved as output.
# Load data into H2O

df = h2o.import_file("../input/train.csv")
df.describe()
y = "C2"

x = df.columns

x.remove(y)

x.remove("C1")
aml = H2OAutoML(max_runtime_secs = 30000, seed = 1,balance_classes=True)

aml.train(x = x, y = y, training_frame = df)
lb = aml.leaderboard
lb.head()
lb.head(rows=lb.nrows)
# Get model ids for all models in the AutoML Leaderboard

model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])

# Get the "All Models" Stacked Ensemble model

se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])

# Get the Stacked Ensemble metalearner model

metalearner = h2o.get_model(se.metalearner()['name'])
metalearner.coef_norm()
%matplotlib inline

metalearner.std_coef_plot()
aml.leader.model_id
h2o.save_model(aml.leader, path = "../working/")
model = h2o.load_model(aml.leader.model_id)
df_test = h2o.import_file("../input/test.csv")

display(df_test.head())

df_test = df_test[1:,:]
predict = aml.predict(df_test)
predict.shape
submission = h2o.import_file("../input/sample_submission.csv")

submission['target1'] = predict

submission = submission.as_data_frame()

submission.columns = ['id', 'target1', 'target']

submission.pop('target1')

submission.to_csv("h2o.csv", index=False)

submission.head()