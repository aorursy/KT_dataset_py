import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



%matplotlib inline
h2o.init(max_mem_size = 2)           

h2o.remove_all()
higgs_train = h2o.import_file('../input/training/training.csv')

higgs_test = h2o.import_file('../input/test/test.csv')
higgs_train.describe()
higgs_test.describe()
train, valid, test = higgs_train.split_frame([0.6, 0.2], seed = 2019)

higgs_X = higgs_train.col_names[1: -1]

higgs_y = higgs_train.col_names[-1]
higgs_model_v1 = H2ODeepLearningEstimator(model_id = 'higgs_v1', epochs = 1, variable_importances = True)

higgs_model_v1.train(higgs_X, higgs_y, training_frame = train, validation_frame = valid)

print(higgs_model_v1)
var_df = pd.DataFrame(higgs_model_v1.varimp(), columns = ['Variable', 'Relative Importance', 'Scaled Importance', 'Percentage'])

var_df.head(10)
higgs_v1_df = higgs_model_v1.score_history()

plt.plot(higgs_v1_df['training_classification_error'], label="training_classification_error")

plt.plot(higgs_v1_df['validation_classification_error'], label="validation_classification_error")

plt.title("Higgs Deep Learner")

plt.legend();
pred = higgs_model_v1.predict(test[1:-1]).as_data_frame(use_pandas=True)

test_actual = test.as_data_frame(use_pandas=True)['Label']

(test_actual == pred['predict']).mean()
higgs_model_v2 = H2ODeepLearningEstimator(model_id = 'higgs_v2', hidden = [32, 32, 32], 

                                          epochs = 1000000, score_validation_samples = 10000, 

                                          stopping_rounds = 2, stopping_metric = 'misclassification', 

stopping_tolerance = 0.01)

higgs_model_v2.train(higgs_X, higgs_y, training_frame = train, validation_frame = valid)
higgs_v2_df = higgs_model_v2.score_history()

plt.plot(higgs_v2_df['training_classification_error'], label="training_classification_error")

plt.plot(higgs_v2_df['validation_classification_error'], label="validation_classification_error")

plt.title("Higgs Deep Learner (Early Stop)")

plt.legend();
pred = higgs_model_v2.predict(test[1:-1]).as_data_frame(use_pandas=True)

test_actual = test.as_data_frame(use_pandas=True)['Label']

(test_actual == pred['predict']).mean()
higgs_model_v2.varimp_plot();
from h2o.automl import H2OAutoML

aml = H2OAutoML(max_models = 10, max_runtime_secs=100, seed = 1)

aml.train(higgs_X, higgs_y, training_frame = train, validation_frame = valid)

aml.leaderboard