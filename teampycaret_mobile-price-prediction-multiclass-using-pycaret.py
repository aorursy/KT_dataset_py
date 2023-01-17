import pandas as pd

train = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')

train.head()
!pip install pycaret
from pycaret.classification import *

clf1 = setup(data = train, target = 'price_range', session_id = 786, silent = True)



#silent is True to perform unattended run when kernel is executed.
%%time

compare_models()
# create knn model

knn = create_model('knn')
# create catboost model

catboost = create_model('catboost')
# tune knn model

tuned_knn = tune_model('knn', optimize = 'Accuracy', n_iter = 100)
# parameters of tuned_knn

print(tuned_knn)
tuned_catboost = tune_model('catboost', optimize = 'Accuracy', n_iter = 100)
tuned_lightgbm = tune_model('lightgbm', optimize = 'Accuracy', n_iter = 100)
tuned_ada = tune_model('ada', optimize = 'Accuracy', n_iter = 100)
tuned_lr = tune_model('lr', optimize = 'Accuracy', n_iter = 100)
dt = create_model('dt')
bagged_dt = ensemble_model(dt, n_estimators = 100)
# auc

plot_model(bagged_dt)
# confusion matrix

plot_model(bagged_dt, plot = 'confusion_matrix')
# boundary

plot_model(bagged_dt, plot = 'boundary')
# vc

plot_model(bagged_dt, plot = 'dimension')
pred_holdout = predict_model(bagged_dt)
final_dt = finalize_model(bagged_dt)
test = pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')

test.head()
predictions = predict_model(final_dt, data=test)
predictions.head()