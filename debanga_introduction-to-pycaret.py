# Install PyCaret
!pip install pycaret
!pip install --upgrade pycaret #if you have installed beta version in past, run the below code to upgrade

import numpy as np
import pandas as pd
import os, sys
from IPython.display import display

from pycaret.utils import version

# PyCaret version
version()
# Importing dataset
from pycaret.datasets import get_data

# For classification examples
print('Classification: Diabetes Data')
diabetes = get_data('diabetes')

# For regression examples
print('Regression: Boston Data')
boston = get_data('boston')

# Importing module and initializing setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')

# comparing all models
compare_models()
# Importing module and initializing setup
from pycaret.regression import *
reg1 = setup(data = boston, target = 'medv')

# comparing all models
compare_models()
clf1 = setup(data = diabetes, target = 'Class variable')
# creating logistic regression model
lr = create_model('lr')
reg1 = setup(data = boston, target = 'medv')

# creating xgboost model
xgboost = create_model('xgboost')
clf1 = setup(data = diabetes, target = 'Class variable')
# tuning LightGBM Model
tuned_lightgbm = tune_model('lightgbm')
# Importing dataset
from pycaret.datasets import get_data
boston = get_data('boston')
# Importing module and initializing setup
from pycaret.regression import *
reg1 = setup(data = boston, target = 'medv')
# tuning Random Forest model
tuned_rf = tune_model('rf', n_iter = 50, optimize = 'mae')
from pycaret.clustering import *
clu1 = setup(data = diabetes)
# Tuning K-Modes Model
tuned_kmodes = tune_model('kmodes', supervised_target = 'Class variable')
from pycaret.anomaly import *
ano1 = setup(data = boston)
# Tuning Isolation Forest Model
tuned_iforest = tune_model('iforest', supervised_target = 'medv')
kiva = get_data('kiva')
# Importing module and initializing setup
from pycaret.nlp import *
nlp1 = setup(data = kiva, target = 'en')
# Tuning LDA Model
tuned_lda = tune_model('lda', supervised_target = 'status')
clf1 = setup(data = diabetes, target = 'Class variable')
# creating decision tree model
dt = create_model('dt')
# ensembling decision tree model (bagging)
dt_bagged = ensemble_model(dt)
# Importing dataset
from pycaret.datasets import get_data
boston = get_data('boston')
# Importing module and initializing setup
from pycaret.regression import *
reg1 = setup(data = boston, target = 'medv')
# creating decision tree model
dt = create_model('dt')
# ensembling decision tree model (boosting)
dt_boosted = ensemble_model(dt, method = 'Boosting')
clf1 = setup(data = diabetes, target = 'Class variable')
# blending all models
blend_all = blend_models()
# Importing dataset
from pycaret.datasets import get_data
boston = get_data('boston')
# Importing module and initializing setup
from pycaret.regression import *

reg1 = setup(data = boston, target = 'medv')
# creating multiple models for blending
dt = create_model('dt')
catboost = create_model('catboost')
lightgbm = create_model('lightgbm')
# blending specific models
blender = blend_models(estimator_list = [dt, catboost, lightgbm])
clf1 = setup(data = diabetes, target = 'Class variable')
# create individual models for stacking
ridge = create_model('ridge')
lda = create_model('lda')
gbc = create_model('gbc')
xgboost = create_model('xgboost')
# stacking models
stacker = stack_models(estimator_list = [ridge,lda,gbc], meta_model = xgboost)
# Importing dataset
from pycaret.datasets import get_data
boston = get_data('boston')
# Importing module and initializing setup
from pycaret.regression import *

reg1 = setup(data = boston, target = 'medv')
# creating multiple models for multiple layer stacking
catboost = create_model('catboost')
et = create_model('et')
lightgbm = create_model('lightgbm')
xgboost = create_model('xgboost')
ada = create_model('ada')
rf = create_model('rf')
gbr = create_model('gbr')
# creating multiple layer stacking from specific models
stacknet = create_stacknet([[lightgbm, xgboost, ada], [et, gbr, catboost, rf]])
clf1 = setup(data = diabetes, target = 'Class variable')
# creating a model
lr = create_model('lr')
# plotting a model
plot_model(lr)
# Importing dataset
from pycaret.datasets import get_data
boston = get_data('boston')
# Importing module and initializing setup
from pycaret.regression import *

reg1 = setup(data = boston, target = 'medv')
# creating a model
lr = create_model('lr')
# plotting a model
plot_model(lr)
# Importing dataset
from pycaret.datasets import get_data
jewellery = get_data('jewellery')
# Importing module and initializing setup
from pycaret.clustering import *
clu1 = setup(data = jewellery)
# creating a model
kmeans = create_model('kmeans')
# plotting a model
plot_model(kmeans)
# Importing dataset
from pycaret.datasets import get_data
anomalies = get_data('anomaly')
# Importing module and initializing setup
from pycaret.anomaly import *
ano1 = setup(data = anomalies)
# creating a model
iforest = create_model('iforest')
# plotting a model
plot_model(iforest)
# Importing dataset
from pycaret.datasets import get_data
kiva = get_data('kiva')
# Importing module and initializing setup
from pycaret.nlp import *
nlp1 = setup(data = kiva, target = 'en')
# creating a model
lda = create_model('lda')
# plotting a model
plot_model(lda)
# Importing dataset
from pycaret.datasets import get_data
france = get_data('france')
# Importing module and initializing setup
from pycaret.arules import *
arul1 = setup(data = france, transaction_id = 'Invoice', item_id = 'Description')
# creating a model
model = create_model(metric = 'confidence')
# plotting a model
plot_model(model)
# Importing dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')
# Importing module and initializing setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')
# creating a model
xgboost = create_model('xgboost')
# interpreting model
interpret_model(xgboost)
# Importing dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')
# Importing module and initializing setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')
# creating a model
xgboost = create_model('xgboost')
# interpreting model
interpret_model(xgboost, plot = 'correlation')
# Importing dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')
# Importing module and initializing setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')
# creating a model
xgboost = create_model('xgboost')
# interpreting model
interpret_model(xgboost, plot = 'reason', observation = 10)
# Importing dataset
from pycaret.datasets import get_data
jewellery = get_data('jewellery')
# Importing module and initializing setup
from pycaret.clustering import *
clu1 = setup(data = jewellery)
# create a model
kmeans = create_model('kmeans')
# Assign label
kmeans_results = assign_model(kmeans)
# Importing dataset
from pycaret.datasets import get_data
anomalies = get_data('anomalies')
# Importing module and initializing setup
from pycaret.anomaly import *
ano1 = setup(data = anomalies)
# create a model
iforest = create_model('iforest')
# Assign label
iforest_results = assign_model(iforest)
# Importing dataset
from pycaret.datasets import get_data
kiva = get_data('kiva')
# Importing module and initializing setup
from pycaret.nlp import *
nlp1 = setup(data = kiva, target = 'en')
# create a model
lda = create_model('lda')
# Assign label
lda_results = assign_model(lda)
# Importing dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')
# Importing module and initializing setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')
# create a model
dt = create_model('dt')
# calibrate a model
calibrated_dt = calibrate_model(dt)
# Importing dataset
from pycaret.datasets import get_data
credit = get_data('credit')
# Importing module and initializing setup
from pycaret.classification import *
clf1 = setup(data = credit, target = 'default')
# create a model
xgboost = create_model('xgboost')
# optimize threshold for trained model
optimize_threshold(xgboost, true_negative = 1500, false_negative = -5000)