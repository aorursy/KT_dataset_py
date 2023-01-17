!pip install pycaret==2.0
# Demonstration of how to use pycaret for classification
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pycaret.utils import version
version()
from pycaret.datasets import get_data
dataset = get_data('iris')
#check the shape of data
dataset.shape
data = dataset.sample(frac=0.9, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
from pycaret.classification import *
iris = setup(data = data, target = 'species', session_id=1)
compare_models()
nbmodel = create_model('nb')
print(nbmodel)
qdamodel=create_model('qda')
print(qdamodel)
xgbmodel=create_model('xgboost')
print(xgbmodel)
tuned_nb=tune_model(nbmodel)
print(tuned_nb)
tuned_qda=tune_model(qdamodel)
print(tuned_qda)
tuned_xgb=tune_model(xgbmodel)
print(tuned_xgb)
plot_model(tuned_nb, plot = 'confusion_matrix')
plot_model(tuned_qda, plot = 'confusion_matrix')
plot_model(tuned_xgb, plot = 'confusion_matrix')
plot_model(tuned_nb, plot = 'class_report')
plot_model(tuned_qda, plot = 'class_report')
plot_model(tuned_xgb, plot = 'class_report')
plot_model(tuned_nb, plot='boundary')
plot_model(tuned_qda, plot='boundary')
plot_model(tuned_xgb, plot='boundary')
predict_model(tuned_qda)
save_model(tuned_qda,'Final QDA Model')
saved_final_qda = load_model('Final QDA Model')
new_prediction = predict_model(saved_final_qda, data=data_unseen)
new_prediction.head()