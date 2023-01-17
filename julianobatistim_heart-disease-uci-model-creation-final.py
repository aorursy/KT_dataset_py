import pandas as pd
import numpy as np

global_seed = 1236

raw = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
raw.dropna()
raw.head()
for column in raw.columns:
    print(f"{column}: {raw[column].unique()}")
from sklearn.model_selection import train_test_split
train, validation = train_test_split(raw, test_size=0.05, random_state=global_seed)

print('raw data shape: ', raw.shape)
print('train data shape: ', train.shape)
print('test data shape: ', validation.shape)
!pip install pycaret
from pycaret.classification import *

exp = setup(
    train, 
    target = 'target', 
    categorical_features=None,
    numeric_features=None,
    date_features=None, 
    ignore_features=None, 
    normalize=False, 
    normalize_method='zscore', 
    transformation=False, 
    transformation_method='yeo-johnson',
    n_jobs=2, 
    use_gpu=True, 
    session_id=global_seed, 
    log_experiment=False, 
    experiment_name=None, 
    log_plots=False, 
    log_profile=False, 
    log_data=False, 
    silent=True, 
    verbose=True, 
    profile=False
)
compare_models()
best = create_model('ridge')
plot_model(best, plot = 'confusion_matrix')
plot_model(best, plot = 'feature')
test_prediction = predict_model(best, validation)
test_prediction.to_csv('validation_prediction.csv', index=False)
test_prediction
test_prediction = test_prediction.apply(pd.to_numeric)
test_prediction['comp'] = np.where(test_prediction['target'] == test_prediction['Label'], 'Correct', 'Incorrect')
test_prediction.groupby('comp').count()['Label']
validation_accuracy = test_prediction.groupby('comp').count()['Label'][0] / (test_prediction.groupby('comp').count()['Label'][0] + test_prediction.groupby('comp').count()['Label'][1])
print('validation_accuracy: ', validation_accuracy)
from sklearn.metrics import confusion_matrix

y_actu = test_prediction['target']
y_pred = test_prediction['Label']

cm = confusion_matrix(y_actu, y_pred)

import seaborn as sn
sn.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})
print('validation_accuracy: ', validation_accuracy)