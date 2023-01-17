# Final 7th solution of Janata Hack NLP in 10 rows with score 0.936 on private LB. First place solution got 0.944 (F1 score metrics).
# Full solution and data cleaning code provided on github by my teammate https://github.com/romanlents/NLP-Janata-/blob/master/README.md 
import numpy as np 
import pandas as pd
import os
        
from sklearn.model_selection import train_test_split
import tensorflow as tf
print("Tensorflow version " + tf.__version__)

# Basic text cleaning technics used, cleaned data uploaded. 

train = pd.read_csv('/kaggle/input/cleanedjanata23/train_cleaned_v2.csv.csv')
test = pd.read_csv('/kaggle/input/cleanedjanata23/test_cleaned_v2.csv.csv')
train.head(3)
# install ktrain library

# !pip install ktrain
# !pip install pandas==0.25.1
import ktrain
from ktrain import text
# split to train and validation with 20%
x_train, x_valid, y_train, y_valid = train_test_split(
                        train['user_review'].values, 
                        train['user_suggestion'].values,
                        test_size = 0.2)
# Model parameters
MAX_LEN = 220 # 
MODEL_NAME = 'roberta-large'
BATCH_SIZE = 8
%%time
  
t = text.Transformer(MODEL_NAME, maxlen = MAX_LEN, class_names = [0, 1])
trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_valid, y_valid)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data = trn, val_data = val, batch_size = BATCH_SIZE)

# 2 Epochs
# Validation accuracy 0.933 on 2nd Epoch
# Private LB score 0.936
learner.fit_onecycle(2e-5, 2)
# Predict values on test
predictor = ktrain.get_predictor(learner.model, preproc=t)
preds = predictor.predict(test.user_review.values)
# Sumbit solution
submission=pd.DataFrame.from_dict(
    {"review_id": test_clean_2['review_id'],
    "user_suggestion": preds})

submission['user_suggestion'] = preds
submission.to_csv('submission_final.csv', index=False)