from sklearn.metrics import fbeta_score

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
df_classes_trainable = pd.read_csv('../input/classes-trainable.csv', index_col=0)
df_classes_trainable.shape
df_tuning_labels = pd.read_csv('../input/tuning_labels.csv', index_col=0, header=None, names=['image_id', 'labels'])
df_tuning_labels.shape
y_true = df_tuning_labels['labels'].apply(lambda lbls: df_classes_trainable.index.isin(lbls.split()))
y_true = pd.DataFrame(y_true.values.tolist(), index=y_true.index)
y_true.shape
y_true.head()
fbeta_score(y_true, y_true, beta=2, average='samples')
y_predict = y_true.copy()
y_predict.shape
y_predict.loc['2b2f44594449326f4e52553d', 0] = True
fbeta_score(y_true, y_predict, beta=2, average='samples')