import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn import metrics
df_train = pd.read_csv('/kaggle/input/dda-p2/train.csv', index_col=0)

df_train.head()
df_train.shape
df_train['Fehlerhaft'].value_counts()
_ = df_train['Fehlerhaft'].plot.hist(bins=2, figsize=(12, 8), title='Verteilung fehlerfreie vs. fehlerhafte Produktionsstücke')
df_train.isnull().sum()
# für schnellere Laufzeit und mehr Übersicht in den Plots: Stichprobe der Daten abbilden

data_sample = df_train.sample(2000, random_state=28)  # random_state sorgt für reproduzierbare Stichprobe, sodass die Stichprobe für uns alle identisch ist



_ = pd.plotting.scatter_matrix(data_sample, c=data_sample['Fehlerhaft'], cmap='seismic', figsize=(16, 20))
# Splitten von Features (X) und Zielgröße (y)

X = df_train.drop('Fehlerhaft', axis=1)

y = df_train['Fehlerhaft']
X_train, X_validierung, y_train, y_validierung = train_test_split(X, y, test_size=0.2)  # nutze 20% der Trainingsdaten als Validierungsset
prediction_pipe = Pipeline([

    ('scaler', StandardScaler()),

    ('clf', LogisticRegression(solver='lbfgs')),

])
_ = prediction_pipe.fit(X_train, y_train)

print('F1-Score auf den Trainingsdaten:', metrics.f1_score(y_train, prediction_pipe.predict(X_train)).round(2))

print('F1-Score auf den Validierungsdaten:', metrics.f1_score(y_validierung, prediction_pipe.predict(X_validierung)).round(2))
X_test = pd.read_csv('/kaggle/input/dda-p2/test.csv', index_col=0)

X_test.head()
predicted_test = prediction_pipe.predict(X_test)
submission = pd.read_csv('/kaggle/input/dda-p2/sample_submission.csv')

submission['Fehlerhaft'] = predicted_test

submission.head()
submission.to_csv('./predicted_values.csv', index=False)