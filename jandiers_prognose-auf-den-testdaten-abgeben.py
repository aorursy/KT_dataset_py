# Import der Bibliotheken

import numpy as np

import pandas as pd
df = pd.read_csv('/kaggle/input/data-driven-business-analytics/train.csv', index_col=0)  # einlesen der Daten

df.head()
mean_price = df['Preis'].mean()  # in diesem Fall ist die Prognose einfach nur der Durchschnittswert aller Preise im Datensatz

mean_price
df_test = pd.read_csv('/kaggle/input/data-driven-business-analytics/test.csv', index_col=0)

df_test.head()
predictions = [mean_price] * 322  # hier steht dann bei euch: predictions = model.predict(X_test) bzw. predictions = pipeline.predict(X_test)

predictions[:5]
df_test['Preis'] = predictions

df_test.head()
df_submission = df_test['Preis'].reset_index()

df_submission.head()
df_submission.to_csv('./submission.csv', index=False)