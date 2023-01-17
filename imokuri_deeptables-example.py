import sys

sys.path.append('../input/deeptables')
import numpy as np

from deeptables.models import deeptable, deepnets

from deeptables.datasets import dsutils

from sklearn.model_selection import train_test_split
#loading data

df = dsutils.load_bank()

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
y = df_train.pop('y')

y_test = df_test.pop('y')
#training

config = deeptable.ModelConfig(nets=deepnets.DeepFM)

dt = deeptable.DeepTable(config=config)

model, history = dt.fit(df_train, y, epochs=10)
#evaluation

result = dt.evaluate(df_test,y_test, batch_size=512, verbose=0)

print(result)
#scoring

preds = dt.predict(df_test)