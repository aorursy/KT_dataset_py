path = '../input/mse-pysupport-hackathon/simulation.csv'

import pandas as pd

df = pd.read_csv(path)

df.head()
# Use comments to make points

SAMPLE_COL = ['Bedrooms']

x = df[SAMPLE_COL].fillna(0)

y = df['SalePrice']



from sklearn.linear_model import LinearRegression 

model  = LinearRegression()

model.fit(x, y)



from sklearn.metrics import r2_score

r2_score(y, model.predict(x))
# URL for Holdout data - https://www.kaggle.com/pranjalrawat/msepysupportholdout

path = '../input/mse-pysupport-hackathon/holdout.csv'

import pandas as pd

holdout = pd.read_csv(path)

holdout.head()
# Scoring using trained model

SAMPLE_COL = ['Bedrooms']

x = holdout[SAMPLE_COL].fillna(0)

ypred = model.predict(x)
ypred
import numpy as np

submission = pd.DataFrame(np.c_[holdout.Id.astype(str), ypred], columns = ['Id', 'SalePrice'])

submission.to_csv('starter_submission.csv', index = False, header = True)