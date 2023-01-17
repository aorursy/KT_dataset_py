# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
DATA_PATH = '../input/israeli-polling-anomaly/'

df = pd.read_csv(DATA_PATH + 'votes_2019.csv', encoding='iso_8859_8')

votes_df = df[df.columns[7:]]
from sklearn.feature_extraction.text import TfidfTransformer

tf_idf_transformer = TfidfTransformer()

transformed_data = tf_idf_transformer.fit_transform(votes_df)

transformed_data = pd.DataFrame(transformed_data.toarray(), columns = votes_df.columns)
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05, behaviour='new')

model.fit(transformed_data)
predictions = pd.Series(data = model.decision_function(transformed_data), index = df.index, name = 'predictions')

print("total predicted anomalies:",predictions.sum())
df_with_anomaly_score = df.copy()

df_with_anomaly_score['anomaly_score'] = model.decision_function(transformed_data)

df_with_anomaly_score.sort_values('anomaly_score',ascending=True).head().T
df[df['שם ישוב'] == 'איתמר'].iloc[0]
sub = pd.read_csv(DATA_PATH + 'sample_sub.csv')

sub['poll'] = -predictions # Probabilistic predictions for AUC

sub.to_csv('submission.csv', index=False)