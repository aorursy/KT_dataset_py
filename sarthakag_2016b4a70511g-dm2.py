import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV
dftrain = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',', index_col='ID')

data = dftrain
dftest = pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',', index_col='ID')

data_test = dftest
obj_col = data.columns[data.dtypes == 'object']

obj_col
df_onehot = pd.get_dummies(data, columns=obj_col)

df_onehot_test = pd.get_dummies(data_test, columns=obj_col)

df_onehot = df_onehot.drop('Class', axis=1)
df1 = df_onehot.copy()

scaler = StandardScaler()

scaled_data = scaler.fit(df1).transform(df1)

scaled_df=pd.DataFrame(scaled_data,columns=df1.columns, index=df1.index)

# scaled_df.head()
df2 = df_onehot_test.copy()

scaler = StandardScaler()

scaled_data_test = scaler.fit(df2).transform(df2)

scaled_df_test=pd.DataFrame(scaled_data_test,columns=df2.columns, index=df2.index)

# scaled_df_test.head()
scaled_df.shape
scaled_df_test.shape
X_train = scaled_df

y_train = dftrain['Class']
rf = RandomForestClassifier(n_estimators=100, max_depth = 6,min_samples_split=2)

rf.fit(X_train, y_train)
y_pred_RF = rf.predict(scaled_df_test)
answer=pd.read_csv('../input/data-mining-assignment-2/Sample Submission.csv')
answer['Class']=list(y_pred_RF)
answer['Class'].value_counts()
answer.to_csv('predict.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "DataRF12.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(answer)