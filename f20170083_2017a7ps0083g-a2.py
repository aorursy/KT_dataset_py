import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
train_csv = "../input/data-mining-assignment-2/train.csv"

test_csv = "../input/data-mining-assignment-2/test.csv"
train_df = pd.read_csv(train_csv)

train_df.head()
test_df = pd.read_csv(test_csv)

test_df.head()
# col = 'col59'

# print(train_df[col].describe())

# print(test_df[col].describe())
shifted_cols =  ['col6', 'col19', 'col30', 'col49', 'col59']
labels = train_df.Class

all_features = pd.concat([train_df.drop('Class', axis=1), test_df]).reset_index()
labels.head()
all_features = all_features.drop(shifted_cols, axis=1)
all_features.head()
all_features.select_dtypes(include=['object']).columns
all_features.drop(['index', 'ID'], axis=1, inplace=True)
all_features = pd.get_dummies(data=all_features)

all_features.head()
# all_features = (all_features - all_features.mean())/all_features.std()

# all_features.head()

# scaler = StandardScaler()

# all_features = scaler.fit_transform(all_features)
# all_features[0]
training_features = all_features[:700]

test_features = all_features[700:1000]
# test_features[0]
X_train, X_val, y_train, y_val = train_test_split(training_features, labels, test_size=0.2)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
model = RandomForestClassifier(n_jobs=-1, verbose=1, n_estimators=500)
model.fit(X_train, y_train)
model.score(X_val, y_val)
results = model.predict(test_features)
result_df = pd.DataFrame()

result_df['ID'] = test_df.ID

result_df['Class'] = results
result_df.head()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(result_df)