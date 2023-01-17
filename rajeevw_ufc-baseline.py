import pandas as pd

import numpy as np



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
raw1 = pd.read_csv('../input/raw_total_fight_data.csv', sep=';')

raw1.head()
raw2 = pd.read_csv('../input/raw_fighter_details.csv')

raw2.head()
data = pd.read_csv('../input/data.csv')

data.head()
df = pd.read_csv('../input/preprocessed_data.csv')

df.head()
df_num = df.select_dtypes(include=[np.float, np.int])
scaler = StandardScaler()



df[list(df_num.columns)] = scaler.fit_transform(df[list(df_num.columns)])
y = df['Winner']

X = df.drop(columns = 'Winner')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=43)
model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=43)
model.fit(X_train, y_train)
model.oob_score_
y_preds = model.predict(X_test)

accuracy_score(y_test, y_preds)
model.feature_importances_