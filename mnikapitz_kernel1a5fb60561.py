

import os

for dirname, _, filenames in os.walk('/kaggle/data'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import numpy as np
df = pd.read_csv("/kaggle/input/data.csv", index_col='COUNTRY')
df.head()
df.info()
df['Deaths per Population*1000000']=df['Deaths per Population*1000000'].apply(lambda x: x.replace(",","."))
df['Deaths per Population*1000000']=df['Deaths per Population*1000000'].astype(float)
df.fillna(value=1000000, inplace=True)
from sklearn.model_selection import train_test_split
X=df.drop('Deaths per Population*1000000', axis=1)

y=df['Deaths per Population*1000000']

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
rf_1=RandomForestRegressor(n_estimators=10, random_state=40)

rf_2=RandomForestRegressor(n_estimators=30, random_state=40)

rf_3=RandomForestRegressor(n_estimators=50, random_state=40)

rf_4=RandomForestRegressor(n_estimators=100, random_state=40)

rf_5=RandomForestRegressor(n_estimators=200, random_state=40)
def score_model(model):

    model.fit(X_train, y_train)

    preds=model.predict(X_test)

    return mean_absolute_error(y_test, preds)
models=[rf_1, rf_2, rf_3, rf_4, rf_5]
for i in range(0, len(models)):

    mae=score_model(models[i])

    print("Random Forest Model %d MAE: %d"%(i+1, mae))
from sklearn.inspection import permutation_importance
model=RandomForestRegressor(n_estimators=200).fit(X,y)

result=permutation_importance(model, X, y, n_repeats=10,random_state=0)
importances = model.feature_importances_
features=X.columns

indices = np.argsort(importances)

plt.figure(figsize=(14,14))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
from treeinterpreter import treeinterpreter as ti
hun=df.loc['Hungary'][:-1]
df.loc['Hungary'][-1:]
hun=pd.DataFrame(hun)
hun=hun.transpose()
X_hun=X.drop('Hungary')

y_hun=y.drop('Hungary')
model_hun=RandomForestRegressor(n_estimators=200).fit(X_hun,y_hun)
model_hun.predict(hun)
prediction, bias, contributions = ti.predict(model_hun, hun)
result=pd.DataFrame(data=contributions.transpose(), index=X.columns, columns=['Contributions']).head()