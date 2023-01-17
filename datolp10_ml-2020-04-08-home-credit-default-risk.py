import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



import os

os.listdir('../input/home-credit-default-risk/')
app_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

app_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')

sample_submission = pd.read_csv('../input/home-credit-default-risk/sample_submission.csv')



SK_ID_CURR = app_test.iloc[:, 0]



app_train.head()
print('0:', app_train.loc[app_train.TARGET == 0].shape[0])

print('1:', app_train.loc[app_train.TARGET == 1].shape[0])



plt.bar(['0', '1'], 

        [app_train.loc[app_train.TARGET == 0].shape[0], app_train.loc[app_train.TARGET == 1].shape[0]], 

        width=0.5)
objct_cols = app_train.select_dtypes(include=object)

objct_cols_list = objct_cols.columns



app_train[objct_cols_list] = app_train[objct_cols_list].fillna('miss') 

app_test[objct_cols_list] = app_test[objct_cols_list].fillna('miss') 



for col in objct_cols_list:

    label_mean = app_train.groupby(col).TARGET.mean()

    app_train[col] = app_train[col].map(label_mean).copy()

    app_test[col] = app_test[col].map(label_mean).copy()





app_train = app_train.dropna()

app_test = app_test.fillna(app_train.median())



target = app_train.pop('TARGET')
X_train, X_test, y_train, y_test = train_test_split(app_train.to_numpy(), target.values, test_size=0.25, random_state=42)
model_1 = RandomForestClassifier()

model_1.fit(X_train, np.ravel(y_train))



print('Train Accuracy:', model_1.score(X_train, y_train))

print('Test Accuracy:', model_1.score(X_test, y_test))
y_pred_proba = model_1.predict_proba(app_test)

y_pred_proba.shape



Submission = pd.DataFrame({ 'SK_ID_CURR': SK_ID_CURR,'TARGET': y_pred_proba[:,1] })

Submission.to_csv("Submission.csv", index=False)



Submission