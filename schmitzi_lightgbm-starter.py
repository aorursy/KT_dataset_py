import json

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from datetime import datetime



from sklearn import preprocessing

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_predict



import lightgbm as lgb
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train: ", train_df.shape)

print("Test: ", test_df.shape)



train_df.head()
maxtime = 200

train_df = train_df.loc[train_df['y']<maxtime]



plt.figure(figsize=(8,6))

plt.title('y distibution')

plt.hist(train_df['y'], bins=100)

plt.xlabel('y value [s]', fontsize=12)

plt.yscale('log')

plt.show()
df_nan = train_df.isnull().sum(axis=0)

print("Null/NaN features:" + str(df_nan.sum()))
for col in train_df.columns:

    if train_df[col].dtype == 'object':

        lenc = preprocessing.LabelEncoder()

        lenc.fit(list(train_df[col]) + list(test_df[col]))

        train_df[col] = lenc.transform(list(train_df[col]))

        test_df[col] = lenc.transform(list(test_df[col]))
X = train_df.drop(['ID','y'], axis=1)

y = train_df['y']



X_test = test_df.drop(['ID'], axis=1)

ID_test = test_df['ID']



print("-> Train: ", X.shape)

print("-> Test: ", X_test.shape)
#with open("parameter_LGB_0.5166900124_2017-06-13-11-46.json") as fp:

#    loaded_pars=json.load(fp)

#fp.close()



model = lgb.LGBMRegressor()

#model.set_params(**loaded_pars)

print("Training...")

model.fit(X,y, init_score=np.mean(y))
y_pred = cross_val_predict(model, X=X, y=y, cv=10, n_jobs=5)

y_diff = np.clip(100 * ( (y_pred - y) / y ), -50, 50)



R2 = r2_score(y, y_pred)



print("CV R2-Score: " + str(R2))



plt.figure(figsize=(8,6))

plt.title('True vs Predicted Y')

plt.scatter(y, y_pred, c=y_diff, cmap=plt.cm.seismic)

plt.colorbar()

plt.xlabel('True y')

plt.ylabel('Predicted y')

plt.show()



plt.figure(figsize=(8,6))

plt.hist(y_pred, 50)

plt.xlabel('Predicted y')

plt.show()
print("Preparing submission and parameters file...")

subm = pd.DataFrame()

subm['ID'] = ID_test

subm['y'] = model.predict(X_test)



sub_file = 'submission_LGB_' + str(R2) + '_' + str(datetime.now().strftime('%Y-%m-%d-%H-%M')) + '.csv'

lgb_file = sub_file.replace('submission', 'parameter')

lgb_file = lgb_file.replace('csv', 'json')



#with open(lgb_file, mode="w") as fp:

#    json.dump(model.get_params(), fp)

#fp.close()



subm.to_csv(sub_file, index=False)

print("done.")