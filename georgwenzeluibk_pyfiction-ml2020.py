import numpy as np
import pandas as pd
import sklearn 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import StandardScaler
train_data_pd = pd.read_csv("train_set.csv", header=0)
test_data_pd = pd.read_csv("test_set.csv", header=0)
train_data_np = train_data_pd.to_numpy()
test_data_np = test_data_pd.to_numpy()
train_data_x = train_data_np[:,2:]
train_data_y = train_data_np[:,1]
test_data_x = test_data_np[:,1:]
#returns the data where the categorical features are removed and the n best hot_encoded_cat_features are appended
def n_hottest_hot(X, y ,n, sel_k_best = None):
    cat_indices = [(X.columns[i].endswith("cat") or X.columns[i].endswith("cat")) for i in np.arange(X.shape[1])]
    cat_data = X.loc[:,cat_indices]
    #take only the one with less than 10 categories - otherwise it takes to much space...
    onehot = sklearn.preprocessing.OneHotEncoder(sparse = False)
    onehot.fit(cat_data)
    hot_cats = onehot.transform(cat_data)
    if sel_k_best is not None:
        k_best = hot_cats[:,sel_k_best.get_support()]
    else:
        k_Best = SelectKBest()
        sel_k_best = SelectKBest(k=n).fit(hot_cats , y)
        k_best = hot_cats[:,sel_k_best.get_support()]
    k_best = pd.DataFrame(k_best)
    not_cat_indices = [ not(X.columns[i].endswith("cat") or X.columns[i].endswith("cat")) for i in np.arange(X.shape[1])]
    result = X.loc[:,not_cat_indices]
    result = pd.concat([result, k_best], axis = 1)
    return result, sel_k_best
def n_best(data_x, data_y, n):
  select_k_best = SelectKBest(f_classif, k=n)
  select_k_best.fit_transform(data_x, data_y)
  return select_k_best.get_support()
#Returns n_best_result-features with the categorical features one-hot-encoded
#data_x is a numpy array of features from data_pandas_x (i.e. train_data_x for train_data_pd)
#data_pandas_x is the original pandas dataset (i.e. train_data_pd)
#n_best_result is the result of calling the n_best function on the numpy array and corresponding y
#offset is the first column of the pandas dataset where features start (2 for training, 1 for testing)
def n_best_onehot(data_x, data_pandas_x, n_best_result, offset=2):
  cols = [i for i,v in enumerate(n_best_result) if v == True]

  manuals = []
  removefromcols = []

  for i in cols:
    if(data_pandas_x.columns[i+offset].endswith("cat")):
      manuals.append(pd.get_dummies(data_pandas_x[data_pandas_x.columns[i+offset]]))
      removefromcols.append(i)

  for i in removefromcols:
    cols.remove(i)

  data_x_new = data_x[:,cols]

  for m in manuals:
    m_np = m.to_numpy()
    data_x_new = np.append(data_x_new, m_np, axis=1)

  return data_x_new

with pd.option_context('display.max_columns', 60):
    #print(df.describe(include='all'))
    print(train_data_pd.describe(include='all'))
def violin(X,y):       
  data = pd.concat([y,X],axis=1)
  data = pd.melt(data,id_vars="target",
                      var_name="features",
                      value_name='value')
  plt.figure(figsize=(20,10))
  sns.violinplot(x="features", y="value", hue="target", data=data,split=True, inner="quart")
  plt.xticks(rotation=90)

cat_indices = [(train_data_pd.columns[i].endswith("cat") or train_data_pd.columns[i].endswith("cat")) for i in np.arange(train_data_pd.shape[1])]
cat_data = train_data_pd.loc[:,cat_indices]

violin(cat_data.loc[:,(cat_data.std() < 1)] ,train_data_pd["target"] )
violin(cat_data.loc[:,(cat_data.std() >= 1)] ,train_data_pd["target"] )
bin_indices = [(train_data_pd.columns[i].endswith("bin") or train_data_pd.columns[i].endswith("bin")) for i in np.arange(train_data_pd.shape[1])]
bin_data = train_data_pd.loc[:,bin_indices]

violin(bin_data, train_data_pd["target"])
rest_indices = [not(bin_indices[i] or cat_indices[i]) for i in np.arange(train_data_pd.shape[1])]
rest_data = train_data_pd.loc[:,rest_indices]
rest_data = rest_data.drop(["id", "target"], axis = 1)

violin( rest_data.loc[:,(rest_data.std() > 1)] ,train_data_pd["target"] )
violin( rest_data.loc[:,(rest_data.std() <= 1)] ,train_data_pd["target"] )
correlation = train_data_pd.corr()

plt.figure(dpi=1200)
plt.figure(figsize=(40,20))
sns.heatmap(correlation, annot=True)
#plt.savefig("correlation.svg")
plt.show()
#get x and y of train set in pandas form
pandas_x = train_data_pd.drop(["target", "id"], axis=1)
pandas_y = train_data_pd["target"]

#get dataset with one-hot-encoding
onehot_pandas_x, selected_cols = n_hottest_hot(pandas_x, pandas_y, 50)
print(onehot_pandas_x.describe)

#convert to numpy array
onehot_np_x = onehot_pandas_x.to_numpy()

print(onehot_np_x.shape)
#standardize data
scaler = StandardScaler().fit(onehot_np_x)
onehot_scaled = scaler.transform(onehot_np_x)
best_feats = n_best(onehot_scaled, train_data_y, 25)
onehot_scaled = onehot_scaled[:,best_feats]
# Create instance of EarlyStopping to stop when loss is no longer decreased for 5 epochs
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
model = Sequential()
model.add(Dense(10, input_dim=25, activation='relu'))
model.add(Dense(1, input_dim=10, activation='sigmoid'))
#compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], weighted_metrics=["accuracy"])
#fit model on large epoch sizes to quickly converge (500 epochs or until no more loss reduction for 20 epochs)
model.fit(onehot_scaled, train_data_y, epochs=500, batch_size=256, class_weight={0:1,1:14.25}, callbacks=[es])
# predict test set
pd_test_x = test_data_pd.drop(["id"], axis=1)
#get dataset with one-hot-encoding
onehot_test, selected_cols = n_hottest_hot(pd_test_x, pandas_y, 50, sel_k_best=selected_cols)

#transform as with the train set
onehot_test = scaler.transform(onehot_test)
onehot_test = onehot_test[:,best_feats]

#predict, round the prediction to 0 or 1
preds = model.predict(onehot_test)
preds = np.rint(preds)
preds = preds.astype(int)

# how many are 0 and 1 class
amount_0 = sum(preds == 0) / len(preds)
amount_1 = sum(preds == 1) / len(preds)

print("Ratio of 0 classified data:", amount_0)
print("Ratio of 1 classified data:", amount_1)
data_out = pd.DataFrame(test_data_pd['id'].copy())
data_out.insert(1, "target", preds, True)
data_out.to_csv('submission.csv',index=False)
#get standard scaler
scaler = StandardScaler()

#get n best columns (here, 14) from training set
n_best_res = n_best(train_data_x, train_data_y, 14)

#fit the training set to the one hot encoding of these columns
train_x = n_best_onehot(train_data_x, train_data_pd, n_best_res, 2)

#fit scaler and scale
scaler.fit(train_x, train_data_y)
train_x = scaler.transform(train_x)

#train model
clf = LogisticRegression(C=1.0, class_weight={0:1, 1:14}, dual=False,
                      fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                      max_iter=10000, multi_class='auto', n_jobs=None,
                      penalty='l1', random_state=None, solver='liblinear',
                      tol=0.0001, verbose=0, warm_start=False)
clf.fit(train_x, train_data_y)

#take the same columns and encoding for the test set and scale the same way
x_test = n_best_onehot(test_data_x, test_data_pd, n_best_res, 1)
x_test = scaler.transform(x_test)

# predict test set
preds = clf.predict(x_test)
preds = np.rint(preds)
preds = preds.astype(int)

# how many are 0 and 1 class
amount_0 = sum(preds == 0) / len(preds)
amount_1 = sum(preds == 1) / len(preds)

print("Ratio of 0 classified data:", amount_0)
print("Ratio of 1 classified data:", amount_1)

#create file
data_out = pd.DataFrame(test_data_pd['id'].copy())
data_out.insert(1, "target", preds, True)
data_out.to_csv('submission.csv',index=False)