import numpy as np

import pandas as pd

import matplotlib as plt

from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline

from category_encoders import CountEncoder

from sklearn.model_selection import KFold

from sklearn.decomposition import PCA



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
fp = "../input/lish-moa/"

trainFeatures = pd.read_csv(fp + "train_features.csv")

trainFeatures.head()
trinTargetsNonsocred = pd.read_csv(fp + "train_targets_nonscored.csv")

trinTargetsNonsocred.head()
trainTargetsScored = pd.read_csv(fp + "train_targets_scored.csv")

trainTargetsScored.head()
testFeatures = pd.read_csv(fp + "test_features.csv")

testFeatures.head()
sampleSubmission = pd.read_csv(fp + "sample_submission.csv")

sampleSubmission.head()
trainFeatures.shape, trainTargetsScored.shape, testFeatures.shape
trainFeatures.groupby("cp_type").count()["sig_id"]
trainTargetsScored.groupby("5-alpha_reductase_inhibitor").count()["sig_id"]
indexNames = trainFeatures[trainFeatures["cp_type"] == "ctl_vehicle"].index

trainFeatures.drop(indexNames , inplace=True)

trainTargetsScored.drop(indexNames, inplace=True)
drop_col = ["sig_id", "cp_type", "cp_time", "cp_dose"]

X = trainFeatures.drop(drop_col, axis = 1).values

X.shape
Y = trainTargetsScored.drop("sig_id", axis = 1).values

Y.shape
#pca

pca = PCA(n_components=20)

pca.fit(X)

W = pca.transform(X)

print(X.shape)

print(W.shape)

print(pca.explained_variance_ratio_)
#X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size = 0.8, test_size = 0.2)

X_train = W

Y_train = Y

#model = LinearRegression()

#model.fit(X_train, Y_train)



#Y_pred = model.predict(X_val)

#Y_train_pred = model.predict(X_train)

#print('MSE train data: ', mean_squared_error(Y_train, Y_train_pred))

#print('MSE test data: ', mean_squared_error(Y_val, Y_pred))   



#scores = cross_val_score(model, X_train, Y_train)

#print(scores)



#model.fit(X, y)
t = testFeatures.drop(drop_col, axis = 1).values

tt = pca.transform(t)

X_test = tt
mof = MultiOutputClassifier(XGBClassifier(tree_method="gpu_hist"))



clf = Pipeline([('encode', CountEncoder(cols=[0, 2])), ('classify', mof)])

params = {'classify__estimator__colsample_bytree': 0.6522,

          'classify__estimator__gamma': 3.6975,

          'classify__estimator__learning_rate': 0.0503,

          'classify__estimator__max_delta_step': 2.0706,

          "classify__estimator__max_depth": 10,

          'classify__estimator__min_child_weight': 31.5800,

          'classify__estimator__n_estimators': 166,

          'classify__estimator__subsample': 0.8639

         }



clf.set_params(**params)



kf = KFold(n_splits=5)

temp = np.zeros((X_test.shape[0], Y_train.shape[1]))



for fn, (kf_indx, kf_indx_val) in enumerate(kf.split(X_train, Y_train)):

    print("Start:", fn)

    

    kf_x, kf_val_x = X_train[kf_indx], X_train[kf_indx_val]

    kf_y, kf_val_y = Y_train[kf_indx], Y_train[kf_indx_val]

  

    clf.fit(kf_x, kf_y)

    val_pred = clf.predict_proba(kf_val_x)

    val_pred = np.array(val_pred)[:, :, 1].T

    print('MSE test data: ', mean_squared_error(kf_val_y, val_pred)) 

    

    test_pred = clf.predict_proba(X_test)

    test_pred = np.array(test_pred)[:,:,1].T

    temp += test_pred

    

Y_pred = temp / 5

sampleSubmission[sampleSubmission.columns.to_list()[1:]] = Y_pred
sampleSubmission.head()
sampleSubmission.to_csv("/kaggle/working/submission.csv", index=False)