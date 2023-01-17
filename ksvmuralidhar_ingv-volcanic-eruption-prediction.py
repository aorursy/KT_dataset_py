import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from IPython.display import clear_output

import gc

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.feature_selection import RFE

from sklearn.pipeline import Pipeline
train_files = os.listdir("../input/predict-volcanic-eruptions-ingv-oe/train")

test_files = os.listdir("../input/predict-volcanic-eruptions-ingv-oe/test")
len(train_files)
len(test_files)
cols = []

rows = []

for i,fname in enumerate(train_files):

    train = pd.read_csv(os.path.join("../input/predict-volcanic-eruptions-ingv-oe/train",fname))

    cols.append(train.shape[0])

    rows.append(train.shape[1])

    print(f'{i+1} / {len(train_files)}')

    clear_output(wait=True)
print(f"Rows of all train files: {pd.Series(rows).unique()}\nColumns of all train files: {pd.Series(cols).unique()}")
cols = []

rows = []

for i,fname in enumerate(test_files):

    test = pd.read_csv(os.path.join("../input/predict-volcanic-eruptions-ingv-oe/test",fname))

    cols.append(test.shape[0])

    rows.append(test.shape[1])

    print(f'{i+1} / {len(test_files)}')

    clear_output(wait=True)
print(f"Rows of all test files: {pd.Series(rows).unique()}\nColumns of all test files: {pd.Series(cols).unique()}")
train = pd.read_csv(os.path.join("../input/predict-volcanic-eruptions-ingv-oe/train",train_files[2]))

train.head()
train.count()
(train==0).sum()
missing_tracker = pd.DataFrame()

for i,fname in enumerate(train_files):

    train = pd.read_csv(os.path.join("../input/predict-volcanic-eruptions-ingv-oe/train",fname))

    missing_tracker = missing_tracker.append(pd.DataFrame(train.count()).T)

    print(f'{i+1} / {len(train_files)}')

    clear_output(wait=True)
missing_tracker
missing_tracker.nunique()
for col in missing_tracker.columns:

    print(f"{col}\n{sorted(missing_tracker[col].unique())}\n\n")
plt.figure(figsize=(15,20))

sns.heatmap(missing_tracker);
mi = ((missing_tracker==60001).sum()/4431)*100

print(f'% of files with complete data per sensor\n\n{mi}')

mi.plot(kind="bar")

plt.axhline(y=100,color="red")

plt.title("Sensors with complete readings")

plt.ylabel("% of files with complete data");
mi = ((missing_tracker==0).sum()/4431)*100

print(f'% of files with no data at all per sensor\n\n{mi}')

mi.plot(kind="bar")

plt.title("Sensors with no readings at all")

plt.ylabel("% of files with no data at all");
train_1 = pd.read_csv(os.path.join("../input/predict-volcanic-eruptions-ingv-oe/train",train_files[0]))

train_1.head()
train_1.shape
fig,ax=plt.subplots(5,2,figsize=(12,17))

r=0

c=0

for i,col in enumerate(train_1.columns):

    ax[r,c].plot(train_1[col])

    ax[r,c].set_title(col)

    c+=1

    if (i+1)%2==0:

        r+=1

        c=0

plt.show();
fig,ax=plt.subplots(5,2,figsize=(12,20))

r=0

c=0

for i,col in enumerate(train_1.columns):

    #ax[r,c].hist(train_1[col])

    sns.distplot(train_1[col],ax=ax[r,c])

    ax[r,c].set_title(col)

    ax[r,c].set_xlabel("")

    c+=1

    if (i+1)%2==0:

        r+=1

        c=0

plt.show();
train = pd.DataFrame()

for n,i in enumerate(train_files):

    df = pd.read_csv(os.path.join("../input/predict-volcanic-eruptions-ingv-oe/train",i))

    #df.fillna(0,inplace=True)

    df = pd.DataFrame(df.describe().values.reshape(1,-1))

    df["segment_id"] = int(i.replace(".csv",""))

    train = train.append(df)

    print(f'{n}')

    gc.collect()

    clear_output(wait=True)
train
(train==0).sum()
(train.isnull()).sum()
((train.iloc[:,10:-1]).min()).min()
pd.read_csv(os.path.join("../input/predict-volcanic-eruptions-ingv-oe/train",train_files[1])).describe()
train_labels = pd.read_csv("../input/predict-volcanic-eruptions-ingv-oe/train.csv")
train = train.merge(train_labels,on="segment_id",how="left")
train=train.drop(columns="segment_id")
train.fillna(0,inplace=True)
X = train.iloc[:,:-1].copy()
y = train.iloc[:,-1].copy()
X.head()
y.isnull().sum()
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold,cross_val_score
#pca = PCA(n_components=25)

#X_pca = pca.fit_transform(X)

#np.cumsum(pca.explained_variance_ratio_)
del df

del train

gc.collect()
X_pca = X.copy()
rf = RandomForestRegressor(random_state=11)

rf.fit(X_pca,y)

print(f"No of Important Features: {sum(rf.feature_importances_>0)}")
MAX_DEPTH = 9

N_ESTIMATORS = 8000

MIN_SAMPLES_LEAF = 300

RFE_FEATURES = 70

L1 = 330000000





rfe = RFE(estimator=LGBMRegressor(random_state=11),n_features_to_select=RFE_FEATURES)

lgbm = LGBMRegressor(n_estimators=N_ESTIMATORS,

                     max_depth=MAX_DEPTH,

                     num_leaves=2**MAX_DEPTH,

                     min_data_in_leaf=MIN_SAMPLES_LEAF,

                     lambda_l1 = L1,

                     random_state=11,

                     n_jobs=-1)



gscv = Pipeline(steps=[('rfe',rfe),

                      ('regressor',lgbm)])
cv = KFold(n_splits=10,random_state=11,shuffle=True)

cv_score = cross_val_score(gscv,X_pca,y,cv=cv,scoring="neg_mean_absolute_error",n_jobs=-1)
-1*(cv_score.astype("int"))
np.mean(-1*(cv_score.astype("int")))
gscv.fit(X_pca,y)
del X

gc.collect()
mean_absolute_error(gscv.predict(X_pca),y)
r2_score(gscv.predict(X_pca),y)
from sklearn.model_selection import learning_curve

train_size,train_acc,test_acc = learning_curve(gscv, X_pca,y,scoring="neg_mean_absolute_error")

learn_df = pd.DataFrame({"Train_size":train_size,"Train_MAE":-1*train_acc.mean(axis=1),"Test_MAE":-1*test_acc.mean(axis=1)}).melt(id_vars="Train_size")

sns.lineplot(x="Train_size",y="value",data=learn_df,hue="variable")

plt.title("Learning Curve")

plt.ylabel("Loss");
del X_pca

gc.collect()
test = pd.DataFrame()

for n,i in enumerate(test_files):

    df = pd.read_csv(os.path.join("../input/predict-volcanic-eruptions-ingv-oe/test",i))

    #df.fillna(0,inplace=True)

    df = pd.DataFrame(df.describe().values.reshape(1,-1))

    df["segment_id"] = int(i.replace(".csv",""))

    test = test.append(df)

    print(f'{n}')

    gc.collect()

    clear_output(wait=True)
test_segment_ids = test["segment_id"]

test = test.drop(columns="segment_id")
test.fillna(0,inplace=True)
#X_test_pca = pca.transform(test)
X_test_pca = test.copy()
del test,cv,cv_score

gc.collect()
pred = gscv.predict(X_test_pca)
pred
submission = pd.DataFrame({"segment_id":test_segment_ids,"time_to_eruption":pred})
submission.to_csv("submission_6.csv",index=False)