# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/final-dm-2020/train_data.csv")

test = pd.read_csv("/kaggle/input/final-dm-2020/test_data.csv")

samp = pd.read_csv("/kaggle/input/final-dm-2020/sample_submission.csv")
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import classification_report, make_scorer, accuracy_score

def cv_model(tx, ty, mdl):

    originalclass = []

    predictedclass = []

    

    def scorer(y_true, y_pred):

        originalclass.extend(y_true)

        predictedclass.extend(y_pred)

        return accuracy_score(y_true, y_pred)

    

    skf   = StratifiedKFold(n_splits=4, shuffle=True, random_state=99)

    score = cross_val_score(mdl, X=tx, y=ty, cv=skf, scoring=make_scorer(scorer))

    

    print('='*60)

    print('CV Scores:', score)

    print(classification_report(originalclass, predictedclass, digits=4))

    print('='*60)
train.describe()
train.isna().sum()
train.tc_time
import seaborn as sns

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,2, figsize=(12,5))



ax1= sns.scatterplot(x='tc_time', y='time_class', data= train)
missing_train = pd.DataFrame({'total_missing': train.isnull().sum(), 'perc_missing': (train.isnull().sum()/len(train.index))*100})

missing_train
missing_test = pd.DataFrame({'total_missing': test.isnull().sum(), 'perc_missing': (test.isnull().sum()/len(test.index))*100})

missing_test
train['frame_class'].value_counts()
train['tc_frame_class'].value_counts()
from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier, LGBMRegressor
def f(x):

    if not np.isnan(x['total_frame']) and not np.isnan(x['predicted_frame']):

        return x['total_frame'] - x['predicted_frame']

    return x['intracoded_frame']

def f2(x):

    if not np.isnan(x['intracoded_frame']) and not np.isnan(x['predicted_frame']):

        return x['intracoded_frame'] + x['predicted_frame']

    return x['total_frame']

def f3(x):

    if not np.isnan(x['intracoded_frame']) and not np.isnan(x['total_frame']):

        return x['total_frame'] - x['intracoded_frame']

    return x['predicted_frame']



def g(x):

    if not np.isnan(x['total_size']) and not np.isnan(x['predicted_size']):

        return x['total_size'] - x['predicted_size']

    return x['intracoded_size']

def g2(x):

    if not np.isnan(x['intracoded_size']) and not np.isnan(x['predicted_size']):

        return x['intracoded_size'] + x['predicted_size']

    return x['total_size']

def g3(x):

    if not np.isnan(x['intracoded_size']) and not np.isnan(x['total_size']):

        return x['total_size'] - x['intracoded_size']

    return x['predicted_size']



dictio_width = {}

dictio_width[640.0] = 480.0

dictio_width[320.0] = 240.0

dictio_width[480.0] = 360.0

dictio_width[1920.0] = 1080.0

dictio_width[1280.0] = 720.0

dictio_width[176.0] = 144.0

dictio_width[None] = None



dictio_height = {}

dictio_height[480.0] = 640.0

dictio_height[240.0] = 320.0

dictio_height[360.0] = 480.0

dictio_height[1080.0] = 1920.0

dictio_height[720.0] = 1280.0

dictio_height[144.0] = 176.0

dictio_height[None] = None
tt = train.copy()

ts = test.copy()



tt['intracoded_frame'] = tt.apply(lambda x: f(x), axis=1)

tt["total_frame"] = tt.apply(lambda x: f2(x), axis=1)

tt['predicted_frame'] = tt.apply(lambda x: f3(x), axis=1)



tt['intracoded_size'] = tt.apply(lambda x: g(x), axis=1)

tt['total_size'] =  tt.apply(lambda x: g2(x), axis=1)

tt['predicted_size'] =  tt.apply(lambda x: g3(x), axis=1)



tt['width'] = tt.apply(

    lambda row: dictio_height[row['height']] if np.isnan(row['width']) and not np.isnan(row['height']) else row['width'],

    axis=1

)



tt['tc_width'] = tt.apply(

    lambda row: dictio_height[row['tc_height']] if np.isnan(row['tc_width']) and not np.isnan(row['tc_height']) else row['tc_width'],

    axis=1

)



tt['height'] = tt.apply(

    lambda row: dictio_width[row['width']] if np.isnan(row['height']) and not np.isnan(row['width']) else row['height'],

    axis=1

)



tt['tc_height'] = tt.apply(

    lambda row: dictio_width[row['tc_width']] if np.isnan(row['tc_height']) and not np.isnan(row['tc_width']) else row['tc_height'],

    axis=1

)



ts['intracoded_frame'] = ts.apply(lambda x: f(x), axis=1)

ts["total_frame"] = ts.apply(lambda x: f2(x), axis=1)

ts['predicted_frame'] = ts.apply(lambda x: f3(x), axis=1)



ts['intracoded_size'] = ts.apply(lambda x: g(x), axis=1)

ts['total_size'] =  ts.apply(lambda x: g2(x), axis=1)

ts['predicted_size'] =  ts.apply(lambda x: g3(x), axis=1)



ts['width'] = ts.apply(

    lambda row: dictio_height[row['height']] if np.isnan(row['width']) and not np.isnan(row['height']) else row['width'],

    axis=1

)



ts['tc_width'] = ts.apply(

    lambda row: dictio_height[row['tc_height']] if np.isnan(row['tc_width']) and not np.isnan(row['tc_height']) else row['tc_width'],

    axis=1

)



ts['height'] = ts.apply(

    lambda row: dictio_width[row['width']] if np.isnan(row['height']) and not np.isnan(row['width']) else row['height'],

    axis=1

)



ts['tc_height'] = ts.apply(

    lambda row: dictio_width[row['tc_width']] if np.isnan(row['tc_height']) and not np.isnan(row['tc_width']) else row['tc_height'],

    axis=1

)

# ty = tf.pop("time_class")
# drop = ['video','height','tc_height']

drop = ['video']



te = tt.drop(columns=['video','tc_time'])

tr = ts.drop(columns=drop)



# te = te.dropna()

tf = te.copy()

tz = tr.copy()

# tf.drop(columns='tc_time', inplace=True)

ty = tf.pop("time_class")
# num_cols = ['duration','width', 'height','bitrate', 'intracoded_frame','predicted_frame','total_frame','intracoded_size', 'predicted_size','total_size','tc_width', 'tc_height','tc_bitrate','tc_time']

num_cols = ['duration','width', 'height','bitrate', 'intracoded_frame','predicted_frame','total_frame','intracoded_size', 'predicted_size','total_size','tc_width', 'tc_height','tc_bitrate']

num_test_cols = ['duration','width','height','bitrate', 'intracoded_frame','predicted_frame','total_frame','intracoded_size', 'predicted_size','total_size','tc_width', 'tc_height','tc_bitrate']
tz.isna().sum()
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

p1 = tz.pop('frame_class')

p2 = tz.pop('tc_frame_class')

p3 = tz.pop('codec')

imputed = IterativeImputer(random_state=1234, min_value=0).fit_transform(tz)



tz = pd.DataFrame(imputed, columns = num_test_cols)

tz['frame_class'] = p1

tz['tc_frame_class'] = p2

tz['codec'] = p3



p1 = tf.pop('frame_class')

p2 = tf.pop('tc_frame_class')

p3 = tf.pop('codec')

imputed = IterativeImputer(random_state=1234, min_value=0).fit_transform(tf)



tf = pd.DataFrame(imputed, columns = num_cols)

tf['frame_class'] = p1

tf['tc_frame_class'] = p2

tf['codec'] = p3
for num in num_cols:

    tf[num] = tf[num].astype('int')

    

for num in num_test_cols:

    tz[num] = tz[num].astype('int')
cat_cols = ['frame_class', 'tc_frame_class', 'codec', 'width','height', 'tc_width', 'tc_height']

# cat_cols = ['frame_class', 'tc_frame_class', 'codec', 'width','tc_width']
parameters = {

    "random_seed": 17,

#     "eval_metric": "F1", # use F1 as eval_metric when training using an eval_set

    "custom_metric": ["F1", "Precision", "Recall"],

    "loss_function": "MultiClass",

#     "task_type": "GPU", # enable GPU

    "random_strength": 3, # random value used at splits

    "od_type": "Iter", # set the overfitting detector type to use iterations

    "verbose":100, # print every 100 iterations

    "cat_features":cat_cols # categorical feature indexes

}

ctb = CatBoostClassifier(**parameters)
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import MultiLabelBinarizer
# tf.isna().sum()
tf = tf.fillna('0')

tz = tz.fillna('0')
ovr = OneVsRestClassifier(estimator=ctb)
# cv_model(tf,ty,ovr) 
for c in cat_cols:

    tf[c] = tf[c].astype("category")

    tz[c] = tz[c].astype("category")    
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(tf, ty, test_size=0.25, random_state=42)
ovr.fit(tf, ty)
pred = ovr.predict(tz)
# ctb.fit(X_train, y_train)
# pred = ctb.predict(X_test)

# print(classification_report(y_test, pred, digits=4))
ctb.fit(tf, ty)
from mlxtend.classifier import EnsembleVoteClassifier

ens = EnsembleVoteClassifier(clfs=[ovr, ctb], weights=[1,1], voting='soft')
ens.fit(tf,ty)

enspred = ens.predict(tz)
le = LabelEncoder()

pre = le.fit_transform(enspred)

samp['time_class'] = pre

samp.to_csv('sub_ensemble.csv', header=True, index=False)

samp
ctbpred = ctb.predict(tz)
len(pred)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

pre = le.fit_transform(pred)
pre
samp['time_class'] = pre
samp.to_csv('sub2.csv', header=True, index=False)
samp