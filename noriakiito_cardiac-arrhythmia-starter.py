import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm.notebook import tqdm
from scipy.io import loadmat
from scipy import stats
from glob import glob

sns.set()
sns.set_context('poster')
%matplotlib inline
# Setting Random Seed

import random
import tensorflow as tf
import numpy as np
import os

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything(seed=42)
idx_ = []  # index
len_ = []  # length
mean_ = []  # mean
median_ = [] # median
mode_ = [] # mpde
std_ = []  # standard deviation
ste_ = []  # standard error
max_ = []  # maximum value
min_ = []  # minimum value
y_ = []
for d in ['normal', 'af']:
    for path in sorted(glob('/kaggle/input/1056lab-cardiac-arrhythmia-detection/' + d +'/*.mat')):
        filename = path.split('/')[-1]  # e.g. B05821.mat
        i = filename.split('.')[0]  # e.g. B05821
        idx_.append(i)
        mat_contents = loadmat(path)
        x = mat_contents['val'][0]
        len_.append(len(x))
        mean_.append(x.mean())
        median_.append(np.median(x))
        mode_.append(stats.mode(x)[0][0])
        std_.append(x.std())
        ste_.append(x.std()/np.sqrt(len(x)))
        max_.append(x.max())
        min_.append(x.min())
        if d == 'normal':
            y_.append(0)
        else:
            y_.append(1)
train_df = pd.DataFrame(index=idx_, columns=['length', 'mean', 'median', 'mode', 'standard deviation', 
                                             'standard error', 'maximum value', 'minimum value','x1',
                                             'x2', 'x3', 'x4', 'y'])
train_df['length'] = len_
train_df['mean'] = mean_
train_df['median'] = median_
train_df['mode'] = mode_
train_df['standard deviation'] = std_
train_df['standard error'] = ste_
train_df['maximum value'] = max_
train_df['minimum value'] = min_
train_df['x1'] = (np.array(mean_) + np.array(median_)) / 2.0
train_df['x2'] = np.array(mean_) + np.array(mode_) / 2.0
train_df['x3'] = np.array(mean_) + np.array(std_) / 2.0
train_df['x4'] = np.array(mean_) + np.array(ste_) / 2.0
train_df['y'] = y_
train_df
idx_ = []  # index
len_ = []  # length
mean_ = []  # mean
median_ = [] # median
mode_ = [] # mpde
std_ = []  # standard deviation
ste_ = []  # standard error
max_ = []  # maximum value
min_ = []  # minimum value
for path in sorted(glob('/kaggle/input/1056lab-cardiac-arrhythmia-detection/test/*.mat')):
    filename = path.split('/')[-1]  # e.g. B05821.mat
    i = filename.split('.')[0]  # e.g. B05821
    idx_.append(i)
    mat_contents = loadmat(path)
    x = mat_contents['val'][0]
    len_.append(len(x))
    mean_.append(x.mean())
    median_.append(np.median(x))
    mode_.append(stats.mode(x)[0][0])
    std_.append(x.std())
    ste_.append(x.std()/np.sqrt(len(x)))
    max_.append(x.max())
    min_.append(x.min())
test_df = pd.DataFrame(index=idx_, columns=['length', 'mean', 'median', 'mode', 'standard deviation', 
                                             'standard error', 'maximum value', 'minimum value','x1',
                                             'x2', 'x3', 'x4'])
test_df['length'] = len_
test_df['mean'] = mean_
test_df['median'] = median_
test_df['mode'] = mode_
test_df['standard deviation'] = std_
test_df['standard error'] = ste_
test_df['maximum value'] = max_
test_df['minimum value'] = min_
test_df['x1'] = (np.array(mean_) + np.array(median_)) / 2.0
test_df['x2'] = np.array(mean_) + np.array(mode_) / 2.0
test_df['x3'] = np.array(mean_) + np.array(std_) / 2.0
test_df['x4'] = np.array(mean_) + np.array(ste_) / 2.0
test_df
y = train_df['y']
length = train_df['length'].apply(np.log)
mean = train_df['mean'].apply(np.abs)
length2 = test_df['length'].apply(np.log)
mean2 = test_df['mean'].apply(np.abs)
train_df = train_df.drop(['y'], axis=1)
train_df['minimum value'] = train_df['minimum value'].apply(np.abs)
train_df['standard error'] = train_df['standard error'].apply(np.log)
train_df['maximum value'] = train_df['maximum value'].apply(np.log)
train_df['minimum value'] = train_df['minimum value'].apply(np.log)
train_df['standard deviation'] = train_df['standard deviation'].apply(np.log)
train_df['standard deviation'] = (train_df['standard deviation'] + length) / 2.0
train_df['length'] = (train_df['maximum value'] * mean + length) / 2.0
train_df['length'] = train_df['length'].apply(np.abs)
train_df['length'] = train_df['length'].apply(np.log)
train_df['x1'] = train_df['x1'].apply(np.abs)
train_df['x1'] = train_df['x1'].apply(np.log)
train_df['x2'] = train_df['x2'].apply(np.abs)
train_df['x2'] = train_df['x2'].apply(np.log)
train_df['x3'] = train_df['x3'].apply(np.abs)
train_df['x3'] = train_df['x3'].apply(np.log)
train_df['x4'] = train_df['x4'].apply(np.abs)
train_df['x4'] = train_df['x4'].apply(np.log)
train_df['x1'] = train_df['x1'] + train_df['maximum value']
train_df['x2'] = train_df['x2'] + train_df['minimum value']
train_df['x4'] = train_df['x4'] + train_df['standard deviation']
test_df['minimum value'] = test_df['minimum value'].apply(np.abs)
test_df['standard error'] = test_df['standard error'].apply(np.log)
test_df['maximum value'] = test_df['maximum value'].apply(np.log)
test_df['minimum value'] = test_df['minimum value'].apply(np.log)
test_df['standard deviation'] = (test_df['standard deviation'].apply(np.log) + length2) / 2.0
test_df['length'] =  (test_df['maximum value'] * mean2 + length2) / 2.0
test_df['length'] = test_df['length'].apply(np.abs)
test_df['length'] = test_df['length'].apply(np.log)
test_df['x1'] = test_df['x1'].apply(np.abs)
test_df['x1'] = test_df['x1'].apply(np.log)
test_df['x2'] = test_df['x2'].apply(np.abs)
test_df['x2'] = test_df['x2'].apply(np.log)
test_df['x3'] = test_df['x3'].apply(np.abs)
test_df['x3'] = test_df['x3'].apply(np.log)
test_df['x4'] = test_df['x4'].apply(np.abs)
test_df['x4'] = test_df['x4'].apply(np.log)
test_df['x1'] = test_df['x1'] + test_df['maximum value']
test_df['x2'] = test_df['x2'] + test_df['minimum value']
test_df['x4'] = test_df['x4'] + test_df['standard deviation']
train_df = (train_df - train_df.mean()) / train_df.std()
test_df = (test_df - test_df.mean()) / test_df.std()
train_df
test_df
import matplotlib.pyplot as plt
import seaborn as sns

x = train_df.join(y)
corr = x.corr()

plt.style.use('ggplot')
plt.figure()
sns.heatmap(corr, square=True, annot=True)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
plt.figure()
sns.pairplot(train_df)
plt.show()
train_X = train_df
train_y = y
from imblearn.under_sampling import RandomUnderSampler

# 正例の数を保存
positive_count_train = np.sum(np.array(train_y))
rus = RandomUnderSampler({0:positive_count_train * 3, 1:positive_count_train}, random_state=0)
# 学習用データに反映
train_X, train_y = rus.fit_resample(train_X, train_y)
from collections import Counter
print('Original dataset shape %s' % Counter(train_y))
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

estimators = [
        ('svc', SVC(random_state=0)),
        ('lvc', LinearSVC(random_state=0)),
        ('gnb', GaussianNB()),
        ('mlp', MLPClassifier(random_state=0)),
        ('rfc', RandomForestClassifier(random_state=0))
        ]

clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=300),
)
clf.fit(X_train, y_train)
from sklearn.metrics import roc_curve, auc

y_pred = clf.predict(X_valid)  # 予測
fpr, tpr, thresholds = roc_curve(y_valid, y_pred)  # ROC曲線を求める
auc(fpr, tpr)  # 評価
from sklearn.metrics import roc_curve, auc

y_pred = clf.predict(train_X)  # 予測
fpr, tpr, thresholds = roc_curve(train_y, y_pred)  # ROC曲線を求める
auc(fpr, tpr)  # 評価
test_X = test_df
p_test = clf.predict_proba(test_X)
p_test
submit_df = pd.read_csv('/kaggle/input/1056lab-cardiac-arrhythmia-detection/sampleSubmission.csv', index_col=0)
submit_df['af'] = p_test[:,1]
submit_df
submit_df.to_csv('submission15.csv')

