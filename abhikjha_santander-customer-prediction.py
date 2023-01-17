# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import gc

gc.collect()
# !pip install pretrainedmodels



%reload_ext autoreload

%autoreload 2

%matplotlib inline



!pip install fastai==1.0.52

import fastai



from fastai import *

from fastai.vision import *

from fastai.tabular import *



# from torchvision.models import *

# import pretrainedmodels



from utils import *

import sys



from fastai.callbacks.hooks import *



from fastai.callbacks.tracker import EarlyStoppingCallback

from fastai.callbacks.tracker import SaveModelCallback
from sklearn.metrics import roc_auc_score



def auroc_score(input, target):

    input, target = input.cpu().numpy()[:,1], target.cpu().numpy()

    return roc_auc_score(target, input)



class AUROC(Callback):

    _order = -20 #Needs to run before the recorder



    def __init__(self, learn, **kwargs): self.learn = learn

    def on_train_begin(self, **kwargs): self.learn.recorder.add_metric_names(['AUROC'])

    def on_epoch_begin(self, **kwargs): self.output, self.target = [], []

    

    def on_batch_end(self, last_target, last_output, train, **kwargs):

        if not train:

            self.output.append(last_output)

            self.target.append(last_target)

                

    def on_epoch_end(self, last_metrics, **kwargs):

        if len(self.output) > 0:

            output = torch.cat(self.output)

            target = torch.cat(self.target)

            preds = F.softmax(output, dim=1)

            metric = auroc_score(preds, target)

            return add_metrics(last_metrics, [metric])
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.shape, df_test.shape
df_train.columns
df_test.columns
df_train['target'].value_counts()
df_train.head()
df_train.drop(['ID_code'], axis=1, inplace=True)
df_train.head()
df_test.drop(['ID_code'], axis=1, inplace=True)
df_test.head()
df_train_new = df_train.drop(['target'], axis=1)

cont_names = df_train_new.columns

dep_var = 'target'

procs = [FillMissing, Categorify, Normalize]

cat_names = []
data = (TabularList.from_df(df_train, procs = procs, cont_names=cont_names, cat_names=cat_names)

        .split_by_rand_pct(0.2, seed=42)

        .label_from_df(cols=dep_var)

        .databunch(bs=64*4))
# data.add_test(TabularList.from_df(df_test, cont_names=cont_names))
data.show_batch()
learn = tabular_learner(data, layers=[1000,500], metrics=accuracy, callback_fns=AUROC, ps=[0.001,0.01], emb_drop=0.04)
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 5e-2

learn.fit_one_cycle(5, max_lr=lr,  pct_start=0.3, wd = 1e-2)
learn.fit_one_cycle(5, max_lr=lr,  pct_start=0.3, wd = 1e-2)
learn.recorder.plot_losses()
learn.fit_one_cycle(5, max_lr=1e-2, wd=1e-2)
learn.recorder.plot_losses()
learn.save('1st-round')

learn.load('1st-round')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
class SaveFeatures():

    features=None

    def __init__(self, m): 

        self.hook = m.register_forward_hook(self.hook_fn)

        self.features = None

    def hook_fn(self, module, input, output): 

        out = output.detach().cpu().numpy()

        if isinstance(self.features, type(None)):

            self.features = out

        else:

            self.features = np.row_stack((self.features, out))

    def remove(self): 

        self.hook.remove()
learn.model
sf = SaveFeatures(learn.model.layers[4])
_= learn.get_preds(data.train_ds)
label = [data.classes[x] for x in (list(data.train_ds.y.items))]
len(label)
df_new = pd.DataFrame({'label': label})
df_new['label'].value_counts()
array = np.array(sf.features)
x=array.tolist()
df_new['img_repr'] = x
df_new.head()
d2 = pd.DataFrame(df_new.img_repr.values.tolist(), index = df_new.index).rename(columns = lambda x: 'img_repr{}'.format(x+1))
df_new_2 = df_new.join(d2)
df_new_2.head(10)
df_new_2.shape
sf = SaveFeatures(learn.model.layers[4])
_=learn.get_preds(DatasetType.Valid)
data.valid_ds.y.items
label = [data.classes[x] for x in (list(data.valid_ds.y.items))]

df_new_valid = pd.DataFrame({'label': label})
df_new_valid['label'].value_counts()
array = np.array(sf.features)
x=array.tolist()
df_new_valid['img_repr'] = x
df_new_valid.head()
d2 = pd.DataFrame(df_new_valid.img_repr.values.tolist(), index = df_new_valid.index).rename(columns = lambda x: 'img_repr{}'.format(x+1))
df_new_valid_2 = df_new_valid.join(d2)
df_new_valid_2.head(10)
df_new_valid_2.shape
df_new_valid_2.drop(['img_repr'], axis=1, inplace=True)
df_new_valid_2.head()
df_new_2.drop(['img_repr'], axis=1, inplace=True)
df_new_2.shape
df_new_2.describe()
corr_matrix = df_new_2.corr()



corr_matrix["label"].sort_values(ascending = False)
X = df_new_2

y = df_new_2.label.copy()



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train = X_train.drop("label", axis =1)

y_train = y_train



X_test = X_test.drop("label", axis =1)

y_test = y_test
X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train.columns
from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    

    def __init__(self, attributes_names):

        self.attributes_names = attributes_names

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        return X[self.attributes_names].values
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



# numerical pipeline



num_pipeline = Pipeline([

    

    ('select_data', DataFrameSelector(X_train.columns)),

    ('Std_Scaler', StandardScaler())

])



X_train_transformed = num_pipeline.fit_transform(X_train)

X_test_transformed = num_pipeline.fit_transform(X_test)
X_train_transformed.shape, X_test_transformed.shape
from sklearn.ensemble import RandomForestClassifier

import time



start = time.time()



rf_clf = RandomForestClassifier(bootstrap=True, class_weight = {0:1, 1:5},

            criterion='gini', max_depth=21, max_features='auto',

            max_leaf_nodes=None, min_impurity_decrease=0.0,

            min_impurity_split=None, min_samples_leaf=3,

            min_samples_split=19, min_weight_fraction_leaf=0.0,

            n_estimators=277, n_jobs=1, oob_score=False, random_state=42,

            verbose=5, warm_start=False)



rf_clf.fit(X_train_transformed, y_train)



end = time.time()



print("run_time:", (end-start)/(60*60))
from sklearn.model_selection import cross_val_predict



import time



start = time.time()



y_train_pred_rf = cross_val_predict(rf_clf, X_train_transformed, y_train, cv=3, verbose=5)



end = time.time()



print("run_time:", (end-start)/(60*60))
from sklearn.metrics import confusion_matrix



confusion_matrix(y_train, y_train_pred_rf)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, cohen_kappa_score



print(precision_score(y_train, y_train_pred_rf))

print(recall_score(y_train, y_train_pred_rf))

print(f1_score(y_train, y_train_pred_rf))

print(cohen_kappa_score(y_train, y_train_pred_rf))



print(classification_report(y_train, y_train_pred_rf))
# import scipy.stats as st

# from sklearn.model_selection import RandomizedSearchCV



# one_to_left = st.beta(10, 1)  

# from_zero_positive = st.expon(0, 50)



# params = {  

#     "n_estimators": st.randint(50, 300),

#     "max_depth": st.randint(3, 40),

#    "min_samples_leaf": st.randint(3, 40),

#     "min_samples_split": st.randint(3, 20)

# }



# gs = RandomizedSearchCV(rf_clf, params)
# gs.fit(X_train_transformed, y_train)  
# gs.best_params_
from sklearn.model_selection import cross_val_predict



y_probas_rf = cross_val_predict(rf_clf, X_train_transformed, y_train, cv=3, method="predict_proba", verbose=0)

y_scores_rf = y_probas_rf[:,1]
roc_score_rf = roc_auc_score(y_train, y_scores_rf)

roc_score_rf
from sklearn.metrics import precision_recall_curve



precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_rf)



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")

    plt.plot(thresholds, recalls[:-1], "r-", label = "Recall")

    plt.xlabel("Thresholds")

    plt.legend(loc="upper left")

    

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()
from sklearn.metrics import roc_curve



fpr_rf, tpr_rf, threshold_rf = roc_curve(y_train, y_scores_rf)



def plot_roc_curve(fpr_rf, tpr_rf, figsize=(15,12), label = None):

    plt.plot(fpr_rf, tpr_rf, linewidth = 2, label = label)

    plt.plot([0,1], [0,1], "k--")

    plt.axis([0,1,0,1])

    plt.xlabel("False Pos Rate")

    plt.ylabel('True Neg Rate')

    

    

plot_roc_curve(fpr_rf, tpr_rf)
plt.plot(thresholds, recalls[1:], 'b', label='Threshold-Recall curve')

plt.title('Recall for different threshold values')

plt.xlabel('Thresholds')

plt.ylabel('Recall')

plt.show()
y_pred_test_rf = rf_clf.predict(X_test_transformed)
confusion_matrix(y_test, y_pred_test_rf)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, cohen_kappa_score



print(precision_score(y_test, y_pred_test_rf))

print(recall_score(y_test, y_pred_test_rf))

print(f1_score(y_test, y_pred_test_rf))

print(cohen_kappa_score(y_test, y_pred_test_rf))



print(classification_report(y_test, y_pred_test_rf))
y_probas_rf = rf_clf.predict_proba(X_test_transformed)

y_scores_rf = y_probas_rf[:,1]

roc_score_rf = roc_auc_score(y_test, y_scores_rf)

roc_score_rf
from sklearn.metrics import precision_recall_curve



precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores_rf)



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")

    plt.plot(thresholds, recalls[:-1], "r-", label = "Recall")

    plt.xlabel("Thresholds")

    plt.legend(loc="upper left")

    

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()
from sklearn.metrics import roc_curve



fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test, y_scores_rf)



def plot_roc_curve(fpr_rf, tpr_rf, figsize=(15,12), label = None):

    plt.plot(fpr_rf, tpr_rf, linewidth = 2, label = label)

    plt.plot([0,1], [0,1], "k--")

    plt.axis([0,1,0,1])

    plt.xlabel("False Pos Rate")

    plt.ylabel('True Neg Rate')

      

plot_roc_curve(fpr_rf, tpr_rf)
df_test.head()
df_test.shape
X = df_new_valid_2

y = df_new_valid_2.label.copy()
X_val = X.drop("label", axis =1)

y_val = y
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



# numerical pipeline



num_pipeline = Pipeline([

    

    ('select_data', DataFrameSelector(X_val.columns)),

    ('Std_Scaler', StandardScaler())

])





X_val_transformed = num_pipeline.fit_transform(X_val)
y_pred_test_rf_val = rf_clf.predict(X_val_transformed)
confusion_matrix(y_val, y_pred_test_rf_val)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, cohen_kappa_score



print(precision_score(y_val, y_pred_test_rf_val))

print(recall_score(y_val, y_pred_test_rf_val))

print(f1_score(y_val, y_pred_test_rf_val))

print(cohen_kappa_score(y_val, y_pred_test_rf_val))



print(classification_report(y_val, y_pred_test_rf_val))
y_probas_rf = rf_clf.predict_proba(X_val_transformed)

y_scores_rf = y_probas_rf[:,1]

roc_score_rf = roc_auc_score(y_val, y_scores_rf)

roc_score_rf
from sklearn.metrics import precision_recall_curve



precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores_rf)



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")

    plt.plot(thresholds, recalls[:-1], "r-", label = "Recall")

    plt.xlabel("Thresholds")

    plt.legend(loc="upper left")

    

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()
from sklearn.metrics import roc_curve



fpr_rf, tpr_rf, threshold_rf = roc_curve(y_val, y_scores_rf)



def plot_roc_curve(fpr_rf, tpr_rf, figsize=(15,12), label = None):

    plt.plot(fpr_rf, tpr_rf, linewidth = 2, label = label)

    plt.plot([0,1], [0,1], "k--")

    plt.axis([0,1,0,1])

    plt.xlabel("False Pos Rate")

    plt.ylabel('True Neg Rate')

      

plot_roc_curve(fpr_rf, tpr_rf)