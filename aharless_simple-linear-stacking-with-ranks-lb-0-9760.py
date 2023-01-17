# File containing validation data
# (These are selected from the last day of the original training set
#  to correspond to the times of day used in the test set.)
VAL_FILE = '../input/training-and-validation-data-pickle/validation.pkl.gz'
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

print(os.listdir("../input"))
almost_zero = 1e-10
almost_one = 1 - almost_zero
# Just names to identify the models
base_models = {
    'lgb1 ': "assemblage of krishna's LGBM with time deltas",
    'wbftl': "anttip's Wordbatch FM-FTRL",
    'nngpu': "Downampled Neural Network run on GPU"
    }
# Files with validation set predictions from each of the base models
# (These were fit on a subset of the training data that ends a day before
#  the end of the full training set.)
cvfiles = {
    'lgb1 ': '../input/validation-of-pas-un-m-lange/val_krishnas_r_lgb_bag3.csv',
    'wbftl': '../input/validate-anttip-s-wordbatch-fm-ftrl-9752-version/wordbatch_fm_ftrl_val.csv',
    'nngpu': '../input/gpu-nn-validation-more-features/gpu_val3.csv'
    }
# Files with test set predictions
# (These were fit on the full training set
#  or on a subset at the end, to accommodate memory limits.)
subfiles = {
    'lgb1 ': '../input/ceci-n-est-pas-un-m-lange/sub_krishnas_r_lgb_bag4.csv',
    'wbftl': '../input/anttip-s-wordbatch-fm-ftrl-9752-version/wordbatch_fm_ftrl.csv',
    'nngpu': '../input/new-talkingdata-gpu-example-with-multiple-runs/gpu_test3.csv'
    }
# Public leaderbaord scores, for comparison
lbscores = {
    'lgb1 ': .9759,
    'wbftl': .9752,
    'nngpu': .9695
}
model_order = [m for m in base_models] # To keep order consistent when converting to array
cvdata = pd.DataFrame( { 
    m:pd.read_csv(cvfiles[m])['is_attributed'].rank()
    for m in base_models
    } )
X_train = np.array(cvdata[model_order])
y_train = pd.read_pickle(VAL_FILE)['is_attributed']
n = len(y_train)
X_train /= n
cvdata.head()
cvdata.corr()
X, X_val, y, y_val = train_test_split(X_train, y_train, test_size=.33, shuffle=False )
for c in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
    mod = LogisticRegression(C=c)
    mod.fit(X, y)
    val_pred = mod.predict_proba(X_val)[:,1]
    print( c, roc_auc_score(y_val, mod.predict_proba(X_val)[:,1]) )
stack_model = LogisticRegression(C=1e6)
stack_model.fit(X_train, y_train)
stack_model.coef_
weights = stack_model.coef_/stack_model.coef_.sum()
columns = cvdata[model_order].columns
scores = [ roc_auc_score( y_train, cvdata[c] )  for c in columns ]
names = [ base_models[c] for c in columns ]
lb = [ lbscores[c] for c in columns ]
pd.DataFrame( data={'LB score': lb, 'CV score':scores, 'weight':weights.reshape(-1)}, index=names )
print(  'Stacker score: ', roc_auc_score( y_train, stack_model.predict_proba(X_train)[:,1] )  )
final_sub = pd.DataFrame()
subs = {m:pd.read_csv(subfiles[m]).rename({'is_attributed':m},axis=1) for m in base_models}
first_model = list(base_models.keys())[0]
final_sub['click_id'] = subs[first_model]['click_id']
df = subs[first_model]
for m in subs:
    if m != first_model:
        df = df.merge(subs[m], on='click_id')  # being careful in case clicks are in different order
df.head()
X_test = np.array( df.drop(['click_id'],axis=1)[model_order].rank()/df.shape[0] )
final_sub['is_attributed'] = stack_model.predict_proba(X_test)[:,1]
final_sub.head(10)
final_sub['is_attributed'] = final_sub['is_attributed'].rank(method='dense') / 1e8
pd.options.display.float_format = ('{:,.8f}').format
final_sub.head(10)
final_sub.to_csv("sub_stacked.csv", index=False, float_format='%.8f')