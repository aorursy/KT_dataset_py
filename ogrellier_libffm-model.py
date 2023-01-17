import numpy as np

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

test.insert(1, 'target', 0)
features = [_f for _f in train if _f not in ['id', 'target']]



def factor_encoding(train, test):

    

    assert sorted(train.columns) == sorted(test.columns)

    

    full = pd.concat([train, test], axis=0, sort=False)

    # Factorize everything

    for f in full:

        full[f], _ = pd.factorize(full[f])

        full[f] += 1  # make sure no negative

        

    return full.iloc[:train.shape[0]], full.iloc[train.shape[0]:]



train_f, test_f = factor_encoding(train[features], test[features])
class LibFFMEncoder(object):

    def __init__(self):

        self.encoder = 1

        self.encoding = {}



    def encode_for_libffm(self, row):

        txt = f"{row[0]}"

        for i, r in enumerate(row[1:]):

            try:

                txt += f' {i+1}:{self.encoding[(i, r)]}:1'

            except KeyError:

                self.encoding[(i, r)] = self.encoder

                self.encoder += 1

                txt += f' {i+1}:{self.encoding[(i, r)]}:1'



        return txt



# Create files for testing and OOF

from sklearn.model_selection import KFold

fold_ids = [

    [trn_, val_] for (trn_, val_) in KFold(5,True,1).split(train)

]

for fold_, (trn_, val_) in enumerate(fold_ids):

    # Fit the encoder

    encoder = LibFFMEncoder()

    libffm_format_trn = pd.concat([train['target'].iloc[trn_], train_f.iloc[trn_]], axis=1).apply(

        lambda row: encoder.encode_for_libffm(row), raw=True, axis=1

    )

    # Encode validation set

    libffm_format_val = pd.concat([train['target'].iloc[val_], train_f.iloc[val_]], axis=1).apply(

        lambda row: encoder.encode_for_libffm(row), raw=True, axis=1

    )

    

    print(train['target'].iloc[trn_].shape, train['target'].iloc[val_].shape, libffm_format_val.shape)

    

    libffm_format_trn.to_csv(f'libffm_trn_fold_{fold_+1}.txt', index=False, header=False)

    libffm_format_val.to_csv(f'libffm_val_fold_{fold_+1}.txt', index=False, header=False)

    

    

# Create files for final model

encoder = LibFFMEncoder()

libffm_format_trn = pd.concat([train['target'], train_f], axis=1).apply(

        lambda row: encoder.encode_for_libffm(row), raw=True, axis=1

)

libffm_format_tst = pd.concat([test['target'], test_f], axis=1).apply(

    lambda row: encoder.encode_for_libffm(row), raw=True, axis=1

)



libffm_format_trn.to_csv(f'libffm_trn.txt', index=False, header=False)

libffm_format_tst.to_csv(f'libffm_tst.txt', index=False, header=False)
!cp /kaggle/input/libffm-binaries/ffm-train .

!cp /kaggle/input/libffm-binaries/ffm-predict .

!chmod u+x ffm-train

!chmod u+x ffm-predict
from sklearn.metrics import log_loss, roc_auc_score



!./ffm-train -p libffm_val_fold_1.txt -r 0.05 -l 0.00001 -k 50 -t 7 libffm_trn_fold_1.txt libffm_fold_1_model

!./ffm-predict libffm_val_fold_1.txt libffm_fold_1_model val_preds_fold_1.txt

(

    log_loss(train['target'].iloc[fold_ids[0][1]], pd.read_csv('val_preds_fold_1.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[0][1]], pd.read_csv('val_preds_fold_1.txt', header=None).values[:,0])

)
!./ffm-train -p libffm_val_fold_2.txt -r 0.05 -l 0.00001 -k 50 -t 7 libffm_trn_fold_2.txt libffm_fold_2_model

!./ffm-predict libffm_val_fold_2.txt libffm_fold_2_model val_preds_fold_2.txt

(

    log_loss(train['target'].iloc[fold_ids[1][1]], pd.read_csv('val_preds_fold_2.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[1][1]], pd.read_csv('val_preds_fold_2.txt', header=None).values[:,0])

)
!./ffm-train -p libffm_val_fold_3.txt -r 0.05 -l 0.00001 -k 50 -t 7 libffm_trn_fold_3.txt libffm_fold_3_model

!./ffm-predict libffm_val_fold_3.txt libffm_fold_3_model val_preds_fold_3.txt

(

    log_loss(train['target'].iloc[fold_ids[2][1]], pd.read_csv('val_preds_fold_3.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[2][1]], pd.read_csv('val_preds_fold_3.txt', header=None).values[:,0])

)
!./ffm-train -p libffm_val_fold_4.txt -r 0.05 -l 0.00001 -k 50 -t 7 libffm_trn_fold_4.txt libffm_fold_4_model

!./ffm-predict libffm_val_fold_4.txt libffm_fold_4_model val_preds_fold_4.txt

(

    log_loss(train['target'].iloc[fold_ids[3][1]], pd.read_csv('val_preds_fold_4.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[3][1]], pd.read_csv('val_preds_fold_4.txt', header=None).values[:,0])

)
!./ffm-train -p libffm_val_fold_5.txt -r 0.05 -l 0.00001 -k 50 -t 7 libffm_trn_fold_5.txt libffm_fold_5_model

!./ffm-predict libffm_val_fold_5.txt libffm_fold_5_model val_preds_fold_5.txt

(

    log_loss(train['target'].iloc[fold_ids[4][1]], pd.read_csv('val_preds_fold_5.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[4][1]], pd.read_csv('val_preds_fold_5.txt', header=None).values[:,0])

)
oof_preds = np.zeros(train.shape[0])

for fold_, (_, val_) in enumerate(fold_ids):

    oof_preds[val_] = pd.read_csv(f'val_preds_fold_{fold_+1}.txt', header=None).values[:, 0]

oof_score = roc_auc_score(train['target'], oof_preds)

print(oof_score)
!./ffm-train -r 0.05 -l 0.00001 -k 50 -t 7 libffm_trn.txt libffm_model
!./ffm-predict libffm_tst.txt libffm_model tst_preds.txt
submission = test[['id']].copy()

submission['target'] = pd.read_csv('tst_preds.txt', header=None).values[:,0]

submission.to_csv('libffm_prediction.csv', index=False)