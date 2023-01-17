import numpy as np

import pandas as pd 

import os

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import rankdata



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



N_Splits = 25

SEED = 2020
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

from sklearn.model_selection import KFold, StratifiedKFold

fold_ids = [

    [trn_, val_] for (trn_, val_) in StratifiedKFold(N_Splits,True,SEED).split(train, train['target'])

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

    libffm_format_tst = pd.concat([test['target'], test_f], axis=1).apply(

        lambda row: encoder.encode_for_libffm(row), raw=True, axis=1

    )

    print(train['target'].iloc[trn_].shape, train['target'].iloc[val_].shape, libffm_format_tst.shape)

    

    libffm_format_trn.to_csv(f'libffm_trn_fold_{fold_+1}.txt', index=False, header=False)

    libffm_format_val.to_csv(f'libffm_val_fold_{fold_+1}.txt', index=False, header=False)

    libffm_format_tst.to_csv(f'libffm_tst_fold_{fold_+1}.txt', index=False, header=False)



    



!cp /kaggle/input/libffm-binaries/ffm-train .

!cp /kaggle/input/libffm-binaries/ffm-predict .

!chmod u+x ffm-train

!chmod u+x ffm-predict
from sklearn.metrics import log_loss, roc_auc_score



!./ffm-train -p libffm_val_fold_1.txt -r 0.05 -l 0.0002 -k 50 --auto-stop libffm_trn_fold_1.txt libffm_fold_1_model

!./ffm-predict libffm_val_fold_1.txt libffm_fold_1_model val_preds_fold_1.txt

!./ffm-predict libffm_tst_fold_1.txt libffm_fold_1_model tst_preds_fold_1.txt

os.remove('libffm_val_fold_1.txt')

os.remove('libffm_trn_fold_1.txt')

os.remove('libffm_fold_1_model')

os.remove('libffm_tst_fold_1.txt')



(

    log_loss(train['target'].iloc[fold_ids[0][1]], pd.read_csv('val_preds_fold_1.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[0][1]], pd.read_csv('val_preds_fold_1.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_2.txt -r 0.05 -l 0.00002 -k 50  --auto-stop libffm_trn_fold_2.txt libffm_fold_2_model

!./ffm-predict libffm_val_fold_2.txt libffm_fold_2_model val_preds_fold_2.txt

!./ffm-predict libffm_tst_fold_2.txt libffm_fold_2_model tst_preds_fold_2.txt

os.remove('libffm_val_fold_2.txt')

os.remove('libffm_trn_fold_2.txt')

os.remove('libffm_fold_2_model')

os.remove('libffm_tst_fold_2.txt')

(

    log_loss(train['target'].iloc[fold_ids[1][1]], pd.read_csv('val_preds_fold_2.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[1][1]], pd.read_csv('val_preds_fold_2.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_3.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_3.txt libffm_fold_3_model

!./ffm-predict libffm_val_fold_3.txt libffm_fold_3_model val_preds_fold_3.txt

!./ffm-predict libffm_tst_fold_3.txt libffm_fold_3_model tst_preds_fold_3.txt

os.remove('libffm_val_fold_3.txt')

os.remove('libffm_trn_fold_3.txt')

os.remove('libffm_fold_3_model')

os.remove('libffm_tst_fold_3.txt')

(

    log_loss(train['target'].iloc[fold_ids[2][1]], pd.read_csv('val_preds_fold_3.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[2][1]], pd.read_csv('val_preds_fold_3.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_4.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_4.txt libffm_fold_4_model

!./ffm-predict libffm_val_fold_4.txt libffm_fold_4_model val_preds_fold_4.txt

!./ffm-predict libffm_tst_fold_4.txt libffm_fold_4_model tst_preds_fold_4.txt

os.remove('libffm_val_fold_4.txt')

os.remove('libffm_trn_fold_4.txt')

os.remove('libffm_fold_4_model')

os.remove('libffm_tst_fold_4.txt')

(

    log_loss(train['target'].iloc[fold_ids[3][1]], pd.read_csv('val_preds_fold_4.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[3][1]], pd.read_csv('val_preds_fold_4.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_5.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_5.txt libffm_fold_5_model

!./ffm-predict libffm_val_fold_5.txt libffm_fold_5_model val_preds_fold_5.txt

!./ffm-predict libffm_tst_fold_5.txt libffm_fold_5_model tst_preds_fold_5.txt

os.remove('libffm_val_fold_5.txt')

os.remove('libffm_trn_fold_5.txt')

os.remove('libffm_fold_5_model')

os.remove('libffm_tst_fold_5.txt')

(

    log_loss(train['target'].iloc[fold_ids[4][1]], pd.read_csv('val_preds_fold_5.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[4][1]], pd.read_csv('val_preds_fold_5.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_6.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_6.txt libffm_fold_6_model

!./ffm-predict libffm_val_fold_6.txt libffm_fold_6_model val_preds_fold_6.txt

!./ffm-predict libffm_tst_fold_6.txt libffm_fold_6_model tst_preds_fold_6.txt

os.remove('libffm_val_fold_6.txt')

os.remove('libffm_trn_fold_6.txt')

os.remove('libffm_fold_6_model')

os.remove('libffm_tst_fold_6.txt')

(

    log_loss(train['target'].iloc[fold_ids[5][1]], pd.read_csv('val_preds_fold_6.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[5][1]], pd.read_csv('val_preds_fold_6.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_7.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_7.txt libffm_fold_7_model

!./ffm-predict libffm_val_fold_7.txt libffm_fold_7_model val_preds_fold_7.txt

!./ffm-predict libffm_tst_fold_7.txt libffm_fold_7_model tst_preds_fold_7.txt

os.remove('libffm_val_fold_7.txt')

os.remove('libffm_trn_fold_7.txt')

os.remove('libffm_fold_7_model')

os.remove('libffm_tst_fold_7.txt')

(

    log_loss(train['target'].iloc[fold_ids[6][1]], pd.read_csv('val_preds_fold_7.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[6][1]], pd.read_csv('val_preds_fold_7.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_8.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_8.txt libffm_fold_8_model

!./ffm-predict libffm_val_fold_8.txt libffm_fold_8_model val_preds_fold_8.txt

!./ffm-predict libffm_tst_fold_8.txt libffm_fold_8_model tst_preds_fold_8.txt

os.remove('libffm_val_fold_8.txt')

os.remove('libffm_trn_fold_8.txt')

os.remove('libffm_fold_8_model')

os.remove('libffm_tst_fold_8.txt')

(

    log_loss(train['target'].iloc[fold_ids[7][1]], pd.read_csv('val_preds_fold_8.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[7][1]], pd.read_csv('val_preds_fold_8.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_9.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_9.txt libffm_fold_9_model

!./ffm-predict libffm_val_fold_9.txt libffm_fold_9_model val_preds_fold_9.txt

!./ffm-predict libffm_tst_fold_9.txt libffm_fold_9_model tst_preds_fold_9.txt

os.remove('libffm_val_fold_9.txt')

os.remove('libffm_trn_fold_9.txt')

os.remove('libffm_fold_9_model')

os.remove('libffm_tst_fold_9.txt')

(

    log_loss(train['target'].iloc[fold_ids[8][1]], pd.read_csv('val_preds_fold_9.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[8][1]], pd.read_csv('val_preds_fold_9.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_10.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_10.txt libffm_fold_10_model

!./ffm-predict libffm_val_fold_10.txt libffm_fold_10_model val_preds_fold_10.txt

!./ffm-predict libffm_tst_fold_10.txt libffm_fold_10_model tst_preds_fold_10.txt

os.remove('libffm_val_fold_10.txt')

os.remove('libffm_trn_fold_10.txt')

os.remove('libffm_fold_10_model')

os.remove('libffm_tst_fold_10.txt')

(

    log_loss(train['target'].iloc[fold_ids[9][1]], pd.read_csv('val_preds_fold_10.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[9][1]], pd.read_csv('val_preds_fold_10.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_11.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_11.txt libffm_fold_11_model

!./ffm-predict libffm_val_fold_11.txt libffm_fold_11_model val_preds_fold_11.txt

!./ffm-predict libffm_tst_fold_11.txt libffm_fold_11_model tst_preds_fold_11.txt

os.remove('libffm_val_fold_11.txt')

os.remove('libffm_trn_fold_11.txt')

os.remove('libffm_fold_11_model')

os.remove('libffm_tst_fold_11.txt')

(

    log_loss(train['target'].iloc[fold_ids[10][1]], pd.read_csv('val_preds_fold_11.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[10][1]], pd.read_csv('val_preds_fold_11.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_12.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_12.txt libffm_fold_12_model

!./ffm-predict libffm_val_fold_12.txt libffm_fold_12_model val_preds_fold_12.txt

!./ffm-predict libffm_tst_fold_12.txt libffm_fold_12_model tst_preds_fold_12.txt

os.remove('libffm_val_fold_12.txt')

os.remove('libffm_trn_fold_12.txt')

os.remove('libffm_fold_12_model')

os.remove('libffm_tst_fold_12.txt')

(

    log_loss(train['target'].iloc[fold_ids[11][1]], pd.read_csv('val_preds_fold_12.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[11][1]], pd.read_csv('val_preds_fold_12.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_13.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_13.txt libffm_fold_13_model

!./ffm-predict libffm_val_fold_13.txt libffm_fold_13_model val_preds_fold_13.txt

!./ffm-predict libffm_tst_fold_13.txt libffm_fold_13_model tst_preds_fold_13.txt

os.remove('libffm_val_fold_13.txt')

os.remove('libffm_trn_fold_13.txt')

os.remove('libffm_fold_13_model')

os.remove('libffm_tst_fold_13.txt')

(

    log_loss(train['target'].iloc[fold_ids[12][1]], pd.read_csv('val_preds_fold_13.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[12][1]], pd.read_csv('val_preds_fold_13.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_14.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_14.txt libffm_fold_14_model

!./ffm-predict libffm_val_fold_14.txt libffm_fold_14_model val_preds_fold_14.txt

!./ffm-predict libffm_tst_fold_14.txt libffm_fold_14_model tst_preds_fold_14.txt

os.remove('libffm_val_fold_14.txt')

os.remove('libffm_trn_fold_14.txt')

os.remove('libffm_fold_14_model')

os.remove('libffm_tst_fold_14.txt')

(

    log_loss(train['target'].iloc[fold_ids[13][1]], pd.read_csv('val_preds_fold_14.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[13][1]], pd.read_csv('val_preds_fold_14.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_15.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_15.txt libffm_fold_15_model

!./ffm-predict libffm_val_fold_15.txt libffm_fold_15_model val_preds_fold_15.txt

!./ffm-predict libffm_tst_fold_15.txt libffm_fold_15_model tst_preds_fold_15.txt

os.remove('libffm_val_fold_15.txt')

os.remove('libffm_trn_fold_15.txt')

os.remove('libffm_fold_15_model')

os.remove('libffm_tst_fold_15.txt')

(

    log_loss(train['target'].iloc[fold_ids[14][1]], pd.read_csv('val_preds_fold_15.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[14][1]], pd.read_csv('val_preds_fold_15.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_16.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_16.txt libffm_fold_16_model

!./ffm-predict libffm_val_fold_16.txt libffm_fold_16_model val_preds_fold_16.txt

!./ffm-predict libffm_tst_fold_16.txt libffm_fold_16_model tst_preds_fold_16.txt

os.remove('libffm_val_fold_16.txt')

os.remove('libffm_trn_fold_16.txt')

os.remove('libffm_fold_16_model')

os.remove('libffm_tst_fold_16.txt')

(

    log_loss(train['target'].iloc[fold_ids[15][1]], pd.read_csv('val_preds_fold_16.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[15][1]], pd.read_csv('val_preds_fold_16.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_17.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_17.txt libffm_fold_17_model

!./ffm-predict libffm_val_fold_17.txt libffm_fold_17_model val_preds_fold_17.txt

!./ffm-predict libffm_tst_fold_17.txt libffm_fold_17_model tst_preds_fold_17.txt

os.remove('libffm_val_fold_17.txt')

os.remove('libffm_trn_fold_17.txt')

os.remove('libffm_fold_17_model')

os.remove('libffm_tst_fold_17.txt')

(

    log_loss(train['target'].iloc[fold_ids[16][1]], pd.read_csv('val_preds_fold_17.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[16][1]], pd.read_csv('val_preds_fold_17.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_18.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_18.txt libffm_fold_18_model

!./ffm-predict libffm_val_fold_18.txt libffm_fold_18_model val_preds_fold_18.txt

!./ffm-predict libffm_tst_fold_18.txt libffm_fold_18_model tst_preds_fold_18.txt

os.remove('libffm_val_fold_18.txt')

os.remove('libffm_trn_fold_18.txt')

os.remove('libffm_fold_18_model')

os.remove('libffm_tst_fold_18.txt')

(

    log_loss(train['target'].iloc[fold_ids[17][1]], pd.read_csv('val_preds_fold_18.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[17][1]], pd.read_csv('val_preds_fold_18.txt', header=None).values[:,0])

)



!./ffm-train -p libffm_val_fold_19.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_19.txt libffm_fold_19_model

!./ffm-predict libffm_val_fold_19.txt libffm_fold_19_model val_preds_fold_19.txt

!./ffm-predict libffm_tst_fold_19.txt libffm_fold_19_model tst_preds_fold_19.txt

os.remove('libffm_val_fold_19.txt')

os.remove('libffm_trn_fold_19.txt')

os.remove('libffm_fold_19_model')

os.remove('libffm_tst_fold_19.txt')

(

    log_loss(train['target'].iloc[fold_ids[18][1]], pd.read_csv('val_preds_fold_19.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[18][1]], pd.read_csv('val_preds_fold_19.txt', header=None).values[:,0])

)



!./ffm-train -p libffm_val_fold_20.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_20.txt libffm_fold_20_model

!./ffm-predict libffm_val_fold_20.txt libffm_fold_20_model val_preds_fold_20.txt

!./ffm-predict libffm_tst_fold_20.txt libffm_fold_20_model tst_preds_fold_20.txt

os.remove('libffm_val_fold_20.txt')

os.remove('libffm_trn_fold_20.txt')

os.remove('libffm_fold_20_model')

os.remove('libffm_tst_fold_20.txt')

(

    log_loss(train['target'].iloc[fold_ids[19][1]], pd.read_csv('val_preds_fold_20.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[19][1]], pd.read_csv('val_preds_fold_20.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_21.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_21.txt libffm_fold_21_model

!./ffm-predict libffm_val_fold_21.txt libffm_fold_21_model val_preds_fold_21.txt

!./ffm-predict libffm_tst_fold_21.txt libffm_fold_21_model tst_preds_fold_21.txt

os.remove('libffm_val_fold_21.txt')

os.remove('libffm_trn_fold_21.txt')

os.remove('libffm_fold_21_model')

os.remove('libffm_tst_fold_21.txt')

(

    log_loss(train['target'].iloc[fold_ids[20][1]], pd.read_csv('val_preds_fold_21.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[20][1]], pd.read_csv('val_preds_fold_21.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_22.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_22.txt libffm_fold_22_model

!./ffm-predict libffm_val_fold_22.txt libffm_fold_22_model val_preds_fold_22.txt

!./ffm-predict libffm_tst_fold_22.txt libffm_fold_22_model tst_preds_fold_22.txt

os.remove('libffm_val_fold_22.txt')

os.remove('libffm_trn_fold_22.txt')

os.remove('libffm_fold_22_model')

os.remove('libffm_tst_fold_22.txt')

(

    log_loss(train['target'].iloc[fold_ids[21][1]], pd.read_csv('val_preds_fold_22.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[21][1]], pd.read_csv('val_preds_fold_22.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_23.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_23.txt libffm_fold_23_model

!./ffm-predict libffm_val_fold_23.txt libffm_fold_23_model val_preds_fold_23.txt

!./ffm-predict libffm_tst_fold_23.txt libffm_fold_23_model tst_preds_fold_23.txt

os.remove('libffm_val_fold_23.txt')

os.remove('libffm_trn_fold_23.txt')

os.remove('libffm_fold_23_model')

os.remove('libffm_tst_fold_23.txt')

(

    log_loss(train['target'].iloc[fold_ids[22][1]], pd.read_csv('val_preds_fold_23.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[22][1]], pd.read_csv('val_preds_fold_23.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_24.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_24.txt libffm_fold_24_model

!./ffm-predict libffm_val_fold_24.txt libffm_fold_24_model val_preds_fold_24.txt

!./ffm-predict libffm_tst_fold_24.txt libffm_fold_24_model tst_preds_fold_24.txt

os.remove('libffm_val_fold_24.txt')

os.remove('libffm_trn_fold_24.txt')

os.remove('libffm_fold_24_model')

os.remove('libffm_tst_fold_24.txt')

(

    log_loss(train['target'].iloc[fold_ids[23][1]], pd.read_csv('val_preds_fold_24.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[23][1]], pd.read_csv('val_preds_fold_24.txt', header=None).values[:,0])

)

!./ffm-train -p libffm_val_fold_25.txt -r 0.05 -l 0.00002 -k 50 --auto-stop libffm_trn_fold_25.txt libffm_fold_25_model

!./ffm-predict libffm_val_fold_25.txt libffm_fold_25_model val_preds_fold_25.txt

!./ffm-predict libffm_tst_fold_25.txt libffm_fold_25_model tst_preds_fold_25.txt

os.remove('libffm_val_fold_25.txt')

os.remove('libffm_trn_fold_25.txt')

os.remove('libffm_fold_25_model')

os.remove('libffm_tst_fold_25.txt')

(

    log_loss(train['target'].iloc[fold_ids[24][1]], pd.read_csv('val_preds_fold_25.txt', header=None).values[:,0]),

    roc_auc_score(train['target'].iloc[fold_ids[24][1]], pd.read_csv('val_preds_fold_25.txt', header=None).values[:,0])

)

oof_preds = np.zeros(train.shape[0])

for fold_, (_, val_) in enumerate(fold_ids):

    oof_preds[val_] = pd.read_csv(f'val_preds_fold_{fold_+1}.txt', header=None).values[:, 0]

oof_score = roc_auc_score(train['target'], oof_preds)

print(oof_score)
test_preds = np.zeros((test.shape[0], N_Splits))

for fold_ in range(N_Splits):

    test_preds[:, fold_] = pd.read_csv(f'tst_preds_fold_{fold_+1}.txt', header=None).values[:, 0]



test_preds_avg = test_preds.mean(axis=1)
submission = test[['id']].copy()

submission['target'] = test_preds_avg

submission.to_csv('libffm_sub_531.csv', index=False)
np.save('test_preds_libffm.npy', test_preds_avg)

np.save('oof_preds_libffm.npy', oof_preds)
subs = [

    '/kaggle/input/bestpublicscores/sub_623.csv',

    '/kaggle/input/bestpublicscores/sub_634.csv',

    '/kaggle/input/bestpublicscores/sub_626.csv',

    '/kaggle/input/bestpublicscores/sub_600.csv',

    '/kaggle/input/bestpublicscores/sub_590.csv',

    '/kaggle/input/otherbestpublicscores/sub_659.csv',

    '/kaggle/input/otherbestpublicscores/sub_606.csv',

    '/kaggle/input/otherbestpublicscores/sub_563.csv',

    '/kaggle/input/otherbestpublicscores/sub_620.csv',

    '/kaggle/input/bestpublicscores3/sub_589.csv',

    'libffm_sub_531.csv'

       ]

predictions = pd.concat([pd.read_csv(sub, index_col='id') for sub in subs], axis=1).reset_index(drop=True)

predictions.columns = ['sub_'+str(i) for i in range(11)]

predictions
for col in predictions.columns:

    predictions[col]=predictions[col].rank()/predictions.shape[0]
corr = predictions.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corr, mask=mask, cmap='Blues', vmin=0.95, center=0, linewidths=1, annot=True, fmt='.4f')

plt.show()
coefs = [0.1, 0.075, 0.1, 0.05, 0.025, 0.375, 0.1, 0.05, 0.05, 0.05, 0.025]

def blend_subs(df, coefs=coefs):

    blend = np.zeros(df.shape[0])

    for idx, column in enumerate(df.columns):

        blend += coefs[idx] * (df[column].values)

    return blend



blend = blend_subs(predictions)
blend
submission['target'] = blend

submission.to_csv('TopPublicBlend.csv',index=False)

submission.head()