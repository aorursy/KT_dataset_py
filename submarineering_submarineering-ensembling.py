import numpy as np, pandas as pd



f_lstm = '../input/improved-lstm-baseline-glove-dropout-lb-0-048/submission.csv'

#f_nbsvm = '../input/nb-svm-strong-linear-baseline-eda-0-052-lb/submission.csv'

f_eaf = '../input/easy-and-fast-lb-044/feat_lr_2cols.csv'

f_agg = '../input/submarineering-ensembling/submission5.csv'
p_lstm = pd.read_csv(f_lstm)

#p_nbsvm = pd.read_csv(f_nbsvm)

p_eaf = pd.read_csv(f_eaf)

p_agg = pd.read_csv(f_agg)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

p_res = p_lstm.copy()

p_res[label_cols] = (2* p_lstm[label_cols] + 3*p_eaf[label_cols] + 4*p_agg[label_cols])/ 9

p_res.to_csv('submission6.csv', index=False)