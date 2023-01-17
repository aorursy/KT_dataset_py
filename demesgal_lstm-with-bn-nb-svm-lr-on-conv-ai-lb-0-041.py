import numpy as np, pandas as pd

f_lstm = '../input/lstm-glove-lr-decrease-bn-cv-lb-0-047/LSTM-submission.csv'
f_nbsvm = '../input/nb-svm-strong-linear-baseline-eda-0-052-lb/submission.csv'
f_eaf = '../input/easy-and-fast-lb-044/feat_lr_2cols.csv'
f_tfidf = '../input/word-character-n-grams-tfidf-regressions-lb-051/output.csv'

p_lstm = pd.read_csv(f_lstm)
p_tfidf = pd.read_csv(f_tfidf)
#p_nbsvm = pd.read_csv(f_nbsvm)
#p_eaf = pd.read_csv(f_eaf)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p_lstm.copy()
#p_res[label_cols] = (2*p_nbsvm[label_cols] + 3*p_lstm[label_cols] + 4*p_eaf[label_cols]) / 9
p_res[label_cols] = (1*p_tfidf[label_cols] + 2*p_lstm[label_cols]) / 3
p_res.to_csv('submission_ensemble.csv', index=False)