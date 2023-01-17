import numpy as np, pandas as pd



f_lstm = '../input/improved-lstm-baseline-glove-dropout-lb-0-048/submission.csv'

f_nbsvm = '../input/nb-svm-strong-linear-baseline-eda-0-052-lb/submission.csv'

f_cnnrnn = '../input/keras-cnn-rnn-0-051-lb/baseline.csv'
p_lstm = pd.read_csv(f_lstm)

p_nbsvm = pd.read_csv(f_nbsvm)

p_cnnrnn = pd.read_csv(f_nbsvm)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

p_res = p_lstm.copy()

p_res[label_cols] = (p_cnnrnn[label_cols]*1 + p_nbsvm[label_cols]*3 + p_lstm[label_cols]*6) / 10
p_res.to_csv('submission.csv', index=False)