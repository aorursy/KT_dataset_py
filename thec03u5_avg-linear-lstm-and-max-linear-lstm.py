import numpy as np, pandas as pd

enAvg = '../input/ensamble-and-max-nbsvm-and-tfidflr/submissionAvg.csv'
enMax = '../input/ensamble-and-max-nbsvm-and-tfidflr/submissionMax.csv'
enCond = '../input/ensamble-and-max-nbsvm-and-tfidflr/submissionCondMax.csv'
lstm = '../input/basic-lstm/submissionLSTM.csv'
p_enAvg = pd.read_csv(enAvg)
p_enMax = pd.read_csv(enMax)
p_enCond = pd.read_csv(enCond)
p_lstm = pd.read_csv(lstm)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# p_res = p_lstm.copy()
# p_res[label_cols] = (p_enCond[label_cols] + p_enAvg[label_cols]) / 2
# p_res.to_csv('submissionlstmAvgAvg.csv', index=False)
p_res = p_lstm.copy()
p_res[label_cols] = (p_lstm[label_cols] + p_enCond[label_cols]) / 2
p_res.to_csv('submissionCondLstmAvg.csv', index=False)
p_res2 = p_lstm.copy()
for i, j in enumerate(p_res2):
    if j == 'id':
        continue
    temp_df = pd.concat([p_lstm[j], p_enCond[j]], axis=1)
    p_res2[j] = temp_df.apply(lambda r: np.max(r) if np.mean(r) > 0.5 else np.mean(r), axis=1)
p_res2.to_csv('submissionlstmCondCondMax.csv', index=False)
p_res2 = p_lstm.copy()
for i, j in enumerate(p_res2):
    if j == 'id':
        continue
    temp_df = pd.concat([p_lstm[j], p_enCond[j]], axis=1)
    p_res2[j] = temp_df.apply(lambda r: np.max(r) if np.mean(r) > 0.5 else np.min(r), axis=1)
p_res2.to_csv('submissionlstmCondCondMaxMin.csv', index=False)
