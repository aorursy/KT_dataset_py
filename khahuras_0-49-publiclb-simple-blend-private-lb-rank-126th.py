import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** .5

test = pd.read_csv('../input/santander-value-prediction-challenge/test.csv')
tst_leak = pd.read_csv('../input/santander-public-outputs/test_leak_37.csv')
test['leak'] = tst_leak['compiled_leak']

merge_files = ['137','138_1','138_2','138_3','138_4','138_5']
weights = [3,1,1,1,1,1] # the weights are invented by sense

score = pd.read_csv('../input/santander-value-prediction-challenge/sample_submission.csv', usecols=['ID'])

for i in range(len(merge_files)):
    score_temp = pd.read_csv('../input/santander-public-outputs/'+ merge_files[i] +'.csv').rename(columns={'target':'score_'+ str(i)})
    score = pd.merge(score, score_temp, how='left', on='ID')
score.head()    
# Compute weighted average
sum_pred = np.zeros(len(score), dtype=float)
for i in range(len(merge_files)):
    sum_pred = sum_pred + list(score['score_'+str(i)].values*weights[i])
    
avg_pred = sum_pred/sum(weights)
filesave = "Blend_Finale"
lgsub = pd.DataFrame(avg_pred,columns=["target"])
lgsub['ID'] = score['ID'].values
lgsub['leak'] = tst_leak['compiled_leak']
# Replace leak rows
lgsub.loc[lgsub.leak.notnull(),'target'] = lgsub.loc[lgsub.leak.notnull(), 'leak'] 
# Write output
lgsub[['ID','target']].to_csv(filesave+".csv",index=False,header=True)
lgsub[['ID','target']].head()
