import numpy as np
import pandas as pd
import os

from scipy.stats import rankdata

LABELS = ["HasDetections"]
!ls ../input/detecting-malwares-with-ftrl-proximal
!ls ../input/outputs-for-microsoft
predict_list = []


predict_list.append(pd.read_csv("../input/lightgbm/submission_lgbm.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/malware-predictions-3500-trees/submission0.72968.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/detecting-malwares-with-ftrl-proximal/submission.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/outputs-for-microsoft/submission_ashish_kfold.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/hung-the-nguyen/submission_lgbm_5.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/hung-the-nguyen/submission_lgbm_6.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/hung-the-nguyen/submission_lgbm_7.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/hung-the-nguyen/submission_lgbm_8.csv")[LABELS].values)
predict_list.append(pd.read_csv("../input/hung-the-nguyen/submission_lgbm_9.csv")[LABELS].values)
print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    for i in range(1):
        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])  
predictions /= len(predict_list)

submission = pd.read_csv('../input/microsoft-malware-prediction/sample_submission.csv')
submission[LABELS] = predictions
submission.to_csv('super_blend.csv', index=False)
submission.head()
import numpy as np # NUMPY
import pandas as pd # PANDAS

# DATA VIZUALIZATION LIBRARIES
from matplotlib import pyplot as plt
import seaborn as sns

# METRICS TO MEASURE RMSE
from math import sqrt
from sklearn import metrics
#ALL PUBLIC SOLUTION RMSE < 0.2269 (WITHOUT REPETITIONS)
df_base0 = pd.read_csv("../input/lightgbm/submission_lgbm.csv",names=["MachineIdentifier","HasDetections0"], skiprows=[0],header=None)
df_base1 = pd.read_csv("../input/malware-predictions-3500-trees/submission0.72968.csv",names=["MachineIdentifier","HasDetections1"], skiprows=[0],header=None)
df_base2 = pd.read_csv("../input/detecting-malwares-with-ftrl-proximal/submission.csv",names=["MachineIdentifier","HasDetections2"], skiprows=[0],header=None)
df_base3 = pd.read_csv("../input/outputs-for-microsoft/submission_ashish_kfold.csv",names=["MachineIdentifier","HasDetections3"], skiprows=[0],header=None)
df_base4 = pd.read_csv("../input/hung-the-nguyen/submission_lgbm_5.csv",names=["MachineIdentifier","HasDetections4"], skiprows=[0],header=None)
df_base5 = pd.read_csv("../input/hung-the-nguyen/submission_lgbm_6.csv",names=["MachineIdentifier","HasDetections5"], skiprows=[0],header=None)
df_base6 = pd.read_csv("../input/hung-the-nguyen/submission_lgbm_7.csv",names=["MachineIdentifier","HasDetections6"], skiprows=[0],header=None)
df_base7 = pd.read_csv("../input/hung-the-nguyen/submission_lgbm_8.csv",names=["MachineIdentifier","HasDetections7"], skiprows=[0],header=None)
df_base8 = pd.read_csv("../input/hung-the-nguyen/submission_lgbm_9.csv",names=["MachineIdentifier","HasDetections8"], skiprows=[0],header=None)
df_base = pd.merge(df_base0,df_base1,how='inner',on='MachineIdentifier')
df_base = pd.merge(df_base,df_base2,how='inner',on='MachineIdentifier')
df_base = pd.merge(df_base,df_base3,how='inner',on='MachineIdentifier')
df_base = pd.merge(df_base,df_base4,how='inner',on='MachineIdentifier')
df_base = pd.merge(df_base,df_base5,how='inner',on='MachineIdentifier')
df_base = pd.merge(df_base,df_base6,how='inner',on='MachineIdentifier')
df_base = pd.merge(df_base,df_base7,how='inner',on='MachineIdentifier')
df_base = pd.merge(df_base,df_base8,how='inner',on='MachineIdentifier')
#CORRELATION MATRIX (Pearson Correlation to measure how similar are 2 solutions)
plt.figure(figsize=(16,12))
sns.heatmap(df_base.iloc[:,1:].corr(),annot=True,fmt=".2f")
# ALTERNATIVE WAY - RMSE MATRIX (RMSE to measure how similar are 2 solutions)
M = np.zeros([df_base.iloc[:,1:].shape[1],df_base.iloc[:,1:].shape[1]])
for i in np.arange(M.shape[1]):
    for j in np.arange(M.shape[1]):
        M[i,j] = sqrt(metrics.mean_squared_error(df_base.iloc[:,i+1], df_base.iloc[:,j+1]))
#SOLUTION = MEAN OF COLUMNS
df_base['HasDetections'] = df_base.iloc[:,1:].mean(axis=1)
df_base[['MachineIdentifier','HasDetections']].to_csv("Bestoutput.csv",index=False)
#ALL PUBLIC SOLUTION RMSE < 0.2269 (WITHOUT REPETITIONS)
df_base0 = pd.read_csv("../input/lightgbm/submission_lgbm.csv",names=["MachineIdentifier","HasDetections0"], skiprows=[0],header=None)
df_base1 = pd.read_csv("../input/malware-predictions-3500-trees/submission0.72968.csv",names=["MachineIdentifier","HasDetections1"], skiprows=[0],header=None)
df_base2 = pd.read_csv("../input/detecting-malwares-with-ftrl-proximal/submission.csv",names=["MachineIdentifier","HasDetections2"], skiprows=[0],header=None)
df_base3 = pd.read_csv("../input/outputs-for-microsoft/submission_ashish_kfold.csv",names=["MachineIdentifier","HasDetections3"], skiprows=[0],header=None)
df_base8 = pd.read_csv("../input/hung-the-nguyen/submission_lgbm_9.csv",names=["MachineIdentifier","HasDetections8"], skiprows=[0],header=None)
df_base = pd.merge(df_base0,df_base1,how='inner',on='MachineIdentifier')
df_base = pd.merge(df_base,df_base2,how='inner',on='MachineIdentifier')
df_base = pd.merge(df_base,df_base3,how='inner',on='MachineIdentifier')
df_base = pd.merge(df_base,df_base8,how='inner',on='MachineIdentifier')
#CORRELATION MATRIX (Pearson Correlation to measure how similar are 2 solutions)
plt.figure(figsize=(16,12))
sns.heatmap(df_base.iloc[:,1:].corr(),annot=True,fmt=".2f")
# ALTERNATIVE WAY - RMSE MATRIX (RMSE to measure how similar are 2 solutions)
M = np.zeros([df_base.iloc[:,1:].shape[1],df_base.iloc[:,1:].shape[1]])
for i in np.arange(M.shape[1]):
    for j in np.arange(M.shape[1]):
        M[i,j] = sqrt(metrics.mean_squared_error(df_base.iloc[:,i+1], df_base.iloc[:,j+1]))
#SOLUTION = MEAN OF COLUMNS
df_base['HasDetections'] = df_base.iloc[:,1:].mean(axis=1)
df_base[['MachineIdentifier','HasDetections']].to_csv("Bestoutput1.csv",index=False)
double_blend = df_base['HasDetections'] * 0.5 + submission['HasDetections'] * 0.5
sample = pd.read_csv('../input/microsoft-malware-prediction/sample_submission.csv')
sample.head()
sample['HasDetections'] = double_blend
sample.to_csv('Bouble_blend.csv', index=False)