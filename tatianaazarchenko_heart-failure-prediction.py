import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm
import seaborn as sns
sns.set()

import os
data_filepath = "../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(data_filepath)
data.head()
x = data[["ejection_fraction", "serum_creatinine"]]
x.describe()
plt.scatter(data["ejection_fraction"], data["serum_creatinine"])
plt.xlabel("Percentage of ejection")
plt.ylabel("Serum Creatinine level")
plt.show()
x1 = data.copy()
x1 = x1.drop(columns=["time", "DEATH_EVENT"])
x1.head()
x_scaled = preprocessing.scale(x1)
x_scaled
x = sm.add_constant(x_scaled)
y = data['DEATH_EVENT']

reg_log = sm.Logit(y,x)
results_log1 = reg_log.fit()
results_log1.summary()
x2 = data.copy()
x2 = x2[["age", "ejection_fraction", "serum_creatinine"]]
x_scaled = preprocessing.scale(x2)
x = sm.add_constant(x_scaled)
reg_log = sm.Logit(y,x)
results_log2 = reg_log.fit()
results_log2.summary()
x3 = data.copy()
x3 = x3[["ejection_fraction", "serum_creatinine"]]
x_scaled = preprocessing.scale(x3)
x = sm.add_constant(x_scaled)
reg_log = sm.Logit(y,x)
results_log3 = reg_log.fit()
results_log3.summary()
results_log3.pred_table()
cm_df = pd.DataFrame(results_log3.pred_table())
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df
cm = np.array(cm_df)
accuracy_train = (cm[0,0]+cm[1,1])/cm.sum()
accuracy_train
results_log2.pred_table()
cm_df = pd.DataFrame(results_log2.pred_table())
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df
cm = np.array(cm_df)
accuracy_train = (cm[0,0]+cm[1,1])/cm.sum()
accuracy_train