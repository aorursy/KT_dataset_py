# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/nab/realKnownCause/realKnownCause/ec2_request_latency_system_failure.csv")
df.head(2)
df.describe()
#changing timestamp to datetime value

df['timestamp']=pd.to_datetime(df['timestamp'])
#plotting values

import plotly.express as px

px.line(df,x='timestamp',y='value')
df['hour']=df['timestamp'].dt.hour
px.box(df,x='hour',y='value')
px.histogram(df['value'])
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from sklearn.metrics import auc

#Source:https://github.com/ngoix/EMMV_benchmarks/blob/master/em.py

def em(t, t_max, volume_support, s_unif, s_X, n_generated):
    EM_t = np.zeros(t.shape[0])
    n_samples = s_X.shape[0]
    s_X_unique = np.unique(s_X)
    EM_t[0] = 1.
    for u in s_X_unique:
        # if (s_unif >= u).sum() > n_generated / 1000:
        EM_t = np.maximum(EM_t, 1. / n_samples * (s_X > u).sum() -
                          t * (s_unif > u).sum() / n_generated
                          * volume_support)
    amax = np.argmax(EM_t <= t_max) + 1
    if amax == 1:
        print("failed to achieve t_max")
        amax = -1
    AUC = auc(t[:amax], EM_t[:amax])
    return AUC, EM_t, amax
df.shape
# parameters of the algorithm:
n_generated = 100000
t_max = 0.9

lim_inf = df['value'].values.min(axis=0)
lim_sup = df['value'].values.max(axis=0)
volume_support = (lim_sup - lim_inf).prod()
t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
unif = np.random.uniform(lim_inf, lim_sup,size=(n_generated, 1))
one_svm=OneClassSVM()
one_svm_result=one_svm.fit_predict(df['value'].values.reshape(-1,1))
one_svm_result_df=pd.DataFrame()
one_svm_result_df['timestamp']=df['timestamp']
one_svm_result_df['value'] = df['value']

#Inliers are labeled 1, while outliers are labeled -1.
one_svm_result_df['anomaly']  = [1 if i==-1 else 0 for i in one_svm_result]
s_X_ocsvm = one_svm.decision_function(df['value'].values.reshape(-1,1)).reshape(1, -1)[0]
s_unif_ocsvm = one_svm.decision_function(unif).reshape(1, -1)[0]
auc_ocsvm, em_ocsvm, amax_ocsvm = em(t, t_max, volume_support,s_unif_ocsvm, s_X_ocsvm, n_generated)
#we will store the EM values for all the models in a list

em_values=[]
model_name=[]
em_values.append(em_ocsvm.mean())
model_name.append("One Clas SVM")
one_svm_result_df['anomaly'].value_counts()
import plotly.graph_objects as go

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=one_svm_result_df['timestamp'], y=one_svm_result_df['value'],
                    mode='lines',
                    name='lines'))

a=one_svm_result_df[one_svm_result_df['anomaly']==1]

fig.add_trace(go.Scatter(x=a.timestamp, y=a.value,
                    mode='markers',
                    name='markers'))

fig.update_layout(title='Anomaly detection using One Class SVM')
fig.show("notebook")
one_svm_result
iso=IsolationForest()
iso_result=iso.fit_predict(df['value'].values.reshape(-1,1))
iso_result_df=pd.DataFrame()
iso_result_df['timestamp']=df['timestamp']
iso_result_df['value'] = df['value']

#Inliers are labeled 1, while outliers are labeled -1.
iso_result_df['anomaly']  = [1 if i==-1 else 0 for i in iso_result]
s_X_iso = iso.decision_function(df['value'].values.reshape(-1,1)).reshape(1, -1)[0]
s_unif_iso = iso.decision_function(unif).reshape(1, -1)[0]
auc_iso, em_iso, amax_iso = em(t, t_max, volume_support,s_unif_iso, s_X_iso, n_generated)
em_values.append(em_iso.mean())
model_name.append("Isolation Forest")
iso_result_df['anomaly'].value_counts()
import plotly.graph_objects as go

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=iso_result_df['timestamp'], y=iso_result_df['value'],
                    mode='lines',
                    name='lines'))

a=iso_result_df[iso_result_df['anomaly']==1]

fig.add_trace(go.Scatter(x=a.timestamp, y=a.value,
                    mode='markers',
                    name='markers'))

fig.update_layout(title='Anomaly detection using Isolation Forest')
fig.show("notebook")
lof=LocalOutlierFactor(novelty=True)
lof.fit(df['value'].values.reshape(-1,1))
lof_result=lof.predict(df['value'].values.reshape(-1,1))
lof_result_df=pd.DataFrame()
lof_result_df['timestamp']=df['timestamp']
lof_result_df['value'] = df['value']

#Inliers are labeled 1, while outliers are labeled -1.
lof_result_df['anomaly']  = [1 if i==-1 else 0 for i in lof_result]

#decision_function is not available when novelty=False. If we make novelty=True, then fit_predict
#is not available

"""
The decision_function method is also defined from the scoring function, 
in such a way that negative values are outliers and non-negative ones are inliers.
"""
s_X_lof = lof.decision_function(df['value'].values.reshape(-1,1))
s_unif_lof = lof.decision_function(unif).reshape(1, -1)
auc_lof, em_lof, amax_lof = em(t, t_max, volume_support,s_unif_lof, s_X_lof, n_generated)
em_values.append(em_lof.mean())
model_name.append("LOF")
lof_result_df['anomaly'].value_counts()
import plotly.graph_objects as go

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=lof_result_df['timestamp'], y=lof_result_df['value'],
                    mode='lines',
                    name='lines'))

a=lof_result_df[lof_result_df['anomaly']==1]

fig.add_trace(go.Scatter(x=a.timestamp, y=a.value,
                    mode='markers',
                    name='markers'))

fig.update_layout(title='Anomaly detection using LOF')
fig.show("notebook")
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(random_state=0)
gm.fit(df['value'].values.reshape(-1,1))

densities = gm.score_samples(df['value'].values.reshape(-1,1))
density_threshold = np.percentile(densities, 1)
gm_result= [-1 if i<density_threshold else 0 for i in densities]
gm_result_df=pd.DataFrame()
gm_result_df['timestamp']=df['timestamp']
gm_result_df['value'] = df['value']

gm_result_df['anomaly']  = [1 if i==-1 else 0 for i in gm_result]
s_X_gm = gm.score_samples(df['value'].values.reshape(-1,1)).reshape(1, -1)[0]
s_unif_gm = gm.score_samples(unif).reshape(1, -1)[0]
auc_gm, em_gm, amax_gm = em(t, t_max, volume_support,s_unif_gm, s_X_gm, n_generated)
gm_result_df['anomaly'].value_counts()
em_values.append(em_gm.mean())
model_name.append("GMM")
import plotly.graph_objects as go

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=gm_result_df['timestamp'], y=gm_result_df['value'],
                    mode='lines',
                    name='lines'))

a=gm_result_df[gm_result_df['anomaly']==1]

fig.add_trace(go.Scatter(x=a.timestamp, y=a.value,
                    mode='markers',
                    name='markers'))

fig.update_layout(title='Anomaly detection using GMM')
fig.show("notebook")
final_result={}

final_result={'Model Name':model_name,'EM Value':em_values}
final_result_df=pd.DataFrame(final_result)
final_result_df