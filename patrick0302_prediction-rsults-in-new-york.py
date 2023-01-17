!pip install seaborn --upgrade
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from matplotlib import dates as md

import seaborn as sns

import plotly.graph_objs as go

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.set_config_file(offline=True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error
df_meta = pd.read_csv('/kaggle/input/building-data-genome-project-v1/meta_open.csv')

df_meta
df_model_prediction = pd.read_pickle('../input/load-prediction-for-bdg1-0/df_model_prediction.pickle.gz')

df_model_prediction['date'] = df_model_prediction['timestamp'].dt.date

df_model_prediction['hour'] = df_model_prediction['timestamp'].dt.hour

df_model_prediction
df_metrics = df_model_prediction[['uid','RSQUARED', 'MAPE']].drop_duplicates().reset_index(drop=True)

df_metrics = df_metrics.merge(df_meta, on='uid')

df_metrics
def percentile(n):

    def percentile_(x):

        return np.percentile(x, n)

    percentile_.__name__ = 'percentile_%s' % n

    return percentile_
sns.displot(df_metrics, x="RSQUARED", kde=True);plt.show()

sns.boxplot(x="primaryspaceusage", y="RSQUARED", data=df_metrics)

sns.swarmplot(x="primaryspaceusage", y="RSQUARED", data=df_metrics, color=".25");plt.show()



sns.displot(df_metrics, x="MAPE", kde=True);plt.show()

sns.boxplot(x="primaryspaceusage", y="MAPE", data=df_metrics)

sns.swarmplot(x="primaryspaceusage", y="MAPE", data=df_metrics, color=".25");plt.show()



pd.DataFrame(df_metrics[['RSQUARED','MAPE']].describe()).rename(columns={'MAPE':'MAPE (%)'}).T.round(2)
for bldgType in df_metrics['primaryspaceusage'].unique():



    list_uid = df_metrics.loc[df_metrics['primaryspaceusage']==bldgType, 'uid'].to_list()



    df_model_prediction_bldgType = df_model_prediction.loc[df_model_prediction['uid'].isin(list_uid)].pivot_table(values=['load_meas','load_pred'],index='timestamp',aggfunc='sum')



    errors = abs(df_model_prediction_bldgType['load_pred'] - df_model_prediction_bldgType['load_meas'])



    RSQUARED = r2_score(df_model_prediction_bldgType.dropna()['load_meas'], df_model_prediction_bldgType.dropna()['load_pred'])



    MAPE = errors/df_model_prediction_bldgType['load_meas']

    MAPE = MAPE.loc[MAPE!=np.inf]

    MAPE = MAPE.loc[MAPE!=-np.inf]

    MAPE = MAPE.dropna().mean()*100



    print("R SQUARED: "+str(round(RSQUARED,3)))

    print("MAPE: "+str(round(MAPE,1))+'%')



    df_model_prediction_bldgType.iplot(title=bldgType)
df_model_prediction_whole = df_model_prediction.pivot_table(values=['load_meas','load_pred'],index='timestamp',aggfunc='sum')



errors = abs(df_model_prediction_whole['load_pred'] - df_model_prediction_whole['load_meas'])



RSQUARED = r2_score(df_model_prediction_whole.dropna()['load_meas'], df_model_prediction_whole.dropna()['load_pred'])



MAPE = errors/df_model_prediction_whole['load_meas']

MAPE = MAPE.loc[MAPE!=np.inf]

MAPE = MAPE.loc[MAPE!=-np.inf]

MAPE = MAPE.dropna().mean()*100



print("R SQUARED: "+str(round(RSQUARED,3)))

print("MAPE: "+str(round(MAPE,1))+'%')



df_model_prediction_whole.iplot(title='All buildings')
df_peak_values = df_model_prediction.pivot_table(index=['uid','date'],values=['load_meas','load_pred'],aggfunc='max')

df_peak_values['peak_value_error'] = df_peak_values['load_pred'] - df_peak_values['load_meas']

df_peak_values['peak_value_error_percentage[%]'] = df_peak_values['peak_value_error'] / df_peak_values['load_meas'] *100

df_peak_values['peak_value_error_percentage_abs[%]'] = df_peak_values['peak_value_error_percentage[%]'].abs()

df_peak_values = df_peak_values.rename(columns={'load_meas':'peak_value_meas','load_pred':'peak_value_pred'})

df_peak_values = df_peak_values.reset_index()

df_peak_values
df_peak_values_metrics = df_peak_values.pivot_table(index='uid',values='peak_value_error_percentage[%]', aggfunc='mean')

df_peak_values_metrics = df_peak_values_metrics.reset_index()

df_peak_values_metrics = df_peak_values_metrics.merge(df_meta, on='uid')

df_peak_values_metrics
sns.displot(df_peak_values_metrics, x="peak_value_error_percentage[%]", kde=True);plt.show()

sns.boxplot(x="primaryspaceusage", y="peak_value_error_percentage[%]", data=df_peak_values_metrics)

sns.swarmplot(x="primaryspaceusage", y="peak_value_error_percentage[%]", data=df_peak_values_metrics, color=".25");plt.show()



pd.DataFrame(df_peak_values_metrics['peak_value_error_percentage[%]'].describe()).T.round(2)
for bldgType in df_metrics['primaryspaceusage'].unique():



    list_uid = df_metrics.loc[df_metrics['primaryspaceusage']==bldgType, 'uid'].to_list()



    df_peak_values_bldgType = df_peak_values.loc[df_peak_values['uid'].isin(list_uid)].pivot_table(values=['peak_value_meas','peak_value_pred'],index='date',aggfunc='sum')



    errors = abs(df_peak_values_bldgType['peak_value_pred'] - df_peak_values_bldgType['peak_value_meas'])



    RSQUARED = r2_score(df_peak_values_bldgType.dropna()['peak_value_meas'], df_peak_values_bldgType.dropna()['peak_value_pred'])



    MAPE = errors/df_peak_values_bldgType['peak_value_meas']

    MAPE = MAPE.loc[MAPE!=np.inf]

    MAPE = MAPE.loc[MAPE!=-np.inf]

    MAPE = MAPE.dropna().mean()*100



    print("R SQUARED: "+str(round(RSQUARED,3)))

    print("MAPE: "+str(round(MAPE,1))+'%')



    df_peak_values_bldgType.reset_index(drop=True).iplot(title=bldgType)
df_peak_values_whole = df_peak_values.pivot_table(values=['peak_value_meas','peak_value_pred'],index='date',aggfunc='sum')



errors = abs(df_peak_values_whole['peak_value_pred'] - df_peak_values_whole['peak_value_meas'])



RSQUARED = r2_score(df_peak_values_whole.dropna()['peak_value_meas'], df_peak_values_whole.dropna()['peak_value_pred'])



MAPE = errors/df_peak_values_whole['peak_value_meas']

MAPE = MAPE.loc[MAPE!=np.inf]

MAPE = MAPE.loc[MAPE!=-np.inf]

MAPE = MAPE.dropna().mean()*100



print("R SQUARED: "+str(round(RSQUARED,3)))

print("MAPE: "+str(round(MAPE,1))+'%')



df_peak_values_whole.reset_index(drop=True).iplot(title='All buildings')
df_peak_hour = df_model_prediction.pivot_table(index=['uid','date'],values=['load_meas','load_pred'],aggfunc='idxmax')

df_peak_hour['peak_hour_meas'] = df_model_prediction.loc[df_peak_hour['load_meas'].values, 'hour'].values

df_peak_hour['peak_hour_pred'] = df_model_prediction.loc[df_peak_hour['load_pred'].values, 'hour'].values

df_peak_hour = df_peak_hour.drop(['load_meas','load_pred'],axis=1)



df_peak_hour['peak_hour_error'] = df_peak_hour['peak_hour_pred'] - df_peak_hour['peak_hour_meas']

df_peak_hour['peak_hour_error_abs'] = df_peak_hour['peak_hour_error'].abs()

df_peak_hour = df_peak_hour.reset_index()



df_peak_hour
df_peak_hour_metrics = df_peak_hour.pivot_table(index='uid',values='peak_hour_error_abs', aggfunc='mean')

df_peak_hour_metrics = df_peak_hour_metrics.reset_index()

df_peak_hour_metrics = df_peak_hour_metrics.merge(df_meta, on='uid')

df_peak_hour_metrics
sns.displot(df_peak_hour_metrics, x="peak_hour_error_abs", kde=True);plt.show()

sns.boxplot(x="primaryspaceusage", y="peak_hour_error_abs", data=df_peak_hour_metrics)

sns.swarmplot(x="primaryspaceusage", y="peak_hour_error_abs", data=df_peak_hour_metrics, color=".25");plt.show()



pd.DataFrame(df_peak_hour_metrics['peak_hour_error_abs'].describe()).T.round(2)
for bldgType in df_metrics['primaryspaceusage'].unique():



    list_uid = df_metrics.loc[df_metrics['primaryspaceusage']==bldgType, 'uid'].to_list()



    df_model_prediction_bldgType = df_model_prediction.loc[df_model_prediction['uid'].isin(list_uid)].groupby('timestamp').sum()

    df_model_prediction_bldgType = df_model_prediction_bldgType.reset_index()

    df_model_prediction_bldgType['date'] = df_model_prediction_bldgType['timestamp'].dt.date

    df_model_prediction_bldgType['hour'] = df_model_prediction_bldgType['timestamp'].dt.hour



    df_peak_hour_bldgType = df_model_prediction_bldgType.pivot_table(index='date',values=['load_meas','load_pred'],aggfunc='idxmax')

    df_peak_hour_bldgType['peak_hour_meas'] = df_model_prediction_bldgType.loc[df_peak_hour_bldgType['load_meas'].values, 'hour'].values

    df_peak_hour_bldgType['peak_hour_pred'] = df_model_prediction_bldgType.loc[df_peak_hour_bldgType['load_pred'].values, 'hour'].values

    df_peak_hour_bldgType = df_peak_hour_bldgType.drop(['load_meas','load_pred'],axis=1)

    

    errors = abs(df_peak_hour_bldgType['peak_hour_pred'] - df_peak_hour_bldgType['peak_hour_meas'])



    RSQUARED = r2_score(df_peak_hour_bldgType.dropna()['peak_hour_meas'], df_peak_hour_bldgType.dropna()['peak_hour_pred'])



    MEAN = errors.mean()



    print("R SQUARED: "+str(round(RSQUARED,3)))

    print("Error(hours): "+str(round(MEAN,1)))



    df_peak_hour_bldgType.reset_index(drop=True).iplot(title=bldgType)
df_model_prediction_whole = df_model_prediction.groupby('timestamp').sum()

df_model_prediction_whole = df_model_prediction_whole.reset_index()

df_model_prediction_whole['date'] = df_model_prediction_whole['timestamp'].dt.date

df_model_prediction_whole['hour'] = df_model_prediction_whole['timestamp'].dt.hour



df_peak_hour_whole = df_model_prediction_whole.pivot_table(index='date',values=['load_meas','load_pred'],aggfunc='idxmax')

df_peak_hour_whole['peak_hour_meas'] = df_model_prediction_whole.loc[df_peak_hour_whole['load_meas'].values, 'hour'].values

df_peak_hour_whole['peak_hour_pred'] = df_model_prediction_whole.loc[df_peak_hour_whole['load_pred'].values, 'hour'].values

df_peak_hour_whole = df_peak_hour_whole.drop(['load_meas','load_pred'],axis=1)



    

errors = abs(df_peak_hour_whole['peak_hour_pred'] - df_peak_hour_whole['peak_hour_meas'])



RSQUARED = r2_score(df_peak_hour_whole.dropna()['peak_hour_meas'], df_peak_hour_whole.dropna()['peak_hour_pred'])



MEAN = errors.mean()



print("R SQUARED: "+str(round(RSQUARED,3)))

print("Error(hours): "+str(round(MEAN,1)))



df_peak_hour_whole.reset_index(drop=True).iplot(title='All buildings')