import os

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
data_folder = '/kaggle/input/weather-dataset-rattle-package'

data_file = 'weatherAUS.csv'



data_path = os.path.join(data_folder, data_file)



df = pd.read_csv(data_path)
df
df.shape
df.groupby('RainTomorrow')['RISK_MM'].describe()
df.dtypes
num_vars = df.columns[df.dtypes == 'float'] ## We get the numeric vars

str_vars = df.columns[df.dtypes == 'object'] ## We get the string vars

df.describe()
vars_to_show = ['Pressure9am', 'Humidity3pm']



plot, ax = plt.subplots(ncols=2, nrows=len(vars_to_show), figsize=(20, 15))



for n, v in enumerate(vars_to_show):

    

    df[v].plot(ax = ax[n, 0], kind='hist', bins=50, title=v)

    df.boxplot(v, ax = ax[n, 1])
plot, ax = plt.subplots(ncols=2, nrows=len(num_vars), figsize=(20, 80))



for n, v in enumerate(num_vars):

    

    df[v].plot(ax = ax[n, 0], kind='hist', bins=50, title=v)

    df.boxplot(v, ax = ax[n, 1])
df_num = df[num_vars]



df_is_outlier = ~df_num[np.abs(df_num - df_num.mean()) > 3 * df_num.std()].isna()

(df_is_outlier).sum()
row_has_outlier = df_is_outlier.sum(axis=1) > 0

df_is_outlier.sum(axis=1)[df_is_outlier.sum(axis=1) > 0].count()
df_check_outlier = pd.DataFrame({'Location': df['Location'], 'RainTomorrow': df['RainTomorrow'],'is_out': df_is_outlier.sum(axis=1) > 0, 'total': df_is_outlier.sum(axis=1) > -1})

df_check_prop = df_check_outlier.groupby(['Location', 'RainTomorrow']).sum().sort_values('is_out')

(df_check_prop['is_out'] / df_check_prop['total']).reset_index().sort_values(0)
sns.heatmap(df[num_vars].corr())
for v in str_vars:

    print('Different values of', v, '\n ')

    print(df[v].value_counts())

    print('\n \n \n')
df['Date'] = pd.to_datetime(df['Date'])
df['day_num'] = df['Date'].apply(lambda x: pd.Period(x, freq='D').dayofyear) # We get the number of the date in the year

df['day_num_angle'] = df['day_num'] / 365 * 2 * np.pi # We get the day inside year as an angle 'angle'

df['day_num_sin'] = np.sin(df['day_num_angle']) # We get the sine and consine of the 'angle'

df['day_num_cos'] = np.cos(df['day_num_angle']) 
df['Year'] = df['Date'].dt.year
wind_dir_vars = ['WindGustDir', 'WindDir9am', 'WindDir3pm']



dir_list = ['E', 'ENE', 'NE', 'NNE', 'N', 'NNW', 'NW', 'WNW', 'W', 'WSW', 'SW', 'SSW', 'S', 'SSE', 'SE', 'ESE']

ang_rad_list = [i * np.pi / 8 for i in range(16)]



wind_dir_map = dict(zip(dir_list, ang_rad_list))



for v in wind_dir_vars:

    v_name_sin = v + '_sin'

    v_name_cos = v + '_cos'

    df[v_name_sin] = np.sin(df[v].map(wind_dir_map))

    df[v_name_cos] = np.cos(df[v].map(wind_dir_map))
df['RainToday_num'] = df['RainToday'].map({'Yes': 1, 'No': 0})

df['RainTomorrow_num'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
feat_cols = ['Year', 'day_num_sin', 'day_num_cos', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',

       'Sunshine', 'WindGustDir_sin', 'WindGustDir_cos', 'WindGustSpeed', 'WindDir9am_sin', 'WindDir9am_cos', 'WindDir3pm_sin', 'WindDir3pm_cos',

       'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',

       'Temp3pm', 'RainToday_num']



label_col = 'RainTomorrow_num'
feat_no_location = [c for c in feat_cols if c != 'Location']



for v in feat_no_location:

    plt.figure()

    df.groupby('RainTomorrow')[v].plot(kind='density', legend=True, title=v)
locs = df['Location'].unique()



for l in locs:

    plt.figure()

    df[df['Location'] == l]['RainTomorrow_num'].plot(kind='hist', legend=True, title=l)
plt.plot(df.groupby('Location')['RainTomorrow_num'].sum() / df.groupby('Location')['RainTomorrow_num'].count() * 100)
feat_no_location = [c for c in feat_cols if c != 'Location']

locs = df['Location'].unique()





for v in feat_no_location:

    print('Variable:', v)



    min_val = df[v].min()

    max_val = df[v].max()

    for l in locs:  

        plt.figure()

        try:

            df[df['Location'] == l].groupby('RainTomorrow')[v].plot(kind='density', legend=True, title=v + ' ' + l, xlim=(min_val, max_val))

            plt.show()

        except ValueError:

            print('There was no null-value for Location', l, 'and variable', v)

            plt.close()
df.groupby('Location').aggregate(lambda x: (~x.isna()).sum())
stack_table = df.groupby('Location').aggregate(lambda x: (~x.isna()).sum()).stack()

stack_table = stack_table[stack_table == 0]



stack_table.reset_index().groupby('Location')['level_1'].agg([list, 'count']).reset_index()
st_reset = stack_table.reset_index()

st_reset.columns = ['Location', 'var', 'val']

st_piv = st_reset.pivot(index='Location', columns='var', values='val')

st_piv.iloc[:, :] = np.where(np.isnan(st_piv), 1 , 0) 



plt.figure(figsize=(10, 10))

sns.heatmap(st_piv, linewidths=1)
print('Number of rows:', df.shape[0])

print('Number of rows with some null value:', df.isna().sum(axis=1)[df.isna().sum(axis=1) > 0].shape[0])
df.isna().sum(axis=0)[df.isna().sum(axis=0) > 0]
num_vars_2 = df.columns[df.dtypes == 'float']

num_vars_2 = [c for c in num_vars_2 if not c in ['Date', 'Quarter']]





df['Quarter'] = df['Date'].dt.quarter



df_fillna_1 = df.groupby(['Year', 'Quarter', 'Location']).agg('mean').reset_index()

df_fillna_2 = df.groupby(['Quarter', 'Location']).agg('mean').reset_index()

df_fillna_3 = df.groupby(['Quarter', 'Year']).agg('mean').reset_index()







df_fillna_1[num_vars_2] = np.where(df_fillna_1[num_vars_2].isna(), \

                          df_fillna_1[['Quarter', 'Location']].apply(lambda x: df_fillna_2.set_index(['Quarter', 'Location']).loc[x[0], x[1]][num_vars_2], axis=1), \

                          df_fillna_1[num_vars_2])



df_fillna_1[num_vars_2] = np.where(df_fillna_1[num_vars_2].isna(), \

                          df_fillna_1[['Quarter', 'Year']].apply(lambda x: df_fillna_3.set_index(['Quarter', 'Year']).loc[x[0], x[1]][num_vars_2], axis=1), \

                          df_fillna_1[num_vars_2])



df[num_vars_2] = np.where(df[num_vars_2].isna(),

                          df[['Year', 'Quarter', 'Location']].apply(lambda x: df_fillna_1.set_index(['Year', 'Quarter', 'Location']).loc[x[0], x[1], x[2]][num_vars_2], axis=1), \

                          df[num_vars_2])
feat_cols = ['Year', 'day_num_sin', 'day_num_cos', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',

       'Sunshine', 'WindGustDir_sin', 'WindGustDir_cos', 'WindGustSpeed', 'WindDir9am_sin', 'WindDir9am_cos', 'WindDir3pm_sin', 'WindDir3pm_cos',

       'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',

       'Temp3pm', 'RainToday_num']



label_col = 'RainTomorrow_num'

feats_exclude = ['Year', 'day_num_sin', 'day_num_cos', 'Location']

feat_cols_model = [f for f in feat_cols if not f in feats_exclude]
from sklearn.feature_selection import SelectKBest, chi2



#We add this offset since chi-sq requires positive values

offset = 1e6



X = df[feat_cols_model] + offset

y = df[label_col]

scores = chi2(X, y)[0]



sorted_vars = [var for _, var in sorted(zip(-scores, feat_cols_model))]



m_scores = -scores

m_scores.sort()



df_scores = pd.DataFrame({'var': sorted_vars, 'score': -m_scores})

df_scores
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df[feat_cols_model], df[label_col], test_size=0.3, random_state=42)
from xgboost import XGBClassifier



model = XGBClassifier()



# fit the model with the training data

model.fit(X_train, y_train)

 

# predict the target on the train dataset

predict_test = model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix



accuracy_train = accuracy_score(y_test, predict_test)

conf_mat = confusion_matrix(y_test, predict_test, labels=[1, 0])



print('Accuracy of the model: ', accuracy_train)

print('Confusion matrix:\n', conf_mat)



tpr = conf_mat[0, 0] / conf_mat[0, :].sum()

tnr = conf_mat[1, 1] / conf_mat[1, :].sum()



print('True positive rate:', tpr)

print('True negative rate:', tnr)
df_metrics = pd.DataFrame({})



for i, _ in enumerate(sorted_vars):

    

    n_feats = i + 1



    important_feats = sorted_vars[:n_feats]

    

    model = XGBClassifier()



    # fit the model with the training data

    model.fit(X_train[important_feats], y_train)

 

    # predict the target on the train dataset

    predict_test = model.predict(X_test[important_feats])



    accuracy_test = accuracy_score(y_test, predict_test)

    conf_mat = confusion_matrix(y_test, predict_test, labels=[1, 0])



    tpr = conf_mat[0, 0] / conf_mat[0, :].sum()

    tnr = conf_mat[1, 1] / conf_mat[1, :].sum()

    

    df_aux = pd.DataFrame({'n_feats': [n_feats], 'accuracy': [accuracy_test],

                           'true_pos_rate': [tpr], 'true_neg_rate': [tnr]})



    df_metrics = pd.concat([df_metrics, df_aux], axis=0)

    

df_metrics
line0, = plt.plot(df_metrics['n_feats'], df_metrics['accuracy'], label='Accuracy')

line1, = plt.plot(df_metrics['n_feats'], df_metrics['true_pos_rate'], label='True Positive Rate')

line2, = plt.plot(df_metrics['n_feats'], df_metrics['true_neg_rate'], label='True Negative Rate')

legend = plt.legend(handles=[line0, line1, line2], loc='upper right')

ax = plt.gca().add_artist(legend)

plt.ylim(0, 1)



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve





model = LogisticRegression()



# fit the model with the training data

model.fit(X_train, y_train)

 

# predict the target on the train dataset

predict_test_proba = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test,predict_test_proba[:,1])



plt.plot(fpr, tpr)

plt.ylabel('TPR')

plt.xlabel('1-TNR')

from sklearn.preprocessing import binarize



which_th = ((1 - fpr) * tpr).argmax()



print('Threshold:', thresholds[which_th])

print('True Positive Rate:', tpr[which_th])

print('True Negative Rate:', 1 - fpr[which_th])





y_pred_class = binarize(predict_test_proba,0.25)[:,1]

accuracy = accuracy_score(y_test,y_pred_class)



print('Accuracy:', accuracy)
df_metrics = pd.DataFrame({})



for i, _ in enumerate(sorted_vars):

    

    n_feats = i + 1



    important_feats = sorted_vars[:n_feats]

    

    model = LogisticRegression()



    # fit the model with the training data

    model.fit(X_train[important_feats], y_train)

 

    # predict the target on the train dataset

    predict_test_proba = model.predict_proba(X_test[important_feats])

    predict_test = y_pred_class = binarize(predict_test_proba,0.25)[:,1]

    

    accuracy_test = accuracy_score(y_test,y_pred_class)

    

    fpr, tpr, thresholds = roc_curve(y_test, predict_test_proba[:,1])



    which_th = ((1 - fpr) * tpr).argmax()

    

    threshold = thresholds[which_th]

    tr_pos_rate = tpr[which_th]

    tr_neg_rate = 1 - fpr[which_th]

    

    df_aux = pd.DataFrame({'n_feats': [n_feats], 'accuracy': [accuracy_test],

                           'true_pos_rate': [tr_pos_rate], 'true_neg_rate': [tr_neg_rate],

                           'threshold': [threshold]})



    df_metrics = pd.concat([df_metrics, df_aux], axis=0)

    

df_metrics
df['RainTomorrow'].value_counts() / df.shape[0]
from math import ceil



model = XGBClassifier()

model.fit(X_train, y_train)



test_df = df.iloc[X_test.index]



max_RISK = ceil(test_df['RISK_MM'].max())



df_tpr = pd.DataFrame()



for i in range(max_RISK):

        

    test_df_filt = test_df[(test_df['RISK_MM'] >= i) & (test_df['RainTomorrow_num'] == 1)]

    

    test_df_labels = test_df_filt[label_col]

    

    predictions = model.predict(test_df_filt[feat_cols_model])



    good_preds = (test_df_labels == predictions).sum()

    total_preds = predictions.shape[0]



    df_aux = pd.DataFrame({'min_RISK_MM': [i], 'true_positive_rate': [good_preds / total_preds]})



    df_tpr = pd.concat([df_tpr, df_aux], axis=0)

    

df_tpr.reset_index(inplace=True, drop=True)
plt.plot(df_tpr['min_RISK_MM'], df_tpr['true_positive_rate'])

plt.xlabel('minimal RISK_MM')

plt.ylabel('True Positive Rate')

plt.ylim(0.5, 1.1)