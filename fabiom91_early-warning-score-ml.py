import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/NEWS_datafile.csv')
df.head()
df.columns
df.dtypes
df.temp = df.temp.str.replace(',','.')

df['MR-proADM'] = df['MR-proADM'].str.replace(',','.')

df.PCT = df.PCT.str.replace(',','.')

df.temp = df.temp.astype('float')

df['MR-proADM'] = df['MR-proADM'].astype('float')

df.PCT = df.PCT.astype('float')

df = df.round({'temp':1})
df.describe()
df['BPD'] = df['BPD'].fillna(df['BPD'].mean())

df['MR-proADM'] = df['MR-proADM'].fillna(df['MR-proADM'].mean())

df['PCT'] = df['PCT'].fillna(df['PCT'].mean())

df['LOS'] = df['LOS'].fillna(df['LOS'].mean())
# create a new feature for EWS based on the values



EWS_Scores = []

for row in df.itertuples():

    row_ews = []

    

    if row.resp_rate >= 12 and row.resp_rate <= 20:

        row_ews.append(0)

    elif row.resp_rate >= 9 and row.resp_rate <= 11:

        row_ews.append(1)

    elif row.resp_rate >= 21 and row.resp_rate <= 24:

        row_ews.append(2)

    elif row.resp_rate <= 8 or row.resp_rate >= 25:

        row_ews.append(3)

        

    if row.confusion == 1:

        row_ews.append(1)

    elif row.confusion == 0:

        row_ews.append(0)

        

    if row.SpO2 >= 96:

        row_ews.append(0)

    elif row.SpO2 >= 94 and row.SpO2 <= 95:

        row_ews.append(1)

    elif row.SpO2 >= 92 and row.SpO2 <= 93:

        row_ews.append(2)

    elif row.SpO2 <= 91:

        row_ews.append(3)

        

    if row.BPS >= 111 and row.BPS <= 249:

        row_ews.append(0)

    elif row.BPS >= 101 and row.BPS <= 110:

        row_ews.append(1)

    elif row.BPS >= 91 and row.BPS <= 100:

        row_ews.append(2)

    elif row.BPS <= 90 or row.BPS >= 250:

        row_ews.append(3)

        

    if row.HR >= 51 and row.HR <= 90:

        row_ews.append(0)

    elif (row.HR >= 41 and row.HR <= 50) or (row.HR >= 91 and row.HR <= 110):

        row_ews.append(1)

    elif row.HR >= 111 and row.HR <= 130:

        row_ews.append(2)

    elif row.HR <= 40 or row.HR >= 131:

        row_ews.append(3)

        

    if row.temp >= 36.1 and row.temp <= 38.0:

        row_ews.append(0)

    elif (row.temp >= 35.1 and row.temp <= 36) or (row.temp >= 38.1 and row.temp <= 39.0):

        row_ews.append(1)

    elif row.temp >= 39.1:

        row_ews.append(2)

    elif row.temp <= 35:

        row_ews.append(3)

        

    if len(row_ews) == 6:

        EWS_Scores.append(row_ews)

    else:

        print('error:', len(row_ews))

        break
ews_list = []

for scores in EWS_Scores:

    ews = sum(scores)

    ews_list.append(ews)



df['EWS'] = ews_list
df.head()
dummy_hospital = pd.get_dummies(df['hospital'])

dummy_country = pd.get_dummies(df['country'])

dummy_gender = pd.get_dummies(df['gender'])

df['discharge location'] = 'discharge_' + df['discharge location']

dummy_discharge = pd.get_dummies(df['discharge location'])

df = df.drop(['hospital','country','gender','discharge location'], axis=1)

df = pd.concat([df,dummy_hospital,dummy_country,dummy_gender,dummy_discharge],axis=1)

df.head()
# Correlation matrix using code found on https://stanford.edu/~mwaskom/software/seaborn/examples/many_pairwise_correlations.html

sns.set(style="white")

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(40, 40))



# Generate a custom colormap - blue and red

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=1, vmin=-1,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

plt.yticks(rotation = 0)

plt.xticks(rotation = 45)
corr_df = df

riskCorr = pd.DataFrame(corr_df.corr()['EWS'])

riskCorr = riskCorr.sort_values('EWS',ascending=False)

plt.figure(figsize=(5, 25))

sns.heatmap(riskCorr, annot=True, fmt="g")

print('Heatmap of correlation of the EWS on the other features')
X = df.drop(['USA','Clearwater Hospital','discharge_Home','discharge_Other','discharge_Institution','discharge_Rehab','Aarau','Switzerland','death30d','France','Hôpital de la Salpêtrière','EWS'],axis=1)

y = df['EWS']
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestRegressor()
rf_model.fit(X_train,y_train)
predictions = rf_model.predict(X_test)
predictions = np.around(predictions)
df_pred = pd.DataFrame({'real_EWS' : np.array(y_test), 'predicted_EWS' : predictions}).astype('int')

df_pred.head(50)
mae = 'MAE: ' + str(metrics.mean_absolute_error(y_test,predictions))

mse = 'MSE: ' + str(metrics.mean_squared_error(y_test,predictions))

rmse = 'RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y_test,predictions)))

variance = 'Variance Score: ' + str(metrics.explained_variance_score(y_test,predictions))

train_metrics = [mae,mse,rmse,variance]

for i in train_metrics:

    print(i)
%%capture

predictions = cross_val_predict(RandomForestRegressor(),X,y, cv=10)
mae = 'MAE: ' + str(metrics.mean_absolute_error(y,predictions))

mse = 'MSE: ' + str(metrics.mean_squared_error(y,predictions))

rmse = 'RMSE: ' + str(np.sqrt(metrics.mean_squared_error(y,predictions)))

variance = 'Variance Score: ' + str(metrics.explained_variance_score(y,predictions))

train_metrics = [mae,mse,rmse,variance]

for i in train_metrics:

    print(i)