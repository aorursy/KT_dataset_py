# Importing relevant libraries 

import pandas as pd

import numpy as np

import scipy.stats

import pylab as plt

from matplotlib.colors import LogNorm



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc



import xgboost as xgb
# Loading 2017 data

data2017 = pd.read_csv('../input/marathon_results_2017.csv')

#np.unique(data2017['M/F'])
# Number of athletes

NoA = len(data2017['Bib'])

NoFA = len(data2017[data2017['M/F']=='F']['Bib'])

NoMA = len(data2017[data2017['M/F']=='M']['Bib'])

print('Number of Athletes',NoA)

print('Number of Female Athletes',NoFA,'%.3g' % float(1.*NoFA/NoA))

print('Number of Male Athletes',NoMA,'%.3g' % float(1.*NoMA/NoA))

print('Female to Male Ratio', '%.3g' % float(1.*NoFA/NoMA))
# Converting Times String into minutes

def getTimeMinutes(time_str):

    if time_str != '-':

        data_time = [int(item) for item in time_str.split(':')]

        return data_time[0]*60 + data_time[1] + data_time[2]/60. 

    else:

        return np.NaN
# Converting Genders into binary 

def getGender(gender_str):

    if gender_str == 'M':

        return 1

    else:

        return 0
# Print Readable Confusion Matrix

def print_cm(confusion_matrix, labels):

    cw = max([len(x) for x in labels]+[5])

    print(cw)

    

    print('{0:10}  {1}'.format(labels[0], labels[1]))
data2017_WB = data2017[['Bib','Age', 'M/F','5K', '10K', '15K', '20K','Half', '25K', '30K', '35K', '40K','Official Time']]

lGENDER = data2017_WB['M/F'].apply(getGender)

lTime5K = data2017_WB['5K'].apply(getTimeMinutes)

lTime10K = data2017_WB['10K'].apply(getTimeMinutes)

lTime15K = data2017_WB['15K'].apply(getTimeMinutes)

lTime20K = data2017_WB['20K'].apply(getTimeMinutes)

lTimeHalf = data2017_WB['Half'].apply(getTimeMinutes)

lTime25K = data2017_WB['25K'].apply(getTimeMinutes)

lTime30K = data2017_WB['30K'].apply(getTimeMinutes)

lTime35K = data2017_WB['35K'].apply(getTimeMinutes)

lTime40K = data2017_WB['40K'].apply(getTimeMinutes)

lTimeFull = data2017_WB['Official Time'].apply(getTimeMinutes)





data2017_CL = pd.DataFrame({'Bib':data2017_WB['Bib'],'Age':data2017_WB['Age']})

data2017_CL['GenderB'] = lGENDER

data2017_CL['Time5K'] = lTime5K

data2017_CL['Time10K'] = lTime10K

data2017_CL['Time15K'] = lTime15K

data2017_CL['Time20K'] = lTime20K

data2017_CL['TimeHalf'] = lTimeHalf

data2017_CL['Time25K'] = lTime25K

data2017_CL['Time30K'] = lTime30K

data2017_CL['Time35K'] = lTime35K

data2017_CL['Time40K'] = lTime40K

data2017_CL['TimeFull'] = lTimeFull



# Selecting Athletes with valid data (all fields)

print(data2017_CL.shape)

print(data2017_CL.dropna().shape)

data2017_CL = data2017_CL.dropna()
# Number of athletes - Clean data

NoA_CL = len(data2017_CL['Bib'])

NoFA_CL = len(data2017_CL[data2017_CL['GenderB']==0]['Bib'])

NoMA_CL = len(data2017_CL[data2017_CL['GenderB']==1]['Bib'])

print('Number of Athletes - Clean Data',NoA_CL)

print('Number of Female Athletes - Clean Data',NoFA_CL,'%.3g' % float(1.*NoFA_CL/NoA_CL))

print('Number of Male Athletes - Clean Data',NoMA_CL,'%.3g' % float(1.*NoMA_CL/NoA_CL))

print('Female to Male Ratio - Clean Data', '%.3g' % float(1.*NoFA_CL/NoMA_CL))
data2017_CL['Pace5K'] = data2017_CL['Time5K']/5

data2017_CL['Pace10K'] = (data2017_CL['Time10K'] - data2017_CL['Time5K'])/5

data2017_CL['Pace15K'] = (data2017_CL['Time15K'] - data2017_CL['Time10K'])/5

data2017_CL['Pace20K'] = (data2017_CL['Time20K'] - data2017_CL['Time15K'])/5

data2017_CL['PaceHalf'] = (data2017_CL['TimeHalf'] - data2017_CL['Time20K'])/(21.1-20)

data2017_CL['Pace25K'] = (data2017_CL['Time25K'] - data2017_CL['TimeHalf'])/(25 - 21.1)

data2017_CL['Pace30K'] = (data2017_CL['Time30K'] - data2017_CL['Time25K'])/5

data2017_CL['Pace35K'] = (data2017_CL['Time35K'] - data2017_CL['Time30K'])/5

data2017_CL['Pace40K'] = (data2017_CL['Time40K'] - data2017_CL['Time35K'])/5

data2017_CL['PaceFull'] = (data2017_CL['TimeFull'] - data2017_CL['Time40K'])/(42.2 - 40)



data2017_CL['PaceMean'] = data2017_CL[['Pace5K', 'Pace10K', 'Pace15K', 'Pace20K', 'PaceHalf', 'Pace25K',

       'Pace30K', 'Pace35K', 'Pace40K', 'PaceFull']].mean(axis=1)

data2017_CL['PaceStd'] = data2017_CL[['Pace5K', 'Pace10K', 'Pace15K', 'Pace20K', 'PaceHalf', 'Pace25K',

       'Pace30K', 'Pace35K', 'Pace40K', 'PaceFull']].std(axis=1)
data2017_CL.head()

data2017_CL.columns.values
# Computing relevant centrality meausures

mode = scipy.stats.mode(data2017_CL['TimeFull'])[0][0]

median = np.median(data2017_CL['TimeFull'])

mean = np.mean(data2017_CL['TimeFull'])



print('Mode of Distribution',mode)

print('Median of Distribution',median)

print('Mean of Distribution',mean)
plt.hist(data2017_CL['TimeFull'],bins=100)

                   

plt.plot([mode,mode],[0,1000],label='Mode')

plt.plot([mean,mean],[0,1000],label='Mean')

plt.plot([median,median],[0,1000],label='Median')



plt.legend()

plt.show()
# Computing relevant dispersion meausures

std = np.std(data2017_CL['TimeFull'])

q25 = data2017_CL['TimeFull'].quantile(.25)

q75 = data2017_CL['TimeFull'].quantile(.75)



print('Standard Deviation of Distribution',std)

print('Quantile 25 of Distribution',q25)

print('Quantile 75 of Distribution',q75)
countsF, binsF, barsF = plt.hist(data2017_CL[data2017_CL['GenderB']==0]['TimeFull'],label='Female',histtype='step',bins=100,range=(100,500))

countsM, binsM, barsM = plt.hist(data2017_CL[data2017_CL['GenderB']==1]['TimeFull'],label='Male',histtype='step',bins=100,range=(100,500))

plt.legend()

plt.show()
n_countsF, n_binsF, n_barsF = plt.hist(data2017_CL[data2017_CL['GenderB']==0]['TimeFull'],label='Female',histtype='step',bins=100,range=(100,500),normed=True)

n_countsM, n_binsM, n_barsM = plt.hist(data2017_CL[data2017_CL['GenderB']==1]['TimeFull'],label='Male',histtype='step',bins=100,range=(100,500),normed=True)

min_distribution_distance = ((n_countsF-n_countsM)**2)[np.all([150<n_binsF,n_binsF<250],axis=0)[:len(n_countsF)]].min()

time_threshold = n_binsF[np.argmin(((n_countsF-n_countsM)**2-min_distribution_distance)**2)]

plt.plot([time_threshold,time_threshold],[0,0.015])

print('Threshold Time', time_threshold)

plt.show()
plt.hist(data2017_CL['PaceMean'])

plt.show()
plt.hist(data2017_CL['PaceStd']/data2017_CL['PaceMean'],bins=100,histtype='step',normed='True')

plt.hist(data2017_CL[data2017_CL['GenderB']==0]['PaceStd']/data2017_CL[data2017_CL['GenderB']==0]['PaceMean'],bins=100,histtype='step',normed='True')

plt.hist(data2017_CL[data2017_CL['GenderB']==1]['PaceStd']/data2017_CL[data2017_CL['GenderB']==1]['PaceMean'],bins=100,histtype='step',normed='True')



plt.show()
plt.subplot(1, 3, 1)

plt.hist2d(data2017_CL['TimeFull'],data2017_CL['PaceStd']/data2017_CL['PaceMean'],bins=(100,100),normed=True,norm=LogNorm())

plt.colorbar()

plt.subplot(1, 3, 3)

plt.hist2d(data2017_CL['TimeFull'],data2017_CL['PaceStd']/data2017_CL['PaceMean'],bins=(100,100),range=((100,500),(0,0.5)),normed=True,norm=LogNorm())

plt.colorbar()

plt.show()
plt.scatter(data2017_CL[data2017_CL['GenderB']==0]['TimeFull'],data2017_CL[data2017_CL['GenderB']==0]['PaceStd']/data2017_CL[data2017_CL['GenderB']==0]['PaceMean'],alpha=0.5,label='Female')

plt.scatter(data2017_CL[data2017_CL['GenderB']==1]['TimeFull'],data2017_CL[data2017_CL['GenderB']==1]['PaceStd']/data2017_CL[data2017_CL['GenderB']==1]['PaceMean'],alpha=0.5,label='Male')

plt.legend()

plt.show()
plt.subplot(1, 3, 1)

plt.hist2d(data2017_CL[data2017_CL['GenderB']==0]['TimeFull'],data2017_CL[data2017_CL['GenderB']==0]['PaceStd']/data2017_CL[data2017_CL['GenderB']==0]['PaceMean'],bins=(100,100),range=((100,500),(0,0.5)),normed=True,norm=LogNorm())

plt.colorbar()

plt.subplot(1, 3, 3)

plt.hist2d(data2017_CL[data2017_CL['GenderB']==1]['TimeFull'],data2017_CL[data2017_CL['GenderB']==1]['PaceStd']/data2017_CL[data2017_CL['GenderB']==1]['PaceMean'],bins=(100,100),range=((100,500),(0,0.5)),normed=True,norm=LogNorm())

plt.colorbar()

plt.show()
X = data2017_CL[['Age', 'Pace5K', 'Pace10K', 'Pace15K', 'Pace20K', 'PaceHalf', 'Pace25K','Pace30K', 'Pace35K', 'Pace40K', 'PaceFull']]

y = data2017_CL['GenderB']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rfc = RandomForestClassifier(n_estimators=1000)

rfc.fit(X_train,y_train)

y_pred_rf = rfc.predict(X_test)

y_score_rf = rfc.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)

print('Random Forest ROC AUC', auc(fpr_rf, tpr_rf))

print(classification_report(y_test, y_pred_rf, target_names=['Female','Male']))
gbc = GradientBoostingClassifier(n_estimators=1000)

gbc.fit(X_train,y_train)

y_pred_gb = gbc.predict(X_test)

y_score_gb = gbc.predict_proba(X_test)[:,1]

fpr_gb, tpr_gb, _ = roc_curve(y_test, y_score_gb)

print('Gradient Boosting ROC AUC', auc(fpr_gb, tpr_gb))

print(classification_report(y_test, y_pred_gb, target_names=['Female','Male']))
xgbc = xgb.XGBClassifier()

xgbc.fit(X_train, y_train)

y_pred_xgb = xgbc.predict(X_test)

y_score_xgb = xgbc.predict_proba(X_test)[:,1]

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_score_xgb)

print('XGBoost ROC AUC', auc(fpr_xgb, tpr_xgb))

print(classification_report(y_test, y_pred_xgb, target_names=['Female','Male']))
fig, ax = plt.subplots()

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_rf, tpr_rf, label='RF')

plt.plot(fpr_gb, tpr_gb, label='GB')

plt.plot(fpr_xgb, tpr_xgb, label='XGB')

ax.set_aspect('equal')

plt.legend()

plt.show()