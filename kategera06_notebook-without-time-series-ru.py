import pandas as pd
import seaborn as sns
from sklearn import preprocessing, ensemble, model_selection

%pylab inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
history = pd.read_csv('../input/softserve-ds-hackathon-2020/history.csv')
history.Date = pd.to_datetime(history.Date)
history.head()
employees = pd.read_csv('../input/softserve-ds-hackathon-2020/employees.csv')
employees.DismissalDate = pd.to_datetime(employees.DismissalDate)
employees.HiringDate = pd.to_datetime(employees.HiringDate)
employees.head()
sample = employees.EmployeeID[0]
employees[employees.EmployeeID == sample]
history[history.EmployeeID == sample]
submission = pd.read_csv('../input/softserve-ds-hackathon-2020/submission.csv')
submission[submission.EmployeeID == sample]
history.Date.describe()
history.dtypes
employees.shape[0]
hired_emplID = employees[employees.DismissalDate.isna()].EmployeeID
hired_emplID.shape[0]
dismissed_emplID = employees[employees.DismissalDate.notna()].EmployeeID
dismissed_emplID.shape[0]
predict_emplID = submission.EmployeeID
submission.shape[0]
hired_empl_history = history[history.EmployeeID.isin(hired_emplID.values)].set_index(['EmployeeID', 'Date'])
hired_empl_history['IsHired'] = True
hired_empl_history
dismissed_empl_history = history[history.EmployeeID.isin(dismissed_emplID.values)].set_index(['EmployeeID', 'Date'])
dismissed_empl_history['IsHired'] = False
dismissed_empl_history.nunique()
countplot_columns = ['PositionLevel', 
                     'IsTrainee', 
                     'LanguageLevelID',
                     'IsInternalProject', 
                     'OnSite', 
                     'PaymentTypeId',
                     'FunctionalOfficeID',
                    ]
fig, ax = plt.subplots(3, 3, figsize=(20, 20))
for idx, ax in enumerate(ax.flat):
    sns.countplot(x=countplot_columns[idx], data=dismissed_empl_history, ax=ax) #row=0, col=0
    if idx == 6:
        break
fig.show()
hired_data = pd.concat([dismissed_empl_history, hired_empl_history])
sns.countplot(x='IsTrainee', data=dismissed_empl_history)
paiplot_columns = ['IsHired', 'Utilization', 'WageGross', 'MonthOnPosition', 'MonthOnSalary']
paiplot_data = pd.concat([dismissed_empl_history[paiplot_columns].groupby('EmployeeID').max(),
                          hired_empl_history[paiplot_columns].groupby('EmployeeID').max()])

plt.figure(figsize=(10,8))
sns.pairplot(paiplot_data, hue = 'IsHired', hue_order=[True, False], palette = 'muted', diag_kind='kde', vars=paiplot_columns[1:])
hist_dates = history.Date.unique()
hist_dates
agg_funcs = {}
              
for col in concatenate([history.columns[2:8], history.columns[10:]]):
    agg_funcs[col] = 'mean'
    if col in ['MonthOnPosition', 'MonthOnSalary', 'IsTrainee','OnSite', 'IsInternalProject', 'LanguageLevelID']:
        agg_funcs[col] = 'max'
    if col in ['BonusOneTime', 'HourVacation', 'PositionLevel']:
        agg_funcs[col] = 'sum'
agg_funcs
data_batches = []
employeesID = employees.EmployeeID
# employees.set_index('EmployeeID')
for date in hist_dates[2::3]:
    period_employees = history[history.Date == date].EmployeeID
    period_data = history[history.EmployeeID.isin(employeesID) & (history.Date <= date)]
    period_data = period_data.drop(columns=['CustomerID', 'ProjectID', 'Date']).groupby('EmployeeID').agg(agg_funcs)
    period_data['target'] = 0
    
    for employee in period_employees:
        dismiss_date = employees[employees.EmployeeID == employee].iloc[0].DismissalDate
        
        if not pd.isnull(dismiss_date):
            
            if (dismiss_date - date) < np.timedelta64(3, 'M'):
                period_data.at[employee, 'target'] = 1
    data_batches.append(period_data)

len(data_batches)
data_batches[0].columns
from sklearn.utils import resample

train_data_x = []
train_data_y = []
for batch in data_batches[-1:]:
    batch_minority = batch[batch.target == 1]
    batch_majority = batch[batch.target == 0]
    batch_minority_upsampled = resample(batch_minority, replace=True, n_samples=batch.shape[0], random_state=123)
    batch_balanced = pd.concat([batch_majority, batch_minority_upsampled])
    train_data_x.append(batch_balanced.drop('target', axis=1))
    train_data_y.append(batch_balanced.target)
    
train_data_x = concatenate(train_data_x)
train_data_y = concatenate(train_data_y)
train_data_x.shape, train_data_y.sum()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_data_x)
train_data_x = scaler.transform(train_data_x)
train_data_x
np.savetxt('train_data_x.csv', train_data_x, delimiter=',')
np.savetxt('train_data_y.csv', train_data_y, delimiter=',')
employeesID = submission.EmployeeID
date = hist_dates[-1]
submit_data_x = history[history.EmployeeID.isin(employeesID)]
submit_data_x = submit_data_x.drop(columns=['CustomerID', 'ProjectID', 'Date']).groupby('EmployeeID').agg(agg_funcs)
submit_data_x = np.array(submit_data_x)
submit_data_x
scaler.fit(submit_data_x)
submit_data_x = scaler.transform(submit_data_x)
submit_data_x
np.savetxt('submit_data_x.csv', submit_data_x, delimiter=',')
train_data_x
submit_data_x
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 0)
lr.fit(train_data_x, train_data_y)
submit_data_y = lr.predict(submit_data_x)
len(submit_data_x), submit_data_y.sum()
submit_df = pd.read_csv('../input/softserve-ds-hackathon-2020/submission.csv')
test_id = submit_df.EmployeeID
submission = pd.DataFrame({'EmployeeID': test_id, 'target':submit_data_y})
submission.target = submission.target.astype(int)
submission.to_csv('my_submission.csv', index=False)