from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import fbeta_score

import numpy as np

import pandas as pd

import shap

from matplotlib import pyplot as plt

%config InlineBackend.figure_format = 'retina'
employees_data = pd.read_csv('../input/softserve-ds-hackathon-2020/employees.csv', parse_dates=['HiringDate', 'DismissalDate'])

history_data = pd.read_csv('../input/softserve-ds-hackathon-2020/history.csv', parse_dates=['Date'])
df = history_data.merge(employees_data, how='outer', on='EmployeeID')

df['months_to_dissmiss'] = (df['DismissalDate'].sub(df['Date']) / np.timedelta64(1, 'M')).round()

df['target'] = (df['months_to_dissmiss'] <= 3).astype(int)

df['experience_months'] = (df['Date'].sub(df['HiringDate']) / np.timedelta64(1, 'M')).round()



df['ProjectID'] = df['ProjectID'].fillna('other')



# drop those who had experience less than 3 months (cause all guys from 2019-02-01 have already worked 3 months)

df = df[df['experience_months'] >= 4].reset_index(drop=True)



# drop those who worked after dismissal

df.drop(df[df['Date'] >= df['DismissalDate']].index, inplace=True)
le = LabelEncoder()

df['ProjectID'] = le.fit_transform(df['ProjectID'].astype(str))

df['CustomerID'] = le.fit_transform(df['CustomerID'].astype(str))
val_date = '2018-11-01'

train_date = ['2018-05-01', '2018-08-01']



val = df[df['Date'] == val_date]

train = df[(df['Date'] >= train_date[0]) & (df['Date'] <= train_date[1])].reset_index(drop=True)
drop_cols = ['Date','EmployeeID', 'months_to_dissmiss', 'DismissalDate', 'HiringDate']



val = val.drop(columns = drop_cols)

X_val = val.drop('target', axis=1)

y_val = val['target']



train = train.drop(columns = drop_cols)

X_train = train.drop('target', axis=1)

y_train = train['target']
def validate_model(model, X_val, y_val, threshold = 0.4, verbose=1):

    y_pred = model.predict_proba(X_val)[:, 1] > threshold

    score = fbeta_score(y_val, y_pred, 1.7)

    if verbose:

        print("mean target result: ", y_pred.mean())

        print("score: ", score)

    return score, y_pred
def variate_factor(factor, model, data, variation_range, n_samples=100, random_state=1):

    sample_ones = data[data['target'] == 1].sample(n_samples, random_state=random_state)

    sample_zeros = data[data['target'] == 0].sample(n_samples, random_state=random_state)

    sample = pd.concat((sample_zeros, sample_ones))

    

    variation_values = []

    prediction_means = []

    

    for value in variation_range:

        sample_copy = sample.copy()

        sample_copy[factor] += value

        

        _, y_pred = validate_model(model, sample_copy.drop('target', axis=1), sample_copy['target'], verbose=0)

        

        variation_values.append((str(value.round(2)) if value <=0 else '+'+str(value.round(2))))

        prediction_means.append(y_pred[y_pred == 1].sum())

    

    plt.plot(variation_values, prediction_means)

    plt.title('Impact of "'+factor+'" on prediction')

    plt.xlabel('variation of ' + factor)

    plt.ylabel('predicted # of guys who dismiss')

    plt.show()
def plot_misclass_hist(df, feature, bins=35, figsize=(15, 5)):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    fig.suptitle(feature)

    ax1.set_title('target = 0')

    ax2.set_title('target = 1')

    ax1.hist([df.loc[(df['pred_target'] == 0) & (df['true_target'] == 0), feature],

             df.loc[(df['pred_target'] == 1) & (df['true_target'] == 0), feature]],

             stacked=True,

             label=['pred = 0', 'pred = 1'],

             bins=bins)

    ax2.hist([df.loc[(df['pred_target'] == 1) & (df['true_target'] == 1), feature],

             df.loc[(df['pred_target'] == 0) & (df['true_target'] == 1), feature]],

             stacked=True,

             label=['pred = 1', 'pred = 0'],

             bins=bins)

    ax1.legend()

    ax2.legend()

def plot_misclass_count(df, feature, figsize=(15, 5)):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    fig.suptitle(feature)

    

    ax1.set_title('target = 0')

    ax2.set_title('target = 1')

    

    pd.crosstab(df.loc[df['true_target'] == 0, feature], 

                df.loc[df['true_target'] == 0, 'pred_target']).plot.bar(stacked=True, ax=ax1)

    pd.crosstab(df.loc[df['true_target'] == 1, feature], 

                df.loc[df['true_target'] == 1, 'pred_target'])[[1, 0]].plot.bar(stacked=True, ax=ax2)
model = RandomForestClassifier(n_estimators=100, 

                             class_weight='balanced',

                             max_leaf_nodes=64,

                             max_depth=12,

                             max_features=5, 

                             min_samples_split=5, 

                             min_samples_leaf=3,

                             criterion='gini',

                             n_jobs=-1,

                             random_state=1)
model.fit(X_train, y_train)

score, y_pred = validate_model(model, X_val, y_val);
val_with_pred = val.copy().rename(columns={'target':'true_target'})

val_with_pred['pred_target'] = y_pred
import shap

model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values[1], X_train, plot_type='bar')
shap.summary_plot(shap_values[1], X_train)
SAMPLE_SIZE = 1000
fig, ax1 = plt.subplots(1, 1, figsize=(15,5))

shap.dependence_plot('experience_months', 

                     pd.DataFrame(shap_values[1]).sample(SAMPLE_SIZE, random_state=1).values,

                     X_train.sample(SAMPLE_SIZE, random_state=1), 

                     interaction_index="WageGross", 

                     ax=ax1)
variate_factor('experience_months', model, val, np.linspace(-5, 20, 6))
plot_misclass_hist(val_with_pred[val_with_pred['experience_months'] > 30], 'experience_months', bins=50)
fig, ax1 = plt.subplots(1, 1, figsize=(15,5))

shap.dependence_plot('DevCenterID', 

                     pd.DataFrame(shap_values[1]).sample(SAMPLE_SIZE, random_state=1).values,

                     X_train.sample(SAMPLE_SIZE, random_state=1), 

                     interaction_index="SBUID", 

                     ax=ax1,

                     )
plot_misclass_count(val_with_pred, 'DevCenterID')
fig, ax1 = plt.subplots(1, 1, figsize=(15,5))

shap.dependence_plot('WageGross', 

                     pd.DataFrame(shap_values[1]).sample(SAMPLE_SIZE, random_state=1).values,

                     X_train.sample(SAMPLE_SIZE, random_state=1), 

                     interaction_index="PositionLevel", 

                     ax=ax1,

                     xmax=10)
variate_factor('WageGross', model, val, np.linspace(-0.5, 0.5, 11))
plot_misclass_hist(val_with_pred, 'WageGross', bins=50)
fig, ax1 = plt.subplots(1, 1, figsize=(15,5))

shap.dependence_plot('SBUID', 

                     pd.DataFrame(shap_values[1]).sample(SAMPLE_SIZE, random_state=1).values,

                     X_train.sample(SAMPLE_SIZE, random_state=1), 

                     interaction_index="DevCenterID", 

                     ax=ax1,

                     )
plot_misclass_hist(val_with_pred[val_with_pred['SBUID'] < 120], 'SBUID', bins=50)
val['HourMobileReserve'].agg(['mean', 'std'])
print('samples with HourMobileReserve = 0: ', val[val['HourMobileReserve'] == 0].shape[0])

print('samples with HourMobileReserve > 0: ', val[val['HourMobileReserve'] > 0].shape[0])
fig, ax1 = plt.subplots(1, 1, figsize=(15,5))

shap.dependence_plot('HourMobileReserve', 

                     pd.DataFrame(shap_values[1]).sample(SAMPLE_SIZE, random_state=1).values,

                     X_train.sample(SAMPLE_SIZE, random_state=1), 

                     interaction_index="PositionID", 

                     ax=ax1,

                     )
# almost all values are "0", so omit them

plot_misclass_hist(val_with_pred[val_with_pred['HourMobileReserve'] > 0], 'HourMobileReserve', bins=50)
plt.figure(figsize=(15,5))

variate_factor('HourMobileReserve', model, val, np.linspace(0, 40, 21))
fig, ax1 = plt.subplots(1, 1, figsize=(15,5))

shap.dependence_plot('PositionID', 

                     pd.DataFrame(shap_values[1]).sample(SAMPLE_SIZE, random_state=1).values,

                     X_train.sample(SAMPLE_SIZE, random_state=1), 

                     interaction_index="PositionLevel", 

                     ax=ax1,

                     )
plot_misclass_hist(val_with_pred[val_with_pred['PositionID'] > 207], 'PositionID')
fig, ax1 = plt.subplots(1, 1, figsize=(15,5))

shap.dependence_plot('MonthOnSalary', 

                     pd.DataFrame(shap_values[1]).sample(SAMPLE_SIZE, random_state=1).values,

                     X_train.sample(SAMPLE_SIZE, random_state=1), 

                     interaction_index="PositionID", 

                     ax=ax1,

                     )
plt.figure(figsize=(15,5))

variate_factor('MonthOnSalary', model, val, np.linspace(-5, 10, 16))
fig, ax1 = plt.subplots(1, 1, figsize=(15,5))

shap.dependence_plot('BonusOneTime', 

                     pd.DataFrame(shap_values[1]).sample(SAMPLE_SIZE, random_state=1).values,

                     X_train.sample(SAMPLE_SIZE, random_state=1), 

                     interaction_index="PositionID", 

                     ax=ax1,

                     xmax=500)
plt.figure(figsize=(15,5))

variate_factor('BonusOneTime', model, val, np.linspace(-10, 200, 22))
sample = {

    'DevCenterID': 1,

    'SBUID': 255,

    'PositionID': 200,

    'PositionLevel': 3,

    'LanguageLevelID': 12,

    'IsTrainee': 0,

    'CustomerID': 110,

    'ProjectID': 900,

    'IsInternalProject': 0,

    'Utilization': 1, 

    'HourVacation': 0,

    'HourMobileReserve': 0,

    'HourLockedReserve': 0,

    'OnSite': 0,

    'CompetenceGroupID': 17,

    'FunctionalOfficeID': 1,

    'PaymentTypeId': 18,

    'BonusOneTime': 0,

    'APM': 26,

    'WageGross': 1,

    'MonthOnPosition': 9,

    'MonthOnSalary': 6,

    'experience_months': 33

}

sample_df = pd.DataFrame([sample])

sample_df.T
def variate_single_sample_factor(factor, model, sample, variation_range, threshold=0.4):

    variation_values = []

    prediction_probas = []

    

    for value in variation_range:

        sample_copy = sample.copy()

        sample_copy[factor] += value

        

        prediction_proba = model.predict_proba(sample_copy)[:, 1][0]

        

        variation_values.append(str(sample_copy[factor].round(2)[0]))

        prediction_probas.append(prediction_proba)

        

    plt.figure(figsize=(15,5))

    plt.axhline(y=threshold, color='r', linestyle='-')

    plt.plot(variation_values, prediction_probas)

    plt.title('Impact of "'+factor+'" on single-sample prediction')

    plt.xlabel('variation of ' + factor)

    plt.ylabel('prediction proba')

    plt.show()
variate_single_sample_factor('experience_months', model, sample_df, np.linspace(-30, 20, 26))
variate_single_sample_factor('DevCenterID', model, sample_df, np.linspace(0, 20, 21))
variate_single_sample_factor('WageGross', model, sample_df, np.linspace(-1, 2, 31))
variate_single_sample_factor('SBUID', model, sample_df, np.linspace(-250, 250, 26))
variate_single_sample_factor('HourMobileReserve', model, sample_df, np.linspace(0, 50, 26))
variate_single_sample_factor('PositionID', model, sample_df, np.linspace(-100, 100, 21))
variate_single_sample_factor('ProjectID', model, sample_df, np.linspace(-600, 1200, 19))
variate_single_sample_factor('CustomerID', model, sample_df, np.linspace(-100, 100, 21))
variate_single_sample_factor('MonthOnSalary', model, sample_df, np.linspace(-6, 10, 17))
variate_single_sample_factor('BonusOneTime', model, sample_df, np.linspace(0, 200, 21))
variate_single_sample_factor('LanguageLevelID', model, sample_df, np.linspace(-9, 9, 19))
variate_single_sample_factor('APM', model, sample_df, np.linspace(-100, 100, 21))
variate_single_sample_factor('OnSite', model, sample_df, np.linspace(0, 1, 2))
variate_single_sample_factor('Utilization', model, sample_df, np.linspace(-1, 4, 6))
variate_single_sample_factor('HourVacation', model, sample_df, np.linspace(0, 200, 21))
variate_single_sample_factor('HourLockedReserve', model, sample_df, np.linspace(0, 50, 26))
variate_single_sample_factor('PositionLevel', model, sample_df, np.linspace(-3, 7, 10))