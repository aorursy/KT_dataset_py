import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
hr_data = pd.read_csv('../input/human-resource/HR_comma_sep.csv')

hr_data.head()
# we should probably rename this to something more intuitive like 'department'

hr_data['sales'].unique()
hr_data['department'] = hr_data['sales']

hr_data.drop('sales', axis=1, inplace=True)
hr_data.describe()
# Let's plot an understanding of the categorical features



sns.set_style('darkgrid')



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,7))

fig.tight_layout()

sns.countplot(x='salary', hue='left', data=hr_data, ax=ax1, palette='bright')

ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=12);

sns.countplot(x='department', hue='left', data=hr_data, ax=ax2, palette='bright')

ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=12, rotation=45);

ax2.set_ylabel(' ');
hr_num_data = hr_data[['satisfaction_level', 'last_evaluation', 'number_project',

                       'average_montly_hours', 'time_spend_company', 'Work_accident',

                       'left', 'promotion_last_5years']]
# Let's get an understanding of the numerical features for employees who left the company



left_nums = hr_num_data[hr_num_data.left==1]



left_nums.describe()
# source: https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas



def CorrelationMatrix(df):

    f = plt.figure(figsize=(8, 10))

    plt.matshow(df.corr(), fignum=f.number)

    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)

    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)

    cb = plt.colorbar()

    cb.ax.tick_params(labelsize=14)
CorrelationMatrix(hr_num_data)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
X_categorical = hr_data[['department', 'salary']]
# Encode values

X_categorical = pd.concat([pd.get_dummies(X_categorical['department']), 

                           pd.get_dummies(X_categorical['salary'])], axis=1)

X_categorical.head()
# Scale values

y = hr_num_data['left']



hr_data_to_scale = hr_num_data.drop(['left', 'Work_accident'], axis=1)



scaler = StandardScaler()



X_numerical = pd.DataFrame(scaler.fit_transform(hr_data_to_scale), columns=hr_data_to_scale.columns)
# Work_accident is binary and needn't be scaled



X = pd.concat([X_categorical, X_numerical, hr_num_data['Work_accident']], axis=1)

print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1867)
# Baseline models

logreg = LogisticRegression()



logreg.fit(X_train, y_train)



lr_preds = logreg.predict(X_test)



lr_results = pd.DataFrame(precision_recall_fscore_support(y_test, lr_preds)).T

lr_results.rename(index={0:'LR_0', 1:'LR_1'},

                  columns={0:'Precision', 1:'Recall', 

                                 2:'F-Score', 3:'Support'}, inplace=True)





randfor = RandomForestClassifier()



randfor.fit(X_train, y_train)



rf_preds = randfor.predict(X_test)



rf_results = pd.DataFrame(precision_recall_fscore_support(y_test, rf_preds)).T

rf_results.rename(index={0:'RF_0', 1:'RF_1'},

                  columns={0:'Precision', 1:'Recall', 

                                 2:'F-Score', 3:'Support'}, inplace=True)

baseline_results = pd.concat([lr_results, rf_results], axis=0)

baseline_results
l_prec, l_rec, _ = precision_recall_curve(y_test, lr_preds)

r_prec, r_rec, _ = precision_recall_curve(y_test, rf_preds)



plt.figure(figsize=(10, 5))

plt.plot(l_prec, l_rec, label='LogisticRegression')

plt.plot(r_prec, r_rec, label='RandomForest')

plt.xlabel('Recall')

plt.ylabel('Precision')



plt.legend();
feat_names = X.columns

importances = randfor.feature_importances_

indices = np.argsort(importances)



plt.figure(figsize=(10,5))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [feat_names[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()