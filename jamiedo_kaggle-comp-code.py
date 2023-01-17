import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from google.colab import files

uploaded = files.upload()
train = pd.read_csv('bank-train.csv')
test = pd.read_csv('bank-test.csv')
train = train.drop(['default'], axis=1)
train = train.drop(['id'], axis=1)
test = test.drop(['default'], axis=1)
test = test.drop(['id'], axis=1)

train.head()
categorical_columns = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

fig, ax = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(20, 20))

counter = 0
for column in categorical_columns:
    trace_x = counter // 3
    trace_y = counter % 3

    counts = train[column].value_counts()
    
    x_pos = np.arange(0, len(counts))
    
    ax[trace_x, trace_y].bar(x_pos, counts.values, tick_label = counts.index)
    ax[trace_x, trace_y].set_title(column)
    
    for tick in ax[trace_x, trace_y].get_xticklabels():
        tick.set_rotation(90)
    
    counter += 1

plt.show()
numerical_columns = ['age', 'duration', 'campaign', 'pdays', 'previous']

fig, ax = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(20, 20))

counter = 0
for column in numerical_columns:
    
    trace_x = counter // 3
    trace_y = counter % 3
    
    ax[trace_x, trace_y].hist(train[column])
    
    ax[trace_x, trace_y].set_title(column)
    
    counter += 1

plt.show()
train = train.replace({'no': 0, 'yes': 1})
test = test.replace({'no': 0, 'yes': 1})
train.head()
train = pd.get_dummies(train)
test = pd.get_dummies(test)
train.head()
upper_lim = train['previous'].quantile(.95)
lower_lim = train['previous'].quantile(.05)
train.loc[(train['previous'] > upper_lim),'previous'] = upper_lim
train.loc[(train['previous'] < lower_lim),'previous'] = lower_lim

upper_lim2 = train['campaign'].quantile(.95)
lower_lim2 = train['campaign'].quantile(.05)
train.loc[(train['campaign'] > upper_lim2),'campaign'] = upper_lim2
train.loc[(train['campaign'] < lower_lim2),'campaign'] = lower_lim2
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

train = train.rename(columns={"emp.var.rate": "emp_var_rate", "cons.price.idx": "cons_price_idx",
                      "cons.conf.idx":"cons_conf_idx", "nr.employed":"nr_employed", "job_admin.":"job_admin",
                      "job_blue-collar":"job_blue_collar","job_self-employed":"job_self_employed",
                      "education_basic.4y":"education_basic_4y", "education_basic.6y": "education_basic_6y",
                      "education_basic.9y":"education_basic_9y", "education_high.school": "education_high_school",
                      "education_professional.course":"education_professional_course","education_university.degree":"education_university_degree"})
test = test.rename(columns={"emp.var.rate": "emp_var_rate", "cons.price.idx": "cons_price_idx",
                      "cons.conf.idx":"cons_conf_idx", "nr.employed":"nr_employed", "job_admin.":"job_admin",
                      "job_blue-collar":"job_blue_collar","job_self-employed":"job_self_employed",
                      "education_basic.4y":"education_basic_4y", "education_basic.6y": "education_basic_6y",
                      "education_basic.9y":"education_basic_9y", "education_high.school": "education_high_school",
                      "education_professional.course":"education_professional_course","education_university.degree":"education_university_degree"})
train.head()

features = "+".join(train.loc[:, train.columns != 'y'])

y_vif, X_vif = dmatrices('y ~' + features, train, return_type='dataframe')

vif = pd.DataFrame()
vif["Features"] = X_vif.columns
vif["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

vif.round(1)
train = train.drop(['pdays','job_unknown','marital_unknown','education_unknown','housing_unknown','loan_unknown'], axis=1)
test = test.drop(['pdays','job_unknown','marital_unknown','education_unknown','housing_unknown','loan_unknown'], axis=1)
train.head()
X = train.drop(["y"], 1)
y = train["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=0)
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
X2_train = X
y2_train = y

lr2 = LogisticRegression()
lr2.fit(X2_train,y2_train)

y_pred_test = pd.DataFrame(lr2.predict(test))
from sklearn.ensemble import RandomForestClassifier

# model
forest = RandomForestClassifier(random_state=2)

# train
forest.fit(X_train, y_train)

# predict
forest_predictions = pd.Series(forest.predict(X_test))
forest2 = RandomForestClassifier(random_state=2)
forest2.fit(X2_train,y2_train)
y_pred_test_2 = pd.Series(forest2.predict(test))
gain_df = pd.DataFrame({'Gain': forest2.feature_importances_}, index=X_train.columns).sort_values('Gain', ascending = False)
gain_df
samp = pd.read_csv('samp_submission.csv')

samp.columns
index = samp['id']
ans = pd.concat([index, y_pred_test],axis=1 )
ans.rename(inplace=True, columns={0:'Predicted'})
print(ans)

samp.Predicted = ans

ans.to_csv('fifth_test.csv', index=False)