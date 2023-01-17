import pandas as pd
raw_data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(raw_data.dtypes)
pd.set_option('display.max_columns', None)
raw_data.head()
import matplotlib.pyplot as plt
import numpy as np

gender_values = raw_data['gender'].unique()
gender_counts = [None] * len(gender_values)
i = 0
for value in gender_values:
    gender_counts[i] = raw_data['gender'].loc[raw_data.gender == value].count()
    i = i + 1
    
plt.bar(gender_values, gender_counts / raw_data['gender'].count())
plt.show()
#let's get rid of code repetition before continuing
def plot_series_bar(series, series_name, ax):
    values = series.unique()
    counts = [None] * len(values)
    i = 0
    for value in values:
        counts[i] = series.loc[series == value].count()
        i = i + 1
    
    ax.bar(values, counts / series.count())
    ax.set_title(series_name)
    
    
fig, axes = plt.subplots(ncols=5, sharex=False, sharey=True, figsize=(15,5))
i = 0
for column in ['SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines']:
    plot_series_bar(raw_data[column], column, axes[i])
    i = i + 1
plt.show()
fig, axes = plt.subplots(ncols=7, sharex=False, sharey=True, figsize=(15,5))
i = 0
for column in ['InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 
               'StreamingTV', 'StreamingMovies']:
    plot_series_bar(raw_data[column], column, axes[i])
    i = i + 1
plt.show()
fig, axes = plt.subplots(ncols=4, sharex=False, sharey=True, figsize=(15,5))
i = 0
for column in ['Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']:
    plot_series_bar(raw_data[column], column, axes[i])
    i = i + 1
plt.show()
import numpy as np
binwidth = 1
tenure_ser = raw_data['tenure']
plt.hist([raw_data.loc[raw_data['Churn']=='Yes']['tenure'], raw_data.loc[raw_data['Churn']=='No']['tenure']],
         bins=np.arange(min(tenure_ser), max(tenure_ser) + binwidth, binwidth), histtype='barstacked',
        label=['Has Churned', 'Has not'])
plt.legend(prop={'size': 10})
plt.show()
import numpy as np

def plot_hist_with_churn(df, series_name, binwidth, ax):
    ser = df[series_name]
    ax.hist([df.loc[df['Churn']=='Yes'][series_name], df.loc[df['Churn']=='No'][series_name]],
         bins=np.arange(min(ser), max(ser) + binwidth, binwidth), histtype='barstacked',
        label=['Has Churned', 'Has not'])
    ax.legend(prop={'size': 10})

fig, ax = plt.subplots()
plot_hist_with_churn(raw_data, 'MonthlyCharges', 1, ax)
plt.show()
total_charges_ser = raw_data['TotalCharges']
total_charges_ser = pd.to_numeric(total_charges_ser, errors='coerce')
raw_data['TotalCharges'] = total_charges_ser
#How many NaN do we have at TotalCharges?
print(raw_data.loc[total_charges_ser.isnull()==True].count()['customerID'])
without_nan = raw_data.loc[total_charges_ser.isnull()==False].infer_objects()    
print(without_nan.TotalCharges.hist())

#Create a boxplot
without_nan.boxplot('TotalCharges', by='Churn')
import statsmodels.api as sm
from statsmodels.formula.api import ols

def print_anova(series_name, data):
    model = ols(series_name+' ~ Churn', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    print('\n')

print_anova('TotalCharges', without_nan) 
print_anova('MonthlyCharges', without_nan)
print_anova('tenure', without_nan)

import scipy.stats as stats

def print_cross_table(df, series_name):
    tab = pd.crosstab(df.Churn, df[series_name], margins = True)
    tab.columns = ["Not "+series_name, series_name, "row_totals"]
    tab.index = ["Has Churned", "Has not", "col_totals"]
    print(tab)
    print("\n")
    return tab
    
tab = print_cross_table(without_nan,'SeniorCitizen')
print(stats.chi2_contingency(observed= tab.iloc[0:2,0:2]))
print("\n")
tab = print_cross_table(without_nan,'PhoneService')
print(stats.chi2_contingency(observed= tab.iloc[0:2,0:2]))
cat_data =  pd.DataFrame()
cat_data['tenure'] = pd.cut(without_nan['tenure'], 6, labels=['1','2','3','4','5','6']) #cat of one year
cat_data['MonthlyCharges'] = pd.cut(without_nan['MonthlyCharges'], 5, labels=['1','2','3','4','5'])

multi_categorical_columns = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                       'TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']

simple_categorical_columns = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','PaperlessBilling','Churn']

preproc_data =  pd.DataFrame()

for category in multi_categorical_columns:
    dummies = pd.get_dummies(without_nan[category], drop_first=True)
    try:
        dummies = dummies.drop(columns=['No phone service'])
    except:
        pass
    try:
        dummies = dummies.drop(columns=['No internet service'])
    except:
        pass
    
    col_names = []
    for col_name in dummies.columns:
        col_names = col_names + [category + '-' + col_name]
    dummies.columns = col_names
    preproc_data = pd.concat([preproc_data, dummies], axis=1)
    
    
for category in ['tenure','MonthlyCharges']:
    dummies = pd.get_dummies(cat_data[category], drop_first=True)
    col_names = []
    for col_name in dummies.columns:
        col_names = col_names + [category + '-' + col_name]
    dummies.columns = col_names
    preproc_data = pd.concat([preproc_data, dummies], axis=1)

for category in simple_categorical_columns:
    dummies = pd.get_dummies(without_nan[category], drop_first=True)
    col_names = []
    for col_name in dummies.columns:
        col_names = col_names + [category + '-' + str(col_name)]
    dummies.columns = col_names
    preproc_data = pd.concat([preproc_data, dummies], axis=1) 

preproc_data['TotalCharges'] = without_nan['TotalCharges']
from sklearn.linear_model import Lasso
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(preproc_data)
norm_data = pd.DataFrame(data=scaler.transform(preproc_data), columns=preproc_data.columns)

ys = []
xs = []

independent = norm_data.drop(columns=['Churn-Yes'])
dependent = norm_data['Churn-Yes']

for alpha in range(1,20):
    
    lassoreg = Lasso(alpha=(alpha/1e2),normalize=False, max_iter=1e4)

    lassoreg.fit(independent,dependent)
    
    ys = ys + [lassoreg.coef_.tolist()]
    
    xs = xs + [(alpha/1e2)]

handles = plt.plot(xs, ys)
plt.legend(handles=handles, labels=independent.columns.tolist(), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
lassoreg = Lasso(alpha=0.05, normalize=False, max_iter=1e4)
lassoreg.fit(independent,dependent)
coefs = np.abs(lassoreg.coef_.tolist())
best_ten = sorted(range(len(coefs)), key=lambda k: coefs[k], reverse=True)[0:10]
best_ten_feats = []
for i in best_ten:
    print(independent.columns[i] + ' with coef: ' + str(lassoreg.coef_[i]))
    best_ten_feats = best_ten_feats + [independent.columns[i]]
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Get the best features, add back the missing categories of the best features and put independent variable
model_data = preproc_data[best_ten_feats + ['PaymentMethod-Credit card (automatic)', 'PaymentMethod-Mailed check',
                                           'tenure-2','tenure-4','tenure-5','tenure-6'] + ['Churn-Yes']]

rf = RandomForestClassifier(n_estimators = 100, random_state = 12)
train_features, test_features = train_test_split(model_data) #by default, 25% of the data is test data

x_train = train_features.drop(columns=['Churn-Yes'])
y_train = (train_features['Churn-Yes']>0.5)

x_test = test_features.drop(columns=['Churn-Yes'])
y_test = (test_features['Churn-Yes']>0.5)

rf.fit(x_train, y_train)

# Use the forest's predict method on the test data
predictions = rf.predict(x_test)

true_positives = ( (predictions==1) & (y_test==1) )
false_negatives = ( (predictions==0) & (y_test==1) )

# Calculate and display accuracy
print('Precision:', round(100*(true_positives.sum()/(predictions>0.5).sum()), 2), '%.')
print('Recall:', round(100*(true_positives.sum()/(y_test==True).sum()), 2), '%.')
import xgboost as xgb

# specify parameters via map
param = {'max_depth':5, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 10

xgtrain = xgb.DMatrix(x_train.values, y_train.values)
xgtest = xgb.DMatrix(x_test.values)

bst = xgb.train(param, xgtrain, num_round)
# make prediction
predictions = bst.predict(xgtest)

true_positives = ( (predictions>0.5) & (y_test==True) )

# Calculate and display accuracy
print('Precision:', round(100*(true_positives.sum()/(predictions>0.5).sum()), 2), '%.')
print('Recall:', round(100*(1 - true_positives.sum()/(y_test==True).sum()), 2), '%.')
from sklearn.neural_network import MLPClassifier

train_data, test_data = train_test_split(preproc_data) #by default, 25% of the data is test data

x_train = train_data.drop(columns=['Churn-Yes'])
y_train = (train_data['Churn-Yes']>0.5)

x_test = test_data.drop(columns=['Churn-Yes'])
y_test = (test_data['Churn-Yes']>0.5)

scaler = preprocessing.StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)  
# apply same transformation to test data
x_test = scaler.transform(x_test)  

clf = MLPClassifier(solver='adam', alpha=1e-7,
                    hidden_layer_sizes=(10, 10, 2), random_state=1, max_iter=200000)

clf.fit(x_train, y_train)

# make prediction
predictions = clf.predict(x_test)

true_positives = ( (predictions>0.5) & (y_test==True) )

# Calculate and display accuracy
print('Precision:', round(100*(true_positives.sum()/(predictions>0.5).sum()), 2), '%.')
print('Recall:', round(100*(1 - true_positives.sum()/(y_test==True).sum()), 2), '%.')