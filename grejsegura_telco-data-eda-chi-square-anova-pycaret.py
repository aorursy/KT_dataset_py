!pip install pycaret
import pandas as pd
import numpy as np
import os
from datetime import datetime
import math
import time
import random

# VISUALIZATION
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# PREPROCESSING
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# STATISTICAL TESTS
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy import stats

# MODELING AND EVALUATION
from pycaret.classification import *
from sklearn import metrics


import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print('The data has {} rows and {} columns.'.format(data.shape[0], data.shape[1]))
data.head()
data.columns
data.info()
data['SeniorCitizen'].value_counts()
data['SeniorCitizen'] = data['SeniorCitizen'].astype(str)
char_not_num = data[['TotalCharges']][~data['TotalCharges'].str.contains('[1-9.]')]
char_not_num
print('There are {} rows that were found to have a special character.'.format(len(char_not_num)))
data['TotalCharges'] = data['TotalCharges'].replace(" ",np.nan).astype(float)
data = data[pd.notnull(data['TotalCharges'])].reset_index().drop('index', axis=1)
data.info()
data.isnull().sum()
data.describe()
# create a bar plot based on frequency

plt.figure()
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
sns.set(rc={'figure.figsize':(10,7)})
cplot = sns.countplot(x="Churn", data=data)
for p in cplot.patches:
         cplot.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', fontsize=15, xytext=(0, 20),
             textcoords='offset points')
cplot.axes.set_title("Churn",fontsize=20)
cplot.set_xlabel("", fontsize=18)
cplot.set_ylabel("Count", fontsize=18)
# this function creates a barplot given the data and the categorical feature against the Churns
def plot_categorical(data, feature, rotate=0, y_axis="percentage of customers", title=None):
    plt.figure()
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15, rotation=rotate)
    sns.set(rc={'figure.figsize':(10,7)})
    five_thirty_eight = [
                        "#30a2da",
                        "#fc4f30",
                        "#e5ae38",
                        "#6d904f",
                        "#8b8b8b",
                        ]
    sns.set_palette(five_thirty_eight)

    graph_data = data.groupby(feature)["Churn"].value_counts().to_frame()
    graph_data = graph_data.rename({"Churn": y_axis}, axis=1).reset_index()
    graph_data[y_axis] = graph_data[y_axis]/len(data)
    bar = sns.barplot(x=feature, y= y_axis, hue='Churn', data=graph_data)
#     bar.set_xticklabels(['{:,.0%}'.format(x) for x in bar.get_xticks()])
    bar.set_yticklabels(['{:,.0%}'.format(x) for x in bar.get_yticks()])
    
    for p in bar.patches:
             bar.annotate("%.2f" % (p.get_height()*100), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=15, xytext=(0, 20),
                 textcoords='offset points')

    bar.axes.set_title(title,fontsize=20)
    bar.set_xlabel("")
    bar.set_ylabel(y_axis, fontsize=18)
    bar.plot()
    plt.savefig('bar.pdf')
# create a table showing values per churn

def show_crosstab(data, feature):
    orange = sns.light_palette("orange", as_cmap=True)
    tab = pd.crosstab(data[feature],data['Churn'], margins=True).style.set_table_attributes('style="font-size: 15px"')
    return tab
excluded_features = ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']
cat_features = data.drop(excluded_features, axis=1)
cat_features.columns
plot_categorical(data=data, feature='SeniorCitizen', title='Senior Citizen')
show_crosstab(data, 'SeniorCitizen')
plot_categorical(data=cat_features, feature='gender', title='Gender')
show_crosstab(data, 'gender')
plot_categorical(data=cat_features, feature='Partner', title='Partner')
show_crosstab(data, 'Partner')
plot_categorical(data=cat_features, feature='Dependents', title='Dependents')
show_crosstab(data, 'Dependents')
plot_categorical(data=cat_features, feature='PhoneService', title='Phone Service')
show_crosstab(data, 'PhoneService')
plot_categorical(data=cat_features, feature='MultipleLines', title='Multiple Lines')
show_crosstab(data, 'MultipleLines')
plot_categorical(data=cat_features, feature='InternetService', title='Internet Service')
show_crosstab(data, 'InternetService')
plot_categorical(data=cat_features, feature='OnlineSecurity', title='Online Security')
show_crosstab(data, 'OnlineSecurity')
plot_categorical(data=cat_features, feature='OnlineBackup', title='Online Backup')
show_crosstab(data, 'OnlineBackup')
plot_categorical(data=cat_features, feature='DeviceProtection', title='Device Protection')
show_crosstab(data, 'DeviceProtection')
plot_categorical(data=cat_features, feature='TechSupport', title='Tech Support')
show_crosstab(data, 'TechSupport')
plot_categorical(data=cat_features, feature='StreamingTV', title='Streaming TV')
show_crosstab(data, 'StreamingTV')
plot_categorical(data=cat_features, feature='StreamingMovies', title='Streaming Movies')
show_crosstab(data, 'StreamingMovies')
plot_categorical(data=cat_features, feature='Contract', title='Contract')
show_crosstab(data, 'Contract')
plot_categorical(data=cat_features, feature='PaperlessBilling', title='Paperless Billing')
show_crosstab(data, 'PaperlessBilling')
plot_categorical(data=cat_features, feature='PaymentMethod', rotate=30, title='Payment Method')
show_crosstab(data, 'PaymentMethod')
# create a density plot sshowing diference between churn groups
def plot_numerical(data, feature, title=None):
    plt.figure()
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15, rotation=0)
    sns.set(rc={'figure.figsize':(10,7)})
    five_thirty_eight = [
                        "#30a2da",
                        "#fc4f30",
                        "#e5ae38",
                        "#6d904f",
                        "#8b8b8b",
                        ]
    sns.set_palette(five_thirty_eight)
    plot = sns.distplot(data[feature][data['Churn']=='No'], hist = False, kde = True,
                     kde_kws = {'shade': True, 'linewidth': 1}, color='red')
    plot = sns.distplot(data[feature][data['Churn']=='Yes'], hist = False, kde = True,
                     kde_kws = {'shade': True, 'linewidth': 1})
    plot.axes.set_title(title,fontsize=20)
    plot.set_xlabel("")
    plot.set_ylabel('density', fontsize=18)
    plot.legend(labels=['No','Yes'], title='Churn')
    plot.plot()
    plt.savefig('bar.pdf')
plot_numerical(data=data, feature='TotalCharges', title='Total Charges')
plot_numerical(data=data, feature='tenure', title='Tenure')
plot_numerical(data=data, feature='MonthlyCharges', title='Monthly Charges')
# this function creates scatterplot with churn groups as hue 
def plot_pair(data, x_feature, y_feature, title=None):
    sns.set(rc={'figure.figsize':(12,8.27)})
    sns.set(font_scale = 1.5)
    scatter = sns.scatterplot(x=x_feature, y=y_feature, data=data, hue='Churn', s=100, alpha=0.7)
    scatter.axes.set_title(title,fontsize=20)
    scatter.plot()

# create data for the scatter plot
scatter_data = data[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]
plot_pair(scatter_data, 'tenure', 'MonthlyCharges', title='Tenure vs. Monthly Charges')
plot_pair(scatter_data, 'tenure', 'TotalCharges', title='Tenure vs. Total Charges')
plot_pair(scatter_data, 'TotalCharges', 'MonthlyCharges', title='Monthly Charges vs. Total Charges')
sns.set(rc={'figure.figsize':(15,10)})
sns.set(font_scale = 1.5)
scatter = sns.scatterplot(x='TotalCharges', y='MonthlyCharges', data=data, hue='Churn', size='tenure', sizes=(20, 300), alpha=0.7)
scatter.axes.set_title('Numerical Features Relationship',fontsize=20)
scatter.plot()
features = pd.Series(cat_features.drop(['Churn'], axis=1).columns).append(pd.Series(['SeniorCitizen']))

# this function creates will automaticall generate the decision for the chi square test
def chi_square_test(data, feature):
    df = pd.crosstab(data[feature],data['Churn'])
    stat, p, dof, expected = chi2_contingency(df)
    print('='*60)
    print('Chi-Square Test for {}'.format(feature))
    print('='*60)
    print('degrees of freedom = %d' % dof)
    print('-'*60)
    # interpret test-statistic
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    print('Results:')
    print(' ')
    print('probability = %.3f, critical = %.3f, stat = %.3f' % (prob, critical, stat))
    print('-'*60)
    print('Decision based on Chi-Square Statistics:')
    print(' ')
    if abs(stat) >= critical:
        print('The Churns are dependent on {} : (reject H0)'.format(feature))
    else:
        print('The Churns are independent on {} : (fail to reject H0)'.format(feature))
    # interpret p-value
    alpha = 1.0 - prob
    print('-'*60)
    print('Decision based on p-value:')
    print(' ')
    print('alpha = %.2f, p = %.3f' % (alpha, p))
    if p <= alpha:
        print('The Churns are dependent on {} : (reject H0)'.format(feature))
    else:
        print('The Churns are independent on {} : (fail to reject H0)'.format(feature))
    print('_'*60)
    print(' '*60)
    print('_'*60)
    print('_'*60)
for feature in features:
    chi_square_test(data, feature)
yes = data['MonthlyCharges'][data['Churn']=='Yes']
yes = yes.sample(round(len(yes)*0.05))
no = data['MonthlyCharges'][data['Churn']=='No']
no = no.sample(round(len(no)*0.05))
stats.f_oneway(yes, no)
yes = data['TotalCharges'][data['Churn']=='Yes']
yes = yes.sample(round(len(yes)*0.05))
no = data['TotalCharges'][data['Churn']=='No']
no = no.sample(round(len(no)*0.05))
stats.f_oneway(yes, no)
yes = data['tenure'][data['Churn']=='Yes']
yes = yes.sample(round(len(yes)*0.05))
no = data['tenure'][data['Churn']=='No']
no = no.sample(round(len(no)*0.05))
stats.f_oneway(yes, no)
cats = cat_features.columns
customerID = data.customerID
cleanData = pd.get_dummies(data.drop(['customerID'],axis=1), prefix = cats)
cleanData = cleanData.drop('Churn_No', axis=1)
cleanData.head()
# this function creates thesplits the data into 75/25 train/test proportion
def split_data(data):
    # change the target feature name to labels
    data = data.rename(columns={'Churn_Yes': 'labels'})
    dataX = data.drop(['labels'], axis = 1)
    dataY = data['labels']
    
    # Create train and test dataset
    X_train, x_test, Y_train, y_test = train_test_split(dataX, dataY, random_state = 0)
    return X_train, x_test, Y_train, y_test

X_train, x_test, Y_train, y_test = split_data(cleanData)
print('There are {} rows for training.'.format(len(X_train)))
print('There are {} rows for testing.'.format(len(x_test)))
data = pd.concat([X_train, Y_train], axis=1)
data = data.rename(columns={'labels':'target'})
clf1 = setup(data, target = 'target')

best1 = compare_models(sort='AUC', whitelist=['dt','rf','xgboost','lightgbm','ada','lr','nb'], fold=5)
clf2 = setup(data, target = 'target', fix_imbalance = True) # this will implement the oversampling using the SMOTE method to balance the data
best2 = compare_models(sort='AUC', whitelist=['dt','rf','xgboost','lightgbm','ada','lr','nb'], fold=5)
clf2 = setup(data, target = 'target', fix_imbalance = True) # this will implement the oversampling using the SMOTE method to balance the data
ada = create_model('ada')
tuned_ada = tune_model(ada, optimize = 'AUC', n_iter = 1000, fold = 5) # this will randomly search a set of parameters, it is based on the randomseach function of sklearn
plot_model(tuned_ada, plot = 'parameter')
plot_model(tuned_ada, plot='auc')
clf2 = setup(data, target = 'target', fix_imbalance = True) # this will implement the oversampling using the SMOTE method to balance the data
lgb = create_model('lightgbm')
tuned_lgb = tune_model(lgb, optimize = 'AUC', n_iter = 1000, fold = 5) # this will randomly search a set of parameters, it is based on the randomseach function of sklearn
plot_model(tuned_lgb, plot = 'parameter')
plot_model(tuned_lgb, plot='auc')
clf2 = setup(data, target = 'target', fix_imbalance = True) # this will implement the oversampling using the SMOTE method to balance the data
lr = create_model('lr')
tuned_lr = tune_model(lr, optimize = 'AUC', n_iter = 1000, fold = 5) # this will randomly search a set of parameters, it is based on the randomseach function of sklearn
plot_model(tuned_lr, plot = 'parameter')
plot_model(tuned_lr, plot='auc')
clf2 = setup(data, target = 'target', fix_imbalance = True) # this will implement the oversampling using the SMOTE method to balance the data
xgb = create_model('xgboost')
tuned_xgb = tune_model(xgb, optimize = 'AUC', n_iter = 1000, fold = 5) # this will randomly search a set of parameters, it is based on the randomseach function of sklearn
plot_model(tuned_xgb, plot = 'parameter')
plot_model(tuned_xgb, plot='auc')
clf2 = setup(data, target = 'target', fix_imbalance = True) # this will implement the oversampling using the SMOTE method to balance the data
nb = create_model('nb')
tuned_nb = tune_model(nb, optimize = 'AUC', n_iter = 1000, fold = 5) # this will randomly search a set of parameters, it is based on the randomseach function of sklearn
plot_model(tuned_nb, plot = 'parameter')
plot_model(tuned_nb, plot='auc')
# save the final model
ada_final = finalize_model(tuned_ada)
lgb_final = finalize_model(tuned_lgb)
lr_final = finalize_model(tuned_lr)
xgb_final = finalize_model(tuned_xgb)
nb_final = finalize_model(tuned_nb)

# reset the index both for x_test and y_test
x_test = x_test.reset_index().drop(['index'], axis=1)
y_test = y_test.reset_index().drop(['index'], axis=1)

# predict using the hold out data
ada_preds = predict_model(ada_final, data = x_test, probability_threshold=.5)
lgb_preds = predict_model(lgb_final, data = x_test, probability_threshold=.45)
lr_preds = predict_model(lr_final, data = x_test, probability_threshold=.45)
xgb_preds = predict_model(xgb_final, data = x_test, probability_threshold=.45)
nb_preds = predict_model(nb_final, data = x_test, probability_threshold=.45)

# save the predicted values
ada = ada_preds['Label']
lgb = lgb_preds['Label']
lr = lr_preds['Label']
xgb = xgb_preds['Label']
nb = nb_preds['Label']
models = {'Adaboost':ada,'Light GBM':lgb,'Logistic Regression':lr,'XGBoost':xgb,'Naive Bayes':nb}

for name, model in models.items():
    print(' ')
    print('{}'.format(name))
    print(' ')
    print('PRECISION {}'.format(metrics.precision_score(y_test, model)))
    print('RECALL {}'.format(metrics.recall_score(y_test, model)))
    print('F1 {}'.format(metrics.f1_score(y_test, model)))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, model)
    auc_score = metrics.auc(fpr, tpr)
    print('AUC {}'.format(auc_score))
    print('ACCURACY {}'.format(metrics.accuracy_score(model, y_test)))
    print('='*50)
blender = blend_models(estimator_list = [ada_final,lgb_final,lr_final,xgb_final,nb_final], method = 'soft', fold=5)
# save the final model
blender_final = finalize_model(blender)

# predict using the hold out data
blender_preds = predict_model(blender_final, data = x_test)

# save the predicted values
blend_value = blender_preds['Label']

print(' ')
print('{}'.format('Blender Model'))
print(' ')
print('PRECISION {}'.format(metrics.precision_score(y_test, blend_value)))
print('RECALL {}'.format(metrics.recall_score(y_test, blend_value)))
print('F1 {}'.format(metrics.f1_score(y_test, blend_value)))

fpr, tpr, thresholds = metrics.roc_curve(y_test, blend_value)
auc_score = metrics.auc(fpr, tpr)
print('AUC {}'.format(auc_score))
print('ACCURACY {}'.format(metrics.accuracy_score(blend_value, y_test)))
print('='*50)
