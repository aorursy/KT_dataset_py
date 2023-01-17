import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import math 
from scipy import stats as ss

df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

pd.options.display.max_columns = None
pd.options.display.max_rows = None
df.head(10)
columns_names = ['age', 'sex_male', 'cp', 'rest_bp', 'chol', 'fbs_above_120', 'rest_ecg',
       'max_bps', 'ex_angina', 'st_depression', 'ecg_st_slope', 'cardiac_fluoro', 'thallium', 'target']
df.columns = columns_names
columns_names_reordered = ['age', 'sex_male', 'rest_bp', 'chol', 'fbs_above_120',
       'max_bps', 'cp', 'ex_angina', 'rest_ecg', 'st_depression', 'ecg_st_slope', 'thallium', 'cardiac_fluoro', 'target']
df = df[columns_names_reordered]
#plot settings
sns.set(style = 'darkgrid')
plt.figure(figsize = (6,3))

#countplot
sns.countplot(x = "target", data = df, palette="Set2")
plt.title('Heart Disease Count')
#plot settings
sns.set(style = 'darkgrid')
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(1,8,figsize=(16,8))
fig.subplots_adjust(hspace=0.4, wspace=0.7)

#countplots
sns.countplot(x = 'sex_male', hue ='target', data = df, ax = ax1, palette = 'Set2')
sns.countplot(x = 'cp', hue = 'target', data = df, ax = ax2, palette = 'Set2')
sns.countplot(x = 'fbs_above_120', hue = 'target', data = df, ax = ax3, palette = 'Set2')
sns.countplot(x = 'rest_ecg', hue = 'target', data = df, ax = ax4, palette = 'Set2')
sns.countplot(x = "ex_angina", hue = 'target', data = df, ax = ax5, palette = 'Set2')
sns.countplot(x = "cardiac_fluoro", hue = 'target', data = df, ax = ax6, palette = 'Set2')
sns.countplot(x = "thallium", hue = 'target', data = df, ax = ax7, palette = 'Set2')
sns.countplot(x = "ecg_st_slope", hue = 'target', data = df, ax = ax8, palette = 'Set2')
#plot settings
sns.set(style = 'darkgrid')
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(16,8))
fig.subplots_adjust(hspace=0.4, wspace=0.7)

#boxplots
sns.boxplot(x = 'target', y ='age', data = df, ax = ax1, palette = 'Set2')
sns.boxplot(x = 'target', y ='rest_bp', data = df, ax = ax2, palette = 'Set2')
sns.boxplot(x = 'target', y ='chol', data = df, ax = ax3, palette = 'Set2')
sns.boxplot(x = 'target', y ='max_bps', data = df, ax = ax4, palette = 'Set2')
sns.boxplot(x = 'target', y ='st_depression', data = df, ax = ax5, palette = 'Set2')
#plot settings
plt.figure(figsize = (14,12))

#plot pairwise graphs
ax = sns.PairGrid(df, vars=['age', 'rest_bp','chol', 'max_bps', 'st_depression'], hue="target", palette = 'Set2')
ax.map_upper(plt.scatter)
ax.map_lower(sns.kdeplot)
ax.map_diag(sns.kdeplot, lw=3, legend=False)
ax.add_legend()
#specify the numerical variables
df_num = df[['age', 'rest_bp', 'chol', 'max_bps', 'st_depression', 'cardiac_fluoro']]
#add a mask to hide identical pairwise correlations
mask = np.zeros_like(df_num.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#plot settings
sns.set(font_scale=1.25)
plt.figure(figsize = (15,10))

#correlation matrix
corrMatrix = df_num.corr()
g = sns.heatmap(corrMatrix, vmin = -1,cmap='coolwarm', fmt='.2f', annot = True,
                square = True, linewidths = .5,mask = mask)
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
g.set_yticklabels(g.get_yticklabels(), rotation=45, horizontalalignment='right')

#specify the categorical variables
df_cramer = df[['cp', 'sex_male', 'fbs_above_120', 'ex_angina', 'rest_ecg', 'ecg_st_slope', 'thallium', 'target']]
#Cramer's V correlation matrix, used for categorical variables
#https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
#create the cramers correlation matrix
rows= []

for var1 in df_cramer:
    col = []
    for var2 in df_cramer:
        cramers =cramers_v(df_cramer[var1], df_cramer[var2])
        col.append(round(cramers,2)) 
    rows.append(col)

cramers_results = np.array(rows)
df_cramer_corr = pd.DataFrame(cramers_results, columns = df_cramer.columns, index =df_cramer.columns)

#mask the diagonial and top triangle
mask = np.zeros_like(df_cramer_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#set correlation graph size
sns.set(font_scale=1)
plt.figure(figsize = (10,10))

#correlation graph settings
ax = sns.heatmap(df_cramer_corr, vmin=0., vmax=1,cmap='coolwarm', fmt='.2f', annot = True, square = True,
                 linewidths = .5,mask = mask)
plt.show()
df.info()
#Create dummy variables for the following categorical attributes
#cp, rest_ecg, ecg_st_slope, cardiac_fluoro, thallium
cp_dummy = pd.get_dummies(df['cp'], drop_first = True)
cp_dummy.columns = ['cp_typical', 'cp_atypical', 'cp_non_anginal']

rest_ecg_dummy = pd.get_dummies(df['rest_ecg'], drop_first = True)
rest_ecg_dummy.columns = ['rest_ecg_st_abnormal', 'rest_ecg_left_vent_hyper']

ecg_st_slope_dummy = pd.get_dummies(df['ecg_st_slope'], drop_first = True)
ecg_st_slope_dummy.columns = ['ecg_st_slope_1', 'ecg_st_slope_2']

thallium_dummy = pd.get_dummies(df['thallium'], drop_first = True)
thallium_dummy.columns = ['thallium_normal', 'thallium_abnormal', 'thallium_rev_defect']

#add the dummy variables to our df
df = pd.concat([df, cp_dummy, rest_ecg_dummy, ecg_st_slope_dummy, thallium_dummy], axis =1)
#remove the original attributes of the dummy variables
df = df.drop(columns = ['cp', 'rest_ecg', 'ecg_st_slope', 'thallium'])
df.head(10)
#randomize the sample set
df = df.sample(frac=1).reset_index(drop=True)
df1 = df.copy()
#specify independent and dependent variables
y = df1['target']
X = df1[['age', 'sex_male', 'rest_bp', 'chol', 'fbs_above_120', 'max_bps', 'ex_angina', 'st_depression',
         'cardiac_fluoro', 'cp_typical', 'cp_atypical', 'cp_non_anginal', 'rest_ecg_st_abnormal',
         'rest_ecg_left_vent_hyper', 'ecg_st_slope_1', 'ecg_st_slope_2', 'thallium_abnormal', 
         'thallium_rev_defect']]

#from imblearn.over_sampling import SMOTE

#smt = SMOTE()
#X, y = smt.fit_sample(X, y)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#model
model = LogisticRegression(max_iter=3000)

#performance metrics
accuracy = cross_val_score(model, X, y, cv=4)
avg_accuracy = np.mean(accuracy)
print('Accuracy:', round((avg_accuracy)*100,2), '%', '+/-', round((max(accuracy)-avg_accuracy)*100,2), '%')

recall = cross_val_score(model, X, y, scoring = 'recall', cv= 4)
avg_recall = np.mean(recall)
print('Recall:', round((avg_recall)*100,2), '%', '+/-', round((max(recall)-avg_recall)*100,2), '%')

precision = cross_val_score(model, X, y, scoring = 'precision', cv= 4)
avg_precsion = np.mean(precision)
print('Precision:', round((avg_precsion)*100,2), '%', '+/-', round((max(precision)-avg_precsion)*100,2), '%')

import statsmodels.api as sm

#run logistic regression
logit_model = sm.Logit(y,X)
result=logit_model.fit()
result.summary()
from sklearn.feature_selection import RFE

# the model
model = LogisticRegression(max_iter=3000)

#run RFE
rfe = RFE(model, 10)
rfe = rfe.fit(X, y)

#display the ranking of each variable
series1 = pd.Series(X.columns.values)
series2 = pd.Series(rfe.ranking_)

rank = pd.DataFrame(data={'Variables': series1, 'Ranking' : series2})
rank.sort_values(by='Ranking')

X_adj = X[['sex_male', 'max_bps', 'ex_angina', 'st_depression',
         'cardiac_fluoro', 'cp_typical', 'cp_atypical', 'cp_non_anginal', 'rest_ecg_st_abnormal', 
         'thallium_rev_defect']]
#model
model = LogisticRegression(max_iter=3000)

#performance metrics
accuracy = cross_val_score(model, X_adj, y, cv=4)
avg_accuracy = np.mean(accuracy)
print('Accuracy:', round((avg_accuracy)*100,2), '%', '+/-', round((max(accuracy)-avg_accuracy)*100,2), '%')

recall = cross_val_score(model, X_adj, y, scoring = 'recall', cv= 4)
avg_recall = np.mean(recall)
print('Recall:', round((avg_recall)*100,2), '%', '+/-', round((max(recall)-avg_recall)*100,2), '%')

precision = cross_val_score(model, X_adj, y, scoring = 'precision', cv= 4)
avg_precsion = np.mean(precision)
print('Precision:', round((avg_precsion)*100,2), '%', '+/-', round((max(precision)-avg_precsion)*100,2), '%')
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

#split the data
X_train, X_test, y_train, y_test = train_test_split(X_adj, y, test_size=0.3, random_state=0)

#the model
model = LogisticRegression(max_iter=3000)
      
#fit and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_quant = model.predict_proba(X_test)[:, 1]

#ROC graph x and y axis
fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)
print('AUC:', round((auc(fpr, tpr))*100,2), '%')

#plot the ROC graph
fig, ax = plt.subplots(figsize = (8,8))
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from statistics import mean 
from scipy import stats as ss

#store the performance values for aggregation of our loop
specificity_store = []
recall_store = []
precision_store = []
f_measure_store = []
accuracy_store = []
auc_store = []

#ROC plot settings
fig, ax = plt.subplots(figsize = (8,8))
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])    
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
    

#specificy how many times to run the model
x = 10

#run the model x times
for i in range(x):
    #randomize the dataset and specify X and y
    df1_rand = df1.sample(frac=1).reset_index(drop=True)
    y = df1_rand['target']
    X_adj = df1_rand[['sex_male', 'max_bps', 'ex_angina', 'st_depression',
                      'cardiac_fluoro', 'cp_typical', 'cp_atypical', 'cp_non_anginal', 'rest_ecg_st_abnormal',
                      'thallium_rev_defect']]
    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X_adj, y, test_size=0.3, random_state=0)
    
    #run the model
    model = LogisticRegression(max_iter=3000) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #ROC plot
    y_pred_quant = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)
    ax.plot(fpr, tpr)
    
    #calculate and store performance metrics
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    
    specificity = TN / (TN + FP)
    specificity_store.append(specificity)

    recall = TP/ (TP + FN)
    recall_store.append(recall)
    
    precision = TP / (TP + FP)
    precision_store.append(precision)
    
    f_measure = (2 * TP) / ((2 * TP) + FP + FN)
    f_measure_store.append(f_measure)

    accuracy = (TP + TN)/ (TN + FP + FN + TP)
    accuracy_store.append(accuracy)

    auc_store.append(auc(fpr, tpr))

#performance metrics
avg_accuracy = np.mean(accuracy_store)
print('Accuracy:', round((avg_accuracy)*100,2), '%', '+/-', round(np.std(accuracy_store) * 1.96 * 100,2), '%')

avg_specificity = np.mean(specificity_store)
print('Specificity:', round((avg_specificity)*100,2), '%', '+/-', round(np.std(specificity_store) * 1.96 * 100,2), '%')

avg_recall = np.mean(recall_store)
print('Recall:', round((avg_recall)*100,2), '%', '+/-', round(np.std(recall_store) * 1.96 * 100,2), '%')

avg_precision = np.mean(precision_store)
print('Precision:', round((avg_precision)*100,2), '%', '+/-', round(np.std(precision_store) * 1.96 * 100,2), '%')

avg_f_measure = np.mean(f_measure_store)
print('F Measure:', round((avg_f_measure)*100,2), '%', '+/-', round(np.std(f_measure_store) * 1.96 * 100,2), '%')

avg_auc = np.mean(auc_store)
print('AUC:', round((avg_auc)*100,2), '%', '+/-', round(np.std(auc_store) * 1.96 * 100,2), '%')


from pdpbox import pdp, get_dataset, info_plots

#specify the variables to grab from
var_names = pd.Series(X_adj.columns.values)

# Create the data that we will plot
pdp_sex_male = pdp.pdp_isolate(model=model, dataset = df1, model_features=var_names, feature = 'sex_male')

# plot it
pdp.pdp_plot(pdp_sex_male, 'Sex Male')
plt.show()
# Create the data that we will plot
pdp_max_bps = pdp.pdp_isolate(model=model, dataset = df1, model_features=var_names, feature='max_bps')

# plot it
pdp.pdp_plot(pdp_max_bps, 'max_bps (Max Heartrate)')
plt.show()
# Create the data that we will plot
pdp_cardiac_fluoro = pdp.pdp_isolate(model = model, dataset = df1,
                                     model_features = var_names, feature = 'cardiac_fluoro')

# plot it
pdp.pdp_plot(pdp_cardiac_fluoro, 'cardiac_fluoro')
plt.show()
# Create the data that we will plot
pdp_st_depression = pdp.pdp_isolate(model = model, dataset = df1,
                                     model_features = var_names, feature = 'st_depression')

# plot it
pdp.pdp_plot(pdp_st_depression, 'st_depression')
plt.show()

features_to_plot = ['max_bps', 'sex_male']
inter1  =  pdp.pdp_interact(model=model, dataset=df1, model_features=var_names,
                            features=features_to_plot,num_grid_points=[10, 10])

fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot,
                                  plot_type='grid')

features_to_plot = ['cardiac_fluoro', 'st_depression']
inter1  =  pdp.pdp_interact(model=model, dataset=df1, model_features=var_names,
                            features=features_to_plot,num_grid_points=[10, 10])

fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot,
                                  plot_type='grid')

features_to_plot = ['max_bps', 'st_depression']
inter1  =  pdp.pdp_interact(model=model, dataset=df1, model_features=var_names,
                            features=features_to_plot,num_grid_points=[10, 10])

fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot,
                                  plot_type='grid')
#create a contingency table for sex_male and target
data_crosstab = pd.crosstab(df1['sex_male'], df1['target'], margins = True)
data_crosstab
#calculate odds ratio
odds_ratio = (93/114)/(72/24)
odds_ratio
#step 1 calculate natural log of odds ratio
ln_odds_ratio = math.log(odds_ratio)

#step 2 calculate standard error of log 
a, b, c, d = 93, 114, 72, 24
se_log = math.sqrt((1/a)+(1/b)+(1/c)+(1/d))

#step 3 calculate the 95% CI on the natual log scale
lower_CI_log = ln_odds_ratio - (1.96*se_log)
upper_CI_log = ln_odds_ratio + (1.96*se_log)

#step 4 convert back to normal scale
math.exp(lower_CI_log), math.exp(upper_CI_log)
#columns 
odds_ratio_storage = []
ci_lower_storage = []
ci_upper_storage = []

#set a copy of the dataframe
X2_adj = X_adj.copy()

#specify cut off points
X2_adj.loc[X2_adj['max_bps'] >= 150, 'max_bps_greater_150'] = 1
X2_adj.max_bps_greater_150 = X2_adj.max_bps_greater_150.fillna(0)

X2_adj.loc[X2_adj['cardiac_fluoro'] >= 1, 'cardiac_fluoro_greater_0'] = 1
X2_adj.cardiac_fluoro_greater_0 = X2_adj.cardiac_fluoro_greater_0.fillna(0)

X2_adj.loc[X2_adj['st_depression'] >= 1, 'st_depression_greater_1'] = 1
X2_adj.st_depression_greater_1 = X2_adj.st_depression_greater_1.fillna(0)

X2_adj = X2_adj.drop(columns = ['max_bps' ,'cardiac_fluoro', 'st_depression'])
X2_adj = X2_adj.astype(int)

#variable names
var_names_X2_adj = pd.Series(X2_adj.columns.values)

#put the dataframe back together
df1_adj = pd.concat([y, X2_adj],axis=1)

#loop through each variable
for i in var_names_X2_adj:
    data_crosstab = pd.crosstab(df1_adj[str(i)], df1_adj['target'], margins = True)

    odds_ratio = (data_crosstab.iloc[1 , 1]/data_crosstab.iloc[1 , 0])/(data_crosstab.iloc[0 , 1]/data_crosstab.iloc[0 , 0])

    #step 1 calculate natural log of odds ratio
    ln_odds_ratio = math.log(odds_ratio)

    #step 2 calculate standard error of log 
    a = data_crosstab.iloc[1 , 1]
    b = data_crosstab.iloc[1 , 0]
    c = data_crosstab.iloc[0 , 1]
    d = data_crosstab.iloc[0 , 0]
    se_log = math.sqrt((1/a)+(1/b)+(1/c)+(1/d))

    #step 3 calculate the 95% CI on the natual log scale
    lower_CI_log = ln_odds_ratio - (1.96*se_log)
    upper_CI_log = ln_odds_ratio + (1.96*se_log)

    #step 4 convert back to normal scale
    lower_ci = math.exp(lower_CI_log) 
    upper_ci = math.exp(upper_CI_log)
    
    #append the odds ratios and ci 
    odds_ratio_storage.append(odds_ratio)
    ci_lower_storage.append(lower_ci)
    ci_upper_storage.append(upper_ci)
    
#odds ratio table
odds_table = pd.DataFrame(data={'Variables': var_names_X2_adj, 'Odds Ratios' : odds_ratio_storage,
                                '95% CI Lower' : ci_lower_storage, '95% CI Upper' : ci_upper_storage})
odds_table.sort_values(by='Odds Ratios')
