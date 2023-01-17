import pandas as pd
import os
from scipy import stats
from statsmodels.stats import anova
import statsmodels.api as sm
from statsmodels.formula.api import ols

wd = './'
mydat = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
print(type(mydat))

# 維度
print(mydat.shape)
print(len(mydat))
cols = (mydat.columns)
print(cols.tolist())

# types of columns
import numpy as np

is_numeric = mydat.dtypes != 'object'
numeric_cols = mydat.columns[is_numeric]


# missing values
missing_cols = mydat.columns[np.where(mydat.isna())[1]]
print('Missing columns:\n', np.unique(missing_cols),'\n')
print('Missing percentage:\n', missing_cols.value_counts() / len(mydat),'\n')

### salary missing deeper explore
not_placed = np.where(mydat['status']=='Not Placed')[0]
salary_missing = np.where(mydat['salary'].isna())[0]

print('Is the data points which has missing salary are also the "not placed" points:\n', (not_placed == salary_missing).all(),'\n')

# numeric distribution
### distribution
pd.set_option("display.max_columns", 15)
print('Summary of numeric variables:\n',mydat.describe(include=[np.number]),'\n')

### outlier detect
def inRangeCheck(x, left, right): # x: series, left: int, right: int; outlier: list
    #outlier = [x[i] for i in range(len(x)) if x[i]<left] + [x[i] for i in range(len(x)) if x[i]>right]
    outlier = pd.concat([x[x<left], x[x>right]]).sort_values()
    return outlier

def outlierDetect(x): # x: series; outlier: list
    x_series = pd.Series(data=x)
    q1 = x_series.quantile(.25)
    q3 = x_series.quantile(.75)
    iqr = q3 - q1
    outlier = pd.concat([x_series[x_series<(q1-1.5*iqr)], x_series[x_series>(q1+1.5*iqr)]]).sort_values()
    return outlier


cols = [nc for nc in numeric_cols if nc not in ['sl_no','salary']]
out_range = mydat.loc[:,cols].apply(inRangeCheck, left=0, right=100)
out_salary = inRangeCheck(mydat.loc[:,'salary'], left=0,right=float('inf'))
print('Out of range:\n', out_range, '\n')
print('Negative salary:\n ', out_salary, '\n')


cols = [nc for nc in numeric_cols if nc not in ['sl_no']]
outlier_normal = []
for col in cols:
    temp = outlierDetect(mydat.loc[:, col])
    outlier_normal.append(temp)

outlier_len = {}
for i in range(len(outlier_normal)):
    outlier_len.update({outlier_normal[i].name :len(outlier_normal[i])})

print("outlier_len: \n", outlier_len, '\n')
### correlation
corr = mydat.loc[:,numeric_cols].corrwith(mydat['salary'])
print('Correlation with salary:\n',corr,'\n')

# categorical distribution
categorical_cols = mydat.columns[mydat.dtypes == 'object']

frequency = {}
i = 0
for cc in categorical_cols:
    if i == 0:
        print('Count Values:')
    temp = mydat[cc].value_counts()
    frequency.update({cc: temp})
    print(temp,'\n')
    i += 1

# Visualization
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

## Single variable distribution check
### Numeric Variables
plot_numeric_cols = [nc for nc in numeric_cols if nc not in ['sl_no']]
n_row = 3
n_col = 2
fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(8,7))
plt.subplots_adjust(hspace=0.7)
count=0
for i in range(n_row):
    for j in range(n_col):
        sns.distplot(mydat[plot_numeric_cols[count]],
            ax=axes[i,j])
        axes[i,j].set_title(plot_numeric_cols[count], fontsize=15)
        count+=1

plt.show()

#### box plot
fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(8,7))
plt.subplots_adjust(hspace=0.7)

count=0
for i in range(n_row):
    for j in range(n_col):
        sns.boxplot(mydat[plot_numeric_cols[count]],
            ax=axes[i,j])
        axes[i,j].set_title(plot_numeric_cols[count], fontsize=15)
        count+=1

plt.show()

### Categorical Variables
n_row = 4
n_col = 2
fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(10,7))
plt.subplots_adjust(hspace=0.7, wspace=0.3)
count=0
for i in range(n_row):
    for j in range(n_col):
            sns.barplot(x=frequency[categorical_cols[count]].values, y=frequency[categorical_cols[count]].index,
                ax=axes[i,j])
            axes[i,j].set_title(categorical_cols[count], fontsize=15)
            count+=1

#plt.savefig(wd + '/categorical_eda1.png')
plt.show()
### Which variable is associated with the "status" variable
##### Numeric variables
plot_numeric_cols = [nc for nc in numeric_cols if nc not in ['sl_no']]

fig, axes = plt.subplots(nrows=len(plot_numeric_cols), ncols=1, figsize=(7,10))
for i in range(len(plot_numeric_cols)):
    if(plot_numeric_cols[i]!='salary'):
        sns.violinplot(data=mydat, x=plot_numeric_cols[i], y='status', #hue='status',
            cut=0, order=['Placed','Not Placed'], scale='count', bw=.3, orient='h',
            ax=axes[i]) # 指定畫在哪個subplots
    else:
        sns.violinplot(data=mydat, x=plot_numeric_cols[i], y='status', hue='status',
            cut=0, order=['Placed','Not Placed'], scale='count', bw=.3, orient='h',
            ax=axes[i])
    axes[i].set_ylabel(plot_numeric_cols[i], rotation=0, fontsize=15, labelpad=27) # ax.set_ylabel
    axes[i].set_yticks(ticks=[])
    axes[i].set_xlabel('')

axes[0].set_title('EDA of Numeric Variables', fontsize=20)

#plt.savefig(wd + '/numeric_eda.png')
plt.show()
##### catgorical variables
plot_categorical_cols = categorical_cols[categorical_cols != 'status']
#cross_dat = pd.crosstab(index=mydat['status'], columns=[mydat['workex']])

n_row = 4
n_col = 2
fix, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(10,7))
plt.subplots_adjust(hspace=0.7, wspace=0.3)
for i in range(n_row):
    end = False
    for j in range(n_col):
        ind = i*2+j
        if ind>(len(plot_categorical_cols)-1):
            end = True
            break
        col = plot_categorical_cols[ind]
        count_table = mydat.groupby(['status',col]).size().reset_index(name='counts')
        count_table['total'] = count_table['counts'].groupby(count_table[col]).transform('sum')
        count_table['proportion'] = (count_table['counts']/count_table['total']*100).round(2)

        sns.barplot(data=count_table, x=col, y="proportion", hue="status", ax=axes[i,j])
        ys = count_table['proportion'].groupby(count_table[col]).max()
        values = count_table.loc[:,[col,'total']].drop_duplicates(subset=col).set_index(col)['total']

        for x, y, value in zip(range(len(ys)), ys.values, values.values):
            axes[i,j].text(x=x, y=y, s=str(value), fontsize=12, horizontalalignment='center')

        axes[i,j].set_title(col, fontsize=14)
        axes[i,j].set_ylabel('')
        axes[i,j].set_xlabel('')

    if end:
        break
plt.show()
# Test of different mean of every numeric variable
# #Before hypothesis test we first do data transfomation and normality test
# #salary, etest_p seem to be more likely to be transform to normality
# #Seems etest_p is a little right skrew

lm_model = ols('etest_p~status', data=mydat).fit()
sns.distplot(lm_model.resid)
plt.show()
 
mydat['etest_p_trans0'] = mydat['etest_p'] ** .1 #Make it less right-skrewed
lm_model = ols('etest_p_trans0~status', data=mydat).fit()
sns.distplot(lm_model.resid)
plt.show()

mydat['etest_p_trans'] = (mydat['etest_p_trans0']-min(mydat['etest_p_trans0']))*100/(max(mydat['etest_p_trans0'])-min(mydat['etest_p_trans0']))

mydat.to_csv(wd + '/Placement_Data_Transformed.csv', index=False)
test_numeric_cols = plot_numeric_cols
test_numeric_cols = ['etest_p_trans' if col == 'etest_p' else col for col in test_numeric_cols]
test_numeric_cols.remove('salary')

t_test = pd.DataFrame(columns=['coef','se','tvalue','pvalue'])
for col in test_numeric_cols:
    X=np.array(mydat['status'].map({'Placed':0, 'Not Placed':1}))
    X = sm.add_constant(X)
    Y = np.array(mydat[col])
    temp = sm.OLS(Y,X).fit()

    df_temp = pd.DataFrame({'coef':[temp.params[1]],
        'se':[temp.bse[1]],
        'tvalue':[temp.tvalues[1]],
        'pvalue':[temp.pvalues[1]]})
    t_test = t_test.append(df_temp, ignore_index=True)

t_test.index = test_numeric_cols
t_test = t_test.apply(lambda x: round(x,2) , axis=0)
print(t_test)

# chi-square independence test
outcome = 'status'
cols = categorical_cols[categorical_cols!='status']
chiind_test = pd.DataFrame(columns=['chi2','pvalue'])
for i in range(len(cols)):
    col = cols[i]
    contingency_table = pd.crosstab(index=mydat[outcome], columns=mydat[col])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table, correction=False)
    chiind_test = chiind_test.append(pd.DataFrame({'chi2':[chi2], 'pvalue':[p]}))

chiind_test.index = cols
chiind_test = chiind_test.apply(lambda x: round(x,2), axis=0)
print(chiind_test)
import pandas as pd
import numpy as np
import math
import os
from joblib import dump, load


wd = os.path.abspath(os.getcwd())
mydat = pd.read_csv(wd + '/Placement_Data_Transformed.csv')

"""
1. Data Preparing
"""
from sklearn import model_selection
categorical_predictor = ['gender', 'specialisation', 'workex']
numeric_predictor = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p_trans']
response = 'status'

X_dummy = pd.get_dummies(mydat.loc[:, categorical_predictor], drop_first=True)
X = pd.concat([X_dummy, mydat.loc[:, numeric_predictor]], axis=1)
Y = mydat.loc[:, response].map({'Placed':0, 'Not Placed':1})
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(X, Y, test_size=0.2, random_state=101)
"""
2. Model Selection

- Compare models including
    1. Logistic regression
    2. Support Vector Machine
    3. Naive Bayes
    4. Decision Tree

- Hyperparameters are obtained by 10-fold CV being the one
    * with the best 'accuracy' using 'balanced' scoring metrics *

- Select models based on metrics including
    1. ROC AUC
    2. F1 score
    3. PR curve
    4. Sensitivity
    5. Accuracy

"""
from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

n_cv = 10
n_Cs = 10
random_state = 101
max_iter = 300
scoring = 'accuracy'
class_weight = 'balanced'
cv_splitter = model_selection.KFold(n_splits=n_cv, shuffle=True, random_state=random_state)


def PerformanceCompare(class_true, class_predict, method, score_predict, exist_score=True):
    if exist_score:
        roc_auc = metrics.roc_auc_score(y_true=class_true, y_score=score_predict)
        pr_score = metrics.average_precision_score(y_true=class_true, y_score=score_predict)
    else:
        roc_auc = float('nan')
        pr_score = float('nan')
    f1_score = metrics.f1_score(y_true=class_true, y_pred=class_predict)
    sensitivity = metrics.recall_score(y_true=class_true, y_pred=class_predict)
    accuracy = metrics.accuracy_score(y_true=class_true, y_pred=class_predict)

    output = pd.DataFrame(np.array([roc_auc, pr_score, f1_score, sensitivity, accuracy]),
        index=['ROC_AUC', 'PR_score', 'F1_score', 'sensitivity', 'accuracy'], columns=method)
    return(output)


# Logistic Regression
def LRPerformance(X, Y):
    param_grid = {
        'C': [10**uu for uu in np.linspace(-4, 4, n_Cs)]
    }

    lr_model = linear_model.LogisticRegression(random_state=random_state, class_weight=class_weight, max_iter=max_iter)
    lr_cv = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=cv_splitter, scoring=scoring).fit(X,Y)
    Y_predict = lr_cv.predict(X)
    score_predict = lr_cv.predict_proba(X)[:,1]
    best_estimator = lr_cv.best_estimator_

    performance = PerformanceCompare(class_true=Y, class_predict=Y_predict, score_predict=score_predict, method=['LR'])
    return({'performance': performance, 'best_estimator': best_estimator})


# Support Vector Machine
def SVMPerformance(X, Y):
    #penalty = 'l1'
    #loss = 'hinge'
    param_grid = {
        'C' : [10**uu for uu in np.linspace(-4, 4, n_Cs)]
    }
    svm_model = svm.SVC(random_state=random_state, class_weight=class_weight)
    svm_cv = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=cv_splitter, scoring=scoring).fit(X, Y)
    Y_predict = svm_cv.predict(X)
    score_predict = svm_cv.decision_function(X)
    best_estimator = svm_cv.best_estimator_

    performance = PerformanceCompare(class_true=Y, class_predict=Y_predict, score_predict=score_predict, method=['SVM'])
    return({'performance': performance, 'best_estimator': best_estimator})


# Naive Bayes
def NBPerformance(X, Y):
    categorical_cols = ['gender_M', 'specialisation_Mkt&HR', 'workex_Yes']
    numerical_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p_trans']

    cate_nb = naive_bayes.BernoulliNB().fit(X.loc[:, categorical_cols], Y)
    conti_nb = naive_bayes.GaussianNB().fit(X.loc[:, numerical_cols], Y)
    prob = cate_nb.predict_proba(X.loc[:, categorical_cols]) * conti_nb.predict_proba(X.loc[:, numerical_cols])
    Y_predict = list(np.argmax(prob, axis=1))
    score_predict = prob[:,1]

    performance = PerformanceCompare(class_true=Y, class_predict=Y_predict, score_predict=score_predict, method=['NB'])
    return({'performance': performance, 'best_estimator': [cate_nb, conti_nb]})


# Decision Tree
def DTPerformance(X, Y):
    param_grid = {
        'max_leaf_nodes': range(3,15),
        'ccp_alpha': np.linspace(0, 0.05, 5)
    }
    dt_model = DecisionTreeClassifier(random_state=random_state, class_weight=class_weight)
    dt_cv = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=cv_splitter, scoring=scoring).fit(X, Y)
    Y_predict = dt_cv.predict(X)
    score_predict = dt_cv.predict_proba(X)[:,1]
    best_estimator = dt_cv.best_estimator_

    performance = PerformanceCompare(class_true=Y, class_predict=Y_predict, score_predict=score_predict, method=['DT'])
    return({'performance': performance, 'best_estimator': best_estimator})


lr_pb = LRPerformance(train_X,train_Y)
svm_pb = SVMPerformance(train_X, train_Y)
nb_pb = NBPerformance(train_X, train_Y)
dt_pb = DTPerformance(train_X, train_Y)

print(lr_pb['best_estimator'])
performance_table = pd.concat([lr_pb['performance'],
    svm_pb['performance'],
    nb_pb['performance'],
    dt_pb['performance']], axis=1)
performance_table = round(performance_table, 3)
print(performance_table)



"""
3. Select Polynomial Features

- CV to select best feature set from polynomial of the basic predictors.
- L1 penalty to eliminate redundant variables.
"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

poly = PolynomialFeatures(2)
train_poly = poly.fit_transform(train_X)
poly_name = poly.get_feature_names(train_X.columns)

stay_id= [i for i, x in enumerate(poly_name) if x not in ['gender_M^2','specialisation_Mkt&HR^2','workex_Yes^2']]
train_poly = train_poly[:,stay_id]
poly_name = [poly_name[i] for i in stay_id]

max_iter = 10000
lr_model = linear_model.LogisticRegression(C=0.36, random_state=random_state, class_weight=class_weight, max_iter=max_iter, solver='liblinear')
rfecv = RFECV(estimator = lr_model, step=1, cv=StratifiedKFold(5), scoring='roc_auc').fit(train_poly, train_Y)

plt.figure()
plt.xlabel('Number of features selected')
plt.ylabel('ROC_AUC score')
plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
plt.show()

rfe = RFE(estimator = lr_model, step=1, n_features_to_select=21).fit(train_poly, train_Y)
poly_variables_rfe = [poly_name[id] for id, x in enumerate(rfe.support_) if x]

# Lasso Feature Selection
param_grid = {
    'C': [10**x for x in np.linspace(-4,4,n_Cs)]
}


lr_model = linear_model.LogisticRegression(penalty='l1', random_state=random_state, class_weight=class_weight, max_iter=max_iter, solver='liblinear')
#lr_cv = GridSearchCV(lr_model, param_grid=param_grid, scoring='roc_auc').fit(train_poly, train_Y)
#importance = np.abs(lr_cv.best_estimator_.coef_)[0]

threshold = 1e-5
sfm = SelectFromModel(lr_model, threshold=threshold).fit(train_poly, train_Y)


poly_variables_l1 = [poly_name[id] for id, x in enumerate(sfm.get_support()) if x]

feature_selection_output = pd.DataFrame({
    'Feature Name': poly_name,
    'RFE': rfe.support_*1,
    'L1': sfm.get_support()*1
})
feature_selection_output.to_csv(wd+'/feature_selection_output.csv', index=False)

print(feature_selection_output)

# Set the final version of FEATURE SET
stay_id = [id for id, x in enumerate(poly_name) if x in poly_variables_rfe]
train_X = train_poly[:, stay_id]

test_poly = poly.transform(test_X)
stay_id = [id for id, x in enumerate(poly.get_feature_names(test_X.columns)) if x in poly_variables_rfe]
test_X = test_poly[:, stay_id]

"""
4. Hyperparameter Tuning

- Tune to the best hyperparameter
"""
param_grid = {
    'C': np.linspace(1e-5, 1e5, 20),
    'class_weight': ['balanced']
}

lr_model = linear_model.LogisticRegression(random_state=random_state, max_iter=max_iter, solver='liblinear')
lr_cv = GridSearchCV(lr_model, param_grid=param_grid, scoring='roc_auc').fit(train_X, train_Y)

dump(lr_cv, wd+'/final_model.joblib')



# Model Performance
Y_predict = lr_cv.predict(test_X)
Y_score = lr_cv.predict_proba(test_X)[:,1]

confusion_matrix_train = metrics.confusion_matrix(train_Y, lr_cv.predict(train_X))
confusion_matrix_train = pd.DataFrame(confusion_matrix_train, index=['Neg','Pos'], columns=['Pred_Neg','Pred_Pos'])

confusion_matrix_test = metrics.confusion_matrix(y_true=test_Y, y_pred=Y_predict)
#tn, fp, fn, tp = confusion_matrix.ravel()
confusion_matrix_test = pd.DataFrame(confusion_matrix_test, index=['Neg','Pos'], columns=['Pred_Neg','Pred_Pos'])

roc_auc = metrics.roc_auc_score(y_true=test_Y, y_score=Y_score)
average_precision = metrics.average_precision_score(y_true=test_Y, y_score=Y_score)
sensitivity = metrics.recall_score(y_true=test_Y, y_pred=Y_predict)
specificity = metrics.recall_score(y_true=test_Y, y_pred=Y_predict, pos_label=0)
accuracy = metrics.accuracy_score(y_true=test_Y, y_pred=Y_predict)
precision = metrics.precision_score(y_true=test_Y, y_pred=Y_predict)
precision_neg = metrics.precision_score(y_true=test_Y, y_pred=Y_predict, pos_label=0)



metrics.plot_roc_curve(estimator=lr_cv.best_estimator_, X=test_X, y=test_Y)
plt.title('ROC Curve')
plt.show()

metrics.plot_precision_recall_curve(estimator=lr_cv.best_estimator_, X=test_X, y=test_Y)
plt.title('Precision-Recall Curve')
plt.show()

print(lr_cv.best_estimator_)
print("best roc_auc:",lr_cv.best_score_)
print('confusion_matrix_train:\n',confusion_matrix_train,'\n')

print('confusion_matrix_test:\n', confusion_matrix_test)
print(pd.Series({
    'roc_auc': roc_auc, 'average_precision':average_precision, 
    'accuracy':accuracy, 'sensitivity':sensitivity, 'specificity':specificity,
    'precision':precision, 'precision_neg':precision_neg
}))
# Feature Impact  
pd_coef = pd.DataFrame({'Feature Name': poly_variables_rfe, 'Coefficient': lr_cv.best_estimator_.coef_[0,:]})


feature_selection_output = pd.merge(feature_selection_output, pd_coef, left_on='Feature Name', right_on='Feature Name', how='outer')
print(feature_selection_output)

default_value = X.loc[:,['ssc_p','hsc_p','degree_p','etest_p_trans']].apply(lambda x: sum(x)/len(x), axis=0)
default_value = round(default_value, 0).tolist()

def makeSample(moving_variable=None, fixed_variable=None, default_value=default_value):
    """
    To generate sample with some variable values being fixed & ONE variable value moving. 
    The sample would be transformed to polynomial and be selected as the feature set we've selected.
    
    * moving_variable is a dict with only one element. The key is column name. The value could be a number or list containing multiple numbers.
    * fixed_variable is a dict containing one or multiple elements. The keys are column names. The values could only be a number.
    * The default values of the categoricals are 0, and which of the numericals are 50.
    """
    
    columns0=['gender_M','specialisation_Mkt&HR','workex_Yes','ssc_p','hsc_p','degree_p','etest_p_trans']
    values0 = [0,0,0]+default_value
    
    if fixed_variable is None:
        X = values0
    else:
        change_names = [x for x in fixed_variable.keys()] # would be a list even if theres only one element
        change_values = [x for x in fixed_variable.values()] # would be a list even if theres only one element
        change_id = [id for cn in change_names for id, x in enumerate(columns0) if x == cn]
        
        X = values0
        for ind in range(len(change_names)):
            X[change_id[ind]] = change_values[ind]
    
    
    if moving_variable is None:
        N=1
        moving_name = [None]
        moving_value = []
    else: 
        moving_name = [x for x in moving_variable.keys()][0]
        moving_value = [x for x in moving_variable.values()][0] # would have been a double list if not choosing the first element
        N = len(moving_value)
    
    
    X = np.array(X*N)
    X = np.reshape(X, (N,7))
    moving_id = [id for id, x in enumerate(columns0) if x in moving_name][0] # would have been a list if not choosing the first element
    X[:,moving_id] = moving_value
    X = pd.DataFrame(X, columns=columns0)
    
    X_poly = poly.fit_transform(X)
    poly_name = poly.get_feature_names(X.columns)
    
    stay_id= [i for i, x in enumerate(poly_name) if x in poly_variables_rfe]
    X_poly = X_poly[:,stay_id]
        
    
    return(X_poly)


def makeComparisonSample(categoricals, numerical):
    output = []
    name = []
    for ind1 in [0,1]:
        name1 = "".join([categoricals[0], str(ind1)])
        for ind2 in [0,1]:
            name2 = "".join([categoricals[1], str(ind2)])
            for ind3 in [0,1]:
                name3 = "".join([categoricals[2], str(ind3)])
                name_temp = " ".join([name1, name2, name3])
                temp = makeSample(moving_variable={numerical:range(101)}, fixed_variable={categoricals[0]:ind1, categoricals[1]:ind2, categoricals[2]:ind3})
                output.append(temp)
                name.append(name_temp)
    return(output, name) # output is a list made of 4 arrays

        
def plotComparison(labels, categoricals, suptitle):
    
    # Show the probablity predicted of different 
    numerical_cols = ['ssc_p','hsc_p','degree_p','etest_p_trans']
    # gender & specialization & numericals
    s = []
    s_name = []
    linestyle = ['solid']*(len(labels)+1)
    count = 0
    n_row = 2
    n_col = 2
    fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(10,7))
    plt.subplots_adjust(hspace=0.7, wspace=0.3)
    for col in numerical_cols:
        s_temp, s_name_temp = makeComparisonSample(categoricals=categoricals, numerical=col)

        r_id = count//n_col
        c_id = count%n_col
        for ind in range(8):
            prob_temp = lr_cv.predict_proba(s_temp[ind])[:,0]

            axes[r_id, c_id].plot(range(101), prob_temp, linestyle=linestyle[ind], label=labels[ind])
        axes[r_id, c_id].set_ylim(-0.1,1.1)
        axes[r_id, c_id].set_yticks([0,.2,.4,.6,.8,1])
        axes[r_id, c_id].hlines(y=0.5, xmin=0, xmax=100, linestyles='dashed', colors='gray', linewidth=1)
        axes[r_id, c_id].set_ylabel('Probability')
        axes[r_id, c_id].set_xlabel(col)
        #axes[r_id, c_id].set_title('Prob. of being PLACED')
        s.append(s_temp)
        s_name.append(s_name_temp)
        count+=1
    
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'right')
    fig.suptitle(suptitle)
    return(fig, axes)



# Create Labels   
label_grid = {'Gender':['Female','Male'], 'Specialization':['Fin','HR'], 'WorkExperience':["Haven't Worked", "Have Worked"]}
grid = model_selection.ParameterGrid(label_grid)
labels = []
for g in grid:
    temp ="/".join(list(g.values()))
    labels.append(temp)

plotComparison(labels=labels, categoricals=['gender_M','specialisation_Mkt&HR', 'workex_Yes'], suptitle='Prob. of being PLACED')    
plt.show()


