!pip install aif360 rfpimp



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load data in pandas dataframe format

hr_data = pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

hr_data.head()
# How is data modeled (datatypes) in the dataset?

# Can also be viewed instead using Data->Summary tab

#hr_data.dtypes

hr_data.columns
# This gives description of only numeric fields (excludes NAN values)

# hr_data.describe()

# This gives description of all columns

hr_data.describe(include = 'all')
def print_missing_values(data):

    data_null = pd.DataFrame(data.isnull().sum(), columns = ['Count'])

    data_null = data_null[data_null['Count'] > 0].sort_values(by='Count', ascending=False)

    data_null = data_null/len(data) * 100



    if data_null.empty:

        print('No missing values in dataset')

    else:

        ax = sns.barplot(x=data_null.index, y=data_null['Count'])

        ax.set_title('Columns with at least one missing value')

        ax.set_ylabel('%age of dataset')

        print('Total missing values: ', data.isnull().sum().sum())
# Check missing values

print_missing_values(hr_data)
#df = hr_data[['HourlyRate', 'DailyRate', 'MonthlyRate', 'MonthlyIncome']]

df_dph = (hr_data['DailyRate']/hr_data['HourlyRate']).to_frame()

df_dph.columns=['DailyPerHourRate']

df_mpd = (hr_data['MonthlyRate']/hr_data['DailyRate']).to_frame()

df_mpd.columns = ['MonthlyPerDayRate']



df = pd.concat([hr_data[['HourlyRate', 'DailyRate', 'MonthlyRate', 'MonthlyIncome']], df_dph, df_mpd], axis=1)

df.head()
# Remove any erroneous data

hr_data.drop(columns=['HourlyRate', 'DailyRate', 'MonthlyRate'], inplace=True)
# Check EmployeeNumber

%matplotlib inline



plt.plot(hr_data['EmployeeNumber'])
# Drop features

hr_data.drop(columns=['EmployeeNumber'], inplace=True)

hr_data.head()
# Copy categorical data

hr_data_cat = hr_data.select_dtypes(exclude=np.number)

hr_data_cat.head()
# Replace Yes and No in Attrition with 1 and 0

#num_val = {'Yes': 1, 'No': 0}

#hr_data_cat['Attrition'] = hr_data_cat['Attrition'].apply(lambda x: num_val[x])

# Convert categorical variable into dummies (this is one-hot encoding)

#hr_data_cat = pd.get_dummies(hr_data_cat)

#hr_data_cat.head()

# OR, use scikit-learn label encoding for mapping

from sklearn import preprocessing

lab_enc = preprocessing.LabelEncoder()

# Deep-copy original data

hr_data_enc = hr_data.copy(deep=True)

for col in hr_data_cat.columns:

    hr_data_enc[col] = lab_enc.fit_transform(hr_data[col])

    le_name_mapping = dict(zip(lab_enc.classes_, lab_enc.transform(lab_enc.classes_)))

    print('Feature', col)

    print('mapping', le_name_mapping)

hr_data_enc.head()
# Visualization methods



# Plots distribution as bar and pie chart

# e.g. plot_bar_and_pie('YearsSinceLastPromotion', hr_data)

def plot_bar_and_pie(y_var, data):

    val = data[y_var]

    

    plt.style.use('seaborn-whitegrid')

    plt.rcParams.update({'font.size': 12})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    

    cnt = val.value_counts().sort_values(ascending=True)

    labels = cnt.index.values

    

    sizes = cnt.values

    colors = sns.color_palette('PuBu', len(labels))

    

    # Count plot rendered as barh

    ax1.barh(cnt.index.values, cnt.values, color=colors)

    ax1.set_title('Count plot of ' + y_var)

    

    # Percentage rendered as pie

    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=145)

    ax2.axis('equal')

    ax2.set_title('Distribution of ' + y_var)

    plt.show()

    

# Plots a histogram

# e.g. plot_hist(hr_data, 'YearsSinceLastPromotion', ['YearsWithCurrManager', 'YearsAtCompany'])

def plot_hist(data, col, y_columns):

    df = data.copy()

    fig, ax = plt.subplots(1, len(y_columns), figsize=(20, 6))

    

    for i in range(0, len(y_columns)):

        cnt = []

        y_col = y_columns[i]

        y_values = df[y_col].dropna().drop_duplicates().values

        for val in y_values:

            cnt += [df[df[y_col] == val][col].values]

        bins = df[col].nunique()

        

        if (len(y_columns) > 1):

            ax[i].hist(cnt, bins=bins, stacked=True)

            ax[i].legend(y_values, loc='upper right')

            ax[i].set_title('Histogram of ' + col + ' column by ' + y_col)

        else:

            ax.hist(cnt, bins=bins, stacked=True)

            ax.legend(y_values, loc='upper right')

            ax.set_title('Histogram of ' + col + ' column by ' + y_col)

    plt.show()
hr_data_enc['Attrition'].value_counts(normalize=True)
# Find all columns where data doesn't change, use .nunique(dropna=False) if we want to count NAs as separate value

uc_columns = hr_data_enc.columns[hr_data_enc.nunique() <= 1]

print('Columns which do not change: {}'.format(uc_columns))

# Remove columns which don't change at all

hr_data_enc = hr_data_enc.drop(columns=uc_columns)

hr_data_enc.head()
# Create a seaborn heatmap

%matplotlib inline



plt.figure(figsize=(10,10), dpi=100)

sns.heatmap(hr_data_enc.corr())
# Extract correlated variables for further analysis

corr_cols = ['Age', 'JobLevel', 'MonthlyIncome', 'TotalWorkingYears', 'MaritalStatus', 'StockOptionLevel']

filtered_data = hr_data_enc[corr_cols]



%matplotlib inline



plt.figure(figsize=(20,5), dpi=100)

sns.heatmap(filtered_data.corr())
plot_hist(hr_data, 'YearsSinceLastPromotion', ['Attrition'])



cnt_yes = hr_data.loc[hr_data['Attrition'] == 'Yes', ['YearsSinceLastPromotion']]['YearsSinceLastPromotion'].value_counts(sort=False)

cnt_no  = hr_data.loc[hr_data['Attrition'] == 'No' , ['YearsSinceLastPromotion']]['YearsSinceLastPromotion'].value_counts(sort=False)



years_since_promo_attr_ratio = []

cnt_yes_cutoff_sum = 0

cnt_no_cutoff_sum = 0



for i in range(8):

    years_since_promo_attr_ratio.append(cnt_yes[i] / (cnt_yes[i] + cnt_no[i]))

for i in range(8, len(cnt_no)):

    if (i != 8) and (i != 12):

        cnt_yes_cutoff_sum += cnt_yes[i]

    cnt_no_cutoff_sum  += cnt_no[i]



years_since_promo_attr_ratio.append(cnt_yes_cutoff_sum / (cnt_yes_cutoff_sum + cnt_no_cutoff_sum))



x_data = np.array(range(9))

y_data = np.array(years_since_promo_attr_ratio)



# Use two subplots, left would be linear, right would be log

fig, ax = plt.subplots(1, 2, figsize=(20, 6))

bbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# Linear plot

curve_fit = np.polyfit(x_data, y_data, 1, full=True)

y = curve_fit[0][0] * x_data + curve_fit[0][1]

ax[0].plot(x_data, y_data, 'o')

ax[0].plot(x_data, y)

ax[0].set_xlabel('YearsSinceLastPromotion')

ax[0].set_ylabel('P(Attrition)')

ax[0].set_ylim(0, 1)

ax[0].set_title('y = {0:.4f}x + {0:.4f}'.format(curve_fit[0][0], curve_fit[0][1]))

ax[0].text(0.05, 0.95, 'Error {0:.4f}'.format(curve_fit[1][0]), transform=ax[0].transAxes, fontsize=12, verticalalignment='top', bbox=bbox_props)

# Log plot

log_x_data = np.log2(x_data + 1)

curve_fit = np.polyfit(log_x_data, y_data, 1, full=True)

y = curve_fit[0][0] * log_x_data + curve_fit[0][1]

ax[1].plot(log_x_data, y_data, 'o')

ax[1].plot(log_x_data, y)

ax[1].set_xlabel('log(YearsSinceLastPromotion)')

ax[1].set_ylabel('P(Attrition)')

ax[1].set_ylim(0, 1)

ax[1].set_title('y = {0:.4f} log2(x + 1) + {0:.4f}'.format(curve_fit[0][0], curve_fit[0][1]))

ax[1].text(0.05, 0.95, 'Error {0:.4f}'.format(curve_fit[1][0]), transform=ax[1].transAxes, fontsize=12, verticalalignment='top', bbox=bbox_props)
# Set input_data for modeling after data is ready

input_data = hr_data_enc

input_data.head()
target = input_data['Attrition']

all_features = input_data.drop('Attrition', axis = 1)



print('No of columns: {}'.format(len(all_features.columns)))

target.head()
sel_feature_cols = []

col_values = list(all_features.columns.values)

print(col_values)
from sklearn.feature_selection import mutual_info_classif





# Find top 10 features with maximum Mutual Information (dependent variables)

feature_scores = mutual_info_classif(all_features, target)

for score, fname in sorted(zip(feature_scores, col_values), reverse=True)[:10]:

    print(fname, score)

    sel_feature_cols.append(fname)
# Find top 10 features with maximum chi-square value

from sklearn.feature_selection import chi2

feature_scores = chi2(all_features, target)[0]

for score, fname in sorted(zip(feature_scores, col_values), reverse=True)[:10]:

    print(fname, score)

    sel_feature_cols.append(fname)
# Select features

print(np.unique(sel_feature_cols))

#features = all_features[np.unique(sel_feature_cols)]



features = all_features # Select all the features (just to test)

features.head()
from sklearn.model_selection import train_test_split



# Create the train / test split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.15, random_state=10)



print('Shape of features: ', features.shape)

print('Shape of Training input data: ', X_train.shape, '\tShape of Training target: ', y_train.shape)

print('Shape of Test     input data: ', X_test.shape,  '\tShape of Test     target: ', y_test.shape)
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_curve, auc



def get_model_performance(X_test, y_true, y_pred, probs):

    # Test the accuracy

    accuracy = accuracy_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    matrix = confusion_matrix(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    preds = probs[:, 1]

    fpr, tpr, threshold = roc_curve(y_true, preds)

    roc_auc = auc(fpr, tpr)

    return accuracy, recall, matrix, f1, fpr, tpr, roc_auc



def plot_model_performance(model, X_test, y_true):

    # Predict the results for test

    y_pred = model.predict(X_test)

    probs = model.predict_proba(X_test)

    accuracy, recall, matrix, f1, fpr, tpr, roc_auc = get_model_performance(X_test, y_true, y_pred, probs)

    print('Accuracy score: ', accuracy)

    print('Recall score:   ', recall)

    print('F1 score:       ', f1)

    

    fig = plt.figure(figsize=(15, 6))

    ax = fig.add_subplot(1, 2, 1)

    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='g')

    plt.title('Confusion Matrix')

    

    ax = fig.add_subplot(1, 2, 2)

    lw = 2

    plt.plot(fpr, tpr, color='orange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.grid(True)

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic Curve')

    plt.legend(loc='lower right')
from sklearn.ensemble import RandomForestClassifier



# Create the model and train

model = RandomForestClassifier()

model.fit(X_train, y_train)
plot_model_performance(model, X_test, y_test)
from aif360.datasets import StandardDataset

from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
# Print metadata for bias mitigation dataset

# Note: dataset needs to be of type StandardDataset

def bm_meta_data(dataset):

    print('Dataset shape: ', dataset.features.shape)

    print('Favorable label: ', dataset.favorable_label)

    print('Unfavorable label: ', dataset.unfavorable_label)

    print('Protected attributes: ', dataset.protected_attribute_names)

    print('Privileged protected attributes: ', dataset.privileged_protected_attributes)

    print('Unprivileged protected attributes: ', dataset.unprivileged_protected_attributes)

    print('Features: ', dataset.feature_names)
# Gender suspected to have bias

privileged_groups   = [{'Gender': 0}] # Female

unprivileged_groups = [{'Gender': 1}] # Male

favorable_label = 0

unfavorable_label = 1

bm_data_test = StandardDataset(input_data,

                               label_name='Attrition',

                               favorable_classes=[favorable_label], 

                               protected_attribute_names=['Gender'], 

                               privileged_classes=[[favorable_label]])

bm_meta_data(bm_data_test)
def fair_metrics(dataset, pred, pred_is_dataset=False):

    if pred_is_dataset:

        dataset_pred = pred

    else:

        dataset_pred = dataset.copy()

        dataset_pred.labels = pred

        

    cols = ['statistical_parity_difference', 'equal_opportunity_difference', 'average_abs_odds_difference', 'disparate_impact', 'theil_index']

    obj_fairness = [[0,0,0,1,0]]

    

    fair_metrics = pd.DataFrame(data=obj_fairness, index=['objective'], columns=cols)

    

    for attr in dataset_pred.protected_attribute_names:

        idx = dataset_pred.protected_attribute_names.index(attr)

        privileged_groups   = [{attr:dataset_pred.privileged_protected_attributes[idx][0]}]

        unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}]

        

        classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

        metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

        acc = classified_metric.accuracy()

        row = pd.DataFrame([[metric_pred.mean_difference(),

                             classified_metric.equal_opportunity_difference(),

                             classified_metric.average_abs_odds_difference(),

                             metric_pred.disparate_impact(),

                             classified_metric.theil_index()

                            ]], columns=cols, index=[attr])

        fair_metrics = fair_metrics.append(row)

    

    fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)

    return fair_metrics



def plot_fair_metrics(fair_metrics):

    fig, ax = plt.subplots(figsize=(20,4), ncols=5, nrows=1)

    

    plt.subplots_adjust(

        left = 0.125,

        bottom = 0.1,

        right = 0.9,

        top = 0.9,

        wspace = .5,

        hspace = 1.1

    )

    y_title_margin = 1.2

    

    plt.suptitle('Fairness metrics', y=1.09, fontsize=20)

    sns.set(style='dark')

    

    cols = fair_metrics.columns.values

    obj = fair_metrics.loc['objective']

    size_rect = [0.2,0.2,0.2,0.4,0.25]

    rect = [-0.1,-0.1,-0.1,0.8,0]

    bottom = [-1,-1,-1,0,0]

    top = [1,1,1,2,1]

    bound = [[-0.1,0.1],[-0.1,0.1],[-0.1,0.1],[0.8,1.2],[0,0.25]]

    

    print('Check bias metrics (model may be biased if even one of these metrics show a bias)')

    for attr in fair_metrics.index[1:len(fair_metrics)].values:

        check = [bound[i][0] < fair_metrics.loc[attr][i] < bound[i][1] for i in range(0,5)]

        print('Attribute: ' + attr + ', with default threshold, bias against unprivileged group detected in {} out of 5 metrics'.format(5 - sum(check)))

    

    for i in range(0, 5):

        plt.subplot(1, 5, i+1)

        ax = sns.barplot(x=fair_metrics.index[1:len(fair_metrics)], y=fair_metrics.iloc[1:len(fair_metrics)][cols[i]])

        for j in range(0, len(fair_metrics)-1):

            a, val = ax.patches[j], fair_metrics.iloc[j+1][cols[i]]

            marg = -0.2 if val < 0 else 0.1

            ax.text(a.get_x() + a.get_width()/5, a.get_y() + a.get_height() + marg, round(val, 3), fontsize=15, color='black')

        plt.ylim(bottom[i], top[i])

        plt.setp(ax.patches, linewidth=0)

        ax.add_patch(patches.Rectangle((-5, rect[i]), 10, size_rect[i], alpha=0.3, facecolor='green', linewidth=1, linestyle='solid'))

        plt.axhline(obj[i], color='black', alpha=0.3)

        plt.title(cols[i])

        ax.set_ylabel('')

        ax.set_xlabel('')



def get_fair_metrics_and_plot(data, model, plot=True, model_aif=False):

    pred = model.predict(data).labels if model_aif else model.predict(data.features)

    fair = fair_metrics(data, pred)

    

    if plot:

        plot_fair_metrics(fair)

        display(fair)

    

    return fair
# Check for bias

fair = get_fair_metrics_and_plot(bm_data_test, model)
# Check if data is unbalanced

target.value_counts(normalize=True)
from sklearn.ensemble import AdaBoostClassifier



# Max number of estimators at which boosting is terminated

estimator = [50, 100, 200, 300, 400, 500, 700, 1000, 1500, 2000]

for i in estimator:

    print('Results for {} estimators:'.format(i))

    cls = AdaBoostClassifier(n_estimators=i)

    cls.fit(X_train, y_train)

    plot_model_performance(cls, X_test, y_test)
# n_estimators=1000 appears to be the best

cls = AdaBoostClassifier(n_estimators=1000)

cls.fit(X_train, y_train)
plot_model_performance(cls, X_test, y_test)
# Adjust hyper-parameters



# Try learning_rate of 0.1

#cls = AdaBoostClassifier(n_estimators=1000, learning_rate = 0.1)

#cls.fit(X_train, y_train)

#plot_model_performance(cls, X_test, y_test)

# Worse than default



# Try Support vector classifier

#from sklearn.svm import SVC

#svc = SVC(probability=True, kernel='linear')

#cls = AdaBoostClassifier(n_estimators=1000, base_estimator=svc, learning_rate = 1.0)

#cls.fit(X_train, y_train)

#plot_model_performance(cls, X_test, y_test)

# Worse than default
from math import ceil, log10



def plot_importances(df_importance):

    plt.style.use('seaborn-whitegrid')

    plt.rcParams.update({'font.size': 12})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))



    cnt = df_importance[df_importance['Importance'] > 0]['Importance'].sort_values(ascending=True)

    labels = cnt.index.values

    colors = sns.color_palette('PuBu', len(labels))



    # Bar

    ax1.barh(labels, cnt.values, color=colors)



    # Donut

    # Pie will not complete if numbers are too small, and their sum < 1

    pie_sum = cnt.values.sum()

    if pie_sum < 1:

        factor = 10^(ceil(log10(pie_sum)))

        pie_values = (cnt * factor).values

    else:

        pie_values = cnt.values

    ax2.pie(pie_values, labels=labels, colors=colors, textprops={'color': 'black'}, autopct='%1.0f%%', startangle=145)

    center_circle = plt.Circle((0,0), 0.7, fc='white')

    fig.gca().add_artist(center_circle)



    ax2.axis('equal')



    plt.tight_layout()

    plt.show()
from rfpimp import importances, dropcol_importances, oob_dropcol_importances, plot_corr_heatmap



imp = pd.DataFrame()

imp['Feature'] = X_train.columns

imp['Importance'] = cls.feature_importances_

imp = imp.sort_values('Importance', ascending=False)

imp = imp.set_index('Feature')

plot_importances(imp)
# Permutation importance

imp = importances(cls, X_test, y_test)

plot_importances(imp)
# Drop column importance

imp = dropcol_importances(cls, X_train, y_train, X_test, y_test)

plot_importances(imp)
# Combine RelationshipSatisfaction with JobSatisfaction

JobAndRelationshipSatisfaction = np.sum(imp.loc[['RelationshipSatisfaction', 'JobSatisfaction']])

JobAndRelationshipSatisfaction.rename('JobAndRelationshipSatisfaction', inplace = True)

p1_feat = imp.loc[['JobInvolvement', 'YearsSinceLastPromotion']].append(JobAndRelationshipSatisfaction)

print('Scaled Weight Values:\n{}'.format(p1_feat / p1_feat.sum()))