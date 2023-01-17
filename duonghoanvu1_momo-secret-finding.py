# Data Processing
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

# Data Visualizing
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from IPython.display import display, HTML

# Data Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC, NuSVC

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Data Validation
from sklearn import metrics

# Math
from scipy import stats  # Computing the t and p values using scipy 
from statsmodels.stats import weightstats 
import math
from scipy.stats import norm

# Warning Removal
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
df = pd.read_csv('../input/Momo_Secret_Finding.csv')

df.head()
zero_count = (df.isnull()).sum() # (df1 == 0).sum()
zero_count_df = pd.DataFrame(zero_count)
zero_count_df.drop('STATUS', axis=0, inplace=True)
zero_count_df.columns = ['Count_Missing_Value']

# https://stackoverflow.com/questions/31859285/rotate-tick-labels-for-seaborn-barplot/60530167#60530167
sns.set(style='whitegrid')
plt.figure(figsize=(13,8))
sns.barplot(x=zero_count_df.index, y=zero_count_df['Count_Missing_Value'])
plt.xticks(rotation=70)
cats = ['CITY', 'AGENT', 'STATUS', 'SHOP_ID']

def plotFrequency(cats):
    #"A plot for visualize categorical data, showing both absolute and relative frequencies"
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()

    for ax, cat in zip(axes, cats):
        total = float(len(df[cat]))
        sns.countplot(df[cat], palette='plasma', ax=ax)

        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 10,
                    '{:1.2f}%'.format((height / total) * 100),
                    ha="center")

        plt.ylabel('Count', fontsize=15, weight='bold')
plotFrequency(cats)
def plotStatus(cats):
    #"A plot for visualize categorical data, showing both absolute and relative frequencies"
    fig, axes = plt.subplots(2, 2, figsize=(25, 20))
    axes = axes.flatten()

    for ax, cat in zip(axes, cats):
        total = float(len(df[cat]))
        sns.countplot(df[cat], palette='plasma',hue=df['STATUS'], ax=ax)
        
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 10,
                    '{:1.2f}%'.format((height / total) * 100),
                    ha="center")
        
        
        ax.legend(title='STATUS?',loc='upper right',labels=['Churn', 'Retain'])
        
        plt.ylabel('Count', fontsize=15, weight='bold')
        display
plotStatus(cats)
plt.figure(figsize=(13,5))
sns.scatterplot(x=df['TIME_TO_CONVERT'], y=df['STATUS'])
plt.figure(figsize=(13,5))
sns.scatterplot(x=df['TIME_TO_CONVERT'], y=df['CITY'], hue=df['STATUS'])
print(df['TRAN_ID'].duplicated().sum())
print(df['USER_ID'].duplicated().sum())
df = df[df['CITY'].notnull()]
# Time_to_convert can't be negative value as following the data description, 
# it is a period that the first payment made by user to the day that the user made the first offline payment
df = df[df['TIME_TO_CONVERT']>0]
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
axes = axes.flatten()

Churn_Convert_Time = df[df['STATUS']=='churn']['TIME_TO_CONVERT']
sns.distplot(Churn_Convert_Time, ax=axes[0]).set_title("Churn_Convert_Time")

Retain_Convert_Time = df[df['STATUS']=='retain']['TIME_TO_CONVERT']
sns.distplot(Retain_Convert_Time, ax=axes[1]).set_title("Retain_Convert_Time")
# Median value of TIME_TO_CONVERT for churn customers
print(df[df['STATUS']=='churn']['TIME_TO_CONVERT'].median())

# Median value of TIME_TO_CONVERT for retain customers
print(df[df['STATUS']=='retain']['TIME_TO_CONVERT'].median())
# Filling the missing values
df.loc[(df['STATUS'] == 'churn' ) & (df['TIME_TO_CONVERT'].isnull()), 'TIME_TO_CONVERT'] = 168
df.loc[(df['STATUS'] == 'retain' ) & (df['TIME_TO_CONVERT'].isnull()), 'TIME_TO_CONVERT'] = 144.5
def plot_3chart(feature):
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    # creating a grid of 3 cols and 3 rows.
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # Customizing the histogram grid.
    ax1 = fig.add_subplot(grid[0, :2])
    # Set the title.
    ax1.set_title('Histogram')
    # plot the histogram.
    sns.distplot(df[feature], hist=True, kde=True, fit=norm, color='#e74c3c', ax=ax1)
    ax1.legend(labels=['Normal', 'Actual'])
    
    # customizing the QQ_plot.
    ax2 = fig.add_subplot(grid[1, :2])
    # Set the title.
    ax2.set_title('Probability Plot')
    # Plotting the QQ_Plot.
    stats.probplot(df[feature].fillna(np.median(df.loc[:, feature])), plot=ax2)
    #ax2.get_lines()[0].set_markerfacecolor('#e74c3c')
    ax2.get_lines()[0].set_markersize(12.0)
    
     # Customizing the Box Plot.
    ax3 = fig.add_subplot(grid[:, 2])
    # Set title.
    ax3.set_title('Box Plot')
    # Plotting the box plot.
    sns.boxplot(df[feature], orient='v', color='#e74c3c', ax=ax3)
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=24))

    plt.suptitle(f'{feature}', fontsize=24)
plot_3chart('TIME_TO_CONVERT')
df['TIME_TO_CONVERT'] = np.log1p(df['TIME_TO_CONVERT'])

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
axes = axes.flatten()

sns.distplot(df['TIME_TO_CONVERT'], ax=axes[0]).set_title("Churn_Convert_Time")

#Get also the QQ-plot
res = stats.probplot(df['TIME_TO_CONVERT'], plot=axes[1])
plt.show()
df.drop(['TRAN_ID', 'USER_ID'], axis=1, inplace=True)
df['SHOP_ID'] = df['SHOP_ID'].astype('str')
df = pd.get_dummies(df, drop_first=True)
df.head()
x = df.drop('STATUS_retain', axis=1)
y = df['STATUS_retain']

# Usually we need to bring all features to the same scale using StandardScaler or Normalizer or Boxcox. 
# But in this case, it won't yield better result
# x = StandardScaler().fit_transform(df.drop('STATUS_retain', axis=1))
# y = df['STATUS_retain']
cv = StratifiedKFold(10, shuffle=True, random_state=0)

def model_check(X, y, estimators, cv):
    model_table = pd.DataFrame()
    
    row_index = 0
    for est in estimators:

        MLA_name = est.__class__.__name__
        model_table.loc[row_index, 'Model Name'] = MLA_name
        #    model_table.loc[row_index, 'MLA Parameters'] = str(est.get_params())

        cv_results = cross_validate(est,
                                    X,
                                    y,
                                    cv=cv,
                                    scoring='accuracy',
                                    return_train_score=True,
                                   )

        model_table.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_score'].mean()
        model_table.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()
        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        row_index += 1

        model_table.sort_values(by=['Test Accuracy Mean'],
                            ascending=False,
                            inplace=True)

    return model_table
logreg = LogisticRegression(n_jobs=-1, solver='newton-cg')

knn = KNeighborsClassifier(n_neighbors=13)

gnb = GaussianNB()

linearSVC = LinearSVC()

RbfSVC = SVC()

dt = DecisionTreeClassifier(max_depth=10)

rf = RandomForestClassifier(random_state=0,n_jobs=-1,verbose=0)

adab = AdaBoostClassifier(random_state=0)

gb = GradientBoostingClassifier(random_state=0)

xgb = XGBClassifier(random_state=0)

lgbm = LGBMClassifier(random_state=0)

votingC = VotingClassifier(estimators=[("XGB", xgb), ("GB", gb), ("DecisionTree", dt),('LightGBM', lgbm)], 
                           voting='soft', n_jobs=4)
estimators = [logreg,knn,gnb,linearSVC,RbfSVC,dt,rf,gb,xgb,lgbm,votingC]
raw_models = model_check(x, y, estimators, cv)
display(raw_models.style.background_gradient(cmap='summer_r'))
xgb = XGBClassifier(random_state=0).fit(x, y)
gb = GradientBoostingClassifier(random_state=0).fit(x, y)
dt = DecisionTreeClassifier(max_depth=10).fit(x, y)
rf = RandomForestClassifier(random_state=0,n_jobs=-1,verbose=0).fit(x, y)
nrows = 2
ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(15,15))

names_classifiers = [("XGB", xgb), ("GB", gb), ('DT', dt), ('RF', rf)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=x.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=gb, dataset=x, model_features=x.columns, feature='TIME_TO_CONVERT')

# plot it
pdp.pdp_plot(pdp_goals, 'TIME_TO_CONVERT')
plt.show()
row_to_show = 1
data_for_prediction = x.iloc[row_to_show]

import shap
# Create object that can calculate shap values
explainer = shap.TreeExplainer(dt)
# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
