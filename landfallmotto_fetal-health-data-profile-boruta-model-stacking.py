# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from collections import Counter
from yellowbrick.classifier import ROCAUC
from yellowbrick.features import Rank1D, Rank2D
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, cross_validate, train_test_split, KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")
!pip install autoviz
data=pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')
sns.countplot(data.fetal_health)
cols=data.columns
print(cols)
data.isnull().any()
from pandas_profiling import ProfileReport
import pandas_profiling as pdp

#https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/advanced_usage.html
    
profile = ProfileReport(data, title='Pandas Profiling Report', minimal=True,progress_bar=False,      
    missing_diagrams={
          'heatmap': False,
          'dendrogram': False,
      } )
profile

'''
from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()



filename = ""
sep = ","
target='fetal_health'
dft = AV.AutoViz(
    filename,
    sep,
    target,
    data,
    header=0,
    verbose=0,
    lowess=False,
    chart_format="svg",
    max_rows_analyzed=150000,
    max_cols_analyzed=30,
)
'''
features = ['baseline value', 'accelerations', 'fetal_movement',
       'uterine_contractions', 'light_decelerations', 'severe_decelerations',
       'prolongued_decelerations', 'abnormal_short_term_variability',
       'mean_value_of_short_term_variability',
       'percentage_of_time_with_abnormal_long_term_variability',
       'mean_value_of_long_term_variability', 'histogram_width',
       'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
       'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
       'histogram_median', 'histogram_variance', 'histogram_tendency']

X = pd.DataFrame(data=data, columns=features)
y = pd.DataFrame(data=data, columns=['fetal_health'])
y = y.astype(int)
# to make labels start from 0 to n_classes, otherwise I couldn't manage to run xgb with labels starting from 1 to n_classes 😟😟😟
y = y-1 
X.head()

from scipy.stats import probplot,skew

for i in cols:
    fig, axes = plt.subplots(1, 3, figsize=(20,4))
    sns.distplot(data[i],kde=False, ax=axes[0])
    sns.boxplot(data[i], ax=axes[1])
    probplot(data[i], plot=axes[2])
    skew_val=round(data[i].skew(), 1)
    axes[1].set_yticklabels([])
    axes[1].set_yticks([])
    axes[0].set_title(i + " | Distplot")
    axes[1].set_title(i + " | Boxplot")
    axes[2].set_title(i + " | Probability Plot - Skew: "+str(skew_val))
    plt.show()
def correlation_heatmap(train):
    correlations = train.corr()
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
                )
    plt.show()

correlation_heatmap(data)
# 30% test and 70% train data as mentioned by dataset Author
# in the task https://www.kaggle.com/andrewmvd/fetal-health-classification/tasks?taskId=2410
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.30, stratify=y)

X_train.shape, y_train.shape, X_test.shape, y_test.shape,
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test)
params = {
    'max_depth': 6,
    'objective': 'multi:softmax',
    'num_class': 3,
    # Set number of GPUs if available   
    'n_gpus': 0
}
bst = xgb.train(params, dtrain)
pred = bst.predict(dtest)
Counter(pred)
print(classification_report(y_test, pred))
cm = confusion_matrix(y_test, pred)
cm
def plot_confusion_matrix(cm, classes, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)

plot_confusion_matrix(cm, ['Normal', 'Suspect', 'Pathological'])
f1_score(y_test, pred, average=None)
plot_importance(bst,importance_type='weight')
pyplot.show()
import shap
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X_test)

classes=['Normal', 'Suspect', 'Pathological']

shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=classes)
rf=RandomForestClassifier(random_state=55,class_weight='balanced_subsample')
lr=LogisticRegression(random_state=55, multi_class='multinomial')
sv = SVC(probability=True,random_state=55)
logreg = LogisticRegression(n_jobs=-1, solver='newton-cg',random_state=55) 
gb = GradientBoostingClassifier(random_state=55)
gnb = GaussianNB()
xgb= XGBClassifier(random_state=55)
models=[rf, lr, sv, logreg, gb, gnb, xgb]
cv = StratifiedKFold(10, shuffle=True, random_state=42)
model_results = pd.DataFrame()
row_number = 0
results = []
names = []

for ml in models:
    model_name=ml.__class__.__name__
    cv_results = cross_validate(ml, X_train, y_train, cv=cv, scoring='f1_macro', return_train_score=True, n_jobs=-1 )
    model_results.loc[row_number,'Model Name']=model_name
    model_results.loc[row_number, 'Train Accuracy Mean']=cv_results['train_score'].mean()
    model_results.loc[row_number, 'Test Accuracy Mean']=cv_results['test_score'].mean()
    model_results.loc[row_number, 'Fit Time Mean']=cv_results['fit_time'].mean()
    results.append(cv_results)
    names.append(model_name)
    
    row_number+=1
cv_results_array = []
for tt in results:
    cv_results_array.append(tt['test_score'])

fig = plt.figure(figsize=(18, 6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(cv_results_array)
ax.set_xticklabels(names)
plt.show()
display(model_results.style.background_gradient(cmap='summer_r'))
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', RandomForestClassifier(random_state=55,class_weight='balanced_subsample')),
    ('gb', GradientBoostingClassifier(random_state=55),
    ('xgb',XGBClassifier(random_state=55)), 
    #('lr',LogisticRegression(random_state=55,multi_class='multinomial'))
    )
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=RandomForestClassifier(random_state=55), cv=5
)

clf.fit(X_train, y_train).score(X_test, y_test)
pred = clf.predict(X_test)
cm = confusion_matrix(y_test, pred)
cm
plot_confusion_matrix(cm, ['Normal', 'Suspect', 'Pathological'])
print(classification_report(y_test, pred))
f1_score(y_test, pred, average=None)
from sklearn.utils import class_weight
class_weights = list(class_weight.compute_class_weight('balanced',
                                             np.unique(y_train['fetal_health']),
                                             y_train['fetal_health']))

w_array = np.ones(y_train.shape[0], dtype = 'float')
for i, val in enumerate(y_train['fetal_health']):
    w_array[i] = class_weights[val-1]
sns.countplot(w_array)
xgb= XGBClassifier(random_state=55)
xgb.fit(X_train, y_train, sample_weight=w_array)
pred = xgb.predict(X_test)
cm = confusion_matrix(y_test, pred)
cm
plot_confusion_matrix(cm, ['Normal', 'Suspect', 'Pathological'])
print(classification_report(y_test, pred))
f1_score(y_test, pred, average=None)
!pip install BorutaShap
from BorutaShap import BorutaShap


model = XGBClassifier(random_state=55)
Feature_Selector = BorutaShap(model=model, importance_measure='shap', classification=True)


Feature_Selector.fit(X=X_train, y=y_train.values, n_trials=100, random_state=0)
Feature_Selector.plot(which_features='all',X_size=14, figsize=(18,8),y_scale='log')
# Returns a subset of the original data with the selected features
subset = Feature_Selector.Subset()
subset.head()
xgb = XGBClassifier(random_state=55)
xgb.fit(X_train[subset.columns],y_train)

pred = xgb.predict(X_test[subset.columns])
cm = confusion_matrix(y_test, pred)
cm
plot_confusion_matrix(cm, ['Normal', 'Suspect', 'Pathological'])
print(classification_report(y_test, pred))
f1_score(y_test, pred, average=None)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

X_train_scaled_df=pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled_df=pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
X_train_scaled_df.head()
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, accuracy_score

xgb = XGBClassifier(random_state=55, nthread=-1)

params = {
            'n_estimators' : [100, 150, 200, 250],
            'max_depth': [10, 15],
        }
        

scorers = {
            'f1_score':make_scorer(f1_score,average='macro'),
            'f1_score_weighted':make_scorer(f1_score,average='weighted'),
            'precision_score': make_scorer(precision_score,average='macro'),
            'recall_score': make_scorer(recall_score,average='weighted'),
            'accuracy_score': make_scorer(accuracy_score)
          }

skf = StratifiedKFold(n_splits=2)

grid = GridSearchCV(xgb, 
                    param_grid = params, 
                    scoring = scorers, 
                    n_jobs = -1, 
                    cv = skf.split(X_train_scaled_df, y_train),
                    refit = "f1_score_weighted")

grid.fit(X_train_scaled_df, y_train)
best_params= grid.best_params_
best_model = grid.best_estimator_
best_params
xgb = XGBClassifier(random_state=55,n_estimators=200,max_depth=10)
xgb.fit(X_train_scaled_df,y_train)
pred = xgb.predict(X_test_scaled_df)
cm = confusion_matrix(y_test, pred)
cm
plot_confusion_matrix(cm, ['Normal', 'Suspect', 'Pathological'])
print(classification_report(y_test, pred))
f1_score(y_test, pred, average=None)