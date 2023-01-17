import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn import preprocessing
from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn import discriminant_analysis
from sklearn import model_selection
from xgboost.sklearn import XGBClassifier 


from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/voice-disorder/data after extraction.csv')
df.head()
df.shape
sns.heatmap(df.isna(),yticklabels=False, cbar=False, cmap='viridis')
data = df
data = df.drop(['status'], axis=1)
label = df['status']
sns.countplot('status', data=df)
plt.xlabel('status')
plt.title('Class - Count Graph')
plt.show()
plt.figure(figsize=(30,20))
for j,i in enumerate(data.columns.values):
    plt.subplot(4,5,j+1)
    plt.plot(data[i].values)
    plt.title(f'{i}')
data.drop(['name', 'PPE'], axis=1, inplace=True)
plt.figure(figsize=(20,10))
for j,i in enumerate(data.columns.values):
    plt.subplot(3,5,j+1)
    plt.plot(data[i].values)
    plt.title(f'{i}')
data.hist(figsize=(20,10),)
plt.plot()
data.plot(kind='kde', subplots=True, figsize=(20,10),
              layout=(-1, 4), sharex=False)
plt.plot()
data.plot(kind='box', subplots=True, figsize=(20,10),
              layout=(-1, 4), sharex=False)
plt.plot()
desc = data.describe().transpose()

desc_copy = desc.reset_index(drop=True).drop('count', axis=1)

desc_copy.plot(kind='line', subplots=True, figsize=(16,8),
              layout=(-1, 4), sharex=False)
plt.plot()
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
standerdise_data = scaler.fit_transform(data)
standerdise_data = pd.DataFrame(standerdise_data, columns=data.columns)
standerdise_data
standerdise_desc = standerdise_data.describe().transpose()
standerdise_desc_copy = standerdise_desc.reset_index(drop=True).drop('count', axis=1)
standerdise_desc_copy
standerdise_desc_copy.plot(kind='line', subplots=True, figsize=(16,8),
              layout=(-1, 4), sharex=False)
plt.plot()
X = standerdise_data
y = label
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.5)
clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
clf.fit(train_X,train_y)
clf.score(train_X,train_y)
clf.score(test_X,test_y)
imp_score = pd.DataFrame(clf.feature_importances_, columns=['Importance Score'])
features = pd.DataFrame(data.columns, columns=['Features'])
feature_imp = pd.concat([features,imp_score], axis=1)
feature_imp = feature_imp.sort_values(by='Importance Score', ascending=False)
sns.barplot(x=feature_imp['Importance Score'], y=feature_imp['Features'])
plt.show()
reduced_X = X[feature_imp.Features[:5]]
reduced_X.head()
plt.figure(figsize=(6,20))
for j,i in enumerate(reduced_X.columns.values):
    plt.subplot(5,1,j+1)
    plt.plot(reduced_X[i].values)
    plt.title(f'{i}')
train_X, test_X, train_y, test_y = train_test_split(reduced_X,y,test_size=0.5)
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    
    #SVM
    svm.SVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost
    XGBClassifier()
    ]
MLA_compare = []

#index through MLA and save performance to table
for alg in MLA:
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
   # cv_results = model_selection.cross_validate(alg, X_train, y_train)
    alg.fit(train_X, train_y)
    y_pred=alg.predict(test_X)
    score=accuracy_score(test_y, y_pred)
    MLA_compare.append((str(alg).split('(')[0],score))
MLA_compare = sorted(MLA_compare, key=lambda x: x[1], reverse=True)
sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.xticks(rotation=90)
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=(lambda x:[i[0] for i in x])(MLA_compare), y=(lambda x:[i[1] for i in x])(MLA_compare))
plt.show()
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
train_X, test_X, train_y, test_y = train_test_split(reduced_X,y,test_size=0.5)
a =['ensemble.AdaBoostClassifier',
    'ensemble.BaggingClassifier',
    'ensemble.ExtraTreesClassifier',
    'ensemble.GradientBoostingClassifier',
    'ensemble.RandomForestClassifier',

    #Gaussian Processes
    'gaussian_process.GaussianProcessClassifier',
    
    #GLM
    'linear_model.LogisticRegressionCV',
    'linear_model.PassiveAggressiveClassifier',
    'linear_model.RidgeClassifierCV',
    'linear_model.SGDClassifier',
    'linear_model.Perceptron',
    
    #Navies Bayes
    'naive_bayes.BernoulliNB',
    'naive_bayes.GaussianNB',
    
    #Nearest Neighbor
    
    #SVM
    'svm.SVC',
    'svm.LinearSVC',
    
    #Trees    
    'tree.DecisionTreeClassifier',
    'tree.ExtraTreeClassifier',
    
    #Discriminant Analysis
    'discriminant_analysis.LinearDiscriminantAnalysis',
    'discriminant_analysis.QuadraticDiscriminantAnalysis',

    
    #xgboost
    'XGBClassifier',
    
    #Hybrid Model
    ]
i=0
for alg in MLA:   # applying smote in different algorithms
    print(a[i])
    k_values = [1, 2, 3, 4, 5, 6, 7]
    for k in k_values:
           # define pipeline
        model = alg
        over = SMOTE(sampling_strategy=0.67, k_neighbors=k)
        under = RandomUnderSampler(sampling_strategy=0.84)
        steps = [('over', over), ('under', under), ('model', model)]
    
        pipeline = Pipeline(steps=steps)
            # evaluate pipeline
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(pipeline, reduced_X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
        score = mean(scores)
        print('> k=%d, Mean ROC AUC: %.3f' % (k, score))
    i=i+1
    
        
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='auto')

train_X, test_X, train_y, test_y = train_test_split(reduced_X,y,test_size=0.5)
X_train_smote, y_train_smote = smote.fit_sample(train_X,train_y)

from collections import Counter
print("Before SMOTE :" , Counter(train_y))
print("After SMOTE :" , Counter(y_train_smote))
k_values = [1, 2, 3, 4, 5, 6, 7]
for k in k_values:
    model = ensemble.GradientBoostingClassifier()
    over = SMOTE(sampling_strategy=0.67, k_neighbors=k)
    under = RandomUnderSampler(sampling_strategy=0.84)
    steps = [('over', over), ('under', under), ('model', model)]
    
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, reduced_X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    score = mean(scores)
    print('> k=%d, Mean ROC AUC: %.3f' % (k, score))
