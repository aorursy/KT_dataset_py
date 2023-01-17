# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
heart_failure_data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
pd.pandas.set_option('display.max_columns', None)

heart_failure_data.head()
heart_failure_data.shape
heart_failure_data.info()
heart_failure_data.duplicated().sum()
heart_failure_data.describe()
sns.distplot(heart_failure_data['age'][heart_failure_data['DEATH_EVENT']==1], color='green')

sns.distplot(heart_failure_data['age'][heart_failure_data['DEATH_EVENT']==0], color='red')

plt.legend("0", "1")

plt.show()
sns.countplot(heart_failure_data['sex'][heart_failure_data['DEATH_EVENT']==1])

plt.show()
sns.countplot(heart_failure_data['smoking'][heart_failure_data['DEATH_EVENT']==1])

plt.show()
sns.countplot(heart_failure_data['high_blood_pressure'][heart_failure_data['DEATH_EVENT']==1])

plt.show()
sns.countplot(heart_failure_data['anaemia'][heart_failure_data['DEATH_EVENT']==1])

plt.show()
sns.pairplot(heart_failure_data, hue = 'DEATH_EVENT', vars = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',

                                                              'serum_creatinine', 'serum_sodium'], palette = 'husl')
sns.boxplot(heart_failure_data['creatinine_phosphokinase'])
sns.boxplot(heart_failure_data['ejection_fraction'])
sns.boxplot(heart_failure_data['platelets'])
sns.boxplot(heart_failure_data['serum_creatinine'])
plt.hist(heart_failure_data['creatinine_phosphokinase'], bins=50)

plt.show()
plt.hist(heart_failure_data['ejection_fraction'], bins=50)

plt.show()
plt.hist(heart_failure_data['platelets'], bins=50)

plt.show()
plt.hist(heart_failure_data['serum_creatinine'], bins=50)

plt.show()
outlier_features = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine']

for feature in outlier_features:

    IQR = heart_failure_data[feature].quantile(0.75) - heart_failure_data[feature].quantile(0.25)

    lower_boundary = heart_failure_data[feature].quantile(0.25) - (IQR*1.5)

    upper_boundary = heart_failure_data[feature].quantile(0.75) + (IQR*1.5)

    print('Feature : ', feature, 'Lower Boundary : ', lower_boundary, 'Upper Boundary : ', upper_boundary)
for feature in outlier_features:

    IQR = heart_failure_data[feature].quantile(0.75) - heart_failure_data[feature].quantile(0.25)

    lower_boundary = heart_failure_data[feature].quantile(0.25) - (IQR*1.5)

    upper_boundary = heart_failure_data[feature].quantile(0.75) + (IQR*1.5)

    heart_failure_data.loc[heart_failure_data[feature] <= lower_boundary, feature] = lower_boundary

    heart_failure_data.loc[heart_failure_data[feature] >= upper_boundary, feature] = upper_boundary
sns.boxplot(heart_failure_data['creatinine_phosphokinase'])
sns.boxplot(heart_failure_data['ejection_fraction'])
sns.boxplot(heart_failure_data['platelets'])
sns.boxplot(heart_failure_data['serum_creatinine'])
plt.figure(figsize=(12,5))

corr = heart_failure_data.corr()

sns.heatmap(corr, annot=True, cmap='RdYlGn')
features = heart_failure_data.drop(['DEATH_EVENT'], axis=1)

label = heart_failure_data['DEATH_EVENT']
plt.figure(figsize=(12,10))

from sklearn.ensemble import ExtraTreesRegressor

etr = ExtraTreesRegressor()

etr.fit(features, label)

feat_importances = pd.Series(etr.feature_importances_, index=features.columns)

feat_importances.nlargest(12).plot(kind='barh')

plt.show
features.drop(['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

train_features, test_features, train_label, test_label = train_test_split(features, label, test_size=0.3, random_state=13)
#Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]



#Number of features to consider in every split

max_features = ['auto', 'sqrt']



#Maximum number of levels in a tree

max_depth = [int(x) for x in np.linspace(start=5, stop=30, num=6)]



#Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]



#Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]



#Random Grid

random_grid = {'n_estimators' : n_estimators,

              'max_features' : max_features,

              'max_depth' : max_depth,

              'min_samples_split' : min_samples_split,

              'min_samples_leaf' : min_samples_leaf}
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

random_forest = RandomForestClassifier()

random_forest_model = RandomizedSearchCV(estimator=random_forest, param_distributions=random_grid, scoring='neg_mean_squared_error',

                                        n_iter=10, cv=5, verbose=2, random_state=13, n_jobs=1)

random_forest_model.fit(train_features, train_label)
random_forest_model.best_params_
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
confusion_matrix(test_label, random_forest_model.predict(test_features))
plot_confusion_matrix(random_forest_model, test_features, test_label)
accuracy_score(test_label, random_forest_model.predict(test_features))
recall_score(test_label, random_forest_model.predict(test_features))
precision_score(test_label, random_forest_model.predict(test_features))
f1_score(test_label, random_forest_model.predict(test_features))
plt.style.use('seaborn')



fpr, tpr, thresholds = roc_curve(test_label, random_forest_model.predict_proba(test_features)[:,1], pos_label=1)



random_probs = [0 for i in range(len(test_label))]

p_fpr, p_tpr, _ = roc_curve(test_label, random_probs, pos_label=1)



plt.plot(fpr, tpr, linestyle='--',color='orange', label='Random Forest')

plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')



plt.title('Random Forest ROC Curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')



plt.legend(loc='best')

plt.savefig('ROC',dpi=300)



plt.show()
auc = roc_auc_score(test_label, random_forest_model.predict_proba(test_features)[:,1])

auc