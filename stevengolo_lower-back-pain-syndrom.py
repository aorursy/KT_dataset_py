# Load the librairies
get_ipython().magic('matplotlib inline')
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
# Import the data
data = pd.read_csv('../input/Dataset_spine.csv', decimal='.', sep=',', header=0)
data = data.drop('Unnamed: 13', 1)
data.columns = ['pelvic_incidence', 'pelvic_tilt',
                'lumbar_lordosis_angle', 'sacral_slope',
                'pelvic_radius', 'degree_spondylolisthesis',
                'pelvic_slope', 'direct_tilt',
                'thoracic_slope', 'cervical_tilt',
                'sacrum_angle', 'scoliosis_slope',
                'class']
data.head()
data.info()
data.describe()
data[data.degree_spondylolisthesis > 180]
data.loc[115, 'degree_spondylolisthesis'] = 41.8543082
data['class'] = pd.get_dummies(data['class'], prefix='class', drop_first=True)
# Compute the correlation matrix.
corr_data = round(data.corr(),2)
corr_data.columns = ['Pelvic Incidence', 'Pelvic Tilt',
                'Lumbar Lordosis Angle', 'Sacral Slope',
                'Pelvic Radius', 'Degree Spondylolisthesis',
                'Pelvic Slope', 'Direct Tilt',
                'Thoracic Slope', 'Cervical Tilt',
                'Sacrum Angle', 'Scoliosis Slope',
                'Class']
corr_data.index = corr_data.columns

f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_data, mask=None, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            annot=True)
plt.show()
f, ax = plt.subplots(figsize=(30, 12))

plt.subplot(161)
sns.boxplot(y='pelvic_incidence', x='class', data=data)
plt.ylabel('Pelvic Incidence')
plt.xlabel('')
plt.xticks(np.arange(2), ('Abnormal', 'Normal'))

plt.subplot(162)
sns.boxplot(y='pelvic_tilt', x='class', data=data)
plt.ylabel('Pelvic Tilt')
plt.xlabel('')
plt.xticks(np.arange(2), ('Abnormal', 'Normal'))

plt.subplot(163)
sns.boxplot(y='lumbar_lordosis_angle', x='class', data=data)
plt.ylabel('Lumbar Lordosis Angle')
plt.xlabel('')
plt.xticks(np.arange(2), ('Abnormal', 'Normal'))

plt.subplot(164)
sns.boxplot(y='sacral_slope', x='class', data=data)
plt.ylabel('Sacral Slope')
plt.xlabel('')
plt.xticks(np.arange(2), ('Abnormal', 'Normal'))

plt.subplot(165)
sns.boxplot(y='degree_spondylolisthesis', x='class', data=data)
plt.ylabel('Degree Spondylolisthesis')
plt.xlabel('')
plt.xticks(np.arange(2), ('Abnormal', 'Normal'))

plt.subplot(166)
sns.boxplot(y='pelvic_radius', x='class', data=data)
plt.ylabel('Pelvic Radius')
plt.xlabel('')
plt.xticks(np.arange(2), ('Abnormal', 'Normal'))

plt.show()
model = ExtraTreesClassifier(n_estimators=200, random_state=0)
model.fit(data.drop('class', axis=1, inplace=False), data['class'])

importances = model.feature_importances_
importances_std = np.std([model_tree.feature_importances_ for model_tree in model.estimators_], axis=0)
res = {'Name':['Pelvic Incidence', 'Pelvic Tilt',
                'Lumbar Lordosis Angle', 'Sacral Slope',
                'Pelvic Radius', 'Degree Spondylolisthesis',
                'Pelvic Slope', 'Direct Tilt',
                'Thoracic Slope', 'Cervical Tilt',
                'Sacrum Angle', 'Scoliosis Slope'],
       'Importances':importances,
       'Importances_std':importances_std}
res = pd.DataFrame(res)
res = res.loc[np.argsort(res.Importances)]

plt.barh(y=range(res.shape[0]), width=res.Importances,
         xerr=res.Importances_std, align='center', tick_label=res.Name)
plt.xlabel('Variable importance')
plt.show()
plt.scatter(data['degree_spondylolisthesis'], data['pelvic_radius'], c=data['class'])
plt.xlabel('Degree Spondylolisthesis')
plt.ylabel('Pelvic Radius')
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(data[['degree_spondylolisthesis', 'pelvic_radius', 'pelvic_tilt', 'pelvic_incidence']], data['class'], test_size=1/3, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train_transformed, Y_train)
Y_pred_dummy = dummy.predict(X_test_transformed)

Y_pred_proba_dummy = dummy.predict_proba(X_test_transformed)[:, 1]
[fpr_dummy, tpr_dummy, thr_dummy] = metrics.roc_curve(Y_test, Y_pred_proba_dummy)

print("The accuracy for the dummy classifier is: %0.2f " % (metrics.accuracy_score(Y_test, Y_pred_dummy)))
param_log_reg = {'tol': np.logspace(-5, 1, 7),
                 'C': np.logspace(-3, 3, 7),
                 'penalty': ['l2']}

log_reg = GridSearchCV(LogisticRegression(solver='lbfgs'), param_log_reg, cv=10, iid=False)
log_reg.fit(X_train_transformed, Y_train)

print("Best parameters set found on development set:", log_reg.best_params_)
Y_pred_log_reg = log_reg.predict(X_test_transformed)

Y_pred_proba_log_reg = log_reg.predict_proba(X_test_transformed)[:, 1]
[fpr_log_reg, tpr_log_reg, thr_log_reg] = metrics.roc_curve(Y_test, Y_pred_proba_log_reg)

print("The accuracy for the Logistic Regression classifier is: %0.2f " % (metrics.accuracy_score(Y_test, Y_pred_log_reg)))
plt.figure(figsize=(18,8))

plt.plot(fpr_dummy, tpr_dummy, color='blue', lw=2, label='Dummy Classifier - AUC = %0.2f' % metrics.auc(fpr_dummy, tpr_dummy))
plt.plot(fpr_log_reg, tpr_log_reg, color='red', lw=2, label='Logistic Regression - AUC = %0.2f' % metrics.auc(fpr_log_reg, tpr_log_reg))

plt.legend(loc = 'lower right')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity', fontsize=14)
plt.ylabel('Sensibility', fontsize=14)
plt.title('ROC curves', fontsize=18)
plt.show()