
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
Ortho = pd.read_csv('Orthopedic_Normality.csv')
Ortho = pd.DataFrame(Ortho)
Ortho.dtypes
Ortho
Ortho.dropna()
Ortho
Ortho['class'] = np.where(Ortho['class'] == 'Abnormal', 1, 0)
Ortho['class'].sum()
print('pelvic_incidence:',Ortho['pelvic_incidence'].unique())
print('pelvic_incidence_max:', Ortho['pelvic_incidence'].max())
print('pelvic_incidence_min:', Ortho['pelvic_incidence'].min())
print('pelvic_incidence_mean:', Ortho['pelvic_incidence'].mean())
print('pelvic_incidence_mode:', Ortho['pelvic_incidence'].mode())
plt.boxplot(Ortho['pelvic_incidence'])
Ortho[Ortho['pelvic_incidence'] > 100]
print('pelvic_tilt:',Ortho['pelvic_tilt'].unique())
print('pelvic_tilt_max:', Ortho['pelvic_tilt'].max())
print('pelvic_tilt_min:', Ortho['pelvic_tilt'].min())
print('pelvic_tilt_mean:', Ortho['pelvic_tilt'].mean())
print('pelvic_tilt_mode:', Ortho['pelvic_tilt'].mode())
print('pelvic_tilt_max:', Ortho['pelvic_tilt'].max())
plt.boxplot(Ortho['pelvic_tilt'])
Ortho[Ortho['pelvic_tilt'] > 45]
print('lumbar_lordosis_angle:',Ortho['lumbar_lordosis_angle'].unique())
print('lumbar_lordosis_angle_max:', Ortho['lumbar_lordosis_angle'].max())
print('lumbar_lordosis_angle_min:', Ortho['lumbar_lordosis_angle'].min())
print('lumbar_lordosis_angle_mean:', Ortho['lumbar_lordosis_angle'].mean())
print('lumbar_lordosis_angle_mode:', Ortho['lumbar_lordosis_angle'].mode())
print('lumbar_lordosis_angle_max:', Ortho['lumbar_lordosis_angle'].max())
plt.boxplot(Ortho['lumbar_lordosis_angle'])
Ortho = Ortho[Ortho['lumbar_lordosis_angle'] < 120]
print('sacral_slope:',Ortho['sacral_slope'].unique())
print('sacral_slope_max:', Ortho['sacral_slope'].max())
print('sacral_slope_min:', Ortho['sacral_slope'].min())
print('sacral_slope_mean:', Ortho['sacral_slope'].mean())
print('sacral_slope_mode:', Ortho['sacral_slope'].mode())
print('sacral_slope_max:', Ortho['sacral_slope'].max())
plt.boxplot(Ortho['sacral_slope'])
Ortho = Ortho[Ortho['sacral_slope'] < 120]
plt.boxplot(Ortho['sacral_slope'])
print('pelvic_radius:',Ortho['pelvic_radius'].unique())
print('pelvic_radius_max:', Ortho['pelvic_radius'].max())
print('pelvic_radius_min:', Ortho['pelvic_radius'].min())
print('pelvic_radius_mean:', Ortho['pelvic_radius'].mean())
print('pelvic_radius_mode:', Ortho['pelvic_radius'].mode())
print('pelvic_radius_max:', Ortho['pelvic_radius'].max())
#plt.boxplot(Ortho['pelvic_radius'])
plt.hist(Ortho['pelvic_radius'])
Ortho[Ortho['pelvic_radius'] > 150]
print('degree_spondylolisthesis:',Ortho['degree_spondylolisthesis'].unique())
print('degree_spondylolisthesis_max:', Ortho['degree_spondylolisthesis'].max())
print('degree_spondylolisthesis_min:', Ortho['degree_spondylolisthesis'].min())
print('degree_spondylolisthesis_mean:', Ortho['degree_spondylolisthesis'].mean())
print('degree_spondylolisthesis_mode:', Ortho['degree_spondylolisthesis'].mode())
print('degree_spondylolisthesis_max:', Ortho['degree_spondylolisthesis'].max())
plt.boxplot(Ortho['degree_spondylolisthesis'])
Ortho[Ortho['degree_spondylolisthesis'] > 100]
Ortho.index = np.arange(0, len(Ortho))
Ortho.index
features = Ortho[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle',
                  'sacral_slope','pelvic_radius','degree_spondylolisthesis']]
labels = Ortho['class']
from sklearn.model_selection import train_test_split 
tr_features, test_features, tr_labels, test_labels = train_test_split(features, labels,
random_state=0)
#ignoring the warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(random_state=0)
Parameters = {'max_depth':[1,2,4,8,16,32, 64, None]}
CV = GridSearchCV(DT, Parameters, cv=5)
CV.fit(tr_features, tr_labels.values.ravel())
CV.best_params_
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
predict = CV.predict(test_features)
print('accuracy:',accuracy_score(test_labels, predict))
print('precision:',precision_score(test_labels, predict))
print('recall:',recall_score(test_labels, predict))
print('f1:',f1_score(test_labels, predict))
print('total predicted 1s:',predict.sum())
print('total actual 1s:',test_labels.sum())
