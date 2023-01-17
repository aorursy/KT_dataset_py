import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, confusion_matrix



sns.set_style("whitegrid")

              

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

df.head()
df.isnull().sum(axis = 0)
df.describe().round(2)
class_size = ds = df['DEATH_EVENT'].value_counts().reset_index()

sns.barplot(x="index", y="DEATH_EVENT", data=class_size)

plt.xlabel('DEATH_EVENT')

plt.ylabel('Count')

plt.show()
# Select Features and Target

x = df.drop(columns='DEATH_EVENT', axis=1)

y = df[['DEATH_EVENT']]
import warnings

warnings.filterwarnings('ignore')



# Scaling

x_scaling = preprocessing.StandardScaler().fit(x).transform(x)



# Create Model

reglog_0 = LogisticRegression()

scores = cross_validate(reglog_0, x_scaling, y, cv=10,

                         scoring=('accuracy', 'precision', 'recall'),

                         return_train_score = True)



print("Evaluation Scores:")

print("Logistic Regression Accuracy:", "{:.2f}%".format(scores['test_accuracy'].mean()*100))

print("Logistic Regression Precission :", "{:.2f}%".format(scores['test_precision'].mean()*100))

print("Logistic Regression Recall :", "{:.2f}%".format(scores['test_recall'].mean()*100))
y_predict_reglog = cross_val_predict(reglog_0, x_scaling, y, cv=10)

conf_mat = confusion_matrix(y, y_predict_reglog)



sns.heatmap(conf_mat, annot=True, cmap="viridis", fmt='g')

plt.xlabel('\nPredict Label')

plt.ylabel('True Label')

plt.show()
# Handling Imbalance Data using Random Over Sampling

from sklearn.datasets import make_classification

from imblearn.over_sampling import RandomOverSampler

from collections import Counter



x_resampled, y_resampled = make_classification(n_samples=1000, n_features=2, n_redundant=0,

                                                   n_clusters_per_class=1, flip_y=0, 

                                                   random_state=1)

ros = RandomOverSampler(random_state=0)

x_resampled, y_resampled = ros.fit_resample(x_resampled, y_resampled)



print(sorted(Counter(y_resampled).items()))
class_size_resample = pd.DataFrame(data=y_resampled, columns=["DEATH_EVENT_RESAMPLE"])

class_size_resample = class_size_resample['DEATH_EVENT_RESAMPLE'].value_counts().reset_index()



sns.barplot(x="index", y="DEATH_EVENT_RESAMPLE", data=class_size_resample)

plt.xlabel('DEATH_EVENT_RESAMPLE')

plt.ylabel('Count')

plt.show()
scores = cross_validate(reglog_0, x_resampled, y_resampled, cv=10,

                         scoring=('accuracy', 'precision', 'recall'),

                         return_train_score = True)



print("Evaluation Scores After Oversampling:")

print("Logistic Regression Accuracy:", "{:.2f}%".format(scores['test_accuracy'].mean()*100))

print("Logistic Regression Precission :", "{:.2f}%".format(scores['test_precision'].mean()*100))

print("Logistic Regression Recall :", "{:.2f}%".format(scores['test_recall'].mean()*100))
y_predict_resample = cross_val_predict(reglog_0, x_resampled, y_resampled, cv=10)

conf_mat_resample = confusion_matrix(y_resampled, y_predict_resample)



sns.heatmap(conf_mat_resample, annot=True, cmap="viridis", fmt='g')

plt.xlabel('\nPredict Label')

plt.ylabel('True Label')

plt.show()