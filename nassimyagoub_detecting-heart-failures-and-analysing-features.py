import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df = df.sample(frac=1)

df.head()
df.hist(figsize=(16,10))
fig, ax = plt.subplots(figsize=(15,13)) 



matrix = np.triu(df.corr(),k=1)

sns.heatmap(df.corr(), annot=True, mask = matrix)

plt.title("Correlation matrix between patient features\n", fontsize=12, color='#009432')
data = df.drop(['DEATH_EVENT', 'time'], axis = 1)

targets = df['DEATH_EVENT']
num_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']

cat_columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

columns = num_columns + cat_columns
num_data = data[num_columns]

cat_data = data[cat_columns]
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



scaled_num_data = pd.DataFrame(scaler.fit_transform(num_data))
processed_data = pd.concat([scaled_num_data, cat_data], axis=1, sort=False)

num_columns_dict = {i : num_columns[i] for i in range(len(num_columns))}

processed_data = processed_data.rename(columns = num_columns_dict)

processed_data.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



forest_clf = RandomForestClassifier()



print(cross_val_score(forest_clf, processed_data, targets, cv = 5))
forest_clf.fit(processed_data, targets)
top_idx = 6



importances = forest_clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest_clf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]

names = [columns[i] for i in indices]



# Plot the impurity-based feature importances of the forest

plt.figure(figsize=(14,8))

plt.title("Feature ranking")

plt.bar(range(top_idx), importances[indices[:top_idx]],

        color="r", yerr=std[indices[:top_idx]], align="center")

plt.xticks(range(top_idx), names[:top_idx])

plt.xlim([-1, top_idx])

plt.ylabel("Importance and standard deviation of the features", fontsize=12)

plt.show()