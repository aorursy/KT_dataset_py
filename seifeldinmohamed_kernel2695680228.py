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

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
df.describe()
df.isna().values.sum()
class_df = pd.DataFrame({'target': df['target'].value_counts().index,

                     'Count': df['target'].value_counts()})



print(class_df)

sns.barplot(x=class_df['target'], y=class_df['Count'])

plt.show()
def resample_data(X, y):

    X_resampled, y_resampled = SMOTE(random_state=21).fit_sample(X, y)

    return X_resampled, y_resampled
f, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df.corr(), annot=True, linewidth=.5, fmt='.1f', ax=ax)
df.drop(columns= ['fbs'])

df.head()
df.hist()
df = pd.get_dummies(df, columns = ['cp', 'restecg', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])
X = df.drop(columns= ['target'])

Y = df['target']
X, Y = resample_data(X, Y)
randomforest_classifier= RandomForestClassifier(n_estimators=10)
knn_classifier = KNeighborsClassifier(n_neighbors = 12)
svm_model = SVC(kernel = 'rbf', C = 0.2)
score=cross_val_score(knn_classifier,X,Y,cv=10)

print("Accuracy: {:.2f} %".format(score.mean()*100))

print("Standard Deviation: {:.2f} %".format(score.std()*100))