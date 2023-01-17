import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split



%matplotlib inline
data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.describe()
data.head()
data.tail()
pd.isnull(data).any()
plt.figure(figsize = (10, 6))

sns.distplot(data['quality'])

plt.xlabel('Quality')

plt.show()
data.corr()
mask = np.zeros_like(data.corr())

triangle_indices = np.triu_indices_from(mask)

mask[triangle_indices] = True

plt.figure(figsize=(16,10))

sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={"size": 14})

sns.set_style('white')

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
%%time



sns.pairplot(data, kind='reg', plot_kws={'line_kws':{'color': 'cyan'}})

plt.show()
bins = (2, 6.5, 8)

group_names = ['bad', 'good']

data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)
quality_labels = LabelEncoder()
data['quality'] = quality_labels.fit_transform(data['quality'])

sns.countplot(data['quality'])
quality = data['quality']

features = data.drop(['quality'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, quality, test_size = 0.2, random_state = 50)

sgd = SGDClassifier(penalty=None)

sgd.fit(X_train, y_train)

pred_sgd = sgd.predict(X_test)
print(f"The classifier has an accuracy of about: {round(sgd.score(X_test, y_test)*100, 0)}%")

print(confusion_matrix(y_test, pred_sgd))