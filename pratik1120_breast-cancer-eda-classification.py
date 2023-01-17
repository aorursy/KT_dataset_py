import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

from pandas_profiling import ProfileReport

from plotly.offline import iplot

!pip install joypy

import joypy



plt.rcParams['figure.figsize'] = 8, 5

plt.style.use("fivethirtyeight")



data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
#describing data



data.describe()
#covariance in data



data.cov()
#correlation in data



data.corr()
sns.heatmap(data.corr())

plt.show()
print('Missing Values Plot')

plt.figure(figsize=(8,8))

sns.barplot(data=data.isnull().sum().reset_index(), y='index', x=0)

plt.ylabel('Variables')

plt.xlabel('Missing value Count')

plt.show()
drop_var = ['Unnamed: 32', 'id']



data.drop(drop_var, axis=1, inplace=True)
features = ['radius','texture','perimeter','area','smoothness','compactness','concavity','concave points','symmetry','fractal_dimension']



for feature in features:

    print("{} distribution".format(feature))

    sns.boxplot(data=data[['{}_mean'.format(feature), '{}_se'.format(feature), '{}_worst'.format(feature)]])

    plt.title('Distribution of {}'.format(feature))

    plt.show()
for feature in features:

    print("{} distribution based on diagnosis".format(feature))

    sns.violinplot(data=data, x="diagnosis", y="{}_mean".format(feature), size=8)

    plt.show()
print('Pairplot')

sns.pairplot(data=data[['diagnosis','area_mean','texture_mean','smoothness_mean','concavity_mean','symmetry_mean']], hue="diagnosis", height=3, diag_kind="hist")

plt.show()
#separating features and labels



X = data.drop('diagnosis',axis=1)

y = data['diagnosis']
#scaling the data



from sklearn import preprocessing



X_scaled = preprocessing.scale(X)
#splitting the data



from sklearn.model_selection import train_test_split, cross_val_score



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=13)
#creating model and fitting the data



from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(X_train, y_train)
#checking the cross val score



scores = cross_val_score(model, X_scaled, y, cv=5)

print(np.mean(scores))
#prediction



pred = model.predict(X_test)
#checking the classificaton report



from sklearn.metrics import classification_report, confusion_matrix



print(classification_report(y_test, pred))
#confurion matrix



from sklearn.metrics import plot_confusion_matrix



plot_confusion_matrix(model, X_test, y_test)

plt.show()
#checking roc curve



from sklearn.metrics import roc_curve



model = LogisticRegression()

model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label='M')

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='Logistic Regression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Logistic Regression ROC Curve')

plt.show()