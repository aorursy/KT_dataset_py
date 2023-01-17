# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import sys

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
from sklearn.preprocessing import StandardScaler, RobustScaler



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, jaccard_score
data = pd.read_csv("../input/creditcardfraud/creditcard.csv")

data
data.info()
data.isnull().sum()
data.describe()
import seaborn as sns
sns.countplot(data.Class, palette=['blue', 'red'] )

plt.title('Number of genuine and fraudulent transactions')

plt.ylabel('Number of transactions')

plt.xlabel('Transactions type')
print('Genuine transactions: ',round(data.Class.value_counts()[0]/len(data.Class)*100,2), '%')

print('Frauds: ',round(data.Class.value_counts()[1]/len(data.Class)*100,2), '%')
fig, ax = plt.subplots( 1, 2, figsize = (18, 4))

sns.boxplot(data.Amount, ax=ax[0])

ax[0].set_title('Amount Variable Boxplot')

sns.boxplot(data.Time, ax=ax[1])

ax[1].set_title('Time Variable Boxplot')
fig, ax = plt.subplots( 1, 2, figsize = (18, 4))

sns.distplot(data.Amount, ax=ax[0])

ax[0].set_title('Amount Variable Boxplot')

sns.distplot(data.Time, ax=ax[1])

ax[1].set_title('Time Variable Boxplot')
corr = data.corr()

ax = plt.figure(figsize=(26, 16))

ax = sns.heatmap(corr)
robust_scaler = RobustScaler()

Amount_scaled = robust_scaler.fit_transform(data.Amount.values.reshape(-1, 1))
var_scaled = robust_scaler.fit_transform(data[['Amount', 'Time']])
var_scaled
data_scaled = data.copy()

data_scaled.Amount = var_scaled[ : , 0]

data_scaled.Time = var_scaled[ : , 1]

data_scaled
X = data.drop( labels = 'Class', axis = 1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
model = ExtraTreesClassifier()

model.fit(X, y)
print(model.feature_importances_)
feature_importances = pd.Series(model.feature_importances_)

feature_importances.sort_values( ascending= False)
ax = plt.figure( figsize = (10, 5))

ax = feature_importances.plot.bar(title = 'Variable importance')

ax
feature_selection = data.iloc[ : , [17, 14, 12, 11, 10, 16, 18]]

feature_selection.loc[ : , 'Class']  = data.Class
X = feature_selection.drop( labels=['Class'], axis = 1)
y = feature_selection['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)
model = RandomForestClassifier()

model.fit(X_train, y_train)

y_predicted_model1 = model.predict(X_test)
model2 = LogisticRegression()

model2.fit(X_train, y_train)

y_predicted_model2 = model2.predict(X_test)
model3 = AdaBoostClassifier( n_estimators=100, random_state=2020, algorithm='SAMME.R',

                         learning_rate=0.8 )

model3.fit(X_train, y_train)

y_predicted_model3 = model3.predict(X_test)
scores = pd.DataFrame()
metrics = ['f1_score','jaccard_score', 'roc_auc_score']
predictions = {

    'Random Forest': y_predicted_model1,

    'Logistic Regression': y_predicted_model2,

    'Ada Boost Classifier': y_predicted_model3

}
for key, item in predictions.items():

    for metric in metrics:

        scores.loc[key, metric] = getattr(sys.modules[__name__], metric)(y_test, item)

        

scores