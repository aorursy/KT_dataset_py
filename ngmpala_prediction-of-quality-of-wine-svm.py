import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# To calculate correlation on features
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report

from warnings import filterwarnings
filterwarnings('ignore')

# allow plots to appear directly in the notebook
%matplotlib inline
data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv', index_col=0)
data.head()
data.shape
data.info()
data.describe().transpose()
f, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)                                      # Set up the matplotlib figure

sns.barplot(x = 'quality', y = 'volatile acidity', data = data, ax=axes[0, 0])
sns.barplot(x = 'quality', y = 'citric acid', data = data, ax=axes[0, 1])
sns.barplot(x = 'quality', y = 'residual sugar', data = data, ax=axes[1, 0])
sns.barplot(x = 'quality', y = 'chlorides', data = data, ax=axes[1, 1])
f, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)                                      # Set up the matplotlib figure

sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = data, ax=axes[0, 0])
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = data, ax=axes[0, 1])
sns.barplot(x = 'quality', y = 'density', data = data, ax=axes[1, 0])
sns.barplot(x = 'quality', y = 'pH', data = data, ax=axes[1, 1])
f, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)                                      # Set up the matplotlib figure

sns.barplot(x = 'quality', y = 'sulphates', data = data, ax=axes[0])
sns.barplot(x = 'quality', y = 'alcohol', data = data, ax=axes[1])

for feature in data.columns:
    if feature == 'quality':
        continue
    sns.jointplot(x=feature, y='quality', data=data, kind='kde', stat_func=stats.pearsonr)
    plt.show()
sns.pairplot(data, vars=["pH", "density", "alcohol", "free sulfur dioxide"], hue="quality", height = 2, aspect = 1.5)
sns.pairplot(data, vars=["residual sugar", "citric acid", "sulphates", "volatile acidity"], hue="quality", height = 2, aspect = 1.5)
#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)
#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()
#Bad becomes 0 and good becomes 1 
data['quality'] = label_quality.fit_transform(data['quality'])
data['quality'].value_counts()
sns.pairplot(data, vars=["pH", "density", "alcohol", "free sulfur dioxide"], hue="quality", size = 2, aspect = 1.5)
sns.heatmap( data.corr(), annot=True );
def handle_missing_values(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percentage = round(total / data.shape[0] * 100)
    return pd.concat([total, percentage], axis = 1, keys = ['total', 'percentage'])
handle_missing_values(data)
X = data.loc[:, data.columns != 'quality']
y = data.quality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#Applying Standard scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
print(classification_report(y_test, pred_svc))
#Finding best parameters for our SVC model
param = {
    'C': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1, 0.8, 0.9, 1, 1.1, 1.2]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)
#Best parameters for our svc model
grid_svc.best_params_
#Let's run our SVC again with the best parameters.
svc2 = SVC(C = 1.4, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))