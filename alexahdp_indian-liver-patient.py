import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, recall_score

from sklearn.metrics import accuracy_score, precision_score

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import Perceptron

from sklearn.svm import SVC



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/indian_liver_patient.csv')
data.head()
data.info()
data['Age'].plot(kind='hist')

plt.show()
data['Gender'].value_counts()
mean_albumin_and_globulin_ratio = data['Albumin_and_Globulin_Ratio'].mean()

data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(mean_albumin_and_globulin_ratio)
data['Gender'] = data['Gender'].astype('category')

data = pd.get_dummies(data)
corr_matrix = data.corr()

sns.heatmap(corr_matrix, annot=True, fmt=".1f", linewidths=.5);
# cols=['Albumin', 'Albumin_and_Globulin_Ratio', 'Gender_Female']

cols=['Albumin','Total_Bilirubin', 'Gender_Female']

# cols = data.columns[data.columns != 'Dataset']

x_train, x_valid, y_train, y_valid = train_test_split(

    data[cols],

    data['Dataset'],

    test_size=0.2,

    random_state=42

)
scaler = StandardScaler()



x_train = scaler.fit_transform(x_train)

x_valid = scaler.transform(x_valid)
def evaluate_model(model_name, model):

    model.fit(x_train, y_train)

    y_predicted = model.predict(x_valid)



    acc = accuracy_score(y_valid, y_predicted)

    prec = precision_score(y_valid, y_predicted)

    rec = recall_score(y_valid, y_predicted)



    print(model_name, ': {acc:', round(acc, 2), 'prec: ', round(prec, 2), 'rec: ', round(rec, 2), '}')
models = [

    {'name': 'LogisticRegression', 'model': LogisticRegression()},

    {'name': 'DecisionTreeClassifier', 'model': DecisionTreeClassifier(random_state=0, max_depth=5)},

    {'name': 'RandomForestClassifier', 'model': RandomForestClassifier(random_state=0, n_estimators=100, max_depth=4)},

    {'name': 'KNeighborsClassifier', 'model': KNeighborsClassifier(n_neighbors=5)},

    {'name': 'SVC', 'model': SVC(kernel='linear', gamma='scale')},

    {'name': 'Perceptron', 'model': Perceptron(tol=1e-3, random_state=0)},

]



for m in models:

    evaluate_model(m['name'], m['model'])