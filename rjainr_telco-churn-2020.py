# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv', sep=',')

df.head()
rows = df.shape[0]

cols = df.shape[1]

print("Rows: {}, cols:{} ".format(rows, cols))
df.isnull().sum().values.sum()
df.nunique()
df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})

df['MultipleLines'].unique()
replace_cols = ['OnlineSecurity', 'OnlineBackup', 'OnlineBackup', 

                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in replace_cols:

    print("Col:{}, unique: {} ".format(col, df[col].unique()))

    df[col].replace({'No internet service': 'No'}, inplace=True)
print(df.dtypes)
df["TotalCharges"] = df["TotalCharges"].astype(float)
df['TotalCharges'] = df["TotalCharges"].replace(" ",np.nan)

df = df.reset_index()[df.columns]

print("Number of null values in Totalcharges: {}".format(len(df) - df['TotalCharges'].count()))



df = df[df['TotalCharges'].notnull()]

df["TotalCharges"] = df["TotalCharges"].astype(float)
bin_labels_5 = ['Tenure1', 'Tenure2', 'Tenure3', 'Tenure4', 'Tenure5']

df['TenureBin'] = pd.qcut(df['tenure'],

                              q=[0, .2, .4, .6, .8, 1],

                              labels=bin_labels_5)



df = df.drop('tenure', axis=1)
df.head()
churn = df[df['Churn']=='Yes']

non_churn = df[df['Churn']=='No']
churn_values = df['Churn'].value_counts().values.tolist()

churn_keys = df['Churn'].value_counts().keys().tolist()



print("labels are ", churn_values)

print("values are ", churn_keys)



fig1, ax1 = plt.subplots()

ax1.pie(churn_values, explode=(0, 0.1), labels=churn_keys, autopct='%1.1f%%',

        shadow=True)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.legend(['Non Churn', 'Churn'])

plt.show()
def plot_pie(col):

    labels = churn[col].value_counts().keys().tolist()

    churn_val = churn[col].value_counts().values.tolist()

    nonchurn_val = non_churn[col].value_counts().values.tolist()



    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (5,5))

    ax1.pie(churn_val, explode=(0, 0.1), labels=labels, autopct='%1.1f%%',

            shadow=True)

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    ax1.set_title('Churn')



    ax2.pie(nonchurn_val, explode=(0, 0.1), labels=labels, autopct='%1.1f%%',

            shadow=True)

    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    ax2.set_title('Non churn')



    f.suptitle(col)

    plt.legend(['Non Churn', 'Churn'], loc= 'best')

    plt.show()
plot_pie('SeniorCitizen')
plot_pie('gender')
plot_pie('Partner')
plot_pie('Dependents')
df.head()


sns.scatterplot(df['MonthlyCharges'], df['TotalCharges'], hue=df['Churn'])

plt.show()
fig = plt.figure(figsize=(7,7))

sns.scatterplot(df['MonthlyCharges'], df['TotalCharges'], hue=df['TenureBin'])

plt.show()
df.head()
df = df.drop('customerID', 1)
binary_cols = [col for col in df.columns.tolist() if df[col].nunique()==2]



categorical_cols = [col for col in df.columns.tolist() if df[col].nunique() < 6]

categorical_cols = [col for col in categorical_cols if col not in binary_cols]



target_col = ['Churn']



numerical_cols = [col for col in df.columns.tolist() if col not in binary_cols+categorical_cols+target_col]
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



le = LabelEncoder()

for col in binary_cols :

    df[col] = le.fit_transform(df[col])

    

df = pd.get_dummies(data=df, columns=categorical_cols)



scaler = StandardScaler()

df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
f = plt.figure(figsize=(19, 15))

plt.matshow(df.corr(), fignum=f.number)

plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)

plt.yticks(range(df.shape[1]), df.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go#visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

from sklearn.metrics import roc_auc_score,roc_curve,scorer

from sklearn.metrics import f1_score

import statsmodels.api as sm

from sklearn.metrics import precision_score,recall_score

from sklearn.model_selection import GridSearchCV
from numpy import mean

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from imblearn.ensemble import BalancedRandomForestClassifier
target = df['Churn']

train = df.drop('Churn', axis=1)
# target = target.to_numpy()
X_train,X_test,y_train,y_test = train_test_split(train, target, test_size=0.15, shuffle = True, stratify=target )
def grid_search(params, model):

    grid_search = GridSearchCV(model, params, scoring='f1')

    model = grid_search.fit(X_train, y_train)

    print ('Best score: %0.3f' % grid_search.best_score_)



    best_parameters = model.best_estimator_

    print ('Best parameters set:', best_parameters)

    return model



def print_classification_report(model):

    predictions = model.predict(X_test)

    # conf_matrix = confusion_matrix(y_test,predictions)

    target_names = ['Not churn', 'Churn']

    print("\n")

    print(classification_report(y_test, predictions, target_names = target_names))
parameters = {'C': (0.1, 0.5,1)}

logit = grid_search(parameters, LogisticRegression())

print_classification_report(logit)
from xgboost import XGBClassifier
churn_values = df['Churn'].value_counts().values.tolist()

pos_weight = churn_values[0]/churn_values[1]

print("pos_weight is ", pos_weight)
parameters = {'max_depth': (5, 8, 10), 'n_estimators': (70, 100, 150)}

xgb_model = grid_search(parameters, XGBClassifier(scale_pos_weight=pos_weight))

print_classification_report(xgb_model)
parameters = {'n_estimators': (15, 20, 50), 'max_depth': (5,10,12)}

rf_clf = grid_search(parameters, RandomForestClassifier(class_weight='balanced_subsample'))

print_classification_report(rf_clf)
