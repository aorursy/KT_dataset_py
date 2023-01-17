# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, RobustScaler

from lightgbm import LGBMClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.head()
def metrics(pred, y_test):

    precision = precision_score(pred, y_test)

    recall = recall_score(pred, y_test)

    f1 = f1_score(pred, y_test)

    roc = roc_auc_score(pred, y_test)

    acc = accuracy_score(pred, y_test)

    print(f"acc : {acc}, f1 : {f1}")

    print(f"precision : {precision}, recall : {recall}, roc : {roc}")
def get_train_test_data(data):

    X = data.drop('Class', axis = 1)

    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    return X_train, X_test, y_train, y_test
def modeling(model, X_train, X_test, y_train, y_test):

    print(model.__class__.__name__, " training")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    metrics(pred, y_test)

    print("**end**")
X_train, X_test, y_train, y_test = get_train_test_data(data)
lr = LogisticRegression()

lgbm = LGBMClassifier(boost_from_average=False)
modeling(lr, X_train, X_test, y_train, y_test)

modeling(lgbm, X_train, X_test, y_train, y_test)

#lf_pred = lf.predict(X_test)

#lgbm_pred = lgbm.predict(X_test)
def draw_graph(data_copy, feature1, feature2):

    fig, ax = plt.subplots(1, 2, figsize=(18, 4))

    feature1_val = data_copy[feature1].values

    feature2_val = data_copy[feature2].values

    

    sns.distplot(feature1_val, ax = ax[0], color = 'r')

    ax[0].set_title(f'distribution of transation {feature1}')

    ax[0].set_xlim([min(feature1_val), max(feature1_val)])

    

    sns.distplot(feature2_val, ax = ax[1], color='b')

    ax[1].set_title(f'distribution of transaction {feature2}')

    ax[1].set_xlim([min(feature2_val), max(feature2_val)])

    plt.show()
draw_graph(data, 'Amount', 'Time')
scaler = StandardScaler()

data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

data['loged_amount'] = np.log1p(data['Amount'].values.reshape(-1, 1))
draw_graph(data, 'scaled_amount', 'loged_amount')
rob_scaler = RobustScaler()
data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

data['rob_scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1, 1))
draw_graph(data, 'scaled_time', 'rob_scaled_time')
data.columns[20:]
data.drop(['Amount', 'Time', 'loged_amount', 'scaled_time'], axis = 1, inplace = True)
data.head()
scaled_amount = data['scaled_amount']

rob_scaled_time = data['rob_scaled_time']

data.drop(['scaled_amount', 'rob_scaled_time'], axis = 1, inplace = True)

data.insert(0, 'scaled_amount', scaled_amount)

data.insert(1, 'rob_scaled_time', rob_scaled_time)
data.head()
X_train, X_test, y_train, y_test = get_train_test_data(data)

modeling(lr, X_train, X_test, y_train, y_test)

modeling(lgbm, X_train, X_test, y_train, y_test)
plt.figure(figsize=(12, 8))

sns.heatmap(data.corr())
def draw_boxplots(data, x_col, y_col_1, y_col_2):

    f, axs = plt.subplots(1, 2, figsize=(15, 8))

    sns.boxplot(x = x_col, y = y_col_1, data = data, ax = axs[0])

    sns.boxplot(x = x_col, y = y_col_2, data = data, ax = axs[1])

    axs[0].set_title(f'{x_col} VS {y_col_1}')

    axs[1].set_title(f'{x_col} VS {y_col_2}')

    plt.show()

    

    
draw_boxplots(data, 'Class', 'V17', 'V14')
draw_boxplots(data, 'Class', 'V20', 'V21')
def remove_outlier_data(data, column):

    fraud_column_data = data[data['Class']==0][column]

    quan_25 = np.percentile(fraud_column_data, 25)

    quan_75 = np.percentile(fraud_column_data, 75)

    IQR = (quan_75 - quan_25) * 1.5

    max_v = quan_75 + IQR

    min_v = quan_25 - IQR

    outlier_index = fraud_column_data[(fraud_column_data < min_v) | (fraud_column_data > max_v)].index

    data.drop(outlier_index, axis = 0, inplace = True)

    return data
data.head()
remove_outlier_data = remove_outlier_data(data, 'V14')

X_train, X_test, y_train, y_test = get_train_test_data(remove_outlier_data)
print(remove_outlier_data.shape)
lr = LogisticRegression()

lgb = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average = False)
modeling(lr, X_train, X_test, y_train, y_test)

modeling(lgb, X_train, X_test, y_train, y_test)