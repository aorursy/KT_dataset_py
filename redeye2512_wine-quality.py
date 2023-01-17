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
import pandas as pd

import numpy as np 

import seaborn as sns

from matplotlib import pyplot as plt 

df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
df.isnull().sum()
fig, ax = plt.subplots(1,1,figsize=(20,10))

sns.heatmap(df.corr(), annot=True, axes=ax)

plt.show()
df.describe()
sns.boxplot(df['quality'])
df['label'] = df['quality'].apply(lambda x: int(x>6))

df['label'].value_counts(True)
import sklearn 

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, PrecisionRecallDisplay, precision_recall_curve

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from time import time
# create X (input) and Y(output)

Y = df['label']

X = df.drop(columns=['label', 'quality'])

for col in X.columns:

    X[col] = X[col]/X[col].max()

    

# split to test and train 

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0) 
def train(model, data, parameters):

    start_time = time()

    train_x, test_x, train_y, test_y = data

    

    clf = GridSearchCV(model, parameters, n_jobs=-1, cv=5)

    clf.fit(train_x, train_y)

    model = clf.best_estimator_

    best_params_dict = clf.best_params_

    print('Best param:\n', best_params_dict)

    

    train_y = train_y.values.reshape(-1,1)

    test_y = test_y.values.reshape(-1,1)

    print('*'*5 + 'TRAIN' + '*'*5)

    pred = model.predict(train_x)

    #print('F1:', f1_score(pred, train_y))

    print('Precision:', precision_score(pred, train_y))

    print('Recall:', recall_score(pred, train_y))

    

    print('*'*5 + 'TEST' + '*'*5)

    pred = model.predict(test_x)

    #print('F1:', f1_score(pred, test_y))

    print('Precision:', precision_score(pred, test_y))

    print('Recall:', recall_score(pred, test_y))

    plot_confusion_matrix(model, test_x, test_y, display_labels=[0,1])

    plt.show()

    prec, recall, _ = precision_recall_curve(test_y, pred, pos_label=1)

    PrecisionRecallDisplay(precision=prec, recall=recall).plot()

    plt.show()

    

    print('Training in %.2f min'%((time()-start_time)/60))

    return model, best_params_dict
lr_parameters = {

    'class_weight': [{1:3,0:1}, {1:1,0:1}, {1:1,0:3}],

    'C': [100, 10, 1, 0.1, 0.01]

}

lr_model = LogisticRegression(random_state=10)

best_lr_model, best_lr_params = train(

    lr_model, 

    data=(train_x, test_x, train_y, test_y), 

    parameters=lr_parameters                     

)
dst_parameters = {

    'criterion':['gini', 'entropy'],

    'max_depth': [3,5,10],

    'class_weight': [{1:3,0:1}, {1:1,0:3}, 'balanced'],

    'max_features': [1,5,10]

}

dst_model = DecisionTreeClassifier()

best_dst_model, best_dst_params = train(

    dst_model, 

    data=(train_x, test_x, train_y, test_y), 

    parameters=dst_parameters                     

)
xgbc_parameters = {

    'n_estimators':[100, 500, 1000],

    'learning_rate':[1e-3],

    'booster': ['gbtree', 'gblinear'],

    'subsample': [0.2, 0.5, 1],

    'max_depth': [3,5,10],

}

xgbc_model = XGBClassifier(n_jobs=-1)

best_xgbc_model, best_xgbc_params = train(

    xgbc_model, 

    data=(train_x, test_x, train_y, test_y), 

    parameters=xgbc_parameters                     

)
rf_parameters = {

    'criterion':['gini', 'entropy'],

    'max_depth': [3,5,10],

    'class_weight': [{1:3,0:1}, {1:1,0:3}, 'balanced'],

    'n_estimators':[10, 50, 100, 500],

    'max_features':['auto', 'sqrt', 'log2']

}

rf_model = RandomForestClassifier()

best_rf_model, best_rf_params = train(

    rf_model, 

    data=(train_x, test_x, train_y, test_y), 

    parameters=rf_parameters                     

)
from sklearn.ensemble import VotingClassifier

vote_model = VotingClassifier(

    estimators=[

        ('lr', best_lr_model), ('rf', best_rf_model), ('xgbc', best_xgbc_model), ('dst', best_dst_model)

    ]

)

vote_parameters = {

#     'voting':['soft', 'hard']

}

best_vote_model, best_vote_params = train(

    dst_model, 

    data=(train_x, test_x, train_y, test_y), 

    parameters=vote_parameters                     

)