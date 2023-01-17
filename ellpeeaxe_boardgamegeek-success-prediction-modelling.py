# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import tree

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.feature_selection import RFECV

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

from sklearn import preprocessing

from sklearn.pipeline import Pipeline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/cleaned_games_detailed_info-2.csv', index_col=0)

data.head()
print(list(data.columns))
# Irrelevant

data.drop(['id','description', 'image','name','thumbnail','bayes_average'], axis = 1, inplace = True) 
data.loc[:, filter(lambda colname: not colname.startswith('category_') and not colname.startswith('mechanic_'), data.columns)]
data[['suggested_player_age_mean', 'official_min_age']].corr()
data[['suggested_num_players_weighted_average', 'official_min_players', 'official_max_players_categorised']].corr()
# Target

y = data['is_success']



# Features

x = data.drop(['is_success'], axis = 1)
# Apply scaling - MinMaxScaler, RobustScaler, StandardScaler

scaler = preprocessing.MinMaxScaler()

scaled_x = scaler.fit_transform(x)

x = pd.DataFrame(scaled_x, columns=x.columns)

x.describe()
# feature selection estimators

LG = LogisticRegression(solver='lbfgs', max_iter=1000)

SVM = LinearSVC(penalty="l1", dual=False, max_iter=10000)
# wrapper methods

RFE_LG = RFECV(LG, cv=5, n_jobs=-1)

RFE_SVM = RFECV(SVM, cv=5, n_jobs=-1)

SFM_LG = SelectFromModel(LG)

SFM_SVM = SelectFromModel(SVM)



selectors = {"RFE_LG": RFE_LG,"RFE_SVM": RFE_SVM,"SFM_LG":SFM_LG,"SFM_SVM": SFM_SVM}
def predict_success(X, y, classifier, selectors):

    

    feature_selectors = list()

    precision_scores = list()

    recall_scores = list()

    f_measure_scores = list()

    remarks = list()

    

    for selector in selectors.keys():

        

        print("Running", selector, "...")

        

        pipeline = Pipeline([

              ('feature_selection', selectors[selector]),

              ('classifier', classifier)

        ])

        

        scoring = {'precision':'precision',

                  'recall': 'recall',

                  'f1':'f1'}

        

        remark = ''



        scores = cross_validate(pipeline, X, y, cv = 5, scoring=scoring)

        

        feature_selectors.append(selector)

        precision_scores.append(scores['test_precision'])

        recall_scores.append(scores['test_recall'])

        f_measure_scores.append(scores['test_f1'])

        remarks.append(remark)

        

    

    return pd.DataFrame({

        'feature_selector': feature_selectors,

        'precision': precision_scores,

        'recall': recall_scores,

        'f1': f_measure_scores,

        'remarks': remarks

    })    



def predict_success_grid(X, y, classifier, selectors, param_grid):



    scoring = {'precision':'precision',

                  'recall': 'recall',

                  'f1':'f1'}

    

    grid = GridSearchCV(classifier, param_grid=param_grid, cv= 5)



    scores = cross_validate(grid, X, y, cv=5, scoring=scoring)

    

    return [scores['test_precision'],scores['test_recall'],scores['test_f1']]
lg_clf = LogisticRegression(random_state=0, max_iter=1000, solver='lbfgs')



lg_results_df = predict_success(x, y, lg_clf, selectors)

lg_results_df
lg_results_mean_df = lg_results_df

lg_results_mean_df['precision'] = lg_results_mean_df['precision'].apply(lambda x: x.mean())

lg_results_mean_df['recall'] = lg_results_mean_df['recall'].apply(lambda x: x.mean())

lg_results_mean_df['f1'] = lg_results_mean_df['f1'].apply(lambda x: x.mean())

lg_results_df
svm_clf = LinearSVC(penalty="l1", dual=False, random_state=0, max_iter=10000)



svm_results_df = predict_success(x, y, svm_clf, selectors)

svm_results_df
svm_results_mean_df = svm_results_df

svm_results_mean_df['precision'] = svm_results_mean_df['precision'].apply(lambda x: x.mean())

svm_results_mean_df['recall'] = svm_results_mean_df['recall'].apply(lambda x: x.mean())

svm_results_mean_df['f1'] = svm_results_mean_df['f1'].apply(lambda x: x.mean())

svm_results_mean_df
from keras.models import Sequential

from keras.layers import Dense
from keras.models import Sequential

from keras.layers import Dense



def neural_network(X_train, X_test, y_train, y_test):

    model = Sequential()

    model.add(Dense(60, activation='sigmoid', input_dim=X_train.shape[1]))

    model.add(Dense(60, activation='sigmoid'))

    model.add(Dense(60, activation='sigmoid'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    model.fit(X_train, y_train, epochs=100, batch_size=50, verbose = 0)

    

    y_pred=model.predict(X_test)

    

    return y_pred
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

precision_scores = list()

recall_scores = list()

f_measure_scores = list()



count = 0



for train_index, test_index in skf.split(x, y):

    

    count += 1

    print("Running fold number",count,"...")

    

    # Perform KFold

    X_train, X_test = x.iloc[train_index], x.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]



    # Fit and predict

    y_pred = neural_network(X_train, X_test, y_train, y_test)

    y_pred_bin = (y_pred>0.40)



    # Metrics

    precision = metrics.precision_score(y_test, y_pred_bin)

    recall = metrics.recall_score(y_test, y_pred_bin)

    f_measure = metrics.f1_score(y_test, y_pred_bin)

    precision_scores.append(precision)

    recall_scores.append(recall)

    f_measure_scores.append(f_measure)

    

nn_results_df = pd.DataFrame({

    'precision': precision_scores,

    'recall': recall_scores,

    'f1': f_measure_scores

})



nn_results_df
print("Mean Precision:", nn_results_df['precision'].mean())

print("Mean Recall:", nn_results_df['recall'].mean())

print("Mean F1:", nn_results_df['f1'].mean())
rf_clf = RandomForestClassifier(random_state=0)

param_grid = { 

    'n_estimators': [500, 800, 1000],

    'max_depth' : [8,10,12],

    'criterion' :['gini', 'entropy']

}



rf_results = predict_success_grid(x, y, rf_clf, selectors, param_grid = param_grid)

rf_results
print("Mean Precision:", rf_results[0].mean())

print("Mean Recall:", rf_results[1].mean())

print("Mean F1:", rf_results[2].mean())