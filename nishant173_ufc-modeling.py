# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import calendar

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
path = "/kaggle/input/ufcdata/raw_total_fight_data.csv"

df_total_fight_data = pd.read_csv(filepath_or_buffer=path, sep=';')

df_total_fight_data.shape
df_total_fight_data.head()
def get_success_percentage(obj):

    obj = str(obj).strip().split('of') # Returns list of length 3

    landed = int(obj[0])

    attempted = int(obj[-1])

    if attempted == 0:

        return -1 # To indicate no attempts

    success_percentage = round(landed * 100 / attempted, 2)

    return success_percentage





def alter_percentages(obj):

    obj = str(obj)[:-1]

    obj = float(obj)

    return obj





def extract_minutes(obj):

    """ Helper function """

    obj = str(obj).split(':')

    secs = int(obj[0]) * 60 + int(obj[-1])

    mins = round(secs / 60, 2)

    return mins





def day_of_week(date):

    """

    Takes in Pandas datetime, and returns the name of day of that date.

    """

    day_index = date.weekday()

    day = calendar.day_name[day_index]

    return day





def alter_date(date_obj):

    month_mapper = {

        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7,

        'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12

    }

    mmdd, yy = str(date_obj).split(',')

    mm, dd = mmdd.split(' ')

    yy = str(yy).strip()

    dd = str(dd)

    mm = str(month_mapper[mm])

    final_date = pd.to_datetime("{}/{}/{}".format(mm, dd, yy))

    return final_date





def map_categories(df, column):

    df_temp = df.copy()

    mapper_dict = dict()

    new_column = column + '_'

    categories = df_temp[column].dropna().unique().tolist()

    for i, category in enumerate(categories):

        mapper_dict[category] = i+1

    df_temp[new_column] = df_temp[column].map(mapper_dict)

    df_temp[new_column] = df_temp[new_column].astype(str)

    return df_temp





def extract_country(obj):

    country = str(obj).split(',')[-1].strip().upper()

    return country
def engineer_features(df_raw):

    df_temp = df_raw.copy()

    df_temp.dropna(inplace=True)

    df_temp['date'] = df_temp['date'].apply(alter_date)

    

    # Success percentages

    features = ['R_KD', 'B_KD', 'R_TD', 'B_TD', 'R_HEAD', 'B_HEAD', 'R_BODY', 'B_BODY', 'R_LEG', 'B_LEG',

                'R_SIG_STR.', 'B_SIG_STR.', 'R_TOTAL_STR.', 'B_TOTAL_STR.', 'R_CLINCH', 'B_CLINCH',

                'R_GROUND', 'B_GROUND']

    for feature in features:

        new_feature = feature + '_success_percent'

        df_temp[new_feature] = df_temp[feature].apply(get_success_percentage)

    df_temp.drop(labels=features, axis=1, inplace=True)

    

    # Clean-up percentage signs (inplace)

    features = ['R_SIG_STR_pct', 'B_SIG_STR_pct', 'R_TD_pct', 'B_TD_pct']

    for feature in features:

        df_temp[feature] = df_temp[feature].apply(alter_percentages)

    

    # Calculate minutes per fight

    fight_mins = ((df_temp['last_round'] - 1) * 5) + df_temp['last_round_time'].apply(extract_minutes)

    df_temp['fight_mins'] = fight_mins

    

    # Alter 'Winner' column into 'Result' column with [Red, Blue, Draw]

    condn_red_wins = (df_temp['Winner'] == df_temp['R_fighter'])

    condn_blue_wins = (df_temp['Winner'] == df_temp['B_fighter'])

    condn_draw = ~ (condn_red_wins | condn_blue_wins)

    

    df_red_wins = df_temp.loc[condn_red_wins, :].copy()

    df_red_wins['Result'] = 'Red'

    df_blue_wins = df_temp.loc[condn_blue_wins, :].copy()

    df_blue_wins['Result'] = 'Blue'

    df_draws = df_temp.loc[condn_draw, :].copy()

    df_draws['Result'] = 'Draw'

    df_results_added = pd.concat(objs=[df_red_wins, df_blue_wins, df_draws], ignore_index=True, sort=False)

    df_results_added.sort_values(by='date', ascending=False, inplace=True)

    df_results_added.reset_index(drop=True, inplace=True)

    

    # Get few additional features and add categorical mapping

    df_results_added['day_of_week'] = df_results_added['date'].apply(day_of_week)

    df_results_added['country'] = df_results_added['location'].apply(extract_country)

    df_results_added = map_categories(df=df_results_added, column='day_of_week')

    df_results_added = map_categories(df=df_results_added, column='country')

    df_results_added = map_categories(df=df_results_added, column='win_by')



    features_to_drop = ['R_DISTANCE', 'B_DISTANCE', 'win_by', 'last_round', 'last_round_time', 'Format', 'Referee',

                        'date', 'location', 'Fight_type', 'Winner', 'day_of_week', 'country']

    if features_to_drop:

        df_results_added.drop(labels=features_to_drop, axis=1, inplace=True)

    return df_results_added
%%time

df_examples = engineer_features(df_raw=df_total_fight_data)
df_examples.head()
df_examples.shape
to_drop = ['R_fighter', 'B_fighter', 'day_of_week_', 'country_', 'win_by_']

if to_drop:

    df_examples.drop(labels=to_drop, axis=1, inplace=True)
df_examples.shape
df_examples.head()
holdout = 100

df_examples = df_examples[df_examples['Result'] != 'Draw']

df_holdout = df_examples.head(holdout)

df_examples = df_examples.tail(len(df_examples) - holdout)
df_examples.shape, df_holdout.shape
if df_examples.isnull().sum().sum() == 0:

    print("No missing values!")

else:

    print("There are missing values!")
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm, naive_bayes

from sklearn.metrics import confusion_matrix, accuracy_score



import joblib
X = df_examples.drop(labels=['Result'], axis=1)

y = df_examples['Result'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



# Feature Scaling

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
classifier = RandomForestClassifier(n_estimators=30, criterion='gini', max_depth=30)

model = classifier.fit(X_train, y_train)

y_pred = model.predict(X_test)

model_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred, normalize=True) * 100

model_accuracy = round(model_accuracy, 2)

model_accuracy
scores = cross_val_score(estimator=model, X=X, y=y, cv=20, scoring='accuracy')

scores.mean()
param_grid = dict(

    n_estimators=np.arange(10,100+1,10),

    criterion=['gini', 'entropy'],

    max_depth=np.arange(10,100+1,10),

    min_samples_split=[1,2,3]

)



grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='accuracy')
grid.estimator
classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=30, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=30,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)

model = classifier.fit(X_train, y_train)

y_pred = model.predict(X_test)

model_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred, normalize=True) * 100

model_accuracy = round(model_accuracy, 2)

model_accuracy
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

cm
df_holdout['Result'].value_counts()
df_holdout_X = df_holdout.drop(labels='Result', axis=1)



sc = StandardScaler()

X_holdout = sc.fit_transform(df_holdout_X)



y_pred = model.predict(X_holdout)



model_accuracy = accuracy_score(y_true=df_holdout['Result'].values, y_pred=y_pred, normalize=True) * 100

model_accuracy = round(model_accuracy, 2)

model_accuracy
def get_feature_importances(model, df):

    feature_column = 'Feature'

    importance_column = 'Importance_Percentage'

    data = {

        feature_column: df.columns,

        importance_column: model.feature_importances_

    }

    df_feature_importances = pd.DataFrame(data=data).sort_values(by=importance_column, ascending=False)

    df_feature_importances[importance_column] = df_feature_importances[importance_column] * 100

    df_feature_importances[importance_column] = df_feature_importances[importance_column].apply(round, args=[2])

    df_feature_importances.reset_index(drop=True, inplace=True)

    return df_feature_importances
feat_importance = get_feature_importances(model=model, df=X)

feat_importance.head(10)
feat_importance.head(10).set_index('Feature').plot(kind='barh', color='purple')

plt.show()
%%time



# Testing different algorithms/models

use_cross_val = True

dict_model_accuracy = dict()

algorithms = ['knn', 'rfc', 'logistic_regression', 'svm', 'naive_bayes']

for algorithm in algorithms:

    if algorithm == 'knn':

        classifier = KNeighborsClassifier(n_neighbors=55, metric='minkowski', p=2)

    elif algorithm == 'rfc':

        classifier = RandomForestClassifier(n_estimators=30, criterion='gini', max_depth=30)

    elif algorithm == 'logistic_regression':

        classifier = LogisticRegression(solver='sag')

    elif algorithm == 'svm':

        classifier = svm.SVC(kernel='rbf')

    elif algorithm == 'naive_bayes':

        classifier = naive_bayes.GaussianNB()



    model = classifier.fit(X_train, y_train)

    

    if use_cross_val:

        # Evaluate with k-fold cross-validation

        scores = cross_val_score(estimator=model, X=X, y=y, cv=20, scoring='accuracy')

        dict_model_accuracy[algorithm] = scores.mean()

    else:        

        # Evaluate with just the model

        y_pred = model.predict(X_test)

        # cm = confusion_matrix(y_test, y_pred)

        model_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred, normalize=True) * 100

        model_accuracy = round(model_accuracy, 2)

        dict_model_accuracy[algorithm] = model_accuracy

#

rename_dict = {

    'index': 'algorithm',

    0: 'accuracy_percentage'

}

df_accuracies = pd.DataFrame(dict_model_accuracy, index=[0]).T.reset_index().rename(mapper=rename_dict, axis=1)

df_accuracies = df_accuracies.sort_values(by='accuracy_percentage', ascending=False).reset_index(drop=True)
df_accuracies
df_accuracies.set_index('algorithm').plot(kind='barh', color='orange')

plt.show()