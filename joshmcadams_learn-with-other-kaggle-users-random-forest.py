import pandas as pd

import numpy as np



def describe(df):

    return pd.DataFrame(np.array([

        df.columns.values,

        df.dtypes.values,

        df.count().values,

        (df.fillna(0).count() - df.count()).values,

        df.min().values,

        df.max().values,

        df.nunique().values,

    ]).T, columns=[

        'Column',

        'Data Type',

        'Non-Null Count',

        'Null Count',

        'Min',

        'Max',

        'Distinct Values'

    ])
train_df = pd.read_csv('../input/learn-together/train.csv')

describe(train_df)
test_df = pd.read_csv('../input/learn-together/test.csv')

describe(test_df)
def test_train_scale(train, test):

    v = []

    for col_name in train.columns.values:

        if col_name not in train or col_name not in test:

            continue



        max_train = train[col_name].max()

        max_test = test[col_name].max()

        max_diff = abs(max_train-max_test)

        numerator = max(max_train, max_test)

        denominator = min(max_train, max_test)

        if max_train < 0 and max_test < 0:

            numerator = min(max_train, max_test) * -1

            denominator = max(max_train, max_test) * -1

        elif max_train < 0:

            lift = abs(max_train)

            numerator = max(max_train+lift, max_test+lift)

            denominator = min(max_train+lift, max_test+lift)

        elif max_test < 0:

            lift = abs(max_test)

            numerator = max(max_train+lift, max_test+lift)

            denominator = min(max_train+lift, max_test+lift)

        max_percent_diff = round(numerator if denominator == 0 else numerator/denominator, 2)



        min_train = train[col_name].min()

        min_test = test[col_name].min()

        min_diff = abs(min_train-min_test)

        numerator = max(min_train, min_test)

        denominator = min(min_train, min_test)

        if min_train < 0 and min_test < 0:

            numerator = min(min_train, min_test) * -1

            denominator = max(min_train, min_test) * -1

        elif min_train < 0:

            lift = abs(min_train)

            numerator = max(min_train+lift, min_test+lift)

            denominator = min(min_train+lift, min_test+lift)

        elif min_test < 0:

            lift = abs(min_test)

            numerator = max(min_train+lift, min_test+lift)

            denominator = min(min_train+lift, min_test+lift)

        min_percent_diff = round(numerator if denominator == 0 else numerator/denominator, 2)



        range_train = max_train - min_train

        range_test = max_test - min_test

        range_diff = abs(range_train-range_test)

        numerator = max(range_train, range_test)

        denominator = min(range_train, range_test)

        if range_train < 0 and range_test < 0:

            numerator = min(range_train, range_test) * -1

            denominator = max(range_train, range_test) * -1

        elif range_train < 0:

            lift = abs(range_train)

            numerator = max(range_train+lift, range_test+lift)

            denominator = min(range_train+lift, range_test+lift)

        elif range_test < 0:

            lift = abs(range_test)

            numerator = max(range_train+lift, range_test+lift)

            denominator = min(range_train+lift, range_test+lift)

        range_percent_diff = round(numerator if denominator == 0 else numerator/denominator, 2)



        v.append([

            col_name, 

            max_train, 

            max_test, 

            max_diff, 

            max_percent_diff, 

            min_train, 

            min_test, 

            min_diff, 

            min_percent_diff, 

            range_train, 

            range_test, 

            range_diff, 

            range_percent_diff

        ])



    return pd.DataFrame(np.array(v), columns=[

        'Column',

        'Max (train)',

        'Max (test)',

        'Max (diff)',

        'Max (% diff)',

        'Min (train)',

        'Min (test)',

        'Min (diff)',

        'Min (% diff)',

        'Range (train)',

        'Range (test)',

        'Range (diff)',

        'Range (% diff)',

    ])
test_train_scale(train_df, test_df)
target_column = train_df.columns.values[-1]

target_column
feature_columns = train_df.columns.values[1:-1]

feature_columns
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score

from statistics import mean



skf = StratifiedKFold(n_splits=10)



X = train_df[feature_columns]

y = train_df[target_column]



scores = []

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    classifier = RandomForestClassifier(n_estimators=100, random_state=8675309)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    scores.append(f1_score(y_test, predictions, average='micro'))

    

mean(scores)
train_df['Soil_Type'] = sum([train_df['Soil_Type' + str(n)] * n for n in range(1, 41)])

train_df['Soil_Type'].hist()
test_df['Soil_Type'] = sum([test_df['Soil_Type' + str(n)] * n for n in range(1, 41)])

test_df['Soil_Type'].hist()
feature_columns
alt_feature_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',

       'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am',

       'Hillshade_Noon', 'Hillshade_3pm',

       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',

       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',

       'Soil_Type']



skf = StratifiedKFold(n_splits=10)



X = train_df[alt_feature_columns]

y = train_df[target_column]



scores = []

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    classifier = RandomForestClassifier(n_estimators=100, random_state=8675309)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    scores.append(f1_score(y_test, predictions, average='micro'))

    

mean(scores)
train_df['Wilderness_Area'] = sum([train_df['Wilderness_Area' + str(n)] * n for n in range(1, 4)])

train_df['Wilderness_Area'].hist()
test_df['Wilderness_Area'] = sum([test_df['Wilderness_Area' + str(n)] * n for n in range(1, 4)])

test_df['Wilderness_Area'].hist()
alt_feature_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',

       'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am',

       'Hillshade_Noon', 'Hillshade_3pm',

       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area',

       'Soil_Type']



skf = StratifiedKFold(n_splits=10)



X = train_df[alt_feature_columns]

y = train_df[target_column]



scores = []

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    classifier = RandomForestClassifier(n_estimators=100, random_state=8675309)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    scores.append(f1_score(y_test, predictions, average='micro'))

    

mean(scores)
ELU_CODES = (

    2702,

    2703,

    2704,

    2705,

    2706,

    2717,

    3501,

    3502,

    4201,

    4703,

    4704,

    4744,

    4758,

    5101,

    5151,

    6101,

    6102,

    6731,

    7101,

    7102,

    7103,

    7201,

    7202,

    7700,

    7701,

    7702,

    7709,

    7710,

    7745,

    7746,

    7755,

    7756,

    7757,

    7790,

    8703,

    8707,

    8708,

    8771,

    8772,

    8776,

)



len(ELU_CODES)
train_df['ELU_Code'] = train_df['Soil_Type'].apply(lambda x: ELU_CODES[x-1])

train_df['ELU_Code'].hist()
test_df['ELU_Code'] = test_df['Soil_Type'].apply(lambda x: ELU_CODES[x-1])

test_df['ELU_Code'].hist()
train_df['ELU_1'] = train_df['ELU_Code'].apply(lambda code: int(str(code)[0]))

train_df['ELU_1'].hist()
test_df['ELU_1'] = test_df['ELU_Code'].apply(lambda code: int(str(code)[0]))

test_df['ELU_1'].hist()
train_df['ELU_2'] = train_df['ELU_Code'].apply(lambda code: int(str(code)[1]))

train_df['ELU_2'].hist()
test_df['ELU_2'] = test_df['ELU_Code'].apply(lambda code: int(str(code)[1]))

test_df['ELU_2'].hist()
train_df['ELU_3'] = train_df['ELU_Code'].apply(lambda code: int(str(code)[2:]))

train_df['ELU_3'].hist()
test_df['ELU_3'] = test_df['ELU_Code'].apply(lambda code: int(str(code)[2:]))

test_df['ELU_3'].hist()
alt_feature_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',

       'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am',

       'Hillshade_Noon', 'Hillshade_3pm',

       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area',

       'Soil_Type', 'ELU_1', 'ELU_2', 'ELU_3']



skf = StratifiedKFold(n_splits=10)



X = train_df[alt_feature_columns]

y = train_df[target_column]



scores = []

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    classifier = RandomForestClassifier(n_estimators=100, random_state=8675309)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    scores.append(f1_score(y_test, predictions, average='micro'))

    

mean(scores)
def direction(azimuth):

    if azimuth < 0 or azimuth > 360:

        raise Exception(f'Azimuth {azimuth} out of bounds')

    if azimuth > 337.5 or azimuth <= 22.5:

        return 1 # 'N'

    if azimuth >= 22.5 and azimuth < 67.5:

        return 2 # 'NE'

    if azimuth >= 67.5 and azimuth < 112.5:

        return 3 # 'E'

    if azimuth >= 112.5 and azimuth < 157.5:

        return 4 # 'SE'

    if azimuth >= 157.5 and azimuth < 202.5:

        return 5 # 'S'

    if azimuth >= 202.5 and azimuth < 247.5:

        return 6 # 'SW'

    if azimuth >= 247.5 and azimuth < 292.5:

        return 7 # 'W'

    if azimuth >= 292.5 and azimuth < 337.5:

        return 8 # 'NW'

    raise Exception(f'Azimuth {azimuth} out of bounds')
train_df['Direction'] = train_df['Aspect'].apply(lambda a: direction(a))

train_df['Direction']
test_df['Direction'] = test_df['Aspect'].apply(lambda a: direction(a))

test_df['Direction']
alt_feature_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',

       'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am',

       'Hillshade_Noon', 'Hillshade_3pm',

       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area',

       'Soil_Type', 'ELU_1', 'ELU_2', 'ELU_3', 'Direction']



skf = StratifiedKFold(n_splits=10)



X = train_df[alt_feature_columns]

y = train_df[target_column]



scores = []

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    classifier = RandomForestClassifier(n_estimators=100, random_state=8675309)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    scores.append(f1_score(y_test, predictions, average='micro'))

    

mean(scores)
import math



train_df['Distance_To_Hydrology'] = (train_df['Horizontal_Distance_To_Hydrology']**2 + train_df['Vertical_Distance_To_Hydrology']**2).apply(lambda n: math.sqrt(n))

test_df['Distance_To_Hydrology'] = (test_df['Horizontal_Distance_To_Hydrology']**2 + test_df['Vertical_Distance_To_Hydrology']**2).apply(lambda n: math.sqrt(n))
alt_feature_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',

       'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am',

       'Hillshade_Noon', 'Hillshade_3pm',

       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area',

       'Soil_Type', 'ELU_1', 'ELU_2', 'ELU_3', 'Direction', 'Distance_To_Hydrology']



skf = StratifiedKFold(n_splits=10)



X = train_df[alt_feature_columns]

y = train_df[target_column]



scores = []

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    classifier = RandomForestClassifier(n_estimators=100, random_state=8675309)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    scores.append(f1_score(y_test, predictions, average='micro'))

    

mean(scores)
tree_count = len(classifier.estimators_)



importances = [0 for _ in range(len(feature_columns))]



for tree in classifier.estimators_:

    for i, importance in enumerate(tree.feature_importances_):

        importances[i] += importance



for i, feature_column in enumerate(alt_feature_columns):

    print(f'{feature_column}: {importances[i]/tree_count}')
from sklearn.ensemble import ExtraTreesClassifier



skf = StratifiedKFold(n_splits=10)



X = train_df[alt_feature_columns]

y = train_df[target_column]



scores = []

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    classifier = ExtraTreesClassifier(n_estimators=100, random_state=8675309)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    scores.append(f1_score(y_test, predictions, average='micro'))

    

mean(scores)
from sklearn.ensemble import GradientBoostingClassifier



skf = StratifiedKFold(n_splits=10)



X = train_df[alt_feature_columns]

y = train_df[target_column]



scores = []

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    classifier = GradientBoostingClassifier(n_estimators=100, random_state=8675309)

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    scores.append(f1_score(y_test, predictions, average='micro'))

    

mean(scores)
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC



X = train_df[alt_feature_columns]

y = train_df[target_column]



scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)



skf = StratifiedKFold(n_splits=10)

scores = []

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    classifier = SVC(random_state=8675309, gamma='scale')

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    scores.append(f1_score(y_test, predictions, average='micro'))

    

mean(scores)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



X = train_df[alt_feature_columns]

y = train_df[target_column]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8675309, stratify=y)



classifier = GridSearchCV(

    n_jobs=3,

    verbose=True,

    cv=5,

    scoring=lambda e, x, y: f1_score(y, e.predict(x), average='micro'),

    estimator=RandomForestClassifier(random_state=8675309),

    param_grid={

        'n_estimators': [250], # [250, 100, 50],

        'criterion': ['entropy'], # ['gini', 'entropy'],

        'min_samples_split': [2], # [2, 6, 12],

        'min_samples_leaf': [1], # [1, 2, 4],

        'max_depth': [None], # [None, 10, 100],

        'bootstrap': [False], # [True, False],

    },

)

classifier.fit(X_train, y_train)

print(f'Best parameters: {classifier.best_params_}\nBest score: {classifier.best_score_}')
classifier = classifier.best_estimator_



test_df['Cover_Type'] = classifier.predict(test_df[alt_feature_columns])

test_df[['Id', 'Cover_Type']].sample(10)
submission = test_df[['Id', 'Cover_Type']]

submission.to_csv('submission.csv', index=False)
import os



os.listdir('.')
with open('submission.csv') as f:

    for _ in range(10):

        print(f.readline(), end='')