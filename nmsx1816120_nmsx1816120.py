import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
training_data = pd.read_csv("../input/train.csv")
testing_data = pd.read_csv("../input/test.csv")

print(training_data.head())
print(training_data.describe())
print(training_data['Slope'].describe())
msno.matrix(training_data.sample(200))
features_non_onehot = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                   'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                   'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                   'Horizontal_Distance_To_Fire_Points','Cover_Type']

features_onehot = [col_name for col_name in training_data.columns if col_name not in features_non_onehot]
features_onehot.append('Cover_Type')
sns.pairplot(training_data[features_non_onehot], hue='Cover_Type')
# sns.boxplot(y='Elevation', x='Cover_Type', data=pd.concat([training_data['Cover_Type'], training_data['Elevation']], axis=1))
protential_features = ['Horizontal_Distance_To_Fire_Points','Horizontal_Distance_To_Roadways', 'Hillshade_9am','Horizontal_Distance_To_Hydrology']
sns.boxplot(y='Horizontal_Distance_To_Fire_Points', x='Cover_Type', data=pd.concat([training_data['Cover_Type'], training_data['Horizontal_Distance_To_Fire_Points']], axis=1))
f, ax = plt.subplots()
sns.boxplot(y='Horizontal_Distance_To_Roadways', x='Cover_Type', data=pd.concat([training_data['Cover_Type'], training_data['Horizontal_Distance_To_Roadways']], axis=1))
f, ax = plt.subplots()
sns.boxplot(y='Horizontal_Distance_To_Hydrology', x='Cover_Type', data=pd.concat([training_data['Cover_Type'], training_data['Horizontal_Distance_To_Hydrology']], axis=1))
f, ax = plt.subplots()
sns.boxplot(y='Hillshade_9am', x='Cover_Type', data=pd.concat([training_data['Cover_Type'], training_data['Hillshade_9am']], axis=1))
def prepare_features_and_labels(training, testing):
    training_X = training.drop(['Id', 'Cover_Type'], axis=1)
    training_Y = training[['Cover_Type']].values
    testing_X = testing.drop(['Id'], axis=1)
    testing_ID = testing['Id'].values
    return training_X, training_Y, testing_X, testing_ID
training_features, training_labels, testing_features, testing_ids = prepare_features_and_labels(training_data, testing_data)
def add_relative_features(df):
    df['HS1'] = df.Hillshade_9am / (df.Hillshade_Noon + 1)
    df['HS2'] = df.Hillshade_Noon / (df.Hillshade_3pm + 1)
    df['HS3'] = df.Hillshade_9am / (df.Hillshade_3pm + 1)
    df['HS4'] = df.Hillshade_9am - df.Hillshade_Noon
    df['HS5'] = df.Hillshade_Noon - df.Hillshade_3pm
    df['HS6'] = df.Hillshade_9am - df.Hillshade_3pm
    df['HSM1'] = (df.Hillshade_9am + df.Hillshade_Noon) / 2
    df['HSM2'] = (df.Hillshade_Noon + df.Hillshade_3pm) / 2
    df['HSM3'] = (df.Hillshade_9am + df.Hillshade_3pm) / 2
    df['HSM4'] = (df.Hillshade_9am + df.Hillshade_Noon + df.Hillshade_3pm) / 3

    df['HD1'] = df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Fire_Points
    df['HD2'] = df.Horizontal_Distance_To_Hydrology - df.Horizontal_Distance_To_Fire_Points
    df['HD3'] = df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways
    df['HD4'] = df.Horizontal_Distance_To_Hydrology - df.Horizontal_Distance_To_Roadways
    df['HD5'] = df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Roadways
    df['HD6'] = df.Horizontal_Distance_To_Fire_Points - df.Horizontal_Distance_To_Roadways
    df['HDM'] = (df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways + df.Horizontal_Distance_To_Fire_Points) / 3

    df['EV1'] = df.Elevation + df.Vertical_Distance_To_Hydrology
    df['EV2'] = df.Elevation - df.Vertical_Distance_To_Hydrology
    
#     df['D1'] = np.divide(df.Elevation, np.sin(df.Slope))
#     df['D1'] = df.D1.map(lambda x: 0 if np.isinf(x) else x)
    df['D2'] = np.sqrt(np.power(df.Horizontal_Distance_To_Hydrology, 2) + np.power(df.Vertical_Distance_To_Hydrology, 2))
    df['D2'] = df.D2.map(lambda x: 0 if np.isinf(x) else x)

    return df

training_features = add_relative_features(training_features)
testing_features = add_relative_features(testing_features)
features_non_onehot = ['Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Hillshade_9am',
                       'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
                       'HS1','HS2','HS3','HS4','HS5','HS6','HSM1','HSM2','HSM3','HSM4',
                       'HD1','HD2','HD3','HD4','HD5','HD6','HDM',
                       'EV1','EV2',
                       'D2']
def normalize_features(train_X, test_X):
    scaler = StandardScaler()
    scaler.fit(train_X)
    
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    
def do_some_tricks(df):
    df = df.drop('Elevation', axis=1)
    return df
normalize_features(training_features[features_non_onehot], testing_features[features_non_onehot])

training_features = do_some_tricks(training_features)
testing_features = do_some_tricks(testing_features)
def get_suitable_features(features, labels):
#     lgb = LGBMClassifier(n_estimators=100, max_depth=3)
    etc = ExtraTreesClassifier(n_estimators=100)

    rfecv = RFECV(estimator=etc, step=1, cv=StratifiedKFold(3, random_state=0),
                  scoring='accuracy', verbose=1)

    rfecv.fit(np.array(features), np.array(labels))

    print("Optimal number of features : %d" % rfecv.n_features_)

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (roc auc)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    return rfecv

rfecv = get_suitable_features(training_features, training_labels)

training_features = rfecv.transform(training_features)
testing_features = rfecv.transform(testing_features)
def begin_fit(features, labels):
    train_X, test_X, train_Y, test_Y = train_test_split(np.array(features), np.array(labels), test_size=0.05, random_state=0, \
                                                        stratify=np.array(labels), shuffle=True)
    
    rfc1 = RandomForestClassifier(n_estimators=500,
                                 random_state=0)

    # paramater getting from tpot classifier
    etc1 = ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.3, min_samples_leaf=1, min_samples_split=3, n_estimators=100, random_state=0)
#     etc1 = ExtraTreesClassifier(n_estimators=500, random_state=0)

    lbc1 = LGBMClassifier(n_estimators=500,
                         learning_rate=0.01,
                          num_leaves=50,
                         random_state=0)
    
    lr1 = LogisticRegression()
    svc1 = SVC(kernel='linear')
    svc2 = SVC()
    
    rfc1.fit(train_X, train_Y)
    y_pred = rfc1.predict(test_X)
    score1 = accuracy_score(test_Y, y_pred)
    print('Predict by random forest classifier: {}'.format(score1))
    
    etc1.fit(train_X, train_Y)
    y_pred = etc1.predict(test_X)
    score2 = accuracy_score(test_Y, y_pred)
    print('Predict by extra trees classifier: {}'.format(score2))

    lbc1.fit(train_X, train_Y)
    y_pred = lbc1.predict(test_X)
    score3 = accuracy_score(test_Y, y_pred)
    print('Predict by lightgbm classifier: {}'.format(score3))
    
    lr1.fit(train_X, train_Y)
    y_pred = lr1.predict(test_X)
    score4 = accuracy_score(test_Y, y_pred)
    print('Predict by logistic regression classifier: {}'.format(score4))
    
    svc1.fit(train_X, train_Y)
    y_pred = svc1.predict(test_X)
    score5 = accuracy_score(test_Y, y_pred)
    print('Predict by svc classifier: {}'.format(score5))
    
    svc2.fit(train_X, train_Y)
    y_pred = svc2.predict(test_X)
    score6 = accuracy_score(test_Y, y_pred)
    print('Predict by linear svc classifier: {}'.format(score6))
    
    scores = [score1, score2, score3, score4, score5, score6]

    print('final validation score: {}'.format(scores))
    
    classifiers = [rfc1, etc1, lbc1, lr1, svc1, svc2]

    return classifiers[scores.index(max(scores))]
my_classifier = begin_fit(training_features, training_labels)
def begin_predict(classifier, features, ids):
    y_pred = classifier.predict(features)
    
    print('[*] Save to CSV...')
    sub = pd.DataFrame()
    sub['Id'] = ids
    sub['Cover_Type'] = y_pred
    sub.to_csv('submission.csv', index=False)
begin_predict(my_classifier, testing_features, testing_ids)