import pandas as pd
def load_data(name, extension=None):

    data = pd.read_csv('../input/{0}_variants{1}'.format(name, '.' + extension if extension is not None else ''))

    text = pd.read_csv('../input/{0}_text{1}'.format(name, '.' + extension if extension is not None else ''), 

                             sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

    data = pd.merge(data, text, on='ID').fillna('')

    

    return data
# training data

train_data = load_data('training')

print(train_data.shape)

train_data.columns.values
# test data

test_data = load_data('test')

print(test_data.shape)

test_data.columns.values
# test data for stage 2

stage2_test_data = load_data('stage2_test', 'csv')

print(stage2_test_data.shape)

stage2_test_data.columns.values
# released labels for stage 2

solution_data = pd.read_csv('../input/stage1_solution_filtered.csv')

solution_data.shape
# data normalization

solution_data_labels = solution_data[['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']]

solution_data_labels.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9]

solution_data['Class'] = solution_data_labels.idxmax(axis=1)

solution_data = solution_data[['ID', 'Class']]

solution_data.head(5)
# merging the released labels with the test data

released_test_data = pd.merge(solution_data, test_data, 

                            left_on=['ID'],

                            right_on=['ID'],

                            how='inner')

released_test_data.shape
released_test_data.head(5)
# extending the train data

extended_train_data = pd.concat([train_data, released_test_data], ignore_index=True)

extended_train_data.shape
def extract_features_from_genes_and_variations(df):

    gene_features = pd.get_dummies(df['Gene'])

    variation_features = pd.get_dummies(df['Variation'])

    features = gene_features.join(variation_features)



    return features
train_and_test_data = pd.concat([train_data, test_data], ignore_index=True)

train_and_test_data.shape
train_features = extract_features_from_genes_and_variations(train_and_test_data)

test_features = extract_features_from_genes_and_variations(stage2_test_data)
common_features = list(set(train_features.columns.values) & set(test_features.columns.values))

train_features = train_features[common_features]

test_features = test_features[common_features]



train_features.shape, test_features.shape
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from scipy.sparse import hstack

import numpy as np
def generate_features_and_labels(train_data, test_data, train_features, test_features):

    '''

    Generating a feature set by combining the two feature sets, 

    extracted from genes and variations and extracted from the texts.

    '''

    x_train = vectorizer.fit_transform(train_data['Text'])

    x_train = hstack((x_train, train_data[['ID']].join(train_features).drop('ID', axis=1).values))

    y_train = train_data['Class']

    x_test = vectorizer.transform(test_data['Text'])

    x_test = hstack((x_test, test_data[['ID']].join(test_features).drop('ID', axis=1).values))

    

    return x_train, y_train, x_test
def predict(x_train, y_train, x_test):

    clf = RandomForestClassifier(n_jobs=3,

                                n_estimators=100,

                                criterion='entropy',

                                random_state=300)



    clf.fit(x_train, y_train)

    return clf.predict(x_test)
def make_submission(test_data):

    submission_data = pd.get_dummies(test_data['predicted_class'])

    submission_data = test_data[['ID']].join(submission_data)

    

    labels = list(range(1, 10))

    submission_data = submission_data[['ID'] + labels]

    submission_data.columns = ['ID'] + ['class' + str(label) for label in labels]



    submission_data.to_csv('submission.csv', index=False)
train, test = train_test_split(extended_train_data, test_size=0.2)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,

                                 stop_words='english')



x_train, y_train, x_test = generate_features_and_labels(train, test, train_features, train_features)

y_test = test['Class']
predicted = predict(x_train, y_train, x_test)

np.mean(predicted == y_test)
x_train = vectorizer.fit_transform(extended_train_data['Text'])

x_train = hstack((x_train, extended_train_data[['ID']].join(train_features).drop('ID', axis=1).values))

y_train = train['Class']

x_test = vectorizer.transform(stage2_test_data['Text'])

x_test = hstack((x_test, stage2_test_data[['ID']].join(test_features).drop('ID', axis=1).values))



x_train, y_train, x_test = generate_features_and_labels(extended_train_data, stage2_test_data, train_features, test_features)



predicted = predict(x_train, y_train, x_test)



stage2_test_data['predicted_class'] = predicted

make_submission(stage2_test_data)