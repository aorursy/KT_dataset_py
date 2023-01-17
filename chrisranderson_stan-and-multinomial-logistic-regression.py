import pystan
import pickle
import os.path, time
import numpy as np
import pandas as pd
from stan_utils import *

age_transforms = {'1 year': 12, '2 years': 24, '3 weeks': .75, '1 month': 1, '5 months': 5, '4 years': 48, '3 months': 3, '2 weeks': .5, '2 months': 2, '10 months': 10, '6 months': 6, '5 years': 60, '7 years': 84, '3 years': 36, '4 months': 4, '12 years': 48, '9 years': 9 * 12, '6 years': 72, '1 weeks': .25, '11 years': 11*12, '4 weeks': 1, '7 months': 7*12, '8 years': 8*12, '11 months': 11, '4 days': .12, '9 months': 9, '8 months': 8, '15 years': 15*12, '10 years': 10*12, '1 week': .25, '0 years': 0, '14 years': 14*12, '3 days': .12, '6 days': .20, '5 days': .18, '5 weeks': 1.25, '2 days': .04, '16 years': 16*12, '1 day': .02, '13 years': 13*12, '17 years': 17*12, '18 years': 18*12, '19 years': 19*12, '20 years': 20*12, '22 years': 22*12 }

conversion_maps = {}

def make_conversion_maps(train, test):
    for column in train.columns:
        if column == 'id' or column == 'datetime':
            continue

        values = train[column].unique()

        if column in test.columns:
            try:
                values = np.hstack((test[column].unique(), values))
            except ValueError as e:
                print('e', e)
                pass

        indices = range(len(values))
        conversion_maps[column] = dict(zip(values, indices))

    return conversion_maps

def append_percentage_dead(df):
    percentages = []

    i = 0
    total = df.shape[0]
    for index, row in df.iterrows():
        i += 1
        if i % 1500 == 0:
            print('percent dead: ', i/total)

        animal_type = row['animal_type']
        age = row['age_at_outcome']

        if animal_type == 'Cat':
            percentages.append(float(age) / (180.0))
        else:
            percentages.append(float(age) / (138.0))

    df['percent_dead'] = percentages

    return df

def numericize_column(df, column):
    unique_values = df[column].unique()
    indices = range(len(unique_values))

    conversion_map = conversion_maps[column]

    return df.applymap(lambda x: 1+int(conversion_map[x]) if x in conversion_map else x)

def prep_all_data(df):
    df = df.applymap(lambda x: x if (x not in age_transforms) else age_transforms[x]); print('map applied.')
    df = append_percentage_dead(df).drop(['name'], axis=1); print('percent dead done.')
    try:
        df = numericize_column(df, 'outcome')
    except KeyError:
        pass

    df = df.fillna(0)
    df = numericize_column(df, 'animal_type'); print('animal_type done.')
    df = numericize_column(df, 'sex_at_outcome')
    df = numericize_column(df, 'breed')
    df = numericize_column(df, 'color')
    df = df.fillna(0)
    return df

def fit_data(all_data, features):
    train_data = all_data[:train_size]
    train_labels = train_data['outcome']

    stan_model = get_model()

    data = {
        'class_count': len(np.unique(all_data['outcome'])),
        'instance_count': train_size,
        'feature_count': len(features),
        'labels': train_labels,
        'train_data': train_data[features].as_matrix()
    }

    fit = stan_model.sampling(
        data=data,
        iter=100,
        chains=4
    )

    beta = np.mean(fit.extract()['beta'], axis=0)

    return beta

def print_validation_accuracy(data, beta, features):
    successes = []

    i = 0
    total = data.shape[0]
    for index, instance in data.iterrows():
        i += 1
        if i % 1500 == 0:
            print('validation accuracy: ', i/total)

        label = instance['outcome']
        scores = beta.dot(instance[features])
        prediction = np.argmax(scores) + 1
        successes.append(1 if prediction == label else 0)

    print('Validation accuracy:', np.mean(successes))

def make_test_predictions(features):
    test_data = prep_all_data(pd.read_csv('test.csv'))[features]

    predictions = []

    i = 0
    total = test_data.shape[0]
    for index, instance in test_data.iterrows():
        i += 1
        if i % 1500 == 0:
            print('test_predictions', i/total)

        try:
            scores = beta.dot(instance[features])
            prediction = np.argmax(scores) + 1
            predictions.append(prediction)
        except TypeError as e:
            print('instance, features', instance, features)
            predictions.append(1)
            print('e', e)

    predictions = pd.DataFrame(predictions)
    numbers_to_labels = {value:key for key, value in conversion_maps['outcome'].items()}
    predictions = predictions.applymap(lambda x: numbers_to_labels[x-1])

    return predictions

def write_to_file(predictions):
    # Write to file
    output = pd.read_csv('sample_submission.csv', ',')
    output['Adoption'] = 0

    for index, row in output.iterrows():
        label = predictions.iloc[[index]]
        output.set_value(index, label, 1)

    output = output.set_index('ID')

    output.to_csv('actual_submission.csv')

train_size = 200

original_data = pd.read_csv('train.csv', ','); print('csv read.')

print(len(original_data[original_data['outcome'] == 'Adoption']) / len(original_data))
conversion_maps = make_conversion_maps(original_data, pd.read_csv('test.csv'))
all_data = prep_all_data(original_data)

features = ['percent_dead', 'animal_type', 'sex_at_outcome']

beta = fit_data(all_data, features)

print_validation_accuracy(all_data[train_size:train_size + 10000], beta, features)

predictions = make_test_predictions(features)

write_to_file(predictions)
import os.path, time, pystan
import pickle


def compile_model():
    with open('model.stan', 'r') as my_file:
        model_code = my_file.read()

    stan_model = pystan.StanModel(model_code=model_code)

    print('Compilation finished, writing to pickle.')

    with open('model.pkl', 'wb') as f:
        pickle.dump(stan_model, f)

    return stan_model

def get_model():
    try:
        model_edited = os.path.getmtime('model.stan')
        pickle_edited = os.path.getmtime('model.pkl')

        if model_edited > pickle_edited:
            print('Model has been changed, recompiling.')
            return compile_model()

        print('Trying pre-compiled model.')
        return pickle.load(open('model.pkl', 'rb'))

    except (FileNotFoundError, EOFError) as exception:
        print('Pre-compiled model not found. Compiling model.')

        return compile_model()