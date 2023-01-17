# Store the training set in a pandas DataFrame

import pandas as pd

train_data = pd.read_csv('../input/train.csv')





def convert_to_indicator_columns(df, column_name):

    """

    Function to convert the list of colums in `column_name` into one indicator variable for each unique value

    """

    unique_values = list(set(df[column_name].tolist()))

    for value in unique_values[:-1]:  # Avoid last category to prevent linear dependence

        indicator_name = '%s: %s' % (column_name, value)

        df[indicator_name] = df[column_name].map(lambda x: x == value)

    del df[column_name]





def clean_data(train_data):

    """

    For all textual columns, either convert to indicator variables or remove. Replace 'N/A' values in Age column.

    """

    data = train_data.copy()

    category_variables = ['Pclass', 'Sex', 'Parch', 'Cabin', 'Embarked']

    for category_variable in category_variables:

        convert_to_indicator_columns(data, category_variable)

    del data['Name']

    del data['Ticket']

    average_age = data['Age'].mean()

    data['Age'].fillna(average_age, inplace=True)

    return data





def normalize(data):

    normal_data = data.copy()

    mean_fare = data['Fare'].mean()

    variance_fare = data['Fare'].std() ** 2

    normal_data['Fare'] = (data['Fare'] - mean_fare)/variance_fare

    normal_data['Fare'].fillna(0, inplace=True)

    return normal_data





cleaned_data = normalize(clean_data(train_data))
# Let's start with linear regression with each data point as a feature and no synthetic features

# To evaluate, we will use k-folds testing with 5 folds

from sklearn.linear_model import LogisticRegression





def train_logistic(train, y_column_name, ignore_columns=[]):

    """

    Train the coefficients of a logistic regression based on training set

    """

    reg = LogisticRegression()

    forbidden_columns = ignore_columns + [y_column_name]

    train_x = train.copy()

    for col in forbidden_columns:

        if col in train_x.columns:

            del train_x[col]

    Y = train.loc[:, y_column_name]

    reg.fit(train_x, Y)

    return reg



def evaluate_logistic(reg, data, y_column_name, ignore_columns=[]):

    if y_column_name:

        forbidden_columns = ignore_columns + [y_column_name]

        params = data.copy()

        for col in forbidden_columns:

            if col in params.columns:

                del params[col]

    else:

        params = data

    return list(reg.predict(params))

    

def test_logistic(reg, test, y_column_name, ignore_columns=[]):

    y_actual = test.loc[:, y_column_name]

    y_predicted = evaluate_logistic(reg, test, y_column_name, ignore_columns)

    success = [(a == p) for a, p in zip(y_actual, y_predicted)]

    return {

        'actual': y_actual,

        'predicted': y_predicted,

        'success': success

    }





reg = train_logistic(cleaned_data, 'Survived', ['PassengerId'])

results = test_logistic(reg, cleaned_data, 'Survived', ['PassengerId'])

print('In-sample accuracy: %.1f%%' % (100 * float(sum(results['success']))/len(results['success'])))
def get_kth_fold(data, k, num_folds):

    """

    Divide data with (k - 1)/k of the data in the train set and 1/k in the test set

    """

    assert k < num_folds  # zero-indexed

    num_data_points = data.shape[0]

    fold_size = int(num_data_points/num_folds)

    train = pd.concat([data.iloc[:fold_size * k, :], data.iloc[fold_size * (k + 1):, :]])

    test = data.iloc[fold_size * k: fold_size * (k + 1), :]

    return train, test





def k_folds_evaluation(data, y_column_name, train_func, test_func, num_folds):

    accuracies = []

    for k in range(num_folds):

        train, test = get_kth_fold(data, k, num_folds)

        evaluator = train_func(train, y_column_name)

        results = test_func(evaluator, test, y_column_name)

        accuracies.append(float(sum(results['success']))/len(results['success']))

    return accuracies





def print_k_folds_results(results):

    print('Out of sample accuracies: ' + ', '.join([('%.1f%%, ' % (r * 100)) for r in results]))

    print('Average out of sample accuracy: %.1f%%' % (100 * sum(results)/float(len(results))))





def out_of_sample_main(data):

    NUM_FOLDS = 10  # Chosen by random dice roll

    results = k_folds_evaluation(data, 'Survived', train_logistic, test_logistic, NUM_FOLDS)

    print_k_folds_results(results)



out_of_sample_main(cleaned_data)
# Age is not a linear variable, so let's break age into buckets and treat them as indicator variables

def convert_age_to_categories(data):

    engineered_data = data.copy()

    engineered_data['IsChild'] = engineered_data['Age'].apply(lambda x: x <= 12)

    engineered_data['IsTeen'] = engineered_data['Age'].apply(lambda x: 12 < x <= 20)

    engineered_data['IsYoungAdult'] = engineered_data['Age'].apply(lambda x: 20 < x <= 40)

    engineered_data['IsOlderAdult'] = engineered_data['Age'].apply(lambda x: 40 < x)

    del engineered_data['Age']

    return engineered_data



engineered_data = convert_age_to_categories(cleaned_data)

out_of_sample_main(engineered_data)



# That didn't improve our accuracy much. What if we add interaction variables between age and gender

print('')

print('Splitting age categories by gender')

def divide_age_categories_by_sex(data):

    engineered_data = data.copy()

    if 'Sex: female' in engineered_data.columns:

        engineered_data['Sex: male'] = engineered_data['Sex: female'].map(lambda x: not x)

    else:

        engineered_data['Sex: female'] = engineered_data['Sex: male'].map(lambda x: not x)

    engineered_data['IsGirl'] = engineered_data['Sex: female'] & engineered_data['IsChild']

    engineered_data['IsBoy'] = engineered_data['Sex: male'] & engineered_data['IsChild']

    engineered_data['IsTeenGirl'] = engineered_data['Sex: female'] & engineered_data['IsTeen']

    engineered_data['IsTeenBoy'] = engineered_data['Sex: male'] & engineered_data['IsTeen']

    engineered_data['IsYoungWoman'] = engineered_data['Sex: female'] & engineered_data['IsYoungAdult']

    engineered_data['IsYoungMan'] = engineered_data['Sex: male'] & engineered_data['IsYoungAdult']

    engineered_data['IsOlderWoman'] = engineered_data['Sex: female'] & engineered_data['IsOlderAdult']

    engineered_data['IsOlderMan'] = engineered_data['Sex: male'] & engineered_data['IsOlderAdult']

    return engineered_data



engineered_data_v2 = divide_age_categories_by_sex(engineered_data)

out_of_sample_main(engineered_data_v2)
# Our accuracy improved modestly, from 80.4% to 82.1%

# From here we will use the test data to submit our first submission



# let's wrap all of our pre-processing into one function we can use

def preprocess_data(data):

    return divide_age_categories_by_sex(convert_age_to_categories(normalize(clean_data(data))))





def evaluate_test_data():

    test_data = pd.read_csv('../input/test.csv')

    y_col_name = 'Survived'

    

    # Arrange data so that columns are identical

    all_data = pd.concat([train_data, test_data])

    all_data['train'] = False

    train_size = train_data.shape[0]

    all_data['train'][0:train_size] = True

    all_data['test'] = all_data['train'].apply(lambda x: not x)

    preprocessed_data = preprocess_data(all_data)

    print('Shape of preprocessed_data: %d, %d' % preprocessed_data.shape)

    preprocessed_train = preprocessed_data[preprocessed_data['train']]

    print('Shape of preprocessed_train: %d, %d' % preprocessed_train.shape)

    preprocessed_test = preprocessed_data[preprocessed_data['test']]

    print('Shape of preprocessed_test: %d, %d' % preprocessed_test.shape)

    del preprocessed_train['train']

    del preprocessed_test['train']

    del preprocessed_train['test']

    del preprocessed_test['test']

    preprocessed_test.drop(['Survived'], inplace=True, axis=1)

    

    # Train a regression and use it to predict the test set

    reg = train_logistic(preprocessed_train, y_col_name, ['passengerId'])

    surived_predications = evaluate_logistic(reg, preprocessed_test, None, ['passengerId', 'Survived'])

    return pd.DataFrame({

       'PassengerId': preprocessed_test['PassengerId'].tolist(),

       'Survived': [int(p) for p in surived_predications]

    })



predictions = evaluate_test_data()

# UNCOMMENT TO SAVE RESULTS TO CSV

# predictions.to_csv('results.csv', index=False)