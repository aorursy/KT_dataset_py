!pip install seaborn==0.9.0

!pip install statsmodels==0.10.1
import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



import sklearn

from sklearn import preprocessing as prep



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



%matplotlib inline
x_columns = ['ID', 'Age', 'Workclass', 'Final Weight', 'Education', 'Education Num', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per Week', 'Native Country']

y_column = ['Income']



train_data_raw = pd.read_csv('../input/adult-pmr3508/train_data.csv', names = (x_columns + y_column), na_values = '?', header = 0)

test_data_raw = pd.read_csv('../input/adult-pmr3508/test_data.csv', names = x_columns, na_values = '?', header = 0)
print(train_data_raw.shape)

train_data_raw.head()
numeric_columns = list(train_data_raw.select_dtypes(include = np.number).columns)

train_data_raw[numeric_columns].describe()
numeric_columns.remove('ID')



view_num_df = train_data_raw[numeric_columns].copy()

for column in numeric_columns:

    view_num_df[column] = pd.to_numeric(view_num_df[column])

    

view_num_df['Income'] = train_data_raw['Income']
pp_graph = sns.pairplot(view_num_df, hue="Income", palette="Set2", diag_kind="kde", height=2.5)
def plot_feature_frequencies(data, feature, kind='line'):

    '''

    Plot frequencie for income <=50k and >50k for a specific feature

    '''

    

    less_50 = data.loc[data['Income'] == '<=50K', feature].value_counts().rename('<=50K')

    more_50 = data.loc[data['Income'] == '>50K', feature].value_counts().rename('>50K')

    plot_data = pd.concat([less_50, more_50], axis=1)

    plot_data.plot(xlabel = feature, ylabel = 'Frequency', kind=kind)

    return plot_data
for feature in ['Age', 'Education Num', 'Hours per Week']:

    plot_feature_frequencies(view_num_df, feature)
categoric_columns = list(train_data_raw.select_dtypes(exclude = np.number).columns)

train_data_raw[categoric_columns].describe()
categoric_columns.remove('Education')

view_cat_df = train_data_raw[categoric_columns].copy()
plot_feature_frequencies(view_cat_df, 'Workclass', 'bar').head()
plot_feature_frequencies(view_cat_df, 'Occupation', 'bar').head()
plot_feature_frequencies(view_cat_df, 'Marital Status', 'bar').head()
plot_feature_frequencies(view_cat_df, 'Relationship', 'bar').head()
plot_feature_frequencies(view_cat_df, 'Sex', 'bar').head()
plot_feature_frequencies(view_cat_df, 'Race', 'bar').head()
plot_feature_frequencies(view_cat_df, 'Native Country', 'bar').head()
def count_missing_values(data):

    '''

    Count missing values for each feature and return a sorted DataFrame with the resuls

    '''



    missing_count = []

    for column in data.columns:

        missing_count.append(data[column].isna().sum())

    missing_count = np.asarray(missing_count)

    missing_count = pd.DataFrame({'feature': data.columns, 'count': missing_count,

                                'freq. [%]': 100*missing_count/data.shape[0]}, index=None)

    missing_count.sort_values('count', ascending=False, inplace=True, ignore_index=True)

    return missing_count
display(count_missing_values(train_data_raw).head())

display(count_missing_values(test_data_raw).head())
def handle_missing_values(original_data, fill_options = None, drop_rest = False):

    '''

    Choose what to do with the missing values.

    fill_options is a dictionary where de features are keys, and the values are how to fill the missing data,

    the options are: unknown (fill with 'unknown'), mean (complete with the mean value), most_frequent (complete with most frequent value).

    The rest of the missing data will be dropped.

    '''



    data = original_data.copy()

    if fill_options is not None:

        for feature, action in fill_options.items():

            # print(feature, action)

            if feature not in data.columns:

                # print(feature)

                continue

            if action == 'unknown':

                data[feature].fillna('unknown', inplace=True)

            elif action == 'mean':

                data[feature].fillna(data[feature].mean(), inplace=True)

            elif action == 'most_frequent':

                top = data[feature].describe().top

                data[feature].fillna(top, inplace=True)

                

    data.dropna()



    return data

def data_pipeline(train_raw, test_raw, fill_options = None, drop_columns = ['ID', 'Education']):

    '''

    Prepare the data to be used in the classifier

    '''



    train_data = train_raw.copy()

    test_data = test_raw.copy()



    # Remove duplicate itens from training data

    train_data.drop_duplicates(keep='first', inplace=True)

    

    # Remove unwanted columns

    if drop_columns is not None:

        train_data.drop(drop_columns, axis = 1, inplace=True)

        test_data.drop(drop_columns, axis = 1, inplace=True)



    # Handle the missing values

    train_data = handle_missing_values(train_data, fill_options, drop_rest=True)

    test_data = handle_missing_values(test_data, fill_options)



    # Separate columns types

    numeric_columns = list(test_data.select_dtypes(include = np.number).columns)

    categoric_columns = list(test_data.select_dtypes(exclude = np.number).columns)

    label_column = 'Income'



    # Apply scaler to numeric features

    scaler = prep.RobustScaler()

    scaler.fit(train_data[numeric_columns])

    train_num = scaler.transform(train_data[numeric_columns])

    test_num = scaler.transform(test_data[numeric_columns])



    # Encode the categoric feature into One Hot

    cat_encoder = prep.OneHotEncoder(sparse=False)

    cat_encoder.fit(train_data[categoric_columns])

    train_cat = cat_encoder.transform(train_data[categoric_columns])

    test_cat = cat_encoder.transform(test_data[categoric_columns])



    # Concatenate X arrays

    X_train = np.concatenate((train_num, train_cat), axis=1)

    X_test = np.concatenate((test_num, test_cat), axis=1)



    # Encode the labels

    label_encoder = prep.LabelEncoder()

    Y_train = label_encoder.fit_transform(train_data[label_column])

    

    # Make sure the test and train data have the same number of features

    assert X_train.shape[1] == X_test.shape[1]

    

    # Make sure the train data has the same number of examples in X and Y

    assert X_train.shape[0] == Y_train.shape[0]



    return X_train, Y_train, X_test, label_encoder
fill_options = {'Occupation': 'unknown', 'Workclass': 'unknown'}

drop_columns = ['ID', 'Education', 'Final Weight', 'Native Country']

X_train, Y_train, X_test, label_encoder = data_pipeline(train_data_raw, test_data_raw, fill_options, drop_columns)
print(f'X_train tem shape: {X_train.shape}')

print(f'Y_train tem shape: {Y_train.shape}')

print(f'X_test tem shape: {X_test.shape}')
for k_value in range(5, 45, 5):

    clf = KNeighborsClassifier(k_value, p=1, weights='uniform', n_jobs=-1)

    score = np.mean(cross_val_score(clf, X_train, Y_train, cv=10))

    

    print(f'Para {k_value} vizinhos, a acurácia foi de {100*score:.5f}%')
max_k = 0

max_score = 0.

knn_clf = None



for k_value in range(20, 35):

    clf = KNeighborsClassifier(k_value, p=1, weights='uniform', n_jobs=-1)

    score = np.mean(cross_val_score(clf, X_train, Y_train, cv=10))

    

    if score > max_score:

        max_score = score

        max_k = k_value

        knn_clf = clf

    

print(f'O melhor número de vizinhos encontrado foi {max_k}, com acurácia de {max_score*100:.5f}%')
knn_clf.fit(X_train, Y_train)

Y_hat_test = knn_clf.predict(X_test)

Y_hat_test = label_encoder.inverse_transform(Y_hat_test)
result_data = pd.DataFrame({'income': Y_hat_test})

display(result_data.head())

result_data.to_csv('submission.csv', index = True, index_label = 'Id')