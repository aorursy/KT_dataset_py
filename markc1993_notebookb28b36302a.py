# Exploratory Analysis of Renthop Competition



# Credit to Mitchell Spryn 

%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import confusion_matrix



def make_auto_pct_string(values):

    def call_fxn(percent):

        total = sum(values)

        value = int(round(percent*total/100.0))

        return '{p:.2f}%  ({v:d})'.format(p=percent,v=value)

    return call_fxn



def pie_by_interest(dataset, title):

    grouped = dataset.groupby(['interest_level']).size().reset_index(name='count')

    fig = plt.figure(figsize=(8,8))

    ax = plt.axes(aspect=1)

    plt.pie(grouped['count'], labels=grouped['interest_level'], autopct=make_auto_pct_string(grouped['count']))

    plt.title(title)

    plt.legend()

    plt.show()

    

def histogram_by_interest(dataset, fieldname, title, bins=50):

    sns.distplot(dataset[fieldname], kde=False, rug=False, bins=bins, color='y', label='all')

    sns.distplot(dataset[dataset['interest_level'] == 'low'][fieldname], kde=False, rug=False, bins=bins, color = 'g', label='low')

    sns.distplot(dataset[dataset['interest_level'] == 'medium'][fieldname], kde=False, rug=False, bins=bins, color = 'c', label='medium')

    sns.distplot(dataset[dataset['interest_level'] == 'high'][fieldname], kde=False, rug=False, bins=bins, color='r', label='high')

    plt.legend()

    plt.title(title)

    plt.show()

    

pie_by_interest(train_data, 'Distribution of Labels in Train Dataset')





train_data_high = train_data[train_data['interest_level'] == 'high']

train_data_medium = train_data[train_data['interest_level'] == 'medium']

train_data_low = train_data[train_data['interest_level'] == 'low']



description_length = train_data.copy()

description_length['description_length'] = description_length.apply(lambda r: len(r['description']), axis=1)



#count number of zero descriptions

no_description = description_length[description_length['description_length'] == 0]

print('Number of data points without description: {0} ({1}%)'.format(\

            no_description.shape[0], no_description.shape[0]/description_length.shape[0]))



#Show histogram

histogram_by_interest(description_length, 'description_length', 'Description length by interest level')





pie_by_interest(description_length[description_length['description_length'] < 90], 'Short Description by Label')



def guid_to_categorical(series, series_name_pfx):

    guid_to_index = {}

    index_to_guid = {}

    new_index = 0

    number_unique = len(series.unique())

    output_data = [[0 for i in range(0, number_unique, 1)] for j in range(0, len(series))]

    for i in range(0, len(series), 1):

        item = series.iloc[i]

        if item not in guid_to_index:

            guid_to_index[item] = new_index

            index_to_guid[new_index] = item

            new_index += 1

        output_data[i][guid_to_index[item]] = 1

    

    column_names = ['{0}_{1}'.format(series_name_pfx, index_to_guid[i]) for i in range(0, number_unique, 1)]

    

    print('Max index: {0}'.format(new_index))

    return pd.DataFrame(data = output_data, columns = column_names)



def features_to_categorical(features_series, series_name_pfx):

    word_column_index = {}

    index_to_word = {}

    new_index = 0

    for feature_set in features_series:

        for word in set(feature_set):

            if word not in word_column_index:

                word_column_index[word] = new_index

                index_to_word[new_index] = word

                new_index += 1

    

    out_data = [[0 for i in range(0, new_index, 1)] for j in range(0, len(features_series), 1)]

    

    for i in range(0, len(features_series), 1):

        features = features_series.iloc[i]

        for j in range(0, len(features), 1):

            current_feature = features[j]

            out_data[i][word_column_index[current_feature]] += 1

    

    out_data_column_names = ['{0}_{1}'.format(series_name_pfx, index_to_word[i]) for i in range(0, new_index, 1)]

    

    print('Max index: {0}'.format(new_index))

    return pd.DataFrame(data = out_data, columns = out_data_column_names)

    

#Create features explored earlier

print('Generating features...')

input_train_data = train_data.copy()

input_train_data['description_length'] = input_train_data.apply(lambda r: len(r['description']), axis=1)

input_train_data['no_description'] = input_train_data.apply(lambda r: len(r['description']) == 0, axis=1)

input_train_data['number_of_photos'] = input_train_data.apply(lambda r: len(r['photos']), axis=1)



#Create categorical features

print('Generating categorical features...')

categorical_building_id = guid_to_categorical(input_train_data['building_id'], 'building')

categorical_manager_id = guid_to_categorical(input_train_data['manager_id'], 'manager')

categorical_features = features_to_categorical(input_train_data['features'], 'feature')

input_train_data = pd.concat([input_train_data, categorical_features, categorical_building_id, categorical_manager_id], axis=1)



print('Deleting excess columns...')

del input_train_data['created']

del input_train_data['description']

del input_train_data['display_address']

del input_train_data['listing_id']

del input_train_data['photos']

del input_train_data['features']

del input_train_data['street_address']

del input_train_data['building_id']

del input_train_data['manager_id']



print('Generating labels...')

labels = input_train_data['interest_level'].apply(lambda r: 0 if r == 'low' else 1 if r == 'medium' else 2)

del input_train_data['interest_level']



print('Splitting data...')

#Split data using sklearn

x_train, x_test, y_train, y_test = train_test_split(input_train_data, labels, test_size=0.20, random_state=42, stratify=labels)



print('Training model...')

#Train tree model

model = ExtraTreesClassifier()

model = model.fit(x_train, y_train)



#Generate confusion matrix

print('Generating confusion matrix and feature importances...')

predictions = model.predict(x_test)

print("Confusion matrix.")

print(confusion_matrix(y_test, predictions))



#Plot feature importances

print("Feature importances")

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(x_train.shape[1]):

    print("{0}. feature {1} ({2}) ({3})".format(f + 1, indices[f], importances[indices[f]], x_train.columns[indices[f]]))



plt.figure(figsize=(15, 15))

plt.title("Feature importances")

plt.bar(range(x_train.shape[1]), importances[indices], color="r", align="center")

plt.xticks(range(x_train.shape[1]), indices)

plt.xlim([-1, x_train.shape[1]])




