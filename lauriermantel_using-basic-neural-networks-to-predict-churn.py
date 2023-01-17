import numpy

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



seed = 7

numpy.random.seed(seed)
churn_df = pd.read_csv('../input/Churn_Modelling.csv')
churn_df.head()
churnWithoutRowNumber = churn_df.drop(churn_df.columns[[0, 2]], axis=1)
churnWithoutRowNumber.head()
# mapping countries to a specific integer

countries = map(lambda x: x[2], churnWithoutRowNumber.values)

unique_countries = set(countries)

country_hash = {}

for i, c in enumerate(unique_countries): country_hash[c] = i # can save this later when getting insight into neural network

integerMapping = list(map(lambda c: country_hash[c], churnWithoutRowNumber['Geography']))

countriesAsIntegers = churnWithoutRowNumber

countriesAsIntegers['Geography'] = integerMapping



#mapping genders as a specific integer

genders = map(lambda x: x[3], countriesAsIntegers.values)

unique_genders = set(genders)

gender_hash = {}

for i, g in enumerate(unique_genders): gender_hash[g] = i # can save this later when getting insight into neural network

integerMapping = list(map(lambda g: gender_hash[g], countriesAsIntegers['Gender']))

gendersAsIntegers = countriesAsIntegers

gendersAsIntegers['Gender'] = integerMapping



gendersAsIntegers.head()
clean_dataset = gendersAsIntegers.values
clean_dataset[0]
X = clean_dataset[:,0:11].astype(float)

Y = clean_dataset[:,11]
# encode class values (Y) as integers

encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)
encoded_Y
# baseline model

def create_baseline():

    # create model

    model = Sequential()

    model.add(Dense(11, input_dim=11, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=10, batch_size=20, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)

#warnings are because this was originally developed using theano backend, not tensorflow.
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# Model with standard scaled data

numpy.random.seed(seed)

estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=10, batch_size=20, verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# smaller net

def create_smaller():

    model = Sequential()

    model.add(Dense(5, input_dim=11, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, nb_epoch=10, batch_size=20, verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
def create_larger():

    # create model

    model = Sequential()

    model.add(Dense(11, input_dim=11, activation='relu'))

    model.add(Dense(5, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasClassifier(build_fn=create_larger, nb_epoch=10, batch_size=20, verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))