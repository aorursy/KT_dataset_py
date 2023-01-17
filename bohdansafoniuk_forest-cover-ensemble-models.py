import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import random
import itertools
import warnings
warnings.filterwarnings('ignore')

from time import time
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

%pylab inline
pylab.rcParams['figure.figsize'] = (10, 6)
sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')

Y_train = train['Cover_Type']
X_train = train.drop('Cover_Type', axis=1)
test_id = X_test['Id']

del train
def preprocess(df):
    df = df.drop('Id', axis=1)

    wilderness_types = [f'Wilderness_Area{i}' for i in range(1, 5)]
    df[wilderness_types] = df[wilderness_types].multiply(range(1, 5), axis=1)
    df['wilderness_types'] = df[wilderness_types].sum(axis=1)
    df = df.drop(wilderness_types, axis=1)

    soil_types = [f'Soil_Type{i}' for i in range(1, 41)]
    df[soil_types] = df[soil_types].multiply(range(1, 41), axis=1)
    df['soil_type'] = df[soil_types].sum(axis=1)
    df = df.drop(soil_types, axis=1)
    
    df['HF1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points'])
    df['HF2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['HR1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
    df['HR2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
    df['FR1'] = abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])
    df['FR2'] = abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])

    df['slope_hyd'] = (df['Horizontal_Distance_To_Hydrology'] ** 2 + df['Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
    df['slope_hyd'] = df['slope_hyd'].map(lambda x: 0 if np.isinf(x) else x)

    df['Mean_Amenities'] = (df['Horizontal_Distance_To_Fire_Points'] +
                          df['Horizontal_Distance_To_Hydrology'] +
                          df['Horizontal_Distance_To_Roadways']) / 3  
    df['Mean_Fire_Hyd'] = (df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Hydrology']) / 2

    return df
def get_random_state():
    return random.randint(1, 1000)

def get_split_coef():
    return round(random.uniform(0.1, 0.35), 2)
X_train = preprocess(X_train)
X_test = preprocess(X_test)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
target_names = {
    1: 'Spruce/Fir',
    2: 'Lodgepole Pine',
    3: 'Ponderosa Pine',
    4: 'Cottonwood/Willow',
    5: 'Aspen',
    6: 'Douglas-fir',
    7: 'Krummholz'
}
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
rand_state = get_random_state()
split_coef = get_split_coef()

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                      test_size=split_coef,
                                                      random_state=rand_state)
print(f'training samples: {x_train.shape[0]}, validating samples: {x_valid.shape[0]}')

model_rf1 = RandomForestClassifier(n_estimators=2000, n_jobs=3, criterion='entropy', random_state=rand_state)

start = time.time()

model_rf1.fit(x_train, y_train)
model_rf1_output = pd.DataFrame({'Id': test_id, 'Cover_Type': model_rf1.predict(X_test)})

print(f'Runtime for RandomForestClassifier1: {time.time() - start}')
print(f'Total accuracy: {accuracy_score(y_valid, model_rf1.predict(x_valid))}')
print(classification_report(y_valid, model_rf1.predict(x_valid), target_names=list(target_names.values())))
model_rf1_output.to_csv('rf1_predictions.csv', index=False)

conf_matr = confusion_matrix(y_valid, model_rf1.predict(x_valid))
plot_confusion_matrix(conf_matr, classes = target_names.values())
rand_state = get_random_state()
split_coef = get_split_coef()

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                      test_size=split_coef,
                                                      random_state=rand_state)
print(f'training samples: {x_train.shape[0]}, validating samples: {x_valid.shape[0]}')

model_rf2 = RandomForestClassifier(n_estimators=2000, n_jobs=3, criterion='entropy', random_state=rand_state)

start = time.time()

model_rf2.fit(x_train, y_train)
model_rf2_output = pd.DataFrame({'Id': test_id, 'Cover_Type': model_rf2.predict(X_test)})

print(f'Runtime for RandomForestClassifier2: {time.time() - start}')
print(f'Total accuracy: {accuracy_score(y_valid, model_rf2.predict(x_valid))}')
print(classification_report(y_valid, model_rf2.predict(x_valid), target_names=list(target_names.values())))
model_rf2_output.to_csv('rf2_predictions.csv', index=False)

conf_matr = confusion_matrix(y_valid, model_rf2.predict(x_valid))
plot_confusion_matrix(conf_matr, classes = target_names.values())
rand_state = get_random_state()
split_coef = get_split_coef()

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                      test_size=split_coef,
                                                      random_state=rand_state)
print(f'training samples: {x_train.shape[0]}, validating samples: {x_valid.shape[0]}')

model_rf3 = RandomForestClassifier(n_estimators=2000, n_jobs=3, criterion='entropy', random_state=rand_state)

start = time.time()

model_rf3.fit(x_train, y_train)
model_rf3_output = pd.DataFrame({'Id': test_id, 'Cover_Type': model_rf3.predict(X_test)})

print(f'Runtime for RandomForestClassifier3: {time.time() - start}')
print(f'Total accuracy: {accuracy_score(y_valid, model_rf3.predict(x_valid))}')
print(classification_report(y_valid, model_rf3.predict(x_valid), target_names=list(target_names.values())))
model_rf3_output.to_csv('rf3_predictions.csv', index=False)

conf_matr = confusion_matrix(y_valid, model_rf3.predict(x_valid))
plot_confusion_matrix(conf_matr, classes = target_names.values())
rand_state = get_random_state()
split_coef = get_split_coef()

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                      test_size=split_coef,
                                                      random_state=rand_state)
print(f'training samples: {x_train.shape[0]}, validating samples: {x_valid.shape[0]}')

model_extra1 = ExtraTreesClassifier(n_estimators=2000, n_jobs=3, bootstrap=True, random_state=rand_state)

start = time.time()

model_extra1.fit(x_train, y_train)
model_extra1_output = pd.DataFrame({'Id': test_id, 'Cover_Type': model_extra1.predict(X_test)})

print(f'Runtime for ExtraTreesClassifier: {time.time() - start}')
print(f'Total accuracy: {accuracy_score(y_valid, model_extra1.predict(x_valid))}')
print(classification_report(y_valid, model_extra1.predict(x_valid), target_names=list(target_names.values())))
model_extra1_output.to_csv('extra1_predictions.csv', index=False)

conf_matr = confusion_matrix(y_valid, model_extra1.predict(x_valid))
plot_confusion_matrix(conf_matr, classes = target_names.values())
rand_state = get_random_state()
split_coef = get_split_coef()

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                      test_size=split_coef,
                                                      random_state=rand_state)
print(f'training samples: {x_train.shape[0]}, validating samples: {x_valid.shape[0]}')

model_extra2 = ExtraTreesClassifier(n_estimators=2000, n_jobs=3, bootstrap=True, random_state=rand_state)

start = time.time()

model_extra2.fit(x_train, y_train)
model_extra2_output = pd.DataFrame({'Id': test_id, 'Cover_Type': model_extra2.predict(X_test)})

print(f'Runtime for ExtraTreesClassifier: {time.time() - start}')
print(f'Total accuracy: {accuracy_score(y_valid, model_extra2.predict(x_valid))}')
print(classification_report(y_valid, model_extra2.predict(x_valid), target_names=list(target_names.values())))
model_extra2_output.to_csv('extra2_predictions.csv', index=False)

conf_matr = confusion_matrix(y_valid, model_extra2.predict(x_valid))
plot_confusion_matrix(conf_matr, classes = target_names.values())
rand_state = get_random_state()
split_coef = get_split_coef()

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                      test_size=split_coef,
                                                      random_state=rand_state)
print(f'training samples: {x_train.shape[0]}, validating samples: {x_valid.shape[0]}')

model_extra3 = ExtraTreesClassifier(n_estimators=2000, n_jobs=3, bootstrap=True, random_state=rand_state)

start = time.time()

model_extra3.fit(x_train, y_train)
model_extra3_output = pd.DataFrame({'Id': test_id, 'Cover_Type': model_extra3.predict(X_test)})

print(f'Runtime for ExtraTreesClassifier: {time.time() - start}')
print(f'Total accuracy: {accuracy_score(y_valid, model_extra3.predict(x_valid))}')
print(classification_report(y_valid, model_extra3.predict(x_valid), target_names=list(target_names.values())))
model_extra3_output.to_csv('extra3_predictions.csv', index=False)

conf_matr = confusion_matrix(y_valid, model_extra3.predict(x_valid))
plot_confusion_matrix(conf_matr, classes = target_names.values())
rand_state = get_random_state()
split_coef = get_split_coef()

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                      test_size=split_coef,
                                                      random_state=rand_state)
print(f'training samples: {x_train.shape[0]}, validating samples: {x_valid.shape[0]}')

model_extra4 = ExtraTreesClassifier(n_estimators=2000, n_jobs=3, bootstrap=True, random_state=rand_state)

start = time.time()

model_extra4.fit(x_train, y_train)
model_extra4_output = pd.DataFrame({'Id': test_id, 'Cover_Type': model_extra4.predict(X_test)})

print(f'Runtime for ExtraTreesClassifier: {time.time() - start}')
print(f'Total accuracy: {accuracy_score(y_valid, model_extra4.predict(x_valid))}')
print(classification_report(y_valid, model_extra4.predict(x_valid), target_names=list(target_names.values())))
model_extra4_output.to_csv('extra4_predictions.csv', index=False)

conf_matr = confusion_matrix(y_valid, model_extra4.predict(x_valid))
plot_confusion_matrix(conf_matr, classes = target_names.values())
rand_state = get_random_state()
split_coef = get_split_coef()

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                      test_size=split_coef,
                                                      random_state=rand_state)
print(f'training samples: {x_train.shape[0]}, validating samples: {x_valid.shape[0]}')

model_extra5 = ExtraTreesClassifier(n_estimators=2000, n_jobs=3, bootstrap=True, random_state=rand_state)

start = time.time()

model_extra5.fit(x_train, y_train)
model_extra5_output = pd.DataFrame({'Id': test_id, 'Cover_Type': model_extra5.predict(X_test)})

print(f'Runtime for ExtraTreesClassifier: {time.time() - start}')
print(f'Total accuracy: {accuracy_score(y_valid, model_extra5.predict(x_valid))}')
print(classification_report(y_valid, model_extra5.predict(x_valid), target_names=list(target_names.values())))
model_extra5_output.to_csv('extra5_predictions.csv', index=False)

conf_matr = confusion_matrix(y_valid, model_extra5.predict(x_valid))
plot_confusion_matrix(conf_matr, classes = target_names.values())
rand_state = get_random_state()
split_coef = get_split_coef()

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,
                                                      test_size=split_coef,
                                                      random_state=rand_state)
print(f'training samples: {x_train.shape[0]}, validating samples: {x_valid.shape[0]}')

model_extra6 = ExtraTreesClassifier(n_estimators=2000, n_jobs=3, bootstrap=True, random_state=rand_state)

start = time.time()

model_extra6.fit(x_train, y_train)
model_extra6_output = pd.DataFrame({'Id': test_id, 'Cover_Type': model_extra6.predict(X_test)})

print(f'Runtime for ExtraTreesClassifier: {time.time() - start}')
print(f'Total accuracy: {accuracy_score(y_valid, model_extra6.predict(x_valid))}')
print(classification_report(y_valid, model_extra6.predict(x_valid), target_names=list(target_names.values())))
model_extra6_output.to_csv('extra6_predictions.csv', index=False)

conf_matr = confusion_matrix(y_valid, model_extra6.predict(x_valid))
plot_confusion_matrix(conf_matr, classes = target_names.values())
final_predictions = pd.DataFrame({'Id': test_id,
                                  'Cover_Type_rf1': model_rf1_output['Cover_Type'],
                                  'Cover_Type_rf2': model_rf2_output['Cover_Type'],
                                  'Cover_Type_rf3': model_rf3_output['Cover_Type'],
                                  'Cover_Type_extra1': model_extra1_output['Cover_Type'],
                                  'Cover_Type_extra2': model_extra2_output['Cover_Type'],
                                  'Cover_Type_extra3': model_extra3_output['Cover_Type'],
                                  'Cover_Type_extra4': model_extra4_output['Cover_Type'],
                                  'Cover_Type_extra5': model_extra5_output['Cover_Type'],
                                  'Cover_Type_extra6': model_extra6_output['Cover_Type']})
predictions = ['Cover_Type_rf1', 'Cover_Type_rf2', 'Cover_Type_rf3',
               'Cover_Type_extra1', 'Cover_Type_extra2', 'Cover_Type_extra3',
               'Cover_Type_extra4', 'Cover_Type_extra5', 'Cover_Type_extra6']
final_predictions['Cover_Type'] = stats.mode(final_predictions[predictions], axis=1).mode
final_predictions = final_predictions.drop(predictions, axis=1)
final_predictions.to_csv('final_predictions.csv', index=False)