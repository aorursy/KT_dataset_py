import numpy as np

import pandas as pd

import os



# printing path to data file

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print('DATA PATH:  {}'.format(os.path.join(dirname, filename)))



# reading data as dataframe, printing shape, printing features list and type,

# printing number of NaNs per feature, printing first 3 rows

data = pd.read_csv('/kaggle/input/austin-weather/austin_weather.csv', parse_dates=['Date'])

print('DATA SHAPE: {}'.format(data.shape))

print('------------------------------------------------------------------')

print('FEATURES LIST AND TYPES:\n{}'.format(data.dtypes))

print('------------------------------------------------------------------')

print('NUMBER OF NaNs PER FEATURE:\n{}'.format(data.isnull().sum(axis=0)))

data.head(3)
# replacing ' ' with 'Clear'

data['Events'] = data['Events'].replace(' ', 'Clear')



# adding feature 'DayOfYear'

days_of_year = [date.dayofyear for date in data['Date']]

data['DayOfYear'] = days_of_year



data[['Date', 'DayOfYear', 'Events']].head(3)
from sklearn.impute import SimpleImputer



# printing '-' and 'T' count before removing them

number_of_nans = data.stack().value_counts()['-']

number_of_ts = data.stack().value_counts()['T']

print('BEFORE imputing and replacing, in the dataset there are {} "-" and {} "Ts"'.format(number_of_nans,number_of_ts))



# replacing '-' with NaN to later use SimpleImputer

data = data.replace('-', float('NaN'))

data['PrecipitationSumInches'] = data['PrecipitationSumInches'].replace('T', 0.0005)



# list of features to be checked for NaNs, imputing

to_be_imputed = list(set(data.columns)-set(['Date', 'Events']))



imp = SimpleImputer(missing_values=float('NaN'), strategy='mean')

data[to_be_imputed] = imp.fit_transform(data[to_be_imputed])



# printing '-' and 'T' count after removing them

if '-' not in data.stack().value_counts(): number_of_nans = 0

if 'T' not in data.stack().value_counts(): number_of_ts = 0

print('AFTER imputing and replacing, in the dataset there are {} "-" and {} "Ts"'.format(number_of_nans,number_of_ts))
from sklearn.preprocessing import MinMaxScaler



# not all features need to be scaled: 'Date' and 'Events' don't need scaling. 

# 'Date' is just there for intepretability and will not be used for classification

# in its place 'DayOfYear' will be used. 

# 'Events' represents the class. 

# Therefore i prepare the list of feature names that need scaling to scale just those

# i keep the original data untouched to later make a comparison

scaled_data = data.copy()

to_be_scaled = list(set(data.columns)-set(['Date', 'Events']))

scaler = MinMaxScaler()

scaler.fit(data[to_be_scaled])

scaled_data[to_be_scaled] = scaler.transform(data[to_be_scaled])
from sklearn.preprocessing import LabelEncoder



# encoding

preprocessed_data = scaled_data

encoder = LabelEncoder()

encoder.fit(scaled_data['Events'])

preprocessed_data['Events'] = encoder.transform(scaled_data['Events'])



# printing

classes_occurrences = preprocessed_data['Events'].value_counts().to_frame()

classes_occurrences['Class'] = encoder.inverse_transform(classes_occurrences.index)

classes_occurrences = classes_occurrences.sort_index(axis=0)

print(classes_occurrences)
# relabeling

preprocessed_data['Events'] = preprocessed_data['Events'].replace([1], 0)

preprocessed_data['Events'] = preprocessed_data['Events'].replace([2,5,6], 1)

preprocessed_data['Events'] = preprocessed_data['Events'].replace([3,4,7,8], 2)



# printing classes and number of occurrences

classes_occurrences = preprocessed_data['Events'].value_counts().to_frame()

classes_occurrences['Class'] = ['Clear', 'Rain', 'Thunderstorm']

classes_occurrences = classes_occurrences.sort_index(axis=0)

print(classes_occurrences)
import matplotlib.pyplot as plt

from matplotlib import cm

%matplotlib inline

plt.style.use('seaborn-darkgrid')



x_axis_original = data['HumidityHighPercent']

y_axis_original = data['PrecipitationSumInches']



x_axis_scaled = preprocessed_data['HumidityHighPercent']

y_axis_scaled = preprocessed_data['PrecipitationSumInches']



rainbow = cm.get_cmap('rainbow', 3)



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

axes[0].scatter(x_axis_original, y_axis_original, c=preprocessed_data['Events'], cmap=rainbow)

right_plot = axes[1].scatter(x_axis_scaled, y_axis_scaled, c=preprocessed_data['Events'], cmap=rainbow)

cbar = fig.colorbar(right_plot, ticks=[0, 1, 2])

cbar.ax.set_yticklabels(classes_occurrences['Class'])

plt.tight_layout()
from sklearn.decomposition import PCA



features = list(set(preprocessed_data.columns)-set(['Date', 'Events']))



# the set of first 10 features extracted with PCA

pca = PCA(n_components=len(features))

pca.fit(preprocessed_data[features])

X_PCA = pd.DataFrame(data=pca.transform(preprocessed_data[features]))



y = preprocessed_data['Events']



print('X_PCA shape: {}'.format(X_PCA.shape))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# function i'll use later to asses the model's performances

def evaluate(y_test, y_pred, avg):

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, average=avg)

    recall = recall_score(y_test, y_pred, average=avg)

    f1 = f1_score(y_test, y_pred, average=avg)



    return (accuracy, precision, recall, f1)
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC



# for storing scores

accuracies = []

precisions = [[],[],[]]

recalls = [[],[],[]]

f1s = [[],[],[]]

# different ways of calculating the average

avgs = ['macro', 'micro', 'weighted']



# each iteration increases by one the number of principal components used for the training

for i in range(2, 21):

    X_train, X_test, y_train, y_test = train_test_split(X_PCA[X_PCA.columns[:i]], y, test_size=0.3, random_state=0) # 70 - 30 split

    clf = SVC(gamma='auto', kernel='linear')

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    

    # saving scores for later plotting

    for j in range(len(avgs)):

        acc, prec, rec, f1 = evaluate(y_test, predictions, avgs[j])

        precisions[j].append(prec)

        recalls[j].append(rec)

        f1s[j].append(f1)   

        if j==0:

            accuracies.append(acc)





fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 8))

lines = []

for i in range(3):

    l1=axes[i].plot(range(2,21), accuracies, color='red')[0]

    l2=axes[i].plot(range(2,21), precisions[i], color='blue')[0]

    l3=axes[i].plot(range(2,21), recalls[i], color='green')[0]

    l4=axes[i].plot(range(2,21), f1s[i], color='orange')[0]

    lines = [l1, l2, l3, l4]

    axes[i].set_title('Scores calculated with {} average'.format(avgs[i]))

    axes[i].set(xlabel='Number of PCA components')

fig.legend(lines, ['Accuracy', 'Precision', 'Recall', 'F1'], loc='center right', fontsize=16)

plt.tight_layout()
X_train, X_test, y_train, y_test = train_test_split(X_PCA[X_PCA.columns[:8]], y, test_size=0.3, random_state=0) # 70 - 30 split

clf = SVC(gamma='auto', kernel='linear')

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

confusion_matrix = pd.crosstab(predictions, y_test, rownames=['Actual'], colnames=['Predictions'])



import seaborn as sn



sn.set(font_scale=1.4)

sn.heatmap(confusion_matrix, annot=True,annot_kws={"size": 12}, fmt="d", cmap="YlGnBu")

plt.show()
from sklearn.model_selection import GridSearchCV, KFold

from time import gmtime, strftime



# converting to numpy array to do THIS later

X = X_PCA[X_PCA.columns[:15]].to_numpy()



# sets of parameters to be tested

kernels = ['rbf', 'poly', 'linear', 'sigmoid']

cs = [2**i for i in range(1, 5)]

degrees = [i for i in range(2, 5)]

gammas = ['auto', 'scale']

coef0s = [2**i for i in range(1, 5)]

parameters = {'kernel': kernels, 'C': cs, 'degree': degrees, 'gamma':gammas, 'coef0': coef0s}

scores = []



folds = KFold(n_splits=5, shuffle=True, random_state=0)



i = 0

print('Starting at: {}'.format(strftime("%H:%M:%S", gmtime())))

print('')

for train_index, test_index in folds.split(X):

    print('Working on {}-th fold... '.format(i))

    i+=1

    

    # THIS can't be (this easily) done with a DataFrame

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]



    svc = SVC()

    clf = GridSearchCV(svc, parameters, cv=5, scoring='accuracy')

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)



    scores.append({'best_score': clf.best_score_, 'best_param': clf.best_params_})

print('')

print('Ending at: {}'.format(strftime("%H:%M:%S", gmtime())))

print('')



max_score = 0

best_params = {}

for score in scores:

    acc = score['best_score']

    params = score['best_param']

    if(acc > max_score):

        max_score = acc

        best_params = params

    print('Best_Score {}'.format(acc))

    print('Parameters {}'.format(params))
X_train, X_test, y_train, y_test = train_test_split(X_PCA[X_PCA.columns[:15]], y, test_size=0.2, random_state=0)



clf = SVC(kernel=best_params['kernel'], C=best_params['C'], degree=best_params['degree'], gamma=best_params['gamma'], coef0=best_params['coef0'])

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)



confusion_matrix = pd.crosstab(predictions, y_test, rownames=['Actual'], colnames=['Predictions'])



sn.set(font_scale=1.4)

sn.heatmap(confusion_matrix, annot=True,annot_kws={"size": 12}, fmt="d", cmap="YlGnBu")

plt.show()



# different ways of calculating the average

avgs = ['macro', 'micro', 'weighted']



accuracies = []

precisions = []

recalls = []

f1s = []

# saving scores for later plotting

for j in range(len(avgs)):

    acc, prec, rec, f1 = evaluate(y_test, predictions, avgs[j])

    precisions.append(prec)

    recalls.append(rec)

    f1s.append(f1)   

    accuracies.append(acc)



scores = {'Average': avgs, 'Accuracy': accuracies, 'Precision': precisions, 'Recall': recalls, 'F1': f1s}

scores_df = pd.DataFrame(data=scores)

scores_df.head(3)