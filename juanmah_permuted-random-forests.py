import numpy as np

import pandas as pd



import warnings

warnings.simplefilter(action = 'ignore', category = FutureWarning)



import itertools

import math



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.utils.multiclass import unique_labels





import matplotlib.pyplot as plt



from tqdm import tnrange
# Read training and test files

X_train = pd.read_csv('../input/learn-together/train.csv', index_col = 'Id', engine = 'python')

X_test = pd.read_csv('../input/learn-together/test.csv', index_col = 'Id', engine = 'python')



# Define the dependent variable 

y_train = X_train['Cover_Type'].copy()



# Define a training set

X_train = X_train.drop(['Cover_Type'], axis = 'columns')
class_names = np.array([None, 'Spruce/Fir','Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz'])

class_names
permutation = list(itertools.permutations(list(range(1,8))))
def get_permuted_class_names(i):

    permuted_class_names = np.copy(class_names)

    for j in range(7):

        permuted_class_names[j + 1] = class_names[permutation[i][j]]

    return permuted_class_names
model = RandomForestClassifier(random_state = 1)
def get_accuracy(X, y):

    scores = cross_val_score(model, X, y, scoring = 'accuracy', n_jobs = -1)

    return np.mean(scores)
def get_prediction(X, y):

    y_pred = cross_val_predict(model, X, y, n_jobs = -1)

    return y_pred
accuracy = get_accuracy(X_train, y_train)

print('Accuracy: {:.4f}'.format(accuracy))
def plot_confusion_matrix(y, y_pred):



    # Compute confusion matrix

    cm = confusion_matrix(y, y_pred)

    # Only use the labels that appear in the data

    classes = class_names[unique_labels(y, y_pred)]



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title='Confusion matrix',

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()



    np.set_printoptions(precision=2)



    # plt.figure(figsize=(24, 24))

    plt.show()
results = pd.DataFrame(columns = ['Permutation', 'Accuracy'])



for i in tnrange(math.factorial(7)):

    accuracy = get_accuracy(X_train, y_train.replace(permutation[0], permutation[i]))

    results = results.append({

        'Permutation': permutation[i],

        'Accuracy': accuracy  

    }, ignore_index = True)



results.to_csv('results.csv', index = True)

print(results.describe())
print('Difference in accuracy from worst to best permutation: {:.2%}'.format(results['Accuracy'].max() - results['Accuracy'].min()))

print('Difference in accuracy from initial to best permutation: {:.2%}'.format(results['Accuracy'].max() - results.iloc[0]['Accuracy']))
plt.hist(results["Accuracy"], 32, density=True, alpha=0.5)



plt.legend(loc='upper right')

plt.xlabel('Smarts')

plt.ylabel('Probability')

plt.title('Histogram of IQ')

plt.grid(True)

plt.show()
nlargest_accuracy = results.nlargest(10, ['Accuracy'])

largest_accuracy = nlargest_accuracy.iloc[0].name

nlargest_accuracy
nsmallest_accuracy = results.nsmallest(10, ['Accuracy'])

smallest_accuracy = nsmallest_accuracy.iloc[0].name

nsmallest_accuracy
y_pred = get_prediction(X_train, y_train.replace(permutation[0], permutation[largest_accuracy]))

plot_confusion_matrix(y_train, pd.Series(y_pred).replace(permutation[largest_accuracy], permutation[0]))
y_pred = get_prediction(X_train, y_train.replace(permutation[0], permutation[smallest_accuracy]))

plot_confusion_matrix(y_train, pd.Series(y_pred).replace(permutation[smallest_accuracy], permutation[0]))
# Get model with the permutation with the largest_accuracy

accuracy = get_accuracy(X_train, y_train.replace(permutation[0], permutation[largest_accuracy]))

print(accuracy)



# Predict with test data

model.fit(X_train, y_train.replace(permutation[0], permutation[largest_accuracy]))

y_test_pred = pd.Series(model.predict(X_test))



# Transform back the forest type

y_test_pred = y_test_pred.replace(permutation[largest_accuracy], permutation[0])
output = pd.DataFrame({'ID': X_test.index,

                       'Cover_Type': y_test_pred})



output.to_csv('submission.csv', index=False)