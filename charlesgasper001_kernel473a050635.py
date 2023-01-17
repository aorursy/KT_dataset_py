# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import tempfile
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#customizing our Matplot lib visualization figure size and colors

mlp.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#Reading in the data
dataframe = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
dataframe.head()
#looking at the statistics of the dataset
dataframe[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V26', 'V27', 'V28', 'Amount', 'Class']].describe()
data = dataframe['V1'], dataframe['V2'], dataframe['Class']
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
plt.scatter(dataframe['V1'], dataframe['V2'])
#looking at dataset imbalance
neg, pos = np.bincount(dataframe['Class'])
total = neg + pos
print('Class:\n  Total: {}\n  Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos/total))
# Checking for missing value
for (column, columnData) in dataframe.iteritems():
    isNull = False
    if dataframe[column].empty:
        isNull = True
        print('Is Null')
print(isNull)
cleaned_df = dataframe.copy()

# We assume the time column is not needed so we remove it
cleaned_df.pop('Time')

# Converting the Amount column to log psace as it covers a huge range
eps=0.001
cleaned_df['Log Amount'] = np.log(cleaned_df.pop('Amount')+eps)
cleaned_df.head()
#splitting the dataset

from sklearn.model_selection import train_test_split

#seperate the dependent from the independent columnes
print('total dataset: ', len(dataframe))
y = np.array(cleaned_df.pop('Class'))
x = np.array(cleaned_df)

random_state = np.random.RandomState(0)

#Shuffling the dataset of features
n_rows, n_features = x.shape
x = np.c_[x, random_state.randn(n_rows, n_features)]

#split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=random_state)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=random_state)

bool_ytrain = y_train != 0

train_features = np.array(x_train)
val_features = np.array(x_val)
test_features = np.array(x_test)
# Normalize with Feature scaling. Given the disparity of the columns time and ammount from the rest

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)
val_features = sc.transform(val_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)

print('Training Class Shape:', y_train.shape)
print('Validation Class Shape:', y_val.shape)
print('Test Class Shape:', y_test.shape)
print('\n')
print('Training features Shape:', train_features.shape)
print('Validation features Shape:', val_features.shape)
print('Test features Shape:', test_features.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pickle

#function for using the k-fold cross validation to find the optimal dept by
#...by fiting trees of various depths on the training data.
# We also use prunning optimize the classification tree

def run_decision_tree(x, y, tree_depths, cv=7, scoring='average_precision', train=True):
    cross_val_scores_list = []
    cross_val_scores_mean = []
    cross_val_scores_std = []
    accuracy_list = []
    
    temp = 0
    
    for depth in tree_depths:
        random_state = np.random.RandomState(0)
        model = DecisionTreeClassifier(criterion="entropy", random_state=random_state, max_depth=depth)
        cross_val_scores = cross_val_score(model,x, y, cv=cv, scoring=scoring)
        cross_val_scores_list.append(cross_val_scores)
        score_mean = cross_val_scores.mean()
        cross_val_scores_mean.append(score_mean)
        cross_val_scores_std.append(cross_val_scores.std())
        accuracy_list.append(model.fit(x, y).score(x, y))
        
        print("Tree depth {:d}:\n\tCross Validation Scores mean: {:f}\n\tCross Validation STD: {:f}\n\tAccuracy: {:f}\n".format(depth, score_mean, cross_val_scores.std(), accuracy_list[-1]))
        
        if(train):
            if(score_mean > temp):
                temp = score_mean
                print("Saving model {}... \n".format(depth))
                model_file = 'best_model.sav'
                pickle.dump(model, open(model_file, 'wb'))
                best_tree_depth = depth
                best_tree_cv_score = cross_val_scores_mean[-1]
                best_tree_cv_std = cross_val_scores_std[-1]
                print("Model saved\n")

            else:
                print("Model not optimal. Skiping save...\n")
        
    cross_val_scores_mean = np.array(cross_val_scores_mean)
    cross_val_scores_std = np.array(cross_val_scores_std)
    accuracy_list = np.array(accuracy_list)
    
    if(train == True):
        return cross_val_scores_mean, cross_val_scores_std, accuracy_list, model_file
    else:
        return cross_val_scores_mean, cross_val_scores_std, accuracy_list
#Fitting Trees for depth 1 to 14
depths = range(1, 15)
scores_mean, scores_std, accuracy_list, model_file = run_decision_tree(train_features, y_train, tree_depths=depths)
val_scores_mean, val_scores_std, val_accuracy_list = run_decision_tree(val_features, y_val, tree_depths=depths, train=False)

print("Score Mean List: ",scores_mean)
print("Score STD: ", scores_std)
print("Accuracy List: ", accuracy_list)
scores = np.array([scores_mean, scores_std, accuracy_list])
v_scores = np.array([val_scores_mean, val_scores_std, val_accuracy_list])
print(scores[2])
print(v_scores[2])
def plot_metrics(scores, v_scores, depth=depths):
  metrics =  ['scores_mean', 'scores_std', 'accuracy_list']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(depth, scores[n], color=colors[0], label='Train')
    plt.plot(depth, v_scores[n], color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Depth')
    plt.ylabel(name)
    if metric == 'scores_mean':
      plt.ylim([0, 1])
    elif metric == 'scores_std':
      plt.ylim([0, 0.2])
    else:
      plt.ylim([0, 2])

    plt.legend()
plot_metrics(scores, v_scores)
loaded_model = pickle.load(open(model_file, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)
#Making predictions and checking the accuracy
from sklearn import metrics

y_pred = loaded_model.predict(x_test)

accuracy = metrics.accuracy_score(y_pred, y_test)

print('The accuracy of the model is {:.2f}% '.format(accuracy*100))

from sklearn.metrics import average_precision_score
avg_precision = average_precision_score(y_test, y_pred)

print("Average precision-recall: {:.2f}%".format(avg_precision * 100))
from sklearn.metrics import confusion_matrix
def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
plot_cm(y_test, y_pred)