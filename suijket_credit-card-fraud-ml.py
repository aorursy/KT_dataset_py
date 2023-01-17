#### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Loading credit card data from kaggle
cc_fraud = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
# show first five columns
cc_fraud.head()
data = cc_fraud.copy()
data['Text'] = ["Non-Fruad" if i == 0 else "Fraud" for i in data['Class']]
data = data[['Text', 'Class']].groupby('Text').count().reset_index()


fig, ax = plt.subplots()

bar_x = [1, 2]
bar_height = data['Class']
bar_tick_label = ['Fraud', 'Non Fraud']
bar_label = data['Class']

bar_plot = plt.bar(bar_x, bar_height, tick_label = bar_tick_label)

def autolabel(rects):
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                bar_label[idx],
                ha='center', va='bottom', rotation=0)

autolabel(bar_plot)

plt.ylim(0,350000)
plt.title('Distribution of Credit Card Transactions')
plt.show()
ax = sns.distplot(cc_fraud['Amount'])
ax.set_title("Distirubtion Amount Of Money Spent")
cc_fraud['Amount^2'] = cc_fraud['Amount'] ** 2
cc_fraud['Amountlog'] = np.log(cc_fraud['Amount'])

cc_fraud = cc_fraud.loc[cc_fraud['Amount'] != 0]

ax = sns.distplot(cc_fraud['Amountlog'])
ax.set_title("Gaussian Distirubtion Amount Spent")
#removing outliers
df = cc_fraud.sample(frac = 1)

fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:2000]

# print then nubmer of rows each dataset has
print("Number of Non-Fraud cases:", len(non_fraud_df), "Number of Fraud cases:", len(fraud_df))

# create a list of all the column names
column_names = list(fraud_df)

# using the 1.5x IQR rule 
for i in range(0, len(column_names)-1):
    Q1 = non_fraud_df[column_names[i]].quantile(0.25)
    Q3 = non_fraud_df[column_names[i]].quantile(0.75)
    IQR = Q3 - Q1
    non_fraud_df = non_fraud_df[~((non_fraud_df[column_names[i]] < (Q1 - (1.75 * IQR))) |(non_fraud_df[column_names[i]] > (Q3 + (1.75 * IQR))))]

# print the number of non_fraud cases
print(len(fraud_df) / (len(non_fraud_df) + len(fraud_df)))

# create a new dataframe with the same amount of fraud as non-fraud cases
normal_distriubted_df = pd.concat([fraud_df, non_fraud_df])

# shuffle the newly created dataframe
new_df = normal_distriubted_df.sample(frac = 1, random_state = 32)

# show number of values left
print("Number of Non-Fraud cases:", len(non_fraud_df), "Number of Fraud cases:", len(fraud_df))
plt.figure(figsize=(30, 30))
matrix = np.triu(new_df.corr())
sns.heatmap(new_df.corr(), annot=True, mask=matrix)
# removing the to predict value from the dataframe
X = new_df.drop(['Class', 'Amount', 'Amount^2', 'Time'], axis = 1)
# only select the value to predict
y = new_df['Class']
# splitting the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 32)

# chagne type from dataframe to numpy array
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# create a dictionary with techniques to evaluate
classifiers = {
    "LogisticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

# test techniques on train and test set using cross validation
for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv = 5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of ", round(training_score.mean(), 2)* 100)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 7)):
   
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

X = new_df.drop('Class', axis = 1)
y = new_df['Class']

title = "Logistic Regression"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = LogisticRegression()
plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = "Decision Tree Classifier"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = DecisionTreeClassifier()
plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()