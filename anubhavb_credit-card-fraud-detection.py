# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model selection and evaluation
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

# Visualization
import seaborn as sns
sns.set(style="dark")
cc='/kaggle/input/creditcardfraud/creditcard.csv'

df = pd.read_csv(cc)
df.head()
df.describe()
print(" ###### Info ######")
print(df.info())

print(" ###### Null Checks ######")
print(df.isna().sum())
df.Class.value_counts()
sns.countplot(y='Class', data=df)
fig, ax = plt.subplots(1, 2, figsize=(18,4))
df.boxplot('Time', ax=ax[0])
df.boxplot('Amount', ax=ax[1])
rr_scalar = RobustScaler()

df['scaled_amt'] = rr_scalar.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rr_scalar.fit_transform(df['Time'].values.reshape(-1,1))

# Drop Original columns
df.drop(['Amount', 'Time'], axis=1, inplace=True)
df.head()
X = df.drop('Class', axis=1) # this creates a copy, since inplace attributes is not provided
y = df['Class']

test_df = df[df.Class==1][:92]
test_df
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=2)

test_y_vals, test_y_counts = np.unique(test_y, return_counts=True)
print(test_y_counts/len(test_y))
train_y_vals, train_y_counts = np.unique(train_y, return_counts=True)
print(train_y_counts/len(train_y))
model_logr = LogisticRegression()

model_logr.fit(train_X, train_y)
result = model_logr.score(test_X, test_y)
print("Accuracy: %.2f%%" % (result*100.0))
y_pred = model_logr.predict(test_X)
from sklearn.metrics import confusion_matrix

confusion_matrix(test_y, y_pred)
def plot_learning_curve(estimator, title, X, y, train_sizes=np.linspace(.1, 1.0, 5)):
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y,
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

plot_learning_curve(model_logr, "LogisticRegression Learning Curve", train_X, train_y)
plt.show()
# shuffle the full dataset
sdf = df.sample(frac=1, random_state=45)
fraud_df = sdf.loc[sdf['Class'] == 1]
non_fraud_df = sdf.loc[sdf['Class'] == 0][:492]

print("Fraud df = ", len(fraud_df))
print("Non Fraud df = ", len(non_fraud_df))

# shuffle the rows 
sampled_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=24)
sampled_df.reset_index(inplace=True)
sampled_df.drop(['index'], axis=1, inplace=True)
sampled_df.head()

sampled_df.head()
sampled_X = sampled_df.drop('Class', axis=1)
sampled_y = sampled_df['Class']

sampled_X = sampled_X.values
sampled_y = sampled_y.values.reshape(-1,1)
print("Sampled X", sampled_X.shape)
print("Sampled y", sampled_y.shape)


train_sX, test_sX, train_sy, test_sy = train_test_split(sampled_X, sampled_y, test_size=0.2, random_state=20)

test_sy_vals, test_sy_counts = np.unique(test_sy, return_counts=True)
print(test_y_counts/len(test_sy))
train_sy_vals, train_sy_counts = np.unique(train_sy, return_counts=True)
print(train_sy_counts/len(train_sy))

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNN Classifier": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}
cv_scores = {}
for k, model in classifiers.items():
    scores = cross_val_score(model, train_sX, train_sy, cv=5)
    cv_scores[k] = round(scores.mean() * 100, 2)
    print("Model: ", k, " Accuracy: ", cv_scores[k])
# Logistic Regression best estimator
lr_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_cv_lr = GridSearchCV(LogisticRegression(), lr_params)
grid_cv_lr.fit(train_sX, train_sy)
model_log_reg = grid_cv_lr.best_estimator_

# KNN  
knn_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knn = GridSearchCV(KNeighborsClassifier(), knn_params)
grid_knn.fit(train_sX, train_sy)
model_knn = grid_knn.best_estimator_

# SVC best estimator
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(train_sX, train_sy)
model_svc = grid_svc.best_estimator_
best_classifiers = {
    "LogisiticRegression": model_log_reg,
    "KNN Classifier": model_knn,
    "Support Vector Classifier": model_svc
}

best_cv_scores = {}
for k, model in best_classifiers.items():
    scores = cross_val_score(model, train_sX, train_sy, cv=5)
    best_cv_scores[k] = round(scores.mean() * 100, 2)
    print("Model: ", k, " Accuracy: ", best_cv_scores[k])
result = model_log_reg.score(test_sX, test_sy)
print("Accuracy: %.2f%%" % (result*100.0))

result = model_knn.score(test_sX, test_sy)
print("Accuracy: %.2f%%" % (result*100.0))

result = model_svc.score(test_sX, test_sy)
print("Accuracy: %.2f%%" % (result*100.0))
result = model_log_reg.score(test_X, test_y)
print("Accuracy: %.2f%%" % (result*100.0))