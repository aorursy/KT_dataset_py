import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')
df
df.head()
df.info()
# df.isnull().sum()
# sns.set_style("whitegrid")

# sns.set_context("poster")



# plt.figure(figsize = (12, 6))

# plt.hist(df['class'])

# plt.title('Histogram of target values in the training set')

# plt.xlabel('Class')

# plt.ylabel('Count')

# plt.show()

# plt.clf()
corr = df.corr()

# sns.heatmap(corr, vmin=0, vmax=1, linewidth=0.5)

corr.style.background_gradient(cmap='coolwarm')
numerical_features = ['chem_0', 'chem_1', 'chem_2', 'chem_3', 'chem_4', 'chem_5', 'chem_6', 'chem_7', 'attribute']

# numerical_features = ['chem_0', 'chem_1', 'chem_3', 'chem_4', 'chem_5', 'chem_6', 'attribute']



X = df[numerical_features]

y = df['class']
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.1, random_state=42)
from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

X_val[numerical_features] = scaler.transform(X_val[numerical_features])



X_train.head()
from sklearn.model_selection import RandomizedSearchCV



learning_rate = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

max_depth = [3, 4, 5, 6, 8, 10, 12, 15]

min_child_weight = [1, 3, 5, 7]

gamma = [0.0, 0.1, 0.2 , 0.3, 0.4]

colsample_bynode = [0.3, 0.4, 0.5 , 0.7]

colsample_bylevel = [0.3, 0.4, 0.5 , 0.7]

colsample_bytree = [0.3, 0.4, 0.5 , 0.7]

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
random_grid = {'learning_rate': learning_rate,

               'max_depth': max_depth,

#                'min_child_weight': min_child_weight,

               'gamma': gamma,

#                'colsample_bytree': colsample_bytree,

               'n_estimators': n_estimators}



print(random_grid)
from sklearn.model_selection import RandomizedSearchCV



n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



print(random_grid)
from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import LinearSVC

from sklearn.ensemble import AdaBoostClassifier
rf = RandomForestClassifier()

classifier = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 1)



classifier.fit(X_train,y_train)
# Utility function to report best scores

def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
report(classifier.cv_results_)
from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import LinearSVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPClassifier
# classifier = XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.2, gamma=0.0).fit(X_train,y_train)

# classifier = GaussianNB().fit(X_train, y_train)

# classifier = DecisionTreeClassifier().fit(X_train,y_train)

classifier = RandomForestClassifier(n_estimators=900, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=5).fit(X_train,y_train)

# classifier = ExtraTreesClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf=2, max_features='sqrt', max_depth=8).fit(X_train,y_train)

# classifier = LinearSVC(random_state=0, tol=1e-5).fit(X_train,y_train)

# classifier = AdaBoostClassifier(n_estimators=500, random_state=0, learning_rate=0.1).fit(X_train,y_train)

# classifier = LinearRegression().fit(X_train,y_train)

# classifier = MLPClassifier(alpha=0.1, max_iter=1000, learning_rate_init=0.1).fit(X_train, y_train)



# classifier.fit(X_train, y_train)
train_accuracy = classifier.score(X_train, y_train)

test_accuracy = classifier.score(X_val, y_val)

print(train_accuracy)

print(test_accuracy)
# predicted = classifier.predict(X_val)

# actual = y_val

# print(predicted)

# print(actual)

# correct = 0

# pos = 0

# for k in actual:

#     if (k == predicted[pos]):

#         correct += 1

#     pos += 1



# print((float(correct)*100)/len(actual))
tst = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

temp = tst.copy()

tst.head()
tst[numerical_features] = scaler.fit_transform(tst[numerical_features])

tst = tst[numerical_features]

tst.head()
tst_pred = classifier.predict(tst)

tst_pred = pd.Series(tst_pred)

frame = {'id':temp.id, 'class':tst_pred}

res = pd.DataFrame(frame)

res.head()
# export_csv = res.to_csv (r'/home/omkar/Desktop/ml_result.csv', index = None, header=True)