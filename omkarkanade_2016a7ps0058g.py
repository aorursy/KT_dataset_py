import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')
df.head(10)
df.info()
df.isnull().sum()
df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
sns.set_style("whitegrid")

sns.set_context("poster")



plt.figure(figsize = (12, 6))

plt.hist(df['rating'])

plt.title('Histogram of target values in the training set')

plt.xlabel('Rating')

plt.ylabel('Count')

plt.show()

plt.clf()
sns.regplot(x='feature3', y='rating', data=df)
df.fillna(value=df.mode().loc[0], inplace=True)

df.head()
df = pd.get_dummies(df, prefix=['type'])

df.head()
corr = df.corr()

# sns.heatmap(corr, vmin=0, vmax=1, linewidth=0.5)

corr.style.background_gradient(cmap='coolwarm')
# shuffled_df = df.sample(frac=1)

# shuffled_df = df
# numerical_features = ['feature6', 'feature7', 'feature3', 'feature5']

# categorical_features = ['type_old']

numerical_features = ['feature1', 'feature2', 'feature4', 'feature8', 'feature9', 'feature10', 'feature11', 'feature6', 'feature7', 'feature3', 'feature5']

categorical_features = ['type_old']

# numerical_features = ['type_new', 'feature2', 'feature4', 'feature8', 'feature11', 'feature6']

# categorical_features = []

X = df[numerical_features+categorical_features]

y = df['rating']
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.01, random_state=42)
from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

X_val[numerical_features] = scaler.transform(X_val[numerical_features])



X_train.head()
from sklearn.ensemble import ExtraTreesClassifier



# Build a forest and compute the feature importances

forest = ExtraTreesClassifier(n_estimators=250,

                              random_state=0)



forest.fit(X_train, y_train)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices)

plt.xlim([-1, X.shape[1]])

plt.show()
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
# from sklearn.model_selection import GridSearchCV



# rf = RandomForestRegressor()

# param_grid = random_grid

# clf = GridSearchCV(rf, param_grid=param_grid, cv=5, iid=False)

# clf.fit(X_train, y_train)
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor



rf = ExtraTreesRegressor()

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
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesClassifier



# classifier = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

#            max_features='sqrt', max_leaf_nodes=None,

#            min_impurity_decrease=0.0, min_impurity_split=None,

#            min_samples_leaf=1, min_samples_split=2,

#            min_weight_fraction_leaf=0.0, n_estimators=2000, n_jobs=1,

#            oob_score=False, random_state=42, verbose=0, warm_start=False)

# classifier = RandomForestRegressor(max_depth=None, n_estimators=2000, random_state=42)

classifier = ExtraTreesRegressor(n_estimators=500, random_state=42, max_features='auto', max_depth=None)

# classifier = ExtraTreesClassifier()

classifier.fit(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



from sklearn.linear_model import LinearRegression

from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor





# classifier = RandomForestClassifier().fit(X_train,y_train)

# classifier = DecisionTreeClassifier(max_depth = 2).fit(X_train,y_train)

# classifier = SVC(kernel='linear', C=1).fit(X_train, y_train)

# classifier = KNeighborsClassifier(n_neighbors = 10).fit(X_train, y_train)

# classifier = GaussianNB().fit(X_train, y_train)

# classifier = MLPClassifier(alpha=0.1, max_iter=10000, learning_rate_init=0.01).fit(X_train, y_train)

# classifier = GaussianProcessClassifier().fit(X_train, y_train)

# classifier = AdaBoostClassifier(n_estimators=20, learning_rate=0.0001).fit(X_train, y_train)

# classifier = QuadraticDiscriminantAnalysis().fit(X_train, y_train)



# classifier = LinearRegression().fit(X_train, y_train)

# classifier = linear_model.Ridge().fit(X_train, y_train)

# classifier = linear_model.Lasso(alpha=0.0001).fit(X_train, y_train)

# classifier = linear_model.LogisticRegression().fit(X_train, y_train)

# classifier = linear_model.ElasticNet(alpha=1.0).fit(X_train, y_train)

# classifier = RandomForestRegressor(max_depth=21, random_state=42, n_estimators=100).fit(X_train, y_train)
predicted = classifier.predict(X_val)

actual = y_val

# print(predicted)

# print(actual)

rmse = 0

pos = 0

for k in actual:

    rmse += ((k - round(predicted[pos]))**2)

    pos += 1

rmse /= len(actual)

rmse = np.sqrt(rmse)

print(rmse)
tst = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

temp = tst.copy()

tst.head()
tst.fillna(value=tst.mean(), inplace=True)

tst.head()
tst.isnull().sum()
tst = pd.get_dummies(tst, prefix=['type'])

tst.head()
tst[numerical_features] = scaler.fit_transform(tst[numerical_features])

tst = tst[numerical_features+categorical_features]

tst.head()
tst_pred = classifier.predict(tst)

tst_pred = pd.Series(tst_pred)

frame = {'id':temp.id, 'rating':round(tst_pred)}

res = pd.DataFrame(frame)

res
# export_csv = res.to_csv (r'/home/omkar/Desktop/result.csv', index = None, header=True)