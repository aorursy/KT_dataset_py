# importing data libraries



import numpy as np

import pandas as pd



# importing visualization libraries



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# importing models from sklearn and tensorflow.keras



# sklearn classification algorithms

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.metrics import confusion_matrix



# tensorflow.keras Sequential model

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.backend import clear_session

data = pd.read_csv('../input/hotel_bookings_cleaned.csv')
data.shape
data.info()
data.describe()



# we have different range of values.

# most of our columns consist dummy variables 
X = data.drop(['is_canceled'], axis=1).values

y = data['is_canceled'].values
X.shape, y.shape
train_data_length = 1000 # X.shape[0] // 3
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X[:train_data_length], y[:train_data_length], test_size=0.1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import StandardScaler



std = StandardScaler()



X_train = std.fit_transform(X_train)

X_test = std.transform(X_test)
pd.DataFrame(X_train).describe() # result of standardization
# PCA Dimension Reduction



from sklearn.decomposition import PCA



pca = PCA(n_components = 0.95)



X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

X_train.shape, X_test.shape
import warnings

from tqdm.notebook import tqdm

warnings.filterwarnings('ignore')



kfold = StratifiedKFold()



# Modeling with different algorithms



state = 42

classifiers = list()



algorithms = ["DecisionTree", "RandomForest","GradientBoosting", "KNeighboors","LogisticRegression",

             "LinearDiscriminantAnalysis", "Keras"]



tree_algorithms = {"DecisionTree": 1, "RandomForest": 2, "GradientBoosting": 3}



classifiers.append(SVC(random_state=state))

classifiers.append(DecisionTreeClassifier(random_state=state))

classifiers.append(RandomForestClassifier(random_state=state))

classifiers.append(GradientBoostingClassifier(random_state=state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results = dict()



for i in tqdm(range(len(classifiers))):

    cv_results[algorithms[i]] = cross_val_score(classifiers[i], X_train, y_train, scoring='accuracy', cv = kfold, verbose=1, n_jobs=-1)

model = Sequential()



model.add(Dense(128, input_shape=(X_train.shape[1],), activation="relu"))

model.add(Dense(256, activation="relu"))

model.add(Dense(128, activation="relu"))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(X_train, y_train, epochs=128, batch_size=128, verbose=0, validation_split=0.2)



score = model.evaluate(X_test, y_test, verbose=0)

cv_results['Keras'] = np.array([score[1]])
cv_means = {}

cv_std = {}



for algorithm in cv_results.keys():

    cv_means[algorithm] = [cv_results[algorithm].mean(), cv_results[algorithm].std()]



cv_means = {k: v for k, v in sorted(cv_means.items(), key=lambda item: item[1], reverse=True)}

cv = np.array(list(cv_means.values()))

    

cv_df = pd.DataFrame({"Algorithm": list(cv_means.keys()), "CVMean": cv[:, 0], "CVStd": cv[:, 1]})
cv_df
plt.figure(figsize=(16, 10))



sns.set_color_codes("pastel")

g = sns.barplot(x='Algorithm', y='CVMean', data=cv_df, color='b')

sns.set_color_codes("muted")

g = sns.barplot(x='Algorithm', y='CVStd', data=cv_df)



g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation Scores")
def plot_feature_importance(classifier, name):

    

    plt.figure(figsize=(16, 8))

    

    indices = np.argsort(classifier.feature_importances_)[::-1][:20]

    

    importances = classifier.feature_importances_[indices]

    columns = data.columns[indices]

    

    g = sns.barplot(x=importances, y=columns)

    

    plt.title(name)

    plt.show(g)
for tree in tree_algorithms.keys():

    classifiers[tree_algorithms[tree]].fit(X_train, y_train)

    plot_feature_importance( classifiers[tree_algorithms[tree]], tree)
test_results = dict()



for i in tqdm(range(len(classifiers))):

    classifiers[i].fit(X_train, y_train)

    test_results[algorithms[i]] = classifiers[i].score(X_test, y_test)



test_results['Keras'] = model.evaluate(X_test, y_test, verbose=0)[1]
test_scores = {}



for algorithm in test_results.keys():

    test_scores[algorithm] = test_results[algorithm]



test_results = {k: v for k, v in sorted(test_scores.items(), key=lambda item: item[1], reverse=True)}

    

test_df = pd.DataFrame({"Algorithm": list(cv_means.keys()), "TestScore": list(test_results.values())})
test_df
plt.figure(figsize=(16, 10))



g = sns.barplot(x='Algorithm', y='TestScore', data=test_df)



g.set_xlabel("Test Accuracy")

g = g.set_title("Test Scores")