# import important pacakages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# sci-kit learn imports
from sklearn.preprocessing import LabelEncoder

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# used to check best classifier
from sklearn.model_selection import cross_val_score

# kfold for the data
from sklearn.model_selection import RepeatedStratifiedKFold
# evaluate a given model using cross-validation
def evaluate_model(model):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores
# models to check for best results
def get_models():
    models = dict()
    models['ExT'] = ExtraTreesClassifier(max_features = 'log2', min_samples_leaf =1, n_estimators = 1000)
    models['Rnd'] = RandomForestClassifier(max_features = 'sqrt', min_samples_leaf =1, n_estimators = 1000)
    models['Gb'] = GradientBoostingClassifier(max_features = 'log2', min_samples_leaf =2, n_estimators = 1000)
    return models
# read the data using pandas
data = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
# let look at the data to see what we have
data.head()
# check data for null values
data.info()
# we need to label encode the data
le = []
labels = data.columns
for i in range(len(labels)):
    le.append(LabelEncoder())
    le[i].fit(data[labels[i]])
    data[labels[i]] = le[i].transform(data[labels[i]])
#split X and y from all the data
X = data[data.columns[1:23]]
y = data[data.columns[0]]
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.show()
