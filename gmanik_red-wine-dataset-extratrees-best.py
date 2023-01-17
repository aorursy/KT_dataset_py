#import important packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#used to scale the data
from sklearn.preprocessing import StandardScaler
#kfold for the data
from sklearn.model_selection import RepeatedStratifiedKFold
#Labelencoder for target
from sklearn.preprocessing import LabelEncoder
#classifiers used
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
#used to check best classifier
from sklearn.model_selection import cross_val_score
#use pandas to import data to a dataframe
data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
#sort data by quality
data.head()
#make the quality either good or bad by using 0 for bad and 1 for good
data['quality'] = data.quality.map({3:0,4:0,5:0,6:0,7:1,8:1})
#split the data and target from all the data
X = data[data.columns[:11]]
y = data[data.columns[11]]
#make the data all float64
X = X.astype('float64')
y = LabelEncoder().fit_transform(y.astype('str'))
#now scale the data to make all the data scaled between 0 to 1
trans_norm = StandardScaler()
X = trans_norm.fit_transform(X)
X= pd.DataFrame(X)
# evaluate a give model using cross-validation
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