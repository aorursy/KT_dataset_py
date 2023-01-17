# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import warnings
warnings.filterwarnings('ignore')

import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.sparse import csr_matrix, hstack

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
from tqdm import tqdm
#tqdm.pandas()

# Feature engineering
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Fitting
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import gc
import os
DATA_PATH = "../input"
print(os.listdir(DATA_PATH))

# Any results you write to the current directory are saved as output.
debug = False
train = pd.read_json(os.path.join(DATA_PATH, 'train.json')).set_index('id')
test = pd.read_json(os.path.join(DATA_PATH, 'test.json')).set_index('id')
if debug is True:
    train = train.sample(100)
    test = test.sample(100)

print("Training Data Shape: ", train.shape)
print("Testing Data Shape: ", test.shape)

print("Number of cuisines: ", train.cuisine.nunique())
# Remove single-ingredient entries 
train = train[train['ingredients'].str.len()>1]
traindex = train.index
testdex = test.index

train_size = train.shape[0]

y_train = train.cuisine.copy()

df = pd.concat([train[['ingredients']], test], axis = 0)
print("All Data Shape: ", df.shape)
df_index = df.index

features_df = pd.DataFrame(index=df.index)
sns.countplot(y=train.cuisine, order=train.cuisine.value_counts().reset_index()["index"])
plt.title("Cuisine Distribution in training data")
plt.show()
train['ings'] = train['ingredients'].apply(lambda x: ",".join(x))
withoz = train[train['ings'].str.contains("oz\.")]
if len(withoz) > 0:
    sns.countplot(y=withoz.cuisine, order=withoz.cuisine.value_counts().reset_index()["index"])
    plt.title("Cuisine Distribution for (oz) in ingredients")
    plt.show()
fr_accents = ['é', 'è', 'ê', 'ë', 'à', 'â', 'î', 'ô', 'ù', 'û', 'ç']
with_fr_accent = train[train['ings'].str.contains("|".join(fr_accents))]
if len(with_fr_accent) > 0:
    sns.countplot(y=with_fr_accent.cuisine, order=with_fr_accent.cuisine.value_counts().reset_index()["index"])
    plt.title("Cuisine Distribution for French accents in ingredients")
    plt.show()
features_df['fr_accents'] = df.ingredients.apply(lambda x: ",".join(x)).str.contains("|".join(fr_accents)).astype(int)
es_accents = ['á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ']
with_es_accent = train[train['ings'].str.contains("|".join(es_accents))]
if len(with_es_accent) > 0:
    sns.countplot(y=with_es_accent.cuisine, order=with_es_accent.cuisine.value_counts().reset_index()["index"])
    plt.title("Cuisine Distribution for Spanish accents in ingredients")
    plt.show()
features_df['es_accents'] = df.ingredients.apply(lambda x: ",".join(x)).str.contains("|".join(es_accents)).astype(int)
train.drop(['ings'], axis=1, inplace=True)
ingredient_dict = {}
for _, row in train.iterrows():
    for ing in row['ingredients']:
        ingredient_dict.setdefault(ing, []).append(row['cuisine'])
endemic_ingredient_dict = {}
for ing, cui in ingredient_dict.items():
    if len(cui) <= 1:
        endemic_ingredient_dict[ing] = cui[0]
len(endemic_ingredient_dict)
cuisines = train.cuisine.unique()
print(cuisines)
for cui in cuisines:
    features_df[cui] = 0
for ing, cui in endemic_ingredient_dict.items():
    features_df.loc[df.ingredients.apply(lambda x: ",".join(x)).str.contains(ing),cui] = 1

features_df.head()
del test; del train; gc.collect();
vect = CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')], lowercase=False)
dummies = vect.fit_transform(df['ingredients'].apply(','.join)) 

full_matrix = csr_matrix(hstack([dummies, features_df]))
X_train = full_matrix[:train_size,:]
X_test = full_matrix[train_size:,:]
print('All data matrix shape:', full_matrix.shape)
print('Train data matrix shape:', X_train.shape)
print('Test data matrix shape:', X_test.shape)
%%time
classifier = LogisticRegression(multi_class='multinomial', 
                                solver='saga', 
                                verbose=1, 
                                n_jobs=-1)
#score = cross_validate(classifier, X_train, y_train, cv=5)
#print(score["test_score"].mean())
#cvscore = score["test_score"].mean()
%%time
from sklearn.model_selection import GridSearchCV
cv_params = {'C': np.logspace(-1, 2, 20), 'multi_class': ['ovr', 'multinomial']}
#gridsearch = GridSearchCV(classifier, cv_params, cv=5, verbose=0, n_jobs=-1)
#gridsearch.fit(X_train, y_train)
#cvscore = gridsearch.best_score_
bestC = 1.8329807108324356
#print(gridsearch.best_params_)
#print(gridsearch.best_score_)
%%time
classifier = LogisticRegression(C=bestC,
                                multi_class='ovr', 
                                solver='saga', 
                                verbose=1, 
                                n_jobs=-1)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_train)
#y_true = label_encoder.inverse_transform(y_train)

print(f'accuracy score on train data: {accuracy_score(y_train, y_pred)}')
def write_submission_file(prediction, index, filename,
                          path_to_sample=os.path.join(DATA_PATH,'sample_submission.csv')):
    #submission = pd.read_csv(path_to_sample, index_col='id')
    submission = pd.Series(prediction, index=index).rename('cuisine')
    #submission['cuisine'] = prediction
    submission.to_csv(filename, header=True, index=True)
# make submission
y_pred = classifier.predict(X_test)
write_submission_file(y_pred, testdex, "logistic_cv_sub.csv")