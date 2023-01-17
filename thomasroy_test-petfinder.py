# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
from scipy.misc import imread
import matplotlib.pyplot as plt
import imageio
path = "../input/"
breeds = pd.read_csv(path + 'breed_labels.csv')
colors = pd.read_csv(path + 'color_labels.csv')
states = pd.read_csv(path + 'state_labels.csv')


#cat_cols = ['Type', 'Gender', 'MaturitySize', 'FurLength',  'Vaccinated', 
#            'Dewormed', 'Sterilized', 'Health', 'Breed1', 'Breed2', 'Color1',
#            'Color2', 'Color3', 'State', 'AdoptionSpeed']

#dtypes = {
#    col: 'category'
#    for col in cat_cols
#}

train = pd.read_csv(path + 'train/train.csv')#, dtype=dtypes)
test = pd.read_csv(path + 'test/test.csv')#, dtype=dtypes)

train.dtypes
sample_sub = pd.read_csv(path +'test/sample_submission.csv')
sample_sub.head()
train.shape
train.describe()
#Utilisation : translator['name_field'][id_cat]
translator = {}
translator['Type'] = ['No','Dog','Cat']
translator['Gender'] = ['Male', 'Female', 'Mixed']
translator['MaturitySize'] = ['Not Specified', 'Small', 'Medium', 'Large', 'Extra Large']
translator['FurLength'] = ['Not Specified', 'Short', 'Medium', 'Long']
translator['Vaccinated'] = ['Not Sure', True, False]
translator['Dewormed'] = ['Not Sure', True, False]
translator['Sterilized'] = ['Not Sure', True, False]
translator['Health'] = ['Not Specified','Healthy', 'Minor Injury', 'Serious Injury']
translator['AdoptionSpeed'] = ['on the same day', 'between 1 and 7 days', 'between 8 and 30 days', 'between 31 and 90 days', 'after 100 days']
indiv1 = train.loc[1704]
indiv1
#Its breed
for numbreed in range(1,3):
    idb = indiv1['Breed'+ str(numbreed)]
    if idb != 0:
        print(breeds[breeds.BreedID==int(idb)])
#Its color
for numcolor in range(1,4):
    idc = indiv1['Color'+ str(numcolor)]
    if idc != 0:
        print(colors[colors.ColorID==int(idc)])
#Its State
print(states[states.StateID==int(indiv1.State)])
translator['MaturitySize'][int(indiv1.MaturitySize)]
#Photos
for numphot in range(1, int(indiv1.PhotoAmt)+1):
    img = imageio.imread(path + 'train_images/' + str(indiv1.PetID) + '-' + str(numphot) + '.jpg')
    plt.imshow(img)
    plt.show()
plt.bar(range (0,5), train['AdoptionSpeed'].value_counts())
plt.title("Number of pets by AdoptionSpeed categorial")
#plt.subplot(1,2,1)
#plt.bar(translator['Type'][1:3], train['Type'].value_counts())
#plt.title("Distribution of pets by type")
#plt.ylabel('AdoptionSpeed')
#plt.subplot(1,2,2)
#adopsp1 = train[train.Type == 1]
#adopsp2 = train[train.Type == 2]
#mean1 = adopsp1['AdoptionSpeed'].mean()
#mean2 = adopsp2['AdoptionSpeed'].mean()
#plt.bar(translator['Type'][1:3], [mean1, mean2])
#plt.title ('Mean AdoptionSpeed by Type')
#plt.xlabel('Type')

train['is_train'] = True
test['is_train'] = False
all=pd.concat((train, test), sort = True)
all=all.reset_index(drop=True)
all = all.astype(test.dtypes)
#train.drop(['Description', 'RescuerID'], axis=1, inplace =True)
assert all['AdoptionSpeed'].isnull().sum() == len(test)
#df['garage_and_paved_driveway'] = df['GarageQual'].notnull() & df['PavedDrive'].notnull()
test1 = all[all.Age >= 8 ] 
test2 = all[all.Age < 8 ] 
print(len(test1)) ; print(len(test2))
all['Young_and_Healthy'] = (all.Age < 8) & (all.Health == 1)
all.dtypes
from sklearn import preprocessing

cat_cols = all.select_dtypes('category').columns
for col in cat_cols:
    if all[col].isnull().sum() > 0:
        all[col] = all[col].cat.add_categories('missing').fillna('missing')
    all[col] = preprocessing.LabelEncoder().fit_transform(all[col])
all.dtypes
import numpy as np

to_drop = ['PetID', 'is_train', 'Description', 'Name', 'RescuerID']

X_train = all.query('is_train == True').drop(columns=to_drop + ['AdoptionSpeed']) #garde les lignes pour lesquels is_train=true
y_train = all.query('is_train == True')['AdoptionSpeed'] #sépare y_train
X_test = all.query('is_train == False').drop(columns=to_drop + ['AdoptionSpeed']) #garde les lignes pour lesquels is_train=false

X_train.head()
#On fait quelques verifs
#assert all(X_train.columns == X_test.columns)
assert X_train.isnull().sum().sum() == 0
assert X_test.isnull().sum().sum() == 0
assert len(X_train) == len(y_train)
from sklearn import decomposition
from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing

#Pipeline = décrire le modèle, ne pas le coder
#Transformeur = prend des données, appliques des opérations dessus, les renvoient
#Estimateur
model = pipeline.Pipeline([
    ('one_hot', preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')), #dummies
    ('rescale', preprocessing.StandardScaler()), #centrer réduire
    ('pca', decomposition.TruncatedSVD(n_components=30)), #faire une ACP, reduction de dimensions
    ('ridge', linear_model.Ridge()) #Estimateur. Regressions linéaire où on pénalise les poids, on les force à être petit
    #Modèle plus timide, prédictions tournent plus autour de la moyenne, s'adapte mieux, va moins se tromper
])
#Méthode de validation croisée
#On s'entraine sur une partie, on teste sur une autre petite partie, on recommence plusieures fois
#On calcule la moyenne des erreurs
from sklearn import metrics
from sklearn import model_selection


def NegRMSE(y_true, y_pred):
    return -metrics.mean_squared_error(y_true, y_pred) ** 0.5

scoring = metrics.make_scorer(NegRMSE) 
cv = model_selection.KFold(n_splits=5, random_state=42)

scores = model_selection.cross_val_score(
    estimator=model,
    X=X_train,
    y=y_train,
    scoring=scoring,
    cv=cv
)

print('Model RMSE: {:.5f} ± {:.5f}'.format(-scores.mean(), scores.std()))
#Prochain but, tester des nouveaux paramètres du modèle
#Lance une validation croisée pour chaque combinaison de paramètres possibles
param_grid = {
    'ridge__alpha': [0.01, 0.1, 1],
    'pca__n_components': [10, 25, 50]
}

grid = model_selection.GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=scoring,
    cv=cv,
    return_train_score=True
)

grid = grid.fit(X_train, y_train)
#On affiche tous les résultats sous forme de tableau
results = pd.concat(
    (
        pd.DataFrame.from_dict(grid.cv_results_['params']),
        pd.DataFrame({
            'mean_train_score': -grid.cv_results_['mean_train_score'],
            'std_train_score': grid.cv_results_['std_train_score'],
            'mean_test_score': -grid.cv_results_['mean_test_score'],
            'std_test_score': grid.cv_results_['std_test_score']
        })
    ),
    axis='columns'
)

results.sort_values('mean_test_score')
#La meilleure combinaison de variable est celle qui minimine mean_test_score
grid.best_estimator_
sub = sample_sub.copy()

# We predict the log of the price
sub['AdoptionSpeed'] = grid.best_estimator_.predict(X_test)
sub['AdoptionSpeed'] = sub['AdoptionSpeed'].apply(round)

# We save the submission; the name of the file has the best validation 
sub.to_csv('submission.csv', index=False)
