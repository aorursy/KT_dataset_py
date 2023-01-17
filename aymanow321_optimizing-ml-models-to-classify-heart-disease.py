import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

plt.rcParams['figure.figsize'] = [15, 6]

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import MinMaxScaler
dt = pd.read_csv("../input/heart-disease/heart.csv")

data = dt.copy()
data.head()
data.dtypes
data.isnull().sum(axis = 0)
data.target.value_counts()
for col in data.columns:

    print(col," : ", data[col].unique().size)
df = data

for col in data.columns:

    if df[col].unique().size < 6:                # 5 is the maximum for 'value_couts' categorical features 

        df = df.drop(columns = [col])

sns.boxplot(data = df, orient = 'h')
positive =data[data['target'] == 1]

negative =data[data['target'] == 0]
def set_kde(col):

    if data[col].unique().size < 7:

        return False

    else : return True

    

    

for col in data.columns :

    plt.figure()

    sns.distplot(positive[col], label = 'positive', kde = set_kde(col))

    sns.distplot(negative[col], label = 'negative', kde = set_kde(col))

    plt.legend()

sns.clustermap(data.corr())
a = pd.get_dummies(data['cp'], prefix = "cp")

b = pd.get_dummies(data['thal'], prefix = "thal")

c = pd.get_dummies(data['slope'], prefix = "slope")



frames = [data, a, b, c]

data = pd.concat(frames, axis = 1)



data = data.drop(columns = ['cp', 'thal', 'slope'])
data.head()
y = data['target']

X = data.drop(columns = ['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
def prepare_data(dt):

    y = dt['target']

    X = dt.drop(columns = ['target'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)



    scaler = MinMaxScaler()

    scaler.fit(X_train, X_test)

    X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled



X_train_scaled, X_test_scaled = prepare_data(dt)
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()



grid_params = {

    'n_neighbors' : range(2,20),

    'weights' : ['uniform', 'distance'],

    'metric' : ['euclidian', 'manhattan']

}



GKNN = GridSearchCV(

    KNN,

    grid_params,

    cv = 4

)
from sklearn.svm import SVC

SVM = SVC(random_state = 1)
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
def evaluation(X_train_scaled, X_test_scaled, msg = ''):

    models = {'RandomForest' : RF,

          'KNearestNeighbors': GKNN,

          'Support Vector' : SVM,

          'Naive Bayes' : NB}



    dfmd = pd.DataFrame(list(models.items()),columns = ['models','params'])

    labels = models.keys()



    scores = []

    for mdl in models.values(): 

        mdl.fit(X_train_scaled, y_train)

        scores.append(100 * mdl.score(X_test_scaled, y_test))

    dfmd['scores'] = scores

    #print(np.mean(scores))

    #if msg != '': print(msg)

    splot = plt.figure()

    splot = sns.barplot(data = dfmd, y = 'scores', x = 'models')

    for p in splot.patches:

        splot.annotate(format(p.get_height(), '.1f'), 

                       (p.get_x() + p.get_width() / 2., p.get_height()), 

                       ha = 'center', va = 'center', 

                       xytext = (0, 9), 

                       textcoords = 'offset points')

    plt.legend(msg)

    plt.show()

    return scores

    



evaluation(X_train_scaled, X_test_scaled)
balanced_sus = ['trestbps', 'fbs']

balanced_sus1 = ['fbs']

balanced_sus2 = ['trestbps']

imbalanced_sus = ['sex']

all_sus = balanced_sus + imbalanced_sus

test_features = [balanced_sus, balanced_sus1, balanced_sus2, imbalanced_sus, all_sus]

test_scores = []

for fts in test_features:

    test_data = data.copy()

    test_data.drop(columns = fts, inplace = True)

    for col in fts :

        X_train_scaled_h, X_test_scaled_h = prepare_data(test_data)

    #print('\nUsing', fts)

    test_scores.append(np.mean(evaluation(X_train_scaled_h, X_test_scaled_h, msg = fts)))

test_scores_a = np.array(test_scores)

print("scores were : ", test_scores_a, "\n \n")

print("the higher score was for ", np.max(test_scores), " by eliminating ", 

      test_features[np.argmax(test_scores_a)][0])



# Then we set the optimal data 

data = dt.drop(columns = [test_features[np.argmax(test_scores_a)][0]])
from sklearn.metrics import recall_score
def recall_evaluation():

    models = {'RandomForest' : RF,

          'KNearestNeighbors': GKNN,

          'Support Vector' : SVM,

          'Naive Bayes' : NB}

    recalls = []

    dfmd = pd.DataFrame(list(models.items()),columns = ['models','params'])

    labels = models.keys()

    scores = []

    for mdl in models.values(): 

        mdl.fit(X_train_scaled, y_train)

        y_pred = mdl.predict(X_test_scaled)

        recalls.append(100 * recall_score(y_test, y_pred, average='binary'))

    dfmd['recalls'] = recalls

    print(recalls)

    splot = sns.barplot(data = dfmd, y = 'recalls', x = 'models')

    for p in splot.patches:

        splot.annotate(format(p.get_height(), '.1f'), 

                       (p.get_x() + p.get_width() / 2., p.get_height()), 

                       ha = 'center', va = 'center', 

                       xytext = (0, 9), 

                       textcoords = 'offset points')

    plt.show()

    

recall_evaluation()
print("On default scoring")

print('  Train score  : ', SVM.score(X_train_scaled, y_train))

print('  Test score   : ', SVM.score(X_test_scaled, y_test))

print('\n\n')





print("On recall scoring")

y_pred = SVM.predict(X_test_scaled)

print('  Train score  :', 100 * recall_score(y_test, y_pred, average='binary'))

y_pred = SVM.predict(X_train_scaled)

print('  Test score   :', 100 * recall_score(y_train, y_pred, average='binary'))
