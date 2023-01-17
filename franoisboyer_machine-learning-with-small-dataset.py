%matplotlib inline



import matplotlib.pyplot as plt

import numpy as np



import pandas as pd

import pandas_profiling



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import make_scorer

def precision_score_micro(y_true, y_pred):

    return(precision_score(y_true, y_pred, average='micro'))



from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



from sklearn.model_selection import GridSearchCV

import seaborn as sns
!conda install -y phik
import phik

from phik import resources, report
phik.__version__
NB_SAMPLES = 150  # Avec 2000 :  perf 80%, et on détecte les 5 coefs importants.  Avec 150 : perf 57% et on ne détecte que les 3 premiers coefs   

NB_VARIABLES = 35



np.random.seed(35)

X = np.random.randint(0,10,(NB_SAMPLES, NB_VARIABLES))

X
X.shape
coefs = np.zeros([NB_VARIABLES])

coefs[0] = 20

coefs[1] = 15

coefs[2] = 10

coefs[3] = 5



coefs
coefs.shape
Y = (coefs.dot(X.T))

Y
Y.shape
Y = pd.cut(pd.DataFrame(Y)[0], bins=5, labels=['F1','F2','F3','F4','F5']).to_numpy()
Y
pd.DataFrame(Y)[0].value_counts().plot.bar(title='Distribution of labels')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle = True)
model = RandomForestClassifier(max_depth=10, max_features=20, max_leaf_nodes=50,

                       n_estimators=5000, random_state=42) 



model.fit(X_train, Y_train)
Y_predict_train = model.predict(X_train)

Y_predict_test = model.predict(X_test)

precision_score(Y_predict_train, Y_train, average='micro')
pscore = precision_score(Y_predict_test, Y_test, average='micro')

rscore = recall_score(Y_predict_test, Y_test, average='micro')
print(f'So we trained a model that can predict the label with a precision of {pscore*100}% and a recall of {rscore*100}% on a test set of data it has never seen')

print('Will that be enough for us to discover which 4 variables were importance to predict the labels ?')
pd.DataFrame(model.feature_importances_).plot.bar(title='Random forest: feature importances of our variables')
df = pd.concat([pd.DataFrame(X),pd.DataFrame(Y).rename(columns={0: 'Y'})], axis=1)
df_phik = df.phik_matrix()
df_phik
df_phik['Y'].sort_values(ascending=False)[0:10]
NB_SAMPLES = 2000  # Avec 2000 :  perf 80%, et on détecte les 5 coefs importants.  Avec 150 : perf 57% et on ne détecte que les 3 premiers coefs   

NB_VARIABLES = 35



X = np.random.randint(0,10,(NB_SAMPLES, NB_VARIABLES))



coefs = np.zeros([NB_VARIABLES])

#coefs[5:] = 0.2

coefs[0] = 20

coefs[1] = 15

coefs[2] = 10

coefs[3] = 5

#coefs[4] = 2



Y = (coefs.dot(X.T))



Y = pd.cut(pd.DataFrame(Y)[0], bins=5, labels=['F1','F2','F3','F4','F5']).to_numpy()

pd.DataFrame(Y)[0].value_counts().plot.bar(title='Distribution of labels')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle = True)



model = RandomForestClassifier(max_depth=10, max_features=20, max_leaf_nodes=50,

                       n_estimators=5000, random_state=42) 



model.fit(X_train, Y_train)



Y_predict_train = model.predict(X_train)

Y_predict_test = model.predict(X_test)

precision_score(Y_predict_train, Y_train, average='micro')



pscore = precision_score(Y_predict_test, Y_test, average='micro')

rscore = recall_score(Y_predict_test, Y_test, average='micro')
print(f'So we trained a model that can predict the label with a precision of {pscore*100}% and a recall of {rscore*100}% on a test set of data it has never seen')

print('Will that be enough for us to discover which 4 variables were importance to predict the labels ?')
pd.DataFrame(model.feature_importances_).plot.bar(title='Random forest: feature importances of our variables')
df = pd.concat([pd.DataFrame(X),pd.DataFrame(Y).rename(columns={0: 'Y'})], axis=1)

df_phik = df.phik_matrix()

df_phik['Y'].sort_values(ascending=False)[0:10]