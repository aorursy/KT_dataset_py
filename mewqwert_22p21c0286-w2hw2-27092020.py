import numpy as np
import pandas as pd
import os
file = {}
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames: file[filename[:-4]] = os.path.join(dirname, filename)
'''
file =
{'test': '/kaggle/input/titanic/test.csv',
'train': '/kaggle/input/titanic/train.csv',
'gender_submission': '/kaggle/input/titanic/gender_submission.csv'}
'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
def clean_data(dataset):
    dataset = dataset[dataset['Embarked'].notnull()]
    x = dataset.drop(['Survived'], axis=1).fillna(0)
    y = dataset['Survived'].values
    sc = StandardScaler()
    ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [1,-1])], remainder='passthrough')
    x = ct.fit_transform(x)
    x = sc.fit_transform(x)
    return x,y
data_train = pd.read_csv(file['train']).drop(['Name', 'Ticket', 'Cabin', 'Fare','PassengerId'], axis = 1)
x_train, y_train = clean_data(data_train)
K_x_train = []
K_y_train = []
for K in range(5):
  K_x_train.append(x_train[K::5])
  K_y_train.append(y_train[K::5])
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
DT = DecisionTreeClassifier(criterion='entropy', random_state = 0)
NB = GaussianNB()
NN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state = 0)
from sklearn.metrics import precision_score, recall_score, f1_score
import copy
def classifier(x, y, model):
    pre = []
    rec = []
    f1 = []
    classi = []
    x = list(K_x_train)
    y = list(K_y_train)
    for fold in range(5):
        classification = copy.deepcopy(model)
        num_k = []
        for k in range(5):
            if k != fold: classification.fit(x[k],y[k])
        pred = classification.predict(x[fold])
        pre.append(precision_score(y[fold], pred))
        rec.append(recall_score(y[fold], pred))
        f1.append(f1_score(y[fold], pred))
        classi.append(classification)
    return pre, rec, f1, classi
pre_DT, rec_DT, f1_DT, model_DT = classifier(K_x_train,K_y_train,DT)
pre_NB, rec_NB, f1_NB, model_NB = classifier(K_x_train,K_y_train,NB)
pre_NN, rec_NN, f1_NN, model_NN = classifier(K_x_train,K_y_train,NN)
print(np.mean(pre_DT), np.mean(rec_DT), np.mean(f1_DT))
print(np.mean(pre_NB), np.mean(rec_NB), np.mean(f1_NB))
print(np.mean(pre_NN), np.mean(rec_NN), np.mean(f1_NN))
test = pd.read_csv(file['test']).drop(['Name', 'Ticket', 'Cabin', 'Fare','PassengerId'], axis = 1)
y_test = pd.read_csv(file['gender_submission'])
test['Survived'] = y_test['Survived']
x_test, y_test = clean_data(test)
def predict(x, y, model):
    pre = []
    rec = []
    f1 = []
    for m in model:
        pred = m.predict(x)
        pre.append(precision_score(y, pred))
        rec.append(recall_score(y, pred))
        f1.append(f1_score(y, pred))
    return pre, rec, f1
pre_DT_t, rec_DT_t, f1_DT_t = predict(x_test, y_test, model_DT)
pre_NB_t, rec_NB_t, f1_NB_t = predict(x_test, y_test, model_NB)
pre_NN_t, rec_NN_t, f1_NN_t = predict(x_test, y_test, model_NN)
pre_DT_t.append(np.mean(pre_DT_t))
rec_DT_t.append(np.mean(rec_DT_t))
f1_DT_t.append(np.mean(f1_DT_t))
pre_NB_t.append(np.mean(pre_NB_t))
rec_NB_t.append(np.mean(rec_NB_t))
f1_NB_t.append(np.mean(f1_NB_t))
pre_NN_t.append(np.mean(pre_NN_t))
rec_NN_t.append(np.mean(rec_NN_t))
f1_NN_t.append(np.mean(f1_NN_t))
dt = [pre_DT_t, rec_DT_t, f1_DT_t]
nb = [pre_NB_t, rec_NB_t, f1_NB_t]
nn = [pre_NN_t, rec_NN_t, f1_NN_t]
css = ['class'+str(i) for i in range(1,6)]+['mean']
col = ['Precision', 'Recall', 'F1-score']
dt = np.array([pre_DT_t, rec_DT_t, f1_DT_t]).T
nb = np.array([pre_NB_t, rec_NB_t, f1_NB_t]).T
nn = np.array([pre_NN_t, rec_NN_t, f1_NN_t]).T
df_dt = pd.DataFrame(dt, index = css, columns = col)
df_dt
df_nb = pd.DataFrame(nb, index = css, columns = col)
df_nb
df_nn = pd.DataFrame(nn, index = css, columns = col)
df_nn