import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from sklearn import ensemble, naive_bayes, svm
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve, matthews_corrcoef, cohen_kappa_score
def f(s):
    if s=="negative":
        return 0
    else:
        return 1
def f1(s):
    if s=="M":
        return 2
    else:
        return 1
df=pd.read_csv("../input/ecoli3.csv")
df.shape
df["Class"]=df.Class.apply(f)
#df["Sex"]=df.Sex.apply(f)
N=df.shape[0]
M=df.shape[1]
x=df.values[:, :M-1]
y=df.values[:, M-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
x[0], y[0]
ALGOS = [
    ensemble.RandomForestClassifier(),
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.HistGradientBoostingClassifier(),
    ensemble.ExtraTreesClassifier(),
    #ensemble.VotingClassifier(),
    naive_bayes.GaussianNB(),
    svm.LinearSVC(),
]
#clf1 = ensemble.VotingClassifier(estimators=[('a', ALGOS[0]), ('b', ALGOS[1]), ('c', ALGOS[2]), ('d', ALGOS[3]), ('e', ALGOS[4]), ('f', ALGOS[5])], voting='hard')
#ALGOS.append(clf1)
ALGOS_columns = []
ALGOS_compare = pd.DataFrame(columns = ALGOS_columns)


row_index = 0
for alg in ALGOS:
    predicted = alg.fit(x_train, y_train).predict(x_test)
    fp, tp, th = roc_curve(y_test, predicted)
    ALGOS_name = alg.__class__.__name__
    ALGOS_compare.loc[row_index,'Algo Name'] = ALGOS_name
    ALGOS_compare.loc[row_index, 'Algo Precission'] = precision_score(y_test, predicted)
    ALGOS_compare.loc[row_index, 'Algo Recall'] = recall_score(y_test, predicted)
    ALGOS_compare.loc[row_index, 'Algo AUC'] = auc(fp, tp)
    ALGOS_compare.loc[row_index, 'Algo FBeta'] = fbeta_score(y_test, predicted, average='binary', beta=2)
    ALGOS_compare.loc[row_index, 'Algo MCC'] = matthews_corrcoef(y_test, predicted)
    ALGOS_compare.loc[row_index, 'Algo Kappa'] = cohen_kappa_score(y_test, predicted)
    sum0=0
    sum1=0
    tot0=0
    tot1=0
    for i in range(len(x_test)):
        if(y_test[i]==0):
            tot0+=1
            if(predicted[i]==y_test[i]):
                sum0+=1
        else:
            tot1+=1
            if(predicted[i]==y_test[i]):
                sum1+=1
    print(len(x_test), tot0, tot1, ALGOS_name)
    print(sum0+sum1, sum0, sum1)
    row_index+=1
    
ALGOS_compare.sort_values(by = ['Algo FBeta'], ascending = False, inplace = True)
ALGOS_compare