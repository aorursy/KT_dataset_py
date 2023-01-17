import numpy as np

import pandas as pd

import math

import random

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing

from sklearn.metrics import fbeta_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve, matthews_corrcoef, cohen_kappa_score

random.seed( 30 )

np.random.seed(30)
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
df=pd.read_csv("../input/dataset/yeast6.csv")

df["Class"]=df.Class.apply(f)

#df["Sex"]=df.Sex.apply(f)

N=df.shape[0]

M=df.shape[1]

x=df.values[:, :M-1]

y=df.values[:, M-1]

scaler=preprocessing.MinMaxScaler()

x[0], y[0], x.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

x.shape, len(x_train)
N=x.shape[0]

M=x.shape[1]

N,M
n = len(x_train)

num_of_best_features=int(math.sqrt(M))

num_of_nodes=(2*(M+1))//3

num_of_nodes
#set hidden layer size based on dataset

clf = MLPClassifier(activation='logistic', solver='lbfgs', random_state=1, alpha=0.05, learning_rate_init=0.5, hidden_layer_sizes=(6,5), max_iter=1000)
#FEATURE SELECTION

bestfeatures = SelectKBest(score_func=f_classif, k=num_of_best_features)

fit = bestfeatures.fit(x_train, y_train)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(df.iloc[:,0:9].columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(num_of_best_features,'Score'))
x_train_sel = bestfeatures.transform(x_train)

x_test_sel = bestfeatures.transform(x_test)

x_train_sel[:4]
len(x_train), len(x_train_sel)
num_of_best_features
#FEATURE CROSSING



preprocessed_x_train = np.empty((num_of_best_features+1, len(x_train), M))

preprocessed_x_train[0] = x_train



preprocessed_x_test = np.empty((num_of_best_features+1, len(x_test), M))

preprocessed_x_test[0] = x_test



for z in range(1, num_of_best_features+1):

    preprocessed_x_train[z] = [[x_train[i][j]*x_train_sel[i][z-1] for j in range(M)] for i in range(len(x_train))]

    

for z in range(1, num_of_best_features+1):

    preprocessed_x_test[z] = [[x_test[i][j]*x_test_sel[i][z-1] for j in range(M)] for i in range(len(x_test))]
len(x_train[0]), len(x_test[0])
#first group is original dataset, second grp is original dataset multiplied by first selected feature, second group is original dataset, multiplied by second selected feature and so on.



#PREDICTING FOR EACH NN FOR EACH FEATURE CROSSED DATASET



num_of_groups = num_of_best_features

num_of_features_in_each_group = M

 

x_train_mid=np.empty((x_train.shape[0],0))

x_test_mid=np.empty((x_test.shape[0],0))

for i in range (num_of_groups):

    clf.fit(preprocessed_x_train[i], y_train)

    y_train_mid = clf.predict(preprocessed_x_train[i])

    y_test_mid = clf.predict(preprocessed_x_test[i])

    #COMBINING PREDICTIONS OF ALL NN FOR FINAL NN

    x_train_mid = np.column_stack((x_train_mid, y_train_mid))

    x_test_mid = np.column_stack((x_test_mid, y_test_mid))
#MODIFIED NN



#PREDICTING FOR THE FINAL NN BY COMBINATION OF ALL NN STORED EARLIER

clf.fit(x_train_mid, y_train)

y_final_train_pred=clf.predict(x_train_mid)



#EVALUATION METRICS FOR THE MODEL

fpo1, tpo1, tho1 = roc_curve(y_train, y_final_train_pred)

fb_own1 = fbeta_score(y_train, y_final_train_pred, average='binary', beta=2) #binary bcos f2 of positive class is needed

mcco1 = matthews_corrcoef(y_train, y_final_train_pred)

kappao1 = cohen_kappa_score(y_train, y_final_train_pred)



y_final_test_pred=clf.predict(x_test_mid)

fpo2, tpo2, tho2 = roc_curve(y_test, y_final_test_pred)

fb_own2 = fbeta_score(y_test, y_final_test_pred, average='binary', beta=2) #binary bcos f2 of positive class is needed

mcco2 = matthews_corrcoef(y_test, y_final_test_pred)

kappao2 = cohen_kappa_score(y_test, y_final_test_pred)



sum0=0

sum1=0

tot0=0

tot1=0

for i in range(len(x_test_mid)):

        if(y_test[i]==0):

            tot0+=1

            if(y_final_test_pred[i]==y_test[i]):

                sum0+=1

        else:

            tot1+=1

            if(y_final_test_pred[i]==y_test[i]):

                sum1+=1

print(len(x_test_mid), tot0, tot1, 'own')

print(sum0+sum1, sum0, sum1)



print(precision_score(y_test, y_final_test_pred), recall_score(y_test, y_final_test_pred), auc(fpo2, tpo2), fb_own2, mcco2, kappao2)
#STANDARD NN

clf.fit(x_train, y_train)

y_train_pred=clf.predict(x_train)

fpstd1, tpstd1, thstd1 = roc_curve(y_train, y_train_pred)

fb_std1 = fbeta_score(y_train, y_train_pred, average='binary', beta=2) #binary bcos f2 of positive class is needed

mccs1 = matthews_corrcoef(y_train, y_train_pred)

kappas1 = cohen_kappa_score(y_train, y_train_pred)

y_test_pred=clf.predict(x_test)

fpstd2, tpstd2, thstd2 = roc_curve(y_test, y_test_pred)

fb_std2 = fbeta_score(y_test, y_test_pred, average='binary', beta=2) #binary bcos f2 of positive class is needed

mccs2 = matthews_corrcoef(y_test, y_test_pred)

kappas2 = cohen_kappa_score(y_test, y_test_pred)

sum0=0

sum1=0

tot0=0

tot1=0

for i in range(len(x_test)):

        if(y_test[i]==0):

            tot0+=1

            if(y_test_pred[i]==y_test[i]):

                sum0+=1

        else:

            tot1+=1

            if(y_test_pred[i]==y_test[i]):

                sum1+=1

print(len(x_test_mid), tot0, tot1, 'std')

print(sum0+sum1, sum0, sum1)



print(precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), auc(fpstd2, tpstd2), fb_std2, mccs2, kappas2)