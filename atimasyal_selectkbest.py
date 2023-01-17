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
df=pd.read_csv("../input/project/abalone9-18.csv")
df["Class"]=df.Class.apply(f)
df["Sex"]=df.Sex.apply(f)
N=df.shape[0]
M=df.shape[1]
x=df.values[:, :M-1]
y=df.values[:, M-1]
scaler=preprocessing.StandardScaler()
x[0], y[0], x.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
#scaler.fit(x_train)
#scaler.transform(x_train)
#scaler.transform(x_test)
x.shape
N=x.shape[0]
M=x.shape[1]
N,M
num_of_best_features=int(math.sqrt(M))
clf = MLPClassifier(activation='logistic', solver='lbfgs', random_state=1, alpha=0.05, learning_rate_init=0.5, hidden_layer_sizes=(5, 2,), max_iter=1000)
x[1, 0:9]
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
my_list = list(range(0, M))
int(my_list[2])
x_train[:4,int(my_list[3])]
num_of_best_features
num_of_elements_in_each_group = int(M//num_of_best_features)
random.Random(4).shuffle(my_list) 
x_train_mid=np.empty((x_train.shape[0],0))
x_test_mid=np.empty((x_test.shape[0],0))
for i in range (num_of_best_features-1):
    subset_train = x_train_sel[:, i]
    subset_test = x_test_sel[:, i]
    print(subset_train[:3])
    for j in range(num_of_elements_in_each_group):
        subset_train = np.column_stack((subset_train, x_train[:,int(my_list[j])]))
        subset_test = np.column_stack((subset_test, x_test[:,int(my_list[j])]))
        print()
        print(int(my_list[j]),x_train[:3,int(my_list[j])])
        print()
    del my_list[:num_of_elements_in_each_group]
    print(subset_train[:3])
    clf.fit(subset_train, y_train)
    y_train_mid = clf.predict(subset_train)
    y_test_mid = clf.predict(subset_test)
    x_train_mid = np.column_stack((x_train_mid, y_train_mid))
    x_test_mid = np.column_stack((x_test_mid, y_test_mid))
    #print(x_train_mid.shape, x_train_mid[3])
    #print(x_test_mid.shape, x_test_mid[3])
subset_train = x_train_sel[:, num_of_best_features-1]
subset_test = x_test_sel[:, num_of_best_features-1]
print(subset_train[:3])
for j in range(len(my_list)):
    subset_train = np.column_stack((subset_train, x_train[:,int(my_list[j])]))
    subset_test = np.column_stack((subset_test, x_test[:,int(my_list[j])]))
    print()
    print(int(my_list[j]),x_train[:3,int(my_list[j])])
    print()
print(subset_train[:3])
clf.fit(subset_train, y_train)
y_train_mid = clf.predict(subset_train)
y_test_mid = clf.predict(subset_test)
x_train_mid = np.column_stack((x_train_mid, y_train_mid))
x_test_mid = np.column_stack((x_test_mid, y_test_mid))
print(x_train_mid.shape, x_train_mid[3])
print(x_test_mid.shape, x_test_mid[3])
clf.fit(x_train_mid, y_train)
y_final_train_pred=clf.predict(x_train_mid)
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


#print(precision_score(y_train, y_final_train_pred), recall_score(y_train, y_final_train_pred), auc(fpo1, tpo1), fb_own1, mcco1, kappao1)
print(precision_score(y_test, y_final_test_pred), recall_score(y_test, y_final_test_pred), auc(fpo2, tpo2), fb_own2, mcco2, kappao2)
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
print(len(x_test_mid), tot0, tot1, 'own')
print(sum0+sum1, sum0, sum1)

#print(precision_score(y_train, y_train_pred), recall_score(y_train, y_train_pred), auc(fpstd1, tpstd1), fb_std1, mccs1, kappas1, )
print(precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), auc(fpstd2, tpstd2), fb_std2, mccs2, kappas2)
