import pandas as pd

import numpy as np

import pylab as pl

import scipy.optimize as opt

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
submission = pd.read_csv('../input/sample_submission.csv')

test_df = pd.read_csv('../input/test.csv')

train_df = pd.read_csv("../input/train.csv")
print(train_df.shape)

train_df.head()
print(test_df.shape)

test_df.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_df.dtypes.to_frame().head()
test_df.dtypes.to_frame().head()
# Some basic stats on the target variable

print ('# target = 1: {}'.format(len(train_df[train_df['target'] == 1])))

print ('# target = 0: {}'.format(len(train_df[train_df['target'] == 0])))  

print ('% target = 1: {}%'.format(round(float(len(train_df[train_df['target'] == 1])) / len(train_df) * 100))) 

tax=round(float(len(train_df[train_df['target'] == 1])) / len(train_df) * 100)
plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')

plt.pcolor(train_df.corr(), cmap='Accent')

plt.colorbar()

plt.show()
train_df_correlations = train_df.corr()

train_df_correlations = train_df_correlations.values.flatten()

train_df_correlations = train_df_correlations[train_df_correlations != 1]



plt.figure(figsize=(20,5))

sns.distplot(train_df_correlations, color="Red", label="train")

plt.xlabel("Correlation values found in train (except 1)")

plt.ylabel("Density")

plt.title("Are there correlations between features?"); 

plt.legend();
def plot_feature_distribution(df0, df1, df2, label0, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(10,10,figsize=(24,18))



    for feature in features:

        i += 1

        plt.subplot(10,10,i)

        sns.kdeplot(df0[feature], bw=0.5,label=label0)

        sns.kdeplot(df1[feature], bw=0.5,label=label1)

        sns.kdeplot(df2[feature], bw=0.5,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.show();
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

t2 = test_df

features = train_df.columns.values[2:102]

plot_feature_distribution(t0, t1, t2, '0', '1', 'test', features)
features = train_df.columns.values[102:202]

plot_feature_distribution(t0, t1, t2, '0', '1', 'test', features)
features = train_df.columns.values[202:302]

plot_feature_distribution(t0, t1, t2, '0', '1', 'test', features)
d=pd.Series(range(0,300))
for df in [test_df, train_df]: 

    d=pd.Series(d)

    d=(d + 1).tolist()

    for x in d:

        df[df.columns[x]]=(df[df.columns[x]]-df[df.columns[x]].min())/(df[df.columns[x]].max()-df[df.columns[x]].min())

        df[df.columns[x]]=np.log((1+df[df.columns[x]]))
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

t2 = test_df

features = train_df.columns.values[2:102]

plot_feature_distribution(t0, t1, t2, '0', '1', 'test', features)
features = train_df.columns.values[102:202]

plot_feature_distribution(t0, t1, t2, '0', '1', 'test', features)
features = train_df.columns.values[202:302]

plot_feature_distribution(t0, t1, t2, '0', '1', 'test', features)
#sample=['target','4','13','16','24','33','65','73','80','91','183','189','194','199','217','276','295','298']

sample= ['target','16', '33', '80', '91', '217', '295']
sns.set(style="ticks", color_codes=True)

pair_sample=train_df[sample]

# Pairwise plots

pplot = sns.pairplot(pair_sample, hue="target", height=3, kind ='scatter', diag_kind='kde', plot_kws=dict(s=20, linewidth=0) ) 
#Ex.: between the variables

sns.lmplot( x='33', y='217', data=train_df, fit_reg=False, hue='target', legend=True)

sns.lmplot( x='33', y='16', data=train_df, fit_reg=False, hue='target', legend=True)
#feat_engin=[4,13,16,24,33,65,73,80,91,183,189,194,199,217,276,295,298]

#feat_engin=[33,217,295,298]

#feat_engin=[2,4,6,11,14,17,21,29,33,45,46,61,70,71,74,76,80,131,132,135,141,177,205,231,246,293]

#feat_engin2=[1,13,16,25,26,42,48,65,66,83,111,116,117,138,147,176,195,228,237,266]
#for df in [test_df, train_df]:

#      

#    d=pd.Series(feat_engin)

#    d=(d + 1).tolist()

#           

#    df['300'] = df[df.columns[d]].sum(axis=1)  

#    df['301'] = df[df.columns[d]].min(axis=1) 

#    df['302'] = df[df.columns[d]].max(axis=1) 

#    df['303'] = df[df.columns[d]].mean(axis=1) 

#    df['304'] = df[df.columns[d]].var(axis=1)

#    df['305'] = df[df.columns[d]].sum(axis=1)+df[df.columns[d]].median(axis=1)

#    df['306'] = df[df.columns[d]].std(axis=1) 

#    df['307'] = df[df.columns[d]].mean(axis=1)

#    df['308'] = df[df.columns[d]].median(axis=1)

#    

#    d2=pd.Series(feat_engin2)

#    d2=(d2 + 1).tolist()

           

#    df['309'] = df[df.columns[d2]].sum(axis=1)  

#    df['310'] = df[df.columns[d2]].min(axis=1) 

#    df['311'] = df[df.columns[d2]].max(axis=1) 

#    df['312'] = df[df.columns[d2]].mean(axis=1) 

#    df['313'] = df[df.columns[d2]].var(axis=1)

#    df['314'] = df[df.columns[d2]].sum(axis=1)+df[df.columns[d2]].median(axis=1)

#    df['315'] = df[df.columns[d2]].std(axis=1) 

#    df['316'] = df[df.columns[d2]].mean(axis=1)

#    df['317'] = df[df.columns[d2]].median(axis=1)
#train_df[train_df.columns[302:]].head()
#test_df[test_df.columns[301:]].head()
#def plot_new_feature_distribution(df1, df2, label1, label2, features):

#    i = 0

#    sns.set_style('whitegrid')

#    plt.figure()

#    fig, ax = plt.subplots(3,6,figsize=(24,9))

#

#    for feature in features:

#        i += 1

#        plt.subplot(3,6,i)

#        sns.kdeplot(df1[feature], bw=0.5,label=label1)

#        sns.kdeplot(df2[feature], bw=0.5,label=label2)

#        plt.xlabel(feature, fontsize=9)

#        locs, labels = plt.xticks()

#        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

#        plt.tick_params(axis='y', which='major', labelsize=6)

#    plt.show();
#t0 = train_df.loc[train_df['target'] == 0]

#t1 = train_df.loc[train_df['target'] == 1]

#features = train_df.columns.values[302:]

#plot_new_feature_distribution(t0, t1, '0', '1', features)
#sns.lmplot( x='305', y='314', data=train_df, fit_reg=False, hue='target', legend=True)

#sns.lmplot( x='305', y='309', data=train_df, fit_reg=False, hue='target', legend=True)
#z_selection=pd.DataFrame()

#d=pd.Series()

#z_box=train_df.groupby(['target']).describe()

#z_box.head()
#i=0

#minim=1

#for x in train_df.columns.drop(['id','target']):

#    z_selection[x]=[abs((z_box[x]['25%'].iloc[(0)]-z_box[x]['25%'].iloc[(1)])/(z_box[x]['25%'].iloc[(0)]-z_box[x]['75%'].iloc[(1)]))+

#                   abs((z_box[x]['75%'].iloc[(0)]-z_box[x]['75%'].iloc[(1)])/(z_box[x]['75%'].iloc[(0)]-z_box[x]['25%'].iloc[(1)]))]

#d=np.where(z_selection>0.5)[1]

#d=feat_engin

d=sample[1:]
result=pd.DataFrame()
y = train_df['target'] 

X = train_df.drop(['target','id'], axis=1) # Data set to create and tunning the model

X_ol=X[d] # Data set with the selected features  - to be studied if is an improvement.

print ('Train set:', X.shape,  y.shape, X_ol.shape)
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=6)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)



X_train_ol=X_train[d]

X_test_ol=X_test[d]



print ('Train set select:', X_train_ol.shape,  y_train.shape)

print ('Test set select:', X_test_ol.shape,  y_test.shape)
print ('% target = 1 in train set: {}%'.format(round(float(len(y_train[y_train==1])) / len(y_train) * 100))) 

print ('% target = 1 in test set: {}%'.format(round(float(len(y_test[y_test==1])) / len(y_test) * 100)))
from sklearn import linear_model

#Model with all 300 features

regr_all = linear_model.LinearRegression(fit_intercept=True, normalize=True)

regr_all.fit(X_train,y_train)
y_hat= regr_all.predict(X_test)

y_hat=y_hat*0.5/(np.sort(y_hat)[100-tax]) # --> ajust the array to have 64% of target > 0.5

result['regr_all']=y_hat

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat>=0.5])) / len(y_hat) * 100))) 



print("Residual sum of squares: %.2f" % np.mean((y_hat - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % regr_all.score(X_test, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat)) #1 is best
#Model with the selected features

regr_ol = linear_model.LinearRegression(fit_intercept=True, normalize=True)

regr_ol.fit(X_train_ol,y_train)
regr_ol.predict(X_test_ol)

#y_hat=(y_hat-y_hat.min())/(y_hat.max()-y_hat.min()) # turn the result to be between 0 and 1

y_hat=y_hat*0.5/(np.sort(y_hat)[100-tax]) # --> ajust the array to have 64% of target > 0.5

result['regr_ol']=y_hat

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % regr_ol.score(X_test_ol, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat)) #1 is best
from sklearn.feature_selection import RFE

#Selecting variables via RFE

i=0

score=pd.DataFrame()

for nfeatures in range(5,301,3):

    selector = RFE(regr_all, nfeatures, step=1,verbose=0)

    selector.fit(X_train,y_train)

    y_hat= selector.predict(X_test)

    score[i]=[nfeatures,roc_auc_score(y_test, y_hat)]

    #print(score, nfeatures)

    i=i+1

    

score=score.transpose()

score.columns=['NumberOfFeatures','AUC ROC']

plt.plot(score['NumberOfFeatures'],score['AUC ROC'])

plt.xlabel('Number of Features')

plt.ylabel('AUC ROC')

plt.title('AUC ROC x Number of features')

plt.grid(True)

plt.show() # best is between 90 and 100 or 201 and 210



#selector.get_support(indices=True) # selected features via RFE
best=score[score['AUC ROC']>score['AUC ROC'].max()*0.95]

best
from sklearn.feature_selection import RFE

regr_sel = RFE(regr_all, 4, step=1,verbose=0)

regr_sel.fit(X_train,y_train)
y_hat= regr_sel.predict(X_test)

y_hat=y_hat*0.5/(np.sort(y_hat)[100-tax]) # --> ajust the array to have 64% of target > 0.5

result['regre_sel']=y_hat

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % regr_sel.score(X_test, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat)) #1 is best
from sklearn.tree import DecisionTreeClassifier

param_grid = { 'criterion': ['gini','entropy'],

              'splitter': ['best','random'],

              'max_depth': [2,4,10,20,None],

              'max_features' : ['auto', 'sqrt', 'log2']}

Tree = DecisionTreeClassifier()
grid = GridSearchCV(estimator=Tree, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1,cv=2)

grid.fit(X_train,y_train)

print("Best Score= " + str(grid.best_score_))

print ("Best Parameters= "+str(grid.best_params_))

best_param=grid.best_params_
Tree = DecisionTreeClassifier(**best_param)

Tree.fit(X_train,y_train)
y_hat= Tree.predict(X_test)

result['Tree']=y_hat

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % Tree.score(X_test, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat)) #1 is best
grid = GridSearchCV(estimator=Tree, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1,cv=2)

grid.fit(X_train_ol,y_train)

print("Best Score= " + str(grid.best_score_))

print ("Best Parameters= "+str(grid.best_params_))

best_param=grid.best_params_
Tree_ol = DecisionTreeClassifier(**best_param)

Tree_ol.fit(X_train_ol,y_train)
y_hat= Tree_ol.predict(X_test_ol)

result['Tree_ol']=y_hat

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % Tree_ol.score(X_test_ol, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat)) #1 is best
#Selecting variables via RFE

i=0

score=pd.DataFrame()

for nfeatures in range(5,301,5):

    selector = RFE(Tree, nfeatures, step=1,verbose=0)

    selector.fit(X_train,y_train)

    y_hat= selector.predict(X_test)

    score[i]=[nfeatures,roc_auc_score(y_test, y_hat)]

    #print(score, nfeatures)

    i=i+1

    

score=score.transpose()

score.columns=['NumberOfFeatures','AUC ROC']

plt.plot(score['NumberOfFeatures'],score['AUC ROC'])

plt.xlabel('Number of Features')

plt.ylabel('AUC ROC')

plt.title('AUC ROC x Number of features')

plt.grid(True)

plt.show() 

Tree_selec = RFE(Tree, 94, step=1,verbose=0)

Tree_selec.fit(X_train,y_train)
y_hat= Tree_selec.predict(X_test)

result['Tree_selec']=y_hat

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % Tree_selec.score(X_test, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat)) #1 is best
from sklearn.linear_model import LogisticRegression

param_grid = { 'C': [0.001,0.1,10.0,100.0],

              'fit_intercept': [True,False],

              'class_weight': ['balanced', None],

              'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

LR = LogisticRegression()
grid = GridSearchCV(estimator=LR, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1,cv=2)

grid.fit(X_train,y_train)

print("Best Score= " + str(grid.best_score_))

print ("Best Parameters= "+str(grid.best_params_))

best_param=grid.best_params_
LR = LogisticRegression(**best_param)

LR.fit(X_train,y_train)
y_hat= LR.predict_proba(X_test)

y_hat=y_hat*0.5/(np.sort(y_hat[:,1])[100-tax]) # --> ajust the array to have 64% of target > 0.5

result['LR']=y_hat[:,1]

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat[:,1]>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat[:,1] - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % LR.score(X_test, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat[:,1])) #1 is best
grid = GridSearchCV(estimator=LR, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1,cv=2)

grid.fit(X_train_ol,y_train)

print("Best Score= " + str(grid.best_score_))

print ("Best Parameters= "+str(grid.best_params_))

best_param=grid.best_params_
LR_ol = LogisticRegression(**best_param)

LR_ol.fit(X_train_ol,y_train)
y_hat= LR_ol.predict_proba(X_test_ol)

y_hat=y_hat*0.5/(np.sort(y_hat[:,1])[100-tax]) # --> ajust the array to have 64% of target > 0.5

result['LR_ol']=y_hat[:,1]

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat[:,1]>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat[:,1] - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % LR_ol.score(X_test_ol, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat[:,1])) #1 is best
#Selecting variables via RFE

i=0

score=pd.DataFrame()

for nfeatures in range(5,301,3):

    selector = RFE(LR, nfeatures, step=1,verbose=0)

    selector.fit(X_train,y_train)

    y_hat= selector.predict(X_test)

    score[i]=[nfeatures,roc_auc_score(y_test, y_hat)]

    #print(score, nfeatures)

    i=i+1

    

score=score.transpose()

score.columns=['NumberOfFeatures','AUC ROC']

plt.plot(score['NumberOfFeatures'],score['AUC ROC'])

plt.xlabel('Number of Features')

plt.ylabel('AUC ROC')

plt.title('AUC ROC x Number of features')

plt.grid(True)

plt.show() 
#take the best results

best=score[score['AUC ROC']>score['AUC ROC'].max()*0.95]

best
LR_selec = RFE(LR, 14, step=1,verbose=0)

LR_selec.fit(X_train,y_train)
y_hat= LR_selec.predict_proba(X_test)

y_hat=y_hat*0.5/(np.sort(y_hat[:,1])[100-tax]) # --> ajust the array to have 64% of target > 0.5

result['LR_select']=y_hat[:,1]

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat[:,1]>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat[:,1] - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % LR_selec.score(X_test, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat[:,1])) #1 is best
from sklearn.neural_network import MLPClassifier

param_grid = { 'activation': ['identity', 'logistic', 'tanh', 'relu'],

              'solver': ['lbfgs', 'sgd', 'adam'],

              'learning_rate': ['constant', 'invscaling', 'adaptive']}

MLPerseptron = MLPClassifier()
grid = GridSearchCV(estimator=MLPerseptron, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1,cv=2)

grid.fit(X_train,y_train)

print("Best Score= " + str(grid.best_score_))

print ("Best Parameters= "+str(grid.best_params_))

best_param=grid.best_params_
MLPerseptron = MLPClassifier(**best_param)

MLPerseptron.fit(X_train,y_train)
y_hat= MLPerseptron.predict_proba(X_test)

y_hat=y_hat*0.5/(np.sort(y_hat[:,1])[100-tax]) # --> ajust the array to have 64% of target > 0.5

result['MLPerseptron']=y_hat[:,1]

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat[:,1]>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat[:,1] - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % MLPerseptron.score(X_test, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat[:,1])) #1 is best
grid = GridSearchCV(estimator=MLPerseptron, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1,cv=2)

grid.fit(X_train_ol,y_train)

print("Best Score= " + str(grid.best_score_))

print ("Best Parameters= "+str(grid.best_params_))

best_param=grid.best_params_
MLPerseptron_ol = MLPClassifier(**best_param)

MLPerseptron_ol.fit(X_train_ol,y_train)
y_hat= MLPerseptron_ol.predict_proba(X_test_ol)

y_hat=y_hat*0.5/(np.sort(y_hat[:,1])[100-tax]) # --> ajust the array to have 64% of target > 0.5

result['MLPerseptron_ol']=y_hat[:,1]

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat[:,1]>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat[:,1] - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % MLPerseptron_ol.score(X_test_ol, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat[:,1])) #1 is best
from sklearn.naive_bayes import GaussianNB

param_grid = { 'priors': [None],

              'var_smoothing': [1e-3,1e-6,1e-9,1e-12]}

g_NB = GaussianNB()
grid = GridSearchCV(estimator=g_NB, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1,cv=2)

grid.fit(X_train,y_train)

print("Best Score= " + str(grid.best_score_))

print ("Best Parameters= "+str(grid.best_params_))

best_param=grid.best_params_
g_NB = GaussianNB(**best_param)

g_NB.fit(X_train, y_train)
y_hat= g_NB.predict_proba(X_test)

y_hat=y_hat*0.5/(np.sort(y_hat[:,1])[100-tax]) # --> ajust the array to have 64% of target > 0.5

result['g_NB']=y_hat[:,1]

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat[:,1]>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat[:,1] - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % g_NB.score(X_test, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat[:,1])) #1 is best
grid.fit(X_train_ol,y_train)

print("Best Score= " + str(grid.best_score_))

print ("Best Parameters= "+str(grid.best_params_))

best_param=grid.best_params_
grid.fit(X_train_ol,y_train)

best_param=grid.best_params_

g_NB_ol = GaussianNB(**best_param)

g_NB_ol.fit(X_train_ol, y_train)
y_hat= g_NB.predict_proba(X_test)

y_hat=y_hat*0.5/(np.sort(y_hat[:,1])[100-tax]) # --> ajust the array to have 64% of target > 0.5

result['g_NB_ol']=y_hat[:,1]

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat[:,1]>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat[:,1] - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % g_NB.score(X_test, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat[:,1])) #1 is best
from sklearn.neighbors import KNeighborsClassifier

param_grid = { 'n_neighbors': list(range(4,100)),

              'weights': ['uniform','distance'],

              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

              'p' : [1,2]}

knn = KNeighborsClassifier()
grid = GridSearchCV(estimator=knn, param_grid=param_grid, scoring=  'roc_auc', verbose=1, n_jobs=-1)

grid.fit(X_train,y_train)

print("Best Score= " + str(grid.best_score_))

print ("Best Parameters= "+str(grid.best_params_))

best_param=grid.best_params_
knn = KNeighborsClassifier(**best_param)

knn.fit(X_train, y_train)
y_hat= knn.predict_proba(X_test)

y_hat=y_hat*0.5/(np.sort(y_hat[:,1])[100-tax]) # --> ajust the array to have 64% of target > 0.5

result['knn']=y_hat[:,1]

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat[:,1]>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat[:,1] - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % knn.score(X_test, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat[:,1])) #1 is best
grid.fit(X_train_ol,y_train)

print("Best Score= " + str(grid.best_score_))

print ("Best Parameters= "+str(grid.best_params_))

best_param=grid.best_params_
grid.fit(X_train_ol,y_train)

best_param=grid.best_params_

knn_ol = KNeighborsClassifier(**best_param)

knn_ol.fit(X_train_ol, y_train)
y_hat= knn_ol.predict_proba(X_test_ol)

y_hat=y_hat*0.5/(np.sort(y_hat[:,1])[100-tax]) # --> ajust the array to have 64% of target > 0.5

result['knn_ol']=y_hat[:,1]

print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat[:,1]>=0.5])) / len(y_hat) * 100)))



print("Residual sum of squares: %.2f" % np.mean((y_hat[:,1] - y_test) ** 2)) # 0 is best score

print('Variance score: %.2f' % knn_ol.score(X_test_ol, y_test)) # 1 is perfect prediction

print('AUC ROC: %.2f' % roc_auc_score(y_test, y_hat[:,1])) #1 is best
temp=pd.DataFrame()

temp['mean']=result.mean(axis=1) 

result2=pd.concat([result, temp], axis=1)
FinalResult=pd.DataFrame()

for results in result2.columns:

    y_hat=result2[results]

    AUC_ROC=roc_auc_score(y_test, y_hat)

    FinalResult[results]=[AUC_ROC]

FinalResult.transpose()
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
def plot_AUCROC(y_test, y_hat, labels):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(4,5,figsize=(20,12))



    for Label in labels:

        # calculate the fpr and tpr for all thresholds of the classification

        fpr, tpr, threshold = metrics.roc_curve(y_test, y_hat[Label])

        roc_auc = metrics.auc(fpr, tpr)



        # method I: plt

        i += 1

        plt.subplot(4,5,i)    

        plt.title(Label)

        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)

        plt.legend(loc = 'lower right')

        plt.plot([0, 1], [0, 1],'r--')

        plt.xlim([0, 1])

        plt.ylim([0, 1])

        plt.ylabel('True Positive Rate')

        plt.xlabel('False Positive Rate')

        plt.tick_params(axis='x', which='major', labelsize=1, pad=-10)

        plt.tick_params(axis='y', which='major', labelsize=1)



plt.show()
labels = result2.columns

plot_AUCROC(y_test,result2, labels)
X_Test = test_df.drop(columns=["id"])

X_Test_ol=X_Test[d]

#LR.fit(X,y)

LR_ol.fit(X_ol,y)

#regr_ol.fit(X_ol, y)  

#g_NB.fit(X, y)

#selector.fit(X,y)
#y_hat=g_NB.predict_proba(X_Test)

#y_hat=regr_ol.predict(X_Test_ol)

#y_hat=(y_hat-y_hat.min())/(y_hat.max()-y_hat.min())

#y_hat= neigh_grid.predict_proba(X_Test_ol)

#y_hat = gbm.predict(X_Test, num_iteration=gbm.best_iteration)

#y_hat= selector.predict(X_Test)

y_hat= knn_ol.predict_proba(X_Test_ol)

y_hat=y_hat*0.5/(np.sort(y_hat[:,1])[round((y_hat[:,1].size)*(100-tax)/100)]) # --> ajust the array to have 64% of target > 0.5



#submission['target']=y_hat

submission['target']=y_hat[:,1]



print ('% target = 1: {}% (in the train sample 64%)'.format(round(float(len(y_hat[y_hat[:,1]>=0.5])) / len(y_hat) * 100)))
submission.to_csv('submission.csv', index=False)

submission.head(20)