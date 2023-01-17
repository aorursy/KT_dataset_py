# tennis match result prediction
# aim is to predict the winner of a tennis match
# this is a classification problem
# before predictive modellind the data will be prepared and analysed
# the ATP men's matches data has been collected from http://tennis-data.co.uk/
import numpy as np
import pandas as pd
from glob import glob

# loading all the files into a dataframe

stock_files=sorted(glob('../input/atpmensmatches20002017/data/atp_matches_*.xlsx'))
len(stock_files)
tmp=pd.concat((pd.read_excel(file).assign(filename=file)
          for file in stock_files), ignore_index=True)
tmp.info() # information about the data frame
# checking missing values and unique values in each column
# features with a huge number of missing values will be removed
# missing values in other columns will be replaced according to th type of data they contain
# mostly features will be replaced by the mean of the rest of the valyes
# in case the data represents gender or surface type where mean cannot be used, forward and backward fill methods will be used
for i in tmp.columns:
    print(f"For variable {i}, percentage of missing values is: \
    {(tmp[i].isnull().sum()/len(tmp[i])*100)} %\nThe number of unique values is:\
    {len(tmp[i].unique())}")
    print("\n")
# the data needs to be reformatted, but before that the missing values need to be replaced
# will not bother replacing missing values of all columns, only those that will be included in model building
# tourney_id, touney_name,draw_size,tourney_date, match_num, winner_id, loser_id, score, best_of are going to be dropped, so their\
#missing valeus need not be replaced
# winner_seed, winner_entry, loser_seed, loser_entry will be dropped due to the presence of a huge number of missing values

# replacing missing values

fnf=['surface','tourney_level','winner_name','loser_name','winner_hand','loser_hand',\
     'winner_rank','loser_rank','round','w_SvGms']
fnm=['winner_ht','loser_ht','winner_age','loser_age',\
     'winner_rank_points','loser_rank_points','minutes']
fni=['w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon',\
     'w_SvGms','w_bpSaved','w_bpFaced','l_ace','l_df','l_svpt',\
     'l_1stIn','l_1stWon','l_2ndWon','l_SvGms','l_bpSaved','l_bpFaced']

j=2
for i in fnf:
    if j%2==0:
        tmp[f'{i}'].fillna(method='bfill',inplace=True)
    else:
        tmp[f'{i}'].fillna(method='ffill',inplace=True)
    j+=1

for k in fnm:
    tmp[f'{k}'].fillna(np.mean(tmp[f'{k}']),inplace=True)

for l in fni:
    tmp[f'{l}'].fillna(int(np.mean(tmp[f'{l}'])),inplace=True)

m=2
for n in tmp['l_SvGms']:
    if n==np.NaN:
        if m%2==0:
            tmp.loc[f'{m-2}','l_SvGms']=tmp.loc[f'{m-2}','w_SvGms']+1
        else:
            tmp.loc[f'{m-2}','l_SvGms']=tmp.loc[f'{m-2}','w_SvGms']-1

# in our final dataframe, minutes will be the only continuous variable
# the outliers and missing values in minutes will be replaced a little\differently

# shortest tennis match: 18 minutes, longest: 665 minutes
# if minute not inside the above range, value is an outlier

otl=[]
for i in tmp['minutes']:
    if i<28 or i>665:
        if i not in otl:
            otl.append(i)
        # replacing the outlying values with Nan
        tmp['minutes']=tmp['minutes'].replace(i,np.NaN)
        # replacing NaN with mean   
        tmp['minutes'].fillna(int(np.mean(tmp['minutes'])),inplace=True)
print("The outliers in minutes column were: -\n",otl)
# reformatting data

data = pd.DataFrame()
losers = pd.DataFrame()
winners = pd.DataFrame()

losers['tourney_level']=tmp['tourney_level']
losers['surface']=tmp['surface']
losers['player']=tmp['loser_name']
losers['opponent']=tmp['winner_name']
losers['pla_hand']=tmp['loser_hand']
losers['opp_hand']=tmp['winner_hand']
losers['pla_taller']=0+(tmp['loser_ht']>tmp['winner_ht'])
losers['pla_younger']=0+(tmp['loser_age']>tmp['winner_age'])
losers['pla_rankedHigher']=0+(tmp['loser_rank']<tmp['winner_rank'])
losers['pla_moreRankPoints']=0+(tmp['loser_rank_points']>tmp['winner_rank_points'])
losers['pla_moreAces']=0+(tmp['l_ace']>tmp['w_ace'])
losers['pla_lessDF']=0+(tmp['l_df']<tmp['w_df'])
losers['pla_1stInMore']=0+(tmp['l_1stIn']>tmp['w_1stIn'])
losers['pla_1stWonMore']=0+(tmp['l_1stWon']>tmp['w_1stWon'])
losers['pla_2ndWonMore']=0+(tmp['l_2ndWon']>tmp['w_2ndWon'])

losers['pla_bpSavedMore']=0+(tmp['l_bpSaved']>tmp['w_bpSaved'])
losers['pla_bpFacedMore']=0+(tmp['l_bpFaced']>tmp['w_bpFaced'])
losers['round']=tmp['round']
losers['minutes']=tmp['minutes']
losers['pla_win']=0

winners['tourney_level']=tmp['tourney_level']
winners['surface']=tmp['surface']
winners['player']=tmp['winner_name']
winners['opponent']=tmp['loser_name']
winners['pla_hand']=tmp['winner_hand']
winners['opp_hand']=tmp['loser_hand']
winners['pla_taller']=0+(tmp['winner_ht']>tmp['loser_ht'])
winners['pla_younger']=0+(tmp['winner_age']>tmp['loser_age'])
winners['pla_rankedHigher']=0+(tmp['winner_rank']<tmp['loser_rank'])
winners['pla_moreRankPoints']=0+(tmp['winner_rank_points']>tmp['loser_rank_points'])
winners['pla_moreAces']=0+(tmp['w_ace']>tmp['l_ace'])
winners['pla_lessDF']=0+(tmp['w_df']<tmp['l_df'])
winners['pla_1stInMore']=0+(tmp['w_1stIn']>tmp['l_1stIn'])
winners['pla_1stWonMore']=0+(tmp['w_1stWon']>tmp['l_1stWon'])
winners['pla_2ndWonMore']=0+(tmp['w_2ndWon']>tmp['l_2ndWon'])

winners['pla_bpSavedMore']=0+(tmp['w_bpSaved']>tmp['l_bpSaved'])
winners['pla_bpFacedMore']=0+(tmp['w_bpFaced']>tmp['l_bpFaced'])
winners['round']=tmp['round']
winners['minutes']=tmp['minutes']
winners['pla_win']=1

data = pd.concat([losers, winners], ignore_index=True, sort=True)
# rearranging the labels
data=data[['tourney_level','surface','player','opponent','pla_hand','opp_hand','pla_taller','pla_younger','pla_rankedHigher',\
           'pla_moreRankPoints','pla_moreAces','pla_lessDF','pla_1stInMore','pla_1stWonMore','pla_2ndWonMore',\
           'pla_bpSavedMore','pla_bpFacedMore','round','minutes','pla_win']]
data.head()
# UNIVARIATE analysis

import matplotlib.pyplot as plt
import seaborn as sns

# comparative columns like pla_taller, pla_younger, pla_rankedHigher...... do\
#not conclude anything meaningful unless checked against another variable (bivariate analysis)
pl=['tourney_level','surface','pla_hand','opp_hand','round','pla_win']
for i in pl:
    figure=sns.countplot(x=f'{i}',data=data, palette='Blues_d')
    plt.show(figure)
# UNIVARIATE analysis

plt.figure(figsize=(18,10))
sns.distplot(data['minutes'])
plt.show()
# UNIVARIATE analysis

tmp['winner_name'].dropna().value_counts()
# UNIVARIATE analysis

tmp['winner_ioc'].dropna().value_counts()
# BIVARIATE analysis

pl1=['tourney_level','surface','pla_hand','opp_hand','pla_taller','pla_younger',\
    'pla_rankedHigher','pla_moreRankPoints','pla_moreAces','pla_lessDF',\
    'pla_1stInMore','pla_1stWonMore','pla_2ndWonMore','pla_bpSavedMore',\
    'pla_bpFacedMore','round','pla_win']
for j in pl1[0:-1]:
    figure1=sns.countplot(x='pla_win',hue=f'{j}',data=data,palette='Blues_d')
    plt.title(f'{j} vs pla_win')
    plt.legend(loc='upper right')
    plt.show(figure1)
# converting few categorical variables into binary categorical variables: -
modsur=pd.get_dummies(data["surface"],drop_first=True)

modph=pd.get_dummies(data["pla_hand"],drop_first=True)
modph.rename(columns={'R':'pla_R','U':'pla_U'},inplace=True) 

modoh=pd.get_dummies(data["opp_hand"],drop_first=True)
modoh.rename(columns={'R':'opp_R','U':'opp_U'},inplace=True)

data=pd.concat((data,modsur,modph,modoh),axis=1)
data.drop(['surface','pla_hand','opp_hand'],axis=1,inplace=True)

data.head()
# model building on this final dataframe
data.info()
# split data into training and test set
from sklearn.model_selection import train_test_split
X=data.drop(['tourney_level','player','opponent','pla_win','round'],axis=1)
y=data['pla_win']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
print(f'{logmodel}\n')

# predictions
logr_pred=logmodel.predict(X_test)
logr=accuracy_score(y_test,logr_pred)*100
print(f"Accuracy from logistic regression: {logr} %")
print(f"F1 score: {f1_score(y_test,logr_pred)}\n")

from sklearn.metrics import confusion_matrix
conm=confusion_matrix(y_test,logr_pred)
print(f"Confusion Matrix: -\n{conm}")

acc,pre,rec=((conm[0,0]+conm[1,1])/conm.sum())*100,conm[0,0]/\
(conm[0,0]+conm[0,1]),conm[0,0]/(conm[0,0]+conm[1,0])
print(f"\nFrom confusion matrix: -\n1. Accuracy = {acc} %\n\
2. Precision = {pre}\n\
3. Recall/Sensitivity = {rec}\n\
4. Specificity = {conm[1,1]/(conm[1,1]+conm[0,1])}")
# decision tree
# function to perform training using entropy
from sklearn.tree import DecisionTreeClassifier
classifier_entropy=DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=3)
#create the model
classifier_entropy.fit(X_train,y_train)
print(classifier_entropy,"\n")

# make predictions
dt_pred=classifier_entropy.predict(X_test)
dt=accuracy_score(y_test,dt_pred)*100
print(f"Accuracy from decision tree: {dt} %")
print(f"F1 score: {f1_score(y_test,dt_pred)}\n")

cond=confusion_matrix(y_test,dt_pred)
print(f"Confusion Matrix:-\n{cond}")
accd,pred,recd=((cond[0,0]+cond[1,1])/cond.sum())*100,cond[0,0]/(cond[0,0]+cond[0,1]),\
cond[0,0]/(cond[0,0]+cond[1,0])
print(f"\nFrom confusion matrix: -\n1. Accuracy = {accd} %\n\
2. Precision = {pred}\n\
3. Recall/Sensitivity = {recd}\n\
4. Specificity = {cond[1,1]/(cond[1,1]+cond[0,1])}")
#knn

from sklearn.neighbors import KNeighborsClassifier
import math
n_neighbors=int(math.sqrt(len(X_test)))
if (n_neighbors%2)==0:
    n_neighbors+=1
    
classifier=KNeighborsClassifier(n_neighbors,metric='euclidean')
classifier.fit(X_train,y_train)
print(f"{classifier}\n")
# knn continued

knn_pred=classifier.predict(X_test)
cm=confusion_matrix(y_test,knn_pred)
knn=f1_score(y_test,knn_pred)*100
print(f"Accuracy from k nearest neighbours: {knn} %\n")
print(f"F1 score: {f1_score(y_test,knn_pred)}\n")
print(f"Confusion Matrix: _\n{cm}")
acck,prek,reck=((cm[0,0]+cm[1,1])/cm.sum())*100,cm[0,0]/(cm[0,0]+cm[0,1]),\
cm[0,0]/(cm[0,0]+cm[1,0])
print(f"\nFrom confusion matrix: -\n1. Accuracy = {acck} %\n\
2. Precision = {prek}\n\
3. Recall/Sensitivity = {reck}\n\
4. Specificity = {cm[1,1]/(cm[1,1]+cm[0,1])}")
# naive bayes
from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()
gnb.fit(X_train,y_train)
print(f"{gnb}\n")

# prediction
gnb_pred=gnb.predict(X_test)
gnba=accuracy_score(y_test,gnb_pred)*100
print(f"Accuracy from gaussian naive bayes: {gnba} %")
print(f"F1 score: {f1_score(y_test,gnb_pred)}\n")

cmn=confusion_matrix(y_test,gnb_pred)
print(f"Confusion Matrix: -\n{cmn}")
accn,pren,recn=((cmn[0,0]+cmn[1,1])/cmn.sum())*100,cmn[0,0]/(cmn[0,0]+cmn[0,1]),\
cmn[0,0]/(cmn[0,0]+cmn[1,0])
print(f"\nFrom confusion matrix: -\n1. Accuracy = {accn} %\n\
2. Precision = {pren}\n\
3. Recall/Sensitivity = {recn}\n\
4. Specificity = {cmn[1,1]/(cmn[1,1]+cmn[0,1])}")
# ensemble learning
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

num_trees=100
max_features=8
kfold=model_selection.KFold(n_splits=10, shuffle=True, random_state=7)
modelr=RandomForestClassifier(n_estimators=num_trees,max_features=max_features)
modelr.fit(X_train,y_train)
print(modelr,"\n")


# prediction
r_pred = modelr.predict(X_test)
ra=accuracy_score(y_test,r_pred)*100
print(f"Accuracy from random forest classifier: {ra}%")
print(f"F1 score: {f1_score(y_test,r_pred)}\n")

cmr=confusion_matrix(y_test,r_pred)
print(f"Confusion Matrix: -\n{cmr}")
accr,prer,recr=((cmr[0,0]+cmr[1,1])/cmr.sum())*100,cmr[0,0]/(cmr[0,0]+cmr[0,1]),\
cmr[0,0]/(cmr[0,0]+cmr[1,0])
print(f"\nFrom confusion matrix: -\n1. Accuracy = {accr} %\n\
2. Precision = {prer}\n\
3. Recall/Sensitivity = {recr}\n\
4. Specificity = {cmr[1,1]/(cmr[1,1]+cmr[0,1])}\n")

results=model_selection.cross_val_score(modelr,X,y,cv=kfold)
print(results.mean())
# accuracies

sc=pd.DataFrame({"model_name":['logistic regression','decision tree',\
                               'kNN','naive bayes','random forest classifier'],\
                 "accuracy(%)":[logr,dt,knn,gnba,ra]})
plt.figure(figsize=(10,5))
plt.title('accuracy comparison')
sns.barplot(x='model_name', y='accuracy(%)', data=sc, palette='Blues_d')
print(sc)
# cross-validation
from sklearn import model_selection
seed=7
models=[]
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DECISION TREE', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RFC', RandomForestClassifier()))

# evaluate each model in turn
results=[]
names=[]
scoring='accuracy'
for name, model in models:
    kfold=model_selection.KFold(n_splits=n_neighbors,shuffle=True,random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f" % (name,cv_results.mean())
    print(msg)
#boxplot algorithm comparison    
fig=plt.figure()
fig.suptitle('Algorithm comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# accuracies post cross-validation

sc=pd.DataFrame({"model_name":['logistic regression','decision tree',\
                               'kNN','naive bayes','random forest classifier'],\
                 "accuracy(%)":[87.3402,82.3244,82.7658,84.1191,84.1407]})
plt.figure(figsize=(10,5))
plt.title('accuracy comparison post cross validation')
sns.barplot(x='model_name', y='accuracy(%)', data=sc, palette='Blues_d')
print(sc)
# non convergence warnings will be received after logistic regressiona and cross-validation
# this is mainly due to the complexity of the predictive function due to the presence of 20+ variables
# this will be rectified after feature selection
# selecting best features
# results should be consistent with bivariate analysis

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif

X1=data.drop(['pla_win','tourney_level','player','opponent','round'],axis=1)


bestfeatures=SelectKBest(score_func=f_classif,k=8)
fit=bestfeatures.fit_transform(X1,y)
cols=X1.columns.values[bestfeatures.get_support()]
print(cols)
# splitting data from selected features
X2=data.loc[:,['pla_rankedHigher','pla_moreRankPoints','pla_moreAces',\
               'pla_lessDF','pla_1stWonMore','pla_2ndWonMore',\
               'pla_bpSavedMore','pla_bpFacedMore']]

X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y,test_size=0.3,\
                                                   random_state=1)
# applying logistic regression with feature selection

logmodel2=LogisticRegression()
logmodel2.fit(X2_train,y2_train)
print(f'{logmodel2}\n')

# predictions
logr2_pred=logmodel2.predict(X2_test)
logr2=accuracy_score(y2_test,logr2_pred)*100
print(f"Accuracy from logistic regression: {logr2} %")
print(f"F1 score: {f1_score(y2_test,logr2_pred)}\n")

from sklearn.metrics import confusion_matrix
conm2=confusion_matrix(y2_test,logr2_pred)
print(f"Confusion Matrix: -\n{conm2}")

acc2,pre2,rec2=((conm2[0,0]+conm2[1,1])/conm2.sum())*100,conm2[0,0]/(conm2[0,0]+conm2[0,1]),\
conm2[0,0]/(conm2[0,0]+conm2[1,0])
print(f"\nFrom confusion matrix: -\n1. Accuracy = {acc2} %\n\
2. Precision = {pre2}\n\
3. Recall/Sensitivity = {rec2}\n\
4. Specificity = {conm2[1,1]/(conm2[1,1]+conm2[0,1])}")
# decision tree with feature selection
# function to perform training using entropy
classifier2_entropy=DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=3)
#create the model
print(classifier2_entropy.fit(X2_train,y2_train),"\n")

# make predictions
dt2_pred=classifier2_entropy.predict(X2_test)
dt2=accuracy_score(y2_test,dt2_pred)*100
print(f"Accuracy from decision tree: {dt2} %")
print(f"F1 score: {f1_score(y2_test,dt2_pred)}\n")

cond2=confusion_matrix(y2_test,dt2_pred)
print(f"Confusion Matrix:-\n{cond2}")
accd2,pred2,recd2=((cond2[0,0]+cond2[1,1])/cond2.sum())*100,cond2[0,0]/\
(cond2[0,0]+cond2[0,1]),\
cond2[0,0]/(cond2[0,0]+cond2[1,0])
print(f"\nFrom confusion matrix: -\n1. Accuracy = {accd2} %\n\
2. Precision = {pred2}\n\
3. Recall/Sensitivity = {recd2}\n\
4. Specificity = {cond2[1,1]/(cond2[1,1]+cond2[0,1])}")
#knn with fature selection

n2_neighbors=int(math.sqrt(len(X2_test)))
if (n2_neighbors%2)==0:
    n2_neighbors+=1
    
classifier=KNeighborsClassifier(n2_neighbors,metric='euclidean')
print(f"{classifier.fit(X2_train,y2_train)}\n")
#knn with fature selection continued

knn2_pred=classifier.predict(X2_test)
cm2=confusion_matrix(y2_test,knn2_pred)
knn2=accuracy_score(y2_test,knn2_pred)*100
print(f"Accuracy from k nearest neighbours: {knn2} %\n")
print(f"F1 score: {f1_score(y2_test,knn2_pred)}\n")
print(f"Confusion Matrix: _\n{cm2}")
acck2,prek2,reck2=((cm2[0,0]+cm2[1,1])/cm2.sum())*100,cm2[0,0]/(cm2[0,0]+\
                                                                cm2[0,1]),\
cm2[0,0]/(cm2[0,0]+cm2[1,0])
print(f"\nFrom confusion matrix: -\n1. Accuracy = {acck2} %\n\
2. Precision = {prek2}\n\
3. Recall/Sensitivity = {reck2}\n\
4. Specificity = {cm2[1,1]/(cm2[1,1]+cm2[0,1])}")
# naive bayes with feature selection

gnb2=GaussianNB()
print(f"{gnb2.fit(X2_train,y2_train)}\n")

# prediction
gnb2_pred=gnb2.predict(X2_test)
gnba2=accuracy_score(y2_test,gnb2_pred)*100
print(f"Accuracy from gaussian naive bayes: {gnba2} %")
print(f"F1 score: {f1_score(y2_test,gnb2_pred)}\n")

cmn2=confusion_matrix(y2_test,gnb2_pred)
print(f"Confusion Matrix: -\n{cmn2}")
accn2,pren2,recn2=((cmn2[0,0]+cmn2[1,1])/cmn2.sum())*100,cmn2[0,0]/(cmn2[0,0]+cmn2[0,1]),\
cmn2[0,0]/(cmn2[0,0]+cmn2[1,0])
print(f"\nFrom confusion matrix: -\n1. Accuracy = {accn2} %\n\
2. Precision = {pren2}\n\
3. Recall/Sensitivity = {recn2}\n\
4. Specificity = {cmn2[1,1]/(cmn2[1,1]+cmn2[0,1])}")
# we are not going to make a second model for random forest classifier 
#using X2, beacause we selected the best 8 features in the first one
# accuracies post feature selection

sc=pd.DataFrame({"model_name":['logistic regression','decision tree',\
                               'kNN','naive bayes','random forest classifier'],\
                 "accuracy(%)":[logr2,dt2,knn2,gnba2,ra]})
plt.figure(figsize=(10,5))
plt.title('accuracy comparison post feature selection')
sns.barplot(x='model_name', y='accuracy(%)', data=sc, palette='Blues_d')
# cross-validation with feature selection

models2=[]
models2.append(('LR2', LogisticRegression()))
models2.append(('KNN2', KNeighborsClassifier()))
models2.append(('DECISION TREE2', DecisionTreeClassifier()))
models2.append(('NB2', GaussianNB()))
models2.append(('RFC2', RandomForestClassifier()))

# evaluate each model in turn
results2=[]
names2=[]
scoring='accuracy'
for name, model in models2:
    kfold=model_selection.KFold(n_splits=n2_neighbors)
    cv_results2 = model_selection.cross_val_score(model, X2, y, cv=kfold, scoring=scoring)
    results2.append(cv_results2)
    names2.append(name)
    msg2="%s: %f" % (name,cv_results2.mean())
    print(msg2)
#boxplot algorithm comparison    
fig=plt.figure()
fig.suptitle('Algorithm comparison')
ax2=fig.add_subplot(111)
plt.boxplot(results2)
ax2.set_xticklabels(names2)
plt.show()
# accuracies post feature selection and cross-validation

sc=pd.DataFrame({"model_name":['logistic regression','decision tree',\
                               'kNN','naive bayes','random forest classifier'],\
                 "accuracy(%)":[87.3087,87.2975,68.5926,85.1361,87.2947]})
plt.figure(figsize=(10,5))
plt.title('accuracy comparison post cross validation and feature selection')
sns.barplot(x='model_name', y='accuracy(%)', data=sc, palette='Blues_d')
print(sc)
# Logistic regression with cross validation gives the highest accuracy
# Accuracy (87.34%)
# But non-convergence warnings are received with this

# LOGISTIC REGRESSION WITH FEATURE SELECTION AND CROSS VALIDATION IS THE BEST MODEL TO USE
# Accuracy (87.31%)
# Warnigs removed, overfitting corrected, high accuracy