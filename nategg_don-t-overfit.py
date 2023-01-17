import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.gridspec as gs

import numpy as np
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print("Dataframe")

print(train.head(2))

print("----------------------------------------------------")

print("Info")

print(train.info())

print(test.info())

print("----------------------------------------------------")

print("Label Count")

print(train["target"].value_counts())

print("----------------------------------------------------")

print("Missing Data")

print(train.isnull().any().any())
train_details = train.describe()

test_details = test.describe()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

new_feature1 = pd.DataFrame(scaler.fit_transform(train.iloc[:,2:]))

test_feature1 = pd.DataFrame(scaler.transform(test.iloc[:,1:]))

new_feature = new_feature1.describe()

test_feature = test_feature1.describe()







fig = plt.figure(figsize=(18,6))

g = gs.GridSpec(1,1,fig)

ax = fig.add_subplot(g[0,0])









ax.hist(train_details.loc["mean","0":], alpha = 0.5,color="magenta", bins=80, label='train')

ax.hist(test_details.loc["mean","0":],color="darkblue",bins=80, label='test')

ax.text(-0.2, 40, r"train $\mu={0:3.2f}$".format(train_details.loc["mean","0":].mean()))

ax.text(-0.2, 35, r"train $\sigma={0:3.2f}$".format(train_details.loc["std","0":].mean()))

ax.text(-0.2, 30, r"test $\mu={0:3.2f}$".format(test_details.loc["mean","0":].mean()))

ax.text(-0.2, 25, r"test $\sigma={0:3.2f}$".format(test_details.loc["std","0":].mean()))



ax.legend(loc=0)

sns.kdeplot(train_details.loc["mean","0":],color="magenta" , ax=ax, legend =False)

sns.kdeplot(test_details.loc["mean","0":],color="darkblue" , ax=ax, legend =False)

ax.set_title("Mean for Testing/ Training set")

from sklearn.linear_model import LogisticRegression



data =pd.concat([ test_feature1 , new_feature1.iloc[:,:]])



lr = LogisticRegression(solver='liblinear')



lr.fit(new_feature1.iloc[:,:], train.iloc[:,1])

coeff = pd.DataFrame(lr.coef_)

mean = coeff.T.mean()

std = coeff.T.std()

coeff_50 = coeff.T.sort_values(by=0,ascending = False).iloc[:50,:]

col = coeff_50.index 

data = data[col]

data = data.describe()



df_corr = new_feature1.iloc[:,:].apply(lambda x: x.corr(train.iloc[:,1]))

df_corr = df_corr.reset_index().sort_values(by=0,ascending = False).iloc[:50,:]











fig = plt.figure(figsize=(25,25))

g = gs.GridSpec(4,1,fig)

ax = fig.add_subplot(g[0,0])

ax2 = fig.add_subplot(g[1,0])

ax3 = fig.add_subplot(g[2,0])

ax4 = fig.add_subplot(g[3,0])



ax.set_title("Coefficient Of all Features")

ax.plot(coeff.T,'ro' )

ax.text(0, 0.75, r"$\mu={0:3.2f}$".format(mean[0]))

ax.text(0, 0.60, r"$\sigma={0:3.2f}$".format(std[0]))



ax2.set_title("Coefficient Of top 50 Features")

coeff_50.plot.bar(ax=ax2)



ax3.set_title("Mean of Top 50 Columns for both Dataset")

data.loc["mean",:].plot.bar(ax=ax3)



ax4.set_title("Correlation Of top 50 Features")

df_corr.iloc[:,1].plot.bar(ax=ax4,color ="Blue")



plt.figure(figsize=(16,6))



plt.title("Distribution of min values per row in the train and test set")

sns.distplot(new_feature1.min(axis=1),color="orange", kde=True,bins=120, label='train')

sns.distplot(test_feature1.min(axis=1),color="red", kde=True,bins=120, label='test')

plt.legend()

plt.show()



plt.figure(figsize=(16,6))



plt.title("Distribution of Mean values per row in the train and test set")

sns.distplot(new_feature1.mean(axis=1),color="magenta", kde=True,bins=120, label='train')

sns.distplot(test_feature1.mean(axis=1),color="darkblue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))



plt.title("Distribution of std values per row in the train and test set")

sns.distplot(new_feature1.std(axis=1),color="orange", kde=True,bins=120, label='train')

sns.distplot(test_feature1.std(axis=1),color="red", kde=True,bins=120, label='test')

plt.legend()

plt.show()
features_90 = train[train["target"] == 1]

features_902 = train[train["target"] == 0]

features_90 = features_90.iloc[:90,:]

features_90_final = pd.concat([features_90, features_902])

features_90_final = pd.DataFrame(scaler.fit_transform(features_90_final.iloc[:,2:]))





plt.figure(figsize=(16,6))



plt.title("Distribution of Equal Dataset compare to Test Data")

sns.distplot(features_90_final.mean(axis=1),color="red", kde=True,bins=120, label='train')

sns.distplot(test_feature1.mean(axis=1),color="orange", kde=True,bins=120, label='test')



plt.legend()

plt.show()



def scores(X ,y, model):

    score = []

    score2 = []

    results = cross_val_score(model, X, y, cv = 3, scoring = 'roc_auc')

    results2 = cross_val_score(model, X, y, cv = 3, scoring = 'accuracy')

    score.append(results)

    score2.append(results2)

    score_df = pd.DataFrame(score).T

    score_df.loc['mean'] = score_df.mean()

    score_df.loc['std'] = score_df.std()

    score_df= score_df.rename(columns={0:'roc_auc'})

    print(score_df.iloc[-2:,:])

    score_df2 = pd.DataFrame(score2).T

    score_df2.loc['mean'] = score_df2.mean()

    score_df2.loc['std'] = score_df2.std()

    score_df2= score_df2.rename(columns={0:'acc'})

    print(score_df2.iloc[-2:,:])
from sklearn.model_selection import GridSearchCV ,cross_val_score

clf = LogisticRegression(max_iter=4000)





param_grid = [

  {'class_weight' : ['balanced', None],'penalty': ['l1'], 'solver': ['liblinear', 'saga'],'C' : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},

  {'class_weight' : ['balanced', None],'penalty': ['l2'], 'solver': ['newton-cg','lbfgs'],'C' : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},

 {'class_weight' : ['balanced', None],'penalty': ['l2','l1'], 'solver': ['saga'],'C' : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},

 ]



grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10, scoring='roc_auc')

grid_search.fit(new_feature1, train.iloc[:,1])

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
new_features_corr = new_feature1.corr()

sns.heatmap(new_features_corr)
model = LogisticRegression(C=0.1,random_state=42,class_weight='balanced',penalty='l1',solver='liblinear')

from sklearn.feature_selection import RFE



rfr_grid =  [5,10,15,25, 50, 75, 100, 125, 150, 175, 200 ,225,250,275,300]



for feature_num in rfr_grid :

    selector = RFE(model, feature_num, step=1)

    selector = selector.fit(new_feature1, train.iloc[:,1])

    new_features =  selector.transform(new_feature1)

    print("Number of Features {}".format(feature_num))

    scores(new_feature1,train.iloc[:,1],selector)
selector = RFE(model,25, step=1)

selector = selector.fit(new_feature1, train.iloc[:,1])

new_features =  selector.transform(new_feature1)

bool_mask = pd.Series(selector.support_,name='bools') 

scores(new_feature1,train.iloc[:,1],selector)

col = []

for num, i  in enumerate(bool_mask):

    if i ==True :

        col.append(num)



rfe_feature = new_feature1[col]

rfe_test = test_feature1[col]

print("The columns that were used were {}".format(col))
plt.figure(figsize=(16,6))



plt.title("Distribution of mean values per row in the train and test set")

sns.distplot(rfe_feature.mean(axis=1),color="red", kde=True,bins=120, label='train')

sns.distplot(rfe_test.mean(axis=1),color="orange", kde=True,bins=120, label='test')

plt.legend()

plt.show()
rfe_feature["sum"] = rfe_feature.sum(axis=1)

rfe_feature["min"] = rfe_feature.min(axis=1)

rfe_feature["max"] = rfe_feature.max(axis=1)

rfe_feature["mean"] = rfe_feature.mean(axis=1)

rfe_feature["std"] = rfe_feature.std(axis=1)

rfe_feature["skew"] = rfe_feature.skew(axis=1)

rfe_feature["var"] = rfe_feature["std"]**2



rfe_feature["target"] = train.iloc[:,1]







rfe_test["sum"] = rfe_test.sum(axis=1)

rfe_test["min"] = rfe_test.min(axis=1)

rfe_test["max"] = rfe_test.max(axis=1)

rfe_test["mean"] = rfe_test.mean(axis=1)

rfe_test["std"] = rfe_test.std(axis=1)

rfe_test["skew"] = rfe_test.skew(axis=1)

rfe_test["var"] = rfe_test["std"]**2





rfe_feature_true = rfe_feature[rfe_feature["target"] == 1]

rfe_feature_false = rfe_feature[rfe_feature["target"] == 0]



fig = plt.figure(figsize=(21,35))

g = gs.GridSpec(7,1,fig)

ax = fig.add_subplot(g[0,0])

ax1 = fig.add_subplot(g[1,0])

ax2 = fig.add_subplot(g[2,0])

ax3 = fig.add_subplot(g[3,0])

ax4= fig.add_subplot(g[4,0])

ax5 = fig.add_subplot(g[5,0])

ax6 = fig.add_subplot(g[6,0])





ax.set_title("SUM Distribuion")

sns.distplot(rfe_feature_true["sum"],color="red", kde=True,bins=120, label='True' ,ax=ax)

sns.distplot(rfe_feature_false["sum"],color="orange", kde=True,bins=120, label='False',ax=ax)



ax1.set_title("MIN Distribuion")

sns.distplot(rfe_feature_true["min"],color="red", kde=True,bins=120, label='True' ,ax=ax1)

sns.distplot(rfe_feature_false["min"],color="orange", kde=True,bins=120, label='False',ax=ax1)



ax2.set_title("MAX Distribuion")

sns.distplot(rfe_feature_true["max"],color="red", kde=True,bins=120, label='True' ,ax=ax2)

sns.distplot(rfe_feature_false["max"],color="orange", kde=True,bins=120, label='False',ax=ax2)



ax3.set_title("MEAN Distribuion")

sns.distplot(rfe_feature_true["mean"],color="red", kde=True,bins=120, label='True' ,ax=ax3)

sns.distplot(rfe_feature_false["mean"],color="orange", kde=True,bins=120, label='False',ax=ax3)



ax4.set_title("STD Distribuion")

sns.distplot(rfe_feature_true["std"],color="red", kde=True,bins=120, label='True' ,ax=ax4)

sns.distplot(rfe_feature_false["std"],color="orange", kde=True,bins=120, label='False',ax=ax4)



ax5.set_title("SKEW Distribuion")

sns.distplot(rfe_feature_true["skew"],color="red", kde=True,bins=120, label='True' ,ax=ax5)

sns.distplot(rfe_feature_false["skew"],color="orange", kde=True,bins=120, label='False',ax=ax5)



ax6.set_title("VAR Distribuion")

sns.distplot(rfe_feature_true["var"],color="red", kde=True,bins=120, label='True' ,ax=ax6)

sns.distplot(rfe_feature_false["var"],color="orange", kde=True,bins=120, label='False',ax=ax6)



ax.legend(loc="best")

rfe_feature = rfe_feature.drop(["target"],axis=1)
model = LogisticRegression(C=0.1,random_state=42,class_weight='balanced',penalty='l1',solver='liblinear',tol=0.02,verbose=0)

rfr_grid =  [23,24,25,26,27,28,29,30,31,32]



for feature_num in rfr_grid :

    selector = RFE(model, feature_num, step=1)

    selector = selector.fit(rfe_feature, train.iloc[:,1])

    new_features =  selector.transform(rfe_feature)

    print("Number of Features {}".format(feature_num))

    scores(new_features,train.iloc[:,1],selector)
selector = RFE(model,23, step=1)

selector = selector.fit(rfe_feature, train.iloc[:,1])

new_features =  selector.transform(rfe_feature)

new_test = selector.transform(rfe_test)



bool_mask = pd.Series(selector.support_,name='bools') 

scores(new_features,train.iloc[:,1],selector)

old_cols = [16, 33, 43, 63, 65, 73, 80, 82, 90, 91, 101, 108, 117, 127, 133, 134, 165, 189, 194, 199, 217, 226, 258, 295, 298,"sum","min","max","mean","std","skew","var"]

col = []

for num, i  in enumerate(bool_mask):

 

    if i ==True :

        col.append(old_cols[num])

        

        





print("The columns that were used were {}".format(col))
model = LogisticRegression(C=0.1,random_state=42,class_weight='balanced',penalty='l1',solver='liblinear')

model = model.fit(new_features, train.iloc[:,1])

prediction5 = model.predict_proba(new_test)



test["target"]= prediction5[:,1]

submission = test[["id","target"]]

submission.to_csv("submission14.csv",index=False)
rfe_feature2  =rfe_feature.drop(["std","var"], axis = 1)

rfe_test2  =rfe_test.drop(["std","var"], axis = 1)


model = LogisticRegression(C=0.1,random_state=42,class_weight='balanced',penalty='l1',solver='liblinear')

rfr_grid =  [23,24,25,26,27,28,29,30,]



for feature_num in rfr_grid :

    selector = RFE(model, feature_num, step=1)

    selector = selector.fit(rfe_feature2, train.iloc[:,1])

    new_features =  selector.transform(rfe_feature2)

    print("Number of Features {}".format(feature_num))

    scores(new_features,train.iloc[:,1],selector)
selector = RFE(model,23, step=1)

selector = selector.fit(rfe_feature2, train.iloc[:,1])

new_features =  selector.transform(rfe_feature2)

new_test = selector.transform(rfe_test2)

bool_mask = pd.Series(selector.support_,name='bools') 

scores(new_features,train.iloc[:,1],selector)

old_cols = [16, 33, 43, 63, 65, 73, 80, 82, 90, 91, 101, 108, 117, 127, 133, 134, 165, 189, 194, 199, 217, 226, 258, 295, 298,"sum","min","max","mean","skew"]

col = []

for num, i  in enumerate(bool_mask):

    if i ==True :

        col.append(old_cols[num])



print("The columns that were used were {}".format(col))
model = LogisticRegression(C=0.1,random_state=42,class_weight='balanced',penalty='l1',solver='liblinear',tol=0.02,verbose=0)

model = model.fit(rfe_feature2, train.iloc[:,1])

prediction5 = model.predict_proba(rfe_test2)



test["target"]= prediction5[:,1]

submission = test[["id","target"]]

submission.to_csv("submission15.csv",index=False)
rfe_feature3  =rfe_feature.drop(["std","var","skew","min","max"], axis = 1)

rfe_test3  = rfe_test.drop(["std","var","skew","min","max"], axis = 1)
model = LogisticRegression(C=0.1,random_state=42,class_weight='balanced',penalty='l1',solver='liblinear',tol=0.02,verbose=0)

rfr_grid =  [23,24,25,26,27]



for feature_num in rfr_grid :

    selector = RFE(model, feature_num, step=1)

    selector = selector.fit(rfe_feature3, train.iloc[:,1])

    new_features =  selector.transform(rfe_feature3)

    print("Number of Features {}".format(feature_num))

    scores(new_features,train.iloc[:,1],selector)
selector = RFE(model,26, step=1)

selector = selector.fit(rfe_feature3, train.iloc[:,1])

new_features =  selector.transform(rfe_feature3)

new_test = selector.transform(rfe_test3)

bool_mask = pd.Series(selector.support_,name='bools') 

scores(new_features,train.iloc[:,1],selector)

old_cols = [16, 33, 43, 63, 65, 73, 80, 82, 90, 91, 101, 108, 117, 127, 133, 134, 165, 189, 194, 199, 217, 226, 258, 295, 298,"sum","mean"]

col = []

for num, i  in enumerate(bool_mask):

    if i ==True :

        col.append(old_cols[num])



print("The columns that were used were {}".format(col))
model = LogisticRegression(C=0.1,random_state=42,class_weight='balanced',penalty='l1',solver='liblinear')

model = model.fit(rfe_feature3, train.iloc[:,1])

prediction5 = model.predict_proba(rfe_test3)



test["target"]= prediction5[:,1]

submission = test[["id","target"]]

submission.to_csv("submission16.csv",index=False)
from sklearn.ensemble import BaggingClassifier, VotingClassifier

from sklearn.naive_bayes import GaussianNB 

from sklearn import linear_model

from sklearn.model_selection import ParameterGrid

from sklearn.linear_model import ElasticNet



model_1= linear_model.SGDClassifier('log',eta0=1, max_iter=1000, tol=0.0001)

model_2 = GaussianNB()

model_3 = LogisticRegression( C = 0.1, class_weight ='balanced', penalty = 'l1', solver = 'liblinear')

model_4 =  ElasticNet(alpha = 0.085, l1_ratio = 0.5)

grid  = [1,3,6,9,12,15,18,20,22,24,28]





                          

for params in grid:

    blg = BaggingClassifier(base_estimator=model_3,n_estimators = params).fit(rfe_feature3,train.iloc[:,1])

    scores(rfe_feature3,train.iloc[:,1],blg)





blg = BaggingClassifier(base_estimator=model_3,n_estimators = 24).fit(rfe_feature3,train.iloc[:,1])

prediction5 = blg.predict_proba(rfe_test3)





test["target"]= prediction5[:,1]

submission = test[["id","target"]]

submission.to_csv("submission17.csv",index=False)
eclf = VotingClassifier(estimators=[('NB', model_2), ('LR', model_3), ('LIN', model_1)],

                       voting='soft', weights=[1,2, 2])



eclf.fit(rfe_feature3,train.iloc[:,1])

scores(rfe_feature3,train.iloc[:,1],eclf)



prediction5 = eclf.predict_proba(rfe_test3)





test["target"]= prediction5[:,1]

submission = test[["id","target"]]

submission.to_csv("submission20.csv",index=False)
