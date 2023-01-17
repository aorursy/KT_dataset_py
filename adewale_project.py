import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


import os
print(os.listdir("../input"))


%matplotlib inline
# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head() # Familiarisation with dataset (prints 1st 5 rows)
train.isnull().sum()
train.describe()
train.shape
train.columns

fig, ax= plt.subplots(figsize=(20,20))
train.hist(ax=ax);
# I suspect the data is artificial because all the features are almost normally distributed 
#Correclation between the features, .CORR RETURNS THE CORRELATION MATRIX
fig, ax= plt.subplots(figsize=(15,10))
#plt.rcParams['figure.figsize']= (10,6) 
sns.heatmap(train.corr(),ax=ax);
train.target.value_counts()
sns.countplot(x='target', data=train);
from sklearn.preprocessing import PolynomialFeatures

train= train[train.era == "era1"]
X = train.drop(['id','era','data_type','target'],axis=1) # We are dropping all the non-numerical columns in our dataset
X2= train.drop(['id','era','data_type','target'],axis=1) # Set for polynomial training

poly = PolynomialFeatures(2)
X2 = poly.fit_transform(X2)


y = train.target
X.shape,y.shape
        # We are using 50 features to predict y
# Now we will split dataset into training set and test set using sklearn library
from sklearn.model_selection import train_test_split 

# Test set is 30% of the whole size 
X_train, X_test,y_train, y_test=train_test_split(X,y, test_size= 0.3, random_state=17)  
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# We will now use X_train, y_train for cross validation  and X_test and y_test to verify if the model works 
from sklearn.tree import DecisionTreeClassifier

# Definition of Model 
tree = DecisionTreeClassifier(random_state=17) # Initialisation of Tree

tree.fit(X_train,y_train) # Fits tree with our data (Fit-predict) : This is where training process occurs
from sklearn.model_selection import cross_val_score , StratifiedKFold
from tqdm import tqdm_notebook #Counts number of executions (widget that visualizes iteration)

skf=StratifiedKFold(n_splits=5, shuffle=True,random_state=17) 

from sklearn.metrics import accuracy_score

cv_accuracies_by_depth, ts_accuracies_by_depth= [],[]

max_depth_values = np.arange(10,30)
# for each value of max depth 
for max_depth in tqdm_notebook(max_depth_values):
    tree= DecisionTreeClassifier(random_state=17, max_depth = max_depth)
    
    #perform cross validation
    val_scores = cross_val_score(estimator=tree, X=X_train, y=y_train, cv= skf)
    cv_accuracies_by_depth.append(val_scores.mean())
    
    # evaluate model on the test set
    tree.fit(X_train,y_train) # fits tree to training set 
    curr_ts_pred = tree.predict(X_test)
    
    ts_accuracies_by_depth.append(accuracy_score(y_test,curr_ts_pred)) 
plt.plot(max_depth_values,cv_accuracies_by_depth, label='CV');
plt.plot(max_depth_values,ts_accuracies_by_depth, label= 'Test');
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Validation Curve');
# Check accuracy with the remaining part of data not trained
from sklearn.metrics import accuracy_score

pred_test = tree.predict_proba(X_test)[:,1] 
# Predicting our target function (i.e y) with the tree constructed from our train set
pred_test.shape, y_test.shape # We obtain a vector with the same shape as our test target 
# Now we can compare the two to see how accurate our model worked 
accuracy_score(y_test,pred_test)
# In order to get a sense of whether our accuracy score is good or bad 
# we try and see the distribution of our target function

y.value_counts(normalize=True)
from sklearn.model_selection import GridSearchCV  , StratifiedKFold
# GridSearchCV finds all the possible combinations of parameters e.g max_depth and min_sample_leaf (Randomized could also be used)
# Then we do cross validation and select the combination that works best
#Definition of our hyper_parameters 
params ={'max_depth':np.arange(2,13),'min_samples_leaf':np.arange(2,13)}

# Way of defining cross validation
skf=StratifiedKFold(n_splits=5, shuffle=True,random_state=17) 

# Getting optimal tree with optimal parameters
best_tree= GridSearchCV(estimator=tree,param_grid=params, cv = skf, n_jobs=1, verbose =1)


best_tree.fit(X_train,y_train);
best_tree.best_params_
best_tree.best_estimator_ # best parameters for tree
#Cross Validation assessment of model quality

best_tree.best_score_ # Accuracy score using best parameters
# Now we will predict with the tuned parameters
pred_test_better = best_tree.predict(X_test)
pred1= best_tree.predict_proba(X_test)[:,1]
pred2 = best_tree.predict_proba(X_train)[:,1]



accuracy_score(y_test,pred_test_better) # accuracy improved compared to before 
from sklearn.tree import export_graphviz
export_graphviz(decision_tree=best_tree.best_estimator_,out_file='tree.dot',filled=True,
                feature_names=train.drop(['id','era','data_type','target'],axis=1).columns )
!cat tree.dot;  # to get the code of the file so that it could be viewed online 
# Exclamation mark tells python I want to execute shell command  
from IPython.display import Image
from IPython.core.display import HTML 

Image(url= "https://chart.googleapis.com/chart?chl=+digraph+Tree+%7B%0D%0Anode+%5Bshape%3Dbox%2C+style%3D%22filled%22%2C+color%3D%22black%22%5D+%3B%0D%0A0+%5Blabel%3D%22feature32+%3C%3D+0.48%5Cngini+%3D+0.5%5Cnsamples+%3D+1153%5Cnvalue+%3D+%5B573%2C+580%5D%22%2C+fillcolor%3D%22%23399de503%22%5D+%3B%0D%0A1+%5Blabel%3D%22feature39+%3C%3D+0.483%5Cngini+%3D+0.487%5Cnsamples+%3D+559%5Cnvalue+%3D+%5B235%2C+324%5D%22%2C+fillcolor%3D%22%23399de546%22%5D+%3B%0D%0A0+-%3E+1+%5Blabeldistance%3D2.5%2C+labelangle%3D45%2C+headlabel%3D%22True%22%5D+%3B%0D%0A2+%5Blabel%3D%22feature7+%3C%3D+0.706%5Cngini+%3D+0.448%5Cnsamples+%3D+274%5Cnvalue+%3D+%5B93%2C+181%5D%22%2C+fillcolor%3D%22%23399de57c%22%5D+%3B%0D%0A1+-%3E+2+%3B%0D%0A3+%5Blabel%3D%22feature42+%3C%3D+0.466%5Cngini+%3D+0.435%5Cnsamples+%3D+263%5Cnvalue+%3D+%5B84%2C+179%5D%22%2C+fillcolor%3D%22%23399de587%22%5D+%3B%0D%0A2+-%3E+3+%3B%0D%0A4+%5Blabel%3D%22gini+%3D+0.494%5Cnsamples+%3D+103%5Cnvalue+%3D+%5B46%2C+57%5D%22%2C+fillcolor%3D%22%23399de531%22%5D+%3B%0D%0A3+-%3E+4+%3B%0D%0A5+%5Blabel%3D%22gini+%3D+0.362%5Cnsamples+%3D+160%5Cnvalue+%3D+%5B38%2C+122%5D%22%2C+fillcolor%3D%22%23399de5b0%22%5D+%3B%0D%0A3+-%3E+5+%3B%0D%0A6+%5Blabel%3D%22feature2+%3C%3D+0.466%5Cngini+%3D+0.298%5Cnsamples+%3D+11%5Cnvalue+%3D+%5B9%2C+2%5D%22%2C+fillcolor%3D%22%23e58139c6%22%5D+%3B%0D%0A2+-%3E+6+%3B%0D%0A7+%5Blabel%3D%22gini+%3D+0.0%5Cnsamples+%3D+7%5Cnvalue+%3D+%5B7%2C+0%5D%22%2C+fillcolor%3D%22%23e58139ff%22%5D+%3B%0D%0A6+-%3E+7+%3B%0D%0A8+%5Blabel%3D%22gini+%3D+0.5%5Cnsamples+%3D+4%5Cnvalue+%3D+%5B2%2C+2%5D%22%2C+fillcolor%3D%22%23e5813900%22%5D+%3B%0D%0A6+-%3E+8+%3B%0D%0A9+%5Blabel%3D%22feature23+%3C%3D+0.6%5Cngini+%3D+0.5%5Cnsamples+%3D+285%5Cnvalue+%3D+%5B142%2C+143%5D%22%2C+fillcolor%3D%22%23399de502%22%5D+%3B%0D%0A1+-%3E+9+%3B%0D%0A10+%5Blabel%3D%22feature9+%3C%3D+0.559%5Cngini+%3D+0.487%5Cnsamples+%3D+198%5Cnvalue+%3D+%5B115%2C+83%5D%22%2C+fillcolor%3D%22%23e5813947%22%5D+%3B%0D%0A9+-%3E+10+%3B%0D%0A11+%5Blabel%3D%22gini+%3D+0.449%5Cnsamples+%3D+135%5Cnvalue+%3D+%5B89%2C+46%5D%22%2C+fillcolor%3D%22%23e581397b%22%5D+%3B%0D%0A10+-%3E+11+%3B%0D%0A12+%5Blabel%3D%22gini+%3D+0.485%5Cnsamples+%3D+63%5Cnvalue+%3D+%5B26%2C+37%5D%22%2C+fillcolor%3D%22%23399de54c%22%5D+%3B%0D%0A10+-%3E+12+%3B%0D%0A13+%5Blabel%3D%22feature27+%3C%3D+0.773%5Cngini+%3D+0.428%5Cnsamples+%3D+87%5Cnvalue+%3D+%5B27%2C+60%5D%22%2C+fillcolor%3D%22%23399de58c%22%5D+%3B%0D%0A9+-%3E+13+%3B%0D%0A14+%5Blabel%3D%22gini+%3D+0.378%5Cnsamples+%3D+79%5Cnvalue+%3D+%5B20%2C+59%5D%22%2C+fillcolor%3D%22%23399de5a9%22%5D+%3B%0D%0A13+-%3E+14+%3B%0D%0A15+%5Blabel%3D%22gini+%3D+0.219%5Cnsamples+%3D+8%5Cnvalue+%3D+%5B7%2C+1%5D%22%2C+fillcolor%3D%22%23e58139db%22%5D+%3B%0D%0A13+-%3E+15+%3B%0D%0A16+%5Blabel%3D%22feature15+%3C%3D+0.56%5Cngini+%3D+0.49%5Cnsamples+%3D+594%5Cnvalue+%3D+%5B338%2C+256%5D%22%2C+fillcolor%3D%22%23e581393e%22%5D+%3B%0D%0A0+-%3E+16+%5Blabeldistance%3D2.5%2C+labelangle%3D-45%2C+headlabel%3D%22False%22%5D+%3B%0D%0A17+%5Blabel%3D%22feature10+%3C%3D+0.401%5Cngini+%3D+0.5%5Cnsamples+%3D+364%5Cnvalue+%3D+%5B178%2C+186%5D%22%2C+fillcolor%3D%22%23399de50b%22%5D+%3B%0D%0A16+-%3E+17+%3B%0D%0A18+%5Blabel%3D%22feature4+%3C%3D+0.541%5Cngini+%3D+0.461%5Cnsamples+%3D+75%5Cnvalue+%3D+%5B48%2C+27%5D%22%2C+fillcolor%3D%22%23e5813970%22%5D+%3B%0D%0A17+-%3E+18+%3B%0D%0A19+%5Blabel%3D%22gini+%3D+0.388%5Cnsamples+%3D+57%5Cnvalue+%3D+%5B42%2C+15%5D%22%2C+fillcolor%3D%22%23e58139a4%22%5D+%3B%0D%0A18+-%3E+19+%3B%0D%0A20+%5Blabel%3D%22gini+%3D+0.444%5Cnsamples+%3D+18%5Cnvalue+%3D+%5B6%2C+12%5D%22%2C+fillcolor%3D%22%23399de57f%22%5D+%3B%0D%0A18+-%3E+20+%3B%0D%0A21+%5Blabel%3D%22feature7+%3C%3D+0.524%5Cngini+%3D+0.495%5Cnsamples+%3D+289%5Cnvalue+%3D+%5B130%2C+159%5D%22%2C+fillcolor%3D%22%23399de52f%22%5D+%3B%0D%0A17+-%3E+21+%3B%0D%0A22+%5Blabel%3D%22gini+%3D+0.499%5Cnsamples+%3D+191%5Cnvalue+%3D+%5B99%2C+92%5D%22%2C+fillcolor%3D%22%23e5813912%22%5D+%3B%0D%0A21+-%3E+22+%3B%0D%0A23+%5Blabel%3D%22gini+%3D+0.433%5Cnsamples+%3D+98%5Cnvalue+%3D+%5B31%2C+67%5D%22%2C+fillcolor%3D%22%23399de589%22%5D+%3B%0D%0A21+-%3E+23+%3B%0D%0A24+%5Blabel%3D%22feature40+%3C%3D+0.341%5Cngini+%3D+0.423%5Cnsamples+%3D+230%5Cnvalue+%3D+%5B160%2C+70%5D%22%2C+fillcolor%3D%22%23e581398f%22%5D+%3B%0D%0A16+-%3E+24+%3B%0D%0A25+%5Blabel%3D%22feature32+%3C%3D+0.595%5Cngini+%3D+0.415%5Cnsamples+%3D+17%5Cnvalue+%3D+%5B5%2C+12%5D%22%2C+fillcolor%3D%22%23399de595%22%5D+%3B%0D%0A24+-%3E+25+%3B%0D%0A26+%5Blabel%3D%22gini+%3D+0.26%5Cnsamples+%3D+13%5Cnvalue+%3D+%5B2%2C+11%5D%22%2C+fillcolor%3D%22%23399de5d1%22%5D+%3B%0D%0A25+-%3E+26+%3B%0D%0A27+%5Blabel%3D%22gini+%3D+0.375%5Cnsamples+%3D+4%5Cnvalue+%3D+%5B3%2C+1%5D%22%2C+fillcolor%3D%22%23e58139aa%22%5D+%3B%0D%0A25+-%3E+27+%3B%0D%0A28+%5Blabel%3D%22feature24+%3C%3D+0.274%5Cngini+%3D+0.396%5Cnsamples+%3D+213%5Cnvalue+%3D+%5B155%2C+58%5D%22%2C+fillcolor%3D%22%23e58139a0%22%5D+%3B%0D%0A24+-%3E+28+%3B%0D%0A29+%5Blabel%3D%22gini+%3D+0.245%5Cnsamples+%3D+7%5Cnvalue+%3D+%5B1%2C+6%5D%22%2C+fillcolor%3D%22%23399de5d4%22%5D+%3B%0D%0A28+-%3E+29+%3B%0D%0A30+%5Blabel%3D%22gini+%3D+0.377%5Cnsamples+%3D+206%5Cnvalue+%3D+%5B154%2C+52%5D%22%2C+fillcolor%3D%22%23e58139a9%22%5D+%3B%0D%0A28+-%3E+30+%3B%0D%0A%7D%0D%0A%0D%0A&cht=gv")
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=300, class_weight='balanced',n_jobs=4,random_state=17)


%%timeit
forest.fit(X_train,y_train)
forest_pred_test= forest.predict_proba(X_test)[:,1]
forest_pred_train=forest.predict_proba(X_train)[:,1]
y_train.shape, forest_pred_train.shape
#Metrics
from sklearn.metrics import  roc_auc_score

score_train_RFC= roc_auc_score(y_train, forest_pred_train)
score_test_RFC = roc_auc_score(y_test,forest_pred_test)

# score_test_DecTr= roc_auc_score(y_test,pred1)# for decision tree
# score_train_DecTr = roc_auc_score(y_train,pred2)# for decision tree

print('RandomForestClassifier ROC_AUC:\n Train_set: %f ; Test_set:  %f' % (score_train_RFC, score_test_RFC))
print('DecisionTreeClassifier ROC_AUC:\n Train_set: %f ; Test_set:  %f' % (score_train_RFC, score_test_DecTr))

#the score is not high enough , but randomforest performs slightly better than Decision treee
plt.hist(forest_pred_test);
# Feature importance 
feat_importance=pd.DataFrame(forest.feature_importances_,index=X.columns,
             columns=['importance']).sort_values(by='importance', ascending=False)
feat_importance.head()
# RidgeClassifier
from sklearn.linear_model import RidgeClassifier
clf_ridge = RidgeClassifier(random_state=17)
clf_ridge.fit(X_train, y_train)

score_train = clf_ridge.score(X_train, y_train) 
score_test=clf_ridge.score(X_test, y_test) 

print('RidgeClassifier Score: \n Train_set: %f ; Test_set:  %f' % (score_train, score_test))

clf_ridge.score(X_test, y_test)
#Logistic RegressionCV
from sklearn.linear_model import LogisticRegressionCV;
from sklearn.metrics import log_loss, roc_auc_score

clf_log_reg = LogisticRegressionCV(cv=5, random_state=17);
clf_log_reg.fit(X_train, y_train);

pred_train=clf_log_reg.predict_proba(X_train)[:,1];
pred_test= clf_log_reg.predict_proba(X_test)[:,1];

# Model evaluation with ROC_AUC Metric (The higher the better)
score_train_Roc = roc_auc_score(y_train, pred_train)
score_test_Roc = roc_auc_score(y_test, pred_test)

#Model validation with Log_loss Metric (The lower the better)
score_train_Log = log_loss(y_train, pred_train)
score_test_Log = log_loss(y_test, pred_test)

print('LogisticRegressionCV ROC_AUC:\n Train_set: %f ; Test_set:  %f' % (score_train_Roc, score_test_Roc))
print('LogisticRegressionCV Log_loss:\n Train_set: %f ; Test_set:  %f' % (score_train_Log, score_test_Log))
clf_log_reg.score(X_test, y_test);

# As expected, the scores are better on the training set
#KNN Classifier 
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train) 

pred_test_KN= neigh.predict_proba(X_test)[:,1]
pred_train_KN = neigh.predict_proba(X_train)[:,1]

# Model validation with ROC_AUC Metric (The higher the better)
score_train_Roc_KN = roc_auc_score(y_train, pred_train_KN)
score_test_Roc_KN = roc_auc_score(y_test, pred_test_KN)

#Model validation with Log_loss Metric (The lower the better)
score_train_Log_KN = log_loss(y_train, pred_train_KN)
score_test_Log_KN = log_loss(y_test, pred_test_KN)

print('KNeighborsClassifier ROC_AUC:\n Train_set: %f ; Test_set:  %f' % (score_train_Roc_KN, score_test_Roc_KN))
print('KNeighborsClassifier Log_loss:\n Train_set: %f ; Test_set:  %f' % (score_train_Log_KN, score_test_Log_KN))
clf_log_reg.score(X_test, y_test);


neigh.score(X_test,y_test)
from sklearn.model_selection import KFold

#Splits training set into 5 different parts
kf = KFold(n_splits=5)
nfold= 0
pred = pd.DataFrame()

for train_index, test_index in kf.split(X_train):
    nfold+=1
    print('nfold:', nfold)
    X_tr, X_ts = X_train.values[train_index], X_train.values[test_index]
    y_tr, y_ts = y_train.values[train_index], y_train.values[test_index]
    
    #Teaches dataset 
    lr1 = LogisticRegressionCV(cv=5, random_state=17)
    lr1.fit(X_tr,y_tr)
    
    neigh1 = KNeighborsClassifier(n_neighbors=3)
    neigh1.fit(X_tr, y_tr) 
    
    pred['pred'+ 'LogReg' + str(nfold)] = lr1.predict_proba(X_test)[:,1]
    pred['pred'+ 'KN'+ str(nfold)] = neigh1.predict_proba(X_test)[:,1]
    
    # Making our predictions with the taught models
    pred_test_Log = lr1.predict_proba(X_ts)[:,1]
    pred_train_Log = lr1.predict_proba(X_tr)[:,1]
    pred_test_KN = neigh1.predict_proba(X_ts)[:,1]
    pred_train_KN = neigh1.predict_proba(X_tr)[:,1]

    # Model evaluation with ROC_AUC metric
    score_test_LogReg = roc_auc_score(y_ts, pred_test_Log)
    score_train_LogReg = roc_auc_score(y_tr, pred_train_Log)
    score_test_KN = roc_auc_score(y_ts, pred_test_KN)
    score_train_KN = roc_auc_score(y_tr, pred_train_KN)
      
    # Model evaluation with Log_loss Metric    
    score_test_ROC_LogReg = log_loss(y_ts, pred_test_Log)
    score_train_ROC_LogReg = log_loss(y_tr, pred_train_Log)
    score_test_Log_KN = log_loss(y_ts, pred_test_KN)
    score_train_Log_KN = log_loss(y_tr, pred_train_KN)
# Average score of K-Fold Validation
pred_all = pred.mean(axis=1)
score_all = roc_auc_score(y_test,pred_all)
print('ROC_AUC score_all:', score_all)
pred
import xgboost as xgb

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
param = {'max_depth':4, 'eta': 0.1, 'silent': 1, 'objective': 'reg:linear','min_child_weight':0}
param['nthread'] = 4
param['eval_metric'] = 'auc'
evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 10
bst = xgb.train(param,dtrain,num_round,evallist)
y_pred = bst.predict(dtest)
y_pred
fig,ax= plt.subplots(figsize=(10,10))
xgb.plot_importance(bst, ax=ax);
fig,ax= plt.subplots(figsize=(200,200));
xgb.plot_tree(bst, num_trees=0,rankdir='LR', ax=ax);







