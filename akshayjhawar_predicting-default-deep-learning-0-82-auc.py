import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
import random
from scipy import stats

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from keras import models
from keras import layers
data_  = pd.read_csv('../input/GiveMeSomeCredit/cs-training.csv')
data_.columns
print(data_['SeriousDlqin2yrs'].value_counts()/data_.shape[0] *100)

## Pie Chart
labels = 'Default', 'Non-Defaults'
sizes = [6.684, 93.316]
explode = (0.2, 0)
cols    = ['#00FFFF', '#008080']

fig = plt.figure(figsize = (4,4))
plt.pie(sizes, explode=explode, colors = cols, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')
plt. title("Percentage of Defaults and Non-Defaults")
plt.show()
def missing_vals(data_):
    miss_     = data_.isnull().sum()
    miss_pct  = data_.isnull().sum()/data_.shape[0]
    
    miss_pct  = pd.concat([miss_, miss_pct], axis =1)
    miss_pct.reset_index(inplace=True)
    miss_cols = miss_pct.rename(columns={'index':'Column Name', 0:'Missings', 1:'Missing_pct'})
    
    miss_cols = miss_cols[miss_cols.iloc[:,1]!=0].sort_values('Missing_pct', ascending=False).round(1)
    miss_cols.reset_index(inplace=True, drop=True)
    
    return miss_cols 
miss = missing_vals(data_)
miss
data_.describe()
cols = list(data_.columns)
cols = cols[1:]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c']
fig = plt.figure(figsize=(15, 12))
for i in range(0, len(cols)):
    plt.subplot(5, 4, i+1)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)

    plt.hist(data_[cols[i]], bins=30, color=colors[i])
    plt.title(cols[i])

plt.tight_layout()
df = data_.copy()
df['MonthlyIncome'] =df['MonthlyIncome'].transform(lambda x: x.fillna(x.mean()))
df['NumberOfDependents'] =df['NumberOfDependents'].transform(lambda x: x.fillna(x.mean()))
miss = missing_vals(df)
miss
random.seed(32)
pca = PCA(n_components = 2)
pca.fit(df)

scores = pca.transform(df)

x,y = scores[:,0] , scores[:,1]
df_ = pd.DataFrame({'x': x, 'y':y, 'clusters':df['SeriousDlqin2yrs']})
grouping_ = df_.groupby('clusters')


fig, ax = plt.subplots(figsize=(10, 5))
names = {0: 'Non-Defaults', 
         1: 'Defaults'}

for name, grp in grouping_:
    ax.plot(grp.x, grp.y, marker='o', label = names[name], linestyle='')
    ax.set_aspect('auto')
    ax.set_ylim([0,200000])     ### I have just kept a upper cap on the axis to see the distribution of them
    
ax.legend()
plt.title('Plot showing Defaults and Non-Defaults')
plt.show()
df = pd.get_dummies(df)
df = df[[c for c in df if c not in ['SeriousDlqin2yrs']]+['SeriousDlqin2yrs']]
df = df.drop(['Unnamed: 0'], axis=1)
df.head()
correlations = pd.DataFrame(df.corr()['SeriousDlqin2yrs'].sort_values())
correlations = correlations.rename(columns = {'SeriousDlqin2yrs':'Correlation value'})
correlations
correlations.plot(kind="bar", color="olive")
corr_ = df.corr()
fig= plt.figure(figsize=(15,7))
sns.heatmap(corr_, cmap = plt.cm.RdYlBu_r, vmin = -0.9, annot = True, vmax = 0.9)
#### Here we will drop variables to better our model
df           = df.drop(['NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(df.drop(['SeriousDlqin2yrs'], axis=1), df['SeriousDlqin2yrs'], test_size=0.2,random_state = 72)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 
  
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 
  
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))
def model_fit_reports(algo,X_,y_,performCV=True,printFeatureImportance=True, cv_folds=5):
    
    #Accuracy, Precision, Recall, F1 Score
    pred = algo.predict(X_)
    accu = accuracy_score(y_, pred)
    f1_  = f1_score(y_, pred)
    rec  = recall_score(y_, pred)
    prec = precision_score(y_, pred)

    
    #GINI & AUC
    fpr, tpr, thresholds = roc_curve(y_, pred)
    roc_auc = auc(fpr, tpr)
    Gini   = 2*roc_auc - 1   
    labels  = ['Accuracy','F1 Score', 'Recall', 'Precision', 'Gini', 'AUC']
    values  = [accu,f1_,rec,prec,Gini,roc_auc]
    
    all_    = pd.Series(values,labels)  
    print(all_)
    all_.plot(kind='bar', title='Model Fit Report')   


    if performCV:
        cv_score = cross_val_score(algo, X_, y_, cv=cv_folds, scoring='roc_auc')
        GINI     = 2 * cv_score -1
        print("AUC : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        print("GINI : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(GINI),np.std(GINI),np.min(GINI),np.max(GINI)))

    cols = list(X_.columns)
    if printFeatureImportance:
        feat_imp = pd.Series(algo.feature_importances_, cols).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')    
    
    return all_
regressor = LogisticRegression(random_state =2, solver='sag', max_iter = 10**2)
regressor.fit(X_train_res, y_train_res)
train = model_fit_reports(algo =regressor ,X_ = X_train,y_ = y_train, performCV=True, printFeatureImportance=False, cv_folds=5)
test  = model_fit_reports(algo =regressor ,X_ = X_test,y_ = y_test, performCV=True, printFeatureImportance=False, cv_folds=5)
rfc = RandomForestClassifier(random_state=8, n_estimators=500)
rfc.fit(X_train_res, y_train_res)
train = model_fit_reports(algo =rfc ,X_ = X_train,y_ = y_train, performCV=True, printFeatureImportance=True, cv_folds=5)
test = model_fit_reports(algo =rfc ,X_ = X_test,y_ = y_test, performCV=True, printFeatureImportance=True, cv_folds=5)
gbc = GradientBoostingClassifier()
gbc.fit(X_train_res, y_train_res)
train = model_fit_reports(algo =gbc ,X_ = X_train,y_ = y_train, performCV=True, printFeatureImportance=False, cv_folds=5)
test = model_fit_reports(algo =gbc ,X_ = X_test,y_ = y_test, performCV=True, printFeatureImportance=False, cv_folds=5)
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1    = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, 
                                                               min_samples_split=500,
                                                               min_samples_leaf=50,
                                                               max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
                                                               param_grid = param_test1, scoring='roc_auc',
                                                               n_jobs=4,iid=False, cv=5)

gsearch1.fit(X_train_res,y_train_res)
gsearch1.best_params_, gsearch1.best_score_
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,400,600)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch2.fit(X_train_res,y_train_res)
gsearch2.best_params_, gsearch2.best_score_
param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}

gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=15,max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_train_res,y_train_res)
gsearch3.best_params_, gsearch3.best_score_
param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=15, min_samples_split=1000, min_samples_leaf=30, subsample=0.8, random_state=10),
param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(X_train_res,y_train_res)
gsearch4.best_params_, gsearch4.best_score_
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_depth= 15, min_samples_split= 1000, min_samples_leaf=30,max_features=7, random_state = 10),
param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(X_train_res,y_train_res)
gsearch5.best_params_, gsearch5.best_score_
gbc = GradientBoostingClassifier(n_estimators=80, max_depth= 15, min_samples_split= 1000, min_samples_leaf=30,subsample=0.9,max_features=7)
gbc.fit(X_train_res, y_train_res)
train = model_fit_reports(gbc,X_train,y_train,performCV=True,printFeatureImportance=False, cv_folds=5)
test = model_fit_reports(gbc,X_test,y_test,performCV=True,printFeatureImportance=False, cv_folds=5)
xgb1 = XGBClassifier(
 learning_rate =0.001,
 n_estimators=1000,
 max_depth=9,
 min_child_weight=1,
 gamma=0.2,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 reg_alpha = 0.1,
 scale_pos_weight=1,
 seed=27)
xgb1.fit(X_train_res, y_train_res)
train = model_fit_reports(xgb1,X_train,y_train,performCV=True,printFeatureImportance=False, cv_folds=5)
test = model_fit_reports(xgb1,X_test,y_test,performCV=True,printFeatureImportance=False, cv_folds=5)
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train_res, y_train_res)
gsearch1.best_params_, gsearch1.best_score_
param_test2 = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(X_train_res, y_train_res)
gsearch2.best_params_, gsearch2.best_score_
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=9,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_train_res, y_train_res)
gsearch3.best_params_, gsearch3.best_score_
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=5,
 min_child_weight=1, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch6.fit(X_train_res, y_train_res)
gsearch6.best_params_, gsearch6.best_score_
n_inputs = X_train_res.shape[1]

model = models.Sequential()
model.add(layers.Dense(16, activation ='relu', input_shape =(n_inputs, )))
model.add(layers.Dense(32,activation = 'relu'))
model.add(layers.Dense(1,activation ='sigmoid'))
model.compile(optimizer = 'rmsprop',
             loss= 'binary_crossentropy',
             metrics = ['accuracy'])
history = model.fit(X_train_res,
                   y_train_res,
                   epochs=150,
                   batch_size=512,
                   validation_data=(X_test,y_test))
score = model.evaluate(X_test, y_test)
#GINI & AUC 
pred  = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, pred)
roc_auc = auc(fpr, tpr)
print("The AUC of the Test model is ", roc_auc)
Gini   = 2*roc_auc - 1
print("The Gini of the Test model is ", Gini)
log_pred                   = regressor.predict(X_test)
rfc_pred                       = rfc.predict(X_test)
gbm_pred                       = gbc.predict(X_test)
xgb_pred                       = xgb1.predict(X_test)
deepl_pred                     = model.predict(X_test)
log_fpr, log_tpr, log_threshold   = roc_curve(y_test, log_pred)
rfc_fpr, rfc_tpr, rfc_threshold   = roc_curve(y_test, rfc_pred)
gbm_fpr, gbm_tpr, gbm_threshold   = roc_curve(y_test, gbm_pred)
xgb_fpr, xgb_tpr, xgb_threshold   = roc_curve(y_test, xgb_pred)
deepl_fpr, deepl_tpr, deepl_threshold   = roc_curve(y_test, deepl_pred)
# Plot ROC curves
fig  = plt.figure(figsize=(10,6))
plt.title('ROC Curve \n Comparison of Classifiers')
plt.plot(log_fpr, log_tpr, label ='Logistic Regression AUC: {:.2f}'.format(roc_auc_score(y_test, log_pred)))
plt.plot(rfc_fpr, rfc_tpr, label ='Random Forest AUC: {:.2f}'.format(roc_auc_score(y_test, rfc_pred)))
plt.plot(gbm_fpr, gbm_tpr, label ='GBM AUC: {:.2f}'.format(roc_auc_score(y_test, gbm_pred)))
plt.plot(xgb_fpr, xgb_tpr, label ='XgBoost AUC: {:.2f}'.format(roc_auc_score(y_test, xgb_pred)))
plt.plot(deepl_fpr, deepl_tpr, label ='Deep Learning AUC: {:.2f}'.format(roc_auc_score(y_test, deepl_pred)))

plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()
## Predict for the whole of the dataset
actual_   = df['SeriousDlqin2yrs']
df_new    = df.copy()
df_new    = df.drop(['SeriousDlqin2yrs'], axis=1)
pred_all  = model.predict(df_new)
fpr, tpr, thresholds = roc_curve(actual_, pred_all)
roc_auc = auc(fpr, tpr)
print("The AUC of the overall model is: {:.2f}".format(roc_auc))
Gini   = 2*roc_auc - 1
print("The Gini of the overall model is: {:.2f}".format(Gini))
## Convert the probability into a score
Base_Score = 600
pdo        = 120
Good_Bads  = 10

## Creating a function to calculate a Score
def score_(x, Offset, Factor):
    score_ = Offset - Factor * np.log(x)
    return score_
Factor          = pdo/np.log(2)
Offset          = Base_Score - Factor * np.log(Good_Bads)
Score_          = score_(pred_all, Offset=Offset, Factor=Factor)
Actual_vals      = pd.DataFrame(actual_).reset_index(drop=True)
Score            = pd.DataFrame(Score_)
pred_all         = pd.DataFrame(pred_all)
combine          = [Actual_vals,pred_all, Score]
combine_         = pd.concat(combine, axis=1)
combine_.columns  = ['Default', 'Predicted_prob', 'Risk_Score']
combine_              = combine_.replace([np.inf, -np.inf], np.nan)
combine_              = combine_[combine_.isna()==False]
combine_.loc[(combine_.Risk_Score>900), ['Risk_Score']] = 899
fig = plt.figure(figsize = (5,5))
ax = sns.distplot(combine_['Risk_Score'], hist=True, kde=True,
                        bins=100, color = 'blue',hist_kws={'edgecolor':'black'},
                         kde_kws={'linewidth': 4})

ax.set_xlabel("Application Score")
ax.set_ylabel("Count")
ax.set_title("Score Distribution")
max_                         = max(combine_['Risk_Score'])
min_                         = min(combine_['Risk_Score'])
combine_['Score_decile']     = pd.cut(combine_['Risk_Score'], bins=[min_,300,415,490,555,600,690,730,max_],labels = [min_,300,415,490,555,600,690,730], include_lowest= True)
no_of_defaults                = combine_.groupby('Score_decile', as_index=False).agg({'Default':'sum', 'Predicted_prob':'count'})
no_of_defaults['default_rate'] = (no_of_defaults['Default']/no_of_defaults['Predicted_prob'] * 100)
no_of_defaults['Score_decile'] = round(no_of_defaults['Score_decile'],0)
no_of_defaults
fig = plt.figure(figsize = (8,6))
plt.plot(no_of_defaults['Score_decile'], no_of_defaults['default_rate'], color = 'c')
plt.title("Default Rates across Score Range", fontsize = 15)
plt.xlabel("Score", fontsize = 12)
plt.ylabel("Default Rate", fontsize=12)
plt.annotate("A rank order is seen \n for all the Score Bands",xy= (350,7), xytext =(380,8), arrowprops = dict(facecolor = "black", shrink = 0.05))
plt.legend()
Final_data     = pd.concat([df,combine_], axis=1)
Final_data.head()