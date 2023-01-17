import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

#df
X = df.values[:,:-1]
y = df.values[:,-1]
beta=sum(y)/(len(y))
beta
plt.hist(df['Class'])
plt.title("distribution of classes, i.e. y")

import seaborn as sns

corr = df.corr() 

# plot correlation matrix 

fig, ax = plt.subplots(figsize = (20,20))
#ax = fig.add_subplot(figsize=(20,20)) #图片大小为20*20
#sns.heatmap(correlations, annot=True, vmax=1,vmin = -1, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
#热力图参数设置（相关系数矩阵，颜色，每个值间隔等）
#ticks = numpy.arange(0,16,1) #生成0-16，步长为1 
ax.set_xticks((np.arange(corr.shape[0]))) #横坐标标注点
ax.set_yticks((np.arange(corr.shape[0]))) #纵坐标标注点
#ax.set_xticks(ticks) #生成刻度 
#ax.set_yticks(ticks)
ax.set_xticklabels(df.columns) #生成x轴标签 
ax.set_yticklabels(df.columns)
ax.set_title('Correlation Matrix \n all factors', fontsize = 18)
#ax.set_ylabel('数字', fontsize = 18)
#ax.set_xlabel('字母', fontsize = 18)

from sklearn import preprocessing

X = preprocessing.scale(X) 

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

plt.hist(y_res)
plt.title("distribution of classes after oversampling")

X, y = X_res, y_res



from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC

svc = LinearSVC(dual=False)

scores = cross_validate(svc,X,y,cv=10,scoring=('precision','recall','f1'))
    # 10 k cross-validation
print("fitting time : ",scores['fit_time'].mean(), "\n scoring time : ", scores['score_time'].mean()
      ,"\n precision score : ", scores['test_precision'].mean(), 
          "\n recall score : ", scores['test_recall'].mean(), "\n f1 score : ", scores['test_f1'].mean(), "\n")
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

dtf = DecisionTreeClassifier()

scores = cross_validate(dtf,X,y,cv=10,scoring=('precision','recall','f1'))
    # 10 k cross-validation
print("fitting time : ",scores['fit_time'].mean(), "\n scoring time : ", scores['score_time'].mean()
      ,"\n precision score : ", scores['test_precision'].mean(), 
          "\n recall score : ", scores['test_recall'].mean(), "\n f1 score : ", scores['test_f1'].mean(), "\n")

from sklearn.neural_network import MLPClassifier

mlpc = MLPClassifier()

scores = cross_validate(mlpc,X,y,cv=10,scoring=('precision','recall','f1'))
    # 10 k cross-validation
print("fitting time : ",scores['fit_time'].mean(), "\n scoring time : ", scores['score_time'].mean()
      ,"\n precision score : ", scores['test_precision'].mean(), 
          "\n recall score : ", scores['test_recall'].mean(), "\n f1 score : ", scores['test_f1'].mean(), "\n")

from sklearn.linear_model import LogisticRegression

LogR = LogisticRegression()

scores = cross_validate(LogR,X,y,cv=10,scoring=('precision','recall','f1'))
    # 10 k cross-validation
print("fitting time : ",scores['fit_time'].mean(), "\n scoring time : ", scores['score_time'].mean()
      ,"\n precision score : ", scores['test_precision'].mean(), 
          "\n recall score : ", scores['test_recall'].mean(), "\n f1 score : ", scores['test_f1'].mean(), "\n")


weights = np.arange(0.05,1,0.05)
precisions = []
recalls = []
f1s = []



for i in weights:
    LogR = LogisticRegression(class_weight = {0:i, 1:1-i})
    scores = cross_validate(LogR,X,y,cv=10,scoring=('precision','recall','f1'))
    precisions.append(scores['test_precision'].mean())
    recalls.append(scores['test_recall'].mean())
    f1s.append(scores['test_f1'].mean())

plt.plot(weights, precisions, label = 'precision')
plt.scatter(weights[precisions.index(max(precisions))],max(precisions), marker = '*', linewidths = 5)
plt.plot(weights, recalls, label = 'recall')
plt.scatter(weights[recalls.index(max(recalls))],max(recalls), marker = '*', linewidths = 5)
plt.plot(weights, f1s, label = 'f1')
plt.scatter(weights[f1s.index(max(f1s))],max(f1s), marker = '*', linewidths = 5)
plt.title("searching for the optimal set of weights")
plt.legend()


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

scores = cross_validate(gnb,X,y,cv=10,scoring=('precision','recall','f1'))
    # 10 k cross-validation
print("fitting time : ",scores['fit_time'].mean(), "\n scoring time : ", scores['score_time'].mean()
      ,"\n precision score : ", scores['test_precision'].mean(), 
          "\n recall score : ", scores['test_recall'].mean(), "\n f1 score : ", scores['test_f1'].mean(), "\n")

from xgboost.sklearn import XGBClassifier

xgb = XGBClassifier(objective='binary:logistic')

scores = cross_validate(xgb,X,y,cv=10,scoring=('precision','recall','f1'))
    # 10 k cross-validation
print("fitting time : ",scores['fit_time'].mean(), "\n scoring time : ", scores['score_time'].mean()
      ,"\n precision score : ", scores['test_precision'].mean(), 
          "\n recall score : ", scores['test_recall'].mean(), "\n f1 score : ", scores['test_f1'].mean(), "\n")

from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(n_jobs = 8)

scores = cross_validate(rfc,X,y,cv=10,scoring=('precision_macro','recall_macro','f1_macro'))
    # 10 k cross-validation


print("fitting time : ",scores['fit_time'].mean(), "\n scoring time : ", scores['score_time'].mean(), 
      "\n precision score : ", scores['test_precision_macro'].mean(), "\n recall score : ", scores['test_recall_macro'].mean(), 
      "\n f1 score : ", scores['test_f1_macro'].mean(), "\n")

from scipy import interp
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold


# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=11)
classifier =  DecisionTreeClassifier()


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = plot_roc_curve(classifier, X[test], y[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver Operating Characteristic")
ax.legend(loc="lower right")
plt.show()
from sklearn.metrics import roc_curve,plot_roc_curve,brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score,auc,brier_score_loss
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
models={'LinearSVC':LinearSVC(dual=False),
        'Naive Bayes':GaussianNB(),
        'Logistic Regression':LogisticRegression(),
        'Random Forest':RandomForestClassifier(),
        'XGBoost':xgb.XGBClassifier(objective='binary:logistic'),
        'Decision Tree':DecisionTreeClassifier(),
        'Neural Network':MLPClassifier()
}
pic={}
cal={}
plt.figure(figsize=(15,15))
for x in list(models.keys()):
    models[x].fit(x_train,y_train)
    #pic[x]=plot_roc_curve(models[x],x_test,y_test)
    try:
        y_pred=models[x].predict_proba(x_test)[:,1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred, normalize=False, n_bins=100)
    except AttributeError:
        y_pred=models[x].decision_function(x_test)
        '''
        prob_pos = y_pred.copy()
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        '''
        #y_pred=(y_pred-np.mean(models[x].decision_function(x_test)))/np.std(models[x].decision_function(x_test))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred, normalize=True, n_bins=100)
        y_pred=(y_pred-y_pred.min())/(y_pred.max()-y_pred.min())
    #fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos,normalize=False, n_bins=10)
    cal[x] = (mean_predicted_value,fraction_of_positives,brier_score_loss(y_test,y_pred))
    fpr,tpr,thresholds=roc_curve(y_test,y_pred)
    plt.plot(fpr,tpr,label=x+'(area %0.8f)'%auc(fpr,tpr))
plt.legend()
plt.title('models comparasion')
plt.show()
plt.figure(figsize=(15,15))
for x in list(cal.keys()):
    plt.plot(cal[x][0],cal[x][1],label=x+'(brier_score %0.6f'%cal[x][2])
plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
plt.legend(loc='best')
plt.ylabel('Fraction of Positives')
plt.xlabel('Mean Predicted Values')
plt.title('Calibration Curves (Reliability Curve)')
plt.show()
cal
