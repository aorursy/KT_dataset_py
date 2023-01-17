import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2
df = pd.read_excel("/kaggle/input/BreastCancer_Prognostic_v1.xlsx")
vals=[]
sum=0
for ele in range(len(df['Lymph_Node_Status'])):
    if type(df['Lymph_Node_Status'].iloc[ele]) == int:
        sum+=df['Lymph_Node_Status'].iloc[ele]
    vals.append(df['Lymph_Node_Status'].iloc[ele])
mean = sum/len(vals)
for ind in range(len(vals)):
    if type(vals[ind])==str:
        vals[ind] = np.round(mean).astype(np.uint8)
df.drop(columns = 'Lymph_Node_Status',inplace = True)
df = df.assign(Lymph_Node_Status = vals)
df['Outcome'].value_counts()
df.index = df['ID']
labels = df['Outcome']
labels = pd.Series([0 if x=='N' else 1 for x in labels])
labeltime = df['Time']
df, x_test, labels, y_test = train_test_split(df, labels, test_size = 0.20, train_size = 0.80, shuffle = True, stratify = labels)
sns.heatmap(df.corr())
df.drop(columns = ["ID","Outcome","Time"],inplace = True)
top_feats = SelectKBest(score_func = chi2, k = 10)
fit = top_feats.fit(df,labels)
scores = pd.DataFrame(fit.scores_)
cols = pd.DataFrame(df.columns)
feat_scores = pd.concat([cols,scores],axis = 1)
feat_scores.columns = ['Features','Chi-2 score']
print(feat_scores.nlargest(10,'Chi-2 score'))
df_chi = df[list(feat_scores.nlargest(10,'Chi-2 score')['Features'])]
cols = df.columns
scaler = StandardScaler()
scaler.fit(df)
df_std = scaler.transform(df)
df_std = pd.DataFrame(df_std,columns = cols)
# x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size = 0.20, train_size = 0.80, shuffle = True, stratify = labels)
cv = StratifiedKFold(n_splits=4,shuffle=True)
cv.get_n_splits(df,labels)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
paramgrid = {'C':list(np.arange(1,6,1)),'degree':np.arange(1,10,1)}
scores = []
predictions = []
actual = []
classifiers1 = [] #to collect the 4 svm predictors and use it at the end
# gscres.best_params_
for trainind,testind in cv.split(df_std,labels):
#     print(trainind)
#     print(testind)
    xtrain,xtest = df_std.iloc[trainind],df_std.iloc[testind]
    ytrain,ytest = labels.iloc[trainind],labels.iloc[testind]
    actual.append(ytest)
    gsc = GridSearchCV(estimator = SVC(kernel='poly'), param_grid= paramgrid,cv=5,scoring = 'f1',n_jobs = 1,verbose = 0)
    gscres = gsc.fit(xtrain,ytrain)
    clfsvm = SVC(C=gscres.best_params_['C'], kernel = 'poly', degree = gscres.best_params_['degree'], class_weight = {0:2.5,1:5.5})
#     clfsvm = SVC(C=10.0, kernel = 'poly', degree = 10)
    clfsvm.fit(xtrain,ytrain)
    predicted_svm = clfsvm.predict(xtest)
    predictions.append(predicted_svm)
    classifiers1.append(clfsvm)
    a = classification_report(ytest, predicted_svm, target_names=['N','R'],output_dict=True)
    print(str(a['N']['f1-score']) + ' ' + str(a['R']['f1-score']))
    scores.append(a)
ytest.value_counts()
f1nsc = [x['N']['f1-score'] for x in scores]
f1navg = np.mean(np.array(f1nsc))
f1rsc = [x['R']['f1-score'] for x in scores]
f1ravg = np.mean(np.array(f1rsc))
print('F1 score average for N  is : {} and F1 score average for R is : {}'.format(f1navg,f1ravg))
print("Confusion matrix for SVM")
print(confusion_matrix(actual[0],predictions[0]))
print(confusion_matrix(actual[1],predictions[1]))
print(confusion_matrix(actual[2],predictions[2]))
print(confusion_matrix(actual[3],predictions[3]))
from imblearn.over_sampling import SVMSMOTE
oversample = SVMSMOTE()
df2_std, labels2 = oversample.fit_resample(df_std, labels)
labels2.shape
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
paramgrid = {'C':list(np.arange(1,6,1)),'degree':np.arange(1,10,1)}
scores2 = []
predictions2 = []
actual2 = []
classifiers2 = []
# gscres.best_params_
for trainind,testind in cv.split(df2_std,labels2):
#     print(trainind)
#     print(testind)
    xtrain,xtest = df2_std.iloc[trainind],df2_std.iloc[testind]
    ytrain,ytest = labels2.iloc[trainind],labels2.iloc[testind]
    actual2.append(ytest)
    gsc = GridSearchCV(estimator = SVC(kernel='poly'), param_grid= paramgrid,cv=5,scoring = 'f1',n_jobs = 1,verbose = 0)
    gscres = gsc.fit(xtrain,ytrain)
    clfsvm = SVC(C=gscres.best_params_['C'], kernel = 'poly', degree = gscres.best_params_['degree'],class_weight = {0:2.5,1:5})
#     clfsvm = SVC(C=10.0, kernel = 'poly', degree = 10)
    clfsvm.fit(xtrain,ytrain)
    predicted_svm = clfsvm.predict(xtest)
    predictions2.append(predicted_svm)
    classifiers2.append(clfsvm)
    a = classification_report(ytest, predicted_svm, target_names=['N','R'],output_dict=True)
    print(str(a['N']['f1-score']) + ' ' + str(a['R']['f1-score']))
    scores2.append(a)
f1nsc2 = [x['N']['f1-score'] for x in scores2]
f1navg2 = np.mean(np.array(f1nsc2))
f1rsc2 = [x['R']['f1-score'] for x in scores2]
f1ravg2 = np.mean(np.array(f1rsc2))
print('F1 score average for N  is : {} and F1 score average for R is : {}'.format(f1navg2,f1ravg2))
print("Confusion matrix for SVMSMOTE AND SVM")
print(confusion_matrix(actual2[0],predictions2[0]))
print(confusion_matrix(actual2[1],predictions2[1]))
print(confusion_matrix(actual2[2],predictions2[2]))
print(confusion_matrix(actual2[3],predictions2[3]))
xtestids = x_test['ID']
xtestoutcomes = x_test['Outcome']
xtesttime = x_test['Time']
x_test.drop(columns = ['Outcome','Time','ID'],inplace = True)
xtest_std = scaler.transform(x_test)
xtest_std = pd.DataFrame(xtest_std,columns = cols)
from xgboost import XGBClassifier
from xgboost import plot_importance
class MyXGBClassifier(XGBClassifier):
    @property
    def coef_(self):
        return None
model = MyXGBClassifier()
model.fit(df,labels)
#print(model.feature_importances_)
fig,ax = plt.subplots(1,1,figsize=(10,10))
impplot = plot_importance(model,ax = ax)
plt.show()
impfeats = [impplot.get_yticklabels()[::-1][i].get_text() for i in range(0,7)]
impfeats
newdf = df_std[impfeats]
from sklearn.ensemble import RandomForestClassifier
clfrf = RandomForestClassifier(bootstrap=True,class_weight={0:1,1:4},criterion='entropy',random_state=7)
# paramsrf = {'max_depth':np.arange(1,25,2)}
# gsrf = GridSearchCV(estimator = clfrf, param_grid= paramsrf,cv=4,scoring = 'f1',n_jobs = 1,verbose = 0)
# gsrfres = gsrf.fit(df,labels)
# gsrfres.best_params_
clfrf.max_depth = 2
clfrf.fit(newdf, labels)
predictedrf = clfrf.predict(xtest_std[impfeats])
print(classification_report(y_test, predictedrf, target_names=['N','R']))
print(confusion_matrix(y_test,predictedrf))
svmpreds = []
smotesvmpreds = [] 
for clsf,clsf2 in zip(classifiers1,classifiers2):
    svmpreds.append(clsf.predict(xtest_std))
    smotesvmpreds.append(clsf2.predict(xtest_std))
svmpreds = np.sum(svmpreds,axis = 0)
smotesvmpreds = np.sum(smotesvmpreds,axis = 0)
for ind,ele in enumerate(svmpreds):
    avg = ele/4
    if avg<0.5:
        svmpreds[ind] = 0
    else:
        svmpreds[ind] = 1
for ind,ele in enumerate(smotesvmpreds):
    avg = ele/4
    if avg<0.5:
        smotesvmpreds[ind] = 0
    else:
        smotesvmpreds[ind] = 1
finalpreds = []
for it in range(xtest_std.shape[0]):
    if smotesvmpreds[it]==1:
        finalpreds.append(1)
    elif svmpreds[it]==0 and predictedrf[it]==0:
        finalpreds.append(0)
    elif svmpreds[it]==1 and predictedrf[it]==1:
        finalpreds.append(1)
    else:
        finalpreds.append(1)
print(classification_report(y_test, finalpreds, target_names=['N','R']))
print(confusion_matrix(y_test,finalpreds))
finallabels = ['N' if y==0 else 'R' for y in finalpreds]
x_test['Time'] = xtesttime
x_test['Outcome'] = finallabels
x_test['ID'] = xtestids
x_test.index = np.arange(0,40,1)
x_test.to_csv('mysubmission1.csv',index=False)
data = pd.read_excel("/kaggle/input/BreastCancer_Prognostic_v1.xlsx")
vals=[]
sum=0
for ele in range(len(data['Lymph_Node_Status'])):
    if type(data['Lymph_Node_Status'].iloc[ele]) == int:
        sum+=data['Lymph_Node_Status'].iloc[ele]
    vals.append(data['Lymph_Node_Status'].iloc[ele])
mean = sum/len(vals)
for ind in range(len(vals)):
    if type(vals[ind])==str:
        vals[ind] = np.round(mean).astype(np.uint8)
data.drop(columns = 'Lymph_Node_Status',inplace = True)
data = data.assign(Lymph_Node_Status = vals)
indr = []
for ind,row in data.iterrows():
    if row['Outcome']=='R':
        indr.append(ind)
data = data.iloc[indr]
timesr = data['Time']
data.index = data['ID']
datatrain, datatest, ytr, ytst = train_test_split(data,timesr,test_size = 0.20,shuffle=True)
dtestoutcomes = datatest['Outcome']
dtestid = datatest['ID']
datatrain.drop(columns=['Outcome','Time','ID'],inplace = True)
datatest.drop(columns=['Outcome','Time','ID'],inplace = True)
colstest = datatrain.columns
sc = StandardScaler()
sc.fit(datatrain)
dtrain_std = scaler.transform(datatrain)
dtrain_std = pd.DataFrame(dtrain_std,columns = colstest)
dtest_std = scaler.transform(datatest)
dtest_std = pd.DataFrame(dtest_std,columns = colstest)
import xgboost as xgb
from sklearn.metrics import mean_squared_error
xgbr = xgb.XGBRegressor() 
xgbr.max_depth = 700
xgbr.booster = "gbtree"
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5)
kf.get_n_splits(dtrain_std)
rmsetime = []
modelsxgb = []
for train_index, test_index in kf.split(dtrain_std):
    X_train, X_test = dtrain_std.iloc[train_index], dtrain_std.iloc[test_index]
    ytrain, ytest = ytr.iloc[train_index], ytr.iloc[test_index]
    rmse = []
    spwvals = []
    for i in np.arange(0.5,8,0.1):    
        spwvals.append(i)
        xgbr.scale_pos_weight = i
        xgbr.fit(X_train,ytrain)
        output = xgbr.predict(X_test)
        rmse.append(mean_squared_error(ytest,output,squared = False))
    xgbr.scale_pos_weight = spwvals[minind]
    modelsxgb.append(xgbr)
    minind = rmse.index(min(rmse))
    print("Best prediction with scale_pos_weight as {} and rmse as {}".format(spwvals[minind],rmse[minind]))
    rmsetime.append(rmse[minind])
np.sum(rmsetime)/len(rmsetime)
plt.plot(range(len(rmsetime)),rmsetime)
plt.xlabel('Model number')
plt.ylabel('RMSE')
plt.show()
testpredslist = []
timepredicted = []
for models in modelsxgb:
    outputtime = models.predict(dtest_std)
    timepredicted.append(outputtime)
    testpredslist.append(mean_squared_error(ytst,outputtime,squared = False))
outputtime = np.sum(timepredicted,axis = 0)/len(modelsxgb)
from catboost import CatBoostRegressor
modelcat = CatBoostRegressor(iterations = 10, learning_rate = 0.7, depth = 8,loss_function = "RMSE",verbose = 0)
modelcat.fit(dtrain_std,ytr)
outputcat = model.predict(dtest_std)
mean_squared_error(ytst,outputcat,squared = False)
datatest['Outcome'] = dtestoutcomes
datatest['ID'] = dtestid
datatest['Time'] = list(outputtime)
datatest.to_csv('timeprediction.csv',index = False)
