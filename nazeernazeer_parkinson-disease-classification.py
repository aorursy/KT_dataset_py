#importing the libraries that we use
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from mlxtend.classifier import StackingClassifier
from sklearn import model_selection
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
sns.set(color_codes=True) # adds a nice background to the graphs
%matplotlib inline
pks_df = pd.read_csv("../input/parkinson-classifications/Data - Parkinsons")
pks_df.head(10).style.background_gradient(cmap="RdYlBu")
pks_df.shape
pks_df.dtypes
pks_df.info()
pks_df.describe().T
pks_df.skew()
pks_df[pks_df.duplicated()]
Target = pks_df["status"]
#Plots to see the distribution of the features individually
def distributionPlot(pks_df):
    plt.figure(figsize= (20,15))
    plt.subplot(5,5,1)
    sns.distplot(pks_df["MDVP:Fo(Hz)"],hist=False,kde=True, color='lightblue')
    plt.xlabel('MDVP:Fo(Hz)')

    plt.subplot(5,5,2)
    sns.distplot(pks_df["MDVP:Fhi(Hz)"],hist=False,kde=True, color='lightgreen')
    plt.xlabel('MDVP:Fhi(Hz)')

    plt.subplot(5,5,3)
    sns.distplot(pks_df["MDVP:Flo(Hz)"],hist=False,kde=True, color='pink')
    plt.xlabel('MDVP:Flo(Hz)')

    plt.subplot(5,5,4)
    sns.distplot(pks_df["MDVP:Jitter(%)"],hist=False,kde=True, color='gray')
    plt.xlabel('MDVP:Jitter(%)')

    plt.subplot(5,5,5)
    sns.distplot(pks_df["MDVP:Jitter(Abs)"],hist=False,kde=True, color='cyan')
    plt.xlabel('MDVP:Jitter(Abs)')

    plt.subplot(5,5,6)
    sns.distplot(pks_df["MDVP:RAP"],hist=False,kde=True, color='Aquamarine')
    plt.xlabel('MDVP:RAP')

    plt.subplot(5,5,7)
    sns.distplot(pks_df["MDVP:PPQ"],hist=False,kde=True, color='lightblue')
    plt.xlabel('MDVP:PPQ')

    plt.subplot(5,5,8)
    sns.distplot(pks_df["Jitter:DDP"],hist=False,kde=True, color='lightgreen')
    plt.xlabel('Jitter:DDP')

    plt.subplot(5,5,9)
    sns.distplot(pks_df["MDVP:Shimmer"],hist=False,kde=True, color='pink')
    plt.xlabel('MDVP:Shimmer')

    plt.subplot(5,5,10)
    sns.distplot(pks_df["MDVP:Shimmer(dB)"],hist=False,kde=True, color='gray')
    plt.xlabel('MDVP:Shimmer(dB)')

    plt.subplot(5,5,11)
    sns.distplot(pks_df["Shimmer:APQ3"],hist=False,kde=True, color='cyan')
    plt.xlabel('Shimmer:APQ3')

    plt.subplot(5,5,12)
    sns.distplot(pks_df["Shimmer:APQ5"],hist=False,kde=True, color='Aquamarine')
    plt.xlabel('Shimmer:APQ5')

    plt.subplot(5,5,13)
    sns.distplot(pks_df["MDVP:APQ"],hist=False,kde=True, color='lightblue')
    plt.xlabel('MDVP:APQ')

    plt.subplot(5,5,14)
    sns.distplot(pks_df["Shimmer:DDA"],hist=False,kde=True, color='lightgreen')
    plt.xlabel('Shimmer:DDA')

    plt.subplot(5,5,15)
    sns.distplot(pks_df["NHR"],hist=False,kde=True, color='pink')
    plt.xlabel('NHR')

    plt.subplot(5,5,16)
    sns.distplot(pks_df["HNR"],hist=False,kde=True, color='gray')
    plt.xlabel('HNR')

    plt.subplot(5,5,17)
    sns.distplot(pks_df["RPDE"],hist=False,kde=True, color='cyan')
    plt.xlabel('RPDE')

    plt.subplot(5,5,18)
    sns.distplot(pks_df["DFA"],hist=False,kde=True, color='Aquamarine')
    plt.xlabel('DFA')

    plt.subplot(5,5,19)
    sns.distplot(pks_df["spread1"],hist=False,kde=True, color='lightblue')
    plt.xlabel('spread1')

    plt.subplot(5,5,20)
    sns.distplot(pks_df["spread2"],hist=False,kde=True, color='lightgreen')
    plt.xlabel('spread2')

    plt.subplot(5,5,21)
    sns.distplot(pks_df["D2"],hist=False,kde=True, color='pink')
    plt.xlabel('D2')

    plt.subplot(5,5,22)
    sns.distplot(pks_df["PPE"],hist=False,kde=True, color='gray')
    plt.xlabel('PPE')


    plt.subplot(5,5,23)
    sns.countplot(pks_df["status"], color='Green')
    plt.xlabel('status')


    plt.show()
distributionPlot(pks_df)
#Plots to see the distribution of the features individually
def outlierPlot(pks_df):
    plt.figure(figsize= (20,15))
    plt.subplot(5,5,1)
    sns.boxplot(pks_df["MDVP:Fo(Hz)"],orient="v", color='lightblue')
    plt.xlabel('MDVP:Fo(Hz)')

    plt.subplot(5,5,2)
    sns.boxplot(pks_df["MDVP:Fhi(Hz)"],orient="v", color='lightgreen')
    plt.xlabel('MDVP:Fhi(Hz)')

    plt.subplot(5,5,3)
    sns.boxplot(pks_df["MDVP:Flo(Hz)"], orient="v",color='pink')
    plt.xlabel('MDVP:Flo(Hz)')

    plt.subplot(5,5,4)
    sns.boxplot(pks_df["MDVP:Jitter(%)"],orient="v", color='gray')
    plt.xlabel('MDVP:Jitter(%)')

    plt.subplot(5,5,5)
    sns.boxplot(pks_df["MDVP:Jitter(Abs)"],orient="v", color='cyan')
    plt.xlabel('MDVP:Jitter(Abs)')

    plt.subplot(5,5,6)
    sns.boxplot(pks_df["MDVP:RAP"],orient="v", color='Aquamarine')
    plt.xlabel('MDVP:RAP')

    plt.subplot(5,5,7)
    sns.boxplot(pks_df["MDVP:PPQ"],orient="v", color='lightblue')
    plt.xlabel('MDVP:PPQ')

    plt.subplot(5,5,8)
    sns.boxplot(pks_df["Jitter:DDP"],orient="v", color='lightgreen')
    plt.xlabel('Jitter:DDP')

    plt.subplot(5,5,9)
    sns.boxplot(pks_df["MDVP:Shimmer"],orient="v", color='pink')
    plt.xlabel('MDVP:Shimmer')

    plt.subplot(5,5,10)
    sns.boxplot(pks_df["MDVP:Shimmer(dB)"],orient="v", color='gray')
    plt.xlabel('MDVP:Shimmer(dB)')

    plt.subplot(5,5,11)
    sns.boxplot(pks_df["Shimmer:APQ3"],orient="v", color='cyan')
    plt.xlabel('Shimmer:APQ3')

    plt.subplot(5,5,12)
    sns.boxplot(pks_df["Shimmer:APQ5"],orient="v", color='Aquamarine')
    plt.xlabel('Shimmer:APQ5')

    plt.subplot(5,5,13)
    sns.boxplot(pks_df["MDVP:APQ"],orient="v", color='lightblue')
    plt.xlabel('MDVP:APQ')

    plt.subplot(5,5,14)
    sns.boxplot(pks_df["Shimmer:DDA"],orient="v", color='lightgreen')
    plt.xlabel('Shimmer:DDA')

    plt.subplot(5,5,15)
    sns.boxplot(pks_df["NHR"],orient="v", color='pink')
    plt.xlabel('NHR')

    plt.subplot(5,5,16)
    sns.boxplot(pks_df["HNR"],orient="v", color='gray')
    plt.xlabel('HNR')

    plt.subplot(5,5,17)
    sns.boxplot(pks_df["RPDE"],orient="v", color='cyan')
    plt.xlabel('RPDE')

    plt.subplot(5,5,18)
    sns.boxplot(pks_df["DFA"],orient="v", color='Aquamarine')
    plt.xlabel('DFA')

    plt.subplot(5,5,19)
    sns.boxplot(pks_df["spread1"],orient="v", color='lightblue')
    plt.xlabel('spread1')

    plt.subplot(5,5,20)
    sns.boxplot(pks_df["spread2"],orient="v", color='lightgreen')
    plt.xlabel('spread2')

    plt.subplot(5,5,21)
    sns.boxplot(pks_df["D2"],orient="v", color='pink')
    plt.xlabel('D2')

    plt.subplot(5,5,22)
    sns.boxplot(pks_df["PPE"],orient="v", color='gray')
    plt.xlabel('PPE')



    plt.show()

outlierPlot(pks_df)
#Checking the pair plot between each feature
sns.pairplot(pks_df)  #pairplot
plt.show()
#person coefficient 
pks_df.corr()
#Correlation
corr = pks_df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(corr,mask=mask,square=True,linewidths=2.5,cmap="viridis",annot=True)
sns.countplot(Target)
plt.show()
print("number of parkinson people in the dataset ",len(pks_df.loc[pks_df["status"]==1]))
print("number of Healthy people in the dataset ",len(pks_df.loc[pks_df["status"]==0]))
pks_df.info()
updated_cols = list(pks_df.columns)
updated_cols.remove('name')
updated_cols.remove('status')
for column in updated_cols:
    print(column," : ", len(pks_df.loc[pks_df[column]<0]))
pks_df["spread1"]
def outliearTreat(df):
    '''
    This function is to treat outliers in the dataframe
    input : dataframe that need to be treated
    output: dataframe that treated with outlier capping treatment
    '''
    cols = list(df.columns)
    cols.remove('name')
    cols.remove('status')
    for columnName in cols:
        Q1 = df[columnName].quantile(0.25)
        Q3 = df[columnName].quantile(0.75)
        IQR = Q3 - Q1
        whisker = Q1 + 1.5 * IQR
        LowerBound = Q1- 1.5 * IQR
        df[columnName] = df[columnName].apply(lambda x : whisker if x>whisker else x)
        df[columnName] = df[columnName].apply(lambda x : LowerBound if x<LowerBound else x)
    return df
outliearTreat(pks_df)
pks_df.skew()
outlierPlot(pks_df)
distributionPlot(pks_df)
#Standardise the numerical columns
def standardScalar(df):
    '''
    This function is to treat outliers in the dataframe
    input : dataframe that need to be treated
    output: dataframe that treated with outlier capping treatment
    '''
    cols = list(df.columns)
    cols.remove('name')
    cols.remove('status')
    # the scaler object (model)
    scalar = StandardScaler()
    for columnName in cols:
        # fit and transform the data
        df[columnName] = scalar.fit_transform(df[[columnName]])
    return df
standardScalar(pks_df)
X = pks_df.drop(['status','name'],axis=1)     # Predictor feature columns 
Y = Target   # Predicted class (1, 0) 

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
# 1 is just any random seed number

x_train.head()
print("{0:0.2f}% data is in training set".format((len(x_train)/len(pks_df.index)) * 100))
print("{0:0.2f}% data is in test set".format((len(x_test)/len(Target.index)) * 100))
#Dropping above mentioned columns
x_test = x_test[X.columns.difference(['MDVP:Jitter(Abs)','MDVP:RAP','Jitter:DDP','spread1','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ'])]
model_stats = {'model_name':[],'train_accuracy':[],'test_accuracy':[],'f1_score':[],'roc_score':[]}
def logistReg(x_train,y_train,solver="liblinear"):
    # Fit the model on train
    model = LogisticRegression(solver=solver)
    model.fit(x_train, y_train)
    #predict on test
    y_predict = model.predict(x_test)
    y_predictprob = model.predict_proba(x_test)

    coef_df = pd.DataFrame(model.coef_,columns=list(x_train.columns))
    model_stats['model_name'].append("LogisticRegression")
    coef_df['intercept'] = model.intercept_
    model_score = model.score(x_train, y_train)
    print(f"Accuracy of Training Data: {model_score}")
    model_stats['train_accuracy'].append(model_score)
    model_score = model.score(x_test, y_test)
    print(f"Accuracy of Test Data: {model_score}")
    model_stats['test_accuracy'].append(model_score)
    print(coef_df)
    print(metrics.classification_report(y_test,y_predict))
    cm=metrics.confusion_matrix(y_test, y_predict, labels=[1, 0])

    df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                      columns = [i for i in ["Predict 1","Predict 0"]])
    plt.figure(figsize = (8,5))
    sns.heatmap(df_cm, annot=True)
    plt.show()
    print("f1 score", metrics.f1_score(y_test,y_predict))
    print("Auc Roc Score: ",metrics.roc_auc_score(y_test,y_predict))
    model_stats['f1_score'].append(metrics.f1_score(y_test,y_predict))
    model_stats['roc_score'].append(metrics.roc_auc_score(y_test,y_predict))
    return y_predictprob,y_predict
#OverSampling the minority to get the better results
xtrain_resampled, ytrain_resampled = SMOTE(sampling_strategy=1,random_state=46).fit_resample(x_train[X.columns.difference(['MDVP:Jitter(Abs)','MDVP:RAP','Jitter:DDP','spread1','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ'])],y_train)

y_predProb,y_predict = logistReg(xtrain_resampled,ytrain_resampled)
fprLR, tprLR, threshLR = metrics.roc_curve(y_test, y_predProb[:,1], pos_label=1)
scores =[]
for k in range(1,30):
    NNH = KNeighborsClassifier(n_neighbors = k, weights = 'distance', metric='euclidean' )
    NNH.fit(xtrain_resampled, ytrain_resampled)
    scores.append(NNH.score(x_test, y_test))
plt.plot(range(1,30),scores)
plt.show()
NNH = KNeighborsClassifier(n_neighbors= 3 , weights = 'distance', metric='euclidean' )
NNH.fit(xtrain_resampled, ytrain_resampled)
y_predKnn = NNH.predict(x_test)
print(metrics.classification_report(y_test,y_predKnn))
cm=metrics.confusion_matrix(y_test, y_predKnn, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (8,5))
sns.heatmap(df_cm, annot=True)
plt.show()
NNH_TrainAcc = NNH.score(xtrain_resampled,ytrain_resampled)
NNH_TestAcc = NNH.score(x_test,y_test)
NNH_Roc = metrics.roc_auc_score(y_test,y_predKnn)
NNH_F1 = metrics.f1_score(y_test,y_predKnn)
print(f"Score of Knn Test Data : {NNH_TestAcc}")
print(f'Score of Knn Train Data : {NNH_TrainAcc}')
print(f"Roc AUC score of KNN : {NNH_Roc}")
print(f"f1 score of KNN : {NNH_F1}\n")
model_stats['model_name'].append("KNN")
model_stats['train_accuracy'].append(NNH_TrainAcc)
model_stats['test_accuracy'].append(NNH_TestAcc)
model_stats['f1_score'].append(NNH_F1)
model_stats['roc_score'].append(NNH_Roc)
pred_prob_NNH = NNH.predict_proba(x_test)
fprNNH, tprNNH, threshNNH = metrics.roc_curve(y_test, pred_prob_NNH[:,1], pos_label=1)
NBmodel = GaussianNB()
NBmodel.fit(xtrain_resampled,ytrain_resampled)
y_NBPred = NBmodel.predict(x_test)
print(metrics.classification_report(y_test,y_NBPred))
cm=metrics.confusion_matrix(y_test, y_NBPred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (8,5))
sns.heatmap(df_cm, annot=True)
plt.show()
NB_TrainAcc = NBmodel.score(xtrain_resampled,ytrain_resampled)
NB_TestAcc = NBmodel.score(x_test,y_test)
NB_Roc = metrics.roc_auc_score(y_test,y_NBPred)
NB_F1 = metrics.f1_score(y_test,y_NBPred)
print(f"Score of NB Test Data : {NB_TestAcc}")
print(f'Score of NB Train Data : {NB_TrainAcc}')
print(f"Roc AUC score of NB : {NB_Roc}")
print(f"f1 score of NB : {NB_F1}\n")
model_stats['model_name'].append("Naive Bayes")
model_stats['train_accuracy'].append(NB_TrainAcc)
model_stats['test_accuracy'].append(NB_TestAcc)
model_stats['f1_score'].append(NB_F1)
model_stats['roc_score'].append(NB_Roc)
pred_prob_NB = NBmodel.predict_proba(x_test)
fprNB, tprNB, threshNB = metrics.roc_curve(y_test, pred_prob_NB[:,1], pos_label=1)
svc_model = SVC(kernel='poly',probability=True)
svc_model.fit(xtrain_resampled,ytrain_resampled)
y_svmPred = svc_model.predict(x_test)
print(metrics.classification_report(y_test,y_svmPred))
cm=metrics.confusion_matrix(y_test, y_svmPred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (8,5))
sns.heatmap(df_cm, annot=True)
plt.show()
SVM_TrainAcc = svc_model.score(xtrain_resampled,ytrain_resampled)
SVM_TestAcc = svc_model.score(x_test,y_test)
SVM_F1 = metrics.f1_score(y_test,y_svmPred)
SVM_Roc = metrics.roc_auc_score(y_test,y_svmPred)
print(f"Score of svm Test Data : {SVM_TestAcc}")
print(f'Score of svm Train Data : {SVM_TrainAcc}')
print(f"Roc AUC score of svm : {SVM_Roc}")
print(f"f1 score of svm : {SVM_F1}\n")
model_stats['model_name'].append("SVM")
model_stats['train_accuracy'].append(SVM_TrainAcc)
model_stats['test_accuracy'].append(SVM_TestAcc)
model_stats['f1_score'].append(SVM_F1)
model_stats['roc_score'].append(SVM_Roc)
pd.DataFrame(model_stats)
pred_prob_svm = svc_model.predict_proba(x_test)
fprsvm, tprsvm, threshsvm = metrics.roc_curve(y_test, pred_prob_svm[:,1], pos_label=1)
cols = list(pks_df.columns)
cols.remove('name')
cols.remove('status')
clf1 = KNeighborsClassifier(n_neighbors=3)
clf2 = SVC(kernel='poly',probability=True)
clf3 = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=lr)
stck_model = sclf.fit(xtrain_resampled,ytrain_resampled)
y_stckPred = sclf.predict(x_test)
print(metrics.classification_report(y_test,y_stckPred))
cm=metrics.confusion_matrix(y_test, y_stckPred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (8,5))
sns.heatmap(df_cm, annot=True)
plt.show()
Stack_TrainAcc = stck_model.score(xtrain_resampled,ytrain_resampled)
Stack_TestAcc = stck_model.score(x_test,y_test)
Stack_Roc = metrics.roc_auc_score(y_test,y_stckPred)
Stack_F1 = metrics.f1_score(y_test,y_stckPred)
print(f"Score of stacking classifier Test Data : {Stack_TestAcc}")
print(f'Score of stacking classifier Train Data : {Stack_TrainAcc}')
print(f"Roc AUC score of stacking classifier : {Stack_Roc}")
print(f"f1 score of stacking classifier : {Stack_F1}\n")
model_stats['model_name'].append("StackingClassifier")
model_stats['train_accuracy'].append(Stack_TrainAcc)
model_stats['test_accuracy'].append(Stack_TestAcc)
model_stats['f1_score'].append(Stack_F1)
model_stats['roc_score'].append(Stack_Roc)
dTree = DecisionTreeClassifier(criterion = 'gini', random_state=1)
dTree.fit(xtrain_resampled, ytrain_resampled)
print(dTree.score(xtrain_resampled, ytrain_resampled))
print(dTree.score(x_test, y_test))
## Reducing over fitting (Regularization)
dTreeR = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, max_leaf_nodes=4, random_state=1)
dTreeR.fit(xtrain_resampled, ytrain_resampled)
print(dTreeR.score(xtrain_resampled, ytrain_resampled))
print(dTreeR.score(x_test, y_test))
# importance of features in the tree building ( The importance of a feature is computed as the 
#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )

print (pd.DataFrame(dTreeR.feature_importances_, columns = ["Imp"], index = xtrain_resampled.columns))
from sklearn.ensemble import BaggingClassifier

bgcl = BaggingClassifier(base_estimator=dTree, n_estimators=50,random_state=1)
#bgcl = BaggingClassifier(n_estimators=50,random_state=1)

bgcl = bgcl.fit(xtrain_resampled, ytrain_resampled)
y_bgclpred = bgcl.predict(x_test)
print(metrics.classification_report(y_test,y_bgclpred))
cm=metrics.confusion_matrix(y_test, y_bgclpred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (8,5))
sns.heatmap(df_cm, annot=True)
plt.show()
Bag_TrainAcc = bgcl.score(xtrain_resampled,ytrain_resampled)
Bag_TestAcc = bgcl.score(x_test,y_test)
Bag_F1 = metrics.f1_score(y_test,y_bgclpred)
Bag_Roc = metrics.roc_auc_score(y_test,y_bgclpred)
print(f"Score of Bagging classifier Test Data : {Bag_TestAcc}")
print(f'Score of Bagging classifier Train Data : {Bag_TrainAcc}')
print(f"Roc AUC score of Bagging classifier : {Bag_Roc}")
print(f"f1 score of Bagging classifier : {Bag_F1}\n")
model_stats['model_name'].append("Bagging")
model_stats['train_accuracy'].append(Bag_TrainAcc)
model_stats['test_accuracy'].append(Bag_TestAcc)
model_stats['f1_score'].append(Bag_F1)
model_stats['roc_score'].append(Bag_Roc)
# param_grid = {'n_estimators': [50, 60, 70, 80, 90, 100],
#                 'max_features': [5,6,7,8,9,10,11,12,15,16,18,20]}

# gbcl = GridSearchCV(estimator=GradientBoostingClassifier(), 
#                         param_grid=param_grid, 
#                         cv=10,
#                         verbose=True, n_jobs=-1)


# gbcl = gbcl.fit(xtrain_resampled, ytrain_resampled)
# y_gbclpred = gbcl.predict(x_test)
# gbcl.best_params_
from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(n_estimators = 100,max_features = 7,random_state=1)
gbcl = gbcl.fit(xtrain_resampled, ytrain_resampled)
y_gbclpred = gbcl.predict(x_test)
print(metrics.classification_report(y_test,y_gbclpred))
cm=metrics.confusion_matrix(y_test, y_gbclpred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (8,5))
sns.heatmap(df_cm, annot=True)
plt.show()
GB_TrainAcc = gbcl.score(xtrain_resampled,ytrain_resampled)
GB_TestAcc = gbcl.score(x_test,y_test)
GB_F1 = metrics.f1_score(y_test,y_gbclpred)
GB_Roc = metrics.roc_auc_score(y_test,y_gbclpred)
print(f"Score of GradientBoost classifier Test Data : {GB_TestAcc}")
print(f'Score of GradientBoost classifier Train Data : {GB_TrainAcc}')
print(f"Roc AUC score of GradientBoost classifier : {GB_Roc}")
print(f"f1 score of GradientBoost classifier : {GB_F1}\n")
model_stats['model_name'].append("GradientBoosting")
model_stats['train_accuracy'].append(GB_TrainAcc)
model_stats['test_accuracy'].append(GB_TestAcc)
model_stats['f1_score'].append(GB_F1)
model_stats['roc_score'].append(GB_Roc)
# param_grid = {'n_estimators': [50, 60, 70, 80, 90, 100],
#                 'max_features': [5,6,7,8,9,10,11,12,15,16,18,20]}

# rfcl = GridSearchCV(estimator=RandomForestClassifier(), 
#                         param_grid=param_grid, 
#                         cv=10,
#                         verbose=True, n_jobs=-1)

# rfcl = rfcl.fit(xtrain_resampled, ytrain_resampled)
# y_rfclpred = rfcl.predict(x_test)
#rfcl.best_params_
rfcl = RandomForestClassifier(n_estimators = 100, random_state=1,max_features=8)
rfcl = rfcl.fit(xtrain_resampled, ytrain_resampled)
y_rfclpred = rfcl.predict(x_test)
print(metrics.classification_report(y_test,y_rfclpred))
cm=metrics.confusion_matrix(y_test, y_rfclpred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (8,5))
sns.heatmap(df_cm, annot=True)
plt.show()
RF_TrainAcc = rfcl.score(xtrain_resampled,ytrain_resampled)
RF_TestAcc = rfcl.score(x_test,y_test)
RF_F1 = metrics.f1_score(y_test,y_rfclpred)
RF_Roc = metrics.roc_auc_score(y_test,y_rfclpred)
print(f"Score of RandomForest classifier Test Data : {RF_TestAcc}")
print(f'Score of RandomForest classifier Train Data : {RF_TrainAcc}')
print(f"Roc AUC score of RandomForest classifier : {RF_Roc}")
print(f"f1 score of RandomForest classifier : {RF_F1}\n")
model_stats['model_name'].append("RandomForest")
model_stats['train_accuracy'].append(RF_TrainAcc)
model_stats['test_accuracy'].append(RF_TestAcc)
model_stats['f1_score'].append(RF_F1)
model_stats['roc_score'].append(RF_Roc)
# param_grid = {'n_estimators': [50, 60, 70, 80, 90, 100,150,200],
#               'max_features': [5,6,7,8,9,10,11,12,15,16,18,20],
#               'learning_rate':[10 ** x for x in range(-3,2)],
#               'max_depth': [x for x in range(1,10)],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005]
#              }

# xgb_estimator = GridSearchCV(estimator=XGBClassifier(), 
#                         param_grid=param_grid, 
#                         cv=10,
#                         verbose=True, n_jobs=-1)


# xgb_estimator = xgb_estimator.fit(xtrain_resampled, ytrain_resampled)
# #y_gbclpred = xgb_estimator.predict(x_test)
# xgb_estimator.best_params_
xgb_estimator = XGBClassifier( learning_rate=1,
                               n_estimators=50,
                               max_depth=2,
                               #min_child_weight=1,
                               gamma=0.0001,
                               #subsample=0.8,
                               #colsample_bytree=0.8,
                               #n_jobs=-1,
                               #reg_alpa=1,
                              max_features= 5,
                               #scale_pos_weight=1,
                               random_state=42,
                               verbose=1)
xgb_estimator.fit(xtrain_resampled,ytrain_resampled)
y_xgbclpred = xgb_estimator.predict(x_test)
print(metrics.classification_report(y_test,y_xgbclpred))
cm=metrics.confusion_matrix(y_test, y_xgbclpred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (8,5))
sns.heatmap(df_cm, annot=True)
plt.show()
xgb_TrainAcc = xgb_estimator.score(xtrain_resampled,ytrain_resampled)
xgb_TestAcc = xgb_estimator.score(x_test,y_test)
xgb_F1 = metrics.f1_score(y_test,y_xgbclpred)
xgb_Roc = metrics.roc_auc_score(y_test,y_xgbclpred)
print(f"Score of XGBoost classifier Test Data : {xgb_TestAcc}")
print(f'Score of XGBoost classifier Train Data : {xgb_TrainAcc}')
print(f"Roc AUC score of XGBoost classifier : {xgb_Roc}")
print(f"f1 score of XGBoost classifier : {xgb_F1}\n")
model_stats['model_name'].append("XGBoost")
model_stats['train_accuracy'].append(xgb_TrainAcc)
model_stats['test_accuracy'].append(xgb_TestAcc)
model_stats['f1_score'].append(xgb_F1)
model_stats['roc_score'].append(xgb_Roc)
pargrid_adb = {'n_estimators':[50,100,200,400,600,800],
              'learning_rate':[10 ** x for x in range(-3,2)],
              'base_estimator':[svc_model]}
adbgscv = GridSearchCV(estimator=AdaBoostClassifier(),param_grid = pargrid_adb,cv=5,verbose=True,n_jobs=-1)
adbgscv.fit(xtrain_resampled,ytrain_resampled)
y_adbpred = adbgscv.predict(x_test)
adbgscv.best_estimator_
adbgscv = AdaBoostClassifier(base_estimator=SVC(kernel='poly', probability=True),
                   learning_rate=0.1, n_estimators=600,random_state=2)
adbgscv.fit(xtrain_resampled,ytrain_resampled)
y_adbpred = adbgscv.predict(x_test)
print(metrics.classification_report(y_test,y_adbpred))
cm=metrics.confusion_matrix(y_test, y_adbpred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (8,5))
sns.heatmap(df_cm, annot=True)
plt.show()
adb_TrainAcc = adbgscv.score(xtrain_resampled,ytrain_resampled)
adb_TestAcc = adbgscv.score(x_test,y_test)
adb_F1 = metrics.f1_score(y_test,y_adbpred)
adb_Roc = metrics.roc_auc_score(y_test,y_adbpred)
print(f"Score of AdaBoost classifier Test Data : {adb_TestAcc}")
print(f'Score of AdaBoost classifier Train Data : {adb_TrainAcc}')
print(f"Roc AUC score of AdaBoost classifier : {adb_Roc}")
print(f"f1 score of AdaBoost classifier : {adb_F1}\n")
model_stats['model_name'].append("AdaBoost")
model_stats['train_accuracy'].append(adb_TrainAcc)
model_stats['test_accuracy'].append(adb_TestAcc)
model_stats['f1_score'].append(adb_F1)
model_stats['roc_score'].append(adb_Roc)
clf1 = KNeighborsClassifier(n_neighbors= 3 , weights = 'distance', metric='euclidean' )
clf2 = GradientBoostingClassifier(n_estimators = 80,max_features = 8,random_state=1)
clf3 = BaggingClassifier(base_estimator=dTree, n_estimators=50,random_state=1)
clf4 = AdaBoostClassifier(base_estimator=SVC(kernel='poly', probability=True),
                   learning_rate=0.1, n_estimators=600,random_state=2)

votingclf = VotingClassifier(estimators=[ ('knn',clf1),('grb', clf2),('bgg', clf3),('adb', clf4)], voting='hard')
votingclf = votingclf.fit(xtrain_resampled,ytrain_resampled)
y_votclpred = votingclf.predict(x_test)
print(metrics.classification_report(y_test,y_votclpred))
cm=metrics.confusion_matrix(y_test, y_votclpred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (8,5))
sns.heatmap(df_cm, annot=True)
plt.show()
vtc_TrainAcc = votingclf.score(xtrain_resampled,ytrain_resampled)
vtc_TestAcc = votingclf.score(x_test,y_test)
vtc_F1 = metrics.f1_score(y_test,y_votclpred)
vtc_Roc = metrics.roc_auc_score(y_test,y_votclpred)
print(f"Score of stacking classifier Test Data : {vtc_TestAcc}")
print(f'Score of stacking classifier Train Data : {vtc_TrainAcc}')
print(f"Roc AUC score of stacking classifier : {vtc_Roc}")
print(f"f1 score of stacking classifier : {vtc_F1}\n")
model_stats['model_name'].append("VotingClf")
model_stats['train_accuracy'].append(vtc_TrainAcc)
model_stats['test_accuracy'].append(vtc_TestAcc)
model_stats['f1_score'].append(vtc_F1)
model_stats['roc_score'].append(vtc_Roc)
results_df = pd.DataFrame(model_stats)
results_df
plt.figure(figsize = (18,10))
sns.barplot(x ="test_accuracy",y = "model_name",data=results_df)
plt.title("Model vs TestAccuracy")
plt.show()
plt.figure(figsize = (18,10))
sns.barplot(x ="f1_score",y = "model_name",data=results_df)
plt.title("Model vs F1Score")
plt.show()
plt.figure(figsize = (18,10))
sns.barplot(x ="roc_score",y = "model_name",data=results_df)
plt.title("Model vs RocScore")
plt.show()