import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import os,time
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
%matplotlib inline
datasets = pd.read_csv("../input/heart-disease-uci/heart.csv")
all_models=[]
all_scores=[]
confusion_matrix_info=[]
def split_data(datasets):
    X= datasets.loc[:,datasets.columns!="target"]
    Y = datasets.loc[:,datasets.columns=="target"]
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=0)
    return X_train,X_test,Y_train,Y_test
#  对特征中非连续型数值(cp,slope,thal)特征进行处理
def process_con_value(datasets):
    thal_dummies=pd.get_dummies(datasets.thal,prefix="thal",prefix_sep="_")
    ca_dummies=pd.get_dummies(datasets.ca,prefix="ca",prefix_sep="_")
    slope_dummies=pd.get_dummies(datasets.slope,prefix="slope",prefix_sep="_")
    cp_dummies=pd.get_dummies(datasets.cp,prefix="cp",prefix_sep="_")
    datasets=pd.concat([datasets,thal_dummies,ca_dummies,slope_dummies,cp_dummies],axis=1)
    datasets.drop(["thal","ca","cp","slope"],inplace=True,axis=1)
    return datasets
from sklearn.preprocessing import StandardScaler

def StandardizeData():
    after_pro_con_v = process_con_value(datasets)
    X_train,X_test,Y_train,Y_test=split_data(after_pro_con_v)
    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    X_train = standardScaler.transform(X_train)
    X_test = standardScaler.transform(X_test) 
    return X_train,X_test,Y_train,Y_test
# 绘制混淆矩阵函数
from sklearn.metrics import confusion_matrix,recall_score
def plot_confusion_metrix(y_true,y_pre,threshold,algorithm):
    y_pre = np.where(y_pre[:,-1]>threshold,1,0)
    cnf = confusion_matrix(y_true,y_pre,labels=[0,1])
    class_names = [0,1]
    fig,ax = plt.subplots(figsize=(6,4))
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks,class_names)
    plt.yticks(tick_marks,class_names)
    # 收集各模型混淆矩阵
    confusion_matrix_info.append((algorithm,cnf))
    #create a heat map
    sns.heatmap(pd.DataFrame(cnf), annot = True, cmap = plt.cm.Blues,fmt = 'g',cbar=False)
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title(algorithm)
    plt.show()
# 模型分值的综合评估
from sklearn.metrics import classification_report
def print_evaluate_values(Y_test,Y_pre,threshold):
    print(classification_report(Y_test,np.where(Y_pre[:,-1]>threshold,1,0)))

# 绘制roc ks曲线
from sklearn.metrics import roc_auc_score,roc_curve,auc
def plot_roc_ks(Y_test,pre_result):
    fpr, tpr, thresholds =roc_curve(Y_test,pre_result[:,1],pos_label=1)
    df=pd.DataFrame({"tpr":tpr,"fpr":fpr,"th":thresholds})
    df["diff"] =np.abs( df["tpr"] - df["fpr"])
    ks = np.max(df["diff"])
    threshold = df.loc[np.argmax(df["diff"]),"th"]
    #画ROC曲线
    fig = plt.figure()
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    #画ks曲线
    plt.plot(tpr,label="goof")
    plt.plot(fpr,label="bad")
    plt.plot(np.abs(tpr-fpr),label="diff")
    auc_value = auc(fpr,tpr)
    all_scores.append(auc_value)
    print(f"ks : {ks}")
    print(f"auc : {auc_value}")
    print(f"threshold : {threshold}")

    # 标记ks
    x = np.argwhere(abs(fpr-tpr) == ks)[0, 0]
    plt.plot((x, x), (0, ks), label='ks - {:.2f}'.format(ks), color='r', marker='o', markerfacecolor='r', markersize=5)
    plt.scatter((x, x), (0, ks), color='r')
    plt.legend()
    plt.show()
    return threshold
def model_evaluate(Y_true,Y_pre,algorithm):
    all_models.append(algorithm)
    threshold=plot_roc_ks(Y_true,Y_pre)
    print_evaluate_values(Y_true,Y_pre,threshold)
    plot_confusion_metrix(Y_true,Y_pre,threshold,algorithm)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

X_train,X_test,Y_train,Y_test = StandardizeData()
lr=LogisticRegression()
lr_params={"C":[0.01,0.1,1.0,10],"penalty":["l2"],"solver":["lbfgs","liblinear"],}# max_iter
glr = GridSearchCV(lr,param_grid=lr_params,cv=3,scoring="roc_auc",n_jobs=-1)
glr.fit(X_train,Y_train.values.ravel())
print("模型的最优参数：",glr.best_params_)
print("最优模型分数：",glr.best_score_)
print("最优模型对象：",glr.best_estimator_)
lr = glr.best_estimator_
pre_lr=lr.predict_proba(X_test)
model_evaluate(Y_test,pre_lr,"LogisticRegression")
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# X_train,X_test,Y_train,Y_test = split_data(process_con_value(datasets))
X_train,X_test,Y_train,Y_test = StandardizeData()
dt_params={"max_depth":[3,4,5,6,7,8],"min_samples_split":[10,13,15,17,20],"criterion":['gini',"entropy"],"random_state":[12]} # 
dt=DecisionTreeClassifier()
dt_gs = GridSearchCV(estimator=dt,param_grid=dt_params,cv=4,scoring="roc_auc")
dt_gs.fit(X_train,Y_train.values.ravel())
print("模型的最优参数：",dt_gs.best_params_)
print("最优模型分数：",dt_gs.best_score_)
print("最优模型对象：",dt_gs.best_estimator_)
new_dt = dt_gs.best_estimator_
pre_dt = new_dt.predict_proba(X_test)
model_evaluate(Y_test,pre_dt,"DecisionTree")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

X_train,X_test,Y_train,Y_test = StandardizeData()
knc_params = {"n_neighbors":[3,5,7,9,11,13],"weights":["uniform","distance"]}
knc= KNeighborsClassifier()
knc_gs = GridSearchCV(knc,cv=3,scoring="roc_auc",param_grid=knc_params)
knc_gs.fit(X_train,Y_train.values[:,0])
print("模型的最优参数：",knc_gs.best_params_)
print("最优模型分数：",knc_gs.best_score_)
new_knc = knc_gs.best_estimator_
pre_knn = new_knc.predict_proba(X_test)
model_evaluate(Y_test,pre_knn,"KNN")
from sklearn.naive_bayes import GaussianNB
X_train,X_test,Y_train,Y_test = split_data(datasets)
nb=GaussianNB()
nb.fit(X_train,Y_train.values.ravel())
pre_nb = nb.predict_proba(X_test)
model_evaluate(Y_test,pre_nb,"NaiveBayes")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

X_train,X_test,Y_train,Y_test = split_data(datasets)

rf=RandomForestClassifier(random_state=2)
rf_params = {"n_estimators":[5,7,9,11],"criterion":["gini","entropy"],"max_depth":[3,5,7],"min_samples_split":[5,10,15],"oob_score":[False]}
rf_gsv = GridSearchCV(rf,cv=5,scoring="roc_auc",param_grid=rf_params)
rf_gsv.fit(X_train,Y_train.values.ravel())
print("模型的最优参数：",rf_gsv.best_params_)
print("最优模型分数：",rf_gsv.best_score_)
print("最优模型对象：",rf_gsv.best_estimator_)
new_rf = rf_gsv.best_estimator_
pre_rf = new_rf.predict_proba(X_test)
importtance = new_rf.feature_importances_
f, ax= plt.subplots(figsize = (8,6))
sns.barplot(x=new_rf.feature_importances_,y=X_train.columns,)
ax.set_yticklabels(ax.get_yticklabels(),fontsize=15)
plt.title("Feature Importance")
model_evaluate(Y_test,pre_rf,"RandomForest")
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
X_train,X_test,Y_train,Y_test = StandardizeData()
svm_params = [
        {"kernel":["rbf"],"C":[1,10,0.1,0.01,0.001],"gamma":[0.1,0.2,0.4,0.6,0.8]},
    {"kernel":["poly"],"C":[1,10,0.1,0.01,0.001],"degree":[1,3,5,7,9]},
    {"kernel":["sigmoid"],"C":[1,10,0.1,0.01,0.001],"gamma":[1,2,3,4],"coef0":[0.2,0.4,0.6,0.8,1]},
             ]
svm_model = SVC(probability=True)
gsc_svm = GridSearchCV(svm_model,cv=4,scoring="roc_auc",param_grid=svm_params)
gsc_svm.fit(X_train,Y_train.values.ravel())
print("模型的最优参数：",gsc_svm.best_params_)
print("最优模型分数：",gsc_svm.best_score_)
print("最优模型对象：",gsc_svm.best_estimator_)
newSvmModel=gsc_svm.best_estimator_
pre_svm=newSvmModel.predict_proba(X_test)
model_evaluate(Y_test,pre_svm,"SVM")
df_scores = pd.DataFrame({"algorithm":all_models,"score":all_scores})
df_scores.sort_values(by="score",inplace=True,ascending=False)
fig=plt.figure(figsize=(14,10))
g=sns.barplot(data=df_scores,x="score",y="algorithm")
for index, value in enumerate(df_scores.score): 
    g.text(value, index, str(round(value, 4)), fontsize = 12)
plt.figure(figsize=(24,12))
plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)
for i,data in enumerate(confusion_matrix_info):
    plt.subplot(2,3,i+1)
    plt.title(f"{data[0]} Confusion Matrix")
    sns.heatmap(data[1],annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
