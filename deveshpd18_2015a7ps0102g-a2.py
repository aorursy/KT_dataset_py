import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data_orig = pd.read_csv('../input/train.csv' , sep=',')
data = data_orig

data_test_orig = pd.read_csv('../input/test.csv' , sep=',')
data_test = data_test_orig
data.head()
data_test.head()
data.shape
data_test.shape
data.info()
data_test.info()
data.shape
#data=data.drop(data.index[10000:99999])
data.shape
import seaborn as sns
f, ax = plt.subplots(figsize=(15, 12))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);
data = data.drop(['ID','Worker Class','Enrolled','MIC','MOC','MLU','Reason','Area','State','MSA','REG','MOVE','Live','PREV','Teen','COB FATHER','COB MOTHER','COB SELF','Fill','Hispanic','Detailed'], 1)
data_test = data_test.drop(['ID','Worker Class','Enrolled','MIC','MOC','MLU','Reason','Area','State','MSA','REG','MOVE','Live','PREV','Teen','COB FATHER','COB MOTHER','COB SELF','Fill','Hispanic','Detailed'], 1)
data = data.drop(['OC','Timely Income','Weight','Own/Self','WorkingPeriod'], 1)
data_test = data_test.drop(['OC','Timely Income','Weight','Own/Self','WorkingPeriod'], 1)
data['Class'].value_counts()
# from imblearn.over_sampling import RandomOverSampler

# ros = RandomOverSampler(random_state=42)
# data = ros.fit_resample(data)
data.shape
data_test.shape
y=data['Class']
X=data.drop(['Class'],axis=1)
X = pd.get_dummies(X, columns=['Schooling','Married_Life','Cast','Sex','Full/Part','Tax Status','Summary','Citizen'])
X_test = pd.get_dummies(data_test, columns=['Schooling','Married_Life','Cast','Sex','Full/Part','Tax Status','Summary','Citizen'])
X.head()
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X,y = ros.fit_resample(X,y)

from collections import Counter

Counter(y)


X_test.head()
X_test.shape
X.shape
for column in X_test:
    print(column)
# for column in X:
#     print(column)
#X.columns
#X_test.columns
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X_train)
X_train = pd.DataFrame(np_scaled)
np_scaled_val = min_max_scaler.transform(X_val)
X_val = pd.DataFrame(np_scaled_val)
X_train.head()
from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X_test)
X_test = pd.DataFrame(np_scaled)
#np_scaled_val = min_max_scaler.transform(X_val)
#X_val = pd.DataFrame(np_scaled_val)
#X_train.head()
X_test.head()
np.random.seed(42)
from sklearn.naive_bayes import GaussianNB as NB
nb = NB()
nb.fit(X_train,y_train)
nb.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_NB = nb.predict(X_val)
print(confusion_matrix(y_val, y_pred_NB))
print(classification_report(y_val, y_pred_NB))
y_pred_test_NB = nb.predict(X_test)
y_pred_test_NB
z = y_pred_test_NB.tolist()
res1 = pd.DataFrame(z)
data_z = pd.read_csv('../input/test.csv' , sep=',')
final = pd.concat([data_z['ID'], res1], axis=1).reindex()
final = final.rename(columns={0: "Class"})
final['Class'] = final.Class.astype(int)
final.head(100)
#res1.head(100)
#final=final.drop(final.index[0:175])
final.to_csv('submission.csv', index = False,  float_format='%.f')
data_z.head()

# from sklearn.neighbors import KNeighborsClassifier
# train_acc = []
# test_acc = []
# for i in range(1,15):
    
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train,y_train)
#     acc_train = knn.score(X_train,y_train)
#     train_acc.append(acc_train)
#     acc_test = knn.score(X_val,y_val)
#     test_acc.append(acc_test)
# plt.figure(figsize=(10,6))
# train_score,=plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed', marker='o',
#          markerfacecolor='green', markersize=5)
# test_score,=plt.plot(range(1,15),test_acc,color='red',linestyle='dashed',  marker='o',
#          markerfacecolor='blue', markersize=5)
# plt.legend( [train_score, test_score],["Train Accuracy", "Test Accuracy"])
# plt.title('Accuracy vs K neighbors')
# plt.xlabel('K neighbors')
# plt.ylabel('Accuracy')
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train,y_train)
# knn.score(X_val,y_val)
# y_pred_KNN = knn.predict(X_val)
# cfm = confusion_matrix(y_val, y_pred_KNN, labels = [0,1,2])
# print(cfm)
# #entry (i,j) in a confusion matrix is the number of observations actually in group i, but predicted to be in group j.

# print"True Positives of Class 0: ", cfm[0][0]
# print"False Positives of Class 0 wrt Class 1: ", cfm[1][0] # Predicted as 0 but actually in 1 
# print"False Positives of Class 0 wrt Class 2: ", cfm[2][0]
# print"False Negatives of Class 0 wrt Class 1: ", cfm[0][1] # Precited as 1 but actually in 0
# print"False Negatives of Class 0 wrt Class 2: ", cfm[0][2]
# print(classification_report(y_val, y_pred_KNN))
# # Precision of class 0: Out of all those that you predicted as 0, how many were actually 0
# # Recall of Class 0: Out of all those that were actually 0, how many you predicted to be 0
# # micro avg = (Total TP)/(Total TP+FP)
# # macro avg = unweighted mean of scores of class 0,1,2
# from sklearn import preprocessing
# #Performing Min_Max Normalization
# min_max_scaler2 = preprocessing.MinMaxScaler()
# np_scaled_full = min_max_scaler2.fit_transform(X)
# X_N = pd.DataFrame(np_scaled_full)
# from sklearn.model_selection import cross_validate
# from sklearn.metrics import make_scorer
# from sklearn.metrics import f1_score

# scorer_f1 = make_scorer(f1_score, average = 'micro')

# cv_results = cross_validate(knn, X_N, y, cv=10, scoring=(scorer_f1), return_train_score=True)
# print cv_results.keys()
# print"Train Accuracy for 3 folds= ",np.mean(cv_results['train_score'])
# print"Validation Accuracy for 3 folds = ",np.mean(cv_results['test_score'])
# y_pred_test_KNN = knn.predict(X_test)
# y_pred_test_KNN
# z_KNN = y_pred_test_KNN.tolist()
# res1_KNN = pd.DataFrame(z_KNN)
# final_KNN = pd.concat([data_test_orig['ID'], res1], axis=1).reindex()
# final_KNN = final_KNN.rename(columns={0: "Class"})
# final_KNN['Class'] = final_KNN.Class.astype(int)
# final_KNN.head(100)
# #res1.head(100)
# final_KNN.to_csv('submission_KNN.csv', index = False,  float_format='%.f')
# from sklearn.linear_model import LogisticRegression
# lg = LogisticRegression(solver = 'liblinear', C = 1, multi_class = 'ovr', random_state = 42)
# lg.fit(X_train,y_train)
# lg.score(X_val,y_val)
# lg = LogisticRegression(solver = 'lbfgs', C = 8, multi_class = 'multinomial', random_state = 42)
# lg.fit(X_train,y_train)
# lg.score(X_val,y_val)
# y_pred_LR = lg.predict(X_val)
# print(confusion_matrix(y_val, y_pred_LR))
# print(classification_report(y_val, y_pred_LR))
# y_pred_test_LR = lg.predict(X_test)
# y_pred_test_LR
# z_LR = y_pred_test_LR.tolist()
# res1_LR = pd.DataFrame(z_LR)
# final_LR = pd.concat([data_test_orig['ID'], res1], axis=1).reindex()
# final_LR = final_LR.rename(columns={0: "Class"})
# final_LR['Class'] = final_LR.Class.astype(int)
# final_LR.head(100)
# #res1.head(100)
# final_LR.to_csv('submission_LR.csv', index = False,  float_format='%.f')
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import DecisionTreeClassifier

# train_acc = []
# test_acc = []
# for i in range(1,15):
#     dTree = DecisionTreeClassifier(max_depth=i)
#     dTree.fit(X_train,y_train)
#     acc_train = dTree.score(X_train,y_train)
#     train_acc.append(acc_train)
#     acc_test = dTree.score(X_val,y_val)
#     test_acc.append(acc_test)
# plt.figure(figsize=(10,6))
# train_score,=plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed', marker='o',
#          markerfacecolor='green', markersize=5)
# test_score,=plt.plot(range(1,15),test_acc,color='red',linestyle='dashed',  marker='o',
#          markerfacecolor='blue', markersize=5)
# plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
# plt.title('Accuracy vs Max Depth')
# plt.xlabel('Max Depth')
# plt.ylabel('Accuracy')
# from sklearn.tree import DecisionTreeClassifier

# train_acc = []
# test_acc = []
# for i in range(2,30):
#     dTree = DecisionTreeClassifier(max_depth = 9, min_samples_split=i, random_state = 42)
#     dTree.fit(X_train,y_train)
#     acc_train = dTree.score(X_train,y_train)
#     train_acc.append(acc_train)
#     acc_test = dTree.score(X_val,y_val)
#     test_acc.append(acc_test)
# plt.figure(figsize=(10,6))
# train_score,=plt.plot(range(2,30),train_acc,color='blue', linestyle='dashed', marker='o',
#          markerfacecolor='green', markersize=5)
# test_score,=plt.plot(range(2,30),test_acc,color='red',linestyle='dashed',  marker='o',
#          markerfacecolor='blue', markersize=5)
# plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
# plt.title('Accuracy vs min_samples_split')
# plt.xlabel('Max Depth')
# plt.ylabel('Accuracy')
# dTree = DecisionTreeClassifier(max_depth=9, random_state = 42)
# dTree.fit(X_train,y_train)
# dTree.score(X_val,y_val)
# y_pred_DT = dTree.predict(X_val)
# print(confusion_matrix(y_val, y_pred_DT))
# print(classification_report(y_val, y_pred_DT))
# y_pred_test_DT = dTree.predict(X_test)
# y_pred_test_DT
# z_DT = y_pred_test_DT.tolist()
# res1_DT = pd.DataFrame(z_DT)
# final_DT = pd.concat([data_test_orig['ID'], res1], axis=1).reindex()
# final_DT = final_DT.rename(columns={0: "Class"})
# final_DT['Class'] = final_DT.Class.astype(int)
# final_DT.head(100)
# final_DT.to_csv('submission_DT.csv', index = False,  float_format='%.f')
from sklearn.ensemble import RandomForestClassifier
score_train_RF = []
score_test_RF = []
for i in range(1,18,1):
    rf = RandomForestClassifier(n_estimators=i, random_state = 42)
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_val,y_val)
    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=11, random_state = 42)
rf.fit(X_train, y_train)
rf.score(X_val,y_val)
y_pred_RF = rf.predict(X_val)
confusion_matrix(y_val, y_pred_RF)
print(classification_report(y_val, y_pred_RF))
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

rf_temp = RandomForestClassifier(n_estimators = 11)        #Initialize the classifier object

parameters = {'max_depth':[3, 5, 8, 10],'min_samples_split':[2, 3, 4, 5]}    #Dictionary of parameters

scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators = 11, max_depth = 10, min_samples_split = 5)
rf_best.fit(X_train, y_train)
rf_best.score(X_val,y_val)
y_pred_RF_best = rf_best.predict(X_val)
confusion_matrix(y_val, y_pred_RF_best)
print(classification_report(y_val, y_pred_RF_best))
y_pred_test_RF = rf_best.predict(X_test)
y_pred_test_RF
z_RF = y_pred_test_RF.tolist()
res1 = pd.DataFrame(z_RF)
final_RF = pd.concat([data_test_orig['ID'], res1], axis=1).reindex()
final_RF = final_RF.rename(columns={0: "Class"})
final_RF['Class'] = final_RF.Class.astype(int)
final_RF.head(100)
#res1.head(100)
final_RF.to_csv('submission_RF.csv', index = False,  float_format='%.f')
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html='<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(final_RF)