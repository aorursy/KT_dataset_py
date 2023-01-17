import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
data_og = pd.read_csv("../input/data-mining-assignment-2/train.csv")
test_og = pd.read_csv("../input/data-mining-assignment-2/test.csv")
data = data_og
x_pred = test_og
data = data.drop(['ID'], axis = 1)
x_pred = x_pred.drop(['ID'],axis = 1)
Y = data['Class']
X = data.drop('Class', axis=1)
X['col2'] = X['col2'].astype('category').cat.codes
X['col11'] = X['col11'].astype('category').cat.codes
X['col37'] = X['col37'].astype('category').cat.codes
X['col44'] = X['col44'].astype('category').cat.codes
X['col56'] = X['col56'].astype('category').cat.codes

x_pred['col2'] = x_pred['col2'].astype('category').cat.codes
x_pred['col11'] = x_pred['col11'].astype('category').cat.codes
x_pred['col37'] = x_pred['col37'].astype('category').cat.codes
x_pred['col44'] = x_pred['col44'].astype('category').cat.codes
x_pred['col56'] = x_pred['col56'].astype('category').cat.codes
rob_scaler = preprocessing.RobustScaler()
rob_scaler.fit(X+x_pred)
X_n = pd.DataFrame(rob_scaler.transform(X))
X_p = pd.DataFrame(rob_scaler.transform(x_pred))
X_train, X_test, y_train, y_test = train_test_split(X_n, Y, test_size=0.15,random_state=42)
# #---------KNN---------#
# train_acc=[]
# test_acc=[]
# for i in range(50,200): # 100 seems to be optimal
#     knn = KNNC(n_neighbors=i)
#     knn.fit(X_train, y_train)
    
#     acc_train = knn.score(X_train,y_train)
#     acc_test = knn.score(X_test, y_test)
    
#     train_acc.append(acc_train)
#     test_acc.append(acc_test)

# plt.figure(figsize=(10,6))
# train_score,=plt.plot(range(50,200),train_acc,color='blue', linestyle='dashed',marker='o',markerfacecolor='green', markersize=5)
# test_score,=plt.plot(range(50,200),test_acc,color='red',linestyle='dashed',marker='o',markerfacecolor='blue', markersize=5)
# plt.legend( [train_score, test_score],["Train Accuracy", "Test Accuracy"])
# plt.title('Accuracy vs K neighbors')
# plt.xlabel('K neighbors')
# plt.ylabel('Accuracy')
# knn = KNNC(n_neighbors=104)
# knn.fit(X_train, y_train)
# ycap = knn.predict(X_test)
# print(classification_report(y_test, ycap))
# y_pred_knn = knn.predict(X_p)
# with open('knn.csv','w') as f:
#     f.write("ID,Class\n")
#     for index, i in enumerate(y_pred_knn):
#         f.write(str(700+index)+ ","+str(i)+"\n")
# #-----Decision tree-----#

# train_acc = []
# test_acc = []
# for i in range(1,10):
#     dTree = DecisionTreeClassifier(max_depth=i) # 6||2 best
#     dTree.fit(X_train,y_train)
#     acc_train = dTree.score(X_train, y_train)
#     train_acc.append(acc_train)
#     acc_test = dTree.score(X_test,y_test)
#     test_acc.append(acc_test)

# plt.figure(figsize=(10,6))
# train_score,=plt.plot(range(1,10),train_acc,color='blue', linestyle='dashed',marker='o',markerfacecolor='green', markersize=5)
# test_score,=plt.plot(range(1,10),test_acc,color='red',linestyle='dashed',marker='o',markerfacecolor='blue', markersize=5)
# plt.legend( [train_score, test_score],["Train Accuracy", "Test Accuracy"])
# plt.title('Accuracy vs Max Depth')
# plt.xlabel('Max Depth')
# plt.ylabel('Accuracy')
# dt = DecisionTreeClassifier(max_depth=6)
# dt.fit(X_n, Y)
# Y_pred_d = dt.predict(X_p)
# with open('dec-tree.csv','w') as f:
#     f.write("ID,Class\n")
#     for index, i in enumerate(Y_pred_d):
#         f.write(str(700+index)+ ","+str(i)+"\n")
# #-----Random Forest-----#

# rf_temp = RandomForestClassifier() #Initialize the classifier object

# n_estimators = [90, 120, 150, 170, 190]
# max_depth = [3,6,9,12,15]
# min_samples_split = [2, 3, 4, 5, 6]
# min_samples_leaf = [1, 2, 4, 5, 6]
# bootstrap = [True, False]

# parameters = {'max_depth':max_depth, 'n_estimators':n_estimators, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf, 'bootstrap':[True]} #Dictionary of parameters
# scorer = make_scorer(f1_score, average = 'micro') #Initialize the scorer using make_scorer
# grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer) #Initialize a GridSearchCV object with above parameters,scorer and classifier
# grid_fit = grid_obj.fit(X_n, Y) #Fit the gridsearch object␣with X_train,y_train
# best_rf = grid_fit.best_estimator_ #Get the best estimator. For this, check documentation of GridSearchCV object
# print(grid_fit.best_params_)
rf_temp = RandomForestClassifier(max_depth = 11, n_estimators=150, min_samples_leaf=6, min_samples_split=6)
rf_temp.fit(X_train,y_train)
print(rf_temp.score(X_test, y_test))
ycap = rf_temp.predict(X_test)
print(classification_report(y_test, ycap))
y_pred_rf = rf_temp.predict(X_p)
# with open('rf.csv','w') as f:
#     f.write("ID,Class\n")
#     for index, i in enumerate(y_pred_rf):
#         f.write(str(700+index)+ ","+str(i)+"\n")
# CHANGE MADE HERE. TO SWITCH FROM OPENING A FILE CREATED BY ME TO CREATING A DF DIRECTLY FROM PREDICTIONS
#
s = np.array([i for i in range(700,1000)])
todf = np.vstack((s,y_pred_rf)).transpose()
print(todf.shape)
df = pd.DataFrame(todf,columns=["ID","Class"])
# #----decision tree bagging-----#
# score_train_bagging = []
# score_test_bagging = []
# for i in range(1,11):
#     bagging = BaggingClassifier(max_samples=.3, max_features=.3)
#     bagging.fit(X_train, y_train)
#     sc_train = bagging.score(X_train,y_train)
#     score_train_bagging.append(sc_train)
#     sc_test = bagging.score(X_test,y_test)
#     score_test_bagging.append(sc_test)

# plt.figure(figsize=(10,6))
# train_score,=plt.plot(range(1,11,1),score_train_bagging,color='blue',linestyle='dashed', marker='o',markerfacecolor='green', markersize=5)
# test_score,=plt.plot(range(1,11,1),score_test_bagging,color='red',linestyle='dashed', marker='o', markerfacecolor='blue', markersize=5)

# plt.legend( [train_score,test_score],["Train Score","Test Score"])
# plt.title('Fig4. Score vs. No. of Trees')
# plt.xlabel('No. of Trees')
# plt.ylabel('Score')
# bagging = BaggingClassifier(max_samples=.1, max_features=.1)
# bagging.fit(X_train,y_train)
# ycap = bagging.predict(X_test)
# print(classification_report(y_test,ycap))
# y_p = bagging.predict(X_p)
# with open('bagging.csv','w') as f:
#     f.write("ID,Class\n")
#     for index, i in enumerate(y_p):
#         f.write(str(700+index)+ ","+str(i)+"\n")
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df)
