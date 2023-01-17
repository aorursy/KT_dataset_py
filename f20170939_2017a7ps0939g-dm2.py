import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
## reading files
train = pd.read_csv('../input/data-mining-assignment-2/train.csv')
test = pd.read_csv('../input/data-mining-assignment-2/test.csv')
train = pd.get_dummies(train, columns=["col11","col37","col44"], drop_first = True)
train = pd.get_dummies(train, columns=["col2","col56"])
test = pd.get_dummies(test, columns=["col11","col37","col44"], drop_first = True)
test = pd.get_dummies(test, columns=["col2","col56"])
## creating a new feature origin
train['origin'] = 0
test['origin'] = 1
training = train.drop('Class',axis=1) #droping target variable
## taking sample from training and test data
training = training.sample(300, random_state=12)
testing = test.sample(300, random_state=11)
## combining random samples
combi = training.append(testing)
y = combi['origin']
combi.drop('origin',axis=1,inplace=True)
## modelling
model = RandomForestClassifier(n_estimators=100, max_depth = 6, min_samples_split = 5)
drop_list = []
for i in combi.columns:
    score = cross_val_score(model,pd.DataFrame(combi[i]),y,cv=2,scoring='roc_auc')
    if (np.mean(score) > 0.8):
        drop_list.append(i)
        print(i,np.mean(score))
drop_list
train_data = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')
train_data.head()
# train_data.info()
df = train_data.copy()
df.drop(['ID'], axis = 1, inplace=True)
null_columns = df.columns[df.isnull().any()]
null_columns
df.select_dtypes(include=['object'])
df_onehot = df.copy()
df_onehot = pd.get_dummies(df_onehot, columns=["col11","col37","col44"], drop_first = True)
df_onehot = pd.get_dummies(df_onehot, columns=["col2","col56"])
# df_onehot.info()
df_onehot.head()
corr = df_onehot.corr()
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
square=True, ax=ax, annot = False)
Y = df_onehot['Class']
X = df_onehot.drop('Class',axis=1)
from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X_N = pd.DataFrame(np_scaled)
# X_N.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_N, Y, test_size=0.20, random_state=42)
y_test.value_counts()
test = pd.read_csv("../input/data-mining-assignment-2/test.csv")
test = pd.get_dummies(test, columns=["col11","col37","col44"], drop_first = True)
test = pd.get_dummies(test, columns=["col2","col56"])
test.drop('ID',axis=1,inplace=True)
# test.head()
from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
test_scaled = min_max_scaler.fit_transform(test)
test_N = pd.DataFrame(test_scaled)
# test_N.head()
rf = RandomForestClassifier(n_estimators=100, max_depth = 6, min_samples_split = 5)
rf.fit(X_train, y_train)
y_pred_RF_best = rf.predict(X_test)
### plotting importances
features = train.drop('Class',axis=1).columns.values
imp = rf.feature_importances_
indices = np.argsort(imp)[::-1][:20]

#plot
plt.figure(figsize=(8,5))
plt.bar(range(len(indices)), imp[indices], color = 'b', align='center')
plt.xticks(range(len(indices)), features[indices], rotation='vertical')
plt.xlim([-1,len(indices)])
plt.show()
drop_list
df_onehot.drop(['col6','col19','col30','col59'],axis = 1, inplace=True)
test.drop(['col6','col19','col30','col59'],axis = 1, inplace=True)
Y = df_onehot['Class']
X = df_onehot.drop('Class',axis=1)
from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X_N = pd.DataFrame(np_scaled)
test_scaled = min_max_scaler.fit_transform(test)
test_N = pd.DataFrame(test_scaled)
test_N.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_N, Y, test_size=0.20, random_state=42)
np.random.seed(42)
from sklearn.neighbors import KNeighborsClassifier

train_acc = []
test_acc = []
for i in range(1,100):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    acc_train = knn.score(X_train,y_train)
    train_acc.append(acc_train)
    acc_test = knn.score(X_test,y_test)
    test_acc.append(acc_test)
plt.figure(figsize=(30,6))
train_score,=plt.plot(range(1,100),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,100),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Accuracy", "Test Accuracy"])
plt.title('Accuracy vs K neighbors')
plt.xlabel('K neighbors',)
plt.ylabel('Accuracy')
plt.xticks(np.arange(1,100, 1.0))
knn = KNeighborsClassifier(n_neighbors=65)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_KNN = knn.predict(X_test)
cfm = confusion_matrix(y_test, y_pred_KNN)
print(cfm)
print(classification_report(y_test, y_pred_KNN))
# Precision of class 0: Out of all those that you predicted as 0, how many were actually 0
# Recall of Class 0: Out of all those that were actually 0, how many you predicted to be 0
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

scorer_f1 = make_scorer(f1_score, average = 'micro')

cv_results = cross_validate(knn, X_N, Y, cv=10, scoring=(scorer_f1), return_train_score=True)
print(cv_results.keys())
print("Train Accuracy for 3 folds= ",np.mean(cv_results['train_score']))
print("Test Accuracy for 3 folds = ",np.mean(cv_results['test_score']))
y_pred_KNN = knn.predict(test_N)
res = pd.DataFrame(np.zeros((1000-700,2)))
 

 
for i in range(700, 1000):
    res[0][i-700]=str(i)
    res[1][i-700]=int(y_pred_KNN[i-700])
 
res[1]=res[1].astype(int)
res[0]=res[0].astype(int)
res
final_sub=pd.DataFrame(res)
final_sub.columns = ['ID','Class']
final_sub
final_sub.to_csv('knn.csv',index=False)
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "knn.csv"):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode())
  payload = b64.decode()
  html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
  html = html.format(payload=payload,title=title,filename=filename)
  return HTML(html)
create_download_link(final_sub)
from sklearn.ensemble import RandomForestClassifier

score_train_RF = []
score_test_RF = []

for i in range(5,20,1):
    rf = RandomForestClassifier(n_estimators = 100, max_depth=i)
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_test,y_test)
    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))
train_score,=plt.plot(range(5,20,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(5,20,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=100, max_depth = 12)
rf.fit(X_train, y_train)
rf.score(X_test,y_test)
rf = RandomForestClassifier(n_estimators=2000, max_depth = 12)
rf.fit(X_train, y_train)
rf.score(X_test,y_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_RF = rf.predict(X_test)
confusion_matrix(y_test, y_pred_RF)
print(classification_report(y_test, y_pred_RF))
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

rf_temp = RandomForestClassifier(n_estimators = 100)        #Initialize the classifier object

parameters = {'max_depth':[6, 12],'min_samples_split':[2, 3, 4, 5]}    #Dictionary of parameters

# parameters = {'n_estimators' : [100, 300, 500, 800, 1200],
# 'max_depth' : [5, 8, 15, 25, 30],
# 'min_samples_split' : [2, 5, 10, 15, 100],
# 'min_samples_leaf' : [1, 2, 5, 10]}

scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators=100, max_depth = 6, min_samples_split = 5)
rf_best.fit(X_train, y_train)
rf_best.score(X_test,y_test)
y_pred_RF_best = rf_best.predict(X_test)
confusion_matrix(y_test, y_pred_RF_best)
print(classification_report(y_test, y_pred_RF_best))
y_pred_RF_best = rf_best.predict(test_N)
y_pred_RF_best
res = pd.DataFrame(np.zeros((1000-700,2)))
 

 
for i in range(700, 1000):
    res[0][i-700]=str(i)
    res[1][i-700]=int(y_pred_RF_best[i-700])
 
res[1]=res[1].astype(int)
res[0]=res[0].astype(int)
res
final_sub=pd.DataFrame(res)
final_sub.columns = ['ID','Class']
final_sub
final_sub.to_csv('rf.csv',index=False)
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "rf.csv"):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode())
  payload = b64.decode()
  html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
  html = html.format(payload=payload,title=title,filename=filename)
  return HTML(html)
create_download_link(final_sub)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

score_train_ada = []
score_test_ada = []

for i in range(5,20,1):
    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=i),n_estimators = 200)
    classifier.fit(X_train, y_train)
    sc_train = classifier.score(X_train,y_train)
    score_train_ada.append(sc_train)
    sc_test = classifier.score(X_test,y_test)
    score_test_ada.append(sc_test)
plt.figure(figsize=(10,6))
train_score,=plt.plot(range(5,20,1),score_train_ada,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(5,20,1),score_test_ada,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Score')
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators = 200)
classifier.fit(X_train, y_train)

classifier.score(X_test,y_test)
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators = 2000)
classifier.fit(X_train, y_train)

classifier.score(X_test,y_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_ada = classifier.predict(X_test)
confusion_matrix(y_test, y_pred_ada)
print(classification_report(y_test, y_pred_ada))
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

ada_temp = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=500)        #Initialize the classifier object

parameters = {'base_estimator__max_depth':[5, 8],
              'base_estimator__min_samples_split': [2, 3 , 5, 10]}   #Dictionary of parameters
# parameters = { 'n_estimators':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000]}

scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(ada_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_ada = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)
# ada_temp.get_params().keys()
ada_best = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8,min_samples_split=5),n_estimators =500)
ada_best.fit(X_train, y_train)
ada_best.score(X_test,y_test)
y_pred_ada_best = ada_best.predict(X_test)
confusion_matrix(y_test, y_pred_ada_best)
print(classification_report(y_test, y_pred_ada_best))
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

scorer_f1 = make_scorer(f1_score, average = 'micro')

cv_results = cross_validate(ada_best, X_N, Y, cv=10, scoring=(scorer_f1), return_train_score=True)
print(cv_results.keys())
print("Train Accuracy for 3 folds= ",np.mean(cv_results['train_score']))
print("Test Accuracy for 3 folds = ",np.mean(cv_results['test_score']))
y_pred_ada_best = ada_best.predict(test_N)
y_pred_ada_best
res = pd.DataFrame(np.zeros((1000-700,2)))
 
for i in range(700, 1000):
    res[0][i-700]=str(i)
    res[1][i-700]=int(y_pred_ada_best[i-700])
 
res[1]=res[1].astype(int)
res[0]=res[0].astype(int)
res
final_sub=pd.DataFrame(res)
final_sub.columns = ['ID','Class']
final_sub
final_sub.to_csv('ada.csv',index=False)
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "ada.csv"):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode())
  payload = b64.decode()
  html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
  html = html.format(payload=payload,title=title,filename=filename)
  return HTML(html)
create_download_link(final_sub)