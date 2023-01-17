import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

main_df=pd.read_csv('/kaggle/input/tool-wear-detection-in-cnc-mill/train.csv')
main_df=main_df.fillna('no')
main_df.head()
import glob
#set working directory
os.chdir('/kaggle/input')
files = list()

for i in range(1,19):
    exp_number = '0' + str(i) if i < 10 else str(i)
    file = pd.read_csv("/kaggle/input/tool-wear-detection-in-cnc-mill/experiment_{}.csv".format(exp_number))
    row = main_df[main_df['No'] == i]
    
     #add experiment settings to features
    file['feedrate']=row.iloc[0]['feedrate']
    file['clamp_pressure']=row.iloc[0]['clamp_pressure']
    
    # Having label as 'tool_conidtion'
    
    file['label'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
    files.append(file)
df = pd.concat(files, ignore_index = True)
df.head()
df.shape
df.dtypes
# Convert 'Machining_process' into numerical values
pro={'Layer 1 Up':1,'Repositioning':2,'Layer 2 Up':3,'Layer 2 Up':4,'Layer 1 Down':5,'End':6,'Layer 2 Down':7,'Layer 3 Down':8,'Prep':9,'end':10,'Starting':11}

data=[df]

for dataset in data:
    dataset['Machining_Process']=dataset['Machining_Process'].map(pro)
df=df.drop(['Z1_CurrentFeedback','Z1_DCBusVoltage','Z1_OutputCurrent','Z1_OutputVoltage','S1_SystemInertia'],axis=1)
corm=df.corr()
corm
#checking the relationship between the variables by applying the correlation 
plt.figure(figsize=(20, 15))
p = sns.heatmap(df.corr(), annot=True)
X=df.drop(['label','Machining_Process'],axis=1)
Y=df['label']
print('The dimension of X table is: ',X.shape,'\n')
print('The dimension of Y table is: ', Y.shape)
from sklearn.model_selection import train_test_split

#divided into testing and training
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
sgd_model=SGDClassifier()
sgd_model.fit(x_train,y_train)
sgd_model_pred=sgd_model.predict(x_test)
acc_sgd_model=round(sgd_model.score(x_train, y_train)*100,2)
acc_sgd_model
rmf_model=RandomForestClassifier()
rmf_model.fit(x_train,y_train)
rmf_model_pred=rmf_model.predict(x_test)
acc_rmf_model=round(rmf_model.score(x_train, y_train)*100,2)
acc_rmf_model
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)
log_reg_pred=log_reg.predict(x_test)
acc_log_reg=round(log_reg.score(x_train,y_train)*100,2)
acc_log_reg
knb_model=KNeighborsClassifier()
knb_model.fit(x_train,y_train)
knb_model_pred=knb_model.predict(x_test)
acc_knb_model=round(knb_model.score(x_train,y_train)*100,2)
acc_knb_model
svm_model=LinearSVC()
svm_model.fit(x_train,y_train)
svm_model_pred=svm_model.predict(x_test)
acc_svm_model=round(svm_model.score(x_train,y_train)*100,2)
acc_svm_model
regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
regr_pred=regr.predict(x_test)
acc_regr=round(regr.score(x_train,y_train)*100,2)
acc_regr
nb=GaussianNB()
nb.fit(x_train,y_train)
nb_pred=nb.predict(x_test)
acc_nb=round(nb.score(x_train,y_train)*100,2)
acc_nb
xgb_model=XGBClassifier()
xgb_model.fit(x_train,y_train)
xgb_pred=xgb_model.predict(x_test)
acc_xgb=round(xgb_model.score(x_train,y_train)*100,2)
acc_xgb
lgbm = LGBMClassifier()
lgbm.fit(x_train,y_train)
lgbm_pred=lgbm.predict(x_test)
acc_lgbm=round(lgbm.score(x_train,y_train)*100,2)
acc_lgbm
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest','Stochastic Gradient Decent','Linear Regression','Naive Bayes','XGBoost','LightGBM'],
    'Score': [acc_svm_model, acc_knb_model, acc_log_reg, 
              acc_rmf_model,acc_sgd_model,acc_regr,acc_nb,acc_xgb,acc_lgbm]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df
from sklearn.model_selection import cross_val_score
rmf_model = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rmf_model, x_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores,'\n')
print("Mean:", scores.mean(),'\n')
print("Standard Deviation:", scores.std())
rmf_model = RandomForestClassifier(n_estimators=100, oob_score = True)
rmf_model.fit(x_train, y_train)
y_prediction = rmf_model.predict(x_test)

rmf_model.score(x_train, y_train)

acc_rmf_model = round(rmf_model.score(x_train, y_train) * 100, 2)
print(round(acc_rmf_model,2,), "%")
print("oob score:", round(rmf_model.oob_score_, 4)*100, "%")
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict

predictions = cross_val_predict(rmf_model, x_train, y_train, cv=3)
predictions[:10] # first 10 predictions
confusion_matrix(y_train,predictions)
print("Precision_score: ", precision_score(y_train,predictions),'\n')
print("Recall: ", recall_score(y_train,predictions),'\n')
print("Accruacy_score: ", accuracy_score(y_train,predictions),'\n')
print("F_score: ", f1_score(y_train, predictions))
from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = rmf_model.predict_proba(x_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()
from sklearn import tree
clf3 = tree.DecisionTreeClassifier().fit(x_train, y_train)
y_pred3 = clf3.predict(x_test)
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
acc_dt=roc_auc_score(y_test, y_pred3)*100
acc_dt