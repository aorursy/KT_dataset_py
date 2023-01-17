import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import  GridSpec
import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True) 
import warnings
warnings.filterwarnings('ignore')
import pandas_profiling as pp
#import DataScienceHelper as dsh

%matplotlib inline

from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import normalize, StandardScaler
from sklearn import metrics 




## We look at the data using the head and tail functions
HF = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

HF.head(5)
HF.tail(5)
HF.shape
HF.info()
HF.describe()
HF.isnull().sum()
## Change the data type of the variables with Binary values to the appropriate data type
HF["anaemia"] = HF["anaemia"].astype(str)
HF["diabetes"] = HF["diabetes"].astype(str)
HF["high_blood_pressure"] = HF["high_blood_pressure"].astype(str)
HF["sex"] = HF["sex"].astype(str)
HF["smoking"] = HF["smoking"].astype(str)
HF["DEATH_EVENT"] = HF["DEATH_EVENT"].astype(str)

print(HF.describe())
print(HF.describe(include = np.object))
columns = list(HF._get_numeric_data().keys())

columns
pp.ProfileReport(HF) ## Another way of generating descriptive statistics using Pandas Profiling package
dsh.show_kdeplot(HF, columns)
dsh.show_boxplot(HF, columns)
## First reconvert the categorical binary variables to intergers
cat_columns = list(HF.select_dtypes(include = 'object').keys())

for column in cat_columns:
    HF[column] = HF[column].astype(int)

print(cat_columns)
## Then we proceed with the correlation matrix
HF_matrix = HF.corr()

f, ax = plt.subplots(figsize = (12,10))
k = 13 ## Number of columns in the matrix
## Use the DEATH EVENT variable as index as it will be compared against other variables
cols = HF_matrix.nlargest(k, 'DEATH_EVENT')['DEATH_EVENT'].index 
hfm = np.corrcoef(HF[cols].values.T)
sns.set(font_scale = 1.5)

sns.heatmap(hfm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 12},
           cmap = 'BrBG', yticklabels = cols.values, xticklabels = cols.values)

plt.show()

HF['DEATH_EVENT'].value_counts()
Death_major = HF[HF['DEATH_EVENT'] == 0]
Death_minor = HF[HF['DEATH_EVENT'] == 1]

UP_min = resample(Death_minor, replace = True, n_samples = 203, random_state = 320)

## Combine the majority class with the upsampled minority class
HFN = pd.concat([Death_major, UP_min])

HFN['DEATH_EVENT'].value_counts()
## get the target variable and the independent variables
target = HFN['DEATH_EVENT']
independent = HFN.drop(['DEATH_EVENT'], axis = 1)
## Normalize the independent variable values
independent = normalize(independent)
independent = StandardScaler().fit_transform(independent)



## OR
# independent = StandardScaler().fit_transform(normalize(independent))
scores_lr = []
train_list = []
for i in range(1,10):
    x_train, x_test, y_train, y_test = train_test_split(independent, target,test_size = i/10, random_state = 123)
    
    
    lr = LogisticRegression()
    lr.fit(x_train,y_train) 
    print("Test accuracy: {}/Test Size: {}".format(np.round(lr.score(x_test,y_test),3),i))
    scores_lr.append(lr.score(x_test,y_test))
    train_list.append(lr.score(x_train,y_train))
     
    

fig, ax = plt.subplots(1,2, figsize = (17,6))
gs = fig.add_gridspec(1, 4)

grid = GridSpec(1, 4, left=0.1, bottom=0.05, right=1.2, top=0.94, wspace=0.3, hspace=0.3)

ax1 = fig.add_subplot(grid[0:3])
ax2 = fig.add_subplot(grid[3:4])

ax1.plot(range(1,10),scores_lr,label = "Test Accuracy")
ax1.plot(range(1,10),train_list, label = "Train Accuracy")
ax1.legend(fontsize = 15)
ax1.set_xlabel("Test Sizes")
ax1.set_ylabel("Accuracy")
ax1.set_title("Scores For Each Test Size",fontsize = 17)
ax1.grid(True, alpha = 0.4)


x_train, x_test, y_train, y_test = train_test_split(independent, 
                                                    target,test_size = (1 + scores_lr.index(np.max(scores_lr)))/10, 
                                                    random_state = 123)

lr_best = LogisticRegression(random_state = 123)
lr_best = lr_best.fit(x_train, y_train)
y_pred = lr_best.predict(x_test)
y_true = y_test


cm = confusion_matrix(y_true,y_pred)

sns.heatmap(cm, annot=True, annot_kws = {"size": 25}, linewidths=0.5, fmt = '.0f',
            ax = ax2,cmap = "Blues",linecolor = "black")
plt.title("Logistic Regression Confusion Matrix",fontsize = 17)
plt.show()

print("Best Accuracy(test): {}/Test Size: {}".format(np.max(scores_lr), 1 + scores_lr.index(np.max(scores_lr))))

       
print(classification_report(y_pred, y_true))
pred_prob = lr_best.predict_proba(x_test)

y_preds = pred_prob[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_true, y_preds)
auc_score = metrics.auc(fpr, tpr)

plt.figure(figsize = (10,10))
plt.title('ROC Curve: Logistic')
plt.plot(fpr, tpr, label = 'AUC = {:.2f}'.format(auc_score))
plt.plot([0, 1], [0, 1], 'r--')

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
x_train, x_test, y_train, y_test = train_test_split(independent, target, test_size = 0.2, random_state = 123)

scores_knn = []
train_list = []
for i in range(1,25):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train,y_train)
    print("test accuracy: {}/Neighbors: {}".format(np.round(knn.score(x_test,y_test), 3),i))
    scores_knn.append(knn.score(x_test,y_test))
    train_list.append(knn.score(x_train,y_train))
    

fig, ax = plt.subplots(1,2, figsize = (17,6))
gs = fig.add_gridspec(1, 4)

grid = GridSpec(1,4,left=0.1, bottom=0.05, right=1.2, top=0.94, wspace=0.3, hspace=0.3)

ax1 = fig.add_subplot(grid[0:3])
ax2 = fig.add_subplot(grid[3:4])    

ax1.plot(range(1,25),scores_knn, label = "Test Accuracy")
ax1.plot(range(1,25),train_list,c = "orange", label = "Train Accuracy")
ax1.legend(fontsize = 15)
ax1.set_xlabel("K Values")
ax1.set_ylabel("Accuracy")
ax1.set_title("Scores For Each K Value",fontsize = 17)
ax1.grid(True , alpha = 0.4)



Best_knn = KNeighborsClassifier(n_neighbors = 1 + scores_knn.index(np.max(scores_knn)))
Best_knn = Best_knn.fit(x_train, y_train)
y_pred = Best_knn.predict(x_test)
y_true = y_test

cm = confusion_matrix(y_true,y_pred)


sns.heatmap(cm, annot=True,annot_kws = {"size": 25}, linewidths=0.5, fmt = '.0f', ax=ax2,cmap = "Blues",linecolor = "black")
plt.title("KNN Confusion Matrix",fontsize = 17)
plt.show()

print("Best Accuracy(test): {}/Neighbors: {}".format(np.max(scores_knn),1 + scores_knn.index(np.max(scores_knn))))

print(classification_report(y_pred, y_true))
knn_prob = Best_knn.predict_proba(x_test)

y_preds = knn_prob[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_true, y_preds)
auc_score = metrics.auc(fpr, tpr)

plt.figure(figsize = (10,10))
plt.title('ROC Curve: KNN')
plt.plot(fpr, tpr, label = 'AUC = {:.2f}'.format(auc_score))
plt.plot([0, 1], [0, 1], 'r--')

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
scores_svm = []
train_list = []
for i in range(100,500,50):
    svm = SVC(cache_size = i)
    svm.fit(x_train,y_train)
    print("test accuracy: {}/Cache Size: {}".format(np.round(svm.score(x_test,y_test),3),i))
    scores_svm.append(svm.score(x_test,y_test))
    train_list.append(svm.score(x_train,y_train))



fig, ax = plt.subplots(1,2, figsize = (17,6))
gs = fig.add_gridspec(1, 4)

grid = GridSpec(1,4,left=0.1, bottom=0.05, right=1.2, top=0.94, wspace=0.3, hspace=0.3)

ax1 = fig.add_subplot(grid[0:3])
ax2 = fig.add_subplot(grid[3:4])  
    
ax1.plot(range(100,500,50), scores_svm, label = "Test Accuracy")
ax1.plot(range(100,500,50), train_list,c = "orange", label = "Train Accuracy")
ax1.legend(fontsize = 15)
ax1.set_xlabel("Cache Sizes")
ax1.set_ylabel("Accuracy")
ax1.set_title("Scores For Each Cache Size",fontsize = 17)
ax1.grid(True , alpha = 0.4)

Best_SVM = SVC(cache_size = 50*(1+scores_svm.index(np.max(scores_svm))))
Best_SVM = Best_SVM.fit(x_train, y_train)
y_pred = Best_SVM.predict(x_test)
y_true = y_test

cm = confusion_matrix(y_true,y_pred)

sns.heatmap(cm, annot=True,annot_kws = {"size": 25}, linewidths=0.5, fmt = '.0f', ax=ax2,cmap = "Blues",linecolor = "black")
plt.title("Confusion Matrix",fontsize = 17)
plt.show()

print("Best Accuracy(test): {}/Cache Size: {}".format(np.max(scores_svm), 
                                                      50 + 50 * (1 + scores_svm.index(np.max(scores_svm)))))
print(classification_report(y_pred, y_true))
scores_dt = []
train_list = []
for d in range(1,10):
    clf = DecisionTreeClassifier(max_depth = d,random_state = 123)
    clf = clf.fit(x_train, y_train)
    print("Test accuracy: {}/Max Depth: {}".format(np.round(clf.score(x_test,y_test),3),d))
    scores_dt.append(clf.score(x_test,y_test))
    train_list.append(clf.score(x_train,y_train))
    
fig, ax = plt.subplots(1,2, figsize = (17,6))
gs = fig.add_gridspec(1, 4)

grid = GridSpec(1,4,left=0.1, bottom=0.05, right=1.2, top=0.94, wspace=0.3, hspace=0.3)

ax1 = fig.add_subplot(grid[0:3])
ax2 = fig.add_subplot(grid[3:4])  
    
ax1.plot(range(1,10),scores_dt,label = "Test Score")
ax1.plot(range(1,10),train_list,label = "Train Score")
ax1.legend(fontsize = 15)
ax1.set_xlabel("Max Depth")
ax1.set_ylabel("Accuracy")
ax1.grid(True, alpha = 0.5)
ax1.set_title("Accuricies for each Max Depth Value",fontsize = 17)

Best_DT = DecisionTreeClassifier(max_depth = 1 + scores_dt.index(np.max(scores_dt)))
Best_DT = Best_DT.fit(x_train, y_train)
y_pred = Best_DT.predict(x_test)
y_true = y_test

cm = confusion_matrix(y_true,y_pred)

sns.heatmap(cm, annot=True,annot_kws = {"size": 25}, linewidths=0.5, fmt = '.0f', ax=ax2,cmap = "Blues",linecolor = "black")
plt.title("Confusion Matrix",fontsize = 17)
plt.show()

print("Best Accuracy: {}/Max Depth: {}".format(np.max(scores_dt), 1 + scores_dt.index(np.max(scores_dt))))
print(classification_report(y_pred, y_true))
DT_prob = Best_DT.predict_proba(x_test)

y_preds = DT_prob[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_true, y_preds)
auc_score = metrics.auc(fpr, tpr)

plt.figure(figsize = (10,10))
plt.title('ROC Curve: Decision Tree')
plt.plot(fpr, tpr, label = 'AUC = {:.2f}'.format(auc_score))
plt.plot([0, 1], [0, 1], 'r--')

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
scores_rf = []
train_list = []

for i in range(20,160,20):
    rf = RandomForestClassifier(n_estimators = i, random_state = 123) #100
    rf.fit(x_train,y_train)
    print("Test Score: {}/Number of Estimators: {} ".format(np.round(rf.score(x_test,y_test),3),i))
    scores_rf.append(rf.score(x_test,y_test))
    train_list.append(rf.score(x_train,y_train))

fig, ax = plt.subplots(1,2, figsize = (17,6))
gs = fig.add_gridspec(1, 4)

grid = GridSpec(1,4,left=0.1, bottom=0.05, right=1.2, top=0.94, wspace=0.3, hspace=0.3)

ax1 = fig.add_subplot(grid[0:3])
ax2 = fig.add_subplot(grid[3:4])  

ax1.plot(range(20,160,20),scores_rf,label = "Test Accuracy")
ax1.plot(range(20,160,20),train_list,label = "Train Accuracy")
ax1.legend(fontsize = 15)
ax1.set_xlabel("N Estimators")
ax1.set_ylabel("Accuracy")
ax1.set_title("Scores for each N Estimator",fontsize = 17)
ax1.grid(True, alpha=0.5)

Best_rf = RandomForestClassifier(n_estimators = 20*(1+scores_rf.index(np.max(scores_rf))))
Best_rf = Best_rf.fit(x_train, y_train)
y_pred = Best_rf.predict(x_test)
y_true = y_test

cm = confusion_matrix(y_true,y_pred)

sns.heatmap(cm, annot=True,annot_kws = {"size": 25}, linewidths=0.5, fmt = '.0f', ax=ax2,cmap = "Blues",linecolor = "black")
plt.title("Confusion Matrix",fontsize = 17)
plt.show()


print("Best Accuracy: {}/Max Depth: {}".format(np.max(scores_rf),
                                               20*(1+scores_rf.index(np.max(scores_rf)))))
print(classification_report(y_pred, y_true))
rf_prob = Best_rf.predict_proba(x_test)

y_preds = rf_prob[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_true, y_preds)
auc_score = metrics.auc(fpr, tpr)

plt.figure(figsize = (10,10))
plt.title('ROC Curve: Random Forest')
plt.plot(fpr, tpr, label = 'AUC = {:.2f}'.format(auc_score))
plt.plot([0, 1], [0, 1], 'r--')

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
scores_per = []
train_list = []
for i in np.arange(0.0001, 0.001, 0.0001):
    perceptron = Perceptron(alpha = i, random_state = 123) 
    perceptron.fit(x_train,y_train)
    print("Test Score: {}/Alpha: {} ".format(np.round(perceptron.score(x_test,y_test),3),np.round(i,5)))
    scores_per.append(perceptron.score(x_test,y_test))
    train_list.append(perceptron.score(x_train,y_train))

fig, ax = plt.subplots(1,2, figsize = (17,6))
gs = fig.add_gridspec(1, 4)

grid = GridSpec(1,4,left=0.1, bottom=0.05, right=1.2, top=0.94, wspace=0.3, hspace=0.3)

ax1 = fig.add_subplot(grid[0:3])
ax2 = fig.add_subplot(grid[3:4])      

ax1.plot(np.arange(0.0001,0.001, 0.0001),scores_per,label = "Test Accuracy")
ax1.plot(np.arange(0.0001,0.001, 0.0001),train_list,label = "Train Accuracy")
ax1.legend(fontsize = 15)
ax1.set_xlabel("Alpha")
ax1.set_ylabel("Accuracy")
ax1.set_title("Scores for each Alpha",fontsize = 17)
ax1.grid(True, alpha=0.5)    

Best_per = Perceptron(alpha = 0.0001+0.0001*(1+scores_per.index(np.max(scores_per))))
Best_per = Best_per.fit(x_train, y_train)
y_pred = Best_per.predict(x_test)
y_true = y_test

cm = confusion_matrix(y_true,y_pred)

sns.heatmap(cm, annot=True,annot_kws = {"size": 25}, linewidths=0.5, fmt = '.0f', ax=ax2,cmap = "Blues",linecolor = "black")
plt.title("Confusion Matrix",fontsize = 17)
plt.show()

print("Best Accuracy: {}/Alpha: {}".format(np.max(scores_per),0.0001*(1+scores_per.index(np.max(scores_per)))))
print(classification_report(y_pred, y_true))
scores_SGD = []
train_list = []
for i in np.arange(0.05, 0.3, 0.02):
    sgd = SGDClassifier(epsilon = i, random_state = 123) 
    sgd.fit(x_train,y_train)
    print("Test Score: {}/Epsilon: {} ".format(np.round(sgd.score(x_test,y_test),3),np.round(i,4)))
    scores_SGD.append(sgd.score(x_test,y_test))
    train_list.append(sgd.score(x_train,y_train))

fig, ax = plt.subplots(1,2, figsize = (17,6))
gs = fig.add_gridspec(1, 4)

grid = GridSpec(1,4,left=0.1, bottom=0.05, right=1.2, top=0.94, wspace=0.3, hspace=0.3)

ax1 = fig.add_subplot(grid[0:3])
ax2 = fig.add_subplot(grid[3:4])    

ax1.plot(np.arange(0.05, 0.3, 0.02),scores_SGD,label = "Test Accuracy")
ax1.plot(np.arange(0.05, 0.3, 0.02),train_list,label = "Train Accuracy")
ax1.legend(fontsize = 15)
ax1.set_xlabel("Epsilons")
ax1.set_ylabel("Accuracy")
ax1.set_title("Scores for each Epsilon", fontsize = 17)
ax1.grid(True, alpha=0.5)

Best_SGD = SGDClassifier(epsilon = 0.03+0.02*(1 + scores_SGD.index(np.max(scores_SGD))))
Best_SGD = Best_SGD.fit(x_train, y_train)
y_pred = Best_SGD.predict(x_test)
y_true = y_test

cm = confusion_matrix(y_true,y_pred)

sns.heatmap(cm, annot=True,annot_kws = {"size": 25}, linewidths=0.5, fmt = '.0f', ax=ax2,cmap = "Blues",linecolor = "black")
plt.title("Confusion Matrix",fontsize = 17)
plt.show()

print("Best Accuracy: {}/Epsilon: {}".format(np.max(scores_SGD),
                                             0.03+0.02*(1+scores_SGD.index(np.max(scores_SGD)))))
print(classification_report(y_pred, y_true))
scores_ridge = []
train_list = []
for i in np.arange(0.0005, 0.003, 0.0005):
    ridge = RidgeClassifier(tol = i, random_state = 123) 
    ridge.fit(x_train,y_train)
    print("Test Score: {}/Tol: {} ".format(np.round(ridge.score(x_test,y_test),3),np.round(i,4)))
    scores_ridge.append(ridge.score(x_test,y_test))
    train_list.append(ridge.score(x_train,y_train))

fig, ax = plt.subplots(1,2, figsize = (17,6))
gs = fig.add_gridspec(1, 4)

grid = GridSpec(1,4,left=0.1, bottom=0.05, right=1.2, top=0.94, wspace=0.3, hspace=0.3)

ax1 = fig.add_subplot(grid[0:3])
ax2 = fig.add_subplot(grid[3:4])  

ax1.plot(np.arange(0.0005, 0.003, 0.0005),scores_ridge,label = "Test Accuracy")
ax1.plot(np.arange(0.0005, 0.003, 0.0005),train_list,label = "Train Accuracy")
ax1.legend(fontsize = 15)
ax1.set_xlabel("Tols")
ax1.set_ylabel("Accuracy")
ax1.set_title("Scores for each Tol",fontsize = 17)
ax1.grid(True, alpha=0.5)

Best_Ridge = RidgeClassifier(tol = 0.0005*(1+scores_ridge.index(np.max(scores_ridge))))
Best_Ridge = Best_Ridge.fit(x_train, y_train)
y_pred = Best_Ridge.predict(x_test)
y_true = y_test

cm = confusion_matrix(y_true,y_pred)

sns.heatmap(cm, annot=True,annot_kws = {"size": 25}, linewidths=0.5, fmt = '.0f', ax=ax2,cmap = "Blues",linecolor = "black")
plt.title("Confusion Matrix",fontsize = 17)
plt.show()

print("Best Accuracy: {}/Tol: {}".format(np.max(scores_ridge),0.0005*(1+scores_ridge.index(np.max(scores_ridge)))))
print(classification_report(y_true, y_pred))
nb = GaussianNB()
nb.fit(x_train,y_train)

print("Test Accuracy: ",nb.score(x_test,y_test))

y_pred = nb.predict(x_test)
y_true = y_test

cm = confusion_matrix(y_true,y_pred)

plt.figure(figsize = (6,6))
sns.heatmap(cm, annot=True,annot_kws = {"size": 25}, linewidths=0.5, fmt = '.0f',cmap = "Blues",linecolor = "black")
plt.title("NB Confusion Matrix",fontsize = 17)
plt.show()
models = {"Models":["Logistic Regression",
                       "KNN",
                       "SVC",
                       "Decision Tree",
                       "Random Forest",
                       "Perceptron",
                       "Sthocastic Gradient Descent",
                       "Ridge", "Naive Bayes"],
             "Scores":[np.max(scores_lr).round(3),
                       np.max(scores_knn).round(3),
                       np.max(scores_svm).round(3),
                       np.max(scores_dt).round(3),
                       np.max(scores_rf).round(3),
                       np.max(scores_per).round(3),
                       np.max(scores_SGD).round(3),
                       np.max(scores_ridge).round(3),
                       nb.score(x_test,y_test).round(3)]}


modelsDF = pd.DataFrame(models)
modelsDF = modelsDF.sort_values(by = ["Scores"])
modelsDF.head(len(modelsDF)) 

trace = go.Bar(
    x = modelsDF["Models"],
    y = modelsDF["Scores"],
    text = modelsDF["Scores"],
    textposition = "auto",
    marker=dict(color = modelsDF["Scores"],colorbar=dict(
            title="ColorScale"
        ),colorscale="Viridis",))

data = [trace]
layout = go.Layout(title = "Comparison of Models",template = "plotly_white")

fig = go.Figure(data = data, layout = layout)
fig.update_xaxes(title_text = "Models")
fig.update_yaxes(title_text = "Scores")
fig.show()
