#'''Importing Data Manipulation Modules'''
import numpy as np                 # Linear Algebra
import pandas as pd                # Data Processing, CSV file I/O (e.g. pd.read_csv)

#'''Seaborn and Matplotlib Visualization'''
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import seaborn as sns              # Python Data Visualization Library based on matplotlib
plt.style.use('fivethirtyeight')
%matplotlib inline

#'''Plotly Visualizations'''
import plotly as plotly                # Interactive Graphing Library for Python
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.offline as py
init_notebook_mode(connected=True)
import os
%pylab inline
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.dtypes
df.describe()
def quality(dataframe):
    dataframe.loc[(dataframe['quality'] >= 2) & (dataframe['quality'] <= 6.5), 'quality'] = 0
    
    dataframe.loc[(dataframe['quality'] > 6.5) & (dataframe['quality'] <= 8), 'quality'] = 1
           
    return dataframe

quality(df)
x = df.drop(['quality'],axis = 1)
y = df['quality']
labels = (df.quality.unique())
colors = ['Crimson', 'DarkBlue']

trace = go.Histogram(x=df.quality,marker=dict(color=colors,line=dict(color='black', width=2)),opacity=0.75)
layout = go.Layout(
    title='Quality distribution',
    xaxis=dict(
        title='Bad wine - Great wine'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.2,
    bargroupgap=0.1, paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor="rgb(243, 243, 243)")
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization


data = pd.concat([y,data_n_2],axis=1)
data = pd.melt(data,id_vars="quality",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="quality", data=data,split=True, inner="quart")
plt.xticks(rotation=90)
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_n_2],axis=1)
data = pd.melt(data,id_vars="quality",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="quality", data=data,palette=["black", "silver"])

plt.xticks(rotation=90)
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
accuracies = {}
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)
ac = accuracy_score(y_test,clf_rf.predict(x_test))

print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_1 = RandomForestClassifier(random_state = 42) 
rfecv = RFECV(estimator=clf_rf_1, step=1, cv=k_fold,scoring='accuracy')   #10-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x.columns[rfecv.support_])
x_1 = df[['volatile acidity','citric acid','total sulfur dioxide','density','sulphates','alcohol']]
x_train, x_test, y_train, y_test = train_test_split(x_1,y, test_size=0.25, random_state=42)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
clf_rf_1 = RandomForestClassifier(random_state=43)      
clr_rf_1 = clf_rf_1.fit(x_train,y_train)
ac = accuracy_score(y_test,clf_rf_1.predict(x_test))
accuracies['Random_Forest'] = ac

print('Accuracy is: ',ac, '\n')
cm = confusion_matrix(y_test,clf_rf_1.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('RFC Reports\n',classification_report(y_test, clf_rf_1.predict(x_test)))
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression() 
logmodel.fit(x_train,y_train)

ac = accuracy_score(y_test,logmodel.predict(x_test))
accuracies['Logistic regression'] = ac

print('Accuracy is: ',ac, '\n')
cm = confusion_matrix(y_test,logmodel.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('Logistic regression Reports\n',classification_report(y_test, logmodel.predict(x_test)))


from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
#Neighbors
neighbors = np.arange(0,25)

#Create empty list that will hold cv scores
cv_scores = []

#Perform 10-fold cross validation on training set for odd values of k:
for k in neighbors:
    k_value = k+1
    knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')
    kfold = model_selection.KFold(n_splits=10, random_state=123)
    scores = model_selection.cross_val_score(knn, x_train, y_train, cv=k_fold, scoring='accuracy')
    cv_scores.append(scores.mean()*100)
    print("k=%d %0.2f (+/- %0.2f)" % (k_value, scores.mean()*100, scores.std()*100))

optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print ("The optimal number of neighbors is %d with %0.1f%%" % (optimal_k, cv_scores[optimal_k]))

plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Train Accuracy')
plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train, y_train)

ac = accuracy_score(y_test,knn.predict(x_test))
accuracies['KNN'] = ac


print('Accuracy is: ',ac, '\n')
cm = confusion_matrix(y_test,knn.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('KNN Reports\n',classification_report(y_test, knn.predict(x_test)))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini
dtree.fit(x_train, y_train)

ac = accuracy_score(y_test,dtree.predict(x_test))
print('Accuracy is: ',ac, '\n')
cm = confusion_matrix(y_test,dtree.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('DecisionTree Reports\n',classification_report(y_test, dtree.predict(x_test)))
from sklearn.tree import plot_tree
plt.figure(figsize=(20,15))
plot_tree(dtree,
         filled=True,
         rounded=True,
         feature_names=x_1.columns)
path = dtree.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

dtrees = []

for ccp_alpha in ccp_alphas:
    dtree = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    dtree.fit(x_train,y_train)
    dtrees.append(dtree)
train_scores = [dtree.score(x_train,y_train) for dtree in dtrees]
test_scores = [dtree.score(x_test, y_test) for dtree in dtrees]

fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy vs alpha for training and testing sets')
ax.plot(ccp_alphas, train_scores, marker = 'o', label = 'train', drawstyle='steps-post')
ax.plot(ccp_alphas, test_scores, marker = 'o', label = 'test', drawstyle='steps-post')
ax.legend()
plt.show()

# create an array to store the results of each fold during cross validation
f,ax = plt.subplots(figsize=(18, 8))
alpha_loop_values = []

for ccp_alpha in ccp_alphas:
    dtree = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores =cross_val_score(dtree, x_train, y_train, cv=kfold, scoring='accuracy')
    alpha_loop_values.append([ccp_alpha, np.mean(scores),np.std(scores)])
    
alpha_results = pd.DataFrame(alpha_loop_values,
                             columns=['alpha','mean_accuracy','std'])
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='--',
                   ax=ax)
alpha_results[(alpha_results['alpha'] > 0.003)
              & 
              (alpha_results['alpha'] < 0.004)]
dtree1 = DecisionTreeClassifier(random_state=42,
                                ccp_alpha=0.003672)
dtree1 = dtree1.fit(x_train, y_train)

ac = accuracy_score(y_test,dtree1.predict(x_test))
accuracies['decisiontree'] = ac


print('Accuracy is: ',ac,'\n')
cm = confusion_matrix(y_test,dtree1.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('Decision Tree Reports\n',classification_report(y_test, dtree1.predict(x_test)))
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'C':[0.5,1,10,100],
     'gamma': ['scale',1,0.1,0.01,0.001,0.0001],
     'kernel': ['rbf']},
]

optimal_params = GridSearchCV(
        SVC(),
        param_grid,
        cv = k_fold,
        scoring='accuracy',
        verbose = 0
    )
optimal_params.fit(x_train, y_train)
print(optimal_params.best_params_)
svc1= SVC(random_state = 42, C = 10, gamma = 1, kernel = 'rbf')
svc1.fit(x_train, y_train)

ac = accuracy_score(y_test,svc1.predict(x_test))
accuracies['SVM'] = ac


print('Accuracy is: ',ac, '\n')
cm = confusion_matrix(y_test,svc1.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('SVM report\n',classification_report(y_test, svc1.predict(x_test)))
from sklearn.naive_bayes import GaussianNB
gaussiannb= GaussianNB()
gaussiannb.fit(x_train, y_train)

ac = accuracy_score(y_test,gaussiannb.predict(x_test))
accuracies['GaussianNB'] = ac


print('Accuracy is: ',ac,'\n')
cm = confusion_matrix(y_test,gaussiannb.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

print('GaussianNB report\n',classification_report(y_test, gaussiannb.predict(x_test)))
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

plt.rcParams['figure.figsize'] = (18,8)

x=list(accuracies.keys())
y=list(accuracies.values())

bars = plt.bar(x, height=y, width=.4, color = colors)

xlocs, xlabs = plt.xticks()

xlocs=[i for i in x]
xlabs=[i for i in x]

plt.xlabel('Algorithms', size = 20)
plt.ylabel('Accuracy %', size = 20)
plt.xticks(xlocs, xlabs, size = 15)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + .1, yval + .005, yval, size = 15)

plt.show()
fig, ax_arr = plt.subplots(nrows = 2, ncols = 3, figsize = (20,15))
from sklearn import metrics

#RandomForest
probs = clf_rf_1.predict_proba(x_test)
preds = probs[:,1]
fprrfc, tprrfc, thresholdrfc = metrics.roc_curve(y_test, preds)
roc_aucrfc = metrics.auc(fprrfc, tprrfc)

ax_arr[0,0].plot(fprrfc, tprrfc, 'b', label = 'AUC = %0.2f' % roc_aucrfc)
ax_arr[0,0].plot([0, 1], [0, 1],'r--')
ax_arr[0,0].set_title('ROC Random Forest ',fontsize=20)
ax_arr[0,0].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,0].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,0].legend(loc = 'lower right', prop={'size': 16})

#LOGMODEL
probs = logmodel.predict_proba(x_test)
preds = probs[:,1]
fprlog, tprlog, thresholdlog = metrics.roc_curve(y_test, preds)
roc_auclog = metrics.auc(fprlog, tprlog)

ax_arr[0,1].plot(fprlog, tprlog, 'b', label = 'AUC = %0.2f' % roc_auclog)
ax_arr[0,1].plot([0, 1], [0, 1],'r--')
ax_arr[0,1].set_title('ROC Logistic ',fontsize=20)
ax_arr[0,1].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,1].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,1].legend(loc = 'lower right', prop={'size': 16})

#KNN
probs = knn.predict_proba(x_test)
preds = probs[:,1]
fprknn, tprknn, thresholdknn = metrics.roc_curve(y_test, preds)
roc_aucknn = metrics.auc(fprknn, tprknn)

ax_arr[0,2].plot(fprknn, tprknn, 'b', label = 'AUC = %0.2f' % roc_aucknn)
ax_arr[0,2].plot([0, 1], [0, 1],'r--')
ax_arr[0,2].set_title('ROC KNN ',fontsize=20)
ax_arr[0,2].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[0,2].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[0,2].legend(loc = 'lower right', prop={'size': 16})

#DECISION TREE
probs = dtree1.predict_proba(x_test)
preds = probs[:,1]
fprdtree, tprdtree, thresholddtree = metrics.roc_curve(y_test, preds)
roc_aucdtree = metrics.auc(fprdtree, tprdtree)

ax_arr[1,0].plot(fprdtree, tprdtree, 'b', label = 'AUC = %0.2f' % roc_aucdtree)
ax_arr[1,0].plot([0, 1], [0, 1],'r--')
ax_arr[1,0].set_title('ROC Decision Tree ',fontsize=20)
ax_arr[1,0].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,0].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,0].legend(loc = 'lower right', prop={'size': 16})


#Gaussiannb

probs = gaussiannb.predict_proba(x_test)
preds = probs[:,1]
fprgau, tprgau, thresholdgau = metrics.roc_curve(y_test, preds)
roc_aucgau = metrics.auc(fprgau, tprgau)

ax_arr[1,1].plot(fprgau, tprgau, 'b', label = 'AUC = %0.2f' % roc_aucgau)
ax_arr[1,1].plot([0, 1], [0, 1],'r--')
ax_arr[1,1].set_title('ROC Gaussian ',fontsize=20)
ax_arr[1,1].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,1].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,1].legend(loc = 'lower right', prop={'size': 16})

#All plots
ax_arr[1,2].plot(fprrfc, tprrfc, 'b', label = 'rfc', color='black')
ax_arr[1,2].plot(fprlog, tprlog, 'b', label = 'Logistic', color='blue')
ax_arr[1,2].plot(fprknn, tprknn, 'b', label = 'Knn', color='brown')
ax_arr[1,2].plot(fprdtree, tprdtree, 'b', label = 'Decision Tree', color='green')
ax_arr[1,2].plot(fprgau, tprgau, 'b', label = 'Gaussiannb', color='grey')
ax_arr[1,2].set_title('Receiver Operating Comparison ',fontsize=20)
ax_arr[1,2].set_ylabel('True Positive Rate',fontsize=20)
ax_arr[1,2].set_xlabel('False Positive Rate',fontsize=15)
ax_arr[1,2].legend(loc = 'lower right', prop={'size': 16})