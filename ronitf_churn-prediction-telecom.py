import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import GradientBoostingClassifier 

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv')
#print first 10 rows
df.head()
df.info()
df.describe(include=['O'])
label_encoder = preprocessing.LabelEncoder()

df['state'] = label_encoder.fit_transform(df['state'])
df['international plan'] = label_encoder.fit_transform(df['international plan'])
df['voice mail plan'] = label_encoder.fit_transform(df['voice mail plan'])
df['churn'] = label_encoder.fit_transform(df['churn'])

# too specific
df.drop(["phone number"], axis = 1, inplace=True)
stay = df[(df['churn'] ==0) ].count()[1]
churn = df[(df['churn'] ==1) ].count()[1]
print ("num of pepole who stay: "+ str(stay))
print ("num of pepole who churn: "+ str(churn))
# calculate the correlation matrix
corr = df.corr()

# plot the heatmap
fig = plt.figure(figsize=(5,4))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            linewidths=.75)
#we will normalize our data so the prediction on all features will be at the same scale
X = df.iloc[:,0:19].values
y = df.iloc[:,19].values
#nurmalize the data
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
dfNorm = pd.DataFrame(X_std, index=df.index, columns=df.columns[0:19])
# # add non-feature target column to dataframe
dfNorm['churn'] = df['churn']
dfNorm.head(10)

X = dfNorm.iloc[:,0:19].values
y = dfNorm.iloc[:,19].values
# after we learn and preprocessed our data, we need to split it n to train and test.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=0)

X_train.shape, y_train.shape, X_test.shape , y_test.shape
results_test = {}
results_train = {}
def prdict_date(algo_name,X_train,y_train,X_test,y_test,verbose=0):
    algo_name.fit(X_train, y_train)
    Y_pred = algo_name.predict(X_test)
    acc_train = round(algo_name.score(X_train, y_train) * 100, 2)
    acc_val = round(algo_name.score(X_test, y_test) * 100, 2)
    results_test[str(algo_name)[0:str(algo_name).find('(')]] = acc_val
    results_train[str(algo_name)[0:str(algo_name).find('(')]] = acc_train
    if verbose ==0:
        print("acc train: " + str(acc_train))
        print("acc test: "+ str(acc_val))
    else:
        return Y_pred
### helping function

def conf(algo_name,X_test, y_test):
    y_pred = algo_name.predict(X_test)
    forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])
    sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["1", "0"] , yticklabels = ["1", "0"] )
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.title(str(algo_name)[0:str(algo_name).find('(')])
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
#data labels before SMOTE:
import collections
collections.Counter(y_train)
#after SMOTE:
import collections
collections.Counter(y_train_res)
random_forest = RandomForestClassifier(n_estimators=75 , random_state=0  )
prdict_date(random_forest,X_train_res,y_train_res,X_test,y_test)
print(classification_report(y_test, random_forest.predict(X_test)))
conf(random_forest,X_test, y_test)

# Train: Gradient Boosting
gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.2, n_estimators=200 , max_depth=6)
prdict_date(gbc,X_train_res,y_train_res,X_test,y_test)

print(classification_report(y_test, gbc.predict(X_test)))
conf(gbc,X_test, y_test)
###### linear svm:
#  SVM
svm = SVC(kernel='linear', probability=True)
prdict_date(svm,X_train_res,y_train_res,X_test,y_test)
###### linear rbf:
svm = SVC(kernel='rbf', probability=True)
prdict_date(svm,X_train_res,y_train_res,X_test,y_test)
######  poly svm :
#  SVM
svm = SVC(kernel='poly', probability=True)
prdict_date(svm,X_train_res,y_train_res,X_test,y_test)
# Train: SVM
svm = SVC(kernel='poly', probability=True)
prdict_date(svm,X_train_res,y_train_res,X_test,y_test)

print(classification_report(y_test, svm.predict(X_test)))
conf(svm,X_test, y_test)
#we will try to find witch K is the best on our data
# first, we will look which give us the best predictions on the train:
from sklearn import model_selection

#Neighbors
neighbors = [x for x in list(range(1,50)) if x % 2 == 0]

#Create empty list that will hold cv scores
cv_scores = []

#Perform 10-fold cross validation on training set for odd values of k:
seed=0
for k in neighbors:
    k_value = k+1
    knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    scores = model_selection.cross_val_score(knn, X_train, y_train, cv=kfold, scoring='accuracy')
    cv_scores.append(scores.mean()*100)
    #print("k=%d %0.2f (+/- %0.2f)" % (k_value, scores.mean()*100, scores.std()*100))

optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print(( "The optimal number of neighbors is %d with %0.1f%%" % (optimal_k, cv_scores[optimal_k])))

plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Train Accuracy')
plt.show()
# then on the test:
cv_preds = []

#Perform 10-fold cross validation on testing set for odd values of k
seed=0
for k in neighbors:
    k_value = k+1
    knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    preds = model_selection.cross_val_predict(knn, X_test, y_test, cv=kfold)
    cv_preds.append(metrics.accuracy_score(y_test, preds)*100)
    #print("k=%d %0.2f" % (k_value, 100*metrics.accuracy_score(test_y, preds)))

optimal_k = neighbors[cv_preds.index(max(cv_preds))]
print("The optimal number of neighbors is %d with %0.1f%%" % (optimal_k, cv_preds[optimal_k]))

plt.plot(neighbors, cv_preds)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Test Accuracy')
plt.show()
# KNN
knn = KNeighborsClassifier(n_neighbors = 4)
prdict_date(knn,X_train_res,y_train_res,X_test,y_test)

print(classification_report(y_test, knn.predict(X_test)))
conf(knn,X_test, y_test)
# Train: Logistic Regression
logr = LogisticRegression()
prdict_date(logr,X_train,y_train,X_test,y_test)

print(classification_report(y_test, logr.predict(X_test)))
conf(logr,X_test, y_test)
df_test =pd.DataFrame(list(results_test.items()),
                      columns=['algo_name','acc_test'])
df_train =pd.DataFrame(list(results_train.items()),
                      columns=['algo_name','acc_train'])
df_results = df_test.join(df_train.set_index('algo_name'), on='algo_name')
df_results.sort_values('acc_test',ascending=False)

import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

# set jupyter's max row display
pd.set_option('display.max_row', 100)

# set jupyter's max column width to 50
pd.set_option('display.max_columns', 50)

# Load the dataset
ax = df_results[['acc_test', 'acc_train']].plot(kind='barh',
              figsize=(10,7), color=['dodgerblue', 'slategray'], fontsize=13);
ax.set_alpha(0.8)
ax.set_title("The Best ALGO is?",
fontsize=18)
ax.set_xlabel("ACC", fontsize=18)
ax.set_ylabel("Algo Names", fontsize=18)
ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100,110])
ax.set_yticklabels(df_results.iloc[:,0].values.tolist())

# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+4, i.get_y()+.1, \
            str(round((i.get_width()), 2)), fontsize=11, color='dimgrey')

# invert for largest on top 
ax.invert_yaxis()
feature_importance = gbc.feature_importances_
feat_importances = pd.Series(gbc.feature_importances_, index=df.columns[:-1])
feat_importances = feat_importances.nlargest(19)

feature = df.columns.values.tolist()[0:-1]
importance = sorted(gbc.feature_importances_.tolist())


x_pos = [i for i, _ in enumerate(feature)]
# 
plt.barh(x_pos, importance , color='dodgerblue')
plt.ylabel("feature")
plt.xlabel("importance")
plt.title("feature_importances")

plt.yticks(x_pos, feature)
#next, we can try and take only the best parameters and see how they do
gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.2, n_estimators=200 , max_depth=6)
prdict_date(gbc,X_train[:,3:],y_train,X_test[:,3:],y_test)
print(classification_report(y_test, gbc.predict(X_test[:,3:])))
conf(gbc,X_test[:,3:], y_test)