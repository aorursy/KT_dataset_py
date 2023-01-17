import numpy as np 

import pandas as pd

import pandas_profiling 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly as py

import plotly.graph_objs as go

import plotly.express as px

import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, MaxAbsScaler



#Pipelines allow you to create a single object that includes all steps from data preprocessing & classification.

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics 

from sklearn.metrics import accuracy_score, recall_score

from IPython.display import display_html

import warnings

warnings.filterwarnings("ignore")

#custom function to display dataframes    



def displayoutput(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline;margin-left:50px !important;margin-right: 40px !important"'),raw=True)

    



def printoutput(string, color=None):

    colorstr = "<span style='color:{}'>{}</span>".format(color, string)

    display(Markdown(colorstr))    
originalData = pd.read_csv('../input/portuguese-marketing-campaigns-dataset/bank-full.csv')
pandas_profiling.ProfileReport(originalData)
originalData.shape
originalData.describe(include=["object"])
CatCloums = originalData.select_dtypes(include="object")

print ( "Categorical Columns:" )

print ( "_____________________" )

sno = 1



for i in CatCloums.columns:

    print(sno, "." , i)

    sno += 1
for catcol in CatCloums:

    sns.countplot(data=CatCloums,y=catcol,order=CatCloums[catcol].value_counts().index)

    plt.figure(figsize=(20,20))

    plt.show()
for i in CatCloums:

    print("Cloumn Name :", i)

    print ( CatCloums [ i ].value_counts ( ) )
for i in CatCloums:

    f, axes = plt.subplots( figsize = (7,5))

    print(i)

    sns.countplot(originalData[i])

    plt.xticks ( rotation = 50 )

    plt.show ( )
NumColums = originalData.select_dtypes(exclude= 'object')

sno = 1

print ( "Numerical columns:" )

print ( "______________________" )

for i in NumColums.columns:

    print(sno, ".", i)

    sno += 1
_ = originalData.hist(column=NumColums.columns,figsize=(15,15),grid=False,color='#86bf91',zorder=2,rwidth=1.0)
for i in NumColums:

    print("Column :", i)

    sns.distplot(originalData[i])

    plt.show()

originalData.describe(include=["object"])
originalData.describe(exclude =["object"])
originalData.corr()
duplicates = originalData.duplicated()

sum(duplicates)
originalData.isna().sum()
originalData.isnull().sum()
originalData.drop(['duration'],axis=1,inplace=True)

originalData.head(10)
originalData['pdays']=originalData['pdays'].astype('category')

originalData['Target']=originalData['Target'].astype('category')

originalData.head(5)
originalData.default.replace(('yes', 'no'), (1, 0), inplace=True)

originalData.housing.replace(('yes','no'),(1,0),inplace=True)

originalData.loan.replace(('yes','no'),(1,0),inplace=True)





from sklearn.model_selection import train_test_split



X = originalData.loc[:,originalData.columns !='Target']

y = originalData['Target']



#get dummies for catagorical features

X = pd.get_dummies(X, drop_first=True)





X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= .40,random_state=101)

logregmodel = LogisticRegression(solver='liblinear')

# Fit the model on train

logregmodel.fit(X_train,y_train)



#predict on test

y_predict = logregmodel.predict(X_test)

y_predict_df = pd.DataFrame(y_predict)

# Check is the model an overfit model? 

y_pred = logregmodel.predict(X_test)

print(logregmodel.score(X_train, y_train))

print(logregmodel.score(X_test , y_test))
cm = metrics.confusion_matrix(y_test,y_predict)





plt.clf()

plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Wistia)

clsnames = ['Not_Subscribed','Subscribed']

plt.title('Confusion Matrix for Test Data')

plt.ylabel('Actual')

plt.xlabel('Predicted')

tick_marks = np.arange(len(clsnames))

plt.xticks(tick_marks, clsnames, rotation=45)

plt.yticks(tick_marks, clsnames)

s = [['TN','FP'], ['FN', 'TP']]



for i in range(2):

    for j in range(2):

        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))

plt.show()
logisticscore = logregmodel.score(X_test,y_test)

print(logisticscore)
logisticaccuracy = metrics.accuracy_score(y_test,y_predict)

print(logisticaccuracy)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



# instantiate learning model 

knn = KNeighborsClassifier() 



# fitting the model

knn.fit(X_train,y_train)



# predict the response

y_pred = knn.predict(X_test)
# evaluate accuracy

accuracy_score(y_test,y_pred)
# instantiate learning model (k = 3)

knn = KNeighborsClassifier(n_neighbors=3)



# fitting the model

knn.fit(X_train, y_train)



# evaluate accuracy

accuracy_score(y_test,y_pred)
# instantiate learning model (k = 9)

knn = KNeighborsClassifier(n_neighbors=9)



# fitting the model

knn.fit(X_train, y_train)



# evaluate accuracy

accuracy_score(y_test,y_pred)
# creating odd list of K for KNN

myList = list(range(1,20))



# subsetting just the odd ones

neighbors = list(filter(lambda x: x % 2 != 0, myList))



# empty list that will hold accuracy scores

ac_scores = []



# perform accuracy metrics for values from 1,3,5....19

for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    # predict the response

    y_pred = knn.predict(X_test)

    # evaluate accuracy

    scores = accuracy_score(y_test, y_pred)

    ac_scores.append(scores)



# changing to misclassification error

MSE = [1 - x for x in ac_scores]



# determining best k

optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)
plt.plot(MSE,neighbors)

plt.xlabel('Number of Neighbors K')

plt.ylabel('MisClassification Error')

plt.show
lstaccuracy =[]

for k in range(20):

    K_value = k+1

    neigh = KNeighborsClassifier(n_neighbors=K_value)

    neigh.fit(X_train,y_train)

    y_pred = neigh.predict(X_test)

    lstaccuracy.append(accuracy_score(y_test,y_pred)*100)

    print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)

plt.plot(lstaccuracy)

plt.ylabel('Accuracy')

plt.xlabel('Number of neighbors')

plt.title("Accuracy vs # Neighbors")
count_misclassified = (y_test != y_pred).sum()

count_misclassified
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



#create a new KNN model

knn_cv = KNeighborsClassifier(n_neighbors=3)

#train model with cv of 5 

cv_scores = cross_val_score(knn_cv, X_test, y_test, cv=5)

#print each cv score (accuracy) and average them

print(cv_scores)

print(format(np.mean(cv_scores)))
from sklearn.tree import DecisionTreeClassifier



dTree = DecisionTreeClassifier(criterion = 'entropy', random_state=10)

dTree.fit(X_train, y_train)
print(dTree.score(X_train, y_train))

print(dTree.score(X_test, y_test))



print(recall_score(y_test, y_pred,average="binary", pos_label="yes"))
predict_dTree = dTree.predict(X_test)

cm = metrics.confusion_matrix(y_test,predict_dTree)

cm_df = pd.DataFrame(cm)



plt.figure(figsize=(5,5))

sns.heatmap(cm_df,annot=True,fmt='g')
#print (pd.DataFrame(dTree.feature_importances_, columns = ["Importance"], index = X_train.columns))



feat_importance = dTree.tree_.compute_feature_importances(normalize=False)

feat_imp_dict = dict(zip(X_train.columns, dTree.feature_importances_))

feat_imp = pd.DataFrame.from_dict(feat_imp_dict,orient='index')

feat_imp.sort_values(by=0, ascending=False)
dTree_Pruning  = DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=10,min_samples_leaf=3)



dTree_Pruning.fit(X_train,y_train)
#preds_pruned = dTree_Pruning.predict(X_test)

#preds_pruned_train = dTree_Pruning.predict(X_train)



print(dTree_Pruning.score(X_train, y_train))

print(dTree_Pruning.score(X_test,y_test))
predict_dTree_purning = dTree_Pruning.predict(X_test)

cm_purning = metrics.confusion_matrix(y_test,predict_dTree_purning)

cm_df_purning = pd.DataFrame(cm_purning)



plt.figure(figsize=(5,5))

sns.heatmap(cm_df_purning,annot=True,fmt='g')
## Calculating feature importance



feat_importance = dTree_Pruning.tree_.compute_feature_importances(normalize=False)

feat_imp_dict = dict(zip(X_train.columns, dTree_Pruning.feature_importances_))

feat_imp = pd.DataFrame.from_dict(feat_imp_dict,orient='index')

feat_imp.sort_values(by=0, ascending=False)
acc_DT = accuracy_score(y_test, predict_dTree_purning)

recall_DT = recall_score(y_test, predict_dTree_purning, average="binary", pos_label="yes")
#Store the accuracy results for each model in a dataframe for final comparison

resultsDf = pd.DataFrame({'Method':['Decision Tree'], 'accuracy': acc_DT, 'recall': recall_DT})

resultsDf = resultsDf[['Method', 'accuracy', 'recall']]

resultsDf
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=50)

rfc = rfc.fit(X_train,y_train)

rfc
predict_rfc = rfc.predict(X_test)

accuracy_rfc = accuracy_score(y_test,predict_rfc)

recall_rfc = recall_score(y_test, predict_rfc, average="binary", pos_label="yes")



tempResultsDf = pd.DataFrame({'Method':['Random Forest'], 'accuracy': [accuracy_rfc]})

tempResultsDf
tempResultsDf = pd.DataFrame({'Method':['Random Forest'], 'accuracy': [accuracy_rfc], 'recall': [recall_rfc]})

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy', 'recall']]

resultsDf

resultsDf
abc1 = AdaBoostClassifier(n_estimators=10,learning_rate=0.1,random_state=25)

abc1 = abc1.fit(X_train,y_train)



accuracy_AdaBoost = abc1.score(X_test,y_test)

print(accuracy_AdaBoost)
pred_AB = abc1.predict(X_test)

acc_AB = accuracy_score(y_test,pred_AB)

recall_AB = recall_score(y_test,pred_AB,pos_label='yes')

tempResultsDf = pd.DataFrame({'Method':['Adaboost'], 'accuracy': [acc_AB], 'recall':[recall_AB]})

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy', 'recall']]

resultsDf

resultsDf
predict_AdaBoost = abc1.predict(X_test)

cm = metrics.confusion_matrix(y_test,pred_AB)

cm_df = pd.DataFrame(cm)

plt.figure(figsize=(5,5))

sns.heatmap(cm_df,annot=True ,fmt='g',)
bgcl = BaggingClassifier(n_estimators=100, max_samples= .7, bootstrap=True, oob_score=True, random_state=22)

bgcl = bgcl.fit(X_train, y_train)
pred_BG =bgcl.predict(X_test)

acc_BG = accuracy_score(y_test, pred_BG)

recall_BG = recall_score(y_test, pred_BG, pos_label='yes')
tempResultsDf = pd.DataFrame({'Method':['Bagging'], 'accuracy': [acc_BG], 'recall':[recall_BG]})

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy', 'recall']]

resultsDf

resultsDf
gb_model = GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.1, random_state=22)

gb_model = gb_model.fit(X_train, y_train)
predict_GB =gb_model.predict(X_test)

accuracy_GB = accuracy_score(y_test, predict_GB)

recall_GB = recall_score(y_test, predict_GB, pos_label='yes')


tempResultsDf = pd.DataFrame({'Method':['Gradient Boost'], 'accuracy': [accuracy_GB], 'recall':[recall_GB]})

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy', 'recall']]

resultsDf
