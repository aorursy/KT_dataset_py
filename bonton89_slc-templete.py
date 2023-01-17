import numpy as np

import pandas as pd

import statsmodels.api as sm # with statistical assumption

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
df1=pd.read_csv('US_Heart_Patients.csv')
df1.head()
df1.drop(['education'],axis=1,inplace= True)

df1.rename(columns={'male':"Sex_male"},inplace=True)

df1.head()
# droping the null values

df1.dropna(axis=0,inplace=True)
df1.info()
# check data balancesd or not

df1.TenYearCHD.value_counts()
sns.countplot(x='TenYearCHD',data=df1)

plt.show()
from statsmodels.tools import add_constant as add_constant # In sk learn the add constant is true by default
df_constant=add_constant(df1)
df_constant.head()
cols=df_constant.columns[:-1]
cols
model=sm.Logit(df1.TenYearCHD,df_constant[cols])
result= model.fit()
result.summary(

)
# bassed on the p value we select the features
new_features=df1[['Sex_male','age','cigsPerDay','sysBP','glucose','TenYearCHD']]
x=new_features.iloc[:,:-1]

x.head()
y=new_features.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.80,random_state=100)
from sklearn.linear_model import LogisticRegression
# fitting the logistic learn model

logreg=LogisticRegression()

logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
## Model Evaluation



## Model accuracy
print("Accuracy Score: ",accuracy_score(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
cm
cm=confusion_matrix(y_test,y_pred)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
TN=cm[0,0]

TP=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
print('The acuuracy of the model = TP+TN / (TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n\n',



'The Miss-classification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n\n',



'Sensitivity or True Positive Rate = TP / (TP+FN) = ',TP/float(TP+FN),'\n\n',



'Specificity or True Negative Rate = TN / (TN+FP) = ',TN/float(TN+FP),'\n\n',



'Positive Predictive value = TP / (TP+FP) = ',TP/float(TP+FP),'\n\n',



'Negative predictive Value = TN / (TN+FN) = ',TN/float(TN+FN),'\n\n',



'Positive Likelihood Ratio = Sensitivity / (1-Specificity) = ',sensitivity/(1-specificity),'\n\n',



'Negative likelihood Ratio = (1-Sensitivity) / Specificity = ',(1-sensitivity)/specificity)

y_pred_prob=logreg.predict_proba(x_test)[:,:]

y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of no heart disease (0)','Prob of Heart Disease (1)'])

y_pred_prob_df.head()
from sklearn.preprocessing import binarize

for i in range(1,5):

    cm2=0

    y_pred_prob_yes=logreg.predict_proba(x_test)

    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]

    cm2=confusion_matrix(y_test,y_pred2)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
### ROC curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])

plt.plot(fpr,tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for Heart disease classifier')

plt.xlabel('False positive rate (1-Specificity)')

plt.ylabel('True positive rate (Sensitivity)')

plt.grid(True)
roc_auc_score(y_test,y_pred_prob_yes[:,1])
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(max_depth=5)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("COnfusion Matrix: \n",confusion_matrix(y_test,y_pred))
print("Accuracy:", accuracy_score(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
TN=cm[0,0]

TP=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
print('The acuuracy of the model = TP+TN / (TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n\n',



'The Miss-classification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n\n',



'Sensitivity or True Positive Rate = TP / (TP+FN) = ',TP/float(TP+FN),'\n\n',



'Specificity or True Negative Rate = TN / (TN+FP) = ',TN/float(TN+FP),'\n\n',



'Positive Predictive value = TP / (TP+FP) = ',TP/float(TP+FP),'\n\n',



'Negative predictive Value = TN / (TN+FN) = ',TN/float(TN+FN),'\n\n',



'Positive Likelihood Ratio = Sensitivity / (1-Specificity) = ',sensitivity/(1-specificity),'\n\n',



'Negative likelihood Ratio = (1-Sensitivity) / Specificity = ',(1-sensitivity)/specificity)
y_pred_prob=logreg.predict_proba(x_test)[:,:]

y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of no heart disease (0)','Prob of Heart Disease (1)'])

y_pred_prob_df.head()
from sklearn.preprocessing import binarize

for i in range(1,5):

    cm2=0

    y_pred_prob_yes=logreg.predict_proba(x_test)

    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]

    cm2=confusion_matrix(y_test,y_pred2)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
from sklearn.model_selection import GridSearchCV
dtc=DecisionTreeClassifier()
dtc
param_grid= {'max_depth':range(5,15,5),

             'min_samples_leaf':range(50,150,50),

            'min_samples_split':range(50,150,50),

            'criterion':['entropy','gini']}
grid_search=GridSearchCV(estimator=dtc,param_grid=param_grid,cv=5)
grid_search.fit(x_train,y_train)
grid_search.best_estimator_
# since the above hyperparamers are the best estimator we will use it in decision tree classifier

dtc=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,

                       max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=100, min_samples_split=50,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=None, splitter='best')
dtc.fit(x_train,y_train)
# Test Score

dtc.score(x_test,y_test)
# Train Score

dtc.score(x_train,y_train)
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_curve,auc

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.ensemble import RandomForestClassifier

fit_rf=RandomForestClassifier(random_state=100)
fit_rf
param={'max_depth':[2,3,4,5],'bootstrap':[True,False],'max_features':['sqrt','log2'],'criterion':['gini','entropy']}
grid_rf=GridSearchCV(fit_rf,cv=5,param_grid=param)
grid_rf.fit(x_train,y_train)
grid_rf.best_estimator_
rf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=4, max_features='sqrt', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=10,

                       n_jobs=None, oob_score=False, random_state=100,

                       verbose=0, warm_start=False)
rf.fit(x_train,y_train)
# to check the feature importance

rf.feature_importances_
rf.score(x_train,y_train)
rf.score(x_test,y_test)
y_pred=rf.predict(x_test)
print("COnfusion Matrix: \n",confusion_matrix(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
TN=cm[0,0]

TP=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
print('The acuuracy of the model = TP+TN / (TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n\n',



'The Miss-classification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n\n',



'Sensitivity or True Positive Rate = TP / (TP+FN) = ',TP/float(TP+FN),'\n\n',



'Specificity or True Negative Rate = TN / (TN+FP) = ',TN/float(TN+FP),'\n\n',



'Positive Predictive value = TP / (TP+FP) = ',TP/float(TP+FP),'\n\n',



'Negative predictive Value = TN / (TN+FN) = ',TN/float(TN+FN),'\n\n',



'Positive Likelihood Ratio = Sensitivity / (1-Specificity) = ',sensitivity/(1-specificity),'\n\n',



'Negative likelihood Ratio = (1-Sensitivity) / Specificity = ',(1-sensitivity)/specificity)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy
# Model complexity

neig = np.arange(1, 25)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, test_accuracy, label = 'Testing Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('Value VS Accuracy',fontsize=20)

plt.xlabel('Number of Neighbors',fontsize=20)

plt.ylabel('Accuracy',fontsize=20)

plt.xticks(neig)

plt.grid()

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
## 1.14  TUNING AND HYPERPARAMETERS
# creating odd list of K for KNN

myList = list(range(1,20))



# subsetting just the odd ones

neighbors = list(filter(lambda x: x % 2 != 0, myList))


# empty list that will hold accuracy scores

ac_scores = []



# perform accuracy metrics for values from 1,3,5....19

for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train, y_train)

    # predict the response

    y_pred = knn.predict(x_test)

    # evaluate accuracy

    scores = accuracy_score(y_test, y_pred)

    ac_scores.append(scores)



# changing to misclassification error

MSE = [1 - x for x in ac_scores]



# determining best k

optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)
import matplotlib.pyplot as plt

# plot misclassification error vs k

plt.plot(neighbors, MSE)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()
from sklearn.naive_bayes import GaussianNB 
df1.plot(kind= 'box' , subplots=True, layout=(3,5), sharex=False, sharey=False, figsize=(13,10))
from scipy import stats
df,fitted_lambda=stats.boxcox(df1)


df1.hist(bins=15, color='red', edgecolor='black', linewidth=1.0,

              xlabelsize=8, ylabelsize=8, grid=False)    

plt.tight_layout(rect=(0, 0, 2.2, 2.2))   

#rt = plt.suptitle('Red Wine Univariate Plots', x=0.65, y=1.25, fontsize=14)  
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(x_test)

y_pred
cm = confusion_matrix(y_test, y_pred)
print(f1_score(y_test, y_pred))

print(accuracy_score(y_test, y_pred))
label = ["0","1"]

sns.heatmap(cm, annot=True, xticklabels=label, yticklabels=label)
print(classification_report(y_test,y_pred))
## 2.20  Pros and cons