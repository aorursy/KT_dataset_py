import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn import metrics

from sklearn import preprocessing

from sklearn.model_selection  import cross_val_score





from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import plot_roc_curve
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.head()
#observe the different feature type present in the data

df.shape

df.info()
#Changing the data type of Class



df['Class'] = df['Class'].astype('category')



#Renaming the classes

df['Class'] = df['Class'].cat.rename_categories({1:'Fraudulent',0:'Non_Fraudulent'})



df['Class']
classes=df['Class'].value_counts()

normal_share=classes[0]/df['Class'].count()*100

print(normal_share)

fraud_share=classes[1]/df['Class'].count()*100

print(fraud_share)
#Creating a df for percentage of each class

class_share = {'Class':['fraudulent','non_fraudulent'],'Percentage':[fraud_share,normal_share]}

class_share = pd.DataFrame(class_share)

class_share.head()
# Create a bar plot for the number and percentage of fraudulent vs non-fraudulent transcations

sns.set_palette("muted")

plt.figure(figsize=(14,6))

plt.subplot(121)

sns.countplot('Class',data=df)

plt.title('No. of fraudulent vs non-fraudulent')



plt.subplot(122)

sns.barplot(x='Class', y='Percentage',data=class_share)

plt.title('% of fraudulent vs non-fraudulent')

plt.show()

# Create a scatter plot to observe the distribution of classes with time

#sns.set_palette("muted")

plt.figure(figsize=(10,6))

sns.stripplot(x= 'Class', y= 'Time',data=df)

plt.title('Distribution of Classes with Time\n (0: Non-Fraudulent || 1: Fraudulent)')

plt.show()
# Create a scatter plot to observe the distribution of classes with Amount

plt.figure(figsize=(10,6))

sns.stripplot(x= 'Class', y= 'Amount',data=df)

plt.title('Distribution of Classes with Amount\n (0: Non-Fraudulent || 1: Fraudulent)')

plt.show()
# Drop unnecessary columns

# Dropping the column 'Time' since it does not have any impact on deciding a fraud transaction



df=df.drop('Time',axis=1)

df.shape
#Plotting heatmap to check the coorelation



plt.figure(figsize=(8,6))



sns.heatmap(df.corr(),linewidths=0.5,cmap='YlGnBu')



plt.show()
y= df.iloc[:,-1] #class variable

X = df.iloc[:,:-1]

from sklearn import model_selection

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)



#Using stratify=y so that proportion of each class is same in both train and test set
print('Total count for each class:\n', y.value_counts())

print("\nCount of each class in train data:\n",y_train.value_counts())

print("\nCount of each class in test data:\n",y_test.value_counts())
# plot the histogram of a variable from the dataset to see the skewness

# ploting distribution plot for all columns to check the skewness



#Loop for creating distplot.



collist = list(X_train.columns)



c = len(collist)

m = 1

n = 0



plt.figure(figsize=(20,30))



for i in collist:

  if m in range(1,c+1):

    plt.subplot(8,4,m)

    sns.distplot(X_train[X_train.columns[n]])

    m=m+1

    n=n+1



plt.show()





# - Apply : preprocessing.PowerTransformer(copy=False) to fit & transform the train & test data

# Using ‘yeo-johnson’ method since it works with positive and negative values. It is used to improve normality or symmetry





from sklearn.preprocessing import power_transform



X_train = power_transform(X_train,method='yeo-johnson')

X_test = power_transform(X_test,method='yeo-johnson')
# Converting X_train & X_test back to dataframe

cols = X.columns



X_train = pd.DataFrame(X_train)

X_train.columns = cols



X_test = pd.DataFrame(X_test)

X_test.columns = cols

# plot the histogram of a variable from the dataset again to see the result 

# Plotting same set of variables as earlier to identify the difference.



#Loop for creating distplot.



collist = list(X_train.columns)



c = len(collist)

m = 1

n = 0



plt.figure(figsize=(20,30))



for i in collist:

  if m in range(1,c+1):

    plt.subplot(8,4,m)

    sns.distplot(X_train[X_train.columns[n]])

    m=m+1

    n=n+1



plt.show()



# Function to plot ROC curve and classification score which will be used for each model



from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report



def plot_roc(fpr,tpr):

    plt.plot(fpr, tpr, color='green', label='ROC')

    plt.plot([0, 1], [0, 1], color='yellow', linestyle='--')

    plt.title("Receiver Operating Characteristic (ROC) Curve")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend()

    plt.show()



def clf_score(clf):

    prob = clf.predict_proba(X_test)

    prob = prob[:, 1]

    auc = roc_auc_score(y_test, prob)    

    print('AUC: %.2f' % auc)

    fpr, tpr, thresholds = roc_curve(y_test,prob, pos_label='Non_Fraudulent')

    plot_roc(fpr,tpr)

    predicted=clf.predict(X_test)

    report = classification_report(y_test, predicted)

    print(report)

    return auc

# Logistic Regression

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression #import the package

from sklearn.model_selection import GridSearchCV

num_C = [0.001,0.01,0.1,1,10,100] #--> list of values



for cv_num in num_C:

  clf = LogisticRegression(penalty='l2',C=cv_num,random_state = 0)

  clf.fit(X_train, y_train)

  print('C:', cv_num)

  print('Coefficient of each feature:', clf.coef_)

  print('Training accuracy:', clf.score(X_train, y_train))

  print('Test accuracy:', clf.score(X_test, y_test))

  print('')
#perform cross validation



grid={"C":np.logspace(-3,3,7), "penalty":["l2"]}  # l2 ridge



lsr = LogisticRegression()

clf_lsr_cv = GridSearchCV(lsr,grid,cv=3,scoring='roc_auc')

clf_lsr_cv.fit(X_train,y_train)



print("tuned hpyerparameters :(best parameters) ",clf_lsr_cv.best_params_)

print("accuracy :",clf_lsr_cv.best_score_)



#perform hyperparameter tuning







#print the optimum value of hyperparameters
# Fitting the model with best parameters .



lsr_best = LogisticRegression(penalty='l2',C=0.01,random_state = 0)

lsr_clf = lsr_best.fit(X_train,y_train)

clf_score(lsr_clf)
#K-Nearest Neighbor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection  import cross_val_score

from sklearn.metrics import accuracy_score, mean_squared_error



# Taking only odd integers as K values to apply the majority rule. 

k_range = np.arange(1, 20, 2)

scores = [] #to store cross val score for each k

k_range
# Finding the best k with stratified K-fold method. 

# We will use cv=3 in cross_val_score to specify the number of folds in the (Stratified)KFold.



for k in k_range:

  knn_clf = KNeighborsClassifier(n_neighbors=k)

  knn_clf.fit(X_train,y_train)

  score = cross_val_score(knn_clf, X_train, y_train, cv=3, n_jobs = -1)

  scores.append(score.mean())



#Storing the mean squared error to decide optimum k

mse = [1-x for x in scores]

#Plotting a line plot to decide optimum value of K



plt.figure(figsize=(20,8))

plt.subplot(121)

sns.lineplot(k_range,mse,markers=True,dashes=False)

plt.xlabel("Value of K")

plt.ylabel("Mean Squared Error")

plt.subplot(122)

sns.lineplot(k_range,scores,markers=True,dashes=False)

plt.xlabel("Value of K")

plt.ylabel("Cross Validation Accuracy")



plt.show()

#Fitting the best parameter to the model

# 3 fold cross validation with K=3



knn = KNeighborsClassifier(n_neighbors=3)



knn_clf = knn.fit(X_train,y_train)



# Checking AUC 



clf_score(knn_clf)

#importing libraries



from sklearn import tree

from pprint import pprint

# 5 fold cross validation for getting best parameter



depth_score=[]

dep_rng = [x for x in range(1,20)]

for i in dep_rng:

  clf = tree.DecisionTreeClassifier(max_depth=i)

  score_tree = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=5, n_jobs=-1)

  depth_score.append(score_tree.mean())

print(depth_score)
#Plotting depth against score



plt.figure(figsize=(8,6))

sns.lineplot(dep_rng,depth_score,markers=True,dashes=False)

plt.xlabel("Depth")

plt.ylabel("Cross Validation Accuracy")



plt.show()
#Fitting the model with depth=5 and plotting ROC curve



dt = tree.DecisionTreeClassifier(max_depth = 5)

dt_clf = dt.fit(X_train,y_train)



#Plotting ROC

clf_score(dt_clf)
#Import libraries

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

# Using grid search cv to find the best parameters.



param = {'n_estimators': [10, 20, 30, 40, 50], 'max_depth': [2, 3, 4, 7, 9]}

rfc = RandomForestClassifier()

clf_rfc_cv = GridSearchCV(rfc, param, cv=5,scoring='roc_auc', n_jobs=-1)

clf_rfc_cv.fit(X_train,y_train)



print("tuned hpyerparameters :(best parameters) ",clf_rfc_cv.best_params_)

print("accuracy :",clf_rfc_cv.best_score_)



#Fitting model and plotting ROC



rf = RandomForestClassifier(max_depth=9, n_estimators=30)

RFC_clf = rf.fit(X_train,y_train)



#Plotting ROC

clf_score(RFC_clf)

#import libraries



from xgboost import XGBClassifier

from scipy import stats
# Using grid search cv to find the best parameters.



xgbst = XGBClassifier()



param_xgb = {'n_estimators': [130,140,150],

              'max_depth': [3, 5, 7],

               'min_child_weight':[1,2,3]

             } 



clf_xgb_cv = GridSearchCV(xgbst, param_xgb, cv=3,scoring='roc_auc', n_jobs=-1)

clf_xgb_cv.fit(X_train,y_train)



print("tuned hpyerparameters :(best parameters) ",clf_xgb_cv.best_params_)

print("accuracy :",clf_xgb_cv.best_score_)



#Fitting the model with best parameters.



xgbst = XGBClassifier(n_estimators=150,max_depth=5,min_child_weight=3)



xgb_clf = xgbst.fit(X_train,y_train)



#Plotting ROC

clf_score(xgb_clf)

clf = XGBClassifier(n_estimators=150,max_depth=5,min_child_weight=3)  #initialise the model with optimum hyperparameters

clf.fit(X_train, y_train)



# print the evaluation score on the X_test by choosing the best evaluation metric

clf_score(clf)
#importing SMOTE



from imblearn.over_sampling import SMOTE



sm = SMOTE()

X_sm, y_sm = sm.fit_resample(X_train, y_train)
#CHecking shape and class count after smote

from collections import Counter



print('Resampled dataset shape %s' % Counter(y_sm))

print(X_sm.shape)

print(y_sm.shape)
# importing ADASYN



from imblearn.over_sampling import ADASYN



ada = ADASYN()

X_ada, y_ada = ada.fit_resample(X_train, y_train)
# CHecking shape and class count after ADASYN

from collections import Counter



print('Resampled dataset shape %s' % Counter(y_ada))

print(X_ada.shape)

print(y_ada.shape)
# Using the best parameters that we got from the cross validation on imbalanced data.



lsr_best = LogisticRegression(penalty='l2',C=0.01,random_state = 0)

lsr_sm = lsr_best.fit(X_sm,y_sm)



# Printing ROC curve and accuracy scores

clf_score(lsr_sm)
lsr_ada = lsr_best.fit(X_ada,y_ada)



# Printing ROC curve and accuracy scores

clf_score(lsr_ada)
# KNN with SMOTE re-sampled data



knn = KNeighborsClassifier(n_neighbors=3)



knn_sm = knn.fit(X_sm,y_sm)



#Printing ROC 



clf_score(knn_sm)
# KNN with ADASYN re-sampled data



knn = KNeighborsClassifier(n_neighbors=3)



knn_ada = knn.fit(X_ada,y_ada)



#Printing ROC 



clf_score(knn_ada)
# Building model with SMOTE



dt = tree.DecisionTreeClassifier(max_depth = 5)

dt_sm = dt.fit(X_sm,y_sm)



#Plotting ROC

clf_score(dt_sm)
# Building model with ADASYN



dt = tree.DecisionTreeClassifier(max_depth = 5)

dt_ada = dt.fit(X_ada,y_ada)



#Plotting ROC

clf_score(dt_ada)
#Building Random forest with best parameters on SMOTE

rf = RandomForestClassifier(max_depth=9, n_estimators=30)

RFC_sm = rf.fit(X_sm,y_sm)



#Plotting ROC

clf_score(RFC_sm)

#Building Random forest with best parameters on ADASYN

rf = RandomForestClassifier(max_depth=9, n_estimators=30)

RFC_ada = rf.fit(X_ada,y_ada)



#Plotting ROC

clf_score(RFC_ada)

# Since X_sm and X_ada are arrays, we need to covert them to dataframes to avoid feature mismatch error 

X_sm = pd.DataFrame(X_sm)

X_sm.columns = cols



X_ada = pd.DataFrame(X_ada)

X_ada.columns = cols
#Fitting the XGBoost model with best parameters on SMOTE



xgbst = XGBClassifier(n_estimators=150,max_depth=5,min_child_weight=3)



xgb_sm = xgbst.fit(X_sm,y_sm)



#Plotting ROC

clf_score(xgb_sm)
#Fitting the XGBoost model with best parameters on ADASYN



xgbst = XGBClassifier(n_estimators=150,max_depth=5,min_child_weight=3)



xgb_ada = xgbst.fit(X_ada,y_ada)



#Plotting ROC

clf_score(xgb_ada)