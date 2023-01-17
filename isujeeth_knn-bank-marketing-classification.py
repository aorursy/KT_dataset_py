from sklearn.model_selection import StratifiedKFold,train_test_split, cross_val_score, GridSearchCV, learning_curve, KFold

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import roc_curve,roc_auc_score, precision_score, recall_score,f1_score

import pandas as pd

import numpy as np

import time



import matplotlib.pyplot as plt

from sklearn import metrics

import seaborn as sns

%matplotlib inline
data_1=pd.read_csv("/kaggle/input/bankpromotion/bank-additional-full.csv",sep=";")

data_2=pd.read_csv("/kaggle/input/bankpromotion/bank-additional.csv",sep=";")

data=pd.concat([data_1,data_2],axis=0)

data.head()
#Correlation Plot

plt.figure(figsize=(14,14))

sns.set(font_scale=1)

sns.heatmap(data.corr(),cmap='GnBu_r',annot=True, square = True ,linewidths=.5);

plt.title('Variable Correlation')
#To avoid mulicorinality drop the higly correltaed column

data = data.drop(["emp.var.rate","nr.employed"],axis=1)

data.head()
#label encoding



jobDummies = pd.get_dummies(data['job'], prefix = 'job')

maritalDummies = pd.get_dummies(data['marital'], prefix = 'marital')

educationDummies = pd.get_dummies(data['education'], prefix = 'education')

defaultDummies = pd.get_dummies(data['default'], prefix = 'default')

housingDummies = pd.get_dummies(data['housing'], prefix = 'housing')

loanDummies = pd.get_dummies(data['loan'], prefix = 'loan')

contactDummies = pd.get_dummies(data['contact'], prefix = 'contact')

poutcomeDummies = pd.get_dummies(data['poutcome'], prefix = 'poutcome')

data['month']=data['month'].astype('category')

data['day_of_week']=data['day_of_week'].astype('category')

data['y']=data['y'].astype('category')



# Assigning numerical values and storing in another column

data['month'] = data['month'].cat.codes

data['day_of_week'] = data['day_of_week'].cat.codes

data['y'] = data['y'].cat.codes



data['y'].dtype
data["age"]=data["age"].astype("int")

data["duration"]=data["duration"].astype("int")

data["pdays"]=data["pdays"].astype("int")

data["previous"]=data["previous"].astype("int")

data["campaign"]=data["campaign"].astype("int")

data_int=data.select_dtypes(include=['int','float64','bool'])

#data_int

bank_df=pd.concat([data_int,jobDummies,maritalDummies,educationDummies,defaultDummies,housingDummies,loanDummies

                  ,contactDummies,poutcomeDummies,data['month'],data['day_of_week'],data['y']],axis=1)

bank_df.head()
#checking variable distribution

print(len(bank_df.columns))

df_test = bank_df.iloc[:,0:25]

for index in range(25):

    df_test.iloc[:,index] = (df_test.iloc[:,index]-df_test.iloc[:,index].mean()) / df_test.iloc[:,index].std();

df_test.hist(figsize= (14,16));
#Predictors count

bank_df.groupby('y').size()
#Total features after one-hot-encoding

features = bank_df.columns

len(features)
#Variables and Output

y=np.array(bank_df["y"])

X=np.array(bank_df.iloc[:,0:48])
#Partition of Dataset

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.neighbors import KNeighborsClassifier



k_range= range(1,26)

scores={}

scores_list=[]

for k in k_range:

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    y_pred=knn.predict(X_test)

    scores[k]=metrics.accuracy_score(y_test,y_pred)

    scores_list.append(scores[k])
#plot relationship between K and the testing accuracy

plt.plot(k_range,scores_list)

plt.xlabel('Value of K')

plt.ylabel('Testing Accuracy')
print(scores_list)
knn=KNeighborsClassifier(n_neighbors=9)

knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred)

print('Precision: %f' % precision)

# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred)

print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred)

print('F1 score: %f' % f1)
print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
knn=KNeighborsClassifier(n_neighbors=9,weights='distance')

#‘distance’ : weight points by the inverse of their distance. 

#in this case, closer neighbors of a query point will have a greater 

#influence than neighbors which are further away.

knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)

precision = precision_score(y_test, y_pred)

print('Precision: %f' % precision)

# recall: tp / (tp + fn)

recall = recall_score(y_test, y_pred)

print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, y_pred)

print('F1 score: %f' % f1)
print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))




scores = []

n=5

i=0

mean_auc=0

seed=7

accuracy_test=0

accuracy_train=0

#cv = KFold(n_splits=n, random_state=42, shuffle=True)



cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)

knn=KNeighborsClassifier(n_neighbors=5)



for train_index, test_index in cv.split(X_train,y_train):

    X_train_cv, X_test_cv= X_train[train_index], X_train[test_index]

    y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

    

    #Fit Model

    knn.fit(X_train,y_train)



    #predict train

    preds_train = knn.predict(X_train_cv)



    #predict test

    preds_test = knn.predict(X_test_cv)

    

    i+=1

    # compute AUC metric for this CV fold

    fpr, tpr, thresholds = metrics.roc_curve(y_test_cv, preds_test)

    roc_auc = metrics.auc(fpr, tpr)

    print ("AUC (fold "+str(i)+"/"+str(n)+"): "+str(roc_auc))

    mean_auc += roc_auc

    

    print("Accuracy Validation Fold "+str(i)+" : "+str(metrics.accuracy_score(y_test_cv,preds_test)*100))

    accuracy_test+=metrics.accuracy_score(y_test_cv,preds_test)*100

    print("Accuracy Train Fold "+str(i)+" : "+str(metrics.accuracy_score(y_train_cv,preds_train)*100))

    accuracy_train+=metrics.accuracy_score(y_train_cv,preds_train)*100

    print(" ")

    

print ("Mean AUC: "+str(mean_auc/n) )

print ("Mean Validation Accuracy: "+str(accuracy_test/n))

print ("Mean Train Accuracy: "+str(accuracy_train/n))
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
##from sklearn.model_selection import kfold

#from sklearn.svm import SVC

#cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

train_sizes, train_scores, test_scores = learning_curve(knn, 

                                                        X_train, 

                                                        y_train,

                                                        # Number of folds in cross-validation

                                                        cv=cv,

                                                        # Evaluation metric

                                                        scoring='accuracy',

                                                        # Use all computer cores

                                                        n_jobs=-1, 

                                                        # 50 different sizes of the training set

                                                        train_sizes=np.linspace(0.01, 1.0, 50))



# Create means and standard deviations of training set scores

train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)



# Create means and standard deviations of test set scores

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



# Draw lines

plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")

plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")



# Draw bands

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")

plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")



# Create plot

plt.title("Learning Curve")

plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")

plt.tight_layout()

plt.show()