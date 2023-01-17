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
data=pd.read_csv("../input/sgemm/sgemm_product.csv")
data.head()
#take average of 4 run
data["run_avg"]=np.mean(data.iloc[:,14:18],axis=1)

mean_run=np.mean(data["run_avg"])
print(mean_run)

#Binary Classification run_avg>mean_run
data["run_class"]=np.where(data['run_avg']>=mean_run, 1, 0)
data.groupby("run_class").size()

data.describe()

#Drop unwanted fields
sgemm_df=data.drop(columns=['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)','run_avg'])
sgemm_df.to_csv(r'segmm_product_classification.csv')
sgemm_df.head()
#data info
sgemm_df.info()
#No null values in the data
#checking for NULL values
sgemm_df.isnull().sum() #no NULL values

df_test=sgemm_df.iloc[:,0:14]
#checking variable distribution
for index in range(10):
    df_test.iloc[:,index] = (df_test.iloc[:,index]-df_test.iloc[:,index].mean()) / df_test.iloc[:,index].std();
df_test.hist(figsize= (14,16));
plt.figure(figsize=(14,14))
sns.set(font_scale=1)
sns.heatmap(df_test.corr(),cmap='GnBu_r',annot=True, square = True ,linewidths=.5);
plt.title('Variable Correlation')
#Varibale and predictor
y=np.array(sgemm_df["run_class"])

X=np.array(sgemm_df.iloc[:,0:14])

#Train Test Validation Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


#X_train, X_val, y_train, y_val = train_test_split(X_train_80, y_train_80, test_size=0.2, random_state=1)
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
knn=KNeighborsClassifier(n_neighbors=5)
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold


scores = []
n=10
i=0
mean_auc=0
accuracy_test=0
accuracy_train=0
cv = KFold(n_splits=n, random_state=42, shuffle=True)
knn=KNeighborsClassifier(n_neighbors=5)


for train_index, test_index in cv.split(X_train):
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
from sklearn.model_selection import learning_curve
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