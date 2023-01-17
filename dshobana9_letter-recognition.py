import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df_train=pd.read_csv("../input/dataset-of-letter-predictiontraintest/train.csv")
df_test=pd.read_csv("../input/dataset-of-letter-predictiontraintest/test.csv")
#Considering train dataset first and preprocessing
df_train.info()
df_train.describe()
df_train.shape
df_train.head()
df_train.corr()
# "id" column is not necessary to train or analyse the given data. But, storing in some variable for future purpose.
a=df_train.id
a=pd.DataFrame()
del df_train['id']
df_train.head(50)
df_train.isnull().sum()
#The given dataset can be better understood by plotting the data.
#Plotting "letter" column to know the "frequency of each letter" in train dataset and presenting it by a "bar plot" 
df_train['letter'].value_counts().plot(kind='bar')
plt.xlabel('letters')
plt.ylabel('frequency')
plt.title('Alphabetic letters count')
plt.legend(loc='best')

plt.tight_layout()
plt.show()
#Plotting entire "train dataset" using "histograms"
df_train.hist(figsize=[12,12])
plt.show()
df_train.dtypes
#By observing datatypes of each column above, object datatype(categorical column) can  also be seen in the target column.(letter)
#So, it is converted into integer datatype (numerical) so that data can be trained well.
from sklearn.preprocessing import LabelEncoder



le=LabelEncoder()
df_train.iloc[:,0]=le.fit_transform(df_train.iloc[:,0])
df_train.head(10)
#After converting, once checking if the  object datatype column  has been converted to integer data type.
df_train.dtypes
X=df_train.iloc[:,1:]
X.head(10)
Y=df_train.iloc[:,0]
Y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
#Now its time to train and fit our data to the algorithms that we use for classification type of dataset.
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
#lr=LogisticRegression(random_state=0)
#lr.fit(X_train,Y_train)
#lr_pred=lr.predict(X_test)

#print("Logistic Regression:")
#print(" ")
#print("accuracy_score:")
#print(accuracy_score(Y_test,lr_pred))
#print(" ")
#print("confusion_matrix:")
#print(confusion_matrix(Y_test,lr_pred))
#print(" ")
#print("classification _report:")
#print(classification_report(Y_test,lr_pred))
#lda=LinearDiscriminantAnalysis()
#lda.fit(X_train,Y_train)
#lda_pred=lda.predict(X_test)

#print("Linear Discriminant Analysis:")
#print(" ")
#print("accuracy_score:")
#print(accuracy_score(Y_test,lda_pred))
#print(" ")
#print("confusion_matrix:")
#print(confusion_matrix(Y_test,lda_pred))
#print(" ")
#print("classification _report:")
#print(classification_report(Y_test,lda_pred))
#dtc=DecisionTreeClassifier()
#dtc.fit(X_train,Y_train)
#dtc_pred=dtc.predict(X_test)

#print("Decision Tree Classifier:")
#print(" ")
#print("accuracy_score:")
#print(accuracy_score(Y_test,dtc_pred))
#print(" ")
#print("confusion_matrix:")
#print(confusion_matrix(Y_test,dtc_pred))
#print(" ")
#print("classification _report:")
#print(classification_report(Y_test,dtc_pred))
#gnb=GaussianNB()
#gnb.fit(X_train,Y_train)
#gnb_pred=gnb.predict(X_test)

#print("Gaussian Naive Bayes:")
#print(" ")
#print("accuracy_score:")
#print(accuracy_score(Y_test,gnb_pred))
#print(" ")
#print("confusion_matrix:")
#print(confusion_matrix(Y_test,gnb_pred))
#print(" ")
#print("classification _report:")
#print(classification_report(Y_test,gnb_pred))
#knn= KNeighborsClassifier(leaf_size=70,n_neighbors=3)
#knn.fit(X_train,Y_train)
#knn_pred=knn.predict(X_test)

#print("K Nearest Neighbors Classifier:")
#print(" ")
#print("accuracy_score:")
#print(accuracy_score(Y_test,knn_pred))
#print(" ")
#print("confusion_matrix:")
#print(confusion_matrix(Y_test,knn_pred))
#print(" ")
#print("classification _report:")
#print(classification_report(Y_test,knn_pred))
etc= ExtraTreesClassifier(criterion='entropy', min_samples_split=10,n_estimators=500,random_state=42)
etc.fit(X_train,Y_train)
etc_pred=etc.predict(X_test)

print("Extra Trees Classifier:")
print(" ")
print("accuracy_score:")
print(accuracy_score(Y_test,etc_pred))
print(" ")
print("confusion_matrix:")
print(confusion_matrix(Y_test,etc_pred))
print(" ")
print("classification _report:")
print(classification_report(Y_test,etc_pred))
lgb_train = lgb.Dataset(X_train, label = Y_train)
params = {
            'task': 'train',
            'boosting_type': 'goss',
            'objective': 'multiclass',
            'metric':{'multi_logloss'},
            'num_class': 30,
            'num_leaves':70,
            'min_data_in_leaf': 1,
            'learning_rate': 0.1,
            'boost_from_average': True
        }
lgb_cls = lgb.train(params,  lgb_train,  800)
lgb_y_pred = lgb_cls.predict(X_test)
lgb_y_orig = []
for p in lgb_y_pred:
    lgb_y_orig.append(np.argmax(p))
print('LightGBM: ', accuracy_score(Y_test, lgb_y_orig))
rfc=RandomForestClassifier(criterion='entropy',max_depth=200,max_features='auto',min_samples_split=10,n_estimators=300, oob_score=True,random_state=101)
rfc.fit(X_train,Y_train)
rfc_pred=rfc.predict(X_test)

print("Random Forest Classifier:")
print(" ")
print("accuracy_score:")
print(accuracy_score(Y_test,rfc_pred))
print(" ")
print("confusion_matrix:")
print(confusion_matrix(Y_test,rfc_pred))
print(" ")
print("classification _report:")
print(classification_report(Y_test,rfc_pred))
svc=SVC(C=3.0,cache_size=200,decision_function_shape='ovr',kernel='rbf', degree=15,gamma='auto',probability=True,random_state=101,shrinking=True,tol=0.150)

#svc=SVC(C=3,cache_size=200,decision_function_shape='ovr',kernel='rbf', degree=3,gamma='auto',probability=True,random_state=0,shrinking=True,tol=0.157)
#svc=SVC(C=5,random_state=0)
svc.fit(X_train,Y_train)
svc_pred=svc.predict(X_test)

print("SVC :")
print(" ")
print("accuracy_score:")
print(accuracy_score(Y_test,svc_pred))
print(" ")
print("confusion_matrix:")
print(confusion_matrix(Y_test,svc_pred))
print(" ")
print("classification _report:")
print(classification_report(Y_test,svc_pred))
#Till now, train dataset has been considered and  its time to give same kind of treatment to test dataset also.
#As the test dataset has been read already,it is displayed now.
df_test.head()
#In further process,as there is no impact of 'id' column in predicting category of letters, let it be stored in some variable for future purpose.
b=df_test.id
b=pd.DataFrame(b)
del df_test['id']
df_test.isnull().sum()
df_test.dtypes
# SVC algorithm is considered to predict test dataset because ,it is giving more accuracy when compared to other algorithms
svc_pred1=svc.predict(df_test)
svc_pred1
#The submission file has to be presented well with column names and all.
# So,the stored 'id' before and 'letter' values from svc prediction of test data have to be used.
submission_a = pd.DataFrame({'id' : b.id, 'letter' : svc_pred1})
submission_map = {0: "A", 1: "B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J", 10:"K", 11:"L", 12:"M", 13:"N", 14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 20:"U", 21:"V", 22:"W", 23:"X", 24:"Y", 25:"Z"}
submission_a = submission_a.replace({"letter": submission_map})
submission_a.head(5)

submission_a.to_csv('submissiona.csv', index=False)

#Final result is in "submissiona" file.
#Thus letters are categorised from the dataset by using svc algorithm with highest accuracy.

