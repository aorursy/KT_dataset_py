import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df_train=pd.read_csv('train (1).csv')
df_test=pd.read_csv('test (1).csv')
#Considering train dataset first and preprocessing
df_train.info()
df_train.describe()
#If correlation is seen between two columns very closely, then one of them can be deleted as both contribute the same to the dataset if they are correlated.
df_train.corr()
df_train.head()
#We can see many columns and using most of the features,we can classify people to either <=50k or >50k (i.e 0 or 1)
# "id" column is not necessary to train or analyse the given data but, storing in some variable for future purpose.
a= df_train['id']
a=pd.DataFrame(a)
a.head()
del df_train['id']
df_train.head(50)
df_train.isnull().sum()
# "race" column doesn't show any impact on predicting person's salary  range and so it can deleted.
del df_train['race']
df_train.head()
#The given dataset can be better understood by plotting the data.
#Plotting "occupation" column to know the "frequency of each occupation" and presenting it by a "pie plot" 
df_train['occupation'].value_counts().plot(kind='pie',label='occupation')
plt.xlabel('different occupations')
plt.ylabel('frequency')
plt.title('Different occupations count')
plt.legend(loc='best')
plt.axis('equal')
plt.tight_layout()
plt.show()
#Plotting "earn" column to know the "frequency of earnings of persons"(i.e <=50k or >50k) and presenting it by a "bar plot" 
df_train['earn'].value_counts().plot(kind='bar',label='earn')
plt.xlabel('salary count')
plt.ylabel('frequency')
plt.title('no.of persons salary count')
plt.legend()
plt.show()
#Plotting "sex" column to know the "frequency of genders in dataset" and presenting it by a "bar plot" 
df_train['sex'].value_counts().plot(kind='bar',label='sex')
plt.xlabel('sex')
plt.ylabel('frequency')
plt.title('count of male and female')
plt.legend()
plt.show()
df_train.dtypes
#By observing datatypes of each column above, object datatypes (categorical columns) can  also be seen.
#So, they are converted into integer datatypes (numerical) so that data can be trained well.
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_train.iloc[:,1]=le.fit_transform(df_train.iloc[:,1])
df_train.iloc[:,3]=le.fit_transform(df_train.iloc[:,3])
df_train.iloc[:,5]=le.fit_transform(df_train.iloc[:,5])
df_train.iloc[:,6]=le.fit_transform(df_train.iloc[:,6])
df_train.iloc[:,7]=le.fit_transform(df_train.iloc[:,7])
df_train.iloc[:,8]=le.fit_transform(df_train.iloc[:,8])
df_train.iloc[:,12]=le.fit_transform(df_train.iloc[:,12])

#After converting, once checking whether still any  object datatype columns are left over.
df_train.dtypes
df_train.head(10)
df_train.shape
X=df_train.iloc[:,:-1];X.head()
Y=df_train['earn']
#After converting some of the object datatype columns into int datatype, let's plot some of the features so that its easy to analyse.
#Plotting "earn" and "sex" columns to know which gender is occupying which range of salaries to the most and it is presented by "histogram"
df_train.plot(x='earn',y='sex',kind='hist')
plt.xlabel('salaries count')
plt.ylabel('frequency of male and female')
plt.title('frequency of male and female having salaries <=50k or >50k')
plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
# If we see data in "X", all the values in columns are varying a lot.
# so,data in "X" should be scaled.
#But without scaling,the accuracy  that we obtain for this dataset is more.so its not scaled.
#As the 'X' data is splitted into 'X_train','X_test' scaling is applied separately.
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#X_train=sc.fit_transform(X_train)
#X_test=sc.fit_transform(X_test)
#Now its time to train and fit our data to the algorithms that we use for classification type of dataset.
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

#performance evaluators like accuracy_score,confusion_matrix,classification_report to evaluate algorithms. 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
lr=LogisticRegression(random_state=0)
lr.fit(X_train,Y_train)
lr_pred=lr.predict(X_test)

print("Logistic Regression:")
print(" ")
print("accuracy_score:")
print(accuracy_score(Y_test,lr_pred))
print(" ")
print("confusion_matrix:")
print(confusion_matrix(Y_test,lr_pred))
print(" ")
print("classification _report:")
print(classification_report(Y_test,lr_pred))
lda=LinearDiscriminantAnalysis()
lda.fit(X_train,Y_train)
lda_pred=lda.predict(X_test)

print("Linear Discriminant Analysis:")
print(" ")
print("accuracy_score:")
print(accuracy_score(Y_test,lda_pred))
print(" ")
print("confusion_matrix:")
print(confusion_matrix(Y_test,lda_pred))
print(" ")
print("classification _report:")
print(classification_report(Y_test,lda_pred))
dtc=DecisionTreeClassifier()
dtc.fit(X_train,Y_train)
dtc_pred=dtc.predict(X_test)

print("Decision Tree Classifier:")
print(" ")
print("accuracy_score:")
print(accuracy_score(Y_test,dtc_pred))
print(" ")
print("confusion_matrix:")
print(confusion_matrix(Y_test,dtc_pred))
print(" ")
print("classification _report:")
print(classification_report(Y_test,dtc_pred))
rfc=RandomForestClassifier(criterion='entropy',max_depth=100,max_features='log2',min_samples_split=10,n_estimators=750, oob_score=True,random_state=101)
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
gnb=GaussianNB()
gnb.fit(X_train,Y_train)
gnb_pred=gnb.predict(X_test)

print("Gaussian Naive Bayes:")
print(" ")
print("accuracy_score:")
print(accuracy_score(Y_test,gnb_pred))
print(" ")
print("confusion_matrix:")
print(confusion_matrix(Y_test,gnb_pred))
print(" ")
print("classification _report:")
print(classification_report(Y_test,gnb_pred))
knn= KNeighborsClassifier( n_neighbors=10)
knn.fit(X_train,Y_train)
knn_pred=knn.predict(X_test)

print("K Nearest Neighbors Classifier:")
print(" ")
print("accuracy_score:")
print(accuracy_score(Y_test,knn_pred))
print(" ")
print("confusion_matrix:")
print(confusion_matrix(Y_test,knn_pred))
print(" ")
print("classification _report:")
print(classification_report(Y_test,knn_pred))
svc=SVC(C=3.0,decision_function_shape='ovo',kernel='rbf', degree=7,gamma='auto',random_state=0)
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
etc= ExtraTreesClassifier(criterion='gini', n_estimators=1000,random_state=42)
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

#Till now, train dataset has been considered and now its time to give same kind of treatment to test dataset also.
#As the test dataset has been read already,it is displayed now.
df_test.head()
df_test.shape
#In further process,as there is no impact of 'id' column in predicting salaries range, let it be stored in some variable for future purpose.
b=df_test.id
b=pd.DataFrame(b)
#There is no impact of "race" column in predicting salaries range either <=50k or >50k.
del df_test['race']
del df_test['id']
df_test.isnull().sum()
df_test.dtypes
#By observing datatypes of each column, object datatypes (categorical columns) can  also be seen.
#So, they are converted into integer datatypes (numerical) so that data can predict well without leading to any confusion.
#As the label encoder is already imported and used for train dataset, its enough to use it now for test dataset.
df_test.iloc[:,1]=le.fit_transform(df_test.iloc[:,1])
df_test.iloc[:,3]=le.fit_transform(df_test.iloc[:,3])
df_test.iloc[:,5]=le.fit_transform(df_test.iloc[:,5])
df_test.iloc[:,6]=le.fit_transform(df_test.iloc[:,6])
df_test.iloc[:,7]=le.fit_transform(df_test.iloc[:,7])
df_test.iloc[:,8]=le.fit_transform(df_test.iloc[:,8])
df_test.iloc[:,12]=le.fit_transform(df_test.iloc[:,12])

df_test.head()
# In the same way as 'X' in train dataset is scaled, here also test dataset is scaled.
#standard scaler preprocessing technique is already imported and so now its used.
#sc=StandardScaler()
#df1_test=sc.fit_transform(df_test)
#df1_test
#As it is in an array, using dataframe and storing in other variable.
#df2_test=pd.DataFrame(df1_test)
df2_test=pd.DataFrame(df_test)
df2_test.head()
# As dataframe has no names for columns,previous columns are given to new scaled dataset which is df2_test
#df2_test.columns=df_test.columns
#df2_test.head()
# Considering "Random Forest Classifier" algorithm to predict test dataset
#Using 'rfc' which is a random forest Classifier algorithm to predict test dataset as its accuracy(0.8627) is higher when compared to other algorithms.
rfc_pred1=rfc.predict(df2_test)
rfc_pred1

submission_a = pd.DataFrame({'id' : b.id, 'earn' : rfc_pred1})
#submission_map = {0: " <=50k.", 1: " >50k."}
#submission_a = submission_a.replace({"earn": submission_map})
submission_a.head(5)



submission_a.to_csv('submissiona.csv', index=False)
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, label = Y_train)
params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric':{'multi_logloss'},
            'num_class': 10,
            'num_leaves': 30,
            'min_data_in_leaf': 1,
            'learning_rate': 0.1,
            'boost_from_average': True
        }
lgb_cls = lgb.train(params,  lgb_train,  100)
lgb_y_pred = lgb_cls.predict(X_test)
lgb_y_orig = []
for p in lgb_y_pred:
    lgb_y_orig.append(np.argmax(p))
print('LightG: ', accuracy_score(Y_test, lgb_y_orig))
# Since the above algorithm has given highest accuracy the predictions that this algorithm gives is taken into 'submissionb' file
lgb_y_pred = lgb_cls.predict(df2_test)
lgb_y_original = []
for p in lgb_y_pred:
    lgb_y_original.append(np.argmax(p))
#The submission file has to be presented well with column names and all.
# So,the stored 'id' before and 'earn' values from rfc prediction of test data have to be used.
submission_b= pd.DataFrame({'id' : b.id, 'earn' : lgb_y_original})
submission_b.head(5)

submission_b.to_csv('submissionb.csv', index=False)
#Final result is in "submissionb" file.
#Thus people are classified of either <=50k or >50k i.e(0 or 1) and lightgbm algorithm is used.


