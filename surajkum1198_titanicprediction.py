import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('../input/titanic/train.csv')
type(df)
df.info()
df.describe()
df.head(10)
#checking for null values
df.isnull()
#missing values
total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)
#heat map to show which features contain missing values
sns.heatmap(df.isnull(),cmap="YlGnBu",yticklabels=False)
#pie chart visualization whether person survived or not 
TotalPassengers=df['Survived'].value_counts().to_dict()
survived=TotalPassengers[1]
didNotSurvive=TotalPassengers[0]
labels=['Survived','Did Not Survive']
sizes=[survived,didNotSurvive]
fig1,ax1=plt.subplots()
ax1.pie(sizes,labels=labels,shadow=True,startangle=90,autopct='%1.1f%%',colors=['tab:blue','tab:orange'])
ax1.axis('equal')
ax1.title.set_text('Survivors Pie Chart')
plt.show()
#list differnt columns
df.columns.values
#data visualization based on age and gender
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = df[df['Sex']=='female']
men = df[df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')
#survived vs not survived figure
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df,palette="rainbow")
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',palette='rainbow',data=df)
#survivors according to Pclass
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',palette='rainbow',data=df) 
sns.barplot(x='Pclass', y='Survived', data=df)
sns.distplot(df.Pclass)
sns.countplot(x='SibSp',data=df)
#plot shows ages of different Pcalss who survived
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')
sns.swarmplot(x="Pclass", y="Age", data=df, color=".25")
df.head()
df['Cabin'].unique()
#filling the age missing data
data = [df]

for dataset in data:
    mean = df["Age"].mean()
    std = df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = df["Age"].astype(int)
df["Age"].isnull().sum()
df['Embarked'].describe()
#fill Embarked missing data with s as it is the most common one
df['Embarked']=df['Embarked'].fillna('S')
df
total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)
#dropping these features
df=df.drop(['PassengerId','Cabin','Ticket'],axis=1)
df
#encoding 
genders = {"male": 0, "female": 1}
df['Sex'] = df['Sex'].map(genders)
df
#encoding
ports = {"S": 0, "C": 1, "Q": 2}
df['Embarked']=df['Embarked'].map(ports)
df
df['Fare'].astype(int)
df
data = [df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "HighClass": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'HighClass')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
df = df.drop(['Name'], axis=1)

df
#Function for plotting confusion matrix

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    #plt.figure(figsize=[10,10])
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="blue" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
# FUNCTION TO CALCULATE TRUE POSITIVE, TRUE NEGATIVE ,FALSE POSITIVE AND FALSE NEGATIVE 

def perf_measure(y_actual, y_hat):
    y_actual=np.array(y_actual)
    y_hat=np.array(y_hat)
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i] and y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
X_train=df.drop(['Survived'],axis=1)
y_train=df['Survived']
clf_lr= LogisticRegression(solver='liblinear', penalty='l1')
clf_lr.fit(X_train, y_train)
pred_lr=clf_lr.predict(X_train)
clf_lr.score(X_train,y_train)
print(classification_report(y_train,pred_lr))
# VISUALIZNG CONFUSION MATRIX

cnf_matrix_lr=confusion_matrix(y_train,pred_lr)
#print(cnf_matrix_lr)
plot_confusion_matrix(cnf_matrix_lr,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE

probs_lr= clf_lr.predict_proba(X_train)
probs_lr=probs_lr[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_train,probs_lr)
plt.title("AUC-ROC curve--LR",color="green",fontsize=20)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot(fpr,tpr,linewidth=2, markersize=12)
plt.show()
clf_mnb=MultinomialNB(alpha=0.2)

clf_mnb.fit(X_train,y_train)
pred_mnb=clf_mnb.predict(X_train)
acc_mnb=clf_mnb.score(X_train,y_train)

#acc=accuracy_score(y_test,pred)
print("Accuracy : ",acc_mnb)
print(classification_report(y_train,pred_mnb))
# VISUALIZNG CONFUSION MATRIX

cnf_matrix_mnb=confusion_matrix(y_train,pred_mnb)
#print(cnf_matrix_mnb)
plot_confusion_matrix(cnf_matrix_mnb,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE

probs_mnb= clf_mnb.predict_proba(X_train)
probs_mnb=probs_mnb[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_train,probs_mnb)
plt.title("AUC-ROC curve--MNB",color="green",fontsize=20)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot(fpr,tpr,linewidth=2, markersize=12)
plt.show()
clf_knn= KNeighborsClassifier(n_neighbors=49)
clf_knn.fit(X_train,y_train)

pred_knn=clf_knn.predict(X_train)
acc_knn=clf_knn.score(X_train,y_train)

print("Accuracy : ",acc_knn)
print(classification_report(y_train,pred_knn))
# VISUALIZNG CONFUSION MATRIX

cnf_matrix_knn=confusion_matrix(y_train,pred_knn)
#print(cnf_matrix_knn)
plot_confusion_matrix(cnf_matrix_knn,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE

probs_knn= clf_knn.predict_proba(X_train)
probs_knn=probs_knn[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_train,probs_knn)
plt.title("AUC-ROC curve--KNN",color="green",fontsize=20)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot(fpr,tpr,linewidth=2, markersize=12)
plt.show()
clf_svm = svm.SVC(kernel='rbf',probability=True)

clf_svm.fit(X_train,y_train)
pred_svm=clf_svm.predict(X_train)
acc_svm=clf_svm.score(X_train,y_train)

print("Accuracy : ",acc_svm)
print(classification_report(y_train,pred_svm))
# VISUALIZNG CONFUSION MATRIX

cnf_matrix_svm=confusion_matrix(y_train,pred_svm)
#print(cnf_matrix_svm)
plot_confusion_matrix(cnf_matrix_svm,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE

probs_svm= clf_svm.predict_proba(X_train)
probs_svm=probs_svm[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_train,probs_svm)
plt.title("AUC-ROC curve--SVM",color="green",fontsize=20)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot(fpr,tpr,linewidth=2, markersize=12)
plt.show()
clf_dtc=DecisionTreeClassifier(random_state=0)

clf_dtc.fit(X_train,y_train)
pred_dtc=clf_dtc.predict(X_train)
acc_dtc=clf_dtc.score(X_train,y_train)

print("Accuracy : ",acc_dtc)
print(classification_report(y_train,pred_dtc))
# VISUALIZNG CONFUSION MATRIX

cnf_matrix_dtc=confusion_matrix(y_train,pred_dtc)
#print(cnf_matrix_dtc)
plot_confusion_matrix(cnf_matrix_dtc,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE

probs_dtc= clf_dtc.predict_proba(X_train)
probs_dtc=probs_dtc[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_train,probs_dtc)
plt.title("AUC-ROC curve--DTC",color="green",fontsize=20)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot(fpr,tpr,linewidth=2, markersize=12)
plt.show()
clf_rf= RandomForestClassifier(n_estimators=31, random_state=111)

clf_rf.fit(X_train,y_train)
pred_rf=clf_rf.predict(X_train)
acc_rf=clf_rf.score(X_train,y_train)

print("Accuracy : ",acc_rf)
print(classification_report(y_train,pred_rf))
# VISUALIZNG CONFUSION MATRIX

cnf_matrix_rf=confusion_matrix(y_train,pred_rf)
#print(cnf_matrix_rf)
plot_confusion_matrix(cnf_matrix_rf,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE

probs_rf= clf_rf.predict_proba(X_train)
probs_rf=probs_rf[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_train,probs_rf)

plt.title("AUC-ROC curve--RandomForest",color="green",fontsize=20)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot(fpr,tpr,linewidth=2, markersize=12)
plt.show()
from keras import models
from keras.layers import Dense
model=models.Sequential()
model.add(Dense(32,activation='relu',input_shape=(X_train.shape[1],)))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
hist=model.fit(X_train,y_train,batch_size=32,epochs=200)
pred_mlp=model.predict(X_train)
pred_mlp[pred_mlp>=0.5]=1
pred_mlp[pred_mlp<0.5]=0
print(pred_mlp)
acc_mlp=accuracy_score(pred_mlp,y_train)
print(acc_mlp)
print(classification_report(y_train,pred_mlp))
# VISUALIZNG CONFUSION MATRIX

cnf_matrix_mlp=confusion_matrix(y_train,pred_mlp)
#print(cnf_matrix_mlp)
plot_confusion_matrix(cnf_matrix_mlp,[0,1],normalize=False,title="Confusion Matrix")
# PLOTTING AUC-ROC CURVE

probs_mlp= model.predict_proba(X_train)
fpr, tpr, thresholds = metrics.roc_curve(y_train,probs_mlp)
plt.title("AUC-ROC curve--MLP",color="green",fontsize=20)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.plot(fpr,tpr,linewidth=2, markersize=12)
plt.show()
classifiers=[]

classifiers.append(('LogisticRegression',clf_lr))
classifiers.append(('MNB',clf_mnb))
classifiers.append(('KNN',clf_knn))
classifiers.append(('SVM',clf_svm))
classifiers.append(('Desicion Tree',clf_dtc))
classifiers.append(('Random Forest',clf_rf))
classifiers.append(('MLP',model))
result=[]
cnf_matric_parameter=[]
for i,v in classifiers:
    if i=='MLP':
        pred=v.predict(X_train)
        pred[pred>=0.5]=1
        pred[pred<0.5]=0
        #print(pred)
        acc=accuracy_score(y_train,pred)
        precision = precision_score(y_train,pred)
        recall=recall_score(y_train, pred)
        f_measure=f1_score(y_train,pred)
        result.append((i,acc,precision,recall,f_measure))
        
        TP,FP,TN,FN=perf_measure(y_train,pred)
        cnf_matric_parameter.append((i,TP,FP,TN,FN))
        continue
        
    
    pred=v.predict(X_train)
    acc=accuracy_score(y_train,pred)
    precision = precision_score(y_train,pred)
    recall=recall_score(y_train, pred)
    #print(precision)
    f_measure=f1_score(y_train,pred)
    result.append((i,acc,precision,recall,f_measure))
    
    TP,FP,TN,FN=perf_measure(y_train,pred)
    cnf_matric_parameter.append((i,TP,FP,TN,FN))
column_names=['Algorithm','Accuracy','Precision','Recall','F-measure']
df1=pd.DataFrame(result,columns=column_names)
print(df1)
df1.plot(kind='bar', ylim=(0.65,1.0), figsize=(15,6), align='center', colormap="Accent")
plt.xticks(np.arange(8), df1['Algorithm'],fontsize=15)
plt.ylabel('Score',fontsize=20)
plt.title('Distribution by Classifier',fontsize=20)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=20)
test_df=pd.read_csv('../input/titanic/test.csv')
test_df
test_df.describe()
total = test_df.isnull().sum().sort_values(ascending=False)
percent_1 = test_df.isnull().sum()/test_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)
data = [test_df]

for dataset in data:
    mean = df["Age"].mean()
    std = df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = df["Age"].astype(int)
df["Age"].isnull().sum()
test_df=test_df.drop(['PassengerId','Cabin','Ticket'],axis=1)
genders = {"male": 0, "female": 1}
test_df['Sex'] = test_df['Sex'].map(genders)
ports = {"S": 0, "C": 1, "Q": 2}
test_df['Embarked']=test_df['Embarked'].map(ports)
test_df['Fare'].describe()
test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].mean())
test_df['Fare'].astype(int)
data = [test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "HighClass": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'HighClass')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
test_df = test_df.drop(['Name'], axis=1)

total = test_df.isnull().sum().sort_values(ascending=False)
percent_1 = test_df.isnull().sum()/test_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)
test_df.head(10)
X_test=test_df
X_test.shape[0]
pred_mlp=model.predict(X_test)
pred_mlp[pred_mlp>=0.5]=1
pred_mlp[pred_mlp<0.5]=0
pred_mlp.shape

pred_mlp=pred_mlp.flatten()
pred_mlp.shape
results=[]
for i in range(418):
    results.append(int(pred_mlp[i]))
len(results)
results=[int(x) for x in results]
len(results)
result=pd.Series(results,name="Survived")
result.shape
submission=pd.concat([pd.Series(range(892,418+892),name="PassengerId"),result],axis=1)
submission
submission.to_csv('my_submissions',index=False)
my_sub=pd.read_csv('my_submissions')
my_sub

