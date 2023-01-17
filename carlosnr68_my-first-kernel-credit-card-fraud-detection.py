# Import 

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

%matplotlib inline 

RANDOM_SEED = 33

plt.style.use('bmh')
# We started the analysis process, studying the available data

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/creditcard.csv') 
df.head()
df.info()
# Remove unnecessary columns
df2 = df.drop('Time', axis=1)
df2.head()
# density por normed (deprecated)

bins=80
plt.figure(figsize=(20,4))
plt.hist(df2.Class[df2.Class==1],bins=bins,density=True,alpha=0.8,label='Fraud',color='red')
plt.hist(df2.Class[df2.Class==0],bins=bins,density=True,alpha=0.8,label='Not Fraud',color='blue')
plt.legend(loc='upper right')
plt.xlabel('Valor')
plt.ylabel('% de Registros')
plt.title('Transacciones vs Valor')
plt.show()
print(df2['Class'].value_counts())
sns.countplot(x = 'Class', data = df2)

plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
# We generate the correlation matrix and look only at those variables with a high correlation level

corr_base = df2.corr() 
plt.figure(figsize=(12, 10))

sns.heatmap(corr_base[(corr_base >= 0.5) | (corr_base <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
# We study the rest of variables: Class= 1:Fraud , Class= 0:No Fraud
y = df2.Class
x = df2.drop('Class',axis=1)
#PCA      

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=29, whiten=True)
sklearn_pca.fit(x)
features_pca = pd.DataFrame(data = sklearn_pca.transform(x))
n_dim = 29
plt.figure(figsize=(12, 5))
rects1 = plt.bar(np.arange(n_dim),sklearn_pca.explained_variance_, color='r')
print(sklearn_pca.explained_variance_) 
# Group of features. The generation of these graphs takes some computing time.

x_scaled=(x-x.min())/(x.max()-x.min()) 
sub_df1=pd.concat([y,x_scaled.iloc[:,0:10]],axis=1)
sub_df2=pd.concat([y,x_scaled.iloc[:,10:20]],axis=1)
sub_df3=pd.concat([y,x_scaled.iloc[:,20:30]],axis=1)

sub_df11=pd.melt(sub_df1,id_vars="Class",var_name="Variable",value_name='Valor')
sub_df22=pd.melt(sub_df2,id_vars="Class",var_name="Variable",value_name='Valor')
sub_df33=pd.melt(sub_df3,id_vars="Class",var_name="Variable",value_name='Valor')

plt.figure(figsize=(20,8))
sns.violinplot(x="Variable",y="Valor",hue="Class",data=sub_df11, split=True)
plt.figure(figsize=(20,8))
sns.violinplot(x="Variable",y="Valor",hue="Class",data=sub_df22, split=True)
plt.figure(figsize=(20,8))
sns.violinplot(x="Variable",y="Valor",hue="Class",data=sub_df33, split=True)
plt.figure(figsize=(20,8))

count_classes = pd.value_counts(df2['Class'], sort = True).sort_index()
labels = 'Fraud', 'Not Fraud'
sizes = [count_classes[1]/(count_classes[1]+count_classes[0]), count_classes[0]/(count_classes[1]+count_classes[0])]
explode = (0, 0.5,)  
colors = ['red', 'lightblue']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, colors=colors, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=45)
ax1.axis('equal')  
plt.title("Distribution of the Dataset in labeled classes")
plt.show()
df2.shape
tt = df2.describe().transpose()
tt[(tt['max']>1) & (tt['min']< -1)]

plt.figure(figsize=(20,8))
plt.hist(df2.Amount, bins=50)
# We normalize all the columns

columns_to_norm = ['V1','V2','V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 
                   'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler() 
df2[columns_to_norm]=min_max_scaler.fit_transform(df2[columns_to_norm])
tt = df2.describe().transpose()
tt[(tt['max']>1) & (tt['min']< -1)]
# We generate a help function for the rest of modules of face to visualize the arrays of confusion.

from sklearn.metrics import confusion_matrix, classification_report, auc, precision_recall_curve, roc_curve
def plot_confusion_matrix(y_test, pred):
    
    y_test_legit = y_test.value_counts()[0]
    y_test_fraud = y_test.value_counts()[1]
    
    cfn_matrix = confusion_matrix(y_test, pred)
    cfn_norm_matrix = np.array([[1.0 / y_test_legit,1.0/y_test_legit],[1.0/y_test_fraud,1.0/y_test_fraud]])
    norm_cfn_matrix = cfn_matrix * cfn_norm_matrix

    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1,2,1)
    sns.heatmap(cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ax = fig.add_subplot(1,2,2)
    sns.heatmap(norm_cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)

    plt.title('Standardized Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    print('---Report de classifition---')
    print(classification_report(y_test,pred))
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df2, test_size=0.2, random_state=RANDOM_SEED)
Y_train = X_train['Class']
X_train = X_train.drop(['Class'], axis=1)
Y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)
from sklearn import metrics

sgd_clf=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=42, shuffle=True,
       tol=None, verbose=0, warm_start=False)

sgd_clf.fit(X_train, Y_train) 
Y_train_predicted=sgd_clf.predict(X_train)
Y_test_predicted=sgd_clf.predict(X_test)

plot_confusion_matrix(Y_test, Y_test_predicted)
from sklearn.utils import shuffle

Train_Data= pd.concat([X_train, Y_train], axis=1)
X_1 =Train_Data[ Train_Data["Class"]==1 ]
X_0=Train_Data[Train_Data["Class"]==0]

X_0=shuffle(X_0,random_state=42).reset_index(drop=True)
X_1=shuffle(X_1,random_state=42).reset_index(drop=True)

ALPHA=1.15 

X_0=X_0.iloc[:round(len(X_1)*ALPHA),:]
data_d=pd.concat([X_1, X_0])

count_classes = pd.value_counts(data_d['Class'], sort = True).sort_index()
labels = 'Fraud', 'Not Fraud'
sizes = [count_classes[1]/(count_classes[1]+count_classes[0]), count_classes[0]/(count_classes[1]+count_classes[0])]
explode = (0, 0.05,)
colors = ['red', 'lightblue']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, colors=colors, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=45)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Distribución del dataset en clases")
plt.show()
data_d.head()
data_d.shape
# Convertimos el dataframe a matriz(array).
dataset=data_d.values
Y_d=data_d['Class']
X_d=data_d.drop(['Class'],axis=1)
sgd_clf_d=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=42, shuffle=True,
       tol=None, verbose=0, warm_start=False)

sgd_clf_d.fit(X_d, Y_d) 
Y_test_predicted=sgd_clf_d.predict(X_test)

plot_confusion_matrix(Y_test, Y_test_predicted)
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
rf =RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0, n_jobs=-1)
rf.fit(X_d, Y_d) 
Y_test_predicted=rf.predict(X_test)

plot_confusion_matrix(Y_test, Y_test_predicted)