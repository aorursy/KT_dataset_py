import numpy as np

import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
sns.set() # setting seaborn default for plots
%matplotlib inline
#load data files
train_data=pd.read_csv('../input/widbot/training.csv',sep=';',index_col=None)
test_data=pd.read_csv('../input/widbot/validation.csv',sep=';',index_col=None)
train_data.head()
test_data.head()
len(train_data),len(test_data)
#doing Concatenate to full data for perprocessing takeplace on the both files
df=pd.concat([train_data,test_data])
df.head()
len(df)
df.info()
#check if ther any non values
df.isnull().values.any()
#check the numbers for both class
df['classLabel'].value_counts()
df.dtypes
#convert ',' to '.' to transform it as float number
df['variable2']=df['variable2'].str.replace(',', '.').astype(float)
df['variable3']=df['variable3'].str.replace(',', '.').astype(float)
df['variable8']=df['variable8'].str.replace(',', '.').astype(float)
df.head()
#label encoding
df.replace({"no.": 0, "yes.": 1}, inplace=True)
#Dealing with null values and fill it by median values for string features
#df["variable4"].value_counts().idxmax()
Str_features = ['variable1','variable4', 'variable5', 'variable6','variable7','variable9','variable10','variable12','variable13','variable18']
for i in range(len(Str_features)):
    value=df[Str_features[i]].value_counts().idxmax()
    df[Str_features[i]]=df[Str_features[i]].fillna(value)
df.head()
#check again if there any nan values exist
df.isnull().values.any()
#print columns that contain nan values to deal with it separately
df.isna().any()
#Dealing with null values and repalce it by mean values for numbres features
num_features = ['variable2','variable14', 'variable17']
for i in range(len(num_features)):
    df[num_features[i]]=df[num_features[i]].fillna(df[num_features[i]].mean())
#check again if there still any missing values
df.isnull().values.any()
#dealing with string features to be encoded 
features_to_encode=['variable1','variable4', 'variable5', 'variable6','variable7','variable9','variable10','variable12','variable13','variable18']
df_1=df[features_to_encode]
df_1.head()
#apply hot encoding for those features
df_2 = pd.get_dummies(df_1,drop_first=True)
df_2.head()
df_2.dtypes
#drop all features that encoded form orignal df
df1=df.drop(features_to_encode, axis=1)
df1.head()
df=pd.concat([df_2,df1],axis=1)
df.head()
#check data type for df
df.dtypes
df['classLabel'].plot(kind="density", figsize=(10,5))
df['classLabel'].value_counts()
df_Data=df.drop(columns=['classLabel'])
df_Label=df['classLabel']
df_Label=pd.DataFrame(df_Label)
train_dataX=df_Data.iloc[0:len(train_data),:]
val_data=df_Data.iloc[len(train_data):,:]
y_train=df_Label.iloc[0:len(train_data),:]
y_val=df_Label.iloc[len(train_data):,:]
len(train_dataX),len(val_data),len(y_train),len(y_val)
#Dealing with imbalance problem first method 
from sklearn.utils import resample
X = pd.concat([train_dataX, y_train], axis=1)
X.head()
X['classLabel'].value_counts()
# separate minority and majority classes
no_classLabel= X[X.classLabel==0]
yes_classLabel = X[X.classLabel==1]

# upsample minority
no_classLabel_upsampled = resample(no_classLabel,
                          replace=True, # sample with replacement
                          n_samples=len(yes_classLabel), # match number in majority class
                          random_state=27) # reproducible results

upsampled = pd.concat([yes_classLabel, no_classLabel_upsampled])

# check new class counts
upsampled.classLabel.value_counts()
upsampled['classLabel'].plot(kind="density", figsize=(10,5))
upsampled['classLabel'].value_counts()
upsampled.head()
#shuffle data
upsampled = upsampled.sample(frac=1).reset_index(drop=True)
upsampled.head()
y_train = upsampled.classLabel
X_train = upsampled.drop('classLabel', axis=1)
#make sure of features and label dim
len(X_train),len(y_train)
# from sklearn.preprocessing import MinMaxScaler
# sc_X = MinMaxScaler()
# train_dataX_normalized = sc_X.fit_transform(X_train)
# val_data_normalized = sc_X.transform(val_data)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sklearn.ensemble as ens
svm_clf = SVC(probability=True,kernel='rbf')
svm_clf.fit(X_train,y_train)
y_pred_svm=svm_clf.predict(val_data)
confusion_matrix(y_val, y_pred_svm)   

accuracy_score(y_val, y_pred_svm)

f1_score(y_val, y_pred_svm, average='macro')

from sklearn import metrics
def buildROC(y_val, y_pred_svm):
    fpr, tpr, threshold = metrics.roc_curve(y_val, y_pred_svm)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.gcf().savefig('roc.png')
buildROC(y_val, y_pred_svm)

svm_clf = SVC(probability=True)
svm_parm = {'kernel': ['rbf'], 
            'C': [1, 5, 50], 
            'degree': [3, 5, 7,10], 
       'gamma':[0.04,.1,0.2,.3,.4,.6],
           'random_state': [0,1,2,3,4]}
clfs = [svm_clf]
params = [svm_parm ] 
clf_names = [ 'SVM']
clfs_opt = []
clfs_best_scores = []
clfs_best_param = []
for clf_, param in zip(clfs, params):
    clf = RandomizedSearchCV(clf_, param, cv=5)
    clf.fit(X_train,y_train)
    clfs_opt.append(clf.best_estimator_)
    clfs_best_scores.append(clf.best_score_)
    clfs_best_param.append(clf.best_params_)
arg = np.argmax(clfs_best_scores)
clfs_best_param[arg]
max(clfs_best_scores)

svm_clf_ = SVC(random_state= 3, kernel= 'rbf', gamma= 0.5, degree= 3, C= 50)
svm_clf_.fit(X_train,y_train)
y_pred_svm_=svm_clf_.predict(val_data)
accuracy_score(y_val, y_pred_svm_)


Dec_clfb = DecisionTreeClassifier(criterion='entropy')
Dec_clfb.fit(X_train,y_train)
y_preDec=Dec_clfb.predict(val_data)
accuracy_score(y_val, y_preDec)
confusion_matrix(y_val, y_preDec)
print(classification_report(y_val, y_preDec))
from sklearn.ensemble import GradientBoostingClassifier
p_test = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]}

tuning = GridSearchCV(estimator =GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10), 
            param_grid = p_test, scoring='accuracy',n_jobs=4,iid=False, cv=5)
tuning.fit(X_train,y_train)
tuning.cv_results_, tuning.best_params_, tuning.best_score_
GB=GradientBoostingClassifier(learning_rate=0.15, n_estimators= 100)
GB.fit(X=X_train,y=y_train)
pred_GB= GB.predict(val_data)
accuracy_score(y_val, pred_GB)

logi_clf = LogisticRegression(solver='lbfgs', max_iter=500)
logi_parm = {"C": [0.1, 0.5, 1, 5, 10, 50],
            'random_state': [0,1,2,3,4,5]}
clfs = [logi_clf]
params = [logi_parm] 
clf_names = ['logistic']
clfs_opt = []
clfs_best_scores = []
clfs_best_param = []
for clf_, param in zip(clfs, params):
    clf = RandomizedSearchCV(clf_, param, cv=5)
    clf.fit(X_train,y_train)
    clfs_opt.append(clf.best_estimator_)
    clfs_best_scores.append(clf.best_score_)
    clfs_best_param.append(clf.best_params_)
arg = np.argmax(clfs_best_scores)
clfs_best_param[arg]
max(clfs_best_scores)

gnb_clf = GaussianNB()
gnb_clf.fit(X_train,y_train)
pred_nb= gnb_clf.predict(val_data)
accuracy_score(y_val, pred_nb)


knn_clf = KNeighborsClassifier()
knn_parm = {'n_neighbors':[5, 10, 15, 20], 
            'weights':['uniform', 'distance'], 
            'p': [1,2]}

clfs = [knn_clf]
params = [ knn_parm] 
clf_names = ['KNN']
clfs_opt = []
clfs_best_scores = []
clfs_best_param = []
for clf_, param in zip(clfs, params):
    clf = RandomizedSearchCV(clf_, param, cv=5)
    clf.fit(X_train,y_train)
    clfs_opt.append(clf.best_estimator_)
    clfs_best_scores.append(clf.best_score_)
    clfs_best_param.append(clf.best_params_)
max(clfs_best_scores)

arg = np.argmax(clfs_best_scores)
clfs_best_param[arg]
knn_clf = KNeighborsClassifier(weights ='distance', p= 2, n_neighbors=10)
knn_clf.fit(X_train,y_train)
pred_knn= gnb_clf.predict(val_data)
accuracy_score(y_val, pred_knn)
