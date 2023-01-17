#Importing all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.svm import *
#Loading dataset and reading the first few columns
df = pd.read_csv('../input/diabetescsv/diabetes.csv')
df.head()
#  Check the number of rows and columns in the dataset
df.shape
# Get basic statistics - gives statistics of only numerical columns in the dataset. 
#714 columns for age indicates missing values in age column
df.describe()
df.groupby('Glucose').size()
df.groupby('BloodPressure').size()
df.groupby('BMI').size()
sns.set_style('darkgrid')
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
n_rows = 2
n_cols = 4

fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.5, n_rows*3.5))

for r in range(0, n_rows):
    for c in range(0,n_cols):
        i = r*n_cols +c  
        ax = axs[r][c] 
        
        sns.distplot(df[cols[i]], ax = ax)
        
    plt.tight_layout()

sns.set_style('darkgrid')
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
n_rows = 2
n_cols = 4

fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.5, n_rows*3.5))

for r in range(0, n_rows):
    for c in range(0,n_cols):
        i = r*n_cols +c  
        ax = axs[r][c] 
        
        sns.boxplot(df[cols[i]], ax = ax)
        
    plt.tight_layout()
sns.pairplot(df, height = 3, hue = 'Outcome', diag_kind = 'kde')
plt.figure(figsize=(10,10))
plt.show()

df_correlation = df.corr()
df_correlation
fig, ax = plt.subplots(figsize = (16,6))
sns.heatmap(df_correlation, annot = True, annot_kws= {'size': 12})
filt1=(df['Outcome']==0)&(df['Glucose']==0)
filt2=(df['Outcome']==1)&(df['Glucose']==0)
df.loc[filt1,'Glucose']=110
df.loc[filt2,'Glucose']=141

filt1=(df['Outcome']==0)&(df['BloodPressure']==0)
filt2=(df['Outcome']==1)&(df['BloodPressure']==0)
df.loc[filt1,'BloodPressure']=68
df.loc[filt2,'BloodPressure']=71

filt1=(df['Outcome']==0)&(df['SkinThickness']==0)
filt2=(df['Outcome']==1)&(df['SkinThickness']==0)
df.loc[filt1,'SkinThickness']=20
df.loc[filt2,'SkinThickness']= 22

filt1=(df['Outcome']==0)&(df['Insulin']==0)
filt2=(df['Outcome']==1)&(df['Insulin']==0)
df.loc[filt1,'Insulin']=69
df.loc[filt2,'Insulin']=100

filt1=(df['Outcome']==0)&(df['BMI']==0)
filt2=(df['Outcome']==1)&(df['BMI']==0)
df.loc[filt1,'BMI']=30
df.loc[filt2,'BMI']=37
df.head()
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']
print("Shape of X :", X.shape)
print("Shape of y :",y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
def models(X_train, y_train):
    
    #1st we will use Logistic regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, y_train)

    from sklearn.linear_model import RidgeClassifier
    clf = RidgeClassifier()
    clf.fit(X_train, y_train)
    
    # Using KNeighbors 
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski', p=2)
    knn.fit(X_train,y_train)
    
    # Using SVM (linear kernel)
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X_train,y_train)
    
    # Using SVM (RBF kernel)
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train,y_train)
    
    #Use GaussianNB
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train,y_train)
    
    #Using Decision tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, y_train)
    
    #Using Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state =0)
    forest.fit(X_train, y_train)

    import xgboost as xgb
    xb = xgb.XGBClassifier(random_state=0)
    xb.fit(X_train,y_train)

    print('[0]Logistic Regression Training Accuracy', log.score(X_train, y_train))
    print('[1]K Nearest Neighbors Regression Training Accuracy', knn.score(X_train, y_train))
    print('[2]SVC Linear Regression Training Accuracy', svc_lin.score(X_train, y_train))
    print('[3]SVC RBF Regression Training Accuracy', svc_rbf.score(X_train, y_train))
    print('[4]Gaussian Regression Training Accuracy', gauss.score(X_train, y_train))
    print('[5]Decision Tree Regression Training Accuracy', tree.score(X_train, y_train))
    print('[6]Random Forest Regression Training Accuracy', forest.score(X_train, y_train))
    print('[7]Ridge Classifier Training Accuracy', clf.score(X_train, y_train))  
    print('[8]XGB Classifier Training Accuracy', xb.score(X_train, y_train)) 
      
    return log, knn, svc_lin, svc_rbf, gauss, tree, forest, clf,xb
models = models(X_train, y_train)
from sklearn.metrics import confusion_matrix

for i in range(len(models)):
    cm = confusion_matrix(y_test, models[i].predict(X_test))
    
    #Extracting TN, FN, TP, FP
    TN, FN, TP, FP = confusion_matrix(y_test, models[i].predict(X_test)).ravel()
    test_score = (TP + TN)/(TP + TN + FN + FP)
    print(cm)
    print('Model[{}] Testing Accuracy = "{}"'.format(i, test_score))
    print()
from sklearn.linear_model import LogisticRegression 
log = LogisticRegression(random_state=0)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
from sklearn.linear_model import RidgeClassifier
rc = RidgeClassifier()
from sklearn.ensemble import VotingClassifier
vt = VotingClassifier(estimators = [('log',log),("tree",tree),('rc',rc)],voting="hard", flatten_transform=True)
vt.fit(X_train,y_train)
vt.score(X_test, y_test)
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(random_state =0)
gb_clf.fit(X_train,y_train)
gb_score = gb_clf.score(X_test,y_test)
gb_score
import xgboost as xgb
xgb_model = xgb.XGBClassifier(random_state=0)
xgb_model.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
y_pred = xgb_model.predict(X_test)
accuracy_score(y_test,y_pred)
!pip install catboost
from catboost import CatBoostClassifier, Pool
cat_model=CatBoostClassifier()
cat_model.fit(X_train, y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_pred,y_test)
print("Test score is ",score)
pred = cat_model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
from sklearn.metrics import roc_curve,auc
fpr, tpr, thresholds = roc_curve(y_test,pred)
auc_vt = auc(fpr, tpr)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='CatBoost Classifier (auc = %0.3f)'% auc_vt )
plt.xlabel('True Positive Rate')
plt.xlabel('False Positive')
plt.title('CatBoost ROC curve')
plt.legend()
plt.show()
