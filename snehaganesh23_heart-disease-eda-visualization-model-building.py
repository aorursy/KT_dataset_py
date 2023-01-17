import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("../input/heart-disease-combined/Comb_heart_data.csv")
df.head()
df.tail()
df.shape
# Check continuous columns
con=df._get_numeric_data().columns
print("No of Continuous columns:",len(df._get_numeric_data().columns),"\n\n",con)
df.info()
col_name=list(df.columns)
col_name 
#These are all the columns in the dataset
for i in col_name:
    print(df[i].value_counts(),"\n")
df.isna().sum()
df=df.replace("?",np.NaN)
df.isna().sum()
df.describe(include="all") #All features included, categorical and continuous 
df.describe() # Only continuous values
# In this case it would be same as above since there are only continuous columns
df.info()
import missingno as msn
msn.matrix(df)
msn.heatmap(df)
#missingno.heatmap visualizes the correlation matrix about the locations of missing values in columns.
msn.bar(df)
mode_col=['Fast_bld_sugar','Rest_Ecg','Ex_Angina','Slope','Colored_Vessels','Thalassemia']
for col in mode_col:
    df[col].fillna(df[col].mode()[0],inplace=True)
    
df.isna().sum()
df.fillna(df.mean()[0],inplace=True)
df.isna().sum()
column_list = df.columns
for i in column_list:
    print("Values of",i,"column\n",df[i].unique())
    print("--------------\n")
df['Sex'][df['Sex'] == 0] = 'female'
df['Sex'][df['Sex'] == 1] = 'male'

df['ChestPain'][df['ChestPain'] == 1] = 'typical angina'
df['ChestPain'][df['ChestPain'] == 2] = 'atypical angina'
df['ChestPain'][df['ChestPain'] == 3] = 'non-anginal pain'
df['ChestPain'][df['ChestPain'] == 4] = 'asymptomatic'

df['Fast_bld_sugar'][df['Fast_bld_sugar'] == '0'] = 'lower than 120mg/ml'
df['Fast_bld_sugar'][df['Fast_bld_sugar'] == '1'] = 'greater than 120mg/ml'

df['Rest_Ecg'][df['Rest_Ecg'] == '0'] = 'normal'
df['Rest_Ecg'][df['Rest_Ecg'] == '1'] = 'ST-T wave abnormality'
df['Rest_Ecg'][df['Rest_Ecg'] == '2'] = 'left ventricular hypertrophy'

df['Ex_Angina'][df['Ex_Angina'] == '0'] = 'no'
df['Ex_Angina'][df['Ex_Angina'] == '1'] = 'yes'

df['Slope'][df['Slope'] == '1'] = 'upsloping'
df['Slope'][df['Slope'] == '2'] = 'flat'
df['Slope'][df['Slope'] == '3'] = 'downsloping'

# Values of Colored_Vessels column
#  ['0' '3' '2' '1']
# ca: number of major vessels (0-3) colored by flourosopy

df['Thalassemia'][df['Thalassemia'] == '3'] = 'normal'
df['Thalassemia'][df['Thalassemia'] == '6'] = 'fixed defect'
df['Thalassemia'][df['Thalassemia'] == '7'] = 'reversable defect'

df['Target'][df['Target'] == 2] = 1
df['Target'][df['Target'] == 3] = 1
df['Target'][df['Target'] == 4] = 1
df.info()
df.head()
df=df.replace(-0.9,df.mean())
# In the UCI data repository it is mentioned that missing 
# values are distinguised with -0.9, so we replace this 
# value also with the mean value.
#Lets check for missing values once again
msn.bar(df,color='Purple')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
total = len(df['Target'])
ax = sns.countplot(y="Target", data=df, palette="hls")
ax.set_title('Total percentage of people with and without Heart Disease')
plt.xlabel('No of people')

total = len(df['Target'])
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

plt.show()
df.sample(4)
total = len(df['Target'])
sns.set(style="darkgrid",palette="husl")
ax = sns.countplot(x="ChestPain", hue="Target", data=df) 
ax.set_title("Percentage of People with chest pain having heart disease")
ax.set(xlabel="ChestPain",ylabel="Target")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.1%}'.format(height/total),
            ha="center") 
total = len(df['Target'])
sns.set(style="darkgrid",palette="bright")
ax = sns.countplot(x="Fast_bld_sugar", hue="Target", data=df) 
ax.set_title("Percentage of People with high to low blood sugar having heart disease")
ax.set(xlabel="Fast_bld_sugar",ylabel="Target")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.1%}'.format(height/total),
            ha="center") 
total = len(df['Target'])
sns.set(style="darkgrid",palette="colorblind")
ax = sns.countplot(x="Rest_Ecg", hue="Target", data=df) 
ax.set_title("Percentage of People with normal to high ecg rates having heart disease")
ax.set(xlabel="Rest_Ecg",ylabel="Target")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.1%}'.format(height/total),
            ha="center") 
total = len(df['Target'])
sns.set(style="darkgrid",palette="PiYG")
ax = sns.countplot(x="Slope", hue="Target", data=df) 
ax.set_title("Percentage of People with slope level having heart disease")
ax.set(xlabel="Slope",ylabel="Target")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.1%}'.format(height/total),
            ha="center") 
total = len(df['Target'])
sns.set(style="darkgrid",palette="RdGy")
ax = sns.countplot(x="Thalassemia", hue="Target", data=df) 
ax.set_title("Percentage of People with fixed, normal, reversable thalassemia rates having heart disease")
ax.set(xlabel="Thalassemia",ylabel="Target")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.1%}'.format(height/total),
            ha="center") 
total = len(df['Sex'])
ax = sns.countplot(y="Sex", data=df, palette="Blues")
ax.set_title('Percentage of Male and Female')
plt.xlabel('No of people')

total = len(df['Sex'])
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

plt.show()
total = len(df['Target'])
sns.set(style="darkgrid",palette="mako_r")
ax = sns.countplot(x="Sex", hue="Target", data=df) 
ax.set_title("Percentage of People having heart disease according to gender")
ax.set(xlabel="Sex",ylabel="Target")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.1%}'.format(height/total),
            ha="center") 
df.sample(1)
total = len(df['Target'])
sns.set(style="darkgrid",palette="PRGn")
ax = sns.countplot(x="Ex_Angina", hue="Target", data=df) 
ax.set_title("Percentage of People with Angina having heart disease")
ax.set(xlabel="Ex_Angina",ylabel="Target")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.1%}'.format(height/total),
            ha="center") 
pd.crosstab(df.Age,df.Target).plot(kind="bar",figsize=(20,6),cmap='copper')
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()
df.info()
df.sample(1)
df['Rest_bp']=df['Rest_bp'].astype(int)
df['Cholestrol']=df['Cholestrol'].astype(int)
df['St_Depr']=df['St_Depr'].astype(float)
df['Max_Rt']=df['Max_Rt'].astype(int)
sns.distplot(df['Age'],color="sienna")
sns.distplot(df['Rest_bp'],color="cadetblue")
sns.distplot(df['Cholestrol'],color="black")
sns.distplot(df['St_Depr'],color="darkred")
sns.distplot(df['Max_Rt'],color="coral")
df.sample(1)
df.info()
df = pd.concat([df, pd.get_dummies(df['Sex'])], axis=1)
df.drop('Sex',axis=1,inplace=True)
df = pd.concat([df, pd.get_dummies(df['ChestPain'])], axis=1)
df.drop('ChestPain',axis=1,inplace=True)
df = pd.concat([df, pd.get_dummies(df['Fast_bld_sugar'])], axis=1)
df.drop('Fast_bld_sugar',axis=1,inplace=True)
df = pd.concat([df, pd.get_dummies(df['Rest_Ecg'])], axis=1)
df.drop('Rest_Ecg',axis=1,inplace=True)
df = pd.concat([df, pd.get_dummies(df['Ex_Angina'])], axis=1)
df.drop('Ex_Angina',axis=1,inplace=True)
df = pd.concat([df, pd.get_dummies(df['Slope'])], axis=1)
df.drop('Slope',axis=1,inplace=True)
df = pd.concat([df, pd.get_dummies(df['Thalassemia'])], axis=1)
df.drop('Thalassemia',axis=1,inplace=True)
df = pd.concat([df, pd.get_dummies(df['Colored_Vessels'])], axis=1)
df.drop('Colored_Vessels',axis=1,inplace=True)
df.columns
df.shape
df.sample(3)
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
X = df.drop('Target',axis=1)
Y = df['Target']
# splitting data in train and test
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.30, random_state = 10)
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_train=pd.DataFrame(X_train,columns=X.columns)
X_valid=sc.transform(X_valid)
# visualizing the results

from yellowbrick.target import ClassBalance
visualizer = ClassBalance(labels=[0, 1])

visualizer.fit(y_train,y_valid)
visualizer.ax.set_xlabel("Classes")
visualizer.ax.set_ylabel("Amount of Occurrences of Class")
visualizer.show()
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import roc_curve , auc
from sklearn.svm import SVC
svm_classifier= SVC(probability=True, kernel='rbf')
svm_classifier.fit(X_train,y_train)
# Predict
y_pred_svm= svm_classifier.predict(X_valid)

#Classification Report
print(classification_report(y_valid,y_pred_svm))
mat_svm = confusion_matrix(y_valid, y_pred_svm, labels = [1,0])
sns.heatmap(mat_svm.T,  annot=True, fmt='d', cbar=False,
          xticklabels=['Yes','No'],
          yticklabels=['Yes','No'] )
plt.xlabel('true label')
plt.ylabel('predicted label')
print(mat_svm)
TP=mat_svm[0,0]
FN=mat_svm[0,1]
FP=mat_svm[1,0]
TN=mat_svm[1,1]
Recall=TP/(TP+FN)
print("Recall: ",Recall)
Precision=TP/(TP+FP)
print("Precision: ",Precision)
FM=(2*Recall*Precision)/(Recall+Precision)
print("F-Measure: ",FM)
y_pred_svm_proba=svm_classifier.predict_proba(X_valid)[:,1]
fpr_svm, tpr_svm, _svm = roc_curve(y_valid, y_pred_svm_proba)
roc_auc=auc(fpr_svm,tpr_svm)

#Now Draw ROC using fpr , tpr
plt.plot([0, 1], [0, 1], 'k--',label='Random')

plt.plot(fpr_svm,tpr_svm,label='ROC curve (area = %0.2f)' %roc_auc)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

plt.title('SVM ROC curve')
plt.legend(loc='best')
svm_classifier.score(X_valid,y_valid)
svm_classifier.score(X_train,y_train)
from yellowbrick.classifier import ClassificationReport
svccr = ClassificationReport(SVC(probability=True))
svccr.fit(X_train, y_train)
svccr.score(X_valid, y_valid)
svccr.show()
from sklearn.linear_model import LogisticRegression

# initiating the classifier and training the model

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
# Predicting the test set results and calculating the accuracy

y_pred_logit = classifier.predict(X_valid)
classifier.score(X_valid, y_valid)
classifier.score(X_train, y_train)
# confusion matrix

from sklearn.metrics import confusion_matrix

matrix_logit = confusion_matrix(y_valid, y_pred_logit)
print(matrix_logit)
# Classification Report

from sklearn.metrics import classification_report
print(classification_report(y_valid, y_pred_logit))
sns.heatmap(matrix_logit,  annot=True, fmt='d', cbar=False,
          xticklabels=['Yes','No'],
          yticklabels=['Yes','No'] )
plt.xlabel('true label')
plt.ylabel('predicted label')
# Compute precision, recall, F-measure and support

TP=matrix_logit[0,0]
FN=matrix_logit[0,1]
FP=matrix_logit[1,0]
TN=matrix_logit[1,1]

Precision=TP/(TP+FP)
print("Precision: ",Precision)

Recall=TP/(TP+FN)
print("Recall: ",Recall)

FM=(2*Recall*Precision)/(Recall+Precision)
print("F-Measure: ",FM)
logit_roc_auc=classifier.predict_proba(X_valid)[:,1]
fpr,tpr,threshold=roc_curve(y_valid,logit_roc_auc)
roc_auc=auc(fpr,tpr)
plt.figure()

# ROC
plt.plot(fpr,tpr,'g',label='Logistic Regression (AUC = %0.2f)'% roc_auc)

# random FPR and TPR
plt.plot([0,1],[0,1],'r--')

# title and label
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression-Receiver operating characteristic')
plt.legend(loc='lower right')
plt.show()

print('The AUC:', auc(fpr,tpr))
from yellowbrick.classifier import ClassificationReport
lrcr = ClassificationReport(LogisticRegression(max_iter=1000))
lrcr.fit(X_train, y_train)
lrcr.score(X_valid, y_valid)
lrcr.show()
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
from sklearn import metrics
y_pred_knn = classifier.predict(X_valid)
classifier.score(X_valid, y_valid)
classifier.score(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix
mat_knn=confusion_matrix(y_valid, y_pred_knn)
print(mat_knn)
print(classification_report(y_valid, y_pred_knn))
sns.heatmap(mat_knn,  annot=True, fmt='d', cbar=False,
          xticklabels=['Yes','No'],
          yticklabels=['Yes','No'] )
plt.xlabel('true label')
plt.ylabel('predicted label')
TP=mat_knn[0,0]
FN=mat_knn[0,1]
FP=mat_knn[1,0]
TN=mat_knn[1,1]

knn_precision=TP/(TP+FP)
print("KNN Precision: ",knn_precision)

knn_recall=TP/(TP+FN)
print("KNN Recall: ",knn_recall)

knn_FM=(2*knn_recall*knn_precision)/(knn_recall+knn_precision)
print("KNN F-Measure: ",knn_FM)
##computing fpr and tpr we plot tpr vs fpr

knn_roc_auc=classifier.predict_proba(X_valid)[:,1]
fpr,tpr,threshold_smote=roc_curve(y_valid,knn_roc_auc)
roc_auc=auc(fpr,tpr)
plt.figure()

# ROC
plt.plot(fpr,tpr,'blue',label='KNN (AUC = %0.2f)'% roc_auc)

# random FPR and TPR
plt.plot([0,1],[0,1],'r--')

# title and label
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')
plt.show()

print('The AUC (smote):',auc(fpr,tpr))
from yellowbrick.classifier import ClassificationReport
knncr = ClassificationReport(KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2))
knncr.fit(X_train, y_train)
knncr.score(X_valid, y_valid)
knncr.show()
df1=df.drop(['Target'],axis=1)
array = df1.values 
arrayt=df['Target'].values
X = array
Y = arrayt
# splitting data in train and test
seed=600
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.30, random_state = seed)
from yellowbrick.target import ClassBalance
visualizer = ClassBalance(labels=[0, 1])

visualizer.fit(y_train,y_valid)
visualizer.ax.set_xlabel("Classes")
visualizer.ax.set_ylabel("Amount of Occurrences of Class")
visualizer.show()
from numpy import set_printoptions
from sklearn.metrics import confusion_matrix, classification_report
smt = SMOTE(random_state=seed)
X_train_SMOTE, Y_train_SMOTE = smt.fit_sample(X_train, y_train.ravel()) 
print(X_train_SMOTE.shape)
print(Y_train_SMOTE.shape)
set_printoptions(precision=3)
print('\n Oversampled input: \n %s' % X_train_SMOTE[0:5,:])
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_SMOTE == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_SMOTE == 0)))
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_features': [2, 3, 4, 5],
    'n_estimators': [200, 300, 400, 500]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                           cv = 3, n_jobs = -1, verbose = 2)
grid_result= grid_search.fit(X_train_SMOTE, Y_train_SMOTE)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Applying Pipeline to random forest for 300 trees
from sklearn.pipeline import Pipeline 
num_trees=300
max_features=4
estimator=[]                                           
estimator.append(('standardize',StandardScaler()))
estimator.append(('RF',RandomForestClassifier(
    n_estimators=num_trees,max_features=max_features))) 
model=Pipeline(estimator)      
kfold=KFold(n_splits=10,random_state=seed)                 
result1=cross_val_score(model,X_train_SMOTE,Y_train_SMOTE,cv=kfold)  
print(result1.mean()*100.0) 
smt1 = SMOTE(random_state=seed)
X_train_SMOT1, Y_train_SMOT1 = smt1.fit_sample(X_train, y_train) 
estimator1=[]                                           
estimator1.append(('standardize',StandardScaler()))
estimator1.append(('RF',RandomForestClassifier(
    n_estimators=num_trees,max_features=max_features))) 
model1=Pipeline(estimator1) 
model1.fit(X_train_SMOT1, Y_train_SMOT1)
result = model1.score(X_valid, y_valid)
print((result)*100.0)
predictions = model1.predict(X_valid)
# print classification report 
print(classification_report(y_valid, predictions))
from sklearn.metrics import classification_report, confusion_matrix
classifier = RandomForestClassifier(n_estimators=num_trees,max_features=max_features)
classifier.fit(X_train, y_train)
y_pred_rf = classifier.predict(X_valid)
mat_rf=confusion_matrix(y_valid, y_pred_rf)
print(mat_rf)
# Compute precision, recall, F-measure and support

TP=mat_rf[0,0]
FN=mat_rf[0,1]
FP=mat_rf[1,0]
TN=mat_rf[1,1]

Precision=TP/(TP+FP)
print("Precision: ",Precision)

Recall=TP/(TP+FN)
print("Recall: ",Recall)

FM=(2*Recall*Precision)/(Recall+Precision)
print("F-Measure: ",FM)
sns.heatmap(mat_rf,  annot=True, fmt='d', cbar=False,
          xticklabels=['Yes','No'],
          yticklabels=['Yes','No'] )
plt.xlabel('true label')
plt.ylabel('predicted label')

rf_roc_auc_smote=classifier.predict_proba(X_valid)[:,1]
fpr,tpr,threshold_smote=roc_curve(y_valid,rf_roc_auc_smote)
roc_auc_smote=auc(fpr,tpr)
plt.figure()

# ROC
plt.plot(fpr,tpr,'blue',label='Random Forest (AUC = %0.2f)'% roc_auc_smote)

# random FPR and TPR
plt.plot([0,1],[0,1],'r--')

# title and label
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')
plt.show()

print('The AUC (smote):',auc(fpr,tpr))
from yellowbrick.classifier import ClassificationReport
rfcr = ClassificationReport(classifier)
rfcr.fit(X_train, y_train)
rfcr.score(X_valid, y_valid)
rfcr.show()
