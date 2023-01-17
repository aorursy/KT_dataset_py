import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
import itertools
plt.style.use('fivethirtyeight')
from sklearn import model_selection
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
df=pd.read_csv('../input/diabetes.csv')
# Lets look at some of the sample data 
df.head()
df.describe()
df.isna().any() # checking No. of Missing Values.
print(df.dtypes)
df.head(50)
# Calculate the median value for BMI
median_bmi = df['BMI'].median()
# Substitute it in the BMI column of the
# dataset where values are 0
df['BMI'] = df['BMI'].replace(
    to_replace=0, value=median_bmi)

median_bloodp = df['BloodPressure'].median()
# Substitute it in the BloodP column of the
# dataset where values are 0
df['BloodPressure'] = df['BloodPressure'].replace(
    to_replace=0, value=median_bloodp)

# Calculate the median value for PlGlcConc
median_plglcconc = df['Glucose'].median()
# Substitute it in the PlGlcConc column of the
# dataset where values are 0
df['Glucose'] = df['Glucose'].replace(
    to_replace=0, value=median_plglcconc)

# Calculate the median value for SkinThick
median_skinthick = df['SkinThickness'].median()
# Substitute it in the SkinThick column of the
# dataset where values are 0
df['SkinThickness'] = df['SkinThickness'].replace(
    to_replace=0, value=median_skinthick)

# Calculate the median value for SkinThick
median_skinthick = df['Insulin'].median()
# Substitute it in the SkinThick column of the
# dataset where values are 0
df['Insulin'] = df['Insulin'].replace(
    to_replace=0, value=median_skinthick)
df.head(50)
sns.countplot(data=df, x = 'Outcome', label='Count')

DB, NDB = df['Outcome'].value_counts()
print('Number of patients diagnosed with Diabtetes disease: ',DB)
print('Number of patients not diagnosed with Diabtetes disease: ',NDB)
columns=df.columns[:8]
plt.subplots(figsize=(18,15))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    df[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()
df1=df[df['Outcome']==1]
columns=df.columns[:8]
plt.subplots(figsize=(18,15))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    df1[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()
sns.pairplot(df, hue = 'Outcome', vars = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI','DiabetesPedigreeFunction','Age'] )
sns.jointplot("Pregnancies", "Insulin", data=df, kind="reg")
def set_bmi(row):
    if row["BMI"] < 18.5:
        return "Under"
    elif row["BMI"] >= 18.5 and row["BMI"] <= 24.9:
        return "Healthy"
    elif row["BMI"] >= 25 and row["BMI"] <= 29.9:
        return "Over"
    elif row["BMI"] >= 30:
        return "Obese"
df = df.assign(BM_DESC=df.apply(set_bmi, axis=1))

df.head()
def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"
df = df.assign(INSULIN_DESC=df.apply(set_insulin, axis=1))

df.head()
sns.countplot(data=df, x = 'INSULIN_DESC', label='Count')

AB, NB = df['INSULIN_DESC'].value_counts()
print('Number of patients Having Abnormal Insulin Levels: ',AB)
print('Number of patients Having Normal Insulin Levels: ',NB)
sns.countplot(data=df, x = 'BM_DESC', label='Count')

UD,H,OV,OB = df['BM_DESC'].value_counts()
print('Number of patients Having Underweight BMI Index: ',UD)
print('Number of patients Having Healthy BMI Index: ',H)
print('Number of patients Having Overweigth BMI Index: ',OV)
print('Number of patients Having Obese BMI Index: ',OB)
g = sns.FacetGrid(df, col="INSULIN_DESC", row="Outcome", margin_titles=True)
g.map(plt.scatter,"Glucose", "BloodPressure",  edgecolor="w")
plt.subplots_adjust(top=1.1)
g = sns.FacetGrid(df, col="Outcome", row="INSULIN_DESC", margin_titles=True)
g.map(plt.hist, "Age", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by INSULIN and Age');
sns.boxplot(x="Age", y="INSULIN_DESC", hue="Outcome", data=df);
sns.boxplot(x="Age", y="BM_DESC", hue="Outcome", data=df);
df["INSULIN_DESC"] = df.INSULIN_DESC.apply(lambda  x:1 if x=="Normal" else 0)
X=pd.get_dummies(df,drop_first=True)
X=X.drop(['Outcome'],axis=1)
y = df['Outcome']
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,stratify=y, random_state = 1234)
sc_X = StandardScaler()
X_train_scaled = pd.DataFrame(sc_X.fit_transform(X_train))
X_test_scaled = pd.DataFrame(sc_X.transform(X_test))
logi = LogisticRegression(random_state = 0, penalty = 'l1')
logi.fit(X_train_scaled, y_train)
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train_scaled, y_train, verbose=True)
random_forest = RandomForestClassifier(n_estimators = 100,criterion='gini', random_state = 47)
random_forest.fit(X_train_scaled, y_train)
svc_model_l = SVC(kernel='linear',probability=True)
svc_model_l.fit(X_train_scaled, y_train)
svc_model_r = SVC(kernel='rbf',probability=True)
svc_model_r.fit(X_train_scaled, y_train)
kfold = model_selection.KFold(n_splits=10, random_state=7)
scoring = 'accuracy'

acc_logi = cross_val_score(estimator = logi, X = X_train_scaled, y = y_train, cv = kfold,scoring=scoring)
acc_logi.mean()

acc_xgb = cross_val_score(estimator = xgb_classifier, X = X_train_scaled, y = y_train, cv = kfold,scoring=scoring)
acc_xgb.mean()

acc_rand = cross_val_score(estimator = random_forest, X = X_train_scaled, y = y_train, cv = kfold, scoring=scoring)
acc_rand.mean()

acc_svc_l = cross_val_score(estimator = svc_model_l, X = X_train_scaled, y = y_train, cv = kfold,scoring=scoring)
acc_svc_l.mean()

acc_svc_r = cross_val_score(estimator = svc_model_r, X = X_train_scaled, y = y_train, cv = kfold,scoring=scoring)
acc_svc_r.mean()
y_predict_logi = logi.predict(X_test_scaled)
acc= accuracy_score(y_test, y_predict_logi)
roc=roc_auc_score(y_test, y_predict_logi)
prec = precision_score(y_test, y_predict_logi)
rec = recall_score(y_test, y_predict_logi)
f1 = f1_score(y_test, y_predict_logi)

results = pd.DataFrame([['Logistic Regression',acc, acc_logi.mean(),prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results
y_predict_x = xgb_classifier.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict_x)
acc = accuracy_score(y_test, y_predict_x)
prec = precision_score(y_test, y_predict_x)
rec = recall_score(y_test, y_predict_x)
f1 = f1_score(y_test, y_predict_x)

model_results = pd.DataFrame([['XG Boost',acc, acc_xgb.mean(),prec,rec, f1,roc]],
               columns = ['Model','Accuracy', 'Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results
y_predict_r = random_forest.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict_r)
acc = accuracy_score(y_test, y_predict_r)
prec = precision_score(y_test, y_predict_r)
rec = recall_score(y_test, y_predict_r)
f1 = f1_score(y_test, y_predict_r)

model_results = pd.DataFrame([['Random Forest',acc, acc_rand.mean(),prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results
y_predict_s = svc_model_l.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict_s)
acc = accuracy_score(y_test, y_predict_s)
prec = precision_score(y_test, y_predict_s)
rec = recall_score(y_test, y_predict_s)
f1 = f1_score(y_test, y_predict_s)

model_results = pd.DataFrame([['SVC Linear',acc, acc_svc_l.mean(),prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results
y_predict_s1 = svc_model_r.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict_s1)
acc = accuracy_score(y_test, y_predict_s1)
prec = precision_score(y_test, y_predict_s1)
rec = recall_score(y_test, y_predict_s1)
f1 = f1_score(y_test, y_predict_s1)

model_results = pd.DataFrame([['SVC RBF',acc, acc_svc_r.mean(),prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results
from sklearn import metrics
import matplotlib.pyplot as plt

plt.figure()

# Add the models to the list that you want to view on the ROC plot
models = [
{
    'label': 'Logistic Regression',
    'model': LogisticRegression(random_state = 0, penalty = 'l1'),
},
{
    'label': 'XG Boost',
    'model': XGBClassifier(),
},
    {
    'label': 'Random Forest Gini',
    'model': RandomForestClassifier(n_estimators = 100,criterion='gini', random_state = 47),
},
    {
    'label': 'Support Vector Machine-L',
    'model': SVC(kernel='linear',probability=True)} ,
        {
    'label': 'Support Vector Machine-RBF',
    'model': SVC(kernel='rbf',probability=True) ,
}
]

# Below for loop iterates through your models list
for m in models:
    model = m['model'] # select the model
    model.fit(X_train_scaled, y_train) # train the model
    y_pred=model.predict(X_test_scaled) # predict the test data
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test_scaled)[:,1])
# Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(X_test_scaled))
# Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
cm_logi = confusion_matrix(y_test, y_predict_logi)
plt.title('Confusion matrix of the Logistic classifier')
sns.heatmap(cm_logi,annot=True,fmt="d")
plt.show()
cm_x = confusion_matrix(y_test, y_predict_x)
plt.title('Confusion matrix of the XGB classifier')
sns.heatmap(cm_x,annot=True,fmt="d")
plt.show()
cm_r = confusion_matrix(y_test, y_predict_r)
plt.title('Confusion matrix of the Random Forest classifier')
sns.heatmap(cm_r,annot=True,fmt="d")
plt.show()
cm = confusion_matrix(y_test, y_predict_s)
plt.title('Confusion matrix of the SVC Linear classifier')
sns.heatmap(cm,annot=True,fmt="d")
plt.show()
TP = cm_logi[1, 1]
TN = cm_logi[0, 0]
FP = cm_logi[0, 1]
FN = cm_logi[1, 0]
classification_error = (FP + FN) / float(TP + TN + FP + FN)

print(classification_error)
sensitivity = TP / float(FN + TP)

print(sensitivity)
specificity = TN / (TN + FP)

print(specificity)
