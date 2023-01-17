import pandas as pd

import numpy as np

import warnings  

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import itertools

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import  accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.svm import SVC 

plt.style.use('fivethirtyeight')
df=pd.read_csv('../input/audit_data.csv')
df.head()
df.columns
df.tail()
cols_del=['LOCATION_ID','TOTAL']



df.drop(cols_del, axis=1, inplace=True)
df.head()
df.info()
df.describe()
df.isna().sum()
df['Money_Value'].fillna((df['Money_Value'].mean()), inplace=True)
df.isna().sum()
sns.countplot(df['Risk'], label = "Count") 
X=df.drop(['Risk'],axis=1)
X.corrwith(df.Risk).plot.bar(

        figsize = (20, 10), title = "Correlation with Churn", fontsize = 20,

        rot = 90, grid = True)
X.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
X=X.drop(['Detection_Risk'],axis=1)
X.columns
df1=df[df['Risk']==1]

columns=df1.columns[:21]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    df1[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
y=df['Risk']
from sklearn.model_selection import train_test_split,cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,stratify=y, random_state = 123)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train_scaled = pd.DataFrame(sc_X.fit_transform(X_train))

X_test_scaled = pd.DataFrame(sc_X.transform(X_test))
logi = LogisticRegression(random_state = 0, penalty = 'l1')

logi.fit(X_train_scaled, y_train)
kfold = model_selection.KFold(n_splits=10, random_state=7)

scoring = 'accuracy'



acc_logi = cross_val_score(estimator = logi, X = X_train_scaled, y = y_train, cv = kfold,scoring=scoring)

acc_logi.mean()
y_predict_logi = logi.predict(X_test_scaled)

acc= accuracy_score(y_test, y_predict_logi)

roc=roc_auc_score(y_test, y_predict_logi)

prec = precision_score(y_test, y_predict_logi)

rec = recall_score(y_test, y_predict_logi)

f1 = f1_score(y_test, y_predict_logi)



results = pd.DataFrame([['Logistic Regression',acc, acc_logi.mean(),prec,rec, f1,roc]],

               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

results
random_forest_e = RandomForestClassifier(n_estimators = 100,criterion='entropy', random_state = 47)

random_forest_e.fit(X_train_scaled, y_train)
acc_rande = cross_val_score(estimator = random_forest_e, X = X_train_scaled, y = y_train, cv = kfold, scoring=scoring)

acc_rande.mean()
y_predict_r = random_forest_e.predict(X_test_scaled)

roc=roc_auc_score(y_test, y_predict_r)

acc = accuracy_score(y_test, y_predict_r)

prec = precision_score(y_test, y_predict_r)

rec = recall_score(y_test, y_predict_r)

f1 = f1_score(y_test, y_predict_r)



model_results = pd.DataFrame([['Random Forest',acc, acc_rande.mean(),prec,rec, f1,roc]],

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

    'label': 'Random Forest Entropy',

    'model': RandomForestClassifier(n_estimators = 100,criterion='entropy', random_state = 47),

},

    

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
cm_r = confusion_matrix(y_test, y_predict_r)

plt.title('Confusion matrix of the Random Forest classifier')

sns.heatmap(cm_r,annot=True,fmt="d")

plt.show()
importances = random_forest_e.feature_importances_

indices = np.argsort(importances)[::-1]



# Rearrange feature names so they match the sorted feature importances

names = [X.columns[i] for i in indices]



# Create plot

plt.figure()



# Create plot title

plt.title("Feature Importance")



# Add bars

plt.bar(range(X.shape[1]), importances[indices])



# Add feature names as x-axis labels

plt.xticks(range(X.shape[1]), names, rotation=90)



# Show plot

plt.show()