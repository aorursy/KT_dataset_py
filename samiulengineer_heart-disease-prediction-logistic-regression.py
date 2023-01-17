import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as sci
import matplotlib.pyplot as matplt
import seaborn as sb
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as matlab
%matplotlib inline
tubes2_HeartDisease_train = pd.read_csv("../input/tubes2_HeartDisease_train.csv")
tubes2_HeartDisease_train.head()
tubes2_HeartDisease_train.dtypes
tubes2_HeartDisease_train['Column4'] = pd.to_numeric(tubes2_HeartDisease_train['Column4'], errors = 'coerce')
tubes2_HeartDisease_train['Column5'] = pd.to_numeric(tubes2_HeartDisease_train['Column5'], errors = 'coerce')
tubes2_HeartDisease_train['Column6'] = pd.to_numeric(tubes2_HeartDisease_train['Column6'], errors = 'coerce')
tubes2_HeartDisease_train['Column7'] = pd.to_numeric(tubes2_HeartDisease_train['Column7'], errors = 'coerce')
tubes2_HeartDisease_train['Column8'] = pd.to_numeric(tubes2_HeartDisease_train['Column8'], errors = 'coerce')
tubes2_HeartDisease_train['Column9'] = pd.to_numeric(tubes2_HeartDisease_train['Column9'], errors = 'coerce')
tubes2_HeartDisease_train['Column10'] = pd.to_numeric(tubes2_HeartDisease_train['Column10'], errors = 'coerce')
tubes2_HeartDisease_train['Column11'] = pd.to_numeric(tubes2_HeartDisease_train['Column11'], errors = 'coerce')
tubes2_HeartDisease_train['Column12'] = pd.to_numeric(tubes2_HeartDisease_train['Column12'], errors = 'coerce')
tubes2_HeartDisease_train['Column13'] = pd.to_numeric(tubes2_HeartDisease_train['Column13'], errors = 'coerce')
tubes2_HeartDisease_train.dtypes
tubes2_HeartDisease_train.head()
len(tubes2_HeartDisease_train.index)
tubes2_HeartDisease_train.rename(columns={'Column1' : 'age', 'Column2' : 'sex', 'Column3' : 'chest_pain_type', 'Column4' : 'resting_bp', 'Column5' : 'ser_chol', 'Column6' : 'fast_glucose', 'Column7' : 'rest_ecg', 'Column8' : 'heart_rate', 'Column9' : 'exc_angina', 'Column10' : 'depression', 'Column11' : 'peak_exc', 'Column12' : 'maj_vessels', 'Column13' : 'thal', 'Column14' : 'heart_disease'}, inplace = True)
tubes2_HeartDisease_train.head()
tubes2_HeartDisease_train.isnull().sum()
count = 0
for i in tubes2_HeartDisease_train.isnull().sum(axis = 1):
    if i > 0:
        count = count + 1
print("%i instances have missing values which is %i%% of the total data" %(count, round((float(count)/len(tubes2_HeartDisease_train.index))*100)))
tubes2_HeartDisease_train.drop(columns = ['peak_exc', 'maj_vessels', 'thal'], inplace = True)
tubes2_HeartDisease_train.isnull().sum()
count = 0
for i in tubes2_HeartDisease_train.isnull().sum(axis = 1):
    if i > 0:
        count = count + 1
print("%i instances have missing values which is %i%% of the total data" % (count, round((float(count)/len(tubes2_HeartDisease_train.index))*100)))
tubes2_HeartDisease_train.dropna(inplace = True)
tubes2_HeartDisease_train.isnull().sum()
len(tubes2_HeartDisease_train.index)
tubes2_HeartDisease_train['heart_disease'] = (tubes2_HeartDisease_train['heart_disease'] >= 1).astype(int)
tubes2_HeartDisease_train = tubes2_HeartDisease_train.astype(int)
tubes2_HeartDisease_train.dtypes
def histograms(df, ft, rw, cl):
    fig = matplt.figure(figsize = (20,20))
    for i, feature in enumerate (ft):
        ax = fig.add_subplot(rw, cl, i+1)
        df[feature].hist(bins = 20, ax = ax, facecolor = 'red')
        ax.set_title(feature)
    fig.tight_layout()
    matplt.show()
histograms(tubes2_HeartDisease_train, tubes2_HeartDisease_train.columns, 6, 3)
sb.countplot(x = tubes2_HeartDisease_train['heart_disease'], data = tubes2_HeartDisease_train, palette = 'hls')
(tubes2_HeartDisease_train['heart_disease'].value_counts()/tubes2_HeartDisease_train['heart_disease'].count())*100
tubes2_HeartDisease_train.describe()
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x = tubes2_HeartDisease_train.iloc[:,:-1]
y = tubes2_HeartDisease_train.iloc[:,-1]

# train_test_split will return the 4 array
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 3)

# fit the model
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
# predict disease from x_test set
y_pred = logreg.predict(x_test)

accuracy = sklearn.metrics.accuracy_score(y_test,y_pred)

print('Accuracy of logistic Regression classifier on test set is: {}%'.format(round(accuracy,2)*100))
y_pred_count = np.unique(y_pred, return_counts = True)
y_pred_count
from sklearn.metrics import confusion_matrix

cfmx = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data = cfmx, columns = ['Predicted: 0', 'Predicted: 1'], index = ['Actual: 0', 'Actual: 1'])

sb.heatmap(data = conf_matrix, annot = True, fmt = 'd', cmap = "YlGnBu", square = True)
TN = cfmx[0,0]
TP = cfmx[1,1]
FN = cfmx[1,0]
FP = cfmx[0,1]

print("The confusion matrix shows %i correct predictions and %i incorrect predictions" % ((TN + TP), (FN + FP)))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])

matplt.plot(fpr,tpr)
matplt.xlim([0.0, 1.0])
matplt.ylim([0.0, 1.0])
matplt.plot([0, 1], [0, 1], 'r--')
matplt.title('ROC curve for Heart disease classifier')
matplt.xlabel('False positive rate (1-Specificity)')
matplt.ylabel('True positive rate (Sensitivity)')
matplt.grid(True)
roc_auc_score = sklearn.metrics.roc_auc_score(y_test,logreg.predict_proba(x_test)[:,1])

if roc_auc_score >= 0.70:
    print("ROC Curve Covers almost %i%% Area" % (round(roc_auc_score, 2)*100))
else:
     print("ROC Curve Covers almost %i%% Area Which is not Satisfactory" % (round(roc_auc_score, 2)*100))  
x_test['y_test'] = y_test
x_test['y_pred'] = y_pred

x_test = x_test.drop(x_test[x_test.y_test == x_test.y_pred].index)

x_test.head()
tubes2_HeartDisease_test = pd.read_csv("../input/tubes2_HeartDisease_test.csv")
tubes2_HeartDisease_test.head()
tubes2_HeartDisease_test.dtypes
tubes2_HeartDisease_test['Column4'] = pd.to_numeric(tubes2_HeartDisease_test['Column4'], errors = 'coerce')
tubes2_HeartDisease_test['Column5'] = pd.to_numeric(tubes2_HeartDisease_test['Column5'], errors = 'coerce')
tubes2_HeartDisease_test['Column6'] = pd.to_numeric(tubes2_HeartDisease_test['Column6'], errors = 'coerce')
tubes2_HeartDisease_test['Column8'] = pd.to_numeric(tubes2_HeartDisease_test['Column8'], errors = 'coerce')
tubes2_HeartDisease_test['Column9'] = pd.to_numeric(tubes2_HeartDisease_test['Column9'], errors = 'coerce')
tubes2_HeartDisease_test['Column10'] = pd.to_numeric(tubes2_HeartDisease_test['Column10'], errors = 'coerce')
tubes2_HeartDisease_test['Column11'] = pd.to_numeric(tubes2_HeartDisease_test['Column11'], errors = 'coerce')
tubes2_HeartDisease_test['Column12'] = pd.to_numeric(tubes2_HeartDisease_test['Column12'], errors = 'coerce')
tubes2_HeartDisease_test['Column13'] = pd.to_numeric(tubes2_HeartDisease_test['Column13'], errors = 'coerce')
tubes2_HeartDisease_test.rename(columns={'Column1' : 'age', 'Column2' : 'sex', 'Column3' : 'chest_pain_type', 'Column4' : 'resting_bp', 'Column5' : 'ser_chol', 'Column6' : 'fast_glucose', 'Column7' : 'rest_ecg', 'Column8' : 'heart_rate', 'Column9' : 'exc_angina', 'Column10' : 'depression', 'Column11' : 'peak_exc', 'Column12' : 'maj_vessels', 'Column13' : 'thal'}, inplace = True)

tubes2_HeartDisease_test.drop(columns = ['peak_exc', 'maj_vessels', 'thal'], inplace = True)

tubes2_HeartDisease_test.head()
len(tubes2_HeartDisease_test.index)
tubes2_HeartDisease_test.isnull().sum()
count = 0
for i in tubes2_HeartDisease_test.isnull().sum(axis = 1):
    if i > 0:
        count = count + 1
print("%i instances have missing values which is %i%% of the total data" % (count, round((float(count)/len(tubes2_HeartDisease_train.index))*100, 2)))
tubes2_HeartDisease_test.dropna(inplace = True)
tubes2_HeartDisease_test.isnull().sum()
tubes2_HeartDisease_test = tubes2_HeartDisease_test.astype(int)
tubes2_HeartDisease_test.head()
y_pred = logreg.predict(tubes2_HeartDisease_test)

tubes2_HeartDisease_test['heart_disease'] = y_pred

tubes2_HeartDisease_test.head()

