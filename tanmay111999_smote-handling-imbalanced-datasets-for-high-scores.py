import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.options.display.float_format = '{:.2f}'.format
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.head()
data.shape
data.columns
data.info()
data.describe()
sns.heatmap(data.isnull(),cmap = 'magma',cbar = False)
f = data[data['Class'] == 1]
nf = data[data['Class'] == 0]
f.describe()
nf.describe()
fraud = len(data[data['Class'] == 1])/len(data)*100
no_fraud = len(data[data['Class'] == 0])/len(data)*100
fraud_percentage = [fraud,no_fraud]
fig,ax = plt.subplots(nrows = 1,ncols = 3,figsize = (20,5))
plt.subplot(1,3,1)
plt.pie(fraud_percentage,labels = ['Fraud','No Fraud'],autopct='%1.1f%%',startangle = 90,)
plt.title('FRAUD PERCENTAGE')

plt.subplot(1,3,2)
sns.countplot('Class',data = data,)
plt.title('DISTRIBUTION OF FRAUD CASES')

plt.subplot(1,3,3)
sns.scatterplot('Time','Amount',data = data,hue = 'Class')
plt.title('TIME vs AMOUNT w.r.t CLASS')
plt.show()
sns.heatmap(data.corr(),cmap = 'RdBu',cbar = True)
corr = data.corrwith(data['Class']).sort_values(ascending = False).to_frame()
corr.columns = ['Correlations']
plt.subplots(figsize = (5,25))
sns.heatmap(corr,annot = True,cmap = 'RdBu',linewidths = 0.4,linecolor = 'black')
plt.title('CORRELATION w.r.t CLASS')
df = data[['V4','V11','V7','V3','V16','V10','V12','V14','V17','Class']]
df.head()
import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

imblearn.__version__
def model(classifier):
    
    classifier.fit(x_train,y_train)
    prediction = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
    print("CROSS VALIDATION SCORE : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc').mean()))
    print("ROC_AUC SCORE : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))
    plot_roc_curve(classifier, x_test,y_test)
    plt.title('ROC_AUC_PLOT')
    plt.show()
def model_evaluation(classifier):
    
    # CONFUSION MATRIX
    cm = confusion_matrix(y_test,classifier.predict(x_test))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
    
    # CLASSIFICATION REPORT
    print(classification_report(y_test,classifier.predict(x_test)))
over = SMOTE(sampling_strategy= 0.5)
under = RandomUnderSampler(sampling_strategy = 0.1)
features = df.iloc[:,:9].values
target = df.iloc[:,9].values
steps = [('under', under),('over', over)]
pipeline = Pipeline(steps=steps)
features, target = pipeline.fit_resample(features, target)
Counter(target)
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = 2)
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0,C=10,penalty= 'l2') 
model(classifier_lr)
model_evaluation(classifier_lr)
from sklearn.svm import SVC
classifier_svc = SVC(kernel = 'linear',C = 0.1)
model(classifier_svc)
model_evaluation(classifier_svc)
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(random_state = 1000,max_depth = 4,min_samples_leaf = 1)
model(classifier_dt)
model_evaluation(classifier_dt)
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(max_depth = 4,random_state = 0)
model(classifier_rf)
model_evaluation(classifier_rf)
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(leaf_size = 1, n_neighbors = 3,p = 1)
model(classifier_knn)
model_evaluation(classifier_knn)