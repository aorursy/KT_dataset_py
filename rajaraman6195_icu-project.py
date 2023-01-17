
import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble  import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import Perceptron
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import VotingClassifier
%matplotlib inline
from sklearn import model_selection
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
global DATA_PATH
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        DATA_PATH = os.path.join(dirname, filename)
df = pd.read_csv(DATA_PATH)
df.head()
df.info()
df.BloodPressure = pd.to_numeric(df['BloodPressure'],errors='coerce')
df.SkinThickness = pd.to_numeric(df['SkinThickness'],errors='coerce')
df.Insulin = pd.to_numeric(df['Insulin'],errors='coerce')
df.BMI = pd.to_numeric(df['BMI'],errors='coerce')
df.DiabetesPedigreeFunction = pd.to_numeric(df['DiabetesPedigreeFunction'],errors='coerce')
df.Age = pd.to_numeric(df['Age'],errors='coerce')
df.Outcome = pd.to_numeric(df['Outcome'],errors='coerce')
df.info()
df.describe()
sns.countplot(x='Outcome',data=df)
df.corr()
sns.distplot(df['Age'].dropna(),kde=True)
median_bmi = df['BMI'].median()
df['BMI'] = df['BMI'].replace(
    to_replace=0, value=median_bmi)

median_bloodp = df['BloodPressure'].median()
df['BloodPressure'] = df['BloodPressure'].replace(
    to_replace=0, value=median_bloodp)

median_plglcconc = df['Glucose'].median()
df['Glucose'] = df['Glucose'].replace(
    to_replace=0, value=median_plglcconc)

median_skinthick = df['SkinThickness'].median()
df['SkinThickness'] = df['SkinThickness'].replace(
    to_replace=0, value=median_skinthick)

median_skinthick = df['Insulin'].median()
df['Insulin'] = df['Insulin'].replace(
    to_replace=0, value=median_skinthick)

df = df.dropna()
dataset2 = df.drop(columns = ['Outcome'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
dataset2.corrwith(df.Outcome).plot.bar(
        figsize = (20, 10), title = "Correlation with Outcome", fontsize = 15,
        rot = 45, grid = True)
corr = dataset2.corr()
sns.heatmap(corr,annot=True)
X = df.drop(['Outcome'],axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,stratify=y, random_state = 100)
# Feature Scaling
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
             max_depth=3, min_samples_leaf=10)
clf_entropy.fit(X_train_scaled, y_train)
import graphviz 
dot_data = tree.export_graphviz(clf_entropy, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render() 
feature_names = df.drop(columns = ['Outcome']).columns
feature_names = list(feature_names)
# print(feature_names)
target_names = df['Outcome'].astype(str)
target_names = list(target_names)
# print(target_names)
dot_data = tree.export_graphviz(clf_entropy, out_file=None, 
                    feature_names=feature_names,  
                    class_names=target_names,  
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(dot_data)  
graph
y_predict = clf_entropy.predict(X_test_scaled)
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test, y_predict)
roc_dt = roc
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

results = pd.DataFrame([['Decision Tree', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train_scaled, y_train)
y_predict = xgb_classifier.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['XGBOOST', acc,prec,rec, f1,roc]],
columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results_xg_dt = results

results_xg_dt
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
GaussianNB(priors=None)
y_predict = gnb.predict(X_test_scaled)
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test, y_predict)

acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

results_nb = pd.DataFrame([['Naive bayes', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])

results_nb
y_predict = xgb_classifier.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['XGBOOST', acc,prec,rec, f1,roc]],
columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results_nb = results_nb.append(model_results, ignore_index = True)
results_nb
results_nb
results_xg_dt
results = results_xg_dt.append(results_nb)
results.sort_values(['Accuracy','Precision'],ascending=[True,False])
from yellowbrick.classifier import ClassificationReport
visualizer = ClassificationReport(gnb, classes=target_names, support=True)
visualizer.fit(X_train_scaled, y_train)  # Fit the visualizer and the model
visualizer.score(X_test_scaled, y_test)  # Evaluate the model on the test data
visualizer.poof()    
from sklearn import metrics
'''
The first parameter to tune is max_depth. This indicates how deep the tree can be. 
The deeper the tree, the more splits it has and it captures more information about the data. 
We fit a decision tree with depths ranging from 1 to 32 and plot the training and test auc scores.
'''
max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
dt = ''

for max_depth in max_depths:
   dt = DecisionTreeClassifier(max_depth=max_depth)
   dt.fit(X_train_scaled, y_train)
   train_pred = dt.predict(X_train_scaled)
   false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train, train_pred)
   roc_dt = metrics.auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous train results
   train_results.append(roc_dt)
   y_pred = dt.predict(X_test_scaled)
   false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred)
   roc_dt = metrics.auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous test results
   test_results.append(roc_dt)
from matplotlib.legend_handler import HandlerLine2D
line1 = plt.plot(max_depths, train_results, "b", label="Train AUC")
line2 = plt.plot(max_depths, test_results, "r", label="Test AUC")
plt.legend('Decision tree optimizer')
plt.ylabel("AUC score")
plt.xlabel("Tree depth")
plt.show()
'''
min_samples_split represents the minimum number of samples required to split an internal node. 
This can vary between considering at least one sample at each node to considering all of the 
samples at each node. When we increase this parameter, the tree becomes more constrained as 
it has to consider more samples at each node. Here we will vary the parameter from 10% to 100% 
of the samples
'''
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
   dt.fit(X_train_scaled, y_train)
   train_pred = dt.predict(X_train_scaled)
   false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train, train_pred)
   roc_dt = metrics.auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_dt)
   y_pred = dt.predict(X_test_scaled)
   false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred)
   roc_dt = metrics.auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_dt)
from matplotlib.legend_handler import HandlerLine2D
line1 = plt.plot(min_samples_splits, train_results, "b", label="Train AUC")
line2 = plt.plot(min_samples_splits, test_results, "r", label="Test AUC")
plt.legend('Decision tree optimizer')
plt.ylabel("AUC score")
plt.xlabel("Tree depth")
plt.show()
print(len(X_test))
X_test_scaled
model_predict = clf_entropy.predict(X_test_scaled)
model_predict
from sklearn.externals import joblib
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(clf_entropy, filename)

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test_scaled, y_test)
result_loaded = loaded_model.predict(X_test_scaled)
result_loaded
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

cl_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
             max_depth=3, min_samples_leaf=10)
cl_entropy.fit(X_train, y_train)
model_predict = clf_entropy.predict(X_test_scaled)
model_predict
from sklearn.externals import joblib
# save the model to disk
filename = 'final_model.sav'
joblib.dump(cl_entropy, filename)

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
result_loaded = loaded_model.predict(X_test)
result_loaded
