import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.features.importances import FeatureImportances
iview = pd.read_csv('../input/Interview.csv')
iview.head()
iview.shape
iview.columns
iview.isnull().sum()
iview = iview.drop(['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27'],axis = 1)
iview[iview['Date of Interview'].isna() == True]
iview = iview.drop(1233, axis = 0)
iview2 = iview[['Industry',
       'Position to be closed', 'Interview Type','Gender','Interview Venue',
       'Have you obtained the necessary permission to start at the required time',
       'Hope there will be no unscheduled meetings',
       'Can I Call you three hours before the interview and follow up on your attendance for the interview',
       'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much',
       'Have you taken a printout of your updated resume. Have you read the JD and understood the same',
       'Are you clear with the venue details and the landmark.',
       'Has the call letter been shared', 'Expected Attendance',
       'Observed Attendance', 'Marital Status']]
def func():
    
    for i in iview2.columns:
        print(np.unique(pd.DataFrame(iview2[i].value_counts()).reset_index()['index']).tolist())
func()
iview2 = iview2.replace(['Sceduled walkin', 'Scheduled Walk In', 'Scheduled Walkin', 'Walkin', 'Walkin '], 'Walk-in')
iview2 = iview2.replace(['Scheduled '], 'Scheduled')
iview2 = iview2.replace(['Yes', 'yes', 'yes ', 'YES', ' yes', 'Y','10.30 Am', '11:00 AM'], 'y')
iview2 = iview2.replace(['No', 'no', 'no ', 'NO', ' no', 'N', 'Na','na','No I have only thi number',
                        'No- will take it soon', 'n','Havent Checked','No ','No Dont','Not Yet'], 'n')
iview2 = iview2.replace(['Havent Checked', 'Need To Check', 'Not Sure', 'Not sure', 'Not yet', 'Yet to Check', 
                         'Yet to confirm','cant Say','No- I need to check'], 'Uncertain')
iview2 = iview2.replace(['IT Products and Services', 'IT Services'], 'IT')
def func():
    
    for i in iview2.columns:
        print(np.unique(pd.DataFrame(iview2[i].value_counts()).reset_index()['index']).tolist())
        #print(uni)
func()
iview2.columns
iview2.isnull().sum()
iview2 = iview2.fillna('n')
iview2.isnull().sum().sum()
def cat_barplot():
    for n in range(0, (len(iview2.columns))):
        plt.subplot(7, 3, n+1)
        iview2.select_dtypes(include = ['object']).iloc[:,n].value_counts().plot(kind = 'bar')
        plt.xlabel(iview2.select_dtypes(include = ['object']).iloc[:,n].name)
plt.figure(figsize = (25, 60))
cat_barplot()
plt.show()
# Showing the classes in the features (categorical features)

y = pd.DataFrame(iview2.groupby(by = ['Marital Status', 'Observed Attendance'])['Expected Attendance'].count()).reset_index()
y
def func1():
    for n in range(0, len(iview2.columns[1:-3])):
        plt.subplot(8, 2, n+1)
        y = pd.DataFrame(iview2.groupby(by = [iview2.columns[n], 
                                              'Observed Attendance'])['Expected Attendance'].count()).reset_index()
        sns.barplot(x = iview2.columns[n], y = 'Expected Attendance', hue = 'Observed Attendance', data = y)
        plt.xlabel(iview2.columns[n], size = 20)
        plt.xticks(size=14, rotation=45)
        plt.yticks(size=14)
        plt.legend(fontsize= 15)
        plt.ylabel('Count', size= 15)
        plt.subplots_adjust(hspace=.5)
plt.figure(figsize = (25, 80))
func1()
plt.show()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
iview3 = iview2.apply(le.fit_transform)
names = ['No', 'Yes']
X = iview3.drop(['Observed Attendance','Expected Attendance'], axis = 1)
y = iview3['Observed Attendance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from xgboost import XGBClassifier

from yellowbrick.classifier import ConfusionMatrix
def model_fit(x):
    x.fit(X_train, y_train)
    y_pred = x.predict(X_test)
    model_fit.accuracy = accuracy_score(y_pred, y_test)
    print('Accuracy Score',accuracy_score(y_pred, y_test))
    print(classification_report(y_pred, y_test))
        
    classes = names
    
    model_cm = ConfusionMatrix(
    x, classes = classes,
    label_encoder = {0 : 'No', 1 : 'Yes'})
    
    model_cm.fit(X_train, y_train)
    model_cm.score(X_test, y_test)
    
    model_cm.poof()  
# list = []
# for i in range(1,20):
#     model_fit(KNeighborsClassifier(n_neighbors = i))
#     list.append(model_fit.accuracy)
# list
model_fit(KNeighborsClassifier(n_neighbors = 13))
KNN = model_fit.accuracy
from sklearn.linear_model import LogisticRegression
model_fit(LogisticRegression())
Logistic = model_fit.accuracy
from sklearn.naive_bayes import GaussianNB
model_fit(GaussianNB())
Gaussian = model_fit.accuracy
from sklearn import tree
model_fit(tree.DecisionTreeClassifier())
Tree = model_fit.accuracy
from sklearn.ensemble import RandomForestClassifier
model_fit(RandomForestClassifier(n_estimators = 100, max_depth =10, random_state = 1))
RandomForest = model_fit.accuracy
model_fit(XGBClassifier(max_depth=20, learning_rate=0.1, n_estimators=50, silent=True, 
                        objective='binary:logistic', booster='gbtree', n_jobs=1, 
                        nthread=None, gamma=0, min_child_weight=10, max_delta_step=0, 
                        subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, 
                        reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=1, 
                        seed=1, missing=None))
XGBClf = model_fit.accuracy
from sklearn.ensemble import GradientBoostingClassifier
model_fit(GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=220, subsample=1.0, 
                           criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
                           min_weight_fraction_leaf=0.0, max_depth=2, min_impurity_decrease=0.0, 
                           min_impurity_split=None, init=None, random_state=1, max_features=None, 
                           verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', 
                           validation_fraction=0.1, n_iter_no_change=None, tol=0.0001))
GradientClf = model_fit.accuracy
import h2o
from h2o.automl import H2OAutoML
h2o.init()
# Load data into H2O
df = h2o.H2OFrame(iview3)
df.describe()
splits = df.split_frame(ratios = [0.8], seed = 1)
train_aml = splits[0]
test = splits[1]
y = "Observed Attendance"
x = train_aml.columns
x.remove(y)
#test.remove(y)
#x.remove("sku")
aml = H2OAutoML(max_models = 10, seed = 1)
aml.train(x = x, y = y, training_frame = train_aml)
aml.leaderboard
pred = aml.predict(test)
pred.head()
pred.describe()
pred[pred < 0.64] = 0
pred[pred >= 0.64] = 1
pred.as_data_frame()['predict'].value_counts()
amlCLF = accuracy_score(pred.as_data_frame(), test[['Observed Attendance']].as_data_frame())
amlCLF
scores_list_1 = ['KNN','Logistic','Gaussian','Tree','RandomForest','XGBClassifier', 'GradientClassifier','amlCLF']
scores_1 = [KNN, Logistic, Gaussian, Tree, RandomForest, XGBClf, GradientClf, amlCLF]
score_df_classification = pd.DataFrame([scores_list_1, scores_1]).T
score_df_classification.index = score_df_classification[0]
del score_df_classification[0]
score_df_classification
#Generalized confusion matrix
fig = plt.figure(figsize = (10,8))
#ax = fig.add_subplot()
visualizer = ClassPredictionError(GradientBoostingClassifier(loss='deviance', learning_rate=0.1, 
                                                             n_estimators=220, subsample=1.0, 
                                                             criterion='friedman_mse', min_samples_split=2, 
                                                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                                             max_depth=2, min_impurity_decrease=0.0, 
                                                             min_impurity_split=None, init=None, 
                                                             random_state=1, max_features=None, verbose=0, 
                                                             max_leaf_nodes=None, warm_start=False, 
                                                             presort='auto', validation_fraction=0.1, 
                                                             n_iter_no_change=None, tol=0.0001), classes = names)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
g = visualizer.poof()
m = (GradientBoostingClassifier(loss='deviance', learning_rate=0.1, 
                                                             n_estimators=220, subsample=1.0, 
                                                             criterion='friedman_mse', min_samples_split=2, 
                                                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                                             max_depth=2, min_impurity_decrease=0.0, 
                                                             min_impurity_split=None, init=None, 
                                                             random_state=1, max_features=None, verbose=0, 
                                                             max_leaf_nodes=None, warm_start=False, 
                                                             presort='auto', validation_fraction=0.1, 
                                                             n_iter_no_change=None, tol=0.0001).fit(X_train, 
                                                                                                    y_train))
x = pd.DataFrame(m.feature_importances_,X_train.columns)
pd.DataFrame(x[0].sort_values(ascending = False)[0:20]).iloc[:,:1].plot.barh(figsize=(10,5))
plt.show()