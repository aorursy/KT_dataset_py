import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE
def ROC_GEN(Title, Labels, Output): 
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr, tpr, _ = roc_curve(Labels, Output)   
    roc_auc = auc(fpr, tpr)    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(Title)
    plt.legend(loc="lower right")
    plt.show()
    
    return;
data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

rows = data.shape[0]
columns = data.shape[1]
print("The dataset contains {0} rows and {1} columns".format(rows, columns))

data.head(1)
data.mean()
sns.pairplot(data[["Age", "Education", "JobLevel"]])
plt.show()
kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = data._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1],c=labels)
plt.show()
data_copy = data.copy()
data_copy["Attrition"] = data_copy["Attrition"].replace(["Yes","No"],[1,0]);
train = data_copy.sample(frac=0.5, random_state=1)
test = data_copy.loc[~data_copy.index.isin(train.index)]
data.head(20)
Effective_Columns = ["Age", "DailyRate", "DistanceFromHome", "Education", "MonthlyIncome","MonthlyRate" ,"NumCompaniesWorked",
"PercentSalaryHike","PerformanceRating","RelationshipSatisfaction",
"StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany",
"YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager",
"EmployeeNumber","EnvironmentSatisfaction","HourlyRate","JobInvolvement","JobLevel","JobSatisfaction"]
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf.fit(train[Effective_Columns], train["Attrition"])
predictions = rf.predict(test[Effective_Columns])
len = predictions.shape[0];
test_label = [0 for x in range(len)] 
test_attr = test["Attrition"];

for i in range(len):
    test_label[i] = test_attr[test_attr.index[i]];

ROC_GEN('RF1-50%', test_label, predictions)
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)
rf.fit(train[Effective_Columns], train["Attrition"])
predictions = rf.predict(test[Effective_Columns])

ROC_GEN('RF2-50%', test_label, predictions)
data = data_copy[Effective_Columns];
out = data_copy["Attrition"];

len = out.shape[0];
dout = [0 for x in range(len)] 
for i in range(len):
    dout[i] = out[out.index[i]];


din = [[0 for x in range(data.shape[1])] for y in range(len)] 
for i in range(len):
    for j in range(data.shape[1]-9,data.shape[1]):#data.shape[1]):       
        din[i][j] = data[Effective_Columns[j]][i];

X_train, X_test, y_train, y_test = train_test_split(din, dout, test_size=.5,
                                                    random_state=0)
random_state = np.random.RandomState(0)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
ROC_GEN('LinearSVM-50%', y_test, y_score)
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
ROC_GEN('RBFSVM-50%', y_test, y_score)
train = data_copy.sample(frac=0.8, random_state=1)
test = data_copy.loc[~data_copy.index.isin(train.index)]

rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf.fit(train[Effective_Columns], train["Attrition"])
predictions = rf.predict(test[Effective_Columns])

len = predictions.shape[0];
test_label = [0 for x in range(len)] 
test_attr = test["Attrition"];

for i in range(len):
    test_label[i] = test_attr[test_attr.index[i]];

ROC_GEN('RF-80%', test_label, predictions)


X_train, X_test, y_train, y_test = train_test_split(din, dout, test_size=.2,
                                                    random_state=0)
######################################## SVM LINEAR 
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
ROC_GEN('LinearSVM-80%', y_test, y_score)


######################################## SVM RBF 
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
ROC_GEN('RBFSVM-80%', y_test, y_score)
Text_Baesd_columns = ["BusinessTravel","Department", "EducationField","OverTime","Gender","JobRole","MaritalStatus"];
#data[Text_baesd_columns]
data_copy[Text_Baesd_columns]

data_corrected = data_copy.copy()


data_corrected["BusinessTravel"] = data_copy["BusinessTravel"].replace(["Travel_Rarely","Travel_Frequently","Non-Travel"],[0,1,2]);

data_corrected["MaritalStatus"] = data_copy["MaritalStatus"].replace(["Single","Married","Divorced"],[0,1,2]);
data_corrected["Gender"] = data_copy["Gender"].replace(["Female","Male"],[0,1]);

data_corrected["Department"] = data_copy["Department"].replace(["Sales","Research & Development","Human Resources"],[0,1,2]);

data_corrected["EducationField"] = data_copy["EducationField"].replace(["Life Sciences","Other","Medical","Marketing","Technical Degree","Human Resources"],[0,1,2,3,4,5]);

data_corrected["OverTime"] = data_copy["OverTime"].replace(["Yes","No"],[0,1]);

data_corrected["JobRole"] = data_copy["JobRole"].replace(["Sales Executive","Sales Representative","Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Manager","Research Director","Human Resources"],[0,1,2,3,4,5,6,7,8]);

Corrected_columns = Effective_Columns + Text_Baesd_columns 
######################################## Classification with new columns
train = data_corrected.sample(frac=0.5, random_state=1)
test = data_corrected.loc[~data_corrected.index.isin(train.index)]
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf.fit(train[Corrected_columns], train["Attrition"])
predictions = rf.predict(test[Corrected_columns])

len = predictions.shape[0];
test_label = [0 for x in range(len)] 
test_attr = test["Attrition"];

for i in range(len):
    test_label[i] = test_attr[test_attr.index[i]];

ROC_GEN('Corrected-Data-RF1-50%', test_label, predictions)

######################################## RANDOM FOREST2
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)
rf.fit(train[Corrected_columns], train["Attrition"])
predictions = rf.predict(test[Corrected_columns])

ROC_GEN('Corrected-Data-RF2-50%', test_label, predictions)

#---------------SVM PREPARE 
data = data_copy[Corrected_columns];
out = data_copy["Attrition"];
len = out.shape[0];
dout = [0 for x in range(len)] 
#test_attr = test["Attrition"];

for i in range(len):
    dout[i] = out[out.index[i]];


din = [[0 for x in range(data.shape[1])] for y in range(len)] 

data_input = data_corrected[Corrected_columns]


for i in range(len):
    for j in range(data.shape[1]-9,data.shape[1]):  #data_input.shape[1]   
        din[i][j] = data_input[Corrected_columns[j]][i];
    
    
    
X_train, X_test, y_train, y_test = train_test_split(din, dout, test_size=.5,
                                                    random_state=0)

######################################## SVM LINEAR 
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
ROC_GEN('Corrected-Data-LinearSVM-50%', y_test, y_score)


######################################## SVM RBF 
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
ROC_GEN('Corrected-Data-RBFSVM-80%', y_test, y_score)

#§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§ Train 80%
#§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§ Train 80%

train = data_corrected.sample(frac=0.8, random_state=1)
test = data_corrected.loc[~data_corrected.index.isin(train.index)]

######################################## RANDOM FOREST1
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf.fit(train[Corrected_columns], train["Attrition"])
predictions = rf.predict(test[Corrected_columns])

len = predictions.shape[0];
test_label = [0 for x in range(len)] 
test_attr = test["Attrition"];

for i in range(len):
    test_label[i] = test_attr[test_attr.index[i]];

ROC_GEN('Corrected-Data-RF1-80%', test_label, predictions)


X_train, X_test, y_train, y_test = train_test_split(din, dout, test_size=.2,
                                                    random_state=0)
######################################## SVM LINEAR 
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
ROC_GEN('Corrected-Data-LinearSVM-80%', y_test, y_score)


######################################## SVM RBF 
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
ROC_GEN('Corrected-Data-RBFSVM-80%', y_test, y_score)
kind = ['svm'];
sm = [SMOTE(kind=k) for k in kind]
X_resampled = []
y_resampled = []
#X_res_vis = []
for method in sm:
    X_res, y_res = method.fit_sample(din, dout)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=.2,
                                                    random_state=0)
######################################## SVM LINEAR 
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
ROC_GEN('Balanced-Data-LinearSVM-80%', y_test, y_score)


######################################## SVM RBF 
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
ROC_GEN('Balanced-Data-RBFSVM-80%', y_test, y_score)

for method in sm:
    X_res, y_res = method.fit_sample(data_corrected[Corrected_columns], data_corrected["Attrition"])


rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

ROC_GEN('Balanced-Data-RF1-80%', y_test, predictions)