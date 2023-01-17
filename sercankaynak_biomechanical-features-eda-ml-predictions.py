# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# matplotlib
import matplotlib.pyplot as plt

# seaborn
import seaborn as sns

#plotly
import plotly.io as pio
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected = True)
import plotly.graph_objs as go

from collections import Counter

import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_2c = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data_2c.head()
data_2c.info()
data_2c.tail()
data_2c.columns
g = sns.pairplot(data_2c,hue = "class",palette = "husl")
from IPython.display import Image
Image("../input/pelvicimage1/pelvic2.jpg")
data_2c.info()
def hist_plot(variable):
    plt.figure(figsize = (9,4))
    plt.hist(data_2c[variable],bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution".format(variable))
    plt.show
numvar = ["pelvic_incidence", "pelvic_tilt numeric", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"]
for n in numvar:
    hist_plot(n)
f,ax = plt.subplots(figsize = (8,8))
sns.boxplot(data=data_2c, orient="h", palette="Set2")
plt.show()
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indexes
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        
        # store indexes
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i,v in outlier_indices.items() if v > 1)
    
    return multiple_outliers
data_2c.loc[detect_outliers(data_2c,["pelvic_incidence", "pelvic_tilt numeric", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"])]
#drop outliers
data_2c = data_2c.drop(detect_outliers(data_2c,["pelvic_incidence", "pelvic_tilt numeric", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree_spondylolisthesis"]),axis = 0).reset_index(drop = True)
data_2c.info()
data_2c.columns[data_2c.isnull().any()]
data_2c.isnull().sum()
data_2c.head()
mask = np.zeros_like(data_2c.corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(data_2c.corr(),linewidths=0.25,vmax=0.7,square=True,cmap="RdBu", #"BuGn_r" to reverse 
            linecolor='w',annot=True,annot_kws={"size":12},mask=mask,cbar_kws={"shrink": .9});
f,ax = plt.subplots(figsize = (12,12))
data_2c_melt = pd.melt(data_2c,"class",var_name = "measurement")
sns.swarmplot(x="measurement", y="value", hue="class",
              palette=["r", "c", "y"], data=data_2c_melt)
plt.show()
from sklearn.model_selection import train_test_split, StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
data_2c.head()
data_2c.tail()
data_2c["class"] = [1 if i == "Normal" else 0 for i in data_2c["class"]]
data_2c.head()
data_2c.tail()
y = data_2c["class"]
x_data = data_2c.drop(["class"],axis = 1)
y
# normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
# train - test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)

print("x_train",len(x_train))
print("x_test",len(x_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
x_train
x_test
y_train
y_test
logisticreg = LogisticRegression()
logisticreg.fit(x_train,y_train)

acc_log_train = round(logisticreg.score(x_train,y_train)*100,2)
acc_log_test = round(logisticreg.score(x_test,y_test)*100,2)
print("Training = Accuracy : % {}".format(acc_log_train))
print("Testing = Accuracy : % {}".format(acc_log_test))
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
prediction
print("{} nn score : {}".format(3,knn.score(x_test,y_test)))
# find k value
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=3)
accuracies = cross_val_score(estimator=knn,X = x_train,y = y_train,cv = 10)
accuracies
print("average accuracy : ",np.mean(accuracies))
print("average std : ",np.std(accuracies))
# test
knn.fit(x_train,y_train)
print("test accuracy : ",knn.score(x_test,y_test))
param_grid = {"C" : np.logspace(-3,3,7),"penalty" : ["l1","l2"]} # l1= lasso  l2 = ridge
logisticreg = LogisticRegression()
logisticreg_cv = GridSearchCV(logisticreg,param_grid,cv = 10)
logisticreg_cv.fit(x_train,y_train)
print("tuned hyperparameters : (best parameters) :",logisticreg_cv.best_params_)
print("accuracy : ",logisticreg_cv.best_score_)
logisticreg2 = LogisticRegression(C = 100.0,penalty = "l2")
logisticreg2.fit(x_train,y_train)
print("score : ",logisticreg2.score(x_test,y_test))
grid = {"n_neighbors" : np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn,grid,cv = 10)
knn_cv.fit(x,y)
# print hyperparameter => K value in KNN algorithm
print("tuned hyperparameter K : ",knn_cv.best_params_)
print("accuracy according to tuned parameter : ", knn_cv.best_score_)
random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
              SVC(random_state = random_state),
              RandomForestClassifier(random_state = random_state),
              LogisticRegression(random_state = random_state),
              KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                 "max_depth" : range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                  "gamma" : [0.001,0.01,0.1,1],
                  "C" : [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features" : [1,3,10],
                 "min_samples_split" : [2,3,10],
                 "min_samples_leaf" : [1,3,10],
                 "bootstrap" : [False],
                 "n_estimators" : [100,300],
                 "criterion" : ["gini"]}

logreg_param_grid = {"C" : np.logspace(-3,3,7),
                     "penalty" : ["l1","l2"]}

knn_param_grid = {"n_neighbors" : np.linspace(1,19,10,dtype = int).tolist(),
                  "weights" : ["uniform","distance"],
                  "metric" : ["euclidean","manhattan"]}

classifier_param = [dt_param_grid,svc_param_grid,rf_param_grid,logreg_param_grid,knn_param_grid]
cv_result = []
best_estimators = []

for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid = classifier_param[i],cv = StratifiedKFold(n_splits = 10),scoring = "accuracy",n_jobs = -1,verbose = 1)
    clf.fit(x_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means": cv_result,"ML Models" : ["DecisionTreeClassifier","SVM","RandomForestClassifier",
                                                                              "LogisticRegression","KNeighborsClassifier"]})
g  = sns.barplot("Cross Validation Means","ML Models",data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")
plt.show()
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                         ("rfc",best_estimators[2]),
                                         ("lr",best_estimators[3])],
                                         voting = "soft",n_jobs = -1)
votingC = votingC.fit(x_train,y_train)
print(accuracy_score(votingC.predict(x_test),y_test))
