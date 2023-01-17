#Data Analysis Libraries

import numpy as np

import pandas as pd

from pandas import Series,DataFrame

import statistics as st

from scipy import stats

from scipy import interp

import statistics as st

import math

import os

from datetime import datetime

import itertools



#Visualization Libraries

import matplotlib

import matplotlib.pyplot as plt

from matplotlib import rcParams

import seaborn as sns

sns.set(color_codes=True)

from IPython.display import HTML

from IPython.display import display

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected = True)

%matplotlib inline

from IPython.core.display import HTML

from pdpbox import pdp, info_plots

import shap

shap.initjs()

def multi_table(table_list):

    ''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell

    '''

    return HTML(

        '<table><tr style="background-color:white;">' + 

        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +

        '</tr></table>'

    )



#sklearn

from sklearn import metrics

from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score,confusion_matrix, classification_report, confusion_matrix, jaccard_similarity_score, f1_score, fbeta_score



from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler, Imputer,MinMaxScaler, LabelEncoder



from sklearn import model_selection

from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score, validation_curve, RandomizedSearchCV, cross_val_predict, StratifiedKFold



from sklearn import linear_model

from sklearn.linear_model import LogisticRegression, LinearRegression



from sklearn import naive_bayes

from sklearn.naive_bayes import GaussianNB



from sklearn import neighbors

from sklearn.neighbors import KNeighborsClassifier



from sklearn import tree

from sklearn.tree import DecisionTreeClassifier



from sklearn import ensemble

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostRegressor



from sklearn.svm import SVC



from sklearn.ensemble import AdaBoostClassifier



from sklearn import datasets



#misc

from functools import singledispatch

import eli5

from eli5.sklearn import PermutationImportance

import shap

from mpl_toolkits.mplot3d import Axes3D

import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))

from xgboost import XGBClassifier

import lightgbm as lgb

import warnings

heart = pd.read_csv("../input/heart.csv")
heart2= heart.drop(heart.index[164])

heart2.columns=['age', 'sex', 'cpain','resting_BP', 'chol', 'fasting_BS', 'resting_EKG', 

                'max_HR', 'exercise_ANG', 'ST_depression', 'm_exercise_ST', 'no_maj_vessels', 'thal', 'target']



heart2['chol']=heart2['chol'].replace([417, 564], 240)

heart2['chol']=heart2['chol'].replace([407, 409], 249)



heart2['ST_depressionAB']=heart2['ST_depression'].apply(lambda row: 1 if row > 0 else 0)

heart2A=heart2.iloc[:,0:11]

heart2B=heart2.iloc[:,11:14]

heart2C=heart2.loc[:,'ST_depressionAB']

heart2C=pd.DataFrame(heart2C)

heart2C.head()

heart2 = pd.concat([heart2A, heart2C, heart2B], axis=1, join_axes=[heart2A.index])



heart2.loc[48, 'thal']=2.0

heart2.loc[281, 'thal']=3.0



PHD=heart2.loc[heart2.loc[:,'target']==1]

AHD=heart2.loc[heart2.loc[:,'target']==0]

heart2.head()
#heart3 (descriptive)

heart3=pd.DataFrame.copy(heart2)



heart3['sex']=heart3['sex'].replace([1, 0], ['Male', 'Female'])

heart3['cpain']=heart3['cpain'].replace([0, 1, 2, 3], ['Asymptomatic', 'Typical Angina', 'Atypical Angina', 'Non-Angina'])

heart3['fasting_BS']=heart3['fasting_BS'].replace([1, 0], ['BS > 120 mg/dl', 'BS < 120 mg/dl'])

heart3['resting_EKG']=heart3['resting_EKG'].replace([0, 1, 2], ['Normal', 'Left Ventricular Hypertrophy', 'ST-T Wave Abnormality'])

heart3['exercise_ANG']=heart3['exercise_ANG'].replace([0, 1], ['Absent', 'Present'])

heart3['m_exercise_ST']=heart3['m_exercise_ST'].replace([0, 1, 2], ['Upsloping', 'Flat', 'Downsloping'])

heart3['thal']=heart3['thal'].replace([1, 2, 3], ['Fixed Defect', 'Normal', 'Reversible Defect'])

heart3['target']=heart3['target'].replace([0, 1], ['Absent', 'Present'])



heart3['chol']=heart3['chol'].replace([417, 564], 240)

heart3['chol']=heart3['chol'].replace([407, 409], 249)



heart3.loc[48, 'thal']="Normal"

heart3.loc[281, 'thal']="Reversible Defect"



PHD3=heart3.loc[heart3.loc[:,'target']=="Present"]

AHD3=heart3.loc[heart3.loc[:,'target']=="Absent"]

heart3.head()

numrows= heart3.shape[0]

numcolumns=heart3.shape[1]

display(heart3.head(5), heart3.describe(), print("Number of Rows:", numrows),print("Number of Columns:", numcolumns))
ax1 = sns.countplot(heart3['target'], palette="BuPu")

plt.title("Distribution of HD Diagnosis", size=30)

plt.ylabel("Frequency", labelpad=40, size=20)

plt.xlabel("HD Diagnosis", labelpad=40, size=20)
fig = plt.figure(figsize=(20,20))





plt.subplot(3, 2, 1)

warnings.filterwarnings('ignore')

ax1 = sns.distplot(heart2['age'], kde=False, color='blueviolet')

ax1.set_xlabel("Age (yrs)")

second_ax1 = ax1.twinx()

second_ax1.yaxis.set_label_position("left")

sns.distplot(heart2['age'], ax=second_ax1, kde=True, hist=False, color='blue')

second_ax1.set_yticks([])

plt.title("Distribution of Age", size=15)

plt.ylabel("Frequency")



plt.subplot(3, 2, 2)

warnings.filterwarnings('ignore')

ax1 = sns.distplot(heart2['max_HR'], kde=False, color='blueviolet')

ax1.set_xlabel("Maximum HR (bpm)")

second_ax1 = ax1.twinx()

second_ax1.yaxis.set_label_position("left")

sns.distplot(heart2['max_HR'], ax=second_ax1, kde=True, hist=False, color='blue')

second_ax1.set_yticks([])

plt.title("Distribution of Maximum Heart Rate", size=15)

plt.ylabel("Frequency")



plt.subplot(3, 2, 3)

warnings.filterwarnings('ignore')

ax1 = sns.distplot(heart2['resting_BP'], kde=False, color='blueviolet')

ax1.set_xlabel("Resing Systolic BP (mm Hg)")

second_ax1 = ax1.twinx()

second_ax1.yaxis.set_label_position("left")

sns.distplot(heart2['resting_BP'], ax=second_ax1, kde=True, hist=False, color='blue')

second_ax1.set_yticks([])

plt.title("Distribution of Resting Systolic Blood Pressure", size=15)

plt.ylabel("Frequency")



plt.subplot(3, 2, 4)

warnings.filterwarnings('ignore')

ax1 = sns.distplot(heart2['ST_depression'], kde=False, color='blueviolet')

ax1.set_xlabel("ST Depression")

second_ax1 = ax1.twinx()

second_ax1.yaxis.set_label_position("left")

sns.distplot(heart2['ST_depression'], ax=second_ax1, kde=True, hist=False, color='blue')

second_ax1.set_yticks([])

plt.title("Distribution of Exercise Induced ST Depression", size=15)

plt.ylabel("Frequency")

plt.xlim(0,7)



plt.subplot(3, 2, 5)

warnings.filterwarnings('ignore')

ax1 = sns.distplot(heart2['chol'], kde=False, color='blueviolet')

ax1.set_xlabel("Serum Cholesterol (mg/dl)")

second_ax1 = ax1.twinx()

second_ax1.yaxis.set_label_position("left")

sns.distplot(heart2['chol'], ax=second_ax1, kde=True, hist=False, color='blue')

second_ax1.set_yticks([])

plt.title("Distribution of Serum Cholesterol",size=15)

plt.ylabel("Frequency")



plt.show()



fig = plt.figure(figsize=(20,20))

plt.subplot(3, 3, 1)

sns.countplot(heart3['sex'], palette="BuPu")

plt.title("Gender Distribution", size=15)

plt.ylabel("Frequency")

plt.xlabel("Provider-Identified Gender")



plt.subplot(3, 3, 2)

sns.countplot(heart3['cpain'], palette="BuPu")

plt.title("Distribution of Chest Pain Type", size=15)

plt.ylabel("Frequency")

plt.xlabel("Chest Pain Description")



plt.subplot(3, 3, 3)

sns.countplot(heart3['fasting_BS'], palette="BuPu")

plt.title("Fasting Blood Sugar Distribution", size=15)

plt.ylabel("Frequency")

plt.xlabel("Level of Fasting BS (mmol/L)")



plt.subplot(3, 3, 4)

sns.countplot(heart3['resting_EKG'], palette="BuPu")

plt.title("Distribution of Resting EKG Results", size=15)

plt.ylabel("Frequency")

plt.xlabel("EKG Results")



plt.subplot(3, 3, 5)

sns.countplot(heart3['exercise_ANG'], palette="BuPu")

plt.title("Distribution of Exercise Induced Angina", size=15)

plt.ylabel("Frequency")

plt.xlabel("Exercise Induced Angina")



plt.subplot(3, 3, 6)

sns.countplot(heart3['m_exercise_ST'], palette="BuPu")

plt.title("Distribution of the ST Segment Slope", size=15)

plt.ylabel("Frequency")

plt.xlabel("Slope  (Peak Exercise)")



plt.subplot(3, 3, 7)

sns.countplot(heart3['ST_depressionAB'], palette="BuPu")

plt.title("ST Depression Abnormalities", size=15)

plt.ylabel("Frequency")

plt.xlabel("ST Depression Abnormalities")



plt.subplot(3, 3, 8)

sns.countplot(heart3['no_maj_vessels'], palette="BuPu")

plt.title("No. of Major Vessels Colored by Flouroscopy", size=15)

plt.ylabel("Frequency")

plt.xlabel("Number of Major Vessels")



plt.subplot(3, 3, 9)

sns.countplot(heart3['thal'], palette="BuPu")

plt.title("Thalium Stress Test Results", size=15)

plt.ylabel("Frequency")

plt.xlabel("Results")
mask = np.zeros_like(heart2.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True 

plt.figure(figsize=(20,20))

sns.heatmap(heart2.corr(),vmax=.8, center=0,

            square=True, linewidths=.1, mask=mask, cbar_kws={"shrink": .5},annot=True)
heart2.corr()
corre=heart2.corr()

TargetCorr=corre.loc[:'thal','target']

TargetCorr=pd.DataFrame(TargetCorr)

TargetCorr['AbsVal']=TargetCorr['target'].apply(lambda row: abs(row))

TargetCorr['Rank']=pd.DataFrame.rank(TargetCorr['AbsVal'])

TargetCorr['Feature']=TargetCorr.index

TargetCorr = TargetCorr.set_index('Rank') 

TargetCorr = TargetCorr.sort_index(ascending=0)

TargetCorr = TargetCorr.set_index('Feature') 

TargetCorr=TargetCorr.loc[:,'target']

TargetCorr=pd.DataFrame(TargetCorr)

TargetCorr.columns=["Correlation with Target"]

TargetCorr
PHD=heart2.loc[heart2.loc[:,"target"]==1]

AHD=heart2.loc[heart2.loc[:,"target"]==0]



from scipy.stats import ttest_ind

def rowz(ttest): 

    name=ttest_ind(PHD[ttest], AHD[ttest])

    name=list(name)

    name = pd.DataFrame(np.array(name))

    name=name.T

    col=["t-statistic", "p_value"]

    name.columns=col

    return name



AGE=rowz('age')

AGE.loc[:,"Names"]="Age"

RESTING_BP=rowz('resting_BP')

RESTING_BP.loc[:,"Names"]="Resting_BP"

CHOLESTEROL=rowz('chol')

CHOLESTEROL.loc[:,"Names"]="Cholesterol"

MAX_HR=rowz('max_HR')

MAX_HR.loc[:,"Names"]="Max_HR"

ST_DEP=rowz('ST_depression')

ST_DEP.loc[:,"Names"]="ST_Depression"



PVALS = pd.concat([AGE, RESTING_BP,CHOLESTEROL,MAX_HR, ST_DEP], axis=0)

PVALS=PVALS.set_index(PVALS["Names"])

P_VALS= PVALS.drop('Names',axis=1)



P_VALS
sns.pairplot(heart2,vars = ['resting_BP', 'chol','max_HR','ST_depression', 'age'],hue='target')
#seperate independent (feature) and dependent (target) variables

#KNN cannot process text/ categorical data unless they are be converted to numbers

#For this reason I did not input the heart3 DataFrame created above

X=heart2.drop('target',1)

y=heart2.loc[:,'target']



#Scale the data

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



#Split the data into training and testing sets

X_train,X_test,y_train,y_test = train_test_split(X_scaled, y,test_size=.2,random_state=40)



#Call classifier and, using GridSearchCV, find the best parameters

knn = KNeighborsClassifier()

params = {'n_neighbors':[i for i in range(1,33,2)]}

modelKNN = GridSearchCV(knn,params,cv=10)

modelKNN.fit(X_train,y_train)

modelKNN.best_params_   



#Use the above model (modelKNN) to predict the y values corresponding to the X testing set

predictKNN = modelKNN.predict(X_test)



#Compare the results of the model's predictions (predictKNN) to the actual y values

accscoreKNN=accuracy_score(y_test,predictKNN)

print('Accuracy Score: ',accuracy_score(y_test,predictKNN))

print('Using k-NN we get an accuracy score of: ',

      round(accuracy_score(y_test,predictKNN),5)*100,'%')
perm = PermutationImportance(modelKNN).fit(X_test, y_test)

eli=eli5.show_weights(perm, feature_names = X.columns.tolist())

eli
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=.2)

X_test=pd.DataFrame(X_test)

X_test



base_features = X.columns.values.tolist()



feat_name = 'no_maj_vessels'



pdp_dist = pdp.pdp_isolate(model=modelKNN, dataset=X, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.ylim(-0.015,0.01)

#plt.xticks(np.arange(0, 4, step=1))

plt.show()
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=.2)

X_test=pd.DataFrame(X_test)

X_test



base_features = X.columns.values.tolist()



feat_name = 'age'



pdp_dist = pdp.pdp_isolate(model=modelKNN, dataset=X, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

#plt.ylim(-0.025,0.01)

#plt.xticks(np.arange(0, 4, step=1))

plt.show()
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=.2)

X_test=pd.DataFrame(X_test)

X_test



base_features = X.columns.values.tolist()



feat_name = 'chol'



pdp_dist = pdp.pdp_isolate(model=modelKNN, dataset=X, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

#plt.ylim(-0.025,0.01)

#plt.xticks(np.arange(0, 4, step=1))

plt.show()
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=.2)

X_test=pd.DataFrame(X_test)

X_test



base_features = X.columns.values.tolist()



feat_name = 'ST_depression'



pdp_dist = pdp.pdp_isolate(model=modelKNN, dataset=X, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

#plt.ylim(-0.025,0.01)

#plt.xticks(np.arange(0, 4, step=1))

plt.show()
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=.2)

X_test=pd.DataFrame(X_test)

X_test



base_features = X.columns.values.tolist()



feat_name = 'max_HR'



pdp_dist = pdp.pdp_isolate(model=modelKNN, dataset=X, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.ylim(-0.005,0.2)

#plt.xticks(np.arange(0, 4, step=1))

plt.show()
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=.2)

X_test=pd.DataFrame(X_test)

X_test



base_features = X.columns.values.tolist()



feat_name = 'resting_BP'



pdp_dist = pdp.pdp_isolate(model=modelKNN, dataset=X, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.ylim(-0.005,0.2)

#plt.xticks(np.arange(0, 4, step=1))

plt.show()
X= heart2.drop('target',1)

y= heart2['target']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y, test_size=.3,random_state=40)



clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)



feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)

feature_imp
# Creating a bar plot

sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels to your graph

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")

plt.legend()

plt.show()                 