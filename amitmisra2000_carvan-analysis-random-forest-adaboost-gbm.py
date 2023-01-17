# 1.1 Call Libraries and import Dataset.

%reset -f

import numpy as np

import pandas as pd

from pandas.io.parsers import read_csv

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeClassifier

import matplotlib as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error as MSE



#2.0 Import OS directory and import data from CSV file

import os          

carvan = read_csv('../input/caravan-insurance-challenge/caravan-insurance-challenge.csv')



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#2.1 Rename column names for better understanding of database



carvan_column_names  = {

'ORIGIN':     'Datacategory',

'MOSTYPE':    'Customer_Subtype',

'MAANTHUI':   'Number_of_houses',

'MGEMOMV':    'Avg_size_household',

'MGEMLEEF':   'Avg_age',

'MOSHOOFD':   'Customer_main_type',

'MGODRK':     'Roman_catholic',

'MGODPR':     'Protestant',

'MGODOV':     'Other_religion',

'MGODGE':     'No_religion',

'MRELGE':     'Married',

'MRELSA':     'Living_together',

'MRELOV':     'Other_relation',

'MFALLEEN':   'Singles',

'MFGEKIND':   'Household_without_children',

'MFWEKIND':   'Household_with_children',

'MOPLHOOG':   'High_level_education',

'MOPLMIDD':   'Medium_level_education',

'MOPLLAAG':   'Lower_level_education',

'MBERHOOG':   'High_status',

'MBERZELF':   'Entrepreneur',

'MBERBOER':   'Farmer',

'MBERMIDD':   'Middle_management',

'MBERARBG':   'Skilled_labourers',

'MBERARBO':   'Unskilled_labourers',

'MSKA':       'Social_class_A',

'MSKB1':      'Social_class_B1',

'MSKB2':      'Social_class_B2',

'MSKC':       'Social_class_C',

'MSKD':       'Social_class_D',

'MHHUUR':     'Rented_house',

'MHKOOP':     'Home_owners',

'MAUT1':      '1_car',

'MAUT2':      '2_cars',

'MAUT0':      'No_car',

'MZFONDS':    'National_Health_Service',

'MZPART':     'Private_health_insurance',

'MINKM30':    'Income_<_30000',

'MINK3045':   'Income_30-45000',

'MINK4575':   'Income_45-75000',

'MINK7512':   'Income_75-122000',

'MINK123M':   'Income_>123000',

'MINKGEM':    'Average_income',

'MKOOPKLA':   'Purchasing_power_class',

'PWAPART':    'Contribution_private_third_party_insurance',

'PWABEDR':    'Contribution_third_party_insurance_firms',

'PWALAND':    'Contribution_third_party_insurane_agriculture',

'PPERSAUT':   'Contribution_car_policies',

'PBESAUT':    'Contribution_delivery_van_policies',

'PMOTSCO':    'Contribution_motorcycle-scooter_policies',

'PVRAAUT':   'Contribution_lorry_policies',

'PAANHANG':   'Contribution_trailer_policies',

'PTRACTOR':   'Contribution_tractor_policies',

'PWERKT':     'Contribution_agricultural_machines_policies',

'PBROM':      'Contribution_moped_policies',

'PLEVEN':     'Contribution_life_insurances',

'PPERSONG':   'Contribution_private_accident_insurance_policies',

'PGEZONG':    'Contribution_family_accidents_insurance_policies',

'PWAOREG':   'Contribution_disability_insurance_policies',

'PBRAND':     'Contribution_fire_policies',

'PZEILPL':    'Contribution_surfboard_policies',

'PPLEZIER':   'Contribution_boat_policies',

'PFIETS':     'Contribution_bicycle_policies',

'PINBOED':   'Contribution_property_insurance_policies',

'PBYSTAND':  'Contribution_social_security_insurance_policies',

'AWAPART':   'Number_of_private_third_party_insurance_1-12',

'AWABEDR':   'Number_of_third_party_insurance_firms',

'AWALAND':   'Number_of_third_party_insurance_agriculture',

'APERSAUT':  'Number_of_car_policies',

'ABESAUT':   'Number_of_delivery_van_policies',

'AMOTSCO':   'Number_of_motorcycle-scooter_policies',

'AVRAAUT':   'Number_of_lorry_policies',

'AAANHANG':  'Number_of_trailer_policies',

'ATRACTOR':  'Number_of_tractor_policies',

'AWERKT':   'Number_of_agricultural_machines_policies',

'ABROM':      'Number_of_moped_policies',

'ALEVEN':     'Number_of_life_insurances',

'APERSONG':  'Number_of_private_accident_insurance_policies',

'AGEZONG':   'Number_of_family_accidents_insurance_policies',

'AWAOREG':   'Number_of_disability_insurance_policies',

'ABRAND':     'Number_of_fire_policies',

'AZEILPL':   'Number_of_surfboard_policies',

'APLEZIER':  'Number_of_boat_policies',

'AFIETS':     'Number_of_bicycle_policies',

'AINBOED':   'Number_of_property_insurance_policies',

'ABYSTAND':  'Number_of_social_security_insurance_policies',

'CARAVAN':   'Number_of_mobile_home_policies_0-1'

                        }

carvan.rename(

         columns = carvan_column_names,

         inplace = True

         )

carvan['Number_of_mobile_home_policies_0-1'].value_counts()
#2.2 Divide the Data into Training and Testing Data. 

#After division Carvan_train is my training dataset and Carvan_test is my testing dataset.

carvan_train = carvan.loc[carvan['Datacategory'] == 'train']

carvan_train = carvan_train.drop(['Datacategory'],axis =1)

carvan_test = carvan.loc[carvan['Datacategory'] == 'test']

carvan_test = carvan_test.drop(['Datacategory'],axis =1)



#2.3 Divide the carvan_train and carvan_test data into X_train,y_train,X_test and y_test



y_train = carvan_train.pop("Number_of_mobile_home_policies_0-1")

X_train = carvan_train

y_test = carvan_test.pop("Number_of_mobile_home_policies_0-1")

X_test = carvan_test

colunique=X_train.nunique

cols= (X_train.nunique() < 41 )

cols

cat_cols = cols[cols==True].index.tolist()

num_cols = cols[cols==False].index.tolist()

cat_cols

num_cols





#2.5 First Doing decision Tree Classification we find the score of dataset.



from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(max_depth=6)

dt.fit(X_train,y_train)

y_predict_dt = dt.predict(X_test)

score_dt =np.sum(y_predict_dt ==y_test)/len(y_test)

score_dt
#2.6 Since All the Columns in this Dataset is of type Categorical having nunique values less than 50 

#we now use Target Encoder for better results.

import pandas as pd

from category_encoders import TargetEncoder

from sklearn.preprocessing import StandardScaler

ct = ColumnTransformer([('cde',TargetEncoder(),cat_cols)],remainder ="passthrough")





from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators =400,oob_score = True,bootstrap=True)    

#2.5 Define Pipeline

pipe_rf = Pipeline([('ct',ct),('rf',rf)])

pipe_rf.fit(X_train,y_train)

y_predict_rf = pipe_rf.predict(X_test)

score_rf =np.sum(y_predict_rf ==y_test)/len(y_test)

score_rf

#2.6 Evaluate the performance of model by Confusion Metrics and Classification Report

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



cf_rf =confusion_matrix(y_test,y_predict_rf)

cf_rf

cr_rf = classification_report(y_test,y_predict_rf)

cr_rf
#3.0 Evaluating the Dataset Using Adaboost Classifier Model



#3.1 Import Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

#Import AdaBoost Classifier

from sklearn.ensemble import AdaBoostClassifier

ct = ColumnTransformer([('cde',TargetEncoder(),cat_cols)],remainder ="passthrough")



#Instantiate Decision Tree Classifier

dt = DecisionTreeClassifier(max_depth =1,random_state =1) 



#Instantiate AdaBoost Classifier

ada = AdaBoostClassifier(base_estimator =dt,n_estimators =100,random_state = 1)



#3.1 Define Pipeline

pipe_ada = Pipeline([('ct',ct),('ada',ada)])

pipe_ada.fit(X_train,y_train)

y_predict_ada = pipe_ada.predict(X_test)

score_ada =np.sum(y_predict_ada ==y_test)/len(y_test)

score_ada



#2.6 Evaluate the performance of model by Confusion Metrics and Classification Report for AdaBoost Model

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



cf_ada =confusion_matrix(y_test,y_predict_ada)

cf_ada

cr_ada = classification_report(y_test,y_predict_ada)

cr_ada
#4.0 Evaluating the Dataset Using Gradiant Boosting Model



#Import and Instantiate Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators = 100,max_depth = 1, random_state =2)

pipe_gb = Pipeline([('ct',ct),('gb',gb)])

pipe_gb.fit(X_train,y_train)

y_predict_gb = pipe_gb.predict(X_test)

score_gb =np.sum(y_predict_gb ==y_test)/len(y_test)

score_gb

#2.6 Evaluate the performance of model by Confusion Metrics and Classification Report using Gradient Boosting Model

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



cf_gb =confusion_matrix(y_test,y_predict_gb)

cf_gb

cr_gb = classification_report(y_test,y_predict_gb)

cr_gb
# 2.7 Generating ROC Curve

import pandas as pd

from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression



ct = ColumnTransformer([('cde',TargetEncoder(),cat_cols)],remainder ="passthrough")

logreg = LogisticRegression(max_iter =5000)

#2.5 Define Pipeline

pipe_logreg = Pipeline([('ct',ct),('logreg',logreg)])



pipe_logreg.fit(X_train,y_train)









import matplotlib.pyplot as plt

# Computing predicted probablities of this dataset

y_predictprob_lr = logreg.predict_proba(X_test)[:,1]



#Generate ROC Curve



fpr_lr,tpr_lr,thresholds = roc_curve(y_test,y_predictprob_lr)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_lr,tpr_lr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()

# Finding of Area Under the Curve (AUC score) of the metrics

from sklearn.metrics import roc_auc_score



auc = roc_auc_score(y_test,y_predictprob_lr)

auc
#5.0 Finding Important Features of Carvan Dataset.

from sklearn.datasets import make_classification

from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot

# define dataset

X_train, y_train = make_classification(n_samples=1000, n_features=100, n_informative=5, n_redundant=5, random_state=1)

# define the model

model = RandomForestClassifier()

# fit the model

model.fit(X_train, y_train)

# get importance

importance = model.feature_importances_

# summarize feature importance

for i,v in enumerate(importance):

	print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance

pyplot.bar([x for x in range(len(importance))], importance)

pyplot.show()