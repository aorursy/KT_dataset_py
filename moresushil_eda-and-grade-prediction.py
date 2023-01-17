import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
fpmath = '/kaggle/input/student-alcohol-consumption/student-mat.csv'
fportugese = '/kaggle/input/student-alcohol-consumption/student-por.csv'
datamath = pd.read_csv(fpmath)
dataport = pd.read_csv(fportugese)
datamath.columns
binary_features = [
    
        'school', 'sex', 'address', 'famsize', 'Pstatus','schoolsup', 'famsup', 'paid', 
        'activities', 'nursery', 'higher', 'internet', 'romantic'
    
]

numeric_features = [
    
        'age', 'Medu', 'Fedu', 'traveltime', 'studytime','failures', 'famrel', 
        'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'
    
]

nominal_features = [
    
        'Mjob', 'Fjob', 'reason', 'guardian'
    
]

y = ['G1', 'G2', 'G3']
datamath[binary_features]
datamath[numeric_features]
datamath[nominal_features]
datamath[y]
datamath.info()
#for col in datamath.columns:
 #   print(col+'    \t', len(datamath[col].unique()), ' categories')
# creating initial dataframe
#bridge_types = ('Arch','Beam','Truss','Cantilever','Tied Arch','Suspension','Cable')
#bridge_df = pd.DataFrame(bridge_types, columns=['Bridge_Types'])
# converting type of columns to 'category'
#bridge_df['Bridge_Types'] = bridge_df['Bridge_Types'].astype('category')
# Assigning numerical values and storing in another column
#bridge_df['Bridge_Types_Cat'] = bridge_df['Bridge_Types'].cat.codes
#bridge_df
adata = pd.read_csv(fpmath)
for col in adata[nominal_features]:
    adata[col] = adata[col].astype('category')
    # Assigning numerical values and storing in another column
    #data[col+'_coded'] = data[col].cat.codes
for col in adata[numeric_features]:
    adata[col] = adata[col].astype('int')
adata['age'] = adata['age'].astype('int')
adata.info()
datanew = pd.get_dummies(adata)
datanew
datanew.shape
X = [
    [
        'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
        'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'school_GP', 'school_MS', 'sex_F', 'sex_M', 'address_R',
        'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T',
        'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services',
        'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other',
        'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home',
        'reason_other', 'reason_reputation', 'guardian_father',
        'guardian_mother', 'guardian_other', 'schoolsup_no', 'schoolsup_yes',
        'famsup_no', 'famsup_yes', 'paid_no', 'paid_yes', 'activities_no',
        'activities_yes', 'nursery_no', 'nursery_yes', 'higher_no',
        'higher_yes', 'internet_no', 'internet_yes', 'romantic_no',
        'romantic_yes'
    ]
]
y = [['G1', 'G2', 'G3']]
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)
from sklearn.ensemble import RandomForestClassifier
#X = [[0, 0], [1, 1]]
#Y = [0, 1]
clf = RandomForestClassifier(n_estimators=150, max_features=7)
clf = clf.fit(X_train, y_train)
pd.DataFrame(clf.predict(X_test), columns=['G1','G2','G3'], index=X_test.index)
#help(pd.DataFrame)
corr = datanew.corr()
plt.figure(figsize=(100,100))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot= True)
datanew.columns
