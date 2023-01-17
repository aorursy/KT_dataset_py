# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
('../input/ny-mental-patient-survey/Patient_Characteristics_Survey__PCS___2015.csv')
ng = pd.read_csv('../input/ny-mental-patient-survey/Patient_Characteristics_Survey__PCS___2015.csv')
ng.columns
ng['Diabetes'].unique()
ng1 = ng[ng['Diabetes']!="UNKNOWN"]
ng.shape
ng1.head()
ng1.groupby(['Diabetes']).size()/ng1.shape[0]
ng1.columns
ng2 = ng1[['Program Category', 'Region Served']]
ng2 = ng1[['Program Category', 'Region Served', 'Age Group', 'Sex','Race','Living Situation', 'Household Composition', 'Preferred Language','Veteran Status', 'Employment Status','Number Of Hours Worked Each Week', 'Education Status','Mental Illness','Intellectual Disability', 'Autism Spectrum','Other Developmental Disability', 'Alcohol Related Disorder','Drug Substance Disorder', 'Mobility Impairment Disorder','Hearing Visual Impairment', 'Hyperlipidemia', 'High Blood Pressure', 'Diabetes', 'Obesity', 'Heart Attack','Stroke', 'Other Cardiac','Pulmonary Asthma', 'Alzheimer or Dementia', 'Kidney Disease','Liver Disease', 'Endocrine Condition', 'Neurological Condition','Traumatic Brain Injury', 'Joint Disease', 'Cancer', 'Other Chronic Med Condition', 'No Chronic Med Condition','Unknown Chronic Med Condition', 'Smokes','Received Smoking Medication', 'Received Smoking Counseling','Serious Mental Illness']]
ng2.shape
ng2.head()
y = ng2['Diabetes']

X = ng2.drop(['Diabetes'],1)
from sklearn.preprocessing import LabelEncoder
cols = list(ng2.columns)
#ng2.isna().sum()
#ng2.columns
le = LabelEncoder()

le.fit_transform(ng2['Program Category'])
for c in cols:

    le  = LabelEncoder()

    ng2[c] = le.fit_transform(ng2[c])
ng2.head()
y = ng2['Diabetes']

X = ng2.drop(['Diabetes'],1)
X.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape
X_test.shape
#X_train.groupby(['Diabetes']).size()/X_train.shape[0]
#X_test.groupby(['Diabetes']).size()/X_test.shape[0]
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier
classifier.fit(X_train, y_train)
y_predict = classifier.predict_proba(X_test)[:,1]
import seaborn as sns

sns.set_style("darkgrid")

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score
fpr, tpr, thresholds = roc_curve( y_test, y_predict)

auc_score = auc(fpr, tpr)

lgb_auc_score = auc_score

print("AUC : "+str(lgb_auc_score))

plt.figure(figsize=(16, 6))

lw = 2

plt.plot(fpr, tpr, lw=lw, label='ROC curve (area = %0.5f)' % lgb_auc_score)  

plt.plot([0, 1], [0, 1], color='g', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.show() 