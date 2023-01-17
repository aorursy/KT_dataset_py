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
%ls ../input
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc,classification_report

from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

pd.options.mode.chained_assignment = None

np.random.seed(687)
df = pd.read_csv('../input/heart-disease-uci/heart.csv')

df.head(10)

df.rename({'cp':'chest_pain_type','trestbps':'resting_blood_pressure','chol':'cholesterol','fbs':'fasting_blood_sugar','restecg':'rest_ecg','thalach':'max_heart_rate_achieved','exang':'exercise_induced_angina','oldpeak':'st_depression','slope':'st_slope','ca':'num_major_vessels','thal':'thalassemia'},axis=1,inplace=True)
df.head(5)
df['sex'][df['sex'] == 0] = 'female'

df['sex'][df['sex'] == 1] = 'male'

#print(len(df.columns))

df['chest_pain_type'][df['chest_pain_type'] == 0] = 'typical angina'

df['chest_pain_type'][df['chest_pain_type'] == 1] = 'atypical angina'

df['chest_pain_type'][df['chest_pain_type'] == 2] = 'non-anginal pain'

df['chest_pain_type'][df['chest_pain_type'] == 3] = 'asymptomatic'



df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'

df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'



df['rest_ecg'][df['rest_ecg'] == 0] = 'normal'

df['rest_ecg'][df['rest_ecg'] == 1] = 'ST-T wave abnormality'

df['rest_ecg'][df['rest_ecg'] == 2] = 'left ventricular hypertrophy'



df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no'

df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes'



df['st_slope'][df['st_slope'] == 0] = 'upsloping'

df['st_slope'][df['st_slope'] == 1] = 'flat'

df['st_slope'][df['st_slope'] == 2] = 'downsloping'



df['thalassemia'][df['thalassemia'] == 1] = 'normal'

df['thalassemia'][df['thalassemia'] == 2] = 'fixed defect'

df['thalassemia'][df['thalassemia'] == 3] = 'reversable defect'
df.head()
df['sex'] = df['sex'].astype('object')

df['chest_pain_type'] = df['chest_pain_type'].astype('object')

df['fasting_blood_sugar'] = df['fasting_blood_sugar'].astype('object')

df['rest_ecg'] = df['rest_ecg'].astype('object')

df['exercise_induced_angina'] = df['exercise_induced_angina'].astype('object')

df['st_slope'] = df['st_slope'].astype('object')

df['thalassemia'] = df['thalassemia'].astype('object')
df.dtypes
df = pd.get_dummies(df,prefix=['st_slope'],columns=['st_slope'])
df.head()
df = pd.get_dummies(df, drop_first=True)
df.dtypes
df.head()
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', 1), df['target'], test_size = .271, random_state=37)

model = SVC(kernel='linear',gamma='scale',probability=True)

model.fit(X_train, y_train)
acc = model.score(X_test,y_test)*100

print("Accuracy = ",acc)



y_predict = model.predict(X_test)

y_pred_quant = model.predict_proba(X_test)[:, 1]

y_pred_bin = model.predict(X_test)



confusion_matrix = confusion_matrix(y_test, y_pred_bin)

total=sum(sum(confusion_matrix))



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])

print('Sensitivity : ', sensitivity )



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])

print('Specificity : ', specificity)



fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

print(auc(fpr, tpr))