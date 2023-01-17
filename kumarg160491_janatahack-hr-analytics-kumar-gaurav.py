# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# https://www.kaggle.com/iamhungundji/trees-and-neural-network-auc-0-6913

#some codes are inspired by above link





##Importing the packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Visualization packages

import matplotlib.pyplot as plt 

import seaborn as sns 



#Machine Learning packages

from sklearn.svm import SVC,NuSVC

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

#from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB,MultinomialNB

#from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score

#from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA

import lightgbm

#Suppress warnings

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/train_jqd04QH.csv')

train.head()
test=pd.read_csv('/kaggle//input/test_KaymcHn.csv')

test.head()
train.shape,test.shape
combine=train.append(test)
combine.shape
combine.isnull().sum()
combine['company_size'].value_counts()
combine['company_size'].fillna("Unknown", inplace=True)

combine['company_size'] = combine['company_size'].replace('50-99','com_Tier3')

combine['company_size'] = combine['company_size'].replace('100-500','com_Tier4')

combine['company_size'] = combine['company_size'].replace('10/49','com_Tier2')

combine['company_size'] = combine['company_size'].replace('10000+','com_Tier8')

combine['company_size'] = combine['company_size'].replace('1000-4999','com_Tier6')

combine['company_size'] = combine['company_size'].replace('<10','com_Tier1')

combine['company_size'] = combine['company_size'].replace('500-999','com_Tier5')

combine['company_size'] = combine['company_size'].replace('5000-9999','com_Tier7')

combine['company_size'].value_counts()
combine['gender'].value_counts()
combine.gender.fillna('Unknown',inplace=True)
combine['relevent_experience'].value_counts()
combine.relevent_experience=combine.relevent_experience.replace('Has relevent experience','Yes_RExp')

combine.relevent_experience=combine.relevent_experience.replace('No relevent experience','No_RExp')
combine['enrolled_university'].value_counts()
combine.enrolled_university.fillna('Unknown',inplace=True)

combine.education_level.value_counts()
combine.education_level.fillna(value=0,inplace=True)

combine.education_level=combine.education_level.replace('Graduate',3)

combine.education_level=combine.education_level.replace('Masters',4)

combine.education_level=combine.education_level.replace('High School',2)

combine.education_level=combine.education_level.replace('Phd',5)

combine.education_level=combine.education_level.replace('Primary School',1)
combine['major_discipline'].value_counts()
combine.major_discipline.fillna('Unknown',inplace=True)

combine.major_discipline=combine.major_discipline.replace('Business Degree','Business_Degree')

combine.major_discipline=combine.major_discipline.replace('No Major','No_Major')
combine.experience.fillna(-1,inplace=True)

combine.experience=combine.experience.replace('>20',21)

combine.experience=combine.experience.replace('<1',0)

combine['experience'] = combine['experience'].astype('int')
bins= [-1,0,3,6,9,12,15,18,21]

labels = ['Unknown','Exp_Tier1','Exp_Tier2','Exp_Tier3','Exp_Tier4','Exp_Tier5','Exp_Tier6','Exp_Tier7']

combine.experience = pd.cut(combine.experience, bins=bins, labels=labels, right=False)

combine.experience.value_counts()
combine['company_type'].value_counts()
combine.company_type.fillna("Unknown", inplace=True)

combine.company_type=combine.company_type.replace('Pvt Ltd','Pvt_Ltd')

combine.company_type=combine.company_type.replace('Funded Startup','Funded_Startup')

combine.company_type=combine.company_type.replace('Public Sector','Public_Sector')

combine.company_type=combine.company_type.replace('Early Stage Startup','Early_Stage_Startup')
combine.last_new_job.fillna(-1,inplace=True)

combine.last_new_job=combine.last_new_job.replace('>4',5)

combine.last_new_job=combine.last_new_job.replace('never',0)

combine.last_new_job=combine.last_new_job.astype(int)
combine.training_hours.describe()
combine.training_hours=np.log(combine.training_hours)
combine.training_hours.describe()
combine.info()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

combine['city'] = encoder.fit_transform(combine['city'])
combine.head()
train_cleaned = combine[combine['target'].isnull()!=True].drop(['enrollee_id'], axis=1)
combine=pd.get_dummies(combine)
combine.shape
X= combine[combine['target'].isnull()!=True].drop(['enrollee_id','target'], axis=1)

y = combine[combine['target'].isnull()!=True]['target']



x_test = combine[combine['target'].isnull()==True].drop(['enrollee_id','target'], axis=1)



X.shape, y.shape, x_test.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.pipeline import Pipeline



pipeline_lr=Pipeline([('scaler1',StandardScaler()),

                      ('pca1',PCA(n_components=2)),

                      ('lr_classifier',LogisticRegression(random_state=0))])



pipeline_svc=Pipeline([('scaler2',StandardScaler()),

                      ('pca2',PCA(n_components=2)),

                      ('svc_classifier',SVC(random_state=0))])



pipeline_dt=Pipeline([('scaler3',StandardScaler()),

                      ('pca3',PCA(n_components=2)),

                      ('dt_classifier',DecisionTreeClassifier(random_state=0))])



pipeline_rf=Pipeline([('scaler4',StandardScaler()),

                      ('pca4',PCA(n_components=2)),

                      ('rf_classifier',RandomForestClassifier(random_state=0))])



pipeline_xgb=Pipeline([('scaler5',StandardScaler()),

                      ('pca5',PCA(n_components=2)),

                      ('xgb_classifier',XGBClassifier(random_state=0))])



pipeline_lgb=Pipeline([('scaler6',StandardScaler()),

                      ('pca6',PCA(n_components=2)),

                      ('lgb_classifier',LGBMClassifier(random_state=0))])
pipelines=[pipeline_lr,pipeline_svc,pipeline_dt,pipeline_rf,pipeline_xgb,pipeline_lgb]
best_accuracy=0.0

best_classifier=0

best_pipeline=""
pipe_dic={0:'logistic regression',1:'SVC',2:'Decision Tree',3:'Random Forest',4:'XGBoost',5:'LGBoost'}



for pipe in pipelines:

    pipe.fit(X_train,y_train)
for i,model in enumerate(pipelines):

    print("{} Test Accuracy: {}".format(pipe_dic[i],model.score(X_test,y_test)))
for i,model in enumerate(pipelines):

    if model.score(X_test,y_test)>best_accuracy:

        best_accuracy=model.score(X_test,y_test)

        best_pipeline=model

        best_classifier=i

print('Classifier with best accuracy score:{}'.format(pipe_dic[best_classifier]))
from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()



log_reg.fit(X_train, y_train)
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,roc_auc_score
# make class predictions for the testing set

y_pred_test = log_reg.predict(X_test)
y_pred_train = log_reg.predict(X_train)
report_train = classification_report(y_train, y_pred_train)

print('Train Report')

print('\n')

print(report_train)



print('\n')



report_test = classification_report(y_test, y_pred_test)

print('Test Report')

print('\n')

print(report_test)
from lightgbm import LGBMClassifier

model = LGBMClassifier(max_depth=5,

                       learning_rate=0.4, 

                       n_estimators=100)
model.fit(X_train,y_train,

          eval_set=[(X_train,y_train),(X_test, y_test.values)],

          eval_metric='auc',

          early_stopping_rounds=200,

          verbose=500)
pred_y = model.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, pred_y))

confusion_matrix(y_test, pred_y>0.5)
fpr, tpr, thresholds = roc_curve(y_test, pred_y)

plt.plot(fpr,tpr)
