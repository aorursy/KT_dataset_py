# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Reading the data from file
data=pd.read_csv('../input/heart.csv')

data.head(5)
data.shape  # Shape of data
data.describe(include = 'all')
data.info()
df = data.copy()
df.columns
# changing the column to its full name
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
       'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
df.head() # Visualising the firs't five row of data
df.target = df.target.replace({0:'No Heart Disease', 1:'Heart Disease'})
df.sex = df.sex.replace({0:'female', 1:'male'})
df.chest_pain_type = df.chest_pain_type.replace({1:'agina pectoris', 2:'atypical agina', 3:'non-anginal pain', 4:'SMI', 0:'absent'})
df.st_slope = df.st_slope.replace({1:'upsloping', 2:'horizontal', 3:'downsloping', 0:'absent'})
df.fasting_blood_sugar = df.fasting_blood_sugar.replace({0:'lower than 120mg/ml', 1:'greater than 120mg/ml'})
df.exercises_angina = df.exercise_angina.replace({0:'no', 1:'yes'})
df.thalassemia = df.thalassemia.replace({1:'normal', 2:'fixed defect', 3:'reversable defect', 0:'absent'})
df.head()
#Plotting the histogram out of it
df.hist(figsize = (20,20))
plt.show()
# Countplot foir target and sex variable
sns.countplot(x="target", hue="sex", data = df)
plt.show()
#Count plot for sex variable alone
sns.countplot(x= 'sex', data = df)
plt.show()
sns.distplot(df[df.sex=="male"].age, color="b")
sns.distplot(df[df.sex=="female"].age, color="r")
plt.xlabel("Age Distribution (blue = male, red = female)")
plt.show()
sns.pairplot(data, hue = 'target', diag_kind = 'kde',markers = ['o','+'])
plt.show()
sns.distplot(df['serum_cholesterol'])

df.columns
df.groupby(df['target']).count()
df.groupby(df['fasting_blood_sugar']).count()
pd.crosstab(df['sex'], df['target'])
pd.crosstab(df['sex'],df['chest_pain_type'])
pd.crosstab(df['sex'],df['fasting_blood_sugar'])
pd.crosstab(df['sex'], df['st_slope'])
sns.boxplot(x = df['sex'], y = df['st_depression'], data = df)
plt.show()
sns.boxplot(x = df['sex'], y = df['serum_cholesterol'], data = df)
plt.show()
sns.boxplot(x = df['target'], y = df['max_heart_rate'], data = df)
plt.show()
sns.boxplot(x = df['sex'], y = df['max_heart_rate'], data = df)
plt.show()
corr = data.corr()
corr
plt.figure(figsize = (20,10))
sns.heatmap(corr,annot = True)
plt.show()
#Getting the information about data types of data
df.info()
#Total unique values in chest pain
df['chest_pain_type'].value_counts()
#total unique values in gender
df['sex'].value_counts()
df['fasting_blood_sugar'].value_counts()
# Function to show the unique values in columns of data
def show_value_counts(heart_data):
    for column in heart_data:
        if column in ['chest_pain_type','fasting_blood_sugar','rest_ecg','exercise_angina','thalassemia','target']:
            print(heart_data[column].value_counts())
            
            
show_value_counts(df)
#Plot the realtion between features and target value
plt.style.use('dark_background')
corr1_new_train=data.corr()
plt.figure(figsize=(5,15))
sns.heatmap(corr1_new_train[['target']].sort_values(by=['target'],ascending=False),annot_kws={"size": 16},vmin=-1, cmap='PiYG', annot=True)
sns.set(font_scale=2)
data["resting_blood_pressure/serum_chelostral"] = data["trestbps"]/data["chol"]
data["max_heart_rate/depression"] = data["oldpeak"]/data["thalach"]
data["max_heart_rate/resting_blood"]=data["thalach"]/data["trestbps"]
data["max_heart_rate/serum_cholestral"]=data["thalach"]/data["chol"]
data["age/st_depression"]=data['oldpeak']/data["age"]
data["resting_blood_pressure/age"]=data["trestbps"]/data["age"]
data["serum_cholestral/age"]=data["chol"]/data["age"]
data["max_heart_rate/age"]=data["thalach"]/data["age"]
#Again ploting the relation between target and features
plt.style.use('dark_background')
corr1_new_train=data.corr()
plt.figure(figsize=(5,15))
sns.heatmap(corr1_new_train[['target']].sort_values(by=['target'],ascending=False),annot_kws={"size": 16},vmin=-1, cmap='PiYG', annot=True)
sns.set(font_scale=2)
#Drawing the pairplot between data's to get some information how they are responsible to classify the data
sns.pairplot(data, hue = 'target', diag_kind = 'kde',markers = ['o','+'])
plt.show()
#plt.bar()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["target"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
strat_train_set=strat_train_set.reset_index(drop = True)
strat_train_set.head()
strat_test_set=strat_test_set.reset_index(drop = True)
strat_test_set.head()
heart = strat_train_set.drop("target", axis=1)
heart_labels = strat_train_set["target"].copy()
from tpot import TPOTClassifier
tpot = TPOTClassifier(verbosity=2,cv = 3,early_stop = 10)
tpot.fit(heart.copy(),heart_labels.copy())
y_pred = tpot.predict(strat_test_set.drop('target',axis = 1))
# Making the Confusion Matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_pred, strat_test_set['target'].copy())
sns.heatmap(cm,annot = True)
accuracy_score(y_pred,strat_test_set['target'].copy())
print(metrics.classification_report(y_pred,strat_test_set['target'].copy()))
#3.XGBoost one of the powefull ML Algorithm
from xgboost import XGBClassifier

my_model = XGBClassifier(n_estimators=1000, learning_rate=0.01, n_job= 2,min_child_weight=5)
my_model.fit(heart.copy(),heart_labels.copy(), 
             early_stopping_rounds=10, 
             eval_set=[(heart.copy(),heart_labels.copy())], 
             verbose=1)

y_pred = my_model.predict(strat_test_set.drop('target',axis = 1))
# Making the Confusion Matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_pred, strat_test_set['target'].copy())
sns.heatmap(cm,annot = True)
