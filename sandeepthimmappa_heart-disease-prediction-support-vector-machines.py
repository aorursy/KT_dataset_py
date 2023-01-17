import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.rcParams['figure.figsize'] = (12,5)
plt.style.use('seaborn-darkgrid')
plt.rcParams['font.size']=12
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
sns.pairplot(df, diag_kind='hist')
sns.heatmap(df.corr(), annot=True, cbar=True)
sns.boxplot('target', 'age', data=df)
df[df['target']==1]['age'].plot(kind='hist', bins=50)
## Importing library to split data
from sklearn.model_selection import train_test_split
## Splitting data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
## Importing classifier
from sklearn.svm import SVC
## Creatng object and fitting the train data
svc = SVC()
svc.fit(X_train, y_train)
## Prediction
pred = svc.predict(X_test)
## Importing classification report and confusion matrix to view accuracy

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))
## Using gridsearchcv
from sklearn.model_selection import GridSearchCV
param = {'C': [0.1, 1, 10, 100], 'gamma':[1, 0.1, 0.001, 0.001], 'kernel': [ 'rbf']}

grid = GridSearchCV(SVC(), param_grid=param, cv=10)
grid.fit(X_train, y_train)
grid.best_params_
grid_pred = grid.predict(X_test)
print(classification_report(y_test, grid_pred))
df.head()
## Renaming columns

df.columns = ['age', 'sex', 'chest_pain', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'ecg', 'max_heart_rate',
       'Exercise_induced_angina', 'st_depression', 'st_slope', 'major_vessels', 'thalassemia', 'target']
df.head(2)
## Checking the features highly correlated with 'target' variable
df.corr()['target'][abs(df.corr()['target'])>0.1].sort_values(ascending=False)
##Change the values of the categorical variables
df.loc[df['sex']==1, 'sex']='male'
df.loc[df['sex']==0, 'sex']='female'

df.loc[df['chest_pain'] == 1, 'chest_pain'] = 'typical angina'
df.loc[df['chest_pain'] == 2, 'chest_pain'] = 'atypical angina'
df.loc[df['chest_pain'] == 3, 'chest_pain'] = 'non-anginal pain'
df.loc[df['chest_pain'] == 4, 'chest_pain'] = 'asymptomatic'

df.loc[df['fasting_blood_sugar'] == 0, 'fasting_blood_sugar'] = 'lower than 120mg/ml'
df.loc[df['fasting_blood_sugar'] == 1, 'fasting_blood_sugar'] = 'greater than 120mg/ml'

df.loc[df['ecg'] == 0, 'ecg'] = 'normal'
df.loc[df['ecg'] == 1, 'ecg'] = 'ST-T wave abnormality'
df.loc[df['ecg'] == 2, 'ecg'] = 'left ventricular hypertrophy'

df.loc[df['Exercise_induced_angina'] == 0, 'Exercise_induced_angina'] = 'no'
df.loc[df['Exercise_induced_angina'] == 1, 'Exercise_induced_angina'] = 'yes'

df.loc[df['st_slope']==1, 'st_slope'] = 'upsloping'
df.loc[df['st_slope']==2, 'st_slope'] = 'flat'
df.loc[df['st_slope']==3, 'st_slope'] = 'downsloping'

df.loc[df['thalassemia'] == 1, 'thalassemia'] = 'normal'
df.loc[df['thalassemia'] == 2, 'thalassemia'] = 'fixed defect'
df.loc[df['thalassemia'] == 3, 'thalassemia'] = 'reversable defect'
## Simple function for value counts
def value_count(df, feature):
    """
    Function for value counts
    """
    count = df[feature].value_counts()
    return count
value_count(df, 'chest_pain')
df.loc[df['chest_pain'] == 0, 'chest_pain'] = 'asymptomatic'
value_count(df, 'st_slope')
df.loc[df['st_slope']==0, 'st_slope'] = 'downsloping'
value_count(df, 'thalassemia')
df[df['thalassemia']==0]
## thalassemia = 0 for 2 rows, will try finding best value for thalassemia using EDA.
sns.boxplot('thalassemia', 'age', hue='sex', data=df)
df[(df['sex']=='female') & (df['age']>=50) & (df['age']<=60)]
## Based on observation from above data, decided to make row 48 as 'fixed defect'
df.loc[48, 'thalassemia'] = 'fixed defect'
df[(df['sex']=='male') & (df['age']>=45) & (df['age']<=60) & (df['chest_pain']=='asymptomatic')&(df['fasting_blood_sugar']=='greater than 120mg/ml')]
## Based on observation from above data, decided to make row 281 as 'reversable defect'
df.loc[281, 'thalassemia']= 'reversable defect'
sns.violinplot('target', 'age', data=df)
## Grouping age, intention to create categorical values 
def age_group(col):
    """
    Function to group age
    """
    
    if col<=30:
        return 'young'
    elif col<=40:
        return 'adult'
    elif col<=55:
        return 'middle aged'
    elif col<=65:
        return 'senior citizen'
    else:
        return 'old'
df['age_group'] = df['age'].apply(age_group)
df.head(3)
sns.violinplot('target', 'resting_blood_pressure', data=df)
## Grouping blood pressure

def blood_pressure_group(col):
    """
    Function to group blood pressure
    """
    
    if col<=120:
        return 'normal'
    elif col<=129:
        return 'elevated'
    elif col<=139:
        return 'high bp stg1'
    elif col<=180:
        return 'high bp stg2'
    else:
        return 'hipertensive crisis'
df['blood_pressure_group'] = df['resting_blood_pressure'].apply(blood_pressure_group)
sns.boxplot('target', 'cholesterol', data = df)
## grouping cholesterol

def cholesterol_group(col):
    """
    Function to group cholestrol values
    """
    if col<=200:
        return 'desirable'
    elif col<=240:
        return 'moderate risk'
    else:
        return 'high risk'
df['cholesterol_group'] = df['cholesterol'].apply(cholesterol_group)
df.head()
## We will stop with feature engineering here, taking copy of df

df_final = df.copy()
## Creating dummy variables
df_final = pd.get_dummies(df_final, drop_first=True)
df_final
X = df_final.drop('target', axis=1)
y = df_final['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler

std_scale = StandardScaler()
X_train = std_scale.fit_transform(X_train)
X_test = std_scale.fit_transform(X_test)
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.001], 'kernel':['rbf']}
grid2 = GridSearchCV(SVC(), param_grid= param_grid, cv=10, n_jobs=-1)
grid2.fit(X_train, y_train)
pred2 = grid2.predict(X_test)
print(classification_report(y_test, pred2))
print('\n')
print(confusion_matrix(y_test, pred2))
print(grid2.best_params_)
print(grid2.best_score_)
from sklearn.metrics import roc_curve, auc
fpr, tpr, treshold = roc_curve(y_test, pred2)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label="ROC,auc="+str(auc(fpr, tpr)))
ax.plot([0,1], [0,1], ls='--', lw=2, color='black')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False positive rate')
plt.ylabel('True psoitive rate')
plt.title('ROC curve')
plt.legend(loc=4)
print('Sensitivity: ', 45/(45+5))
print('Specificity: ', 34/(34+7))
