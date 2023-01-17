# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report, roc_curve, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ds = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
ds.head()
ds.drop('id',axis = 1, inplace = True)
ds.info()
ds['Response'].value_counts(normalize = True).mul(100) #dataset is highly imbalanced
sns.countplot(x = ds['Gender'], hue = ds['Response'], data = ds)
sns.countplot(x = ds['Previously_Insured'], hue = ds['Response'], data = ds)
sns.countplot(x = ds['Driving_License'], hue = ds['Response'], data = ds)
f = sns.FacetGrid(ds, hue = 'Response', aspect = 5)
f.map(sns.kdeplot, "Age", shade = True)
f.add_legend()
ds['Vehicle_Age'].value_counts().plot.pie(y = ds['Response'], autopct="%0.1f%%")
sns.catplot(x = 'Gender', hue = 'Response', col = 'Vehicle_Damage', data = ds, kind = 'count' )
sns.catplot(x = 'Gender', hue = 'Response', col = 'Vehicle_Age', data = ds, kind = 'count' )
sns.catplot(x = 'Gender', hue = 'Response', col = 'Driving_License', data = ds, kind = 'count' )
sns.catplot(x = 'Gender', hue = 'Response', col = 'Previously_Insured', data = ds, kind = 'count' )
f = sns.FacetGrid(ds, hue = 'Response', aspect = 5)
f.map(sns.kdeplot, "Annual_Premium", shade = True)
f.add_legend()
f = sns.FacetGrid(ds, hue = 'Response', aspect = 5)
f.map(sns.kdeplot, "Vintage", shade = True)
f.add_legend()
f = sns.FacetGrid(ds, hue = 'Gender', aspect = 5)
f.map(sns.kdeplot, "Age", shade = True)
f.add_legend()
plt.figure(figsize = (20,20))
sns.heatmap(ds.corr(), annot = True)
## Feature Scaling and Feature Selection.
ds1 = ds.copy()
ds1['Driving_License'] = ds1['Driving_License'].astype(str)
ds1['Previously_Insured'] = ds1['Previously_Insured'].astype(str)
ds1.info()
ds1.head()
skr = ExtraTreesClassifier() # selecting only the required features based on their score.
score = skr.fit(ds1.drop(['Gender','Vehicle_Age', 'Response', 'Vehicle_Damage'], axis = 1), ds1.iloc[:,-1])
co = ['Gender','Vehicle_Age', 'Response', 'Vehicle_Damage']
columns = [x for x in ds1.columns if x not in co] # creating a list which only consist of continous features.
columns
ser = pd.Series(score.feature_importances_, index = columns) # converting the scores into series
ser.nlargest(10).plot(kind = 'barh') # plotting the scores as we can see that features like Driving License and 
# Ploicy Sales Channel have least scores.
ds1.head()
ds1.drop(['Driving_License','Previously_Insured'], axis  = 1, inplace = True) # removing unwanted features.
va = {
    '> 2 Years':2,
    '1-2 Year': 1.5,
    '< 1 Year': 1
}
ds1['Vehicle_Age'] = ds1['Vehicle_Age'].map(va) # label encoding vehicle age.
le = LabelEncoder()
sc = StandardScaler()
ds1.head()
ds1['Gender'] = le.fit_transform(ds1['Gender'])
ds1['Vehicle_Damage'] = le.fit_transform(ds1['Vehicle_Damage']) 
# encoding categorical features.
## Model Building.
ds1.head()
x = ds1.iloc[:,:8] # seperating data into x and y.
y = ds1.iloc[:,-1]
x = sc.fit_transform(x) # scaling the features.
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size = 0.2) # splitting the data into training and testing.
sm = SMOTE(random_state=42) # oversampling.
X_sm, y_sm = sm.fit_sample(x, y)
#print(X_sm.shape),print(y_sm.shape)
print('before oversampling',y_tr.value_counts()) #before over sampling.
print('after oversampling',y_sm.value_counts()) # after over sampling.
mod = RandomForestClassifier()
mod.fit(X_sm,y_sm) # training.
y_hat = mod.predict(x_te) # predicting.
accuracy_score(y_te, y_hat) # checking the accuracy which is very high.
sns.heatmap(confusion_matrix(y_te, y_hat), annot = True) # looking at the confusion matrix.
print(classification_report(y_te, y_hat)) # checking other metrics for better idea.
roc_auc_score(y_te, y_hat) # checking the roc auc score.
fpr, tpr, _ = roc_curve(y_te, y_hat) # plotting the auc curve.

plt.title('ROC curve')
plt.xlabel('FPR ')
plt.ylabel('TPR ')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()
## For Test Data Sumbission.
te = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
te.head()
te['Gender'] = le.fit_transform(te['Gender'])
te['Vehicle_Damage'] = le.fit_transform(te['Vehicle_Damage'])
te['Vehicle_Age'] = te['Vehicle_Age'].map(va)
te.head()
te.drop(['Driving_License','Previously_Insured'], axis = 1, inplace = True)
x1 = te.iloc[:,1:]
x1
x1 = sc.fit_transform(x1)
final_op = mod.predict(x1)
df = pd.DataFrame()
df['id'] = te['id'] 
df['response'] = final_op
df.to_csv('submission.csv',index = False)
