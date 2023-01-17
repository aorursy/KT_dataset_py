# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
train=pd.read_csv('/kaggle/input/train.csv')
test=pd.read_csv('/kaggle/input/test.csv')
print(train.shape[0],train.shape[1])
print(test.shape[0],test.shape[1])
train.head()
test.head()
sns.heatmap(train.corr(),annot=True)
train.isna().sum()
meanvalue=train['Number_Weeks_Used'].mean()
train['Number_Weeks_Used']=train['Number_Weeks_Used'].fillna(meanvalue)
train.isna().sum()
test.isna().sum()
meanvalue=test['Number_Weeks_Used'].mean()
test['Number_Weeks_Used']=test['Number_Weeks_Used'].fillna(meanvalue)
test.isna().sum()
train.Crop_Damage.value_counts().plot.pie(autopct="%0.2f%%")
train.Crop_Damage.value_counts()
plt.title('Estimated_Insects_Count vs Crop_Damage')
sns.boxplot(x='Crop_Damage',y='Estimated_Insects_Count',data=train)
plt.figure(figsize=(12,8))
plt.title('Estimated Insects count vs crop damage')
sns.distplot(train['Estimated_Insects_Count'][train['Crop_Damage']==0],bins=30,kde=True,hist=False,kde_kws={"color": "red", "label": "0-Alive"})
sns.distplot(train['Estimated_Insects_Count'][train['Crop_Damage']==1],bins=30,kde=True,hist=False,kde_kws={"color": "green", "label": "1-Damage due to other reasons"})
sns.distplot(train['Estimated_Insects_Count'][train['Crop_Damage']==2],bins=30,kde=True,hist=False,kde_kws={"color": "blue", "label": "2-Damage due to pesticides"})
for i,row in train.iterrows():
    if(train['Estimated_Insects_Count'][i]<=1100):
        train['Estimated_Insects_Count'][i]=0;
    elif (train['Estimated_Insects_Count'][i]>1100 and train['Estimated_Insects_Count'][i]<=2250):
        train['Estimated_Insects_Count'][i]=1;
    else:
        train['Estimated_Insects_Count'][i]=2;

        
        
for i,row in test.iterrows():
    if(test['Estimated_Insects_Count'][i]<=1100):
        test['Estimated_Insects_Count'][i]=0;
    elif (test['Estimated_Insects_Count'][i]>1100 and test['Estimated_Insects_Count'][i]<=2250):
        test['Estimated_Insects_Count'][i]=1;
    else:
        test['Estimated_Insects_Count'][i]=2;

plt.title('Number of doses per week vs Crop_Damage')
sns.boxplot(x='Crop_Damage',y='Number_Doses_Week',data=train)
plt.figure(figsize=(12,8))
plt.title('Number of doses per week vs crop damage')
sns.distplot(train['Number_Doses_Week'][train['Crop_Damage']==0],bins=30,kde=True,hist=False,kde_kws={"color": "red", "label": "0-Alive"})
sns.distplot(train['Number_Doses_Week'][train['Crop_Damage']==1],bins=30,kde=True,hist=False,kde_kws={"color": "green", "label": "1-Damage due to other reasons"})
sns.distplot(train['Number_Doses_Week'][train['Crop_Damage']==2],bins=30,kde=True,hist=False,kde_kws={"color": "blue", "label": "2-Damage due to pesticides"})
plt.title('Weeks used vs Crop_Damage')
sns.boxplot(x='Crop_Damage',y='Number_Weeks_Used',data=train)
plt.figure(figsize=(12,8))
plt.title('Number of weeks used vs crop damage')
sns.distplot(train['Number_Weeks_Used'][train['Crop_Damage']==0],bins=10,kde=True,hist=False,kde_kws={"color": "red", "label": "0-Alive"})
sns.distplot(train['Number_Weeks_Used'][train['Crop_Damage']==1],bins=10,kde=True,hist=False,kde_kws={"color": "green", "label": "1-Damage due to other reasons"})
sns.distplot(train['Number_Weeks_Used'][train['Crop_Damage']==2],bins=10,kde=True,hist=False,kde_kws={"color": "blue", "label": "2-Damage due to pesticides"})
for i,row in train.iterrows():
    if(train['Number_Weeks_Used'][i]<=5):
        train['Number_Weeks_Used'][i]=0;
    elif (train['Number_Weeks_Used'][i]>5 and train['Number_Weeks_Used'][i]<=30):
        train['Number_Weeks_Used'][i]=1;
    elif(train['Number_Weeks_Used'][i] >30 and train['Number_Weeks_Used'][i] <35):
        train['Number_Weeks_Used'][i]=2;
    else:
        train['Number_Weeks_Used'][i]=3;

        
for i,row in test.iterrows():
    if(test['Number_Weeks_Used'][i]<=5):
        test['Number_Weeks_Used'][i]=0;
    elif (test['Number_Weeks_Used'][i]>5 and test['Number_Weeks_Used'][i]<=30):
        test['Number_Weeks_Used'][i]=1;
    elif(test['Number_Weeks_Used'][i] >30 and test['Number_Weeks_Used'][i] <35):
        test['Number_Weeks_Used'][i]=2;
    else:
        test['Number_Weeks_Used'][i]=3;
plt.title('Weeks Quit vs Crop_Damage')
sns.boxplot(x='Crop_Damage',y='Number_Weeks_Quit',data=train)
plt.figure(figsize=(12,8))
plt.title('Weeks Quit vs crop damage')
sns.distplot(train['Number_Weeks_Quit'][train['Crop_Damage']==0],bins=10,kde=True,hist=False,kde_kws={"color": "red", "label": "0-Alive"})
sns.distplot(train['Number_Weeks_Quit'][train['Crop_Damage']==1],bins=10,kde=True,hist=False,kde_kws={"color": "green", "label": "1-Damage due to other reasons"})
sns.distplot(train['Number_Weeks_Quit'][train['Crop_Damage']==2],bins=10,kde=True,hist=False,kde_kws={"color": "blue", "label": "2-Damage due to pesticides"})
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.title('Soil type vs crop damage')
sns.countplot(x='Crop_Damage',data=train,hue='Soil_Type')
plt.subplot(122)
plt.title('Crop type vs crop damage')
sns.countplot(x='Crop_Damage',data=train,hue='Crop_Type')
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.title('Season vs crop damage')
sns.countplot(x='Crop_Damage',data=train,hue='Season')
plt.subplot(122)
plt.title('Pesticide vs crop damage')
sns.countplot(x='Crop_Damage',data=train,hue='Pesticide_Use_Category')
train['Total_Doses']=train['Number_Doses_Week']*train['Number_Weeks_Used']
test['Total_Doses']=test['Number_Doses_Week']*test['Number_Weeks_Used']
plt.title('Total doses vs Crop_Damage')
sns.boxplot(x='Crop_Damage',y='Total_Doses',data=train)
plt.figure(figsize=(12,8))
plt.title('Total_Doses vs crop damage')
sns.distplot(train['Total_Doses'][train['Crop_Damage']==0],bins=30,kde=True,hist=False,kde_kws={"color": "red", "label": "0-Alive"})
sns.distplot(train['Total_Doses'][train['Crop_Damage']==1],bins=30,kde=True,hist=False,kde_kws={"color": "green", "label": "1-Damage due to other reasons"})
sns.distplot(train['Total_Doses'][train['Crop_Damage']==2],bins=30,kde=True,hist=False,kde_kws={"color": "blue", "label": "2-Damage due to pesticides"})
train.head()
test.head()
traindata=train.drop(['ID','Soil_Type','Number_Weeks_Quit','Crop_Damage'],axis=1)
testdata=test.drop(['ID','Soil_Type','Number_Weeks_Quit'],axis=1)
traindata.head()
testdata.head()
traindata['Crop_Type']=traindata.Crop_Type.astype(str)
traindata['Pesticide_Use_Category']=traindata.Pesticide_Use_Category.astype(str)
traindata['Season']=traindata.Season.astype(str)
traindata['Estimated_Insects_Count']=traindata.Estimated_Insects_Count.astype(str)
traindata['Number_Weeks_Used']=traindata.Number_Weeks_Used.astype(str)

testdata['Crop_Type']=testdata.Crop_Type.astype(str)
testdata['Pesticide_Use_Category']=testdata.Pesticide_Use_Category.astype(str)
testdata['Season']=testdata.Season.astype(str)
testdata['Estimated_Insects_Count']=testdata.Estimated_Insects_Count.astype(str)
testdata['Number_Weeks_Used']=testdata.Number_Weeks_Used.astype(str)
X=traindata.iloc[:,:]
X=pd.get_dummies(X,['Crop_Type','Pesticide_Used_Category','Season','Estimated_Insects_Count','Number_Weeks_Used'])

X_test=testdata.iloc[:,:]
X_test=pd.get_dummies(X_test,['Crop_Type','Pesticide_Used_Category','Season','Estimated_Insects_Count','Number_Weeks_Used'])
y=train.iloc[:,9]
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(criterion='entropy',random_state=0,max_depth=8)
classifier.fit(X,y)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score
from sklearn.model_selection import cross_val_score,GridSearchCV
accuracies=cross_val_score(estimator=classifier,X=X,y=y,cv=10)
print(accuracies.mean())
print(accuracies.std())
predictions=classifier.predict(X_test)
output = pd.DataFrame({'ID': test.ID, 'Crop_Damage': predictions})
output.to_csv('random.csv', index=False)
print("Your submission was successfully saved!")
