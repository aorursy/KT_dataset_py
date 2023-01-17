# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Read the file
df=pd.read_csv('../input/heart-disease-dataset/heart.csv')
df.head(5)
df.info()
df.describe()
#Number of the rows and columns
df.shape
#drop some columns(ignore some unuseful attributes )
df=df.drop(columns=['exang','ca','thal','oldpeak'],axis=1)
#Data with 10 attributes
df.head(5)
df.nunique()
df['target'].unique()
target_sex=pd.crosstab(df['target'],df['sex'])
target_sex
#people with the age more than 60 and the target is 1
df[(df['target']==1)& (df['age']>60)]
df['sex'].unique()
sex_cp=pd.crosstab(df['sex'],df['cp'])
sex_cp
#Number of the patient with thalach(max heart rate 180)
def clip_thalach(thalach):
    if thalach>180:
        thalach=180
        return thalach
df['thalach'].apply(lambda x:clip_thalach(x))[:30]
df.groupby('target').mean().sort_values('age',ascending=False)
pd.pivot_table(df,index=['sex','age'],values='target')
sns.countplot(data=df,y='target',palette='hls',hue='sex')
plt.title('number of the patient by their target')
plt.figure(figsize=(10,5))
plt.show()
sns.countplot(data=df,y='sex',hue='fbs')
plt.title('people by their age')
plt.figure(figsize=(10,5))
plt.show()
sns.swarmplot(df['age'])
sns.catplot('target','thalach',data=df,kind='box')
sns.jointplot('age','thalach',data=df,kind='kde',color='pink',hue='sex')
fig,ax=plt.subplots(figsize=(10,5))
sns.countplot(x=df['sex'],hue=df['target'],data=df,ax=ax)
plt.xlabel('Categorized by sex')
plt.ylabel('target')
plt.xticks(rotation=50)
plt.show()
fig,ax=plt.subplots(figsize=(10,5))
sns.countplot(x=df['age'],hue=df['sex'],ax=ax)
plt.xlabel('age')
plt.ylabel('amount of the max heart rate')
plt.xticks(rotation=50)
plt.show()
sns.relplot('thalach','age',data=df,kind='line',ci=None)
nums=['sex','thalach','cp','trestbps','restecg','slope']
for i in nums:
    sns.jointplot(x=df[i],y=df['target'],kind='reg')
    plt.xlabel(i)
    plt.ylabel('count')
    plt.grid()
    plt.show()
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
df.isnull().sum()
df.head()
X = df.drop(columns=['target','age','fbs','age','chol','trestbps','restecg','slope'],axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=145)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
import pickle
Pkl_Filename = "Pickle_DT_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_SVM_model = pickle.load(file)

Pickled_SVM_model
# Calculate the Score 
score = Pickled_DT_model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_SVM_model.predict(X_test)  

Ypredict
sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")
model=LogisticRegression()
model.fit(X_train,y_train)
import pickle
Pkl_Filename = "Pickle_RF_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_RF_model = pickle.load(file)

Pickled_RF_model
model = SVC()
# Calculate the Score 
score = Pickled_RF_model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_RF_model.predict(X_test)  

Ypredict
sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")
model.fit(X_train,y_train)
import pickle
Pkl_Filename = "Pickle_SVM_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_SVM_model = pickle.load(file)

Pickled_SVM_model
# Calculate the Score 
score = Pickled_SVM_model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_SVM_model.predict(X_test)  

Ypredict