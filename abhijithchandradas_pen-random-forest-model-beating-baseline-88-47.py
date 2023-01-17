import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')
#Read Data
df=pd.read_csv("../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")
#New Column Ctry
df['Ctry']=df.Country.apply(lambda x: 'Estonia' if x=='Estonia' 
                            else ('Sweden' if x=='Sweden' else 'Others'))

# Delete PassengerId, Country, Firstname and Lastname as they are not required for further analysis
df=df[['Sex', 'Age','Category', 'Survived', 'Ctry']]
#without ctry
#df=df[['Sex', 'Age','Category', 'Survived']]
df.head()
scalar=StandardScaler()
df.Age=scalar.fit_transform(np.array(df.Age).reshape(-1,1))

df_dummy=pd.get_dummies(df[['Sex','Category','Ctry']], drop_first=True)
#without ctry
#df_dummy=pd.get_dummies(df[['Sex','Category']], drop_first=True)
df=pd.concat([df,df_dummy], axis=1)

df.drop(['Sex','Category','Ctry'], axis=1, inplace=True)
#without ctry
#df.drop(['Sex','Category'], axis=1, inplace=True)
df.head()
X=df.drop('Survived', axis=1)
y=df.Survived
model_rf=RandomForestClassifier(class_weight='balanced')
model_rf.fit(X,y)
y_pred_rf=model_rf.predict(X)
y_proba_rf=model_rf.predict_proba(X)
y_proba_rf=[p[1] for p in y_proba_rf]

print("Score : ", accuracy_score(y,y_pred_rf))
print("Confusion Matrix : \n", confusion_matrix(y, y_pred_rf))
print("ROC AUC Score : " , roc_auc_score(y,y_proba_rf))
print(classification_report(y,y_pred_rf))