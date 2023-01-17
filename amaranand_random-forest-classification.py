import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

data = pd.read_csv('../input/Autism_Data.arff')
data.head(10)

data.replace("?",np.nan,inplace=True)

data=data.drop('used_app_before',axis=1)

data['age']=data['age'].apply(lambda x:float(x))
data.head(10)

data['age'].max()

data_p=data
data_p.dropna(inplace=True)
data_t=data_p[data_p['age']!=383]
data_t['age'].mean()
data.loc[data.age==383,'age']=30

data['age']=data['age'].fillna(30)
data=data.drop('ethnicity',axis=1)
data.drop(['contry_of_res','age_desc','relation'],axis=1,inplace=True)

sex=pd.get_dummies(data['gender'],drop_first=True)
jaund=pd.get_dummies(data['jundice'],drop_first=True,prefix="Had_jaundice")
rel_autism=pd.get_dummies(data['austim'],drop_first=True,prefix="Rel_had")
detected=pd.get_dummies(data['Class/ASD'],drop_first=True,prefix="Detected")

data=data.drop(['gender','jundice','austim','Class/ASD'],axis=1)
dataset=pd.concat([data,sex,jaund,rel_autism,detected],axis=1)

dataset.head()

#X=dataset.iloc[:,:-1].values
#Y=dataset.iloc[:,[15]].values

X=dataset[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'result', 'm',
       'Had_jaundice_yes', 'Rel_had_yes']]
Y=dataset['Detected_YES']

X
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators=5, criterion='entropy')
classifier.fit(X_train,Y_train)
pred=classifier.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_true=Y_test,y_pred=pred))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,pred)
cm



# Any results you write to the current directory are saved as output.
