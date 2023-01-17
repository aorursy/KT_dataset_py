import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('train1.csv')
df.head()
df.info()
count_missing= df.isnull().sum()
count_missing[count_missing > 0]
unique_val = pd.concat([df.dtypes, df.nunique()],axis=1)
unique_val.columns = ["dtype","unique"]
unique_val
df.fillna(value=df.mean(),inplace=True)
df.head()
df.isnull().any().any()
df.columns
plt.figure(figsize=(20,20))
sns.heatmap(data=df.corr(),cmap='Blues',annot=True)
numerical_features = [ 'feature1','feature2','feature3','feature4', 'feature5',
       'feature6','feature7','feature8','feature9','feature10','feature11']
#categorical_features = ['type']
X_train = df[numerical_features]
y_train = df["rating"]
#typeCode = {'old':0,'new':1}
#X_train['type'] = X_train['type'].map(typeCode)

#One-hot
#X_train = pd.get_dummies(data=X_train,columns=['type'])



from sklearn.preprocessing import StandardScaler


X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.25,random_state = 0)


scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])

#X_train[numerical_features].head()


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_features='sqrt',min_samples_leaf=1,max_depth=100, min_samples_split=2,n_estimators=10000)
rf.fit(X_train, y_train)



from sklearn.model_selection import GridSearchCV
parameters=[{'bootstrap': [False],
 'max_depth': [100,150],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [ 4,6],
 'min_samples_split': [ 5, 10],
 'n_estimators': [ 3000, 4000]}]
rf_random=GridSearchCV(estimator=rf,param_grid=parameters,cv=3,n_jobs=-1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
df1=pd.read_csv('test1.csv')
df1.fillna(value=df.mean(),inplace=True)

X_test_numerical_features = ['feature1','feature2','feature3','feature4', 'feature5',
       'feature6','feature7','feature8','feature9','feature10','feature11']
#X_test_categorical_features = ['type']
X_test = df1[X_test_numerical_features]
#type_code = {'old':0,'new':1}
#X_test['type'] = X_test['type'].map(type_code)

#X_test = pd.get_dummies(data=X_test,columns=['type'])
#X_test.head()

scaler = StandardScaler()
X_test[X_test_numerical_features] = scaler.fit_transform(X_test[X_test_numerical_features])

X_test[X_test_numerical_features].head()
pred1=rf.predict(X_test)
pred1
df1['rating']=np.array(pred1)
df1.head()
#df1.round({'rating': 0})
out=df1[['id','rating']]
out=out.round({'rating': 0})
out.head()
#out.to_csv('submit_eval_lab_one.csv',index='True')
from google.colab import files
out.to_csv('submit19.csv') 
files.download('submit19.csv')