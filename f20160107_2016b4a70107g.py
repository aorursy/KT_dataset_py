import numpy as np
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
##from google.colab import files
##uploaded = files.upload()

df = pd.read_csv('../input/eval-lab-1-f464-v2/train.csv')
df.head()
df.info()

count_missing= df.isnull().sum()
count_missing[count_missing > 0]
unique_values = pd.concat([df.dtypes, df.nunique()],axis=1)
unique_values.columns = ["dtype","unique"]
unique_values
df.fillna(value=df.mean(),inplace=True)
df.head()
df.isnull().any().any()
df.columns
plt.figure(figsize=(20,20))
sns.heatmap(data=df.corr(),cmap='coolwarm',annot=True)
cat_features = ['type']
num_features = [ 'feature1','feature2','feature3','feature4', 'feature5',
       'feature6','feature7','feature8','feature9','feature10','feature11']

X_train = df[num_features+cat_features]
y_train = df["rating"]
typeCode = {'old':0,'new':1}
X_train['type'] = X_train['type'].map(typeCode)




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.20,random_state = 0)


scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])

#X_train[num_features].head()


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_features='sqrt',bootstrap=True,min_impurity_decrease=0.0, min_impurity_split=None,warm_start=False,min_samples_leaf=1,max_depth=100, min_samples_split=2,n_estimators=11000)
rf.fit(X_train, y_train)




##tuning
'''from sklearn.model_selection import GridSearchCV
parameters=[{'bootstrap': [False],
 'max_depth': [100,150,300],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [ 4,6,9],
 'min_samples_split': [ 5, 10,50],
 'n_estimators': [ 3000, 4000,1000]}]
rf_random=GridSearchCV(estimator=rf,param_grid=parameters,cv=3,n_jobs=-1)
rf_random.fit(X_train,y_train)'''

#rf_random.best_params_
##hyperparameter tuning
'''from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

#TODO
clf = RandomForestClassifier()        #Initialize the classifier object

parameters = {'n_estimators':range(10,80)}    #Dictionary of parameters

scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train

best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions
optimized_predictions = best_clf.predict(X_val)        #Same, but use the best estimator

acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model
acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model

print("Accuracy score on unoptimized model:{}".format(acc_unop))
print("Accuracy score on optimized model:{}".format(acc_op))'''



##from google.colab import files
##uploaded = files.upload()
df1=pd.read_csv('../input/eval-lab-1-f464-v2/test.csv')
df1.fillna(value=df.mean(),inplace=True)
##df1=pd.read_csv('test1.csv')
##df1.fillna(value=df.mean(),inplace=True)
X_test_numerical_features = ['feature1','feature2','feature3','feature4', 'feature5',
       'feature6','feature7','feature8','feature9','feature10','feature11']
X_test_categorical_features = ['type']
X_test = df1[X_test_numerical_features+X_test_categorical_features]
type_code = {'old':0,'new':1}
X_test['type'] = X_test['type'].map(type_code)

scaler = StandardScaler()
X_test[X_test_numerical_features] = scaler.fit_transform(X_test[X_test_numerical_features])

X_test[X_test_numerical_features].head()

pred1=rf.predict(X_test)
pred1
df1['rating']=np.array(pred1)
df1.head()

out=df1[['id','rating']]
out=out.round({'rating': 0})
out.head()

#from google.colab import files
#out=pd.DataFrame(out)

out.to_csv('myfile.csv') 
#files.download('myfile.csv')