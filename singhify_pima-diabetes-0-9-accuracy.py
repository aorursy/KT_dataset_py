import pandas as pd
# pd.set_option("max_rows", None)
df= pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df

### value in the Glucose is 0 so we have to replace it 

mean_of_glucose = df.Glucose.mean()

df['Glucose'] = df['Glucose'].apply(lambda x: mean_of_glucose if x ==0 else x )
mean_of_glucose

### value in the Insulin is 0 so we have to replace it 

mean_of_insulin = df.Insulin.mean()

df['Insulin'] = df['Insulin'].apply(lambda x: mean_of_insulin if x ==0 else x )
mean_of_insulin

### value in the SkinThickness is 0 so we have to replace it 

mean_of_thickness = df.SkinThickness.mean()

df['SkinThickness'] = df['SkinThickness'].apply(lambda x: mean_of_thickness if x ==0 else x )
mean_of_insulin

df

## Balancing the data set 

df.Outcome.value_counts()

from sklearn.utils import resample

# Separate majority and minority classes
df_majority = df[df.Outcome==0]
df_minority = df[df.Outcome==1]


# Upsample minority class
df_minority_upsampled = resample(df_minority,replace=True,n_samples=500,random_state=123)

df = pd.concat([df_majority, df_minority_upsampled])

df.Outcome.value_counts()

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
plt.figure(figsize=(10,7))

sns.heatmap(df.drop('Outcome',axis=1).corr()).plot()

#splitting 
X= df.drop('Outcome',axis=1)
y = df['Outcome']

X

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size= 0.2,random_state=12)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10)
clf_rf = rf.fit(Xtrain,ytrain)
pred_y = clf_rf.predict(Xtest)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print('Cofusion matrix Random Forest:\n ',confusion_matrix(ytest,pred_y))
print('Accuracy Score Random Forest: ',accuracy_score(ytest,pred_y))
print('Classification report Random Forest: ', classification_report(ytest,pred_y))

import numpy as np

n_estimators = [int(x) for x in np.linspace(10,200,15)]
max_features = ['sqrt','auto','log2']
max_depth = [int(x) for x in np.linspace(10,1000,15)]
min_sample_split = [int(x) for x in range(1,20,2)]
min_sample_leaf = [int(x) for x in range(1,20,2)]
criterion = ['entropy','gini']
random_grid = dict(n_estimators=n_estimators,criterion=criterion,max_features = max_features,max_depth=max_depth,min_samples_split=min_sample_split,min_samples_leaf=min_sample_leaf)

print(random_grid)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
clf_random = RandomizedSearchCV(rf,random_grid,n_iter=100,cv=3,verbose=2,n_jobs=-1,random_state=10)
clf_random.fit(Xtrain,ytrain)

clf_random.best_params_,clf_random

best_random_grid = clf_random.best_estimator_
best_random_grid

pred_y_new = best_random_grid.predict(Xtest)

print('Cofusion matrix RandomSearchCV:\n ',confusion_matrix(ytest,pred_y_new))
print('Accuracy Score RandomSearchCV: ',accuracy_score(ytest,pred_y_new))
print('Classification report RandomSearchCV: ', classification_report(ytest,pred_y_new))


n_estimators = [186]
max_features = ['sqrt']
max_depth = [263,300,363,400,563]
min_sample_split = [4,5,6,7]
min_sample_leaf = [1,2,3,4]
criterion = ['entropy']
grid_grid = dict(n_estimators=n_estimators,criterion=criterion,max_features = max_features,max_depth=max_depth,min_samples_split=min_sample_split,min_samples_leaf=min_sample_leaf)

print(grid_grid)


from sklearn.model_selection import GridSearchCV

rf=RandomForestClassifier()
clf_grid = GridSearchCV(rf,grid_grid,cv=3,verbose=2,n_jobs=-1)
clf_grid.fit(Xtrain,ytrain)

best_grid = clf_grid.best_estimator_
best_grid

pred_y_new = best_grid.predict(Xtest)

print('Cofusion matrix GridSearch:\n ',confusion_matrix(ytest,pred_y_new))
print('Accuracy Score GridSearch: ',accuracy_score(ytest,pred_y_new))
print('Classification report GridSearch:', classification_report(ytest,pred_y_new))



