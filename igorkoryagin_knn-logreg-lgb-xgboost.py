import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve

#models and tuning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMClassifier
from xgboost import plot_importance

# For tuning
from sklearn.model_selection import GridSearchCV, cross_val_score
import optuna 

# Metrics for models evaluation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
df.head()
df.info()
df.describe()
labelencoder=LabelEncoder()

for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
df.describe()
df.drop('veil-type', axis = 1, inplace=True)
rat_df = df['class'].value_counts()

# ratio is created if it's needed in models
ratio = rat_df[1] / rat_df[0]

weights_dict = {0: ratio,
               1:1}
X = df.drop('class',axis=1) 
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# sc = StandardScaler()

# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
neighbors = np.arange(4,16)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    print('Training KNN ', k)
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test)
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
knn = KNeighborsClassifier(n_neighbors=6,p=1, n_jobs=-1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('Accuracy:', accuracy_score(y_test ,y_pred))
print('F1-score:', f1_score(y_test ,y_pred))
print('Confusion matrix:')
print(confusion_matrix(y_test,y_pred))
# Find best hyperparameters (roc_auc)
random_state = 42

log_clf = LogisticRegression(random_state = random_state)
param_grid = {'class_weight' : ['auto', weights_dict, 'None'], 
                'penalty' : ['l2','l1'],  
                'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid = GridSearchCV(estimator = log_clf, param_grid = param_grid , scoring = 'roc_auc', verbose = 1, n_jobs = -1)

grid.fit(X_train,y_train)

print("Best Score:" + str(grid.best_score_))
print("Best Parameters: " + str(grid.best_params_))

best_parameters = grid.best_params_
log_reg = LogisticRegression(**best_parameters)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print('Accuracy:', accuracy_score(y_test ,y_pred))
print('F1-score:', f1_score(y_test ,y_pred))
print('Confusion matrix:')
print(confusion_matrix(y_test,y_pred))
lgb_model = LGBMClassifier(nthread=-1, silent=True)

#Fit to training data
lgb_model.fit(X_train, y_train)
#Generate Predictions
y_pred=lgb_model.predict(X_test)

accuracy = accuracy_score(y_test ,y_pred)
f1 = f1_score(y_test ,y_pred)

print('Accuracy:', accuracy)
print('F1-score:', f1)
print('Confusion matrix:')
print(confusion_matrix(y_test,y_pred))
lgb.plot_importance(lgb_model, figsize=(12,8))
lgb_fimp = pd.DataFrame(sorted(zip(lgb_model.feature_importances_,X.columns)), columns=['Value_LGB','Feature_LGB'])
lgb_fimp = lgb_fimp.sort_values(by=['Value_LGB'], ascending=False).reset_index(drop=True)
lgb_fimp
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)

accuracy = accuracy_score(y_test ,y_pred)
f1 = f1_score(y_test ,y_pred)

print('Accuracy:', accuracy)
print('F1-score:', f1)
print('Confusion matrix:')
print(confusion_matrix(y_test,y_pred))
plot_importance(gbm, max_num_features=30) # top 10 most important features
plt.show()
dct = gbm.get_booster().get_score(importance_type="gain")
xgb_fimp = pd.DataFrame(dct.items(), columns=['Feature_XGB', 'Value_XGB']).sort_values(by=['Value_XGB'], ascending=False)


xgb_fimp,lgb_fimp

d = {'XGBoost_Feature' : xgb_fimp['Feature_XGB'], 
     'XGBoost_Value' : xgb_fimp['Value_XGB'],
      'LGBM_Feature' : lgb_fimp['Feature_LGB'],
      'LGBM_Value' : lgb_fimp['Value_LGB'],
    }

features = pd.DataFrame(d)

features