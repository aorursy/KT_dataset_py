import pandas as pd # functionality for holding data in row/column format, many useful utility functions, works very well with sklearn
import numpy as np # array utility
import re # regular expressions, as we are dealing in part with text
import matplotlib # y'know
%matplotlib inline

# sklearn functions used here
from sklearn.model_selection import train_test_split # not actually used below, but very useful in general!
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix,accuracy_score
train_data = pd.read_csv('../input/train.csv')
print('Training data:\n1 person per row, 11 variables we can use to predict, 1 outcome label (Survived).\n',train_data.shape)
test_data = pd.read_csv('../input/train.csv')
print('Test data:\nSame as training, but without label column.\n', test_data.shape)
train_data.head()
def get_title(x):
    'Take a single string and if includes one of a number of titles (e.g. Dr. Mrs., etc) and then returns the title'
    m = re.search(r'(Mr.|Ms.|Miss|Master|Reverend)+', x)
    out=m.group(0) if m else 'NoTitle'
    return out

def get_X(data):
    'This function turns the input dataframe into a dummy coded array'
    X = pd.get_dummies(data['Embarked'].apply(lambda x: 'embarked' + str(x) if not pd.isnull(x) else 'embarkednone'))
    X = X.join(pd.get_dummies(data['Cabin'].apply(lambda x: 'cabinsec' + re.sub('[^A-Z]','',x)[0] if not pd.isnull(x) else 'cabinsec' + 'None')))
    X = X.join(pd.get_dummies(data['Cabin'].apply(lambda x: 'cabinnum' + re.sub('[A-Z]','',x).split(' ')[0] if not pd.isnull(x) else 'cabinnumNone'))) #X.drop('cabinnumNone',axis=1,inplace=True)
    X = X.join(pd.get_dummies(data['Pclass'].apply(lambda x: 'pclass' + str(x))))
    X = X.join(data['Sex'].apply(lambda x: 0 if x=='male' else 1))
    X = X.join(data['Age'].apply(lambda x: x if not pd.isnull(x) else train_data['Age'].mean()))
    X = X.join(data['Fare'].apply(lambda x: x if not pd.isnull(x) else train_data['Fare'].mean()))
    X = X.join(pd.get_dummies(data['Parch'].apply(lambda x: 'Parch' + str(x))))
    X = X.join(pd.get_dummies(data['SibSp'].apply(lambda x: 'SibSp' + str(x)))) # if x<4 else 'SibSp' + '4+')))
    X = X.join(data['Cabin'].apply(lambda x: 1 if not pd.isnull(x) and len(re.sub('[^A-Z]','',x))>1 else 0))
    X = X.join(pd.get_dummies(data['Name'].apply(lambda x: get_title(x))))# if x else 'NoTitle')))
    #X = X.join(data['PassengerId'])
    return X

data=get_X(pd.concat([train_data, test_data],ignore_index=True)) # I join train and test set so that the dummy coding takes into account all existing classes      
y_train = train_data['Survived']
X_train = data.iloc[:891,:] # splitting back into train/test sets
X_test = data.iloc[891:,:]
X_train.head()
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)
parameters = {'C':[0.001,0.1,1,10,100,1000],'class_weight':['balanced']}  # the class_weight option is quite important as we have many more dead than living people in the dataset
svc = LogisticRegression()
clf = GridSearchCV(svc, parameters,cv=10)
clf.fit(X_train, y_train)
print("Best score: %0.3f" % clf.best_score_)
print("Best parameters set:")
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

predictions=pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':clf.predict(X_test)})
print(predictions.head())
#predictions.to_csv('predictions_logistic.csv',index=False)

# Using the best model from above we can print out the 10 most important features and their weights
best = LogisticRegression(**clf.best_estimator_.get_params())
best.fit(X_train, y_train)
features = dict((zip(X_train.columns,best.coef_[0])))
[(i,features[i]) for i in sorted(features,key=features.get,reverse=True)[:10]]
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=2, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=0, verbose=0, warm_start=False)
parameters = {'max_depth':[2,4,10,None], 'min_samples_leaf':[1, 5, 10], 'max_features':('sqrt','log2')}
svc = RandomForestClassifier(class_weight='balanced')
clf = GridSearchCV(svc, parameters,cv=10)
clf.fit(X_train, y_train)
print("Best score: %0.3f" % clf.best_score_)
print("Best parameters set:")
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
predictions=pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':clf.predict(X_test)})
print(predictions.head())
#predictions.to_csv('predictions_random_forest.csv',index=False)

# Using the best model from above we can print out the most important features and their weights
best = RandomForestClassifier(**clf.best_estimator_.get_params())
best.fit(X_train, y_train)
features = dict((zip(X_train.columns,best.feature_importances_)))
[(i,features[i]) for i in sorted(features,key=features.get,reverse=True)[:10]]

parameters = {'C':[0.001,0.1,1,10,100],'gamma':[0.00001, 0.001, 0.01, 0.1], 'kernel': ['poly', 'rbf'],'class_weight':['balanced']}
svc = SVC()
clf = GridSearchCV(svc, parameters,cv=5)
clf.fit(X_train_scale, y_train)
print("Best score: %0.3f" % clf.best_score_)
print("Best parameters set:")
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
predictions=pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':clf.predict(X_test_scale)})
print(predictions.head())
#predictions.to_csv('predictions_SVM.csv',index=False)
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.models import load_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
def get_weight_dict(data):
    d=dict()
    ucl=np.unique(np.asarray(data))
    weight=class_weight.compute_class_weight('balanced', ucl, np.asarray(data))
    count=0
    for i in ucl:
        d[i] = weight[count]
        count+=1
    return d
model = Sequential()
model.add(Dense(10,input_dim=X_train_scale.shape[1],activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
opt=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])

# get class weights
weights=get_weight_dict(y_train)
# model checkpoints - this is a list of functions that gets called after every training epoch
callback=[ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)] #_'+time.strftime("%Y%m%d-%H%M%S")+'

# fit (uncomment)
#hist = model.fit(X_train_scale, y_train, epochs=1000, verbose=1, class_weight=weights,shuffle=True)
#predictions=pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':[0 if n<0.5 else 1 for n in model.predict(X_test_scale)]})
#print(predictions.head())
#predictions.to_csv('predictions_nn.csv',index=False)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_train_pruned=sel.fit_transform(X_train)
X_test_pruned=sel.transform(X_test)

# we have to standardize again after pruning
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train_pruned)
X_test_scale = scaler.transform(X_test_pruned)
parameters = {'C':[0.001,0.1,1,10,100],'gamma':[0.00001, 0.001, 0.01, 0.1], 'kernel': ['poly', 'rbf'],'class_weight':['balanced']}
svc = SVC()
clf = GridSearchCV(svc, parameters,cv=5)
clf.fit(X_train_scale, y_train)
print("Best score: %0.3f" % clf.best_score_)
print("Best parameters set:")
best_parameters = clf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
predictions=pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':clf.predict(X_test_scale)})
print(predictions.head())
#predictions.to_csv('predictions_SVM_pruned.csv',index=False)
X_train_scale.shape