import sys
print('Python version: {}'.format(sys.version))

import pandas as pd
print('pandas version: {}'.format(pd.__version__))

import numpy as np

print('numpy version: {}'.format(np.__version__))

import matplotlib as mlp
import matplotlib.pyplot as plt
%matplotlib inline
print('matplotlib version: {}'.format(mlp.__version__))

import seaborn as sns
print('seaborn version: {}'.format(sns.__version__))

import os
print('\nFile list:',os.listdir('../input'))

import time
start_time = time.time()
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/train.csv')
df.head()
df.info()
df.columns.to_series().groupby(df.dtypes).groups
#show few first rows
df.head()
#Overview
df.describe()
df.isna().sum()
sns.heatmap(data=df.isna(),yticklabels=False,cmap='coolwarm',cbar=False)
#Group them by age and find the mean of each Pclass
plt.figure(figsize=(12, 5))
ax = sns.boxplot(data=df,x=df['Pclass'],y=df['Age'],palette='coolwarm') # create plot object.
medians = df.groupby(['Pclass'])['Age'].median().values #get median values
median_labels = [str(np.round(s, 2)) for s in medians] #create label from median values
pos = range(len(medians)) # get range of median values
#Loop to put value label
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], medians[tick] + 0.5, median_labels[tick], 
            horizontalalignment='center', size=13, color='r', weight='semibold')
#create function to fill age
def fill_age_na(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
df['Age'] = df[['Age','Pclass']].apply(fill_age_na,axis=1)
df['Age'].isna().sum() # no more missing value
col_to_drop = ['Cabin']
df.drop(columns=col_to_drop,axis=1,inplace=True)
df.columns # 'Cabin' is now removed.
sns.countplot(x=df['Embarked'])
#or use mode
df['Embarked'].mode()[0]
#Let's fill it
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)
#check if all are filled
df['Embarked'].isna().sum()
# create name length feature, since I think longer name may harder to call by staff and lead to death
# you may improve this by removing those initial first(remove Mr. Mrs, Ms, Dr. etc)
df['NameLength'] = df['Name'].apply(len)
df['NameLength'].hist(bins=30) #most of passenger has name length around 20-30 character
# create family size since bigger family may help each other and all survive
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 # plus 1 for passenger itself
df['FamilySize'].hist(bins=20) #most of passenger travel alone
# create feature IsAlone to see if the passenger travel alone
def IsTravelAlone(col):
    if col == 1:
        return 1
    else:
        return 0
df['IsAlone'] = df['FamilySize'].apply(IsTravelAlone)
sns.countplot(data=df,x=df['IsAlone']) # most of passenger travel alone
cols_drop = ['PassengerId','Name','Ticket']
df.drop(cols_drop, axis=1, inplace = True)
#let's see how each feature interact to each other
sns.pairplot(data=df,hue='Sex',size=1.2)
print(df.groupby(['Sex'])['Survived'].mean())
sns.countplot(x=df['Sex'],hue=df['Survived']) # total number of survived female is higher and survived mean is also higher than male
fig = plt.figure(figsize=(10,8))
sns.violinplot(x='Sex',y='Age',hue='Survived',data=df,split=True)
print(df.groupby(['Pclass'])['Survived'].mean()) # highest class has 62% survaival rate while lowest class has only 24% survival rate
sns.catplot(x='Sex',y='Fare',hue='Survived',data=df,col='Pclass',kind='swarm')
grid = sns.FacetGrid(data=df,col='Survived',size=8)
grid.map(plt.hist,'Age',bins=50)
#check family size
sns.countplot(x=df['FamilySize'])
# is there any relationship between age, fare and class
sns.jointplot(x='Age',y='Fare',data=df)
#set overall size
fig = plt.figure(figsize=(15,10))
#set total number of rows and columns
row = 5
col = 2
#set title
fig.suptitle('Various plot',fontsize=20)

#box 1
fig.add_subplot()
ax = fig.add_subplot(2,2,1)
sns.countplot(x='Sex',data=df,hue='IsAlone')
#box 2
ax = fig.add_subplot(2,2,2)
df.groupby('Pclass')['Age'].plot(kind='hist',alpha=0.5,legend=True,title='Pclass vs Age')
#box 3
ax = fig.add_subplot(2,2,3)
df.groupby('Pclass')['Fare'].plot(kind='hist',alpha=0.5,legend=True,title='Pclass vs Fare')
#box 4
ax = fig.add_subplot(2,2,4)
sns.violinplot(x='Sex',y='Age',data=df,hue='Survived',split=True)

#some more setting
plt.tight_layout(pad=4,w_pad=1,h_pad=1.5)
plt.show()
df.head() #which feature are still categorical
categorical_feature = []
#loop each column
for i in range(df.shape[1]):
    #if column datatype is object/categorical
    if df[df.columns[i]].dtype == 'object':
        categorical_feature.append(df.columns[i])
        
#show
categorical_feature
#convert categorical feature to numerical
#drop_first=True, will help avoid variable dummy trap
df = pd.get_dummies(data=df,columns=categorical_feature,drop_first=True) 
df.head()
fig = plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',linewidths=0.2)
plt.show()
from sklearn.model_selection import train_test_split
dfX = df.drop('Survived',axis=1)
dfY = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(dfX, dfY, test_size=0.20, 
                                                    random_state=0)
#I saw some kernel split into train, test and validation. Should I do that to improve the model ?
#check size of data
X_train.shape,y_train.shape,X_test.shape,y_test.shape
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) #fit scaler with training data
X_test = sc.transform(X_test) #apply scaler to test data
X_train[0,:]
X_test[0,:]
df.corr().loc['Survived']
# import library to evaluate model
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
# this part just to show why should not use LinearRegression for binary outcome
from sklearn.linear_model import LinearRegression,LogisticRegression
model_lm = LinearRegression()
model_lm.fit(X_train,y_train)
pred_lm = model_lm.predict(X_test)

# find bad output
bad_output = []
for i in pred_lm:
    if i < 0 or i > 1:
        bad_output.append(i)

bad_output # so let's use LogisticRegression
model_lg = LogisticRegression(solver='lbfgs')
model_lg.fit(X_train,y_train)
pred_lg = model_lg.predict(X_test)
pred_lg
print(classification_report(y_test,pred_lg)) #classification report

#confusion matrix
cm = confusion_matrix(y_test, pred_lg)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True,cmap='RdYlGn')
plt.title('Model: LogisticRegression \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, pred_lg)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
acc_score = [] # create list to store accuracy score
def build_train_predict(clf,X_train,y_train,X_test,strAlg,acc_score):
    '''
    1. Create model
    2. Train model
    3. Prediction
    4. Evaluate
    5. Keep score
    '''
    model = clf
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    plot_score(y_test,pred,strAlg,acc_score)
    return clf,pred
# create function to plot score for later use
def plot_score(y_test,y_pred,strAlg,lstScore):
    '''
    1. Compare prediction versus real result and plot confusion matrix
    2. Store model accuracy score to list
    '''
    lstScore.append([strAlg,accuracy_score(y_test, y_pred)])
    #print(classification_report(y_test,y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True,cmap='RdYlGn')
    plt.title('Model: {0} \nAccuracy:{1:.3f}'.format(strAlg,accuracy_score(y_test, y_pred)))
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()
model_lg,pred_lg = build_train_predict(LogisticRegression(),
                                       X_train,y_train,X_test,
                                       'LogisticRegression',acc_score)
from sklearn.neighbors import KNeighborsClassifier
model_knn,pred_knn = build_train_predict(KNeighborsClassifier(),
                                       X_train,y_train,X_test,
                                       'KNN',acc_score)
from sklearn.svm import SVC
model_svm,pred_svm = build_train_predict(SVC(),
                                       X_train,y_train,X_test,
                                       'SVM',acc_score)
from sklearn.naive_bayes import GaussianNB
model_gnb,pred_gnb = build_train_predict(GaussianNB(),
                                       X_train,y_train,X_test,
                                       'GaussianNB',acc_score)
from sklearn.naive_bayes import BernoulliNB
model_bnb,pred_bnb = build_train_predict(BernoulliNB(),
                                       X_train,y_train,X_test,
                                       'BernoulliNB',acc_score)
from sklearn.tree import DecisionTreeClassifier
model_dt,pred_dt = build_train_predict(DecisionTreeClassifier(),
                                       X_train,y_train,X_test,
                                       'DecisionTreeClassifier',acc_score)
from sklearn.ensemble import RandomForestClassifier
model_rfc,pred_rfc = build_train_predict(RandomForestClassifier(),
                                       X_train,y_train,X_test,
                                       'RandomForestClassifier',acc_score)
from sklearn.ensemble import GradientBoostingClassifier
model_gbc,pred_gbc = build_train_predict(GradientBoostingClassifier(),
                                       X_train,y_train,X_test,
                                       'GradientBoostingClassifier',acc_score)
from sklearn.ensemble import ExtraTreesClassifier
model_et,pred_et = build_train_predict(ExtraTreesClassifier(),
                                       X_train,y_train,X_test,
                                       'ExtraTreesClassifier',acc_score)
from sklearn.ensemble import AdaBoostClassifier
model_adb,pred_adb = build_train_predict(AdaBoostClassifier(),
                                       X_train,y_train,X_test,
                                       'AdaBoostClassifier',acc_score)
from xgboost import XGBClassifier
model_xgb,pred_xgb = build_train_predict(XGBClassifier(),
                                       X_train,y_train,X_test,
                                       'XGBClassifier',acc_score)
import keras
from keras.models import Sequential
from keras.layers import Dense

#get number of input node and number of neuron in hidden layer
dims = X_train.shape[1]
h_dims = int((dims+1)/2)
dims,h_dims

#create model
model_ann = Sequential() #initialize
#input
model_ann.add(Dense(units=h_dims,kernel_initializer='uniform',activation='relu',input_dim=dims))
#hidden
model_ann.add(Dense(units=h_dims,kernel_initializer='uniform',activation='relu'))
#output
model_ann.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
#compile
model_ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#train
model_ann.fit(X_train,y_train,batch_size=32,epochs=100,verbose=0)

#evaluate
pred_ann = model_ann.predict(X_test)
pred_ann = pred_ann > 0.5
plot_score(y_test,pred_ann,'ANN',acc_score)
# See the summary, which model is leading
df_acc = pd.DataFrame(acc_score,columns=['Name','TestScore']).sort_values(by=['TestScore','Name'],ascending=False)
df_acc
from sklearn.model_selection import cross_val_score

#create function to store
def cross_val_MinMaxMean(clf,X_train,y_train,fold):
    scores = cross_val_score(clf,X_train,y_train,cv=fold)
    print('Min: {} \nMax: {} \nMean: {}'.format(scores.min(),scores.max(),scores.mean()))
cross_val_MinMaxMean(LogisticRegression(),X_train,y_train,10)
cross_val_MinMaxMean(KNeighborsClassifier(),X_train,y_train,10)
cross_val_MinMaxMean(SVC(),X_train,y_train,10)
cross_val_MinMaxMean(GaussianNB(),X_train,y_train,10)
cross_val_MinMaxMean(DecisionTreeClassifier(),X_train,y_train,10)
cross_val_MinMaxMean(RandomForestClassifier(),X_train,y_train,10)
cross_val_MinMaxMean(GradientBoostingClassifier(),X_train,y_train,10)
cross_val_MinMaxMean(ExtraTreesClassifier(),X_train,y_train,10)
cross_val_MinMaxMean(AdaBoostClassifier(),X_train,y_train,10)
cross_val_MinMaxMean(XGBClassifier(),X_train,y_train,10)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold

def create_model():
    model = Sequential()
    model.add(Dense(h_dims,input_dim=dims,activation='relu'))
    model.add(Dense(h_dims,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,epochs=100,batch_size=10,verbose=0)
kfold = StratifiedKFold(n_splits=10,shuffle=True)
cross_val_MinMaxMean(model,X_train,y_train,kfold)
#import library for model improvement
from sklearn.model_selection import GridSearchCV

# function to reduce coding
def wrap_gridsearchCV(clf,X_train,y_train,X_test,param_grid,strAlg,acc_score):
    '''
    1. Create GridSearch model
    2. Train model
    3. Predict
    4. Evaluate
    5. Keep score
    '''
    model = GridSearchCV(estimator=clf,param_grid=param_grid,cv=10,
                         refit=True,verbose=0,n_jobs=-1)
    model.fit(X_train,y_train)
    print('\nBest hyper-parameter: {} \n'.format(model.best_params_))
    pred = model.predict(X_test)
    plot_score(y_test,pred,strAlg,acc_score)
    return model,pred
param_grid = {
    'C': [0.1,1, 10, 100, 1000],
    'solver': ['newton-cg','lbfgs','liblinear','sag','saga'],
}
model_grid_lg,pred_grid_lg = wrap_gridsearchCV(LogisticRegression(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'LogisticRegression GCV',acc_score)
param_grid = {
    'n_neighbors': [i for i in range(1,51)]
}
model_grid_knn,pred_grid_knn = wrap_gridsearchCV(KNeighborsClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'KNN GCV',acc_score)
param_grid = {
    'C': [0.1,1, 10, 100, 1000],
    'gamma': [1,0.1,0.01,0.001,0.0001]
}
model_grid_svm,pred_grid_svm = wrap_gridsearchCV(SVC(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'SVM GCV',acc_score)
# there is no hyper-parameter to play with
param_grid = {
    'max_depth': [None,1,2,3,4,5,7,8,9,10],
    'criterion': ['gini', 'entropy']
}
model_grid_dt,pred_grid_dt = wrap_gridsearchCV(DecisionTreeClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'DecisionTreeClassifier GCV',acc_score)
param_grid = {
    'n_estimators': [i for i in range(100,1000,100)],
    'max_depth': [i for i in range(5,10)],
    'min_samples_leaf': [2,3,4,5]
}
model_grid_rfc,pred_grid_rfc = wrap_gridsearchCV(RandomForestClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'RandomForestClassifier GCV',acc_score)
param_grid = {
    'loss': ['deviance', 'exponential'],
    'n_estimators': [i for i in range(100,1000,100)],
    'min_samples_leaf': [1,2,3,4,5]
}
model_grid_gbc,pred_grid_gbc = wrap_gridsearchCV(GradientBoostingClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'GradientBoostingClassifier GCV',acc_score)
param_grid = {
    'n_estimators': [i for i in range(100,1000,100)],
    'max_depth': [i for i in range(5,10)],
    'min_samples_leaf':[2,3,4,5]
}
model_grid_et,pred_grid_et = wrap_gridsearchCV(ExtraTreesClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'ExtraTreesClassifier GCV',acc_score)
param_grid = {
    'n_estimators': [i for i in range(100,1000,100)],
    'learning_rate' : [0.25, 0.75, 1.00]
}
model_grid_et,pred_grid_et = wrap_gridsearchCV(AdaBoostClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'AdaBoostClassifier GCV',acc_score)
param_grid = {
    'n_estimators': [i for i in range(100,1000,100)],
    'max_depth': [i for i in range(5,10)]
}
model_grid_et,pred_grid_et = wrap_gridsearchCV(XGBClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'XGBClassifier GCV',acc_score)
## later
#Take feature importance to select feature for next training
dt_fi = model_dt.feature_importances_
rfc_fi = model_rfc.feature_importances_
gbc_fi = model_gbc.feature_importances_
et_fi = model_et.feature_importances_
ada_fi = model_adb.feature_importances_
xgb_fi = model_xgb.feature_importances_

fi = [dt_fi,rfc_fi,gbc_fi,et_fi,ada_fi,xgb_fi]
model_name = ['DecisionTree','RandomForrest','GradientBoost',
        'ExtraTree','AdaBoost','XGBoost']
model_name = pd.Series(model_name)
df_fi = pd.DataFrame(fi,columns=dfX.columns)
df_fi.index = model_name
df_fi
#set overall size
fig = plt.figure(figsize=(20,10))
#set total number of rows and columns
row = 2
col = 3
#set title
fig.suptitle('Feature importance',fontsize=20)

# boxes
for index,i in enumerate(df_fi.index):
    fig.add_subplot()
    ax = fig.add_subplot(2,3,index+1)
    sns.barplot(df_fi.loc[i],df_fi.columns)
    

#some more setting
plt.tight_layout(pad=4,w_pad=1,h_pad=1.5)
plt.show()
# Final score table
pd.DataFrame(acc_score,columns=['model','score']).sort_values(by=['score','model'],
                                                              ascending=False)
#display tree graph
import graphviz
from sklearn import tree
tree_dot = tree.export_graphviz(model_dt,out_file=None, 
                                feature_names = dfX.columns, class_names = True,
                                filled = True, rounded = True)
tree_img = graphviz.Source(tree_dot) 
tree_img
print("--- %s seconds ---" % (time.time() - start_time))