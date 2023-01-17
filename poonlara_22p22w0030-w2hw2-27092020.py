# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
gender_submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train=pd.read_csv('/kaggle/input/titanic/train.csv')
print('gender_submission shape '+str(gender_submission.shape))
print('train shape '+str(train.shape))
print('test shape '+str(test.shape))
train.isnull().sum()
female_age=int(train.groupby('Sex').agg({'Age':'mean'}).iloc[0:1,0:1].values)
male_age=int(train.groupby('Sex').agg({'Age':'mean'}).iloc[1:,0:1].values)
avg_fare=train["Fare"].mean()
print('female age= '+str(female_age))
print('male age= '+str(male_age))
train.loc[(train['Sex']=='female')&(train['Age'].isnull()),'Age']=female_age
train.loc[(train['Sex']=='male')&(train['Age'].isnull()),'Age']=male_age
train.loc[(train['Cabin'].isnull()),'Cabin']="Unknown"
train.dropna(axis=0, how='any',subset=['Embarked'],inplace=True)
train.isnull().sum()
train.drop(columns=['PassengerId','Name', 'Ticket'],inplace=True)
train.isnull().sum()
train.groupby('Pclass').size()
train.groupby('Sex').size()
train.groupby('SibSp').size()
train.groupby('Fare').size()

train.groupby('Cabin').size()

train.groupby('Embarked').size()
print(gender_submission.shape)
print(train.shape)
print(test.shape)
test.isnull().sum()
test.head()
test.loc[(test['Sex']=='female')&(test['Age'].isnull()),'Age']=female_age
test.loc[(test['Sex']=='male')&(test['Age'].isnull()),'Age']=male_age
test.loc[(test['Cabin'].isnull()),'Cabin']="Unknown"
test.loc[(test['Fare'].isnull()),'Fare']=avg_fare
test.drop(columns=['PassengerId','Name', 'Ticket'],inplace=True)
gender_submission.drop(columns=['PassengerId'],inplace=True)

print(gender_submission.shape)
print(train.shape)
print(test.shape)
X_train=train.copy()
y_train=X_train['Survived'].values
X_train=X_train.drop(columns='Survived')
y_test=test
y_test=gender_submission
gender_submission.isnull().sum()
print(X_train.shape,y_train.shape)
print(test.shape,gender_submission.shape)

gender_submission=gender_submission['Survived'].values
X_train.isnull().sum()
test.isnull().sum()
New_df=X_train.append(test)
New_df.shape
New_df.isnull().sum()
New_df["Sex"] = New_df["Sex"].astype('category')
New_df["Cabin"] = New_df["Cabin"].astype('category')
New_df["Embarked"] = New_df["Embarked"].astype('category')
# Create one_hot_encoder function
def one_hot_encoder(dataframe):

  # Select category columns
  cat_df = dataframe.select_dtypes(include=['category']).columns.to_list()

  # Convert to one-hot dataframe
  one_hot_df = pd.get_dummies(dataframe, columns=cat_df, drop_first=True)
  
  return one_hot_df
New_df_dummie = one_hot_encoder(New_df)
X_train_dummie=New_df_dummie[0:889]
X_test_dummie=New_df_dummie[889:]
X_train_dummie
X_train_dummie.isnull().sum()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
param_grid = [
        {
            'criterion' : ['gini', 'entropy'],
            'splitter' :['best', 'random'],
            'max_depth' :[2,3,4,5],
            'min_samples_split' : [2,3,4,5,6,7,8,9,10],
            'min_samples_leaf' :[1,2,3,4,5,6,7,8,9,10],
            'max_features': ['auto', 'sqrt', 'log2']
        }
       ]
tree_clf_tune = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=10,
                           scoring='accuracy')
tree_clf_tune.fit(X_train_dummie,y_train)


print("Best parameters set found on development set:")
print(tree_clf_tune.best_params_)
tree_clf = DecisionTreeClassifier(criterion='gini',max_depth= 5, max_features='auto', min_samples_leaf=2, min_samples_split=4, splitter='random',random_state=30)
tree_model = tree_clf.fit(X_train_dummie, y_train)

print("Decison Tree Metric Evalaute all train,test")
print("Decision Tree : accuracy score = "+str(accuracy_score(y_test, tree_model.predict(X_test_dummie), normalize=True)*100))
print("Decision Tree : precision score = "+str(precision_score(y_test, tree_model.predict(X_test_dummie),average='macro')*100))
print("Decision Tree : recall score = "+str(recall_score(y_test, tree_model.predict(X_test_dummie),average='macro')*100))
print("Decision Tree : f1 score = "+str(f1_score(y_test, tree_model.predict(X_test_dummie),average='macro')*100))


tree_clf1 = DecisionTreeClassifier(random_state=30)
tree_model1 = tree_clf1.fit(X_train_dummie, y_train)
print("Decison Tree Metric Evalaute all train,test")
print("Decision Tree : accuracy score = "+str(accuracy_score(y_test, tree_model1.predict(X_test_dummie), normalize=True)*100))
print("Decision Tree : precision score = "+str(precision_score(y_test, tree_model1.predict(X_test_dummie),average='macro')*100))
print("Decision Tree : recall score = "+str(recall_score(y_test, tree_model1.predict(X_test_dummie),average='macro')*100))
print("Decision Tree : f1 score = "+str(f1_score(y_test, tree_model1.predict(X_test_dummie),average='macro')*100))
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb_model = gnb.fit(X_train_dummie, y_train)
print("naive_bayes Metric Evalaute all train,test")
print("naive_bayes : accuracy score = "+str(accuracy_score(y_test, gnb_model.predict(X_test_dummie), normalize=True)*100))
print("naive_bayes : precision score = "+str(precision_score(y_test, gnb_model.predict(X_test_dummie),average='macro')*100))
print("naive_bayes : recall score = "+str(recall_score(y_test, gnb_model.predict(X_test_dummie),average='macro')*100))
print("naive_bayes : f1 score = "+str(f1_score(y_test, gnb_model.predict(X_test_dummie),average='macro')*100))
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(2,),(3,),(4,),(5,2),(6,2),(7,2),(8,2),(9,2),(10,2),(11,2), (12,2),(13,2),(14,2),(15,2)
             ]
        }
       ]
clf_tune = GridSearchCV(MLPClassifier(), param_grid, cv=5,
                           scoring='accuracy')
clf_tune.fit(X_train_dummie,y_train)


print("Best parameters set found on development set:")
print(clf_tune.best_params_)
mlpc_clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4, ),activation= 'tanh',random_state=30)
mlpc_clf_model =mlpc_clf.fit(X_train_dummie, y_train)
print("MLP Classifier Metric Evalaute all train,test")
print("MLP Classifier : accuracy score = "+str(accuracy_score(y_test, mlpc_clf_model.predict(X_test_dummie), normalize=True)*100))
print("MLP Classifier : precision score = "+str(precision_score(y_test, mlpc_clf_model.predict(X_test_dummie),average='macro')*100))
print("MLP Classifier : recall score = "+str(recall_score(y_test, mlpc_clf_model.predict(X_test_dummie),average='macro')*100))
print("MLP Classifier : f1 score = "+str(f1_score(y_test, mlpc_clf_model.predict(X_test_dummie),average='macro')*100))
# KFold Cross Validation approach
X=X_train_dummie
y=y_train
kf = KFold(n_splits=5,shuffle=True,random_state=30)
kf.split(X)    


# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
accuracy_model1 = []
precision_score_model1 = []
recall_score_model1 = []
f1_score_model1=[] 

tree_clf = DecisionTreeClassifier(random_state=30)

# Iterate over each train-test split
for train_index, test_index in kf.split(X):
    # Split train-test
    X_train_f, X_test_f = X.iloc[train_index], X.iloc[test_index]
    y_train_f, y_test_f = y[train_index], y[test_index]
    # Train the model
    model1 = tree_clf.fit(X_train_f, y_train_f)
    # Append to accuracy_model the accuracy of the model
    
    accuracy_model1.append(accuracy_score(y_test_f, model1.predict(X_test_f), normalize=True)*100)
    precision_score_model1.append(precision_score(y_test_f, model1.predict(X_test_f),average='macro')*100)
    recall_score_model1.append(recall_score(y_test_f, model1.predict(X_test_f),average='macro')*100)
    f1_score_model1.append(f1_score(y_test_f, model1.predict(X_test_f),average='macro')*100)
    

print("Decision Tree Classification 5 Fold Cross validate Metric \n")
print("accuary score 5 Fold cross validate : "+ str(accuracy_model1)+ "\n avg ="+str(sum(accuracy_model1) / len(accuracy_model1))+"\n")
print("precision score 5 Fold cross validate : "+str(precision_score_model1)+"\n avg ="+str(sum(precision_score_model1) / len(precision_score_model1))+"\n")
print("recall score 5 Fold cross validate : "+str(recall_score_model1)+"\n avg ="+str(sum(recall_score_model1) / len(recall_score_model1))+"\n")
print("f1 score 5 Fold cross validate : "+str(f1_score_model1)+"\n avg ="+str(sum(f1_score_model1) / len(f1_score_model1))+"\n")

import seaborn as sns
import matplotlib.pyplot as plt
### Visualize accuracy for each iteration
 
accuracy_scores = pd.DataFrame(accuracy_model1,columns=['accuracy scores'])
precision_scores = pd.DataFrame(precision_score_model1,columns=['precision scores'])
recall_scores = pd.DataFrame(recall_score_model1,columns=['recall scores'])
f1_scores = pd.DataFrame(f1_score_model1,columns=['f1 scores'])


fig, axs = plt.subplots(ncols=4,figsize=(20,5))



sns.set(style="white", rc={"lines.linewidth": 3})
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="accuracy scores",data=accuracy_scores,ax=axs[0]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="precision scores",data=precision_scores,ax=axs[1]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="recall scores",data=recall_scores,ax=axs[2]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="f1 scores",data=f1_scores,ax=axs[3]).set(ylim=(0, 100))

#plt.subtitle("Decision Tree metrix Score 5 Fold cross validate",y=1, fontsize = 16,loc='center')

plt.show()
sns.set()
# KFold Cross Validation approach
X=X_train_dummie
y=y_train
kf = KFold(n_splits=5,shuffle=True,random_state=30)
kf.split(X)    


# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
accuracy_model2 = []
precision_score_model2 = []
recall_score_model2 = []
f1_score_model2=[] 

gnb = GaussianNB()

# Iterate over each train-test split
for train_index, test_index in kf.split(X):
    # Split train-test
    X_train_f, X_test_f = X.iloc[train_index], X.iloc[test_index]
    y_train_f, y_test_f = y[train_index], y[test_index]
    # Train the model
    model2 = gnb.fit(X_train_f, y_train_f)
    # Append to accuracy_model the accuracy of the model
    
    accuracy_model2.append(accuracy_score(y_test_f, model2.predict(X_test_f), normalize=True)*100)
    precision_score_model2.append(precision_score(y_test_f, model2.predict(X_test_f),average='macro')*100)
    recall_score_model2.append(recall_score(y_test_f, model2.predict(X_test_f),average='macro')*100)
    f1_score_model2.append(f1_score(y_test_f, model2.predict(X_test_f),average='macro')*100)
    
print("Naive Bayes 5 Fold Cross validate Metric \n") 
print("accuary score 5 Fold cross validate : "+ str(accuracy_model2)+ "\n avg ="+str(sum(accuracy_model2) / len(accuracy_model2))+"\n")
print("precision score 5 Fold cross validate : "+str(precision_score_model2)+"\n avg ="+str(sum(precision_score_model2) / len(precision_score_model2))+"\n")
print("recall score 5 Fold cross validate : "+str(recall_score_model2)+"\n avg ="+str(sum(recall_score_model2) / len(recall_score_model2))+"\n")
print("f1 score 5 Fold cross validate : "+str(f1_score_model2)+"\n avg ="+str(sum(f1_score_model2) / len(f1_score_model2))+"\n")
### Visualize accuracy for each iteration
 
accuracy_scores = pd.DataFrame(accuracy_model2,columns=['accuracy scores'])
precision_scores = pd.DataFrame(precision_score_model2,columns=['precision scores'])
recall_scores = pd.DataFrame(recall_score_model2,columns=['recall scores'])
f1_scores = pd.DataFrame(f1_score_model2,columns=['f1 scores'])


fig, axs = plt.subplots(ncols=4,figsize=(20,5))



sns.set(style="white", rc={"lines.linewidth": 3})
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="accuracy scores",data=accuracy_scores,ax=axs[0]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="precision scores",data=precision_scores,ax=axs[1]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="recall scores",data=recall_scores,ax=axs[2]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="f1 scores",data=f1_scores,ax=axs[3]).set(ylim=(0, 100))

#plt.subtitle("Naieve Bayes metric Score 5 Fold cross validate",y=1, fontsize = 16,loc='center')

plt.show()
sns.set()
# KFold Cross Validation approach
X=X_train_dummie
y=y_train
kf = KFold(n_splits=5,shuffle=True,random_state=30)
kf.split(X)    


# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
accuracy_model3 = []
precision_score_model3 = []
recall_score_model3 = []
f1_score_model3=[] 

mlpc_clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4, ),activation= 'tanh',random_state=30)

# Iterate over each train-test split
for train_index, test_index in kf.split(X):
    # Split train-test
    X_train_f, X_test_f = X.iloc[train_index], X.iloc[test_index]
    y_train_f, y_test_f = y[train_index], y[test_index]
    # Train the model
    model3 = mlpc_clf.fit(X_train_f, y_train_f)
    # Append to accuracy_model the accuracy of the model
    
    accuracy_model3.append(accuracy_score(y_test_f, model3.predict(X_test_f), normalize=True)*100)
    precision_score_model3.append(precision_score(y_test_f, model3.predict(X_test_f),average='macro')*100)
    recall_score_model3.append(recall_score(y_test_f, model3.predict(X_test_f),average='macro')*100)
    f1_score_model3.append(f1_score(y_test_f, model3.predict(X_test_f),average='macro')*100)
print("MLPC 5 Fold Cross validate Metric Best Hyperparameter\n") 
print("accuary score 5 Fold cross validate : "+ str(accuracy_model3)+ "\n avg ="+str(sum(accuracy_model3) / len(accuracy_model3))+"\n")
print("precision score 5 Fold cross validate : "+str(precision_score_model3)+"\n avg ="+str(sum(precision_score_model3) / len(precision_score_model3))+"\n")
print("recall score 5 Fold cross validate : "+str(recall_score_model3)+"\n avg ="+str(sum(recall_score_model3) / len(recall_score_model4))+"\n")
print("f1 score 5 Fold cross validate : "+str(f1_score_model3)+"\n avg ="+str(sum(f1_score_model3) / len(f1_score_model3))+"\n")
### Visualize accuracy for each iteration
 
accuracy_scores = pd.DataFrame(accuracy_model3,columns=['accuracy scores'])
precision_scores = pd.DataFrame(precision_score_model3,columns=['precision scores'])
recall_scores = pd.DataFrame(recall_score_model3,columns=['recall scores'])
f1_scores = pd.DataFrame(f1_score_model3,columns=['f1 scores'])


fig, axs = plt.subplots(ncols=4,figsize=(20,5))



sns.set(style="white", rc={"lines.linewidth": 3})
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="accuracy scores",data=accuracy_scores,ax=axs[0]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="precision scores",data=precision_scores,ax=axs[1]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="recall scores",data=recall_scores,ax=axs[2]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="f1 scores",data=f1_scores,ax=axs[3]).set(ylim=(0, 100))

#plt.subtitle("Naieve Bayes metric Score 5 Fold cross validate",y=1, fontsize = 16,loc='center')

plt.show()
sns.set()
# KFold Cross Validation approach
X=X_train_dummie
y=y_train
kf = KFold(n_splits=5,shuffle=True,random_state=30)
kf.split(X)    


# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
accuracy_model4 = []
precision_score_model4 = []
recall_score_model4 = []
f1_score_model4=[] 

mlpc_clf = MLPClassifier(solver='adam',hidden_layer_sizes=(30,30 ), random_state=30)

# Iterate over each train-test split
for train_index, test_index in kf.split(X):
    # Split train-test
    X_train_f, X_test_f = X.iloc[train_index], X.iloc[test_index]
    y_train_f, y_test_f = y[train_index], y[test_index]
    # Train the model
    model4 = mlpc_clf.fit(X_train_f, y_train_f)
    # Append to accuracy_model the accuracy of the model
    
    accuracy_model4.append(accuracy_score(y_test_f, model4.predict(X_test_f), normalize=True)*100)
    precision_score_model4.append(precision_score(y_test_f, model4.predict(X_test_f),average='macro')*100)
    recall_score_model4.append(recall_score(y_test_f, model4.predict(X_test_f),average='macro')*100)
    f1_score_model4.append(f1_score(y_test_f, model4.predict(X_test_f),average='macro')*100)
print("MLPC 5 Fold Cross validate Metric Manual Tune\n") 
print("accuary score 5 Fold cross validate : "+ str(accuracy_model4)+ "\n avg ="+str(sum(accuracy_model4) / len(accuracy_model4))+"\n")
print("precision score 5 Fold cross validate : "+str(precision_score_model4)+"\n avg ="+str(sum(precision_score_model4) / len(precision_score_model4))+"\n")
print("recall score 5 Fold cross validate : "+str(recall_score_model4)+"\n avg ="+str(sum(recall_score_model4) / len(recall_score_model4))+"\n")
print("f1 score 5 Fold cross validate : "+str(f1_score_model4)+"\n avg ="+str(sum(f1_score_model4) / len(f1_score_model4))+"\n")
### Visualize accuracy for each iteration
 
accuracy_scores = pd.DataFrame(accuracy_model4,columns=['accuracy scores'])
precision_scores = pd.DataFrame(precision_score_model4,columns=['precision scores'])
recall_scores = pd.DataFrame(recall_score_model4,columns=['recall scores'])
f1_scores = pd.DataFrame(f1_score_model4,columns=['f1 scores'])


fig, axs = plt.subplots(ncols=4,figsize=(20,5))



sns.set(style="white", rc={"lines.linewidth": 3})
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="accuracy scores",data=accuracy_scores,ax=axs[0]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="precision scores",data=precision_scores,ax=axs[1]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="recall scores",data=recall_scores,ax=axs[2]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5'],y="f1 scores",data=f1_scores,ax=axs[3]).set(ylim=(0, 100))

#plt.subtitle("Naieve Bayes metric Score 5 Fold cross validate",y=1, fontsize = 16,loc='center')

plt.show()
sns.set()
# KFold Cross Validation approach
X=X_train_dummie
y=y_train
kf = KFold(n_splits=10,shuffle=True,random_state=30)
kf.split(X)    


# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
accuracy_model5 = []
precision_score_model5 = []
recall_score_model5 = []
f1_score_model5=[] 

mlpc_clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4, ),activation= 'tanh',random_state=30)

# Iterate over each train-test split
for train_index, test_index in kf.split(X):
    # Split train-test
    X_train_f, X_test_f = X.iloc[train_index], X.iloc[test_index]
    y_train_f, y_test_f = y[train_index], y[test_index]
    # Train the model
    model5 = mlpc_clf.fit(X_train_f, y_train_f)
    # Append to accuracy_model the accuracy of the model
    
    accuracy_model5.append(accuracy_score(y_test_f, model5.predict(X_test_f), normalize=True)*100)
    precision_score_model5.append(precision_score(y_test_f, model5.predict(X_test_f),average='macro')*100)
    recall_score_model5.append(recall_score(y_test_f, model5.predict(X_test_f),average='macro')*100)
    f1_score_model5.append(f1_score(y_test_f, model5.predict(X_test_f),average='macro')*100)
    

print("MLPC Classification 10 Fold Cross validate Metric \n") 
print("accuary score 10 Fold cross validate :\n"+ str(accuracy_model5)+ "\n avg=" +str(sum(accuracy_model5) / len(accuracy_model5))+"\n")
print("precision score 10 Fold cross validate :\n"+str(precision_score_model5)+"\n avg="+str(sum(precision_score_model5) / len(precision_score_model5))+"\n")
print("recall score 10 Fold cross validate :\n"+str(recall_score_model5)+"\n avg="+ str(sum(recall_score_model5) / len(recall_score_model5))+"\n")
print("f1 score 10 Fold cross validate :\n"+str(f1_score_model5)+"\n avg=" +str(sum(f1_score_model5) / len(f1_score_model5))+"\n")

### Visualize accuracy for each iteration
 
accuracy_scores = pd.DataFrame(accuracy_model5,columns=['accuracy scores'])
precision_scores = pd.DataFrame(precision_score_model5,columns=['precision scores'])
recall_scores = pd.DataFrame(recall_score_model5,columns=['recall scores'])
f1_scores = pd.DataFrame(f1_score_model5,columns=['f1 scores'])


fig, axs = plt.subplots(ncols=4,figsize=(20,5))



sns.set(style="white", rc={"lines.linewidth": 3})
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5','Iter6','Iter7','Iter8','Iter9','Iter10'],y="accuracy scores",data=accuracy_scores,ax=axs[0]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5','Iter6','Iter7','Iter8','Iter9','Iter10'],y="precision scores",data=precision_scores,ax=axs[1]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5','Iter6','Iter7','Iter8','Iter9','Iter10'],y="recall scores",data=recall_scores,ax=axs[2]).set(ylim=(0, 100))
sns.barplot(x=['Iter1','Iter2','Iter3','Iter4','Iter5','Iter6','Iter7','Iter8','Iter9','Iter10'],y="f1 scores",data=f1_scores,ax=axs[3]).set(ylim=(0, 100))

#plt.subtitle("Naieve Bayes metric Score 5 Fold cross validate",y=1, fontsize = 16,loc='center')

plt.show()
sns.set()