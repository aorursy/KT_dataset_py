# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import Pipeline # Import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV # SVM Parameter optimization/tuning
from sklearn.cross_validation import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import cross_validation
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
testpath="../input/test.csv"
trainpath="../input/train.csv"
# Any results you write to the current directory are saved as output.

#****************************************************************************************************************************
def get_combined_data(sel_features):
    # reading train data
    train = pd.read_csv(trainpath)
    
    # reading test data
    test = pd.read_csv(testpath)

    # extracting and then removing the targets from the training data 
    targets = train.Survived
   # train.drop(['Survived'], 1, inplace=True)
    

    # merging train data and test data for future feature engineering
    # we'll also remove the PassengerID since this is not an informative feature
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'PassengerId'], inplace=True, axis=1)
    
    combined=combined[sel_features] # Restrict dataset to desired columns
    
    return combined

def create_xgboost(X_train,y_train,X_test,y_test,testX):
    
    # Using Grid Search parameters, but without GridSearch
  # Best: 0.887395 using {'gamma': 0.3, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 200, 'reg_alpha': 0.1}
  #  'gamma': 0.4, 'learning_rate': 0.05, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0.1
 #  Best: 0.879566 using {'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 11, 'min_child_weight': 1, 'n_estimators': 50, 'reg_alpha': 0.1}
    classifier=XGBClassifier(
         learning_rate =0.05,
         n_estimators=50,
         max_depth=11,
         min_child_weight=1,
         gamma=0.1,
         subsample=0.8,
         colsample_bytree=0.8,
         reg_alpha=0.1,
         objective= 'binary:logistic',
         nthread=4,
         scale_pos_weight=1,
         seed=27)
    #Training
    classifier.fit(X_train,y_train)
    #print(classifier.best_params_)
    # Training Accuracy
    accuracy = classifier.score(X_train, y_train)
    print("\nAccuracy on sample training data - numeric, no nans: ", accuracy*100)
    print(" Training Data Set Type : ", type(X_train), " Size : ", len(X_train))
    
    # Prediction on Test Data Which is part of Training Split
    accuracyTest = classifier.score(X_test, y_test)
    print("\nAccuracy on sample test data - numeric, no nans: ", accuracyTest*100)
    
    prediction=classifier.predict(testX)
    return prediction

def xgboost_gridsearch(X_train,y_train,X_test,y_test,testX):
   #Best: 0.877856 using {'gamma': 0.3, 'learning_rate': 0.05, 'max_depth': 9, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 1}
    # reproducibility
    seed = 342
    np.random.seed(seed)

    param_test1 = {
    'max_depth':range(3,20,2),
    'n_estimators':range(50,250,50),
    'min_child_weight':range(1,6,2),
    'learning_rate':[0.01,0.05,0.1],
    'gamma':[i/10.0 for i in range(0,5)],
    'reg_alpha':[0.1, 1, 100]
    }
    kcv = StratifiedKFold(y_train, n_folds=3, shuffle=True, random_state=seed)
    classifier = GridSearchCV(estimator = XGBClassifier(gamma=0.2, learning_rate=0.1, max_delta_step=0, max_depth=3,
            min_child_weight=1, missing=None, n_estimators=100, subsample=0.8, colsample_bytree=0.8,
            objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
            param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=kcv)
    
    grid_result = classifier.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


#********************************************************************************************************************************
def get_targets(path,col):
    # reading train data
    target = pd.read_csv(path)
  #  print("Target ", col, " : ",(target[col]).head())
    target_col=target[col]
    return target_col
    
#********************************************************************************************************************************
def status(feature):
    print ('Processing', feature, ': ok')

#********************************************************************************************************************************
def process_pclass():
    
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    
    # adding dummy variable
    combined = pd.concat([combined, pclass_dummies],axis=1)
    
    # removing "Pclass"
    combined.drop('Pclass',axis=1,inplace=True)
    
    status('Pclass')
    return combined

def process_sex():
    global combined
    # mapping string values to numerical one 
  #  combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})
    sex_Dummy=pd.get_dummies(combined['Sex']) # Preprocessing to handle categorical Sex column
    combined=combined.drop('Sex', axis=1) # dropping categorical sex column
    combined=pd.concat([combined,sex_Dummy],axis=1) # adding dummy sex columns to the training dataset

    status('Sex')
    return combined

def process_cabin():
    global combined
    # Feature that tells whether a passenger had a cabin on the Titanic
    combined['Has_Cabin'] = combined["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    # removing "Cabin"
    combined.drop('Cabin',axis=1,inplace=True)
    return combined

def process_age():
    global combined
    combined['Age'] = combined['Age'].fillna(combined['Age'].median())
    combined.loc[ combined['Age'] <= 16, 'Age'] = 0
    combined.loc[(combined['Age'] > 16) & (combined['Age'] <= 32), 'Age'] = 1
    combined.loc[(combined['Age'] > 32) & (combined['Age'] <= 48), 'Age'] = 2
    combined.loc[(combined['Age'] > 48) & (combined['Age'] <= 64), 'Age'] = 3
    combined.loc[ combined['Age'] > 64, 'Age']
    combined['Age'] = combined['Age'].astype(int)
    status('Age')
    return combined

def process_title():
    global combined
   # for dataset in combined:
    #    dataset['Title'] = dataset.Name.extract(' ([A-Za-z]+)\.', expand=False)
    #pd.crosstab(train_df['Title'], train_df['Sex'])
    combined['Title']=combined.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    #print(pd.crosstab(combined['Title'], combined['Sex']))
    
    
    combined['Title'] = combined['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
     	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    combined['Title'] = combined['Title'].replace('Mlle', 'Miss')
    combined['Title'] = combined['Title'].replace('Ms', 'Miss')
    combined['Title'] = combined['Title'].replace('Mme', 'Mrs')
    
    #print(combined[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    combined['Title'] = combined['Title'].map(title_mapping)
    combined['Title'] = combined['Title'].fillna(0)
    #Remove Name Column
    combined.drop(['Name'],inplace=True,axis=1)
    #print(combined.head())
    
    status('Title')
    #print("\n Titles ", combined.Title)
    return combined
#*********************************************************************************************************************************


def process_embarked():
    global combined
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    combined.Embarked.fillna('S', inplace=True)
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    #Remove Embarked
    combined.drop('Embarked', axis=1, inplace=True)
    status('embarked')
    return combined

def process_fares():
    global combined
    #train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    #train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)
    combined.loc[ combined['Fare'] <= 7.91, 'Fare'] = 0
    combined.loc[(combined['Fare'] > 7.91) & (combined['Fare'] <= 14.454), 'Fare'] = 1
    combined.loc[(combined['Fare'] > 14.454) & (combined['Fare'] <= 31), 'Fare']   = 2
    combined.loc[ combined['Fare'] > 31, 'Fare'] = 3
    combined['Fare'] = combined['Fare'].astype(int)

    status('fare')
    return combined
#**********************************************************************************************************************************
def process_isalone():
    global combined
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    combined['IsAlone'] = 0
    combined.loc[combined['FamilySize'] == 1, 'IsAlone'] = 1
    # dripping features
    drop_elements = ['Parch', 'SibSp']
    combined = combined.drop(drop_elements, axis = 1)

    return combined
#**********************************************************************************************************************************
def preprocessing_scale():
    global combined
    std_X = StandardScaler()
    combined = std_X.fit_transform(combined)
    #combined=combined[:,1:] # Removing index column
    combined=pd.DataFrame(data=combined,index=combined[0:,0])
    return combined



def prepare_submission(prediction):
    pid=get_targets("../input/test.csv",'PassengerId')
    my_submission = pd.DataFrame({'PassengerId': pid, 'Survived': prediction})
    # you could use any filename. We choose submission here
    my_submission.to_csv('gender_submission.csv', index=False)

def xgboost_classification(trainX,trainy,X_test,y_test,testX):
    prediction=create_xgboost(trainX,trainy,X_test,y_test,testX)
    # Prediction on Test Data
    rounded = np.round(prediction)#[int(round(x[0],2)) for x in prediction]
    rounded=rounded.astype(int)
    prepare_submission(rounded)

def classify_randomforest(trainX,trainy,X_test,y_test,testX):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(trainX, trainy)
    prediction = random_forest.predict(testX)
    print(" Training Score for Random Forest : ",random_forest.score(trainX, trainy)*100)
    print(" Testing Score for Random Forest : ", round(random_forest.score(X_test, y_test) * 100, 2))
    rounded = np.round(prediction)#[int(round(x[0],2)) for x in prediction]
    rounded=rounded.astype(int)
    prepare_submission(rounded)
    #return prediction

# Stochastic Gradient Descent
def classify_SGD(trainX,trainy,X_test,y_test,testX):
    sgd = SGDClassifier()
    sgd.fit(trainX, trainy)
    prediction = sgd.predict(testX)
    print(" Training Score for Stochastic Gradient Descent : ",sgd.score(trainX, trainy)*100)
    print(" Testing Score for Stochastic Gradient Descent : ", round(sgd.score(X_test, y_test) * 100, 2))
    rounded = np.round(prediction)#[int(round(x[0],2)) for x in prediction]
    rounded=rounded.astype(int)
    #Submission
    prepare_submission(rounded)

def create_SeqNNmodel():
    # Set up the NN model
    global combined
    n_cols=len(combined.columns)
    model = Sequential()
    # Layer unit configuration
    units=[1600,1600,800,800]
    # Add the first layer
    model.add(Dense(units[0],activation='relu', input_shape=(n_cols,)))
    # Add the hidden layer
    model.add(Dense(units[1],activation='relu'))
    # Add the hidden layer
    model.add(Dense(units[2],activation='relu'))
    # Add the hidden layer
    model.add(Dense(units[3],activation='relu'))
    
    # Add the output layer
    model.add(Dense(1,activation='sigmoid'))
    
    # Compile the model
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    return model

def classify_voting(trainX,trainy,X_test,y_test,testX):
    seed = 7
    kfold = model_selection.KFold(n_splits=2, random_state=seed)
    # create the sub models
    estimators = []
    #model1 = LogisticRegression(penalty = 'l2', C = 100,random_state = 0)
    #estimators.append(('logistic', model1))
    model1 = SGDClassifier(loss="hinge", alpha=0.1, max_iter=500, fit_intercept=True)
    estimators.append(('sgd', model1))
    model2 = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=8, min_samples_leaf=5)
    estimators.append(('decisiontree', model2))
    model3 = RandomForestClassifier(bootstrap= True, max_depth= 16, max_features= 'auto',
                               min_samples_leaf= 3, min_samples_split= 10, n_estimators= 100)
    estimators.append(('randomforest', model3))
    model4=XGBClassifier(
         learning_rate =0.1,
         n_estimators=1000,
         max_depth=11,
         min_child_weight=1,
         gamma=0.3,
         subsample=0.8,
         colsample_bytree=0.8,
         reg_alpha=0.1,
         objective= 'binary:logistic',
         nthread=4,
         scale_pos_weight=1,
         seed=27)
    estimators.append(('xgboost', model4))
    #model5= SVC()
    #estimators.append(('svc', model5))
    model5 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                         algorithm="SAMME",
                         n_estimators=1000, learning_rate=0.1,random_state=10)
    estimators.append(('adaboost', model5))
    model6 = GradientBoostingClassifier(n_estimators=1000,loss="exponential",max_features=4,
    max_depth=5,subsample=0.5,learning_rate=0.005, random_state=10)
    estimators.append(('grboost', model6))
    model7 = MLPClassifier((100, 100, 100), early_stopping=False, random_state=10,max_iter=500)
    estimators.append(('mlp', model7))
   # model6=create_SeqNNmodel()
    #estimators.append(('seqNN', model6))
    # create the ensemble model
    ensemble = VotingClassifier(estimators,voting='hard',weights=[1,1,1.5,1.5,1.5,1.5,1.5])
    results = model_selection.cross_val_score(ensemble, trainX, trainy, cv=kfold)
    print("Ensemble Score : ",results.mean())
    ensemble.fit(trainX, trainy)
    prediction = ensemble.predict(testX)
    print(" Training Score for Voting Method : ",ensemble.score(trainX, trainy)*100)
    print(" Testing Score for Voting Method : ", round(ensemble.score(X_test, y_test) * 100, 2))
    rounded = np.round(prediction)#[int(round(x[0],2)) for x in prediction]
    rounded=rounded.astype(int)
    
    #Printing Results of Classifiers
    for clf, label in zip([model1, model2, model3, model4, model5,model6,model7, ensemble], ['sgd','Decision Tree',
                                                     'Random Forest','XGBoost','adaboost','grboost','mlp', 'Ensemble']):
        scores = cross_validation.cross_val_score(clf, trainX, trainy, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    #Submission
    prepare_submission(rounded)

def feature_preprocessing(sel_ch,scaling):
    global combined
    if sel_ch==1:
        sel_col= ['Fare','Age','Sex','Embarked']
        combined=get_combined_data(sel_col)
        combined=process_fares()
        combined=process_embarked()
        combined=process_sex()
        #combined=process_pclass()
        combined=process_age()
    elif sel_ch==2:
        sel_col= ['Pclass','Fare','Age','Sex','Embarked']
        combined=get_combined_data(sel_col)
        combined=process_fares()
        combined=process_embarked()
        combined=process_sex()
        combined=process_pclass()
        combined=process_age()
    elif sel_ch==3:
        sel_col= ['Pclass','Fare','Age','Sex','SibSp','Embarked']
        combined=get_combined_data(sel_col)
        combined=process_fares()
        combined=process_embarked()
        combined=process_sex()
        combined=process_pclass()
        combined=process_age()
    elif sel_ch==4:
        sel_col= ['Age','Sex','Embarked']
        combined=get_combined_data(sel_col)
        combined=process_embarked()
        combined=process_sex()
        combined=process_age()
    elif sel_ch==5:
        sel_col= ['Pclass','Fare','Age','Sex','SibSp','Embarked','Name','Survived','Parch','Cabin']
       # sel_col= ['Pclass','Fare','Sex','SibSp','Embarked','Name','Survived','Parch']
        combined=get_combined_data(sel_col)
        combined=process_isalone()
        combined=process_fares()
        combined=process_embarked()
        combined=process_sex()
        combined=process_pclass()
        combined=process_age()
        combined=process_title()
        combined=process_cabin()
        PearsonCorrelation(combined)
        combined.drop(['Survived'],inplace=True,axis=1)
    if scaling==True:
        combined=preprocessing_scale()

    return combined

import matplotlib.pyplot as plt
import seaborn as sns

def PearsonCorrelation(dataset):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(dataset.astype(float).corr(),linewidths=0.1,vmax=1.0, 
                square=True, cmap=colormap, linecolor='white', annot=True)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# Configs
sel_ch=5
scaling=False
tst_size=0.8
tree_depth=9
n_est=100
# Preprocessing
combined=feature_preprocessing(sel_ch,scaling)
#Analyzing Training and Testing Set after preprocessing
#combined.to_csv('combined.csv',index=False)

#Training Setup
trainX=combined.iloc[:891]
print("Training Set :\n",trainX.head())
#trainX=train_df[sel_col]
trainy=np.array(get_targets("../input/train.csv",'Survived'))


# Creating Train/Test Split
# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(trainX,trainy,test_size=0.3, random_state=21)

#Setting up Test X
testX=combined.iloc[891:]
#Analyzing Training and Testing Set after preprocessing
#combined.to_csv('combined.csv',index=False)
#Classification and prediction
classify_voting(trainX,trainy,X_test,y_test,testX)
#classify_voting(X_train,y_train,X_test,y_test,testX)
#xgboost_classification(trainX,trainy,X_test,y_test,testX)
#xgboost_classification(X_train,y_train,X_test,y_test,testX)
#classify_randomforest(trainX,trainy,X_test,y_test,testX)
#classify_SGD(trainX,trainy,X_test,y_test,testX)
#******************
# parameter optimization
#prediction=xgboost_gridsearch(trainX,trainy,X_test,y_test,testX)