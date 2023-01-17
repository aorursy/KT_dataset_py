# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,precision_score, accuracy_score

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Total_train_records = 0

Total_test_records = 0

Train_Features = []

Train_Features_Type = []

Train_Features_records = []

Train_NAN_records = []

Train_Percentage_NAN = [] 

Test_Features = []

Test_Features_Type = []

Test_Features_records = []

Test_NAN_records = []

Test_Percentage_NAN = []



Train_Test_records = 0

Train_Test_Features = []

Train_Test_Features_Type = []

Train_Test_Features_records = []

Train_Test_NAN_records = []

Train_Test_Percentage_NAN = []



Train_results = []

Test_results = []

DSTree_depth = []

Samples_split = []

Samples_leaf = []

RFTree_depth = []

    
def Main():

    titanic_train, titanic_test = ReadFiles()

    titanic_train_test = KnowYourData(titanic_train,titanic_test)

    DataAnalysis(titanic_train_test)

    titanic_train, titanic_test1 = Preprocessing(titanic_train_test)

    Correlation(titanic_train)

    X_train,X_test,Y_train,Y_test,titanic_test1 = Model_Selection(titanic_train,titanic_test1)

    LogRegression(X_train,X_test,Y_train,Y_test,titanic_test1,titanic_test)

    DecisionTree(X_train,X_test,Y_train,Y_test)

    RandomForest(X_train,X_test,Y_train,Y_test,titanic_test1,titanic_test)

#Read Both data files

def ReadFiles():

    titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv")

    titanic_test = pd.read_csv("/kaggle/input/titanic/test.csv")

    print("Shape of the Train Dataset:", titanic_train.shape)

    print("Shape of the Test Dataset:", titanic_test.shape)

    return titanic_train, titanic_test
def KnowYourData(titanic_train,titanic_test):

    global Total_train_records,Total_test_records

    global Train_Features,Train_Features_Type,Train_Features_records,Train_NAN_records,Train_Percentage_NAN

    global Test_Features,Test_Features_Type,Test_Features_records,Test_NAN_records,Test_Percentage_NAN

    print("Know Your Data")

    Total_train_records = len(titanic_train)

    Total_test_records = len(titanic_test)

    

    #Train DataSet

    for col in titanic_train.columns:

        Train_Features.append(col)

        Train_Features_Type.append(titanic_train[col].dtypes)

        Train_Features_records.append(Total_train_records)

        Train_NAN_records.append(titanic_train[col].isnull().sum())

        Train_Percentage_NAN.append((titanic_train[col].isnull().sum() / Total_train_records) * 100)

    

    

    #Test DataSet

    for col in titanic_test.columns:

        Test_Features.append(col)

        Test_Features_Type.append(titanic_test[col].dtypes)

        Test_Features_records.append(Total_test_records)

        Test_NAN_records.append(titanic_test[col].isnull().sum())

        Test_Percentage_NAN.append((titanic_test[col].isnull().sum() / Total_test_records) * 100)

    

    print("About Train Data:")

    display(pd.DataFrame({"Features":Train_Features,"Features Type":Train_Features_Type,"Total Records": Train_Features_records,

    "Total NAN": Train_NAN_records, "Percentage of NAN": Train_Percentage_NAN}).T)

    

    Survived_df = pd.DataFrame(titanic_train['Survived'].value_counts().reset_index())

    Survived_df.rename(columns={"index":"Survived","Survived":"Count"},inplace=True)

    Survived_df['Percentage'] = Survived_df['Count'].apply(lambda x: (x/ Survived_df['Count'].sum())*100)

    display(Survived_df)

    #display(pd.DataFrame({"Survived-0": len(titanic_train['Survived'][0]), "Survived-1": len(titanic_train['Survived'][1])}))

        

    print("About Test Data:")

    display(pd.DataFrame({"Features":Test_Features,"Features Type":Test_Features_Type,"Total Records": Test_Features_records,

    "Total NAN": Test_NAN_records, "Percentage of NAN": Test_Percentage_NAN}).T)

    

    #Total titanic dataset

    titanic_train_test = pd.concat([titanic_train,titanic_test],sort=False)

    Train_Test_records = len(titanic_train_test)

    for col in titanic_train_test.columns:

        if col != 'Survived':

            Train_Test_Features.append(col)

            Train_Test_Features_Type.append(titanic_train_test[col].dtypes)

            Train_Test_Features_records.append(Train_Test_records)

            Train_Test_NAN_records.append(titanic_train_test[col].isnull().sum())

            Train_Test_Percentage_NAN.append((titanic_train_test[col].isnull().sum() / Train_Test_records) * 100)

    

    print("About Total Data:")

    display(pd.DataFrame({"Features":Train_Test_Features,"Features Type":Train_Test_Features_Type,

                          "Total Records": Train_Test_Features_records,"Total NAN": Train_Test_NAN_records,

                          "Percentage of NAN": Train_Test_Percentage_NAN}).T)

    

    return titanic_train_test
def DataAnalysis(titanic_train_test):

    print("Data Analysis:")

    display(titanic_train_test.head(30))

    display(titanic_train_test[['Pclass','Cabin','Survived']].head(100).T)

    display(titanic_train_test.groupby(['Pclass','Survived']).count())

    display("Minimum Age by Sex:",titanic_train_test.groupby('Sex')['Age'].agg(['min','max'],axis=1))

    display("NaN records by Fare : \n",titanic_train_test[titanic_train_test['Fare'].isnull()])

    display("Mean Fare by Pclass:",titanic_train_test.groupby('Pclass')['Fare'].agg(['mean','median'],axis=1))

    display("NaN records by Embarked:\n",titanic_train_test[titanic_train_test['Embarked'].isnull()])

    display("Mode of Embarked by Pclass:",titanic_train_test.groupby('Pclass')['Embarked'].agg(pd.Series.mode))



def Preprocessing(titanic_train_test):

    print("Preprocessing Starts.....")

    print("Number of duplicated data:",titanic_train_test.duplicated().sum())

    titanic_train_test = HandleMissingValues(titanic_train_test)

    titanic_train_test = IrrelevantFeatures(titanic_train_test)

    titanic_train_test = Feature_Extraction(titanic_train_test)

    titanic_train_test = RedunantData(titanic_train_test)

    #titanic_train_test = PrepareForEncode(titanic_train_test)

    titanic_train_test = Encoding(titanic_train_test)

    titanic_train = titanic_train_test.head(891)

    titanic_test = titanic_train_test.tail(418)

    display("Preprocessing Ends.....")

    return titanic_train, titanic_test
def HandleMissingValues(titanic_train_test):

    print("Missing Values Phase Starts:")

    for col in titanic_train_test.columns:

        if (titanic_train_test[col].isnull().sum() / len(titanic_train_test)) * 100 >= 50:

            titanic_train_test.drop(col,axis=1,inplace=True)

    

    #titanic_train_test[col].fillna(titanic_train_test)

    print("Mean Age by Sex:\n",titanic_train_test.groupby('Sex')['Age'].mean())

    titanic_train_test['Age'] = titanic_train_test['Age'].fillna(titanic_train_test['Age'].mean())

    print("Missing Count After Imputation of Age:",titanic_train_test['Age'].isnull().sum())

    titanic_train_test['Fare'] = titanic_train_test['Fare'].fillna(titanic_train_test.groupby('Pclass')['Fare'].mean()[3])

    print("Missing Count After Imputation of Fare:",titanic_train_test['Fare'].isnull().sum())

    titanic_train_test['Embarked'] = titanic_train_test['Embarked'].fillna(titanic_train_test.groupby('Pclass')['Embarked'].agg(pd.Series.mode)[1])

    print("Missing Count After Imputation of Embarked:",titanic_train_test['Embarked'].isnull().sum())

    display("Missing Count in DataFrame:",titanic_train_test.isnull().sum())

    print("Missing Values Phase Ends:")

    return titanic_train_test

def IrrelevantFeatures(titanic_train_test):

    print("Removing Irrelevant Features Process Starts:")

    titanic_train_test.drop(columns=['PassengerId','Name','Ticket'],axis=1,inplace=True)

    print("Removing Irrelevant Features Process Ends:")

    display(titanic_train_test.head(5))

    return titanic_train_test

def Feature_Extraction(titanic_train_test):

    print("Feature Extraction Phase Starts:")

    titanic_train_test['From_Embarked'] = titanic_train_test['Embarked'].apply(lambda x: 1 if x == 'C' else (2 if x == 'Q' else 3))

    titanic_train_test.drop('Embarked',axis=1,inplace=True)

    print("Feature Extraction Phase Ends:")

    display(titanic_train_test.head(5))

    return titanic_train_test

def RedunantData(titanic_train_test):

    print("Redunant Data Phase Starts:")

    print("Both Pclass and From_Embarked are redunant. Removing From_Embarked feature")

    titanic_train_test.drop('From_Embarked',axis=1,inplace=True)

    print("Redunant Data Phase Ends:")

    display(titanic_train_test.head(5))

    return titanic_train_test

def Encoding(titanic_train_test):

    print("Encodeing Phase Starts:")

    titanic_train_test_encode = pd.get_dummies(titanic_train_test,prefix_sep='_')

    display(titanic_train_test_encode)

    print("Encoding Phase Ends")

    return titanic_train_test_encode

def Correlation(titanic_train):

    print("Correlation Starts:")

    display(titanic_train.corr())

    print("Correlation Ends:")

def Model_Selection(titanic_train,titanic_test):

    print("Model Selection Phase Starts:")

    X = titanic_train.drop("Survived",axis=1)

    Y = titanic_train['Survived']

    titanic_test = titanic_test.drop('Survived',axis=1)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=10)

    print("Model Selection Phase Ends:")

    return X_train,X_test,Y_train,Y_test,titanic_test

def LogRegression(X_train,X_test,Y_train,Y_test,titanic_test1,titanic_test):

    print("Logistic Regression Model Starts:")

    algo = "Log"

    log_model = LogisticRegression()

    log_model.fit(X_train,Y_train)

    Y_train_predict = log_model.predict(X_train)

    Y_test_predict = log_model.predict(X_test)

    Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,algo)
def DecisionTree(X_train,X_test,Y_train,Y_test):

    global DSTree_depth,Train_results,Test_results,Samples_split

    print("Decision Tree Model Starts:")

    algo = "DS"

    max_depths = np.linspace(1, 32, 32, endpoint=True)

    min_samples = np.linspace(0.1, 1.0, 10, endpoint=True)

    min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)

    

    for depth in max_depths:

        DSTree_depth.append(depth)

        ds_model = DecisionTreeClassifier(max_depth=depth)

        ds_model.fit(X_train,Y_train)

        Y_train_predict = ds_model.predict(X_train)

        Y_test_predict = ds_model.predict(X_test)

        Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,algo)

    pname = "Tree Depth"

    ROCAUC_Curve(Train_results,Test_results, DSTree_depth,pname)

        

    for sample in min_samples:

        Samples_split.append(sample)

        ds_model = DecisionTreeClassifier(min_samples_split=sample)

        ds_model.fit(X_train,Y_train)

        Y_train_predict = ds_model.predict(X_train)

        Y_test_predict = ds_model.predict(X_test)

        Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,algo)

    

    pname = "Min_Samples_Split"

    ROCAUC_Curve(Train_results,Test_results, Samples_split,pname)

    

    

    for leaf in min_samples_leaf:

        Samples_leaf.append(leaf)

        ds_model = DecisionTreeClassifier(min_samples_leaf=leaf)

        ds_model.fit(X_train,Y_train)

        Y_train_predict = ds_model.predict(X_train)

        Y_test_predict = ds_model.predict(X_test)

        Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,algo)

    

    pname = "Min_Samples_Leaf"

    ROCAUC_Curve(Train_results,Test_results, Samples_leaf,pname)

    

    ds_model = DecisionTreeClassifier(max_depth=4, min_samples_split = 0.2,min_samples_leaf=0.30)

    ds_model.fit(X_train,Y_train)

    Y_train_predict = ds_model.predict(X_train)

    Y_test_predict = ds_model.predict(X_test)

    Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,algo="DSBestFit")

    

    

    print("Decision Tree Model Ends:")

def RandomForest(X_train,X_test,Y_train,Y_test,titanic_test1,titanic_test):

    global RFTree_depth, Samples_split,Samples_leaf

    print("RandomForest Model Starts:")

    algo = "RF"

    max_depths = np.linspace(1, 32, 32, endpoint=True)

    min_samples = np.linspace(0.1, 1.0, 10, endpoint=True)

    min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)

    

    

    for depth in max_depths:

        RFTree_depth.append(depth)

        rf_model = RandomForestClassifier(n_estimators = 200,max_depth=depth)

        rf_model.fit(X_train,Y_train)

        Y_train_predict = rf_model.predict(X_train)

        Y_test_predict = rf_model.predict(X_test)

        Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,algo)

    pname = "Tree Depth"

    ROCAUC_Curve(Train_results,Test_results, RFTree_depth,pname)

        

    for sample in min_samples:

        Samples_split.append(sample)

        rf_model = RandomForestClassifier(n_estimators = 200,min_samples_split=sample)

        rf_model.fit(X_train,Y_train)

        Y_train_predict = rf_model.predict(X_train)

        Y_test_predict = rf_model.predict(X_test)

        Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,algo)

    

    pname = "Min_Samples_Split"

    ROCAUC_Curve(Train_results,Test_results, Samples_split,pname)

    

    

    for leaf in min_samples_leaf:

        Samples_leaf.append(leaf)

        rf_model = RandomForestClassifier(n_estimators = 200,min_samples_leaf=leaf)

        rf_model.fit(X_train,Y_train)

        Y_train_predict = rf_model.predict(X_train)

        Y_test_predict = rf_model.predict(X_test)

        Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,algo)

    

    pname = "Min_Samples_Leaf"

    ROCAUC_Curve(Train_results,Test_results, Samples_leaf,pname)

    

    algo = 'RFBest_Fit'

    rf_model = RandomForestClassifier(n_estimators = 200,min_samples_split=0.6,min_samples_leaf=0.20)

    rf_model.fit(X_train,Y_train)

    Y_train_predict = rf_model.predict(X_train)

    Y_test_predict = rf_model.predict(X_test)

    titanic_test_predict = rf_model.predict(titanic_test1)

    Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,algo)

    print("RandomForest Model Ends:")

    

    display("RanfomForest model better than other models and the final result :")

    my_submission = pd.DataFrame({"PassengerId": titanic_test.PassengerId,"Survived":titanic_test_predict})

    my_submission['Survived'] = my_submission['Survived'].apply(lambda x: int(x))

    my_submission.to_csv('submission.csv',index=False)

    print("Final output:")

    display(my_submission)
def Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,algo):

    global Train_results, Test_results, Tree_depth

    #print("Measuring the model phase starts:")

            

    if algo == 'Log':

        print("Logistic Regression Measures:")

        print("Train dataset accuracy:")

        print(accuracy_score(Y_train,Y_train_predict))

        print(classification_report(Y_train,Y_train_predict))

        print("Test dataset accuracy:")

        print(accuracy_score(Y_test,Y_test_predict))

        print(classification_report(Y_test,Y_test_predict))

    elif algo == 'DS':

        #print("Decision Tree Classifier Measures:")

        #print("Train dataset accuracy:")

        #print(classification_report(Y_train,Y_train_predict))

        #print("Test dataset accuracy:")

        #print(classification_report(Y_test,Y_test_predict))

        FPR_train,TPR_train,threshold_train = roc_curve(Y_train,Y_train_predict)

        roc_auc_train = auc(FPR_train,TPR_train)

        Train_results.append(roc_auc_train)

        

        FPR_test,TPR_test,threshold_test = roc_curve(Y_test,Y_test_predict)

        roc_auc_test = auc(FPR_test,TPR_test)

        Test_results.append(roc_auc_test)

        

    elif algo == 'DSBestFit':

        

        print("Decision Tree Best Fit Measures:")

        print("Train dataset accuracy:")

        print(accuracy_score(Y_train,Y_train_predict))

        print(classification_report(Y_train,Y_train_predict))

        print("Test dataset accuracy:")

        print(accuracy_score(Y_test,Y_test_predict))

        print(classification_report(Y_test,Y_test_predict))

        

        

    elif algo == 'RF':

        

        FPR_train,TPR_train,threshold_train = roc_curve(Y_train,Y_train_predict)

        roc_auc_train = auc(FPR_train,TPR_train)

        Train_results.append(roc_auc_train)

        

        FPR_test,TPR_test,threshold_test = roc_curve(Y_test,Y_test_predict)

        roc_auc_test = auc(FPR_test,TPR_test)

        Test_results.append(roc_auc_test)

        

    else:

        print("Random Forest Classifier Measures:")

        print("Train dataset accuracy:")

        print(accuracy_score(Y_train,Y_train_predict))

        print(classification_report(Y_train,Y_train_predict))

        print("Test dataset accuracy:")

        print(accuracy_score(Y_test,Y_test_predict))

        print(classification_report(Y_test,Y_test_predict))
def ROCAUC_Curve(Train,Test,parameter,pname):

    global Train_results, Test_results, DSTree_depth, Samples_split, Samples_leaf

    train = plt.plot(parameter,Train,'b',label="Train")

    test = plt.plot(parameter,Test,'r',label="Test")

    plt.legend()

    plt.xlabel(pname)

    plt.ylabel("AUC-Score")

    plt.show()

    Train_results = []

    Test_results = []

    DSTree_depth = []

    Samples_split = []

    Samples_leaf = []

Main()