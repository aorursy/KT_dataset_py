# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



#Libraries for Linear regression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,mean_absolute_error



#Library for RFR

from sklearn.ensemble import RandomForestRegressor



#Library for Lasso,Ridge, Elastic

from sklearn.linear_model import Ridge,Lasso,ElasticNet



#Library for Decision Tree

from sklearn import tree



#Library for XGBoost

import xgboost as xgb

from xgboost.sklearn import XGBClassifier



#For doing scaling

from sklearn.preprocessing import StandardScaler

MSE_Train=[]

RMSE_Train=[]

MAE_Train=[]

MSE_Test=[]

RMSE_Test=[]

MAE_Test=[]

Hyper=[]

rfestimator=[]

tree_depth=[]
def ModelFlow1():

    

    CreditScoreData=[]

    CreditTrain=pd.read_csv('/kaggle/input/credit-score-prediction/CreditScore_train.csv')

    

    CreditTest=pd.read_csv('/kaggle/input/credit-score-prediction/CreditScore_test.csv')

    

    #Concatenate both Train & Test

    CreditScoreMerge=pd.concat([CreditTrain,CreditTest],axis=0)

    print(CreditScoreMerge.shape)

    

   

    #for i in CreditScoreMerge.columns:

        #print(i + '-' + str(CreditScoreMerge[i].isnull().sum()))

    
def ModelFlow():

    CreditTrain=pd.read_csv('/kaggle/input/credit-score-prediction/CreditScore_train.csv')

    

    CreditTest=pd.read_csv('/kaggle/input/credit-score-prediction/CreditScore_test.csv')

    

    #Concatenate both Train & Test

    CreditScoreMerge=pd.concat([CreditTrain,CreditTest],axis=0)

    

    print("CreditScoreTrain:",CreditTrain.head())

    print("CreditScoreTest:",CreditTest.head())

    print("CreditScoreMerge:",CreditScoreMerge.head())

    

    #calling the function KnowTheData

    KnowTheData(CreditScoreMerge)

    

    print('******************************************')

    print("Linear Model Starts")

    

    CreditTrainTestLinear=ModelType(CreditScoreMerge,modeltype='Linear')

    

    

    #Splitting from merged Data Train & Test

    CreditTrain1=CreditScoreMerge.head(80000)

    CreditTest1=CreditScoreMerge.tail(20000)

    

    #ModelSelection

    print("Model Selection Starts for Linear...")

    

    

    X_train,X_test,Y_train,Y_test,CreditTest1=ModelSelection(CreditTrain1,CreditTest1)

    

    print("Model Selection Ends for Linear...")

    

    #BuildLinearModel

    LinearReg(X_train,X_test,Y_train,Y_test,CreditTest1)

    

    print("Linear Model Type Build Process ends.....")

    print('**********************************************')

    

    #Add Other Models below

    

    #Build Ridge Model

    RidgeReg(X_train,X_test,Y_train,Y_test,CreditTest1)

    

    #Tree Model

    print('**********************************************')

    print("Tree Model Type Build Process starts.....")

    

    CreditScoreMergeTree = ModelType(CreditScoreMerge,modeltype = 'Tree')

    

    #split the dataset into train and test after data cleaning and select the important features for the model

    CreditTrain1=CreditScoreMergeTree.head(80000)

    CreditTest1=CreditScoreMergeTree.tail(20000)

    

    print("Model Selection Starts for Tree.....")

    

    #Model Selection

    X_train,X_test,Y_train,Y_test,CreditTest1=ModelSelection(CreditTrain1,CreditTest1)

    

    print("Model Selection Ends for Tree.....")

    

    #Build DSRegressor Model

    DSReg(X_train,X_test,Y_train,Y_test,CreditTest1)

    

    #Build RandomForest Regressor Model

    #RandomForestReg(X_train,X_test,Y_train,Y_test,CreditTest1)

    

    #Build XGBoost Regressor Model

    #XGBoost(X_train,X_test,Y_train,Y_test,CreditTest1)

    

    print('**********************************************')

    print("Tree Model Type Build Process ends.....")

def TrainShape(CreditTrain):

    display("Shape of CreditScoreTrain")

    CreditTrain.shape

def TestShape(CreditTest):

    display("Shape of CreditScoreTest")

    CreditTest.shape

def KnowTheData(CreditScoreMerge):

    display("Shape of the Merged Dataframe")

    CreditScoreMerge.shape

    

    CreditScoreData=[]

    CreditScoreData.append(['FeatureName','DataType','No.ofMiss_Values','Percent_of_Miss_Values'])

    for col in CreditScoreMerge.columns:

        CreditScoreData.append([col,CreditScoreMerge[col].dtypes,CreditScoreMerge[col].isnull().sum(),(CreditScoreMerge[col].isnull().sum()/len(CreditScoreMerge)*100)])

        

    display(CreditScoreData)

        

def ModelType(CreditScoreMerge,modeltype):

    if modeltype=='Linear':

        CreditScoreMerge=DataCleaning(CreditScoreMerge,modeltype)

        CreditScoreMerge=Feature_Selection(CreditScoreMerge)

    else:

        CreditScoreMerge=DataCleaning(CreditScoreMerge,modeltype)

   

        

    return CreditScoreMerge



def DataCleaning(CreditScoreMerge,modeltype):

    DuplicatedData(CreditScoreMerge)

    if modeltype=="Linear":

        print("Data Cleaning Process starts for Linear..........")

        CreditScoreMerge=DataForImputation(CreditScoreMerge,modeltype)

        CreditScoreMerge=Imputation(CreditScoreMerge,modeltype)

        print("DataCleaning Process Ends For Linear.....")

    else:

        print("DataCleaning Process Starts For Tree.....")

        CreditScoreMerge = Imputation(CreditScoreMerge,modeltype)

        print("DataCleaning Process Ends For Tree.....")

    

    

    return CreditScoreMerge
def DuplicatedData(CreditScoreMerge):

    display("Finding Duplicated Data....")

    display("Total number of Duplicated Records:", CreditScoreMerge.duplicated().sum())

    
def DataForImputation(CreditScoreMerge,modeltype):

    Feature_For_Model=[]

    Missing_Count=0.0

    

    if modeltype == "Linear":

        print('DataAnalysis for Imputation starts for Linear...')

        print("Drop the features whose null % greater than or equal to 50% of Missing Values:")

        #print('Before Err')

        for col in CreditScoreMerge.columns:

            Missing_Count=((CreditScoreMerge[col].isnull().sum()/len(CreditScoreMerge))*100)

            #print('Before IF')

            if Missing_Count < 50:

                Feature_For_Model.append(col)

            else:

                CreditScoreMerge.drop(col,axis=1,inplace=True)



    #print("After FOR")

    Feature_For_Model

    display(np.array(Feature_For_Model).T)

    

    print("DataAnalysis For Imputation Ends For Linear...")

    

    return CreditScoreMerge

    
#ImputeMissingValues



def Imputation(CreditScoreMerge,modeltype):

    if modeltype == 'Linear':

        print("Imputation starts for Linear...")

        for col in CreditScoreMerge.columns:

            CreditScoreMerge[col].fillna(CreditScoreMerge[col].mean(),inplace=True)

            print("Shape of the Dataframe after Imputation...")

            display(CreditScoreMerge.shape)

            

        print("Imputation ends for Linear.....")

    else:

        print("Imputation starts for Tree.....")

        display(CreditScoreMerge.shape)

        CreditScoreMerge.fillna(0,inplace=True)

        print("Imputation ends for Tree.....")



    

    return CreditScoreMerge
def Feature_Selection(CreditScoreMerge):

    print("Identifying the relationship of independent and dependent variables for Linear")

    CreditScoreCorr=CreditScoreMerge.corr(method='pearson')[['y']].T

    CreditScoreCorr=CreditScoreCorr[CreditScoreCorr>0.3]

    ImportantFeatures=[]

    for col in CreditScoreCorr.columns:

        if CreditScoreCorr[col].isnull()[0]!=True:

            ImportantFeatures.append(col)

    

    print("Important Features from Correlation for Linear")

    display(ImportantFeatures)

    

   

    

    return CreditScoreMerge[ImportantFeatures]

        

def ModelSelection(CreditTrain1,CreditTest1):

    print("Model Selection Starts..")

    X=CreditTrain1.drop('y',axis=1)

    Y=CreditTrain1['y']

    

    CreditTest1.drop('y',axis=1,inplace=True)

    

    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)

    

    

    print("X_train shape:", X_train.shape)

    print("X_test shape:",X_test.shape)

    print("Y_train shape:",Y_train.shape)

    print("Y_test shape:",Y_test.shape)

    

    print("Model Selection Ends.....")

    

    return X_train,X_test,Y_train,Y_test,CreditTest1

    
def LinearReg(X_train,X_test,Y_train,Y_test,CreditTest1):

    algo="Linear"

    model=LinearRegression()

    model.fit(X_train,Y_train)

    print("Linear Regression...")

    

    Y_train_predict=model.predict(X_train)

    Y_test_predict=model.predict(X_test)

    

    Y_Credit_test_predict=model.predict(CreditTest1)

    print("*********Linear Regression Measures************")

    

    Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,Y_Credit_test_predict)

    

    LinearAccuracy=pd.DataFrame({"MSE-Train":MSE_Train,"MSE-Test":MSE_Test,"RMSE-Train":RMSE_Train,"RMSE-Test":RMSE_Test,"MAE-Train":MAE_Train,"MAE-Test":MAE_Test})

    ModelPerformance(algo,LinearAccuracy)

    
def RidgeReg(X_train,X_test,Y_train,Y_test,Credit_Test):

    global Hyper

    algo='Ridge'

    print("Ridgeregression:")

    print("********Ridge Regression Measures*************")

    

    for alpha in range(0,10):

        model=Ridge(alpha,normalize=True)

        Hyper.append(alpha)

        model.fit(X_train,Y_train)

        Y_train_predict=model.predict(X_train)

        Y_test_predict=model.predict(X_test)

        Y_Credit_test_predict=model.predict(Credit_Test)

        Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,Y_Credit_test_predict)

        

    Ridge_optimize=pd.DataFrame({"Penalty":Hyper,"MSE-Train":MSE_Train,"MSE-Test":MSE_Test,"RMSE-Train":RMSE_Train,"RMSE-Test":RMSE_Test,"MAE-Train":MAE_Train,"MAE-Test":MAE_Test})

    ModelPerformance(algo, Ridge_optimize)

def DSReg(X_train,X_test,Y_train,Y_test,CreditTest1):

    global Hyper

    algo='ds'

    print('Decision Tree')

    print("***********Decision Tree Regression Measures*****************")

    for depth in range(1,20):

        dsmodel=tree.DecisionTreeRegressor(max_depth=depth)

        Hyper.append(depth)

        dsmodel.fit(X_train,Y_train)

        Y_train_predict=dsmodel.predict(X_train)

        Y_test_predict=dsmodel.predict(X_test)

        Y_Credit_test_predict = dsmodel.predict(CreditTest1)

        Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,Y_Credit_test_predict)

    

    ds_optimize = pd.DataFrame({"TreeDepth":Hyper,"MSE-Train":MSE_Train,"MSE-Test":MSE_Test,"RMSE-Train":RMSE_Train,"RMSE-Test":RMSE_Test,"MAE-Train":MAE_Train,"MAE-Test":MAE_Test})

    

    ModelPerformance(algo,ds_optimize)
def RandomForestReg(X_train,X_test,Y_train,Y_test,CreditTest1):

    global rfestimator, tree_depth

    algo="RandomForest"

    print("**************Random Forest Regressor*****************")

    

    for estimator in range(10,21):

        for depth in range(1,10):

            rfestimator.append(estimator)

            tree_depth.append(depth)

            random_model=RandomForestRegressor(n_estimators=estimator,max_depth=depth,random_state=0)

            random_model.fit(X_train,Y_train)

            

            Y_train_predict=random_model.predict(X_train)

            Y_test_predict=random_model.predict(X_test)

            Y_Credit_test_predict=random_model.predict(CreditTest1)

            

            Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,Y_Credit_test_predict)

            

    random_accuracy = pd.DataFrame({"No of Trees":rfestimator,"Tree Depth": tree_depth,"MSE-Train":MSE_Train,"MSE-Test":MSE_Test,"RMSE-Train":RMSE_Train,"RMSE-Test":RMSE_Test,"MAE-Train":MAE_Train,"MAE-Test":MAE_Test})

    

    ModelPerformance(algo,random_accuracy)

    rfestimator = []

    tree_depth = []

            
def XGBoost(X_train,X_test,Y_train,Y_test,CreditTest1):

    print("**********XGBoost**************")

    global rfestimator, tree_depth

    algo = "XGBoost"

    depth=9

    

    for estimator in range(100,1001,100):

        rfestimator.append(estimator)

        tree_depth.append(depth)

        xgb_model=xgb.XGBRegressor(n_estimators=estimator,learning_rate=0.1,subsample=0.75,colsample_bytree=1,max_depth=depth,random_state=10,gamma=1)

        xgb_model.fit(X_train,Y_train)

        Y_train_predict=xgb_model.predict(X_train)

        Y_test_predict=xgb_model.predict(X_test)

        Y_Credit_test_predict=xgb_model.predict(CreditTest1)

        

        Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,Y_Credit_test_predict)

    xgb_accuracy = pd.DataFrame({"No of Trees":rfestimator,"Tree Depth": tree_depth,"MSE-Train":MSE_Train,"MSE-Test":MSE_Test,"RMSE-Train":RMSE_Train,"RMSE-Test":RMSE_Test,"MAE-Train":MAE_Train,"MAE-Test":MAE_Test})

        

    ModelPerformance(algo,xgb_accuracy)

    rfestimator = []

    tree_depth = []

    
def Measures(Y_train,Y_train_predict,Y_test,Y_test_predict,Y_Credit_test_predict):

    global MSE_Train,RMSE_Train,MAE_Train,MSE_Test,RMSE_Test,MAE_Test,Hyper

    

    MSE=mean_squared_error(Y_train,Y_train_predict)

    RMSE=np.sqrt(mean_squared_error(Y_train,Y_train_predict))

    MAE=mean_absolute_error(Y_train,Y_train_predict)

    

    MSE_Train.append(MSE)

    RMSE_Train.append(RMSE)

    MAE_Train.append(MAE)

    

    MSE = mean_squared_error(Y_test,Y_test_predict)

    RMSE = np.sqrt(mean_squared_error(Y_test,Y_test_predict))

    MAE = mean_absolute_error(Y_test,Y_test_predict)

    

    MSE_Test.append(MSE)

    RMSE_Test.append(RMSE)

    MAE_Test.append(MAE)

    

    

    
def ModelPerformance(algo,performance):

    print("Performance of the model:", algo)

    global Hyper, MSE_Train, RMSE_Train,MAE_Train,MSE_Test, RMSE_Test,MAE_Test

    

    display(performance)

    if algo=='XGBoost':

        plt.figure(figsize=(10,5))

        sns.lineplot(x=performance["No of Trees"],y=performance["MSE-Train"])

        sns.lineplot(x=performance["No of Trees"],y=performance["MSE-Test"])

        plt.xlabel("No of Trees")

        plt.ylabel("MSE-Error")

        

        plt.show()

        

    Hyper = []

    MSE_Train = []

    RMSE_Train = []

    MAE_Train = []

    MSE_Test = []

    RMSE_Test = []

    MAE_Test = []

def Measures1(Y_train,Y_train_predict,Y_test,Y_test_predict,Y_Credit_test_predict,algo):

    if algo=="Linear":

        print("MSE Train:",mean_squared_error(Y_train,Y_train_predict))

        print("RMSE Train:", np.sqrt(mean_squared_error(Y_train,Y_train_predict)))

        print("MAE Train:",mean_absolute_error(Y_train,Y_train_predict))

        print("********************************************************")

        print("MSE Test:",mean_squared_error(Y_test,Y_test_predict))

        print("RMSE Test:",np.sqrt(mean_squared_error(Y_test,Y_test_predict)))

        print("MAE Test:",mean_absolute_error(Y_test,Y_test_predict))

        print("********************************************************")

        print("Credit Score Test Final Result")

        print(Y_Credit_test_predict)

    

    elif algo=='Random':

        rfr = RandomForestRegressor(n_jobs=-1)

        estimators=np.arange(10,200,10)

        scores=[]

        for n in estimators:

            rfr,set_params(n_estimators=n)

            rfr.fit(X_train,Y_train)

            rfr.append(rfr.score(X_test,Y_test))

        plt.title("Effect of n_estimators")

        plt.xlabel("n_estimator")

        plt.ylabel("score")

        plt.plot(estimators, scores)

        
ModelFlow()