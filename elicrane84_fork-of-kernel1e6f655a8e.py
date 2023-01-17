import pandas as pd

import numpy as np

import random

import math



from matplotlib.ticker import NullFormatter

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation



import itertools
def plot_confusion_matrix(cm, classes,

                              normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve



def evaluate_model(y_test,predictions, probs):

    """Compare machine learning model to baseline performance.

    Computes statistics and shows ROC curve."""

    

    baseline = {}

    

    baseline['recall'] = recall_score(y_test, [1 for _ in range(len(y_test))])

    baseline['precision'] = precision_score(y_test, [1 for _ in range(len(y_test))])

    baseline['roc'] = 0.5

    

    results = {}

    

    results['recall'] = recall_score(y_test, predictions)

    results['precision'] = precision_score(y_test, predictions)

    results['roc'] = roc_auc_score(y_test, probs)

    

    # Calculate false positive rates and true positive rates

    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])

    model_fpr, model_tpr, _ = roc_curve(y_test, probs)



    plt.figure(figsize = (4, 3))

    plt.rcParams['font.size'] = 10

    

    # Plot both curves

    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')

    plt.plot(model_fpr, model_tpr, 'r', label = 'model')

    plt.legend();

    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
def predictAndCreateConfusionMatrix(model,x_data,y_true,y_pred,y_probs,addTitle=''):

    

    #Confusion Matrix

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=[1,0])

    np.set_printoptions(precision=2)



    # Plot non-normalized confusion matrix

    plt.figure(figsize = (6,4))

    plot_confusion_matrix(cnf_matrix, classes=['1','0'],normalize= False,  title='Confusion Matrix: %s %s' % (type(model).__name__,addTitle))

    plt.show()

    

    #Create ROC

    AUC_rf = roc_auc_score(y_true, y_probs)

    print("AUC for %s %s:" % (type(model).__name__,addTitle))

    print(AUC_rf)

    evaluate_model(y_true,y_pred,y_probs)

    
def cleanDataset(dfOrig, lsContVars, missingStrategy = 'DropNA', dropAge = 0, dropOutliers_low = .00, dropOutliers_high = 1.0):

    print("\nPrior to cleaning the original dataframe contains %d rows" % len(dfOrig.index))

    

    #Minimum Age

    if dropAge is not None:

        lsIndexesAge = list(dfOrig[dfOrig['age'] <= dropAge].index)

        print("There are %d indexes deleted for having an age of %d or less." % (len(lsIndexesAge),dropAge))

        

    #Drop Missing

    lsMissingIndex = []

    if missingStrategy:

        lenBeforeMissing = len(dfOrig.index)

        lsMissingDataHandling = missingDataHandling(dfOrig, missingStrategy, lsContVars)

        dfOrig = lsMissingDataHandling[0]

        lsMissingIndex = lsMissingDataHandling[1]

        print("There are %d indexes deleted for having missing variables." % (lenBeforeMissing-len(dfOrig.index)))



    #Clean Dataset for Outlier Detection\

    dfClean = dfOrig.loc[~dfOrig.index.isin(lsIndexesAge)]

    lsIndexesClean = dfClean.index

    print("The clean dataset now has %d rows before deleting outliers." % len(dfClean.index))



    #Detect and store Outlier indexes

    dfQuantile = dfClean.quantile([dropOutliers_low,dropOutliers_high])

    

    dfClean = dfClean.apply(lambda x: x[(x >= dfQuantile.loc[dropOutliers_low,x.name]) &

                            (x <= dfQuantile.loc[dropOutliers_high,x.name])], axis = 0).dropna()

    lsIndexesNonOutliers = dfClean.index

    print("The clean dataset now has %d rows after deleting outliers.\n" % len(dfClean.index))

    

    lsIndexesOutliers = [ix for ix in lsIndexesClean if ix not in lsIndexesNonOutliers]

    

    return {'dfClean':dfClean,'lsIndexesAge':lsIndexesAge,'lsMissingIndex':lsMissingIndex,'lsIndexesOutliers':lsIndexesOutliers}
def missingDataHandling(dfOrig, missingStrategy, lsColImpute = []):



    if len(lsColImpute) == 0:

        lsColImpute = dfOrig.columns



    if missingStrategy == "DropNA":

        lsOrigIndex = list(dfOrig.index)

        dfDropNA = dfOrig.dropna()

        lsDropNAIndex = list(dfDropNA.index)

        lsMissingIndex = list(set(lsOrigIndex) - set(lsDropNAIndex))

        

        return [dfDropNA, lsMissingIndex]



    if missingStrategy == "Mean" or missingStrategy == "Mode":

        dfDescribe = dfOrig.describe()



        for col in lsColImpute:



            valImpute = dfDescribe.loc[missingStrategy.lower(), col]



            lsMissingIndex = pd.isna(dfOrig[col])



            dfOrig.loc[lsMissingIndex,col] = valImpute



    return [dfOrig, []]
def createVariables(dfModel, blExtraCont=True, blDiscrete=True, blInteraction=True):



    #Continuous

    if blExtraCont:

        dfModel['NumberOfNonRealEstateCreditLines'] = dfModel['NumberOfOpenCreditLinesAndLoans']  - dfModel['NumberRealEstateLoansOrLines']

        dfModel['TotalDebtIncome'] = dfModel['DebtRatio'] * dfModel['MonthlyIncome']

        dfModel['NumberOfTimesPastDue'] = dfModel['NumberOfTime30-59DaysPastDueNotWorse'] + dfModel['NumberOfTime60-89DaysPastDueNotWorse'] + dfModel['NumberOfTimes90DaysLate']

    

    #Discretize Age

    if blDiscrete:

        dfModel.loc[:,'disc_age'] = '<29'



        dfModel.loc[:,'age_21_29'] = 0

        lsIx = dfModel.loc[(dfModel['age'] <= 29)].index

        dfModel.loc[lsIx,'age_21_29'] = 1



        dfModel.loc[:,'age_30_69'] = 0

        lsIx = dfModel.loc[(dfModel['age'] >= 30) & (dfModel['age']<= 69)].index

        dfModel.loc[lsIx,'age_30_69'] = 1

        dfModel.loc[lsIx,'disc_age'] = '30_69'



        dfModel.loc[:,'age_70_'] = 0

        lsIx = dfModel.loc[(dfModel['age'] >= 70)].index

        dfModel.loc[lsIx,'age_70_'] = 1

        dfModel.loc[lsIx,'disc_age'] = '>70'



        #Number Credit Lines

        dfModel.loc[:,'disc_numcreditlines'] = '<4'

        lsIx = dfModel.loc[(dfModel['NumberOfOpenCreditLinesAndLoans'] >=4)].index

        dfModel.loc[lsIx,'disc_numcreditlines'] = '4+'



        #Income

        dfModel.loc[:,'disc_income'] = '<1000'



        dfModel.loc[:,'income_0_1999'] = 0

        lsIx = dfModel.loc[(dfModel['MonthlyIncome'] <= 1999)].index

        dfModel.loc[lsIx,'income_0_1999'] = 1



        dfModel.loc[:,'income_1999_13000'] = 0

        lsIx = dfModel.loc[(dfModel['MonthlyIncome'] > 1999) & (dfModel['MonthlyIncome']<= 13000)].index

        dfModel.loc[lsIx,'income_1999_13000'] = 1

        dfModel.loc[lsIx,'disc_income'] = '1999_13000'



        dfModel.loc[:,'income_13000_'] = 0

        lsIx = dfModel.loc[(dfModel['MonthlyIncome'] > 13000)].index

        dfModel.loc[lsIx,'income_13000_'] = 1

        dfModel.loc[lsIx,'disc_income'] = '>13000'

        

        dfModel.loc[:,'DRgt0_MIeq0'] = 0

        dfModel.loc[(dfModel['DebtRatio'] > 0) & (dfModel['MonthlyIncome'] == 0),'DRgt0_MIeq0'] = 1

        

        dfModel.loc[:,'NREgt0_MIeq0'] = 0

        dfModel.loc[(dfModel['NumberRealEstateLoansOrLines'] == dfModel['NumberOfOpenCreditLinesAndLoans']),'NREgt0_MIeq0'] = 1

        

        dfModel.loc[:,'RUeq0_NREeq0_DRgt0'] = 0

        dfModel.loc[(dfModel['RevolvingUtilizationOfUnsecuredLines'] == 0) & (dfModel['NumberRealEstateLoansOrLines'] == 0) & (dfModel['DebtRatio'] > 0),'RUeq0_NREeq0_DRgt0'] = 1

    

    #Interaction

    if blInteraction:

        dfModel['irc_age_21_29'] = dfModel['age_21_29'] * dfModel['age']

        dfModel['irc_age_30_69'] = dfModel['age_30_69'] * dfModel['age']

        dfModel['irc_age_70_'] = dfModel['age_70_'] * dfModel['age']



        dfModel['irc_income_0_1999'] = dfModel['income_0_1999'] * dfModel['MonthlyIncome']

        dfModel['irc_income_1999_13000'] = dfModel['income_1999_13000'] * dfModel['MonthlyIncome']

    

    print(dfModel.columns)

    print(dfModel.dtypes)

    

    return dfModel
def transformData(dfModel,blNormalizeCont,x_var_cont,x_var_disc=[],x_var_intc=[]):

    print("Transforming data...")

    #To Preserve Index

    dfModel['Index'] = dfModel.index

    #Continuous

    x_cont = np.asarray(dfModel[x_var_cont])

    if blNormalizeCont:

        x_cont = preprocessing.StandardScaler().fit(x_cont).transform(x_cont)



    #Add to full variable

    x_full = x_cont#np.concatenate((x_full,x_cont),axis=1)

    

    #Add discrete

    if len(x_var_disc) > 0:

        x_disc = np.asarray(dfModel[x_var_disc])

        x_full = np.concatenate((x_full,x_disc),axis=1)

    

    #Interaction

    if len(x_var_intc) > 0:

        x_intc = np.asarray(dfModel[x_var_intc])

        x_full = np.concatenate((x_full,x_intc),axis=1)

    

    print("arrays:")

    print(type(x_full))

    print(type(x_full[1]))

    

    #np.concatenate((x_full,np.asarray(dfModel['Index']))

    print(x_full)

    return x_full
def createRandomDataFrameSplit(dfSplit, leftSplit=.5, randomSeed=4):

    

    lsIndex = list(dfSplit.index)

    

    #Shuffle

    random.seed(randomSeed)

    random.shuffle(lsIndex)



    ixLeft = math.ceil(len(lsIndex)*(1-leftSplit))

    

    lsIndexLeft = lsIndex[:ixLeft+1]

    lsIndexRight = lsIndex[ixLeft+1:]

    

    return {'dfLeft':dfSplit.loc[lsIndexLeft],'dfRight':dfSplit.loc[lsIndexRight]}
def returnTrainTestSplit(dfModel, y_var, blNormalizeCont, x_var_cont, x_var_disc, x_var_intc, testSplit=.2, randomSeed=4):

    lsRegressors = x_var_cont + x_var_disc + x_var_intc

    

    dictTrainTest = createRandomDataFrameSplit(dfModel,testSplit,randomSeed)

    dfTrain = dictTrainTest['dfLeft']

    dfTest = dictTrainTest['dfRight']

    

    x_train = transformData(dfTrain.loc[:,lsRegressors],blNormalizeCont,x_var_cont,x_var_disc,x_var_intc)

    y_train = np.asarray(dfTrain[y_var])

    x_test = transformData(dfTest.loc[:,lsRegressors],blNormalizeCont,x_var_cont,x_var_disc,x_var_intc)

    y_test = np.asarray(dfTest[y_var])

    

    return x_train, x_test, y_train, y_test, dfTrain.index, dfTest.index
def runModels(dfModel, y_var, x_var_cont, x_var_disc, x_var_intc, blNormalizeCont, test_split, randomSeed, blRunLogistic, blRunLogisticCV, blRunDecisionTree, blRunRandomForest, numTrees, blRunCV):

    #Create arrays and define regressors

    lsRegressors = x_var_cont + x_var_disc + x_var_intc

    

    dictModels = {}

    

    #Define train and test split

    x_train, x_test, y_train, y_test, x_train_index, x_test_index = returnTrainTestSplit(dfModel, y_var, blNormalizeCont, x_var_cont, x_var_disc, x_var_intc, test_split, randomSeed)

    print ('Train set:', x_train.shape,  y_train.shape)

    print ('Test set:', x_test.shape,  y_test.shape)

    

    #Define Display Function

    def displayModel(model):

        nameModel = type(model).__name__

        print("\nDisplaying results for model %s:\n" % nameModel)

        

        y_pred_train = model.predict(x_train)

        y_probs_train = model.predict_proba(x_train)[:,1]

        y_pred_test = model.predict(x_test)

        y_probs_test = model.predict_proba(x_test)[:,1]

        

        predictAndCreateConfusionMatrix(model,x_train,y_train,y_pred_train,y_probs_train,'Train')

        predictAndCreateConfusionMatrix(model,x_test,y_test,y_pred_test,y_probs_test,'Test')

        

        dictModels[nameModel] = {'model': model,'pred_train_split' : y_pred_train, 'pred_test_split' : y_pred_test}

        print('')

    

    #Run Logistic

    if blRunLogistic:

        model_LR = LogisticRegression(C=.01,solver='liblinear').fit(x_train,y_train)

        print(model_LR)

        from collections import OrderedDict

        dictCoeff = OrderedDict()

        inc = 0

        for coef in lsRegressors:

            dictCoeff[coef] = model_LR.coef_[0][inc]/(1-model_LR.coef_[0][inc])

            print(coef + ': ' + str(dictCoeff[coef]))

            inc+=1

        

        displayModel(model_LR)

    

    if blRunLogisticCV:

        model_LRCV = LogisticRegressionCV(cv=10).fit(x_train,y_train)

        

        displayModel(model_LRCV)

 

    #Run Decision Tree Classifier

    if blRunDecisionTree:

        model_clf = DecisionTreeClassifier()

        model_clf = model_clf.fit(x_train,y_train)

        

        displayModel(model_clf)

    

    #Run Random Forest

    if blRunRandomForest:

        # Create the model with 100 trees

        model_RF = RandomForestClassifier(n_estimators=numTrees, 

                                   bootstrap = True,

                                   max_features = 'sqrt',random_state = randomSeed)

        # Fit on training data

        model_RF.fit(x_train,y_train)

        

        #Create Confusion Matrices

        displayModel(model_RF)



    if blRunCV:

        # Hyperparameter grid

        param_grid = {

            'n_estimators': np.linspace(10, 200).astype(int),

            'max_depth': [None] + list(np.linspace(3, 20).astype(int)),

            'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),

            'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),

            'min_samples_split': [2, 5, 10],

            'bootstrap': [True, False]

        }



        # Estimator for use in random search

        estimator = RandomForestClassifier()



        # Create the random search model

        model_CV = RandomizedSearchCV(estimator, param_grid, n_jobs = -1, 

                                scoring = 'roc_auc', cv = 3, 

                                n_iter = 10, verbose = 1, random_state=randomSeed)



        # Fit 

        model_CV.fit(x_train, y_train)

        

        #Params and Best Model

        model_CV.best_params_

        model_CV_best = model_CV.best_estimator_

        

        displayModel(model_CV_best)

    

    dictRunModels = {}

    dictRunModels['dictModels'] = dictModels

    dictRunModels['train_index'] = x_train_index

    dictRunModels['test_index'] = x_test_index

    

    print("Models returned:")

    print(dictRunModels['dictModels'].keys())

    return dictRunModels

def createResultFrame(model, cs_test, x_test, colID, lsMissingIndex):

   

    #Predict

    predics_test = model.predict(x_test)

    probs_test = model.predict_proba(x_test)

    print("There were %d probabilities generated." % len(probs_test))



    #Create Result Frame

    namePredCol = colID +'_pred'

    nameProbCol = colID +'_prob'

    dfPred = pd.DataFrame(index=cs_test.index,columns=[namePredCol])                         

    #dfResult[namePredCol] = 1

    dfPred.loc[~dfPred.index.isin(lsMissingIndex),namePredCol] = predics_test

    dfPred.loc[~dfPred.index.isin(lsMissingIndex),nameProbCol] = probs_test[:,1]

    dfPred[nameProbCol] = dfPred[nameProbCol].astype(float)

    dfPred[namePredCol] = dfPred[namePredCol].astype(float)



    return dfPred
def cleanTestDataSet(cs_test, x_var_cont, x_var_disc, x_var_intc, colID, missingStrategy, blNormalizeCont, outlier_low, outlier_high):

    print("Cleaning test data set for predictions....")

        

    #Drop ID

    cs_test = cs_test.drop([colID], axis=1)

    

    #Get Clean Dict

    dictClean = cleanDataset(cs_test, x_var_cont, missingStrategy, 0, outlier_low, outlier_high)

    dfClean = dictClean['dfClean']

    

    lsMissingIndex = []

    for nameMissing in dictClean.keys():

        if nameMissing != 'dfClean':

            lsMissingIndex += dictClean[nameMissing]

    lsMissingIndex = list(set(lsMissingIndex))

    print("Length Missing: %d/n" % len(lsMissingIndex))

    

    #Create Variables

    dfModel = createVariables(dfClean)

    #print(dfModel.dtypes)

    #print(dfModel.describe())

    

    x_test = transformData(dfModel, blNormalizeCont, x_var_cont, x_var_disc, x_var_intc)

    

    return [x_test,lsMissingIndex]

    
def runProcess(cs_training, cs_test, y_col='SeriousDlqin2yrs',

               x_var_cont=[], x_var_disc=[], x_var_intc=[], missingStrategy = 'DropNA', blNormalizeCont=True, 

               test_split = .2, randomSeed = 4, outlier_low = .02, outlier_high = .98,

               blRunLogistic=True, blRunLogisticCV=True, blRunDecisionTree=True, blRunRandomForest=True, numTrees=100, blRunCV=True):

    

    #Get Clean Dict

    dictClean = cleanDataset(cs_training, x_var_cont, missingStrategy, 0, outlier_low, outlier_high)

    dfClean = dictClean['dfClean']

    

    #Create Variables

    dfModel = createVariables(dfClean)

    

    dictRunModels = runModels(dfModel, y_col, x_var_cont, x_var_disc, x_var_intc, blNormalizeCont, 

                           test_split, randomSeed, blRunLogistic, blRunLogisticCV, blRunDecisionTree, blRunRandomForest, numTrees, blRunCV)

    

    #Create test data set and run predictions

    lsCleanDataItems = cleanTestDataSet(cs_test, x_var_cont, x_var_disc, x_var_intc, y_col, missingStrategy, blNormalizeCont, outlier_low, outlier_high)

    x_test = lsCleanDataItems[0]

    lsMissingIndex = lsCleanDataItems[1]

    

    dictResults = {}

    for model in dictRunModels['dictModels'].keys():

        dfPredict = createResultFrame(dictRunModels['dictModels'][model]['model'],cs_test,x_test,y_col,lsMissingIndex)#x_var_cont,x_var_disc,x_var_intc,

        dfResult = cs_test.merge(dfPredict,how='left',left_index=True,right_index=True)

        dictResults[model] = dfResult

    

    lsIndexTrain = dictRunModels['train_index']

    return {'dictRunModels':dictRunModels,'dictResults':dictResults,'dfTrain':dfModel.loc[lsIndexTrain,:],'dfTrainTest':dfModel.loc[~dfModel.index.isin(lsIndexTrain),:]}
#Run everything before here

cs_test = pd.read_csv("../input/give-me-some-credit-dataset/cs-test.csv")

cs_training = pd.read_csv("../input/give-me-some-credit-dataset/cs-training.csv")

sampleEntry = pd.read_csv("../input/give-me-some-credit-dataset/sampleEntry.csv")

cs_training.describe()
#Define Regressors

lsContinuous = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',

                         'MonthlyIncome','NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',#'DebtRatio',

                         'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',

                         'NumberOfDependents']

lsExtraContinuous = ['NumberOfNonRealEstateCreditLines','TotalDebtIncome','NumberOfTimesPastDue']

lsInteraction = ['irc_age_21_29','irc_age_30_69','irc_age_70_','irc_income_0_1999','irc_income_1999_13000']

lsDiscrete = ['DRgt0_MIeq0','NREgt0_MIeq0']#'age_21_29','age_70_','income_0_1999','income_13000_']
cs_test_debtratio0_monthinc0 = cs_test.loc[(cs_training['DebtRatio'] > 0) & (cs_training['MonthlyIncome'] == 0),]

cs_training_debtratio0_monthinc0 = cs_training.loc[(cs_training['DebtRatio'] > 0) & (cs_training['MonthlyIncome'] == 0),]

dictProcess_debtratio0_monthinc0 = runProcess(cs_training_debtratio0_monthinc0, cs_test_debtratio0_monthinc0, x_var_cont=lsContinuous, missingStrategy = "Mean", outlier_low = .005, outlier_high = .995, blRunDecisionTree=False, blRunRandomForest=False, blRunCV=False)
cs_test_NumRealeqNumOpenCredit = cs_test.loc[(cs_test['NumberRealEstateLoansOrLines'] == cs_test['NumberOfOpenCreditLinesAndLoans']),]

cs_training_NumRealeqNumOpenCredit = cs_training.loc[(cs_training['NumberRealEstateLoansOrLines'] == cs_training['NumberOfOpenCreditLinesAndLoans']),]

dictProcess_NumRealeqNumOpenCredit = runProcess(cs_training_NumRealeqNumOpenCredit, cs_test_NumRealeqNumOpenCredit, x_var_cont=lsContinuous, missingStrategy = "Mean", outlier_low = .005, outlier_high = .995, blRunDecisionTree=False, blRunRandomForest=False, blRunCV=False)
cs_test_NumRevolveeqNumReal = cs_test.loc[(cs_test['RevolvingUtilizationOfUnsecuredLines'] == 0) & (cs_test['NumberRealEstateLoansOrLines'] == 0) & (cs_test['DebtRatio'] > 0),]

cs_training_NumRevolveeqNumReal = cs_training.loc[(cs_training['RevolvingUtilizationOfUnsecuredLines'] == 0) & (cs_training['NumberRealEstateLoansOrLines'] == 0) & (cs_training['DebtRatio'] > 0),]

dictProcess_NumRevolveeqNumReal = runProcess(cs_training_NumRevolveeqNumReal, cs_test_NumRevolveeqNumReal, x_var_cont=lsContinuous, missingStrategy = "Mean", outlier_low = .005, outlier_high = .995, blRunDecisionTree=False, blRunRandomForest=False, blRunCV=False)
cs_training.loc[(cs_training['DebtRatio'] > 0) & (cs_training['MonthlyIncome'] == 0)].describe()
cs_training.loc[(cs_training['DebtRatio'] >= 5000)].describe()
cs_training.loc[(cs_training['NumberRealEstateLoansOrLines'] == cs_training['NumberOfOpenCreditLinesAndLoans'])].describe()
cs_training.loc[ (cs_training['NumberRealEstateLoansOrLines'] == cs_training['NumberOfOpenCreditLinesAndLoans'])  & (cs_training['RevolvingUtilizationOfUnsecuredLines'] > 0)].describe()
cs_training.loc[(cs_training['RevolvingUtilizationOfUnsecuredLines'] == 0) & (cs_training['NumberRealEstateLoansOrLines'] == 0) & (cs_training['DebtRatio'] > 0)].describe()
dictProcess = runProcess(cs_training, cs_test, x_var_cont=lsContinuous, missingStrategy = "Mean", outlier_low = .002, outlier_high = .998, blRunDecisionTree=False, blRunRandomForest=False, blRunCV=False)
dictProcess = runProcess(cs_training, cs_test, x_var_cont=lsContinuous, x_var_disc=lsDiscrete, missingStrategy = "Mean", outlier_low = .002, outlier_high = .998, blRunDecisionTree=False, blRunRandomForest=False, blRunCV=False)
dictProcess.keys()

dfTrain = dictProcess['dfTrain']

dfTrainTest = dictProcess['dfTrainTest']

lsLogisticPreds_train = dictProcess['dictRunModels']['dictModels']['LogisticRegression']['pred_train_split']

lsLogisticPreds_traintest = dictProcess['dictRunModels']['dictModels']['LogisticRegression']['pred_test_split']

dfTrain['SeriousDlqin2yrs_pred'] = lsLogisticPreds_train

dfTrainTest['SeriousDlqin2yrs_pred'] = lsLogisticPreds_traintest

dfTrain_Disagree = dfTrain.loc[(dfTrain['SeriousDlqin2yrs'] != dfTrain['SeriousDlqin2yrs_pred'])]

dfTrainTest_Disagree = dfTrainTest.loc[(dfTrainTest['SeriousDlqin2yrs'] != dfTrainTest['SeriousDlqin2yrs_pred'])]
dfTrain.describe()
dfTrain_Disagree.describe()

#dfTrain_Disagree.iloc[:50,]
dfTrain_Disagree.loc[(dfTrain['SeriousDlqin2yrs']==1)].describe()
dfTrain_Disagree.loc[(dfTrain['SeriousDlqin2yrs']==0)].describe()

#dfTrain_Disagree.loc[(dfTrain['SeriousDlqin2yrs']==0)].iloc[:50,]
dfTrainTest.describe()
dfTrainTest_Disagree.describe()
dfResults_Logistic = dictProcess['dictResults']['LogisticRegression']

dfResults_Logistic.describe()
dfResults_Logistic.loc[(pd.isnull(dfResults_Logistic['SeriousDlqin2yrs_pred'])),'SeriousDlqin2yrs_prob'] = .5
dfSubmit = pd.DataFrame(columns=['Id','Probability'],index=dfResults_Logistic.index)



dfSubmit['Id'] = [item for item in range(1,len(cs_test.index)+1)]

dfSubmit['Probability'] = dfResults_Logistic['SeriousDlqin2yrs_prob']



dfSubmit.iloc[2000:5000]

dfSubmit.to_csv('GiveMeCredit_Sub3.csv',index=False)
#dictResult = runProcess(cs_training, cs_test, x_var_cont=lsContinuous, blRunDecisionTree=False, blRunRandomForest=True, blRunCV=False)