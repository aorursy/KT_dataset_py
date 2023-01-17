import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns
# Let's see the correlations between the attributes

# To do it, we need only numercial values



importTestAdult = pd.read_csv("../input/adultbasefiles/adult.test.txt",

                    names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country", "Target"],

                    sep=r'\s*,\s*',

                    engine='python',

                    na_values="?")



importAdult = pd.read_csv("../input/adultbasefiles/adult.data.txt",

                    names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country", "Target"],

                    sep=r'\s*,\s*',

                    engine='python',

                    na_values="?")



attributes = ["Age", "Workclass", "Education-Num", "Martial Status",

              "Occupation", "Relationship", "Race", "Sex", "Capital Gain",

              "Capital Loss", "Hours per week", "Country", "Target"]



indexColumnsToConvert = [1,3,4,5,6,7,11,12]



cleanedImportTestAdult1 = importTestAdult.dropna(axis=0, how='any', subset=attributes, inplace=False)



# I figured out I needed to import a csv file to successfully calculate the correlations matrix

# When I tried to simply convert those numpy.array into panda.DataFrame it simply didn't work...

# So I made this function to pass from a numpy array to a complete pandas DataFrame

def numpyToDF(numpyArray,colLabels):

    f1 = open('./temp.txt', 'w')

    for line in numpyArray:

        string = ""

        for data in line:

            string += str(data) + ", "

        f1.write(string[:-2] + "\n")

    f1.close()



    return(pd.read_csv('./temp.txt',

                         names=colLabels,

                         sep=r'\s*,\s*',

                         engine='python'))



#let's convert the NaN columns into numbers, if necessary

def convertDatasToNumbers(indexColumnsToConvert,dataFrame,colLabels):

    numpyDF = dataFrame[colLabels].values

    for col in indexColumnsToConvert:

        valueList=[]

        for i in range(len(numpyDF)):

            value = numpyDF[i][col]

            if col == indexColumnsToConvert[-1]:

                if value in ['>50K','>50K.','>50k','>50k.',' >50K',' >50K.',' >50k',' >50k.']:

                    numpyDF[i][col] = 1

                else:

                    numpyDF[i][col] = 0

            else:

                if value not in valueList:

                    valueList.append(value)

                numpyDF[i][col] = valueList.index(value)

    return numpyToDF(numpyDF,colLabels)



testAdult1 = convertDatasToNumbers(indexColumnsToConvert,cleanedImportTestAdult1,attributes)
corr_mat = testAdult1.corr().round(2)



# Draw a correlation heatmap

plt.rcParams['font.size'] = 18

plt.figure(figsize = (20, 20))

sns.heatmap(corr_mat, vmin = -0.8, vmax = 0.8, center = 0, 

            cmap = plt.cm.RdYlGn_r, annot = True);
# So now, we can see that there is no hide duplicate datas.

# In plus, we can see that some of those datas are poorly significant to the Target.

# For this, we will reject the following attributes:

attributesToReject = ["Martial Status","Occupation","Relationship","Race","Country",'Target']



keyColumns = [x for x in attributes if x not in attributesToReject]

print(keyColumns)
cleanedImportTestAdult = importTestAdult.dropna(axis=0, how='any', subset=keyColumns+['Target'], inplace=False)

cleanedImportAdult = importAdult.dropna(axis=0, how='any', subset=keyColumns+['Target'], inplace=False)
indexColumnsToConvert = [1,3,7] # Workclass, Sex, Target

testAdult = convertDatasToNumbers(indexColumnsToConvert,cleanedImportTestAdult,keyColumns+['Target'])

adult = convertDatasToNumbers(indexColumnsToConvert,cleanedImportAdult,keyColumns+['Target'])
XTestAdult = testAdult[keyColumns]

YTestAdult = testAdult['Target']
XAdult = adult[keyColumns]

YAdult = adult['Target']
# Let's import most commons models

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
# DataFrame to hold results

model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])



def cv_model(Xdata, Ydata, model, name, model_results=None):

    """Perform 10 fold cross validation of a model"""

    

    cv_scores = cross_val_score(model, Xdata, Ydata, cv = 10, n_jobs = -1)

    print(f'10 Fold CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')

    

    if model_results is not None:

        model_results = model_results.append(pd.DataFrame({'model': name, 

                                                           'cv_mean': cv_scores.mean(), 

                                                            'cv_std': cv_scores.std()},

                                                           index = [0]),

                                             ignore_index = True)



        return model_results
model_results = cv_model(XTestAdult,YTestAdult,ExtraTreesClassifier(),"ExtraTrees",model_results)
model_results = cv_model(XTestAdult,YTestAdult,LinearSVC(),"LinearSV",model_results)
model_results = cv_model(XTestAdult,YTestAdult,GaussianNB(),"GaussianNB",model_results)
model_results = cv_model(XTestAdult,YTestAdult,MLPClassifier(),"Multi-layer Perceptron",model_results)
model_results = cv_model(XTestAdult,YTestAdult,LogisticRegressionCV(),"LogisticReg",model_results)
model_results = cv_model(XTestAdult,YTestAdult,RidgeClassifierCV(),"RidgeReg",model_results)
model_results = cv_model(XTestAdult,YTestAdult,LinearDiscriminantAnalysis(),"LinearDiscriminant",model_results)
model_results = cv_model(XTestAdult,YTestAdult,RandomForestClassifier(),"RdmForest",model_results)
model_results
model_results.set_index('model', inplace = True)

model_results['cv_mean'].plot.bar(figsize = (8, 6), color='orange',

                                  yerr = list(model_results['cv_std']),

                                  grid=True)



plt.title('Model 10 Cross Validation Score Results');

plt.ylabel('Mean Score (with error bar)');

model_results.reset_index(inplace = True)
MLP_results = pd.DataFrame(columns = ['alpha', 'accuracy', 'stdErr'])



# Filter out warnings from models

import warnings 

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category = ConvergenceWarning)

warnings.filterwarnings('ignore', category = DeprecationWarning)

warnings.filterwarnings('ignore', category = UserWarning)



alphas = np.geomspace(10**-20,10**5,num=25)

for A in alphas:

    accuracies = []

    for i in range(10):

        MLP = MLPClassifier(max_iter=100, alpha=A)

        MLP.fit(XTestAdult,YTestAdult)

        YAdultPred_MLP = MLP.predict(XAdult)

        accuracies.append(accuracy_score(YAdult,YAdultPred_MLP))

    

    MLP_results = MLP_results.append(pd.DataFrame({'alpha': A, 'accuracy': np.mean(accuracies), 'stdErr': np.std(accuracies)},

                                    index = [0]),ignore_index = True)
index = MLP_results.index[MLP_results['accuracy'] == MLP_results['accuracy'].max()].tolist()

MLP_results.iloc[index]
MLP_results
MLP_results.set_index('alpha', inplace = True)

MLP_results['accuracy'].plot.bar(figsize = (15, 6), color='orange',

                                 yerr = list(MLP_results['stdErr']),

                                 grid=True)



plt.title('MLP accuracy in function of alpha');

plt.ylabel('accuracy');

MLP_results.reset_index(inplace = True)
#Let's create a function to pass back from the 0 and 1 of the target to the '<=50k' and '>50k'.

def binToHumanIncome (binaire):

    if binaire == 0:

        return '<=50k'

    return '>50k'



def createOutput(fileName, targetArray):

    Id = [i for i in range(len(targetArray))]

    d = {'Id' : Id, 'Income' : targetArray}

    DF = pd.DataFrame(d)

    DF.to_csv(fileName+'.csv',index=False, sep=',', line_terminator = '\n', header = ["Id", "Income"])
# Let's create the output file.

MLP = MLPClassifier(max_iter=200, alpha=3*10**-5)

MLP.fit(XTestAdult,YTestAdult)

YAdultPred_MLP = MLP.predict(XAdult)



createOutput('MLP_Prediction',YAdultPred_MLP)
import warnings 

from sklearn.exceptions import ConvergenceWarning



# Filter out warnings from models

warnings.filterwarnings('ignore', category = ConvergenceWarning)

warnings.filterwarnings('ignore', category = DeprecationWarning)

warnings.filterwarnings('ignore', category = UserWarning)



LR_results = pd.DataFrame(columns = ['solver', 'accuracy', 'stdErr'])

solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']





for sol in solver:

    accuracies = []

    for i in range(3):

        LR = LogisticRegressionCV(max_iter=100, solver=sol)

        LR.fit(XTestAdult,YTestAdult)

        YAdultPred_LR = LR.predict(XAdult)

        accuracies.append(accuracy_score(YAdult,YAdultPred_LR))

    

    LR_results = LR_results.append(pd.DataFrame({'solver': sol, 'accuracy': np.mean(accuracies), 'stdErr': np.std(accuracies)},

                                    index = [0]),ignore_index = True)
index = LR_results.index[LR_results['accuracy'] == LR_results['accuracy'].max()].tolist()

LR_results.iloc[index]
LR_results
LR_results.set_index('solver', inplace = True)

LR_results['accuracy'].plot.bar(figsize = (15, 6), color='orange',

                                 yerr = list(LR_results['stdErr']),

                                 grid=True)



plt.title('LR accuracy in function of the solver');

plt.ylabel('accuracy');

LR_results.reset_index(inplace = True)
# Let's create the output file.

LR = LogisticRegressionCV(max_iter=250, solver='liblinear')

LR.fit(XTestAdult,YTestAdult)

YAdultPred_LR = LR.predict(XAdult)



createOutput('LogisticRegression_Prediction',YAdultPred_LR)
RdF_results = pd.DataFrame(columns = ['forestSize', 'accuracy', 'stdErr'])

forestSize = np.concatenate((np.arange(1,6,1),np.arange(10,51,5),np.arange(100,501,50)))



for size in forestSize:

    accuracies = []

    for i in range(5):

        RdF =  RandomForestClassifier(n_estimators=size, criterion='gini', bootstrap=False, n_jobs=-1)

        RdF.fit(XTestAdult,YTestAdult)

        YAdultPred_RdF = RdF.predict(XAdult)

        accuracies.append(accuracy_score(YAdult,YAdultPred_RdF))

    

    RdF_results = RdF_results.append(pd.DataFrame({'forestSize': size, 'accuracy': np.mean(accuracies), 'stdErr': np.std(accuracies)},

                                    index = [0]),ignore_index = True)
index = RdF_results.index[RdF_results['accuracy'] == RdF_results['accuracy'].max()].tolist()

RdF_results.iloc[index]
RdF_results
RdF_results.set_index('forestSize', inplace = True)

RdF_results['accuracy'].plot.bar(figsize = (15, 6), color='orange',

                                 yerr = list(RdF_results['stdErr']),

                                 grid=True)



plt.title('RdF accuracy in function of the number of trees');

plt.ylabel('accuracy');

RdF_results.reset_index(inplace = True)
RdF_results = pd.DataFrame(columns = ['forestSize', 'accuracy', 'stdErr'])

forestSize = np.concatenate((np.arange(1,6,1),np.arange(10,51,5),np.arange(100,501,50)))



for size in forestSize:

    accuracies = []

    for i in range(5):

        RdF =  RandomForestClassifier(n_estimators=size, criterion='entropy', bootstrap=False, n_jobs=-1)

        RdF.fit(XTestAdult,YTestAdult)

        YAdultPred_RdF = RdF.predict(XAdult)

        accuracies.append(accuracy_score(YAdult,YAdultPred_RdF))

    

    RdF_results = RdF_results.append(pd.DataFrame({'forestSize': size, 'accuracy': np.mean(accuracies), 'stdErr': np.std(accuracies)},

                                    index = [0]),ignore_index = True)
index = RdF_results.index[RdF_results['accuracy'] == RdF_results['accuracy'].max()].tolist()

RdF_results.iloc[index]
RdF_results
RdF_results.set_index('forestSize', inplace = True)

RdF_results['accuracy'].plot.bar(figsize = (15, 6), color='orange',

                                 yerr = list(RdF_results['stdErr']),

                                 grid=True)



plt.title('RdF accuracy in function of the number of trees');

plt.ylabel('accuracy');

RdF_results.reset_index(inplace = True)
RdF_results = pd.DataFrame(columns = ['forestSize', 'accuracy', 'stdErr'])

forestSize = np.concatenate((np.arange(1,6,1),np.arange(10,51,5),np.arange(100,501,50)))



for size in forestSize:

    accuracies = []

    for i in range(5):

        RdF =  RandomForestClassifier(n_estimators=size, criterion='gini', bootstrap=True, n_jobs=-1)

        RdF.fit(XTestAdult,YTestAdult)

        YAdultPred_RdF = RdF.predict(XAdult)

        accuracies.append(accuracy_score(YAdult,YAdultPred_RdF))

    

    RdF_results = RdF_results.append(pd.DataFrame({'forestSize': size, 'accuracy': np.mean(accuracies), 'stdErr': np.std(accuracies)},

                                    index = [0]),ignore_index = True)
index = RdF_results.index[RdF_results['accuracy'] == RdF_results['accuracy'].max()].tolist()

RdF_results.iloc[index]
RdF_results
RdF_results.set_index('forestSize', inplace = True)

RdF_results['accuracy'].plot.bar(figsize = (15, 6), color='orange',

                                 yerr = list(RdF_results['stdErr']),

                                 grid=True)



plt.title('RdF accuracy in function of the number of trees');

plt.ylabel('accuracy');

RdF_results.reset_index(inplace = True)
RdF_results = pd.DataFrame(columns = ['forestSize', 'accuracy', 'stdErr'])

forestSize = np.concatenate((np.arange(1,6,1),np.arange(10,51,5),np.arange(100,501,50)))



for size in forestSize:

    accuracies = []

    for i in range(5):

        RdF =  RandomForestClassifier(n_estimators=size, criterion='entropy', bootstrap=True, n_jobs=-1)

        RdF.fit(XTestAdult,YTestAdult)

        YAdultPred_RdF = RdF.predict(XAdult)

        accuracies.append(accuracy_score(YAdult,YAdultPred_RdF))

    

    RdF_results = RdF_results.append(pd.DataFrame({'forestSize': size, 'accuracy': np.mean(accuracies), 'stdErr': np.std(accuracies)},

                                    index = [0]),ignore_index = True)
index = RdF_results.index[RdF_results['accuracy'] == RdF_results['accuracy'].max()].tolist()

RdF_results.iloc[index]
RdF_results
RdF_results.set_index('forestSize', inplace = True)

RdF_results['accuracy'].plot.bar(figsize = (15, 6), color='orange',

                                 yerr = list(RdF_results['stdErr']),

                                 grid=True)



plt.title('RdF accuracy in function of the number of trees');

plt.ylabel('accuracy');

RdF_results.reset_index(inplace = True)
# Let's create the output file.

RdF =  RandomForestClassifier(n_estimators=450, criterion='gini', bootstrap=True, n_jobs=-1)

RdF.fit(XTestAdult,YTestAdult)

YAdultPred_RdF = RdF.predict(XAdult)



createOutput('RandomForest_Prediction',YAdultPred_RdF)