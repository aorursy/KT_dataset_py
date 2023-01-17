# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as scp # scientific data crunching tools

import seaborn as sns # nice visualizations 



from sklearn import linear_model

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from sklearn import ensemble

from sklearn.model_selection import GridSearchCV



import itertools



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



sns.set(context = "notebook")



# Print numpy arrays human readable

# Thx2: https://stackoverflow.com/a/2891805

np.printoptions(precision=3, suppress=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



DataFilePath = ''



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        filepth = os.path.join(dirname, filename)

        print(filepth)

        if ('data' in filename) and ('csv' in filename):

            DataFilePath = filepth

            

if len(DataFilePath) < 1:

    print('Oh no! We have no data file! :(')

else:

    print('Data File:\n', DataFilePath)



# Any results you write to the current directory are saved as output.
CandyDataSetDF = pd.read_csv(DataFilePath)

CandyDataSetDF.head()
CandyDataSetDF.describe()
# Define a function that will provide labels for the thirs of data points



def get_third_label(value, borders):

    """This function expects a value (value:float) that it can compare against 

    two other values given (borders:list of two numbers).

    It returnds a string containing either min, med or max, depending on in which

    third the given value lies."""  

    

    third_label = ''

    

    if len(borders) > 2:

        #print('No more than two values!')

        raise ValueError('get_third_label: No more than two border values!')

    

    if value <= borders[0]:

        third_label = 'min'

    if (value > borders[0]) and (value <= borders[1]):

        third_label = 'med'

    if value > borders[1]:

        third_label = 'max'

    

    if third_label == '':

        raise RuntimeError('get_third_label: Could not assign a label!')

        

    return(third_label)

SPDFboundaries = CandyDataSetDF[['sugarpercent', 'pricepercent']].describe(percentiles = [.33, .66])



print('### Boundaries for Sugar')

print(SPDFboundaries['sugarpercent']['min'], SPDFboundaries['sugarpercent']['33%'], '- 1/3 for Sugar') 

print(SPDFboundaries['sugarpercent']['33%'], SPDFboundaries['sugarpercent']['66%'], '- 2/3 for Sugar') 

print(SPDFboundaries['sugarpercent']['66%'], SPDFboundaries['sugarpercent']['max'], '- 3/3 for Sugar') 

print('### Boundaries for Price')

print(SPDFboundaries['pricepercent']['min'], SPDFboundaries['pricepercent']['33%'], '- 1/3 for Price') 

print(SPDFboundaries['pricepercent']['33%'], SPDFboundaries['pricepercent']['66%'], '- 2/3 for Price') 

print(SPDFboundaries['pricepercent']['66%'], SPDFboundaries['pricepercent']['max'], '- 3/3 for Price') 





tmppricelabels = list()

for idx, row in CandyDataSetDF.iterrows():

    tmppricelabels.append(get_third_label(

        row['sugarpercent'], [SPDFboundaries['sugarpercent']['33%'],SPDFboundaries['sugarpercent']['66%']]))



CandyDataSetDF['sugarlabel'] = tmppricelabels 



tmpsugarlabels = list()

for idx, row in CandyDataSetDF.iterrows():

    tmpsugarlabels.append(get_third_label(

        row['pricepercent'], [SPDFboundaries['pricepercent']['33%'],SPDFboundaries['pricepercent']['66%']]))



CandyDataSetDF['pricelabel'] = tmpsugarlabels    



CandyDataSetDF.head()
sns.distplot(CandyDataSetDF['winpercent'], bins=20, color = 'blue')
scp.stats.kstest(CandyDataSetDF['winpercent'],'norm')
sns.distplot(CandyDataSetDF['sugarpercent'], bins=20, color = 'red')
scp.stats.kstest(CandyDataSetDF['sugarpercent'],'norm')
sns.distplot(CandyDataSetDF['pricepercent'], bins=20, color = 'green')
scp.stats.kstest(CandyDataSetDF['pricepercent'],'norm')
# Thx2: https://becominghuman.ai/multi-layer-perceptron-mlp-models-on-real-world-banking-data-f6dd3d7e998f

# Thx2: https://becominghuman.ai/@awhan.mohanty



# What is the difference between flatten and ravel functions in numpy?

# Thx2: https://stackoverflow.com/a/28930580/12171415



# How to use Data Scaling Improve Deep Learning Model Stability and Performance

# Thx2: https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/



from sklearn.preprocessing import MinMaxScaler,StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split



SVRGBXscaler = StandardScaler()

SVRGBYscaler = StandardScaler()



Xindy = np.array(CandyDataSetDF.iloc[:,1:10])

Ydept = np.array(CandyDataSetDF['winpercent'])



# The data appears to be centered and scaled properly, however not good enough for MLPR and SVM, need a standard distibution for best results

X4SVGB = SVRGBXscaler.fit_transform(Xindy)

Y4SVGB = SVRGBYscaler.fit_transform(Ydept.reshape(-1, 1)).flatten()

CandyDataSetDF.head()
# Expand the DataFrame with categorial variables for price and sugar

# Thx2: https://kaijento.github.io/2017/04/22/pandas-create-new-column-sum/

# Thx2: https://thispointer.com/python-how-to-use-if-else-elif-in-lambda-functions/







MLPR_Xbinscaler = MinMaxScaler(feature_range=(0.000001,0.999999)) #for ReLu activation function

MLPR_Ybinscaler = MinMaxScaler(feature_range=(0.000001,0.999999))

MLPR_XOHscaler = OneHotEncoder(categories='auto')

MLPR_yscaler = MinMaxScaler(feature_range=(0.01,0.99))



# Furthermore Neural Nets like to have a test and training set

Xtr, Xte, Ytr, Yte = train_test_split(Xindy, Ydept, test_size = 20) # Twenty samples in the test set to control outliers



XOHlabels = MLPR_XOHscaler.fit_transform(CandyDataSetDF[['sugarlabel', 'pricelabel']])

XOHtr = np.hstack([Xtr,XOHlabels.toarray()[0:len(Xtr),:]])

XOHte = np.hstack([Xte,XOHlabels.toarray()[0:(len(Xindy)-len(Xtr)),:]])



MLPR_Xbinscaler.fit(XOHtr)

XNNTrain = MLPR_Xbinscaler.transform(XOHtr)

XNNTest = MLPR_Xbinscaler.transform(XOHte)



MLPR_Ybinscaler.fit(Ytr.reshape(-1, 1))

YNNTrain = MLPR_Ybinscaler.transform(Ytr.reshape(-1, 1)).flatten()

YNNTest = MLPR_Ybinscaler.transform(Yte.reshape(-1, 1)).flatten()
# Create a view of only the ratio scaled columns

CandyRatioVar = CandyDataSetDF[['sugarpercent', 'pricepercent', 'winpercent']]



CandyRatioCorr = CandyRatioVar.corr(method = 'pearson')



sns.heatmap(CandyRatioCorr, vmin=0, vmax=1)

CandyRatioCorr
from sklearn.neighbors import LocalOutlierFactor



mylof = LocalOutlierFactor(contamination = 0.1) #Only necessary to supress the version 0.22 warning message

mylof.fit(np.array(CandyDataSetDF.iloc[:,1:12])) # Resembles Xdept

sns.lineplot(data = mylof.negative_outlier_factor_)

mylof.fit(np.array(CandyDataSetDF.iloc[:,1:13])) # Resembles all numerical values in a line

sns.lineplot(data = mylof.negative_outlier_factor_)
NNlof = LocalOutlierFactor(contamination = 0.1, n_neighbors = 8)

NNlof.fit(XNNTrain) # Training values 

sns.lineplot(data = NNlof.negative_outlier_factor_)
NNlof.fit(XNNTest) # Testing values 

sns.lineplot(data = NNlof.negative_outlier_factor_)
myregr = linear_model.LinearRegression()

myregr.fit(Xindy, Ydept)
sns.relplot(data = pd.DataFrame(data = myregr.coef_, index = CandyDataSetDF.iloc[:,1:10].columns ), 

            kind = "line", legend = False, aspect = 3)
# Thx2: https://stackoverflow.com/questions/36306555/scikit-learn-grid-search-with-svm-regression

# Python2 code and for sklearn.svm.SVC, so some changes had to be made to make this work

# for Python3 and sklearn.svm.SVR



# About the parameter C (penalty parameter)

# Thx2: https://stats.stackexchange.com/a/159051



c = list(np.linspace(0.0000001, 10, 100, endpoint = True))

param_gridSVR = dict(C = c)

mysvregr = SVR(max_iter = -1, kernel = 'linear', gamma = 'auto')



gridsvr = GridSearchCV(mysvregr, param_gridSVR, cv = 5)

gridsvr.fit(X4SVGB, Y4SVGB) 

print('AND THE WINNER IS:\n', gridsvr.best_params_)

gridsvrDF = pd.DataFrame(gridsvr.cv_results_)
myplt = sns.relplot(data = pd.DataFrame(data = gridsvr.best_estimator_.coef_.reshape(9), index = CandyDataSetDF.iloc[:,1:10].columns), 

                    kind = "line", legend = False, aspect = 3)
# Thx2: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py

# Worked:

# mygradboostregr = ensemble.GradientBoostingRegressor(n_estimators = 500, max_depth = 4, min_samples_split = 2, learning_rate = 0.01, loss = 'ls')



# Tune parameters for small datasets

mygradboostregr = ensemble.GradientBoostingRegressor(max_depth = 2)  # minimal max_depth for keeping overfitting low

mygradboostregr.fit(X4SVGB, Y4SVGB)
sns.relplot(data = pd.DataFrame(data = mygradboostregr.feature_importances_, index = CandyDataSetDF.iloc[:,1:10].columns),

            kind = "line", legend = False, aspect = 3)
# Define the parameters that stay constant and are not default.

param_fixMLPR = dict(

    #alpha =

    early_stopping = False,

    nesterovs_momentum = False,

    learning_rate = 'constant',

    #solver = 'sgd',

    activation = 'relu',

    learning_rate_init = 0.001,

    momentum = 0.9,

    max_iter = 100000,

    hidden_layer_sizes = (200,),

    verbose = False

)



# Prepare the parameters for grid search 



param_gridMLPR = dict(

    #activation = ['identity', 'logistic', 'tanh', 'relu'],

    #momentum = [0.25, 0.5, 0.9],

    #learning_rate_init = [0.001, 0.0001],

    #learning_rate = ['adaptive', 'constant'],

    #nesterovs_momentum = [True, False],

    #early_stopping = [True, False],

    solver = ['adam', 'sgd'],

    #hidden_layer_sizes = [(20,), (50,), (100,), (200,), (100, 50), (100, 50, 50), (100, 100, 50, 50)], #, (500,)    

    alpha = [0.1, 1, 10],    # Better higher penality for small sample sizes

)
myMLPR = MLPRegressor(**param_fixMLPR)



gridMLPR = GridSearchCV(myMLPR, param_gridMLPR, verbose = 2, cv = 5) # Five folds

gridMLPR.fit(XNNTrain, YNNTrain) 

print('AND THE WINNER IS:\n', gridMLPR.best_params_)

gridMLPRDF = pd.DataFrame(gridMLPR.cv_results_)



myMLPR = gridMLPR.best_estimator_
myOLDerrs = np.absolute(YNNTrain) - np.absolute(myMLPR.predict(XNNTrain)) # makes little sense but kept for backward comparability

myNEWerrs = np.absolute(YNNTrain - myMLPR.predict(XNNTrain)) 

print("OLD Home Brew Error Estimator", np.mean(myOLDerrs))

print("NEW Home Brew Error Estimator", np.mean(myNEWerrs))

# sns.relplot(data = pd.DataFrame(data = myNEWerrs, index = CandyDataSet['competitorname']),

#            kind = "line", legend = False, aspect = 3)



# Get the error score against the test set

myMLPR.score(XNNTest, YNNTest)
MLPRparams = myMLPR.get_params()

if MLPRparams['solver'] in ['sgd', 'adam']:

    MyDFLoss = pd.DataFrame(data = myMLPR.loss_curve_, columns=['Error'])

    sns.relplot(data = pd.DataFrame(data = myMLPR.loss_curve_, columns=['Error']),

                kind = "line", legend = False, aspect = 3)
# Thx2: https://docs.python.org/3/library/functions.html#bin



myinputarr = list()

CandyNames = list()

FoundNames = list()

FoundCandy = False



XLabelIndy = np.vstack((XOHtr,XOHte))



# Calculate the number of possible binary combinations and desired permutations for price and sugar content.

# 100 sugar and 100 price equals 100^2 combinations

NumBinPermutations = pow(2,9) * pow(3,2) # 6 new dimensions since one hot encoding for price (3) and sugar (3)

#NumSPPermutations = pow(100,2)

#NumTotPermutations = NumBinPermutations * NumSPPermutations



myNPinputBin = np.empty(shape=(NumBinPermutations, (9+6)))

myOneHotStrings = ['100', '010', '001']

myOHcntS = 0

myOHcntP = 0

bincnt = 0



for mynum in range(NumBinPermutations):

    mybinarystr = "{:09b}".format(bincnt) #, fill=0, witdth=20')

    bincnt += 1

    if  bincnt >= pow(2,9):

         bincnt = 0

    mybinarystr += (myOneHotStrings[myOHcntS] + myOneHotStrings[myOHcntP])

    #print(mybinarystr)

    myOHcntS += 1

    if myOHcntS > 2: # Permutate the one hot combinations

        myOHcntS = 0

        myOHcntP += 1

        if myOHcntP > 2:

            myOHcntP = 0

    for mypos in range(len(mybinarystr)):

        myinputarr.append(int(mybinarystr[mypos]))

    #print(mynum, len(mybinarystr), mybinarystr, myinputarr)

    for (x,testarr) in enumerate(XLabelIndy):

        if myinputarr == list(testarr): #Before: [0:9]

            #print(CandyDataSetDF.iloc[x,0])

            #print(mybinarystr)

            FoundNames.append(CandyDataSetDF.iloc[x,0])

            FoundCandy = True

    if FoundCandy == True:

        CandyNames.append(FoundNames)

        #print(FoundNames)

        FoundNames = []

        FoundCandy = False

    else: 

        CandyNames.append(mybinarystr)

        #print(mybinarystr)

    

    myNPinputBin[mynum] = myinputarr

    

    #Keep variables tidy

    myinputarr = []

    mybinarystr = ''

    

print('Processed {} binary permutations, generating an {} numpy array.'.format(mynum+1, np.shape(myNPinputBin)))
res = myMLPR.predict(MLPR_Xbinscaler.transform(myNPinputBin)) #fit_transform(myNPinputTot)

netmax = np.amax(res)

idx = np.argmax(res)

print('The winning combination is:')

print(myNPinputBin[idx], 'with the index {} and the value {}'.format(idx, MLPR_Ybinscaler.inverse_transform(netmax.reshape(-1, 1))))
# #################################

# Now create a nice DataFrame with all possible permutations of the binary columns

# Sort with index also sorted:

# Thx2: https://stackoverflow.com/a/52720936



MLPPredictDF = pd.DataFrame(data = myNPinputBin[:,0:9], columns = CandyDataSetDF.iloc[:,1:10].columns)



# Transform the one hot fields back to DataFrame Format



MLPPredictDF['sugarlabel'] = MLPR_XOHscaler.inverse_transform(myNPinputBin[:,9:16])[:,0]

    

MLPPredictDF['pricelabel'] = MLPR_XOHscaler.inverse_transform(myNPinputBin[:,9:16])[:,1]



MLPPredictDF['prediction'] = myMLPR.predict(MLPR_Xbinscaler.transform(myNPinputBin)) # yes, calculating a second time is not elegant but... ah well...



MLPPredictDF['PredNames'] = CandyNames 



MLPPredictDFsort = MLPPredictDF.sort_values(by = 'prediction', ascending = False)



MLPPredictDFsort.reset_index(drop=True, inplace=True)



sns.lineplot(x = MLPPredictDFsort.index, y = MLPPredictDFsort['prediction'], sort = True)

MLPPredictDFsort.head()
from time import gmtime, strftime



curgmtstr = strftime("%Y%m%d%H%M%S", gmtime())



MLPPredictDFsort.to_excel("prediction_" + curgmtstr + ".xlsx")



# Exporting the whole myNPinputTot array via a DataFrame in an EXCEL caused:

# ValueError: This sheet is too large! Your sheet size is: 5120000, 12 Max sheet size is: 1048576, 16384

# WORKAROUND: Save to CSV:

# MLPPredictDFsort.to_csv("prediction_" + curgmtstr + ".csv")



gridsvrDF.to_excel("SVRegSearch_" + curgmtstr + ".xlsx")

gridMLPRDF.to_excel("MLPRSearch_" + curgmtstr + ".xlsx")

#np.savetxt('NP_buffer_' + curgmtstr + '.csv',buffer) ###################### DEBUG



MLPRparams



##################### DEBUG

# For exporting files directly from the kernel

# Thx2: https://www.kaggle.com/general/65351#600457



#import os

#os.chdir(r'/kaggle/working')
curgmtstr
!ls -l
##################### DEBUG

#from IPython.display import FileLink

#FileLink(r'XXXXXXXX.xlsx')