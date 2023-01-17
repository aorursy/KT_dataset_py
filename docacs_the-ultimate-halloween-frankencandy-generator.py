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

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

from sklearn.neighbors import LocalOutlierFactor

from sklearn.model_selection import GridSearchCV

from sklearn import ensemble



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



# Defining some constant like variables

MODEL_NAMES = ['MLPR', 'LRG', 'SVR', 'GBR']



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
sns.distplot(CandyDataSetDF['winpercent'], bins=20, color = 'blue')
scp.stats.kstest(CandyDataSetDF['winpercent'],'norm')
sns.distplot(CandyDataSetDF['sugarpercent'], bins=20, color = 'red')
scp.stats.kstest(CandyDataSetDF['sugarpercent'],'norm')
sns.distplot(CandyDataSetDF['pricepercent'], bins=20, color = 'green')
scp.stats.kstest(CandyDataSetDF['pricepercent'],'norm')
# Define a dictionary for the Y scalers, we need them later...

# MODEL_NAMES = ['MLPR', 'LRG', 'SVR', 'GBR']

YScalDict = dict.fromkeys(MODEL_NAMES) # Dictionary for the scalers, later needed for inverse transform

ModelDict = dict.fromkeys(MODEL_NAMES) # Dictionary of the models for convenient multi model handling

ScoreDict = dict.fromkeys(MODEL_NAMES) # Dictionary of the model scores



Xindy = np.array(CandyDataSetDF.iloc[:,1:10])

Ydept = np.array(CandyDataSetDF['winpercent'])



# For support vector regression and eventually gradient boosting

SVRGBXscaler = StandardScaler()

YScalDict['SVR'] = StandardScaler()

YScalDict['GBR'] = FunctionTransformer(lambda x : x, validate = True) # returns idenicals, needed for later processing



X4SVGB = SVRGBXscaler.fit_transform(Xindy)

Y4SVGB = YScalDict['SVR'].fit_transform(Ydept.reshape(-1, 1)).flatten()

YI4GBR = YScalDict['GBR'].fit_transform(Ydept.reshape(-1, 1)).flatten()

YScalDict['LRG'] = YScalDict['SVR'] #Linear regression also uses the same scaler



YScalDict['MLPR'] = MinMaxScaler(feature_range=(1,100)) # Suprisingly this feature range works far better than (0,1) or (0,100)!

MLPR_xscaler = MinMaxScaler(feature_range=(0.001,0.999)) # Making X a bit less binary



XtrTmp, XteTmp, YtrTmp, YteTmp = train_test_split(Xindy[:,0:9], Ydept, test_size = 20)



MLPR_xscaler.fit(Xindy[:,0:9])



YScalDict['MLPR'].fit(Ydept.reshape(-1, 1))

YNNTrain = YScalDict['MLPR'].transform(YtrTmp.reshape(-1, 1)).flatten()

XNNTrain = MLPR_xscaler.transform(XtrTmp)



YNNTest = YScalDict['MLPR'].transform(YteTmp.reshape(-1, 1)).flatten()

XNNTest = MLPR_xscaler.transform(XteTmp)
mylof = LocalOutlierFactor(contamination = 0.1) # contamination only necessary to supress the version 0.22 warning message



plt.figure(figsize=(10, 4))



mylof.fit(np.array(CandyDataSetDF.iloc[:,1:10])) # Resembles Xdept

sns.lineplot(data = mylof.negative_outlier_factor_)

mylof.fit(np.array(CandyDataSetDF.iloc[:,1:13])) # Resembles all numerical values in a line

sns.lineplot(data = mylof.negative_outlier_factor_)
NNlof = LocalOutlierFactor(contamination = 0.1) #, n_neighbors = 8)

NNlof.fit(XNNTrain) # Training values 

plt.figure(figsize=(10, 4))

sns.lineplot(data = NNlof.negative_outlier_factor_)
NNlof.fit(XNNTest) # Testing values 

plt.figure(figsize=(10, 4))

sns.lineplot(data = NNlof.negative_outlier_factor_)
# Create a view of only the ratio scaled columns

CandyRatioVar = CandyDataSetDF[['sugarpercent', 'pricepercent', 'winpercent']]



CandyRatioCorr = CandyRatioVar.corr(method = 'pearson')



sns.heatmap(CandyRatioCorr, vmin=0, vmax=1)

CandyRatioCorr
ModelDict['LRG'] = linear_model.LinearRegression()

ModelDict['LRG'].fit(X4SVGB, Y4SVGB) 

ScoreDict['LRG'] = ModelDict['LRG'].score(X4SVGB, Y4SVGB)
coef_linreg = ModelDict['LRG'].coef_

sns.relplot(data = pd.DataFrame(data = coef_linreg, index = CandyDataSetDF.iloc[:,1:10].columns ), 

            kind = "line", legend = False, aspect = 3)
param_gridSVR = dict(C = list(np.linspace(0.0000001, 10, 100, endpoint = True)))



ModelDict['SVR'] = SVR(max_iter = -1, kernel = 'linear', gamma = 'auto')



gridsvr = GridSearchCV(ModelDict['SVR'], param_gridSVR, cv = 5)

gridsvr.fit(X4SVGB, Y4SVGB) 

print('AND THE WINNER IS:\n', gridsvr.best_params_)

gridsvrDF = pd.DataFrame(gridsvr.cv_results_)

ModelDict['SVR'] = gridsvr.best_estimator_

ScoreDict['SVR'] = ModelDict['SVR'].score(X4SVGB, Y4SVGB)
coef_SVR = ModelDict['SVR'].coef_.reshape(9)



myplt = sns.relplot(data = pd.DataFrame(data = coef_SVR, index = CandyDataSetDF.iloc[:,1:10].columns),

                    kind = "line", legend = False, aspect = 3)
param_fixGBR = dict(

    #n_estimators = 3000,

    max_depth = 2,

    learning_rate = 0.01,

    min_samples_leaf = 9,

    max_features = 0.3

)



# Prepare the parameters for grid search 

# Thx2: https://www.slideshare.net/PyData/gradient-boosted-regression-trees-in-scikit-learn-gilles-louppe



param_gridGBR = dict(

    n_estimators = [100, 500, 1500, 3000, 4500, 6000],

    #max_depth = [2, 4, 6],

    #learning_rate = [0.1, 0.05, 0.02, 0.01],

    #min_samples_leaf = [3, 5, 9],

    #max_features = [1.0, 0.3, 0.1],

)
# Thx2: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py

# Worked:

# mygradboostregr = ensemble.GradientBoostingRegressor(n_estimators = 500, max_depth = 4, min_samples_split = 2, learning_rate = 0.01, loss = 'ls')



ModelDict['GBR'] = ensemble.GradientBoostingRegressor(**param_fixGBR)



gridGBR = GridSearchCV(ModelDict['GBR'], param_gridGBR, verbose = True, cv = 5) # Five folds

#gridGBR.fit(X4SVGB, Y4SVGB) 

gridGBR.fit(Xindy, Ydept) # Gradient boosting regression is very robust against scaling effects



ModelDict['GBR'] = gridGBR.best_estimator_

ScoreDict['GBR'] = ModelDict['GBR'].score(Xindy, Ydept) # Attention! Must be same parameters as in: gridGBR.fit above

gridGBRDF = pd.DataFrame(gridGBR.cv_results_)

print('AND THE WINNER IS:\n', gridGBR.best_params_)
coef_GBR = ModelDict['GBR'].feature_importances_

sns.relplot(data = pd.DataFrame(data = coef_GBR, index = CandyDataSetDF.iloc[:,1:10].columns),

            kind = "line", legend = False, aspect = 3)
AllCoefDF = pd.DataFrame()



MMSAllCoef = MinMaxScaler()



AllCoefNP = MMSAllCoef.fit_transform(np.vstack([coef_linreg, coef_SVR, coef_GBR]).T)



AllCoefDF['coef_linreg'] = AllCoefNP[:,0]

AllCoefDF['coef_SVR'] = AllCoefNP[:,1]

AllCoefDF['coef_GBR'] = AllCoefNP[:,2]



AllCoefDF.index = CandyDataSetDF.iloc[:,1:10].columns 



sns.relplot(data = AllCoefDF, kind = "line", aspect = 3)
# Define the parameters that stay constant and are not default.

param_fixMLPR = dict(

    #alpha = 0.0001,

    early_stopping = True,

    solver='adam',

    learning_rate='adaptive',

    learning_rate_init=0.0001,

    hidden_layer_sizes = (100, 100, 50, 50),

    activation= 'relu',

    max_iter=100000,

    verbose=False

)



# Prepare the parameters for grid search 

# Thx2: https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html



param_gridMLPR = dict(

    #activation = ['identity', 'logistic', 'tanh', 'relu'],

    alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10],    # Deal with outliers

    #momentum = [0.25, 0.5, 0.9],

    #learning_rate_init = [0.001, 0.0001],

    #learning_rate = ['adaptive', 'constant'],

    #nesterovs_momentum = [True, False],

    #early_stopping = [True, False],

    #solver = ['adam', 'sgd'],

    #hidden_layer_sizes = [(20,), (50,), (100,), (200,), (100, 50), (100, 50, 50), (100, 100, 50, 50)],   

)
ModelDict['MLPR'] = MLPRegressor(**param_fixMLPR)



gridMLPR = GridSearchCV(ModelDict['MLPR'], param_gridMLPR, verbose = True, cv = 5) # Five folds

gridMLPR.fit(XNNTrain, YNNTrain) 

print('AND THE WINNER IS:\n', gridMLPR.best_params_)

gridMLPRDF = pd.DataFrame(gridMLPR.cv_results_)



ModelDict['MLPR'] = gridMLPR.best_estimator_
ScoreDict['MLPR'] = ModelDict['MLPR'].score(XNNTest,YNNTest)

print('R² score: ', ScoreDict['MLPR'])
MLPRparams = ModelDict['MLPR'].get_params()

if MLPRparams['solver'] in ['sgd', 'adam']:

    sns.relplot(data = pd.DataFrame(data = ModelDict['MLPR'].loss_curve_, columns=['Error']),

                kind = "line", legend = False, aspect = 3)
# ####################################################

# Essentially we can produce all possible combinations by 9 bit encoding, because of nine binary attributes

# So, we build a function that will genereate a numpy array with all possible 9 bit combinations that we can 

# then feed into the MLP network for approximation to find the best candidate.

# Thx2: https://docs.python.org/3/library/functions.html#bin



myinputarr = list()

CandyNames = list()

FoundNames = list()

FoundCandy = False

myNPinput = np.empty(shape=(pow(2,9),9))



for mynum in range(pow(2,9)):

    mybinarystr = "{:09b}".format(mynum)

    for mypos in range(len(mybinarystr)):

        myinputarr.append(int(mybinarystr[mypos]))

    for (x,testarr) in enumerate(Xindy):

        if myinputarr == list(testarr[0:9]):

            FoundNames.append(CandyDataSetDF.iloc[x,0])

            FoundCandy = True

    if FoundCandy == True:

        CandyNames.append(FoundNames)

        FoundNames = []

        FoundCandy = False

    else:

        CandyNames.append(mybinarystr)

                

    myNPinput[mynum] = myinputarr

    myinputarr = []

    

print('DONE!')
# #################################

# Now create nice DataFrame with all possible permutations and save the predictions of each model.

# Sort with index also sorted:

# Thx2: https://stackoverflow.com/a/52720936



PredDFDict = {}

PredDFsortDict = {}



for n in MODEL_NAMES:



    PredDFDict[n] = pd.DataFrame(data = myNPinput, columns = CandyDataSetDF.iloc[:,1:10].columns)

    PredDFDict[n]['prediction'] = YScalDict[n].inverse_transform(ModelDict[n].predict(myNPinput).reshape(-1, 1))

    PredDFDict[n]['PredNames'] = CandyNames



    PredDFsortDict[n] = PredDFDict[n].sort_values(by = 'prediction', ascending = False)

    PredDFsortDict[n].reset_index(drop=True, inplace=True)

    

    print(n, 'Sorted DataFrame with predictions created')
# Plot the graphs

# https://seaborn.pydata.org/tutorial/axis_grids.html



PredictionsAllDF = pd.DataFrame(columns=[a for a in PredDFDict])

for a in PredDFDict:

    PredictionsAllDF[a] = PredDFsortDict[a]['prediction']

sns.relplot(data = PredictionsAllDF, aspect = 2.5)
# Extracting the records with candy names from the output DataFrame and build a dictionary with candy name as key, the true value and the predicted value. 

# Thx2: https://www.journaldev.com/23763/python-remove-spaces-from-string



rankdict = dict()



for k, MLPPredictDFsort in PredDFDict.items():

    is_name = MLPPredictDFsort['PredNames'].map(lambda x: not str(x).isdigit())

    NameSortDF = MLPPredictDFsort[is_name]







    for rpos, namestr in enumerate(CandyDataSetDF['competitorname']):

        nammap = NameSortDF['PredNames'].map(lambda x: not str(x).find(namestr) == -1)

        NamFoundDF = NameSortDF[nammap]

        if len(NamFoundDF) > 1:

            namlst = dict() #Dictionary of lists containing the candy names of the respective DataFrame line 

            for tstidx, namstr in enumerate(NamFoundDF['PredNames']):

                namlst[tstidx] = str(namstr).strip("[]").replace("'","").split(',')

                for x,s in enumerate(namlst[tstidx]):

                    namlst[tstidx][x] = s.strip() # delete the remaining whitespaces

            is_exmatch = False

            for c,v in namlst.items():

                for s in v:

                    if namestr == s:

                        is_exmatch = True

                        break

                if is_exmatch == True:

                    NamFoundDF = NamFoundDF.iloc[c]

                    break

        elif len(NamFoundDF) > 2:

            raise ValueError('Found candy name more than two times!')

        if not (k in rankdict): #Key does not exist yet, make a new entry

            rankdict[k] = dict()

        rankdict[k][namestr] = [float(CandyDataSetDF.at[rpos, 'winpercent']), float(NamFoundDF['prediction'])]



    print('Done with {}!'.format(k))
# Calculate the Pearson rank correllation and Kendall's Tau 

# Thx2: https://stackoverflow.com/a/24888331



spearmans_R = {}

kendalls_Tau = {}

RanksDF = pd.DataFrame(index = [a for a in rankdict], columns=['Kendalls_Tau', 'Spearmans_R'])



for k in rankdict:

    tempDF = pd.DataFrame.from_dict(data = rankdict[k], orient='index', columns = ['winpercent', 'prediction']) 



    RanksDF['Kendalls_Tau'].loc[k] = scp.stats.kendalltau(tempDF['winpercent'], tempDF['prediction'])[0]

    RanksDF['Spearmans_R'].loc[k] = scp.stats.spearmanr(tempDF['winpercent'], tempDF['prediction'])[0]





RanksDF
# Thx2: https://stackoverflow.com/a/33227833

sns.lineplot(data = RanksDF.astype(float))

plt.ylim(0, 1)
ScoreDF = pd.DataFrame.from_dict(data = ScoreDict, orient='index', columns=['R² coeff'])



sns.lineplot(data = ScoreDF.astype(float))

plt.ylim(0, 1)
#######################

# Create a DataFrame with the weights of MLPR

# Thx2: https://stackoverflow.com/a/19736406



MLPRLayersDict = dict()

for c, a in enumerate(ModelDict['MLPR'].coefs_):

    print('Layer {}'.format(c+1), len(a), np.shape(a))

    MLPRLayersDict['Wght_IN_'+str(c)] = a[0]

    MLPRLayersDict['Wght_OUT_'+str(c)] = a[1]

    

MLPRLayersDF = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in MLPRLayersDict.items() ]))

MLPRLayersDF.head()
from time import gmtime, strftime



curgmtstr = strftime("%Y%m%d%H%M%S", gmtime())



# For some reason Kaggle does not like '+' signs in filenames, so here's the workaround

posnegstr = lambda x: 'p' if x >= 0.0 else 'n'

curscorestr = '_' + posnegstr(ScoreDict['MLPR']) + '{:.6f}'.format(ScoreDict['MLPR']).replace('.','_').replace('-','')



###############################################

# Write multiple Dataframes in one EXCEL file

# Thx2: https://xlsxwriter.readthedocs.io/example_pandas_multiple.html



MLPRParamsDFDict = dict.fromkeys(MODEL_NAMES)



for a in MODEL_NAMES:

    MLPRParamsDFDict[a] = pd.DataFrame.from_dict(ModelDict[a].get_params(), orient = 'index', columns = ['value'])



with pd.ExcelWriter('results_' + curgmtstr + curscorestr + ".xlsx", engine='xlsxwriter') as XLSXwriter:

    for a in PredDFsortDict:

        PredDFsortDict[a].to_excel(XLSXwriter, sheet_name = 'pred_'+a)

    AllCoefDF.to_excel(XLSXwriter, sheet_name = 'Lin_Coeffs')

    MLPRLayersDF.to_excel(XLSXwriter, sheet_name = 'MLPR_Coeffs')

    RanksDF.to_excel(XLSXwriter, sheet_name = 'RankCorr')

    ScoreDF.to_excel(XLSXwriter, sheet_name = 'R² score')

    for a in MODEL_NAMES:

        MLPRParamsDFDict[a].to_excel(XLSXwriter, sheet_name = 'params_'+a)

    gridMLPRDF.to_excel(XLSXwriter, sheet_name = 'grid_MLPR')

    gridsvrDF.to_excel(XLSXwriter, sheet_name = 'grid_SVR')

    gridGBRDF.to_excel(XLSXwriter, sheet_name = 'grid_GBR')
##################### DEBUG

# For exporting files directly from the kernel

# Thx2: https://www.kaggle.com/general/65351#600457



#import os

#os.chdir(r'/kaggle/working')
!ls -l *.xlsx
curgmtstr
##################### DEBUG

#from IPython.display import FileLink

#FileLink(r'XXXXXXX.xlsx')