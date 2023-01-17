# General
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import random, re, itertools
from IPython.display import Math

#Visualisations
import matplotlib.pyplot as plt
import seaborn as sns

# Algo
from sklearn.ensemble import RandomForestClassifier
TRAIN_PATH = '/kaggle/input/shelter-animal-outcomes/train.csv.gz'
TEST_PATH = '/kaggle/input/shelter-animal-outcomes/test.csv.gz'
SAMPLESUBMISSION_PATH = '/kaggle/input/shelter-animal-outcomes/sample_submission.csv.gz'


TARGET_VARIABLE = 'OutcomeType'
train_data = pd.read_csv(TRAIN_PATH, index_col=0)
test_data = pd.read_csv(TEST_PATH, index_col=0)
samplesubmission_data = pd.read_csv(SAMPLESUBMISSION_PATH, index_col=0)

print('Shape of Train Data : ', train_data.shape)
print('Shape of Test Data : ', test_data.shape)

class HelperFunctions:
    def __init__(self):
        print('Initialising the HelperFunctions class..')
        
    def calcLogLossError(self, metrics):
        
        #Scaling
        metrics.iloc[:,:5] = metrics.iloc[:,:5].apply(lambda x: x/x.sum(),axis=1)
        metrics.iloc[:,:5] = metrics.iloc[:,:5].applymap(lambda x: max(min(x,1-1e-15),1e-15))
        metrics.iloc[:,:5] = np.log(metrics.iloc[:,:5])
        loglossScore = (-(metrics.iloc[:,:5]*pd.get_dummies(metrics['Actuals'])).sum().sum())/metrics.shape[0]
        return loglossScore
               
    def createProbabHeatmap(self, cv_metrics):
        
        cv_metrics.sort_values('Actuals', inplace=True)
        plt.rcParams['figure.dpi'] = 180
        plt.rcParams['figure.figsize'] = (18,18)
        fig, axes =plt.subplots(5,5, sharex=False, sharey=True)
        fig.text(0.5, -0.05, 'Index', ha='center', fontsize=25)
        fig.text(-0.05, 0.5, 'Predicted Probabilities', va='center', rotation='vertical', fontsize=25)
        plt.tight_layout()

        for idx, eachClass in enumerate(cv_metrics.Actuals.unique()):

            probMatrix = cv_metrics[cv_metrics.Actuals == eachClass].reset_index().drop('AnimalID', axis=1)
            temp2 = {k:0 for k in cv_metrics.Actuals.unique()}
            predictionCounts = probMatrix.Predictions.value_counts().to_dict()
            for k in predictionCounts.keys():
                temp2[k] = predictionCounts[k]
            predictionCounts = temp2

            for idx1, eachPlotAX in enumerate(axes[idx]):

                sns.scatterplot(x = probMatrix.iloc[:,idx1].index, 
                                y = probMatrix.iloc[:,idx1].values, 
                                ax = eachPlotAX,
                                hue = probMatrix.iloc[:,idx1].values,
                                s = 20,
                                legend=False,
                                palette = 'plasma')
                eachPlotAX.hlines(np.mean(probMatrix.iloc[:,idx1].values),
                                  0, probMatrix.iloc[:,idx1].index[-1]*1.1, color='maroon', linewidth=3)
                
                if idx == idx1 : 
                    eachPlotAX.set_facecolor('darkgray')
                    eachPlotAX.text(0.5,0.5, "{0}/{1}".format(predictionCounts[probMatrix.iloc[:,idx1].name],
                                                           probMatrix.shape[0]), size=25,
                                ha="center", color='black', transform=eachPlotAX.transAxes)
                else:
                    eachPlotAX.set_facecolor('lightblue')
                    eachPlotAX.text(0.5,0.5, "{0}/{1}".format(predictionCounts[probMatrix.iloc[:,idx1].name],
                                                           probMatrix.shape[0]), size=15,
                                ha="center", color='black', transform=eachPlotAX.transAxes)
                    
                eachPlotAX.grid()
                eachPlotAX.linewidth =10

                if idx1 == 0 : eachPlotAX.set_ylabel(eachClass)
                if idx == len(axes)-1 : eachPlotAX.set_xlabel(cv_metrics.columns[:-1][idx1])

        plt.close()
        return fig

helper_functions_handler = HelperFunctions()
train_data.head()
train_data.isna().sum()
class DataProcessing:
    def __init__(self):
        print('Initialising Data Processinig Class..')

    def preprocess_dataset(self, df, mode='train', sort_bydate=True):
        
        # Drop OutcomeSubtype
        if mode == 'train':
            df.drop(['OutcomeSubtype'], axis=1, inplace=True)

        # Generate Calendar Variables
        
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        if sort_bydate:
            df.sort_values('DateTime', inplace=True)
        
        _nonnan_idx = df.loc[~df.Name.isna()].index
        df.loc[_nonnan_idx, 'Name'] = ['HAS_NAME']*df.Name[~df.Name.isna()].shape[0]
        df['Name'].fillna('NO_NAME', inplace=True)


        # Transform AgeuponOutcome TO AgeuponOutcomeDays
        def transformAgeuponOutcome(x):
            x = str(x)
            if x == 'nan':
                return np.nan
            elif 'year' in x:
                return int(x.split(' ')[0])*365
            elif 'month' in x:
                return int(x.split(' ')[0])*30
            elif 'day' in x:
                return int(x.split(' ')[0])*1

        df['AgeuponOutcome'] = df.AgeuponOutcome.apply(lambda x : transformAgeuponOutcome(x))
        df.rename(columns={'AgeuponOutcome':'AgeuponOutcomeDays'},inplace=True)

        return df
    
    def feature_engineering(self, df):
        
        # Calendar Variables
        def generateHourOfTheDay(x):
            if x < 12:
                return 'Morning'
            elif x>=12 and x < 17:
                return 'Noon'
            elif x>=17 and x < 20:
                return 'Eevening'
            elif x >= 20:
                return 'Night'
            
        # Simplify Breed..
        df.Breed = df.Breed.apply(lambda x : re.sub(' Mix', '', x))
        df.Breed = df.Breed.apply(lambda x : x.split('/')[0])
        df.Color = df.Color.apply(lambda x : x.split('/')[0])
        
        df['Month'] = df['DateTime'].dt.month
        df['Day'] = df['DateTime'].dt.day
        df['WeekOfYear'] = df['DateTime'].dt.weekofyear
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek
        df['DayType'] = pd.to_datetime(df['DateTime']).dt.hour.apply(lambda x : generateHourOfTheDay(x))
        df.drop(['DateTime'], axis=1, inplace=True)
        
        df['AnimalType_Breed'] = df.apply(lambda x : x.AnimalType+'_'+x.Breed, axis=1)
        df['AnimalType_Color'] = df.apply(lambda x : x.AnimalType+'_'+x.Color, axis=1)
        df['AnimalType_Breed_Color'] = df.apply(lambda x : x.AnimalType_Breed+'_'+x.Color, axis=1)
        df['AnimalType_SexuponOutcome'] = df.apply(lambda x : np.nan if pd.isna(x.SexuponOutcome) else x.AnimalType+'_'+x.SexuponOutcome, 
                                                   axis=1)
        df['AnimalType_Name'] = df.apply(lambda x : x.AnimalType+'_'+x.Name, axis=1)
        df['AnimalType_DayType'] = df.apply(lambda x : x.AnimalType+'_'+x.DayType, axis=1)
        
        df['AgeGroup'] = pd.cut(df.AgeuponOutcomeDays.fillna(0), bins=50)

        return df

    def process_dataset(self, df, transformDict, mode='train', verbose=True):
        """Available Types :
          'NORM_MAX','NORM_MIN', 'NORM_SUM','NORM_CUSTOM', 'ORIGNAL', 'OHE',
          'OHE_DROP', 'LE_O', 'LE_N', 'LE_O_NORM', 'LE_N_NORM', 'DROP'"""
        
        catMappings = {}
        print('Transforming Varables..')
        
        for eachKey, eachValue in transformDict.items():
            
            
            if eachValue['Type'] == 'NORM_MAX':
                df[eachKey] = df[eachKey]/df[eachKey].max()

            elif eachValue['Type'] == 'NORM_MAXMIN':
                df[eachKey] = df[eachKey]/(df[eachKey].max()-df[eachKey].min())

            elif eachValue['Type'] == 'NORM_SUM':
                df[eachKey] = df[eachKey]/df[eachKey].sum()
            
            elif eachValue['Type'] == 'NORM_CUSTOM':
                df[eachKey] = df[eachKey]/ast.literal_eval(ast.literal_eval(eachRow['Params']))

            elif eachValue['Type'] == 'ORIGNAL':
                df[eachKey] = df[eachKey]

            elif eachValue['Type'] == 'OHE':
                _dummies = pd.get_dummies(df[[eachKey]].copy(),
                           prefix=[eachKey], 
                           prefix_sep = '_',
                           columns = [eachKey], 
                           drop_first=True)

                df[_dummies.columns] = _dummies

            elif eachValue['Type'] == 'OHE_DROP':
                _dummies = pd.get_dummies(df[[eachKey]].copy(),
                           prefix=[eachKey], 
                           prefix_sep = '_',
                           columns = [eachKey], 
                           drop_first=True)

                df[_dummies.columns] = _dummies
                df.drop(eachKey, axis=1, inplace=True)

            elif eachValue['Type'] == 'LE_O':
                _param = eachValue['Params']
                _CM = pd.Categorical(df[eachKey], categories=_param)
                df[eachKey] = _CM.codes
                catMappings[eachKey] = _CM
            
            elif eachValue['Type'] == 'LE_N':
                _CM = pd.Categorical(df[eachKey])
                df[eachKey] = _CM.codes
                catMappings[eachKey] = _CM

            elif eachValue['Type'] == 'LE_O_NORM':
                _param = eachValue['Params']
                _CM = pd.Categorical(df[eachKey], categories=_param)
                df[eachKey] = _CM.codes
                df[eachKey] = df[eachKey]/df[eachKey].max()
                catMappings[eachKey] = _CM
            
            elif eachValue['Type'] == 'LE_N_NORM':
                _CM = pd.Categorical(df[eachKey])
                df[eachKey] = _CM.codes
                df[eachKey] = df[eachKey]/df[eachKey].max()
                catMappings[eachIdx] = _CM

            elif eachValue['Type'] == 'DROP':
                df.drop(eachKey, axis=1, inplace=True)

        if mode == 'train':
            _CM = pd.Categorical(df[TARGET_VARIABLE])
            df[TARGET_VARIABLE] = _CM.codes
            catMappings[TARGET_VARIABLE] = _CM
            self.trainCatMappings = catMappings
            
        elif mode == 'test':
            self.testCatMappings = catMappings

        if verbose : print('Shape of the {0} dataset After Processing : {1}\n'.format(mode, df.shape))

        return df    

    def impute(self, df):
        
        df.fillna(-1, inplace=True)
        
        return df
        
    def train_cv_split(self, df, ratio=0.2):
        
        upto = int(df.shape[0]*ratio)
        trainDF = df[:-upto]
        cvDF = df[-upto:]
        
        return trainDF, cvDF
    
dataProcessingHandler = DataProcessing()
transformDict = {'Name': {'Type': 'OHE_DROP',
                          'Params': None},
                 'AnimalType': {'Type': 'LE_N',
                                'Params': None},
                 'SexuponOutcome': {'Type': 'OHE_DROP',
                                    'Params': None},
                 'AgeuponOutcomeDays': {'Type': 'ORIGNAL',
                                        'Params': None},
                 'Breed': {'Type': 'LE_N',
                           'Params': None},
                 'Color': {'Type': 'LE_N',
                           'Params': None},
                 'Month': {'Type': 'ORIGNAL',
                           'Params': None},
                 'Day': {'Type': 'ORIGNAL',
                         'Params': None},
                 'WeekOfYear': {'Type': 'ORIGNAL',
                         'Params': None},
                 'DayOfWeek': {'Type': 'ORIGNAL',
                         'Params': None},
                 'DayType': {'Type': 'LE_O',
                             'Params': ['Morning',  'Noon', 'Eevening', 'Night']},
                 'AnimalType_Breed': {'Type': 'LE_N',
                                      'Params': None},
                 'AnimalType_Color': {'Type': 'LE_N',
                                      'Params': None},
                 'AnimalType_Breed_Color': {'Type': 'LE_N',
                                             'Params': None},
                 'AnimalType_SexuponOutcome': {'Type': 'LE_N',
                                               'Params': None},
                 'AnimalType_Name': {'Type': 'LE_N',
                                     'Params': None},
                 'AnimalType_DayType': {'Type': 'LE_N',
                                        'Params': None},
                 'AgeGroup': {'Type': 'LE_N',
                                        'Params': None}}
# Train Data
_mode = "train"
COMPLETE_TRAINING_DATA = dataProcessingHandler.preprocess_dataset(train_data.copy(), 
                                                                   mode=_mode, 
                                                                   sort_bydate= True)
COMPLETE_TRAINING_DATA = dataProcessingHandler.feature_engineering(COMPLETE_TRAINING_DATA)
COMPLETE_TRAINING_DATA = dataProcessingHandler.process_dataset(COMPLETE_TRAINING_DATA,
                                                                transformDict=transformDict, mode=_mode)
COMPLETE_TRAINING_DATA = dataProcessingHandler.impute(COMPLETE_TRAINING_DATA)


# Test Data
_mode = "test"
TESTING_DATA = dataProcessingHandler.preprocess_dataset(test_data.copy(), 
                                                         mode=_mode)
TESTING_DATA = dataProcessingHandler.feature_engineering(TESTING_DATA)
TESTING_DATA = dataProcessingHandler.process_dataset(TESTING_DATA, 
                                                      transformDict=transformDict, 
                                                      mode=_mode)
TESTING_DATA = dataProcessingHandler.impute(TESTING_DATA)
TRAINING_DATA, CV_DATA = dataProcessingHandler.train_cv_split(COMPLETE_TRAINING_DATA.copy())

print('Shape of the Training Set : ', TRAINING_DATA.shape)
print('Shape of the Cross Validation Set : ', CV_DATA.shape)
print('Shape of the Testing Set : ', TESTING_DATA.shape)
TRAINING_DATA_X = TRAINING_DATA.drop(TARGET_VARIABLE, axis=1).copy()
TRAINING_DATA_y = TRAINING_DATA[TARGET_VARIABLE]

CV_DATA_X = CV_DATA.drop(TARGET_VARIABLE, axis=1).copy()
CV_DATA_y = CV_DATA[TARGET_VARIABLE]

TESTING_DATA_X = TESTING_DATA.copy()
targetCatMapping = {idx:k for idx, k in enumerate(dataProcessingHandler.trainCatMappings[TARGET_VARIABLE].categories)}

trainDataEDA = train_data.copy()
trainDataEDA = dataProcessingHandler.preprocess_dataset(trainDataEDA, 
                                                        mode='train', 
                                                        sort_bydate= True)
trainDataEDA = dataProcessingHandler.feature_engineering(trainDataEDA)
# Analysing features w.r.t target
analysis_Cols = {'AnimalType': None,
                 'SexuponOutcome': None,
                 'AgeGroup': None,
                 'DayOfWeek': None,
                 'AnimalType_DayType': None,
                 'AnimalType_Name': None}

for eachCol in [k for k in analysis_Cols.keys()]:
    
    fig, axes = plt.subplots(1,2, figsize =(25,9))
    
    temp = trainDataEDA.groupby([TARGET_VARIABLE, eachCol]).size().unstack().fillna(0)
    
    
    t1 = temp.T/temp.sum(axis=1)
    t1.T.plot(kind='bar', stacked=True, ax = axes[0])
    t2 = temp/temp.sum(axis=0)
    t2.T.plot(kind='bar', stacked=True, ax = axes[1])
    
    
    axes[0].legend(loc='center left', bbox_to_anchor=(-0.25, 0.5))
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.suptitle('Analysing {0} v/s {1}'.format(eachCol, TARGET_VARIABLE), fontsize=20)
    analysis_Cols[eachCol] = fig
    plt.close()
    
analysis_Cols['AnimalType']
analysis_Cols['SexuponOutcome']
analysis_Cols['AgeGroup']
analysis_Cols['DayOfWeek']
analysis_Cols['AnimalType_DayType']
analysis_Cols['AnimalType_Name']
rfClf = RandomForestClassifier(oob_score=True, n_estimators=500, max_depth = 6, min_samples_leaf = 2)
_=rfClf.fit(TRAINING_DATA_X, TRAINING_DATA_y)
ypred = rfClf.predict(CV_DATA_X)
ypredProba = rfClf.predict_proba(CV_DATA_X)

CV_METRICS = pd.DataFrame(ypredProba, index=CV_DATA_X.index)
CV_METRICS.columns = CV_METRICS.columns.map(targetCatMapping)
CV_METRICS.applymap(lambda x: max(min(x,1-1e-15),1e-15))
CV_METRICS['Actuals'] = CV_DATA_y.map(targetCatMapping)
CV_METRICS['Predictions'] = pd.Series(ypred,
                                      index= CV_DATA_X.index).map(targetCatMapping)


loglossscore = helper_functions_handler.calcLogLossError(CV_METRICS.copy())

fiDF = pd.DataFrame(dict(zip(TRAINING_DATA_X.columns, rfClf.feature_importances_)).items(),columns=['Features', 'Feature_Importance'])
fiDF.set_index('Features', inplace=True)
fiDF.sort_values('Feature_Importance',ascending=False, inplace=True)
plt.rcParams['figure.figsize'] = (25,7)
fiDF.plot(kind='bar')
plt.hlines(fiDF.mean().values[0]*0.8,-20,200,color='r')
_=plt.title('OOB Score : {0}, LogLossError : {1}'.format(rfClf.oob_score_, loglossscore), fontsize=20)
helper_functions_handler.createProbabHeatmap(CV_METRICS.copy())

combinations_no = 20
bestFeatures = []
forced_columns = []
model = RandomForestClassifier(oob_score=True, n_estimators=50, max_depth = 6, min_samples_leaf = 2)

algoModel = 'RandomForestClassifier'
loglossscore = 99999
triedCombinations = []
allFeatures = [k for k in TRAINING_DATA_X.columns]

for preditorInjection in range(6,10):

    pbar = tqdm(range(combinations_no))
    
    for eachR_Pick in pbar:
        
        availableCols = [k for k in allFeatures if k not in forced_columns+[TARGET_VARIABLE]]
        randomsubset_cols = []
        
        if availableCols != []:
            randomsubset_cols = random.sample(availableCols, preditorInjection)
            randomsubset_cols.sort()
        
        if tuple(randomsubset_cols) not in triedCombinations:
            triedCombinations += [tuple(randomsubset_cols)]
            
            selectedCols = forced_columns+randomsubset_cols+[TARGET_VARIABLE]

            _modellingData = COMPLETE_TRAINING_DATA[selectedCols]
            _TRAIN_DATA, _CV_DATA = dataProcessingHandler.train_cv_split(_modellingData, ratio=0.2)
            
            _modellingData_XTrain = _TRAIN_DATA.drop(TARGET_VARIABLE, axis=1).copy()
            _modellingData_yTrain = _TRAIN_DATA[TARGET_VARIABLE].copy()

            _modellingData_XCV = _CV_DATA.drop(TARGET_VARIABLE, axis=1).copy()
            _modellingData_YCV = _CV_DATA[TARGET_VARIABLE].copy()

            _modellingData_XTest = TESTING_DATA_X.copy()

            model.fit(_modellingData_XTrain, _modellingData_yTrain)
            _yPred = model.predict(_modellingData_XCV)
            _yPredProba = model.predict_proba(_modellingData_XCV)
            
            
            CV_METRICS = pd.DataFrame(_yPredProba, index=CV_DATA_X.index)
            CV_METRICS.columns = CV_METRICS.columns.map(targetCatMapping)
            CV_METRICS.applymap(lambda x: max(min(x,1-1e-15),1e-15))
            CV_METRICS['Actuals'] = CV_DATA_y.map(targetCatMapping)
            CV_METRICS['Predictions'] = pd.Series(_yPred, index= CV_DATA_X.index).map(targetCatMapping)

            _loglossscore = helper_functions_handler.calcLogLossError(CV_METRICS)
            pbar.set_description('Injection Level : {0} @ {1}'.format(preditorInjection, round(loglossscore,4)))

            if _loglossscore < loglossscore:
                loglossscore = _loglossscore
                
                bestFeatures.append( (loglossscore, _modellingData_XTrain.columns.tolist()))

        else:
            pass
_features = bestFeatures[-1][1]
_features = ['AgeGroup',
             'AgeuponOutcomeDays',
             'AnimalType_Color',
             'AnimalType_Name',
             'AnimalType',
             'AnimalType_Breed',
             'DayType',
             'SexuponOutcome_Unknown',
             'SexuponOutcome_Neutered Male',
             'SexuponOutcome_Spayed Female']
# Change the parameters according to the model being used...
_max_depth = np.arange(5, 8, 1).astype(int)
_min_samples_leaf = np.arange(4, 6, 1).astype(int)
_n_estimators = np.arange(500, 1000, 50).astype(int)


print('All Hyper Param Combinations : ',len(_max_depth)*len(_min_samples_leaf)*len(_n_estimators))

_hyperp = [_max_depth, _min_samples_leaf, _n_estimators]
hyperCombinations = pd.DataFrame(list(itertools.product(*_hyperp)))
hyperCombinations.columns = ['max_depth', 'min_samples_leaf', 'n_estimators']

hyperCombinations.max_depth = hyperCombinations.max_depth.astype(int)
hyperCombinations.n_estimators = hyperCombinations.n_estimators.astype(int)
hyperCombinations = hyperCombinations.sample(frac=1)
hyperCombinations.sample(3)
loglossscore = 99999
hyperidx = 0
bestHParams= []

pbar = tqdm(hyperCombinations.iterrows())
for idx, eachHyper in pbar:
    
    model = RandomForestClassifier(oob_score=True, max_features='log2', n_jobs=-1)
    hparams = eachHyper.to_dict()
    hparams['max_depth'] = int(hparams['max_depth'])
    hparams['n_estimators'] = int(hparams['n_estimators'])
    model.set_params(**hparams)
    
    _=model.fit(TRAINING_DATA_X[_features], TRAINING_DATA_y)
    ypred = model.predict(CV_DATA_X[_features])
    ypredProba = model.predict_proba(CV_DATA_X[_features])
    CV_METRICS = pd.DataFrame(ypredProba, index=CV_DATA_X.index)
    CV_METRICS.columns = CV_METRICS.columns.map(targetCatMapping)
    CV_METRICS.applymap(lambda x: max(min(x,1-1e-15),1e-15))
    CV_METRICS['Actuals'] = CV_DATA_y.map(targetCatMapping)
    CV_METRICS['Predictions'] = pd.Series(ypred,
                                          index= CV_DATA_X.index).map(targetCatMapping)
    _loglossscore = helper_functions_handler.calcLogLossError(CV_METRICS.copy())
    pbar.set_description('HpIdx : {0} @ {1}'.format(hyperidx, round(loglossscore,4)))
    if _loglossscore < loglossscore:
        loglossscore = _loglossscore
        bestHParams.append([loglossscore, hparams])
        hyperidx = idx
bestHParams
model = RandomForestClassifier(oob_score=True, max_features='log2', n_jobs=-1)
model.set_params(**bestHParams[-1][1])

_=model.fit(TRAINING_DATA_X[_features], TRAINING_DATA_y)
ypred = model.predict(CV_DATA_X[_features])
ypredProba = model.predict_proba(CV_DATA_X[_features])
CV_METRICS = pd.DataFrame(ypredProba, index=CV_DATA_X.index)
CV_METRICS.columns = CV_METRICS.columns.map(targetCatMapping)
CV_METRICS.applymap(lambda x: max(min(x,1-1e-15),1e-15))
CV_METRICS['Actuals'] = CV_DATA_y.map(targetCatMapping)
CV_METRICS['Predictions'] = pd.Series(ypred, index= CV_DATA_X.index).map(targetCatMapping)
_loglossscore = helper_functions_handler.calcLogLossError(CV_METRICS.copy())
print(_loglossscore)
helper_functions_handler.createProbabHeatmap(CV_METRICS.copy())
model = RandomForestClassifier(oob_score=True, max_features='log2', n_jobs=-1)
model.set_params(**bestHParams[-1][1])

_=model.fit(COMPLETE_TRAINING_DATA[_features], COMPLETE_TRAINING_DATA[TARGET_VARIABLE])
ypredProba = model.predict_proba(TESTING_DATA_X[_features])

submissionDF = pd.DataFrame(ypredProba)
submissionDF.columns = submissionDF.columns.map(targetCatMapping)
submissionDF.index = TESTING_DATA_X.index
submissionDF.index.name = TESTING_DATA_X.index.name
submissionDF.to_csv('submissionDF.csv')
from pdpbox import pdp

XTrain = COMPLETE_TRAINING_DATA[_features].copy()

pdp_StoreType = pdp.pdp_isolate(model=model, 
                                dataset=XTrain.sample(100), 
                                num_grid_points = 100,
                                model_features=XTrain.columns.tolist(), 
                                feature='AgeGroup')

fig, axes = pdp.pdp_plot(pdp_StoreType, 'AgeGroup', plot_lines=True)
pdp_StoreType = pdp.pdp_isolate(model=model, 
                                dataset=XTrain.sample(100), 
                                num_grid_points = 100,
                                model_features=XTrain.columns.tolist(), 
                                feature='AgeuponOutcomeDays')

fig, axes = pdp.pdp_plot(pdp_StoreType, 'AgeuponOutcomeDays', plot_lines=True)



