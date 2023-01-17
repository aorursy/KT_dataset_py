import pandas as pd

import numpy  as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime



from xgboost import XGBClassifier 

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.pipeline import Pipeline



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



MAXRows = 30000000



CoverNames = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow',

'Aspen', 'Douglas-fir', 'Krummholz']



#'Vertical_Distance_To_Hydrology',

IMPORTV =  ['Elevation', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 

            'Horizontal_Distance_To_Hydrology', 'Hillshade_9am', 'Hillshade_Noon', 'Vertical_Distance_To_Hydrology', 

            'Wilderness_Area1', 'Aspect', 'Slope', 'Hillshade_3pm', 

            'Soil_Type3', 'Soil_Type10', 'Soil_Type4', 'Wilderness_Area3', 'Soil_Type39', 'Soil_Type30', 'Wilderness_Area4', 

            'Soil_Type2', 'Soil_Type32', 'Soil_Type38', 'Soil_Type33']





"""

Load // setup data

"""

path = '../input/'

Train = pd.read_csv(path + 'train.csv', index_col='Id', nrows=MAXRows)

Train ['Origin'] = 'Train'



Test =  pd.read_csv(path + 'test.csv', index_col='Id', nrows=MAXRows)

Test ['Origin'] = 'Test'



# Join everything so that we can treat Train and Test variables with less code

Todo = pd.concat([Train, Test])

TrainSize = Train.shape[0]

def Graficos():

    for v in IVSkew.index.values:

        Train.plot(kind='hist', y=v, color='orange')

        plt.title(v + ' - Skew: ' +  str(IVSkew.loc[v]) )

        plt.show()



def TreatVars(X, drp, skw):

    

    # Drop variables that make no difference

    print ('\nDropping: ', drp)

    X.drop(drp, inplace=True, axis=1)

    

    # Normalize skewed variables

    print ('\nDe Skewing: ', skw)

    X[skw] = np.log1p( X[skw])

    return

"""

Evaluate Stuff

"""

    

def EvalStuff (X, y, model)  : 

    y_pred    = model.predict(X)

    score     = model.score(X, y)

    accuracy  = accuracy_score(y, y_pred)

    

    print('\nScore: {} Accuracy: {}\n\n\nConfusion Matrix'.format(score, accuracy))

    print(confusion_matrix(y, y_pred))

    plt.matshow(confusion_matrix(y, y_pred))

    plt.colorbar()

    plt.show()

    

    print(classification_report(y, y_pred))

    return 









"""

Initial modeling, raw

"""

    

def ModelStuff (X, y, estimator, estimator_parms):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=32)

    

    model = GridSearchCV(estimator, estimator_parms, cv=3)

    model.fit(X_train, y_train)

    print('GridSearchCV - \n Best Estimator: {}  \n Best Score: {}'.format(model.best_estimator_, model.best_score_))

    

    EvalStuff(X_test, y_test, model)

    

#    ImpFeat = pd.DataFrame(list(zip(X.columns, model.feature_importances_)), 

#                  columns=['Feature', 'Importance'])

#    ImpFeat.sort_values('Importance', ascending=False, inplace=True)

#    print(ImpFeat[ImpFeat['Importance'] > 0])

#    

#    rtrn = {'Model' : model, 'ImpFeat' : ImpFeat}



    rtrn = {'Model' : model}

    return rtrn

"""

Analisis de Variables

"""

# Correlation variables

dfCorr   = Train.corr()

CorrVars = list(dfCorr.columns.values)

dfPairs =pd.DataFrame()

for key in CorrVars:

    dfPairs = dfPairs.append({'Column' : key, 'Corr' :   dfCorr[key]['Cover_Type']}, ignore_index=True)

dfPairs.sort_values('Corr', ascending=False, inplace=True)

HighCorr = list(dfPairs.iloc[1:15]['Column'])



# Interva variables

INTVARS = [x for x in list(Train.columns.values) 

    if 'Wilderness_Area' not in x and 'Soil_Type' not in x]

INTVARS.remove('Origin')

INTVARS.remove('Cover_Type')





#Skewed variables

IVSkew = Train[INTVARS].skew().abs().sort_values(0, ascending=False)

IVSkew = list(IVSkew[:6].index)

# Wilderness / SoilType variables: reverse OneHot encoding to a single column

WildVars = [x for x in list(Train.columns.values)  if 'Wilderness_Area'  in x ]

WildVles = Train[WildVars].apply(lambda x: ''.join(x.astype(str)), axis=1)

WildVles = WildVles.apply(lambda x: x.index('1'))

SoilVars = [x for x in list(Train.columns.values)  if 'Soil_Type'  in x ]

SoilVles = Train[SoilVars].apply(lambda x: ''.join(x.astype(str)), axis=1)

SoilVles = SoilVles.apply(lambda x: x.index('1'))



#NOOP variables

WildChek = Train[WildVars].apply(lambda x: x.value_counts()).fillna(0).transpose()

SoilChek = Train[SoilVars].apply(lambda x: x.value_counts()).fillna(0).transpose()

print(SoilChek[SoilChek[1] < 20])

SoilDrop = list(SoilChek[SoilChek[1] < 20].index)





Train2 = Train.drop(WildVars + SoilVars, axis=1)

Train2['Wilderness'] = WildVles

Train2['SoilType'] = SoilVles



#plt.rc("figure", figsize = (10,5))

sns.countplot(x = 'Wilderness', hue = 'Cover_Type', data = Train2)

plt.show()

sns.countplot(x = 'SoilType', hue = 'Cover_Type', data = Train2)

plt.show()
"""

Modela y predice

"""

xgb_params = {  

    "learning_rate": [ 0.01, 0.02],

    "reg_alpha" : [0.05,  0.1, 0.15, 0.2],

    "nthread" : [-1],

    "silent" : [0]

}

dtc_params = {  

              "criterion": ["gini", "entropy"],

              "min_samples_split": [2, 10, 20],

              "max_depth": [None, 2, 5, 10],

              "min_samples_leaf": [1, 5, 10],

              "max_leaf_nodes": [None, 5, 10, 20]}

rfc_params = {  

              "criterion": ["gini", "entropy"],

              "min_samples_split": [2, 10, 20],

              "max_depth": [None, 2, 5, 10],

              "min_samples_leaf": [1, 5, 10],

              "max_leaf_nodes": [None, 5, 10, 20]}



scl = StandardScaler()

xgb = XGBClassifier(objective='multi:softmax')

dtc = DecisionTreeClassifier()

rfc = RandomForestClassifier()



steps = [('scl', scl),

        ('rfc', rfc)]

pipe_params = {}

for key, val in rfc_params.items():

    pipe_params['rfc__' + key]  = val

pipe = Pipeline(steps)

# Re-scale so no negative values remain. needed for Log1p

Todo ['Vertical_Distance_To_Hydrology'] = Todo ['Vertical_Distance_To_Hydrology']  + abs(np.min(Todo ['Vertical_Distance_To_Hydrology'] ))

TreatVars(Todo, SoilDrop, IVSkew)



Train = Todo[:TrainSize]

Test  = Todo[TrainSize:].drop(['Cover_Type'], axis=1)

X = Train.drop(['Cover_Type', 'Origin'], axis=1)

y = Train['Cover_Type']

rtrn = ModelStuff(X, y, pipe, pipe_params)

model = rtrn['Model']   



# Evaluate on all data

print('\n\nRESULTS: Whole Train dataset')

EvalStuff (X, y, model)      
# Now, predict on Test dataset         

y = model.predict(Test.drop(['Origin'], axis=1))    

result = pd.DataFrame({'Id' : Test.index.values, 'Cover_Type' : y} )



result[['Id', 'Cover_Type']].to_csv('result.csv', index=False)



print ("Current date and time: " , datetime.datetime.now())






