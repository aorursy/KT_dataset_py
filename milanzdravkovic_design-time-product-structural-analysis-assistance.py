import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
import collections
import warnings
warnings.filterwarnings("ignore")
import pickle
df6=pd.read_csv("../input/geometry-and-physical-properties-of-fixator/DOE6.csv", 
                skiprows=4, 
                names=['Name',
                       'Bar length',
                       'Bar diameter',
                       'Bar end thickness',
                       'Radius trochanteric unit',
                       'Radius bar end',
                       'Clamp distance',
                       'Total Deformation Maximum',
                       'Equivalent Stress',
                       'P9',
                       'P10',
                       'P11',
                       'Fixator Mass'], 
                usecols=['Bar length',
                         'Bar diameter',
                         'Bar end thickness',
                         'Radius trochanteric unit',
                         'Radius bar end',
                         'Clamp distance',
                         'Total Deformation Maximum',
                         'Equivalent Stress',
                         'Fixator Mass'])
df6.head()
df6.info()
plt.rcParams["figure.figsize"] = (14,14)
plt.rcParams.update({'font.size': 11})

fig, ax = plt.subplots(1, 3, figsize=(12, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.3)

defdict={'DOE6':df6['Total Deformation Maximum'].values}
ax[0].set_title('Total Deformation Max')
ax[0].boxplot(defdict.values(), widths=(0.4))
ax[0].set_xticklabels(defdict.keys())
ax[0].grid(color='gray', ls = '-.', lw = 0.2)
ax[0].set_ylabel('mm')

strdict={'DOE6':df6['Equivalent Stress'].values}
ax[1].set_title('Equivalent Stress')
ax[1].boxplot(strdict.values(), widths=(0.4))
ax[1].set_xticklabels(strdict.keys())
ax[1].grid(color='gray', ls = '-.', lw = 0.2)
ax[1].set_ylabel('MPa')

masdict={'DOE6':df6['Fixator Mass'].values}
ax[2].set_title('Fixator Mass')
ax[2].boxplot(masdict.values(), widths=(0.4))
ax[2].set_xticklabels(masdict.keys())
ax[2].grid(color='gray', ls = '-.', lw = 0.2)
ax[2].set_ylabel('kg')

plt.show()
print('std(def) = ' + str(df6['Total Deformation Maximum'].values.std()))
print('std(str) = ' + str(df6['Equivalent Stress'].values.std()))
print('std(mas) = ' + str(df6['Fixator Mass'].values.std()))
pd.options.display.float_format = '{:,.3f}'.format
df6.corr(method='pearson')
plt.rcParams["figure.figsize"] = (14,14)
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(3, 3, figsize=(17, 12))
plt.subplots_adjust(wspace=0.4, hspace=0.3)

ax[0,0].set_title('p=-0.950')
ax[0,0].plot(df6['Bar length'].values,df6['Total Deformation Maximum'].values,'g.')
ax[0,0].set_xlabel('Bar length (mm)')
ax[0,0].set_ylabel('Total Deformation Max (mm)')
ax[0,0].grid(color='gray', ls = '-.', lw = 0.2)

ax[0,1].set_title('p=-0.655')
ax[0,1].plot(df6['Bar length'].values,df6['Equivalent Stress'].values,'g.')
ax[0,1].set_xlabel('Bar length (mm)')
ax[0,1].set_ylabel('Equivalent Stress (MPa)')
ax[0,1].grid(color='gray', ls = '-.', lw = 0.2)

ax[0,2].set_title('p=0.899')
ax[0,2].plot(df6['Bar length'].values,df6['Fixator Mass'].values,'g.')
ax[0,2].set_xlabel('Bar length (mm)')
ax[0,2].set_ylabel('Fixator Mass (Kg)')
ax[0,2].grid(color='gray', ls = '-.', lw = 0.2)

ax[1,0].set_title('p=-0.250')
ax[1,0].plot(df6['Bar diameter'].values,df6['Total Deformation Maximum'].values,'y.')
ax[1,0].set_xlabel('Bar diameter (mm)')
ax[1,0].set_ylabel('Total Deformation Max (mm)')
ax[1,0].grid(color='gray', ls = '-.', lw = 0.2)

ax[1,1].set_title('p=-0.051')
ax[1,1].plot(df6['Bar diameter'].values,df6['Equivalent Stress'].values,'r.')
ax[1,1].set_xlabel('Bar diameter (mm)')
ax[1,1].set_ylabel('Equivalent Stress (MPa)')
ax[1,1].grid(color='gray', ls = '-.', lw = 0.2)

ax[1,2].set_title('p=0.397')
ax[1,2].plot(df6['Bar diameter'].values,df6['Fixator Mass'].values,'y.')
ax[1,2].set_xlabel('Bar diameter (mm)')
ax[1,2].set_ylabel('Fixator Mass (Kg)')
ax[1,2].grid(color='gray', ls = '-.', lw = 0.2)

ax[2,0].set_title('p=-0.038')
ax[2,0].plot(df6['Bar end thickness'].values,df6['Total Deformation Maximum'].values,'r.')
ax[2,0].set_xlabel('Bar end thickness (mm)')
ax[2,0].set_ylabel('Total Deformation Max (mm)')
ax[2,0].grid(color='gray', ls = '-.', lw = 0.2)

ax[2,1].set_title('p=-0.633')
ax[2,1].plot(df6['Bar end thickness'].values,df6['Equivalent Stress'].values,'g.')
ax[2,1].set_xlabel('Bar end thickness (mm)')
ax[2,1].set_ylabel('Equivalent Stress (MPa)')
ax[2,1].grid(color='gray', ls = '-.', lw = 0.2)

ax[2,2].set_title('p=0.048')
ax[2,2].plot(df6['Bar end thickness'].values,df6['Fixator Mass'].values,'r.')
ax[2,2].set_xlabel('Bar end thickness (mm)')
ax[2,2].set_ylabel('Fixator Mass (Kg)')
ax[2,2].grid(color='gray', ls = '-.', lw = 0.2)

plt.show()
import collections
import warnings
warnings.filterwarnings("ignore")

def getModels():
    models = []
    models.append(('LRE', LinearRegression()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('SVR', SVR(kernel='linear')))
    models.append(('DTR', DecisionTreeRegressor()))
    models.append(('RFR', RandomForestRegressor()))
    models.append(('GBR', GradientBoostingRegressor()))
    return models

def getRanking(X,y,model):
    fit=RFECV(model, cv=3).fit(X,y)
    return fit.ranking_, fit.support_

def getRFERanks(X,y):
    models=getModels()
    ranks= collections.defaultdict(dict)
    for x in range(y.shape[1]):
        for modelname, model in models:
            if(modelname!='KNN'):
                rank, supp=getRanking(X,y[:,x],model)
                ranks[x+1][modelname]=rank
    return ranks
X=df6.values[:,:6]
y=df6.values[:,6:]
ranks=getRFERanks(X,y)
defranks=ranks[1]
strranks=ranks[2]
masranks=ranks[3]
plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
plt.subplots_adjust(wspace=0.2, hspace=0.8)

x = np.arange(6)
width = 0.15

ax[0].set_xlabel('Input features')
ax[0].set_ylabel('Rank of corrn with Total Deformation Max (1-5)')
ax[0].bar(x, defranks['LRE'], width=width, label='LinReg', color='r')
ax[0].bar(x+width, defranks['SVR'], width=width, label='SVR', color='y')
ax[0].bar(x+2*width, defranks['DTR'], width=width, label='DecTre', color='b')
ax[0].bar(x+3*width, defranks['RFR'], width=width, label='RanFor', color='c')
ax[0].bar(x+4*width, defranks['GBR'], width=width, label='GBoost', color='m')
ax[0].legend()

ax[1].set_xlabel('Input features')
ax[1].set_ylabel('Rank of corr with Equivalent Stress (1-5)')
ax[1].bar(x, strranks['LRE'], width=width, label='LinReg', color='r')
ax[1].bar(x+width, strranks['SVR'], width=width, label='SVR', color='y')
ax[1].bar(x+2*width, strranks['DTR'], width=width, label='DecTre', color='b')
ax[1].bar(x+3*width, strranks['RFR'], width=width, label='RanFor', color='c')
ax[1].bar(x+4*width, strranks['GBR'], width=width, label='GBoost', color='m')
ax[1].legend()

ax[2].set_xlabel('Input features')
ax[2].set_ylabel('Rank of corr with Product Mass (1-5)')
ax[2].bar(x, masranks['LRE'], width=width, label='LinReg', color='r')
ax[2].bar(x+width, masranks['SVR'], width=width, label='SVR', color='y')
ax[2].bar(x+2*width, masranks['DTR'], width=width, label='DecTre', color='b')
ax[2].bar(x+3*width, masranks['RFR'], width=width, label='RanFor', color='c')
ax[2].bar(x+4*width, masranks['GBR'], width=width, label='GBoost', color='m')
ax[2].legend()

plt.show()
cv_doe6=KFold(n_splits=4, shuffle=False, random_state=3)
import collections
import warnings
warnings.filterwarnings("ignore")

def getModels():
    models = []
    models.append(('LRE', LinearRegression()))
    models.append(('KNN', KNeighborsRegressor()))
    models.append(('SVR', SVR()))
    models.append(('DTR', DecisionTreeRegressor()))
    models.append(('RFR', RandomForestRegressor(random_state=7)))
    models.append(('GBR', GradientBoostingRegressor(random_state=7)))
    return models

scaler1 = MinMaxScaler()
scaled_df6=scaler1.fit_transform(df6)

scores=[]
for d in range(3):
    featurescores=[]
    for modelname, model in getModels():
        X=df6.values[:,:6]
        y=df6.values[:,6+d]
        if(modelname=='KNN' or modelname=='SVR'):
            X=scaled_df6[:,:6]
        fscores=cross_val_score(model, X, y, cv=cv_doe6, 
                                             scoring='neg_mean_absolute_error').mean()
        featurescores.append(fscores)
    scores.append(featurescores)
    
dfres = pd.DataFrame(scores, 
                     columns=['LRE','KNN','SVR','DTR','RFR','GBR'], 
                     index=['Total Deformation Maximum','Equivalent Stress','Fixator mass'])
dfres.head()
def testModel(X, y, model, parameters, feature):
    grid = GridSearchCV(estimator=model, param_grid=parameters, cv=cv_doe6, scoring=make_scorer(r2_score))
    grid = grid.fit(X, y=y)
    print('-------------------------')
    print("Parameters :", grid.best_params_)
    print("R2 :", grid.best_score_)
    print('-------------------------')
    return grid.best_params_, grid.best_score_ 
    
    
params={'LRE':
          {'params':
               {'fit_intercept':[True,False],
                'normalize': [True,False],
                'copy_X': [True,False],
                'n_jobs': [1,2,None]
               }
          },
        'KNN':
          {'params':
               {'n_neighbors':[7,13,20],
                'leaf_size':[1,5,30],
                'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p':[1,2,3,5],
                'weights':['uniform', 'distance']
               }
          },
        'SVR':
          {'params':
               {'C': [100], 
                'kernel': ['rbf','poly','linear','sigmoid'],
                'degree':[2,3,5,8],
                'coef0':[0.1,1],
                'shrinking':[True,False],
                'gamma':['scale','auto']
               }
          },
        'DTR':
          {'params':
               {'max_features': ['auto', 'sqrt', 'log2'],
                'splitter':['best','random'],
                'max_depth':[1,10,100,None],
                'min_samples_split': [8,9,10],
                'criterion':['mse', 'friedman_mse', 'mae'],
                'min_samples_leaf':[4,5,6]
               }
          },
        'RFR':
          {'params':
               {'n_estimators': [5,7,10],
                'max_features': ['auto'],
                'criterion': ['mae','mse'],
                'min_samples_split': [2,4],
                'min_samples_leaf': [1,3,4],
                'max_leaf_nodes':[6,8,12],
                'bootstrap':[True,False],
                'max_depth': [3,4]
               }
          },
        'GBR':
          {'params':
               {'loss':['ls', 'lad'],
                'learning_rate': [0.08,0.1],
                'subsample'    : [1.0],
                'criterion': ['mse'],
                'min_samples_split': [0.7,0.8,4],
                'min_samples_leaf': [1],
                'n_estimators' : [100,200,600],
                'max_depth'    : [None]
               }
          }
       }
features = {0:'Total Deformation Maximum',1:'Equivalent Stress',2:'Fixator Mass'}
gsresults= collections.defaultdict(dict)

scaler2 = MinMaxScaler()
scaled_df6=scaler2.fit_transform(df6)

for d in features.keys():
    for modelname, model in getModels():
        X=df6.values[:,:6]
        y=df6.values[:,6+d]
        if(modelname=='KNN' or modelname=='SVR'):
            X=scaled_df6[:,:6]
        print(features[d] + ' with ' + modelname)
        prms,sc=testModel(X, y, model,params[modelname].values(), d)
        gsresults[features[d]][modelname]={'PARAMS':prms}
scaler6 = MinMaxScaler()
scaled_df6=scaler6.fit_transform(df6)


scores=[]
features = {0:'Total Deformation Maximum',1:'Equivalent Stress',2:'Fixator Mass'}
for d in features.keys():
    y=df6.values[:,6+d]
    featurescores=[]
    for modelname, model in getModels():
        X=df6.values[:,:6]
        if(modelname=='KNN' or modelname=='SVR'):
            X=scaled_df6[:,:6]
        model.set_params(**gsresults[features[d]][modelname]['PARAMS'])
        #pickle.dump(model, open('models/'+modelname+'_'+str(d)+'.sav','wb'))
        featurescores.append(cross_val_score(model, X, y, cv=cv_doe6,
                                             scoring='neg_mean_absolute_error').mean())
    scores.append(featurescores)

dfresopt = pd.DataFrame(scores, 
                     columns=['LRE','KNN','SVR','DTR','RFR','GBR'], 
                     index=['Total Deformation Maximum','Equivalent Stress','Fixator mass'])
dfresopt.head()
dfres.head()