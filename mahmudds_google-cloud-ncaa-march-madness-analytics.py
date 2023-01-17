import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import matplotlib as mpl

from matplotlib.patches import Circle, Rectangle, Arc
import seaborn as sns

plt.style.use('seaborn-dark-palette')

mypal = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Grab the color pal

import os

import gc
def mkdir(path):

    import os

    path = path.strip()

    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:

        os.makedirs(path)

        print(path + 'Successufully established')

        return True

    else:

        print(path + 'dir existed')

        return False
DIRC_MENS_PATH = '../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1'

DIRC_WOMENS_PATH = '../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1'
MensTeamsDT = pd.read_csv(f'{DIRC_MENS_PATH}/MTeams.csv')

WoMensTeamsDT = pd.read_csv(f'{DIRC_WOMENS_PATH}/WTeams.csv')
MensTeamsDT
WoMensTeamsDT
M_drop_list = ["FirstD1Season","TeamName","LastD1Season"]

W_drop_list = ['TeamName']

MensTeamsDT.drop(M_drop_list,axis = 1,inplace = True)

WoMensTeamsDT.drop(W_drop_list,axis = 1,inplace = True)

MensTeamsDT.head()
print(len(MensTeamsDT["TeamID"].unique()))

print(len(WoMensTeamsDT["TeamID"].unique()))
# Team_box

time_list = [2015,2016,2017,2018,2019]
MensTeamsDT = pd.read_csv(f'{DIRC_MENS_PATH}/MTeams.csv')

WoMensTeamsDT = pd.read_csv(f'{DIRC_WOMENS_PATH}/WTeams.csv')
M_Seed = pd.read_csv(f'{DIRC_MENS_PATH}/MNCAATourneySeeds.csv')

W_Seed = pd.read_csv(f'{DIRC_WOMENS_PATH}/WNCAATourneySeeds.csv')
M_Team_seed = M_Seed[M_Seed['Season'].isin(time_list)]

W_Team_seed = W_Seed[W_Seed['Season'].isin(time_list)]
print(len(W_Team_seed))

print(len(M_Team_seed))
M_game_result_detailed =  pd.read_csv(f'{DIRC_MENS_PATH}/MRegularSeasonDetailedResults.csv')

W_game_result_detailed =  pd.read_csv(f'{DIRC_WOMENS_PATH}/WRegularSeasonDetailedResults.csv')



print(M_game_result_detailed.head())

print(W_game_result_detailed.head())
M_game_result_detailed = M_game_result_detailed[M_game_result_detailed['Season'].isin(time_list)]

W_game_result_detailed = W_game_result_detailed[W_game_result_detailed['Season'].isin(time_list)]

print(M_game_result_detailed.head())

print(W_game_result_detailed.head())
print(len(M_game_result_detailed))

print(len(W_game_result_detailed))
M_game_result_detailed.columns
W_game_result_detailed.columns
win_list = ['Season','WTeamID','WScore','WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',

       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']

M_game_result_win = M_game_result_detailed[win_list]

W_game_result_win = W_game_result_detailed[win_list]

W_game_result_win.head()
lose_list = ['Season','LTeamID','LScore','LFGM', 'LFGA', 'LFGM3', 'LFGA3',

       'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

M_game_result_lose = M_game_result_detailed[lose_list]

W_game_result_lose = W_game_result_detailed[lose_list]

W_game_result_lose.head()
M_team_win_box = M_game_result_win.groupby(['WTeamID','Season']).count()

W_team_win_box = W_game_result_win.groupby(['WTeamID','Season']).count()
drop_list = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',

       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']

M_team_win_box = M_team_win_box.drop(drop_list,axis = 1)

W_team_win_box = W_team_win_box.drop(drop_list,axis = 1)

M_team_win_box = M_team_win_box.rename(columns = {'WScore':'count'})

W_team_win_box = W_team_win_box.rename(columns = {'WScore':'count'})
M_team_count = M_team_win_box.reset_index()

W_team_count = W_team_win_box.reset_index()
M_team_win_box = M_game_result_win.groupby(['WTeamID','Season']).sum()

M_team_win_box = M_team_win_box.reset_index()

M_team_win_box = pd.merge(M_team_win_box,M_team_count,on = ['WTeamID','Season'])

win_rename_columns = {'WTeamID':"TeamID","WScore":"Score",'WFGM':'FGM', 'WFGA':'FGA', 'WFGM3':'FGM3', 'WFGA3':'FGA3',

       'WFTM':'FTM', 'WFTA':'FTA', 'WOR':'OR', 'WDR':'DR', 'WAst':'Ast', 'WTO':'TO', 'WStl':'Stl', 'WBlk':'Blk', 'WPF':'PF'}

M_team_win_box = M_team_win_box.rename(columns=win_rename_columns)

M_team_win_box.head()
W_team_win_box = W_game_result_win.groupby(['WTeamID','Season']).sum()

W_team_win_box = W_team_win_box.reset_index()

W_team_win_box = pd.merge(W_team_win_box,W_team_count,on = ['WTeamID','Season'])

W_team_win_box = W_team_win_box.rename(columns=win_rename_columns)

W_team_win_box.head()
len(W_team_win_box)
W_game_result_lose
M_team_lose_box = M_game_result_lose.groupby(['LTeamID','Season']).count()

drop_list = ['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR',

       'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

M_team_lose_box = M_team_lose_box.drop(drop_list,axis = 1)

M_team_lose_box = M_team_lose_box.rename(columns = {'LScore':'count'})
W_team_lose_box = W_game_result_lose.groupby(['LTeamID','Season']).count()

W_team_lose_box = W_team_lose_box.drop(drop_list,axis = 1)

W_team_lose_box = W_team_lose_box.rename(columns = {'LScore':'count'})
M_team_lose_count = M_team_lose_box.reset_index()

W_team_lose_count = W_team_lose_box.reset_index()
M_result_lose = M_game_result_lose.groupby(['LTeamID','Season']).sum()

M_team_lose_box = M_result_lose.reset_index()

M_team_lose_box = pd.merge(M_team_lose_box,M_team_lose_count,on = ['LTeamID','Season'])

rename_columns = {'LTeamID':"TeamID","LScore":"Score",'LFGM':'FGM', 'LFGA':'FGA', 'LFGM3':'FGM3', 'LFGA3':'FGA3',

       'LFTM':'FTM', 'LFTA':'FTA', 'LOR':'OR', 'LDR':'DR', 'LAst':'Ast', 'LTO':'TO', 'LStl':'Stl', 'LBlk':'Blk', 'LPF':'PF'}

M_team_lose_box = M_team_lose_box.rename(columns=rename_columns)

M_team_lose_box.head()
W_result_lose = W_game_result_lose.groupby(['LTeamID','Season']).sum()

W_team_lose_box = W_result_lose.reset_index()

W_team_lose_box = pd.merge(W_team_lose_box,W_team_lose_count,on = ['LTeamID','Season'])

W_team_lose_box = W_team_lose_box.rename(columns=rename_columns)

W_team_lose_box.head()
assert len(W_team_win_box.columns) == len(W_team_lose_box.columns)
M_result = pd.merge(M_team_win_box,M_team_lose_box,on=['TeamID','Season'])

W_result = pd.merge(W_team_win_box,W_team_lose_box,on=['TeamID','Season'])
M_result = M_team_win_box.append(M_team_lose_box)

W_result = W_team_win_box.append(W_team_lose_box)
W_result
M_result = M_result.groupby(['TeamID','Season']).sum()

W_result = W_result.groupby(['TeamID','Season']).sum()
W_result
element_list = ['Score','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','TO','Stl','Blk','PF','Ast']
M_result = M_result[element_list].apply(lambda x:x/M_result['count'])

W_result = W_result[element_list].apply(lambda x:x/W_result['count'])
W_result
M_result_withseed = pd.merge(M_result,M_Team_seed,on=['TeamID','Season'],how = 'outer')

W_result_withseed = pd.merge(W_result,W_Team_seed,on=['TeamID','Season'],how = 'outer')
W_result_withseed
M_team_win_box = M_result_withseed.rename(columns = {'TeamID':'WTeamID'})

M_team_lose_box = M_result_withseed.rename(columns = {'TeamID':'LTeamID'})

W_team_win_box = W_result_withseed.rename(columns = {'TeamID':'WTeamID'})

W_team_lose_box = W_result_withseed.rename(columns = {'TeamID':'LTeamID'})
M_result_withseed.to_csv('M_result.csv')

W_result_withseed.to_csv('W_result.csv')
M_game_result =  pd.read_csv(f'{DIRC_MENS_PATH}/MNCAATourneyCompactResults.csv')

M_game_result = M_game_result[M_game_result['Season'].isin(time_list)]

W_game_result =  pd.read_csv(f'{DIRC_WOMENS_PATH}/WNCAATourneyCompactResults.csv')

W_game_result = W_game_result[W_game_result['Season'].isin(time_list)]
W_game_result
W_team_lose_box
M_game_result_1 = pd.merge(M_game_result,M_team_win_box,on=['WTeamID','Season'],how = 'left')

W_game_result_1 = pd.merge(W_game_result,W_team_win_box,on=['WTeamID','Season'],how = 'left')
W_game_result_1
M_game_result_final = pd.merge(M_game_result_1,M_team_lose_box,on=['LTeamID','Season'],how = 'left')

W_game_result_final = pd.merge(W_game_result_1,W_team_lose_box,on=['LTeamID','Season'],how = 'left')
W_game_result_final[['Score_x','Score_y']]
M_game_result_final.to_csv('M_result_by_game_tourney.csv')

W_game_result_final.to_csv('W_result_by_game_tourney.csv')
import statsmodels.api as sm



from matplotlib import pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.datasets import load_diabetes

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score



import warnings

warnings.filterwarnings('ignore')
df = M_game_result_final.copy()

df1=df.copy()

df_WTeamID=df['WTeamID']

df_LTeamID=df['LTeamID']

df_WScore=df['WScore']

df_LScore=df['LScore']

df1['WTeamID']=df_LTeamID

df1['LTeamID']=df_WTeamID

df1['WScore']=df_LScore

df1['LScore']=df_WScore

label_x=df1.columns[8:23]

label_y=df1.columns[23:38]

label_x=list(label_x)

label_y=list(label_y)

for i in range(len(label_x)):

    df1[label_x[i]]=df[label_y[i]]

    df1[label_y[i]]=df[label_x[i]]
df['target']=1

df1['target']=0

df_final=df.append(df1)
df_W = W_game_result_final.copy()

df_W1=df_W.copy()

df_W_WTeamID=df_W['WTeamID']

df_W_LTeamID=df_W['LTeamID']

df_W_WScore=df_W['WScore']

df_W_LScore=df_W['LScore']

df_W1['WTeamID']=df_W_LTeamID

df_W1['LTeamID']=df_W_WTeamID

df_W1['WScore']=df_W_LScore

df_W1['LScore']=df_W_WScore

label_x_W=df_W1.columns[8:23]

label_y_W=df_W1.columns[23:38]

label_x_W=list(label_x_W)

label_y_W=list(label_y_W)

for i in range(len(label_x_W)):

    df_W1[label_x_W[i]]=df_W[label_y_W[i]]

    df_W1[label_y_W[i]]=df_W[label_x_W[i]]
df_W['target']=1

df_W1['target']=0

df_W_final=df_W.append(df_W1)
df_final.to_csv('M_result_by_game_tourney_editored.csv')

df_W_final.to_csv('W_result_by_game_tourney_editored.csv')
m,n=np.shape(df_final)
df_final.reset_index(inplace = True)
df_final
df_final = df_final.drop(columns = 'index')
df_final
data=df_final.copy()

wseed = data["Seed_x"]

lseed = data["Seed_y"]

Wseed  = np.zeros([m])

Lseed = np.zeros([m])
data
for i in range(m):

    Wseed[i] = wseed[i][1:3]

    Lseed[i] = lseed[i][1:3]

    

seeddiff=Wseed-Lseed
df_final=df_final.drop(['WLoc','Seed_x','Seed_y','WTeamID','WScore','LTeamID','LScore'],axis=1)
m,n=np.shape(df_final)
df_final.insert(n-1,'Seeddiff',seeddiff) 
df_final.tail()
# Split the data 

df_train=df_final[df_final['Season']<2019]

df_test=df_final[df_final['Season']>2018]
x_train = df_train.iloc[:,0:n].values

y_train = df_train.target.values
x_test = df_test.iloc[:,0:n].values

y_test = df_test.target.values
from sklearn.linear_model import LogisticRegressionCV



logreg = LogisticRegressionCV(cv=5,random_state=0, solver='newton-cg')

logreg.fit(x_train, y_train)



y_pred_train = logreg.predict(x_train)

y_pred_test = logreg.predict(x_test)



print("Coefficients :", np.round(logreg.intercept_,4), np.round(logreg.coef_,4))
y_pred_train = logreg.predict(x_train)

y_pred_test = logreg.predict(x_test)



accuracy_train = accuracy_score(y_train, y_pred_train)

accuracy_test = accuracy_score(y_test, y_pred_test)

print('Accuracy on the training set =', np.round(accuracy_train,4))

print('Accuracy on the test set =', np.round(accuracy_test,4))
m,n=np.shape(df_W_final)
df_W_final.reset_index(inplace = True)
df_W_final = df_W_final.drop(columns = 'index')
data=df_W_final.copy()

wseed = data["Seed_x"]

lseed = data["Seed_y"]

Wseed  = np.zeros([m])

Lseed = np.zeros([m])
for i in range(m):

    Wseed[i] = wseed[i][1:3]

    Lseed[i] = lseed[i][1:3]

    

seeddiff=Wseed-Lseed
df_W_final=df_W_final.drop(['WLoc','Seed_x','Seed_y','WTeamID','WScore','LTeamID','LScore'],axis=1)
m,n=np.shape(df_W_final)
df_W_final.insert(n-1,'Seeddiff',seeddiff) 
# Split the data 

df_train_W=df_W_final[df_W_final['Season']<2019]

df_test_W=df_W_final[df_W_final['Season']>2018]
x_train_W = df_train_W.iloc[:,0:n].values

y_train_W = df_train_W.target.values
x_test_W = df_test_W.iloc[:,0:n].values

y_test_W = df_test_W.target.values
from sklearn.linear_model import LogisticRegressionCV



logreg = LogisticRegressionCV(cv=5,random_state=0, solver='newton-cg')

logreg.fit(x_train, y_train)



y_pred_train = logreg.predict(x_train_W)

y_pred_test = logreg.predict(x_test_W)



print("Coefficients :", np.round(logreg.intercept_,4), np.round(logreg.coef_,4))
y_pred_train = logreg.predict(x_train_W)

y_pred_test = logreg.predict(x_test_W)



accuracy_train = accuracy_score(y_train_W, y_pred_train)

accuracy_test = accuracy_score(y_test_W, y_pred_test)

print('Accuracy on the training set =', np.round(accuracy_train,4))

print('Accuracy on the test set =', np.round(accuracy_test,4))
import xgboost as xgb
data_xgb = pd.read_csv("M_result_by_game_tourney_editored.csv")

data_xgb
wseed = data_xgb["Seed_x"]

lseed = data_xgb["Seed_y"]

Wseed  = np.zeros([data_xgb.shape[0]])

Lseed = np.zeros([data_xgb.shape[0]])

for i in range(data_xgb.shape[0]):

    Wseed[i] = wseed[i][1:3]

    Lseed[i] = lseed[i][1:3]

data_xgb["Seed_x"] = Wseed

data_xgb["Seed_y"] = Lseed
data_xgb["Score_diff"] = data_xgb["Score_x"]-data_xgb["Score_y"]

data_xgb["FGM_diff"] = data_xgb["FGM_x"]-data_xgb["FGM_y"]

data_xgb["FGA_diff"] = data_xgb["FGA_x"]-data_xgb["FGA_y"]

data_xgb["FGM3_diff"] = data_xgb["FGM3_x"]-data_xgb["FGM3_y"]

data_xgb["FGA3_diff"] = data_xgb["FGA3_x"]-data_xgb["FGA3_y"]

data_xgb["FTM_diff"] = data_xgb["FTM_x"]-data_xgb["FTM_y"]

data_xgb["FTA_diff"] = data_xgb["FTA_x"]-data_xgb["FTA_y"]

data_xgb["OR_diff"] = data_xgb["OR_x"]-data_xgb["OR_y"]

data_xgb["DR_diff"] = data_xgb["DR_x"]-data_xgb["DR_y"]

data_xgb["TO_diff"] = data_xgb["TO_x"]-data_xgb["TO_y"]

data_xgb["Stl_diff"] = data_xgb["Stl_x"]-data_xgb["Stl_y"]

data_xgb["Blk_diff"] = data_xgb["Blk_x"]-data_xgb["Blk_y"]

data_xgb["PF_diff"] = data_xgb["PF_x"]-data_xgb["PF_y"]

data_xgb["Seed_diff"] = data_xgb["Seed_x"]-data_xgb["Seed_y"]

trainlabel = data_xgb[data_xgb['Season']<2019]['target']

testlabel = data_xgb[data_xgb['Season']==2019]['target']

trainlabel
traindata = data_xgb[data_xgb['Season']<2019]

testdata = data_xgb[data_xgb['Season']==2019]
droplist = ["Unnamed: 0","LTeamID","WTeamID","WLoc","target","WScore","Score_x","FGM_x","FGA_x","FGM3_x","FGA3_x",

            "LScore","FTM_x","FTA_x","OR_x","DR_x","TO_x","Stl_x","Blk_x",

           "PF_x","Score_y","FGM_y","FGA_y","FGM3_y","FGA3_y","FTM_y","FTA_y",

            "OR_y","DR_y","TO_y","Stl_y","Blk_y","PF_y","Seed_x","Seed_y"]

traindata.drop(droplist,axis=1,inplace = True)

testdata.drop(droplist,axis=1,inplace = True)

traindata
xg_reg = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.8, learning_rate = 0.001,

                max_depth = 10, alpha = 7, n_estimators = 50000)

xg_reg.fit(traindata,trainlabel)
preds  = xg_reg.predict(testdata)

preds = np.floor(preds+0.5)

np.sum(preds==testlabel)/testdata.shape[0]
xgb.plot_tree(xg_reg,num_trees=0)

plt.rcParams['figure.figsize'] = [500, 400]

plt.show()
# xgb.plot_importance(xg_reg)

#plt.rcParams['figure.figsize'] = [500,400]

# plt.show()
data_xgb = pd.read_csv("W_result_by_game_tourney_editored.csv")



wseed = data_xgb["Seed_x"]

lseed = data_xgb["Seed_y"]

Wseed  = np.zeros([data_xgb.shape[0]])

Lseed = np.zeros([data_xgb.shape[0]])

for i in range(data_xgb.shape[0]):

    Wseed[i] = wseed[i][1:3]

    Lseed[i] = lseed[i][1:3]

data_xgb["Seed_x"] = Wseed

data_xgb["Seed_y"] = Lseed
data_xgb[['Score_x','Score_y']]
data_xgb["Score_diff"] = data_xgb["Score_x"]-data_xgb["Score_y"]

data_xgb["FGM_diff"] = data_xgb["FGM_x"]-data_xgb["FGM_y"]

data_xgb["FGA_diff"] = data_xgb["FGA_x"]-data_xgb["FGA_y"]

data_xgb["FGM3_diff"] = data_xgb["FGM3_x"]-data_xgb["FGM3_y"]

data_xgb["FGA3_diff"] = data_xgb["FGA3_x"]-data_xgb["FGA3_y"]

data_xgb["FTM_diff"] = data_xgb["FTM_x"]-data_xgb["FTM_y"]

data_xgb["FTA_diff"] = data_xgb["FTA_x"]-data_xgb["FTA_y"]

data_xgb["OR_diff"] = data_xgb["OR_x"]-data_xgb["OR_y"]

data_xgb["DR_diff"] = data_xgb["DR_x"]-data_xgb["DR_y"]

data_xgb["TO_diff"] = data_xgb["TO_x"]-data_xgb["TO_y"]

data_xgb["Stl_diff"] = data_xgb["Stl_x"]-data_xgb["Stl_y"]

data_xgb["Blk_diff"] = data_xgb["Blk_x"]-data_xgb["Blk_y"]

data_xgb["PF_diff"] = data_xgb["PF_x"]-data_xgb["PF_y"]

data_xgb["Seed_diff"] = data_xgb["Seed_x"]-data_xgb["Seed_y"]

data_xgb['Score_y']
trainlabel = data_xgb[data_xgb['Season']<2019]['target']

testlabel = data_xgb[data_xgb['Season']==2019]['target']

trainlabel
traindata = data_xgb[data_xgb['Season']<2019]

testdata = data_xgb[data_xgb['Season']==2019]
traindata
droplist = ["Unnamed: 0","LTeamID","WTeamID","WLoc","target","WScore","Score_x","FGM_x","FGA_x","FGM3_x","FGA3_x",

            "LScore","FTM_x","FTA_x","OR_x","DR_x","TO_x","Stl_x","Blk_x",

           "PF_x","Score_y","FGM_y","FGA_y","FGM3_y","FGA3_y","FTM_y","FTA_y",

            "OR_y","DR_y","TO_y","Stl_y","Blk_y","PF_y","Seed_x","Seed_y"]

traindata.drop(droplist,axis=1,inplace = True)

testdata.drop(droplist,axis=1,inplace = True)

traindata
xg_reg = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.8, learning_rate = 0.001,

                max_depth = 10, alpha = 7, n_estimators = 50000)

xg_reg.fit(traindata,trainlabel)
preds  = xg_reg.predict(testdata)

preds = np.floor(preds+0.5)

np.sum(preds==testlabel)/testdata.shape[0]
#xgb.plot_tree(xg_reg,num_trees=0)

#plt.rcParams['figure.figsize'] = [500, 400]

#plt.show()
# xgb.plot_importance(xg_reg)

#plt.rcParams['figure.figsize'] = [500,400]

# plt.show()
!pip install pygam
from pygam import LinearGAM,f,s,l

import eli5

from eli5.sklearn import PermutationImportance
X=x_train

y=y_train
# perform LASSO CV

# Note that the regularization strength is denoted by alpha in sklearn.

from sklearn import linear_model

cv = 10

lassocv = linear_model.LassoCV(cv=cv)

lassocv.fit(X, y)

print('alpha =',lassocv.alpha_.round(4))
# draw solution path

alphas = np.logspace(-8,1,21)

alphas_lassocv, coefs_lassocv, _ = lassocv.path(X, y, alphas=alphas)

log_alphas_lassocv = np.log10(alphas_lassocv)



plt.figure(figsize=(12,8)) 

plt.plot(log_alphas_lassocv,coefs_lassocv.T)

plt.vlines(x=np.log10(lassocv.alpha_), ymin=np.min(coefs_lassocv), ymax=np.max(coefs_lassocv), 

           color='b',linestyle='-.',label = 'alpha chosen')

plt.axhline(y=0, color='black',linestyle='--')

plt.xlabel(r'$\log_{10}(\alpha)$', fontsize=12)

plt.ylabel(r'$\hat{\beta}$', fontsize=12, rotation=0)

plt.title('Solution Path',fontsize=12)

plt.legend()

plt.show()


# fit a multiple layer percepton (neural network)

from sklearn.neural_network import MLPClassifier

names=df_train.drop('target',axis=1).columns

names=list(names)

clf = MLPClassifier(max_iter=1000, random_state=0)

clf.fit(X, y)

# define a permutation importance object

perm = PermutationImportance(clf).fit(X, y)

# show the importance

eli5.show_weights(perm, feature_names=names)
df_train[['Seeddiff','Score_y','FGM_y','FGA_y','FTM_x','FTA_x']].values
x_train =df_train[['Seeddiff','Score_y','FGM_y','FGA_y','FTM_x','FTA_x']].values

x_test =df_test[['Seeddiff','Score_y','FGM_y','FGA_y','FTM_x','FTA_x']].values
names = ['Seeddiff','Score_y','FGM_y','FGA_y','FTM_x','FTA_x']

names
from pygam import LogisticGAM,f,s,l



gam = LogisticGAM().fit(x_train,y_train)

# f: factor term

# some parameters combinations in grid search meet the error exception.

gam.gridsearch(x_train,y_train)
# plotting

fig, axs = plt.subplots(2,3,figsize=(20,8))

for i, ax in enumerate(axs.flatten()):

    XX = gam.generate_X_grid(term=i)

    plt.subplot(ax)

    plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))

    plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='grey', ls='--')

    plt.title(names[i])

plt.tight_layout()
y_pred_train = gam.predict(x_train)

y_pred_test = gam.predict(x_test)



print('The Acc on training set:',accuracy_score(y_train,y_pred_train))

print('The Acc on testing set:',accuracy_score(y_test,y_pred_test))
X=x_train_W

y=y_train_W
# perform LASSO CV

# Note that the regularization strength is denoted by alpha in sklearn.

from sklearn import linear_model

cv = 10

lassocv = linear_model.LassoCV(cv=cv)

lassocv.fit(X, y)

print('alpha =',lassocv.alpha_.round(4))
# draw solution path

alphas = np.logspace(-8,1,21)

alphas_lassocv, coefs_lassocv, _ = lassocv.path(X, y, alphas=alphas)

log_alphas_lassocv = np.log10(alphas_lassocv)



plt.figure(figsize=(12,8)) 

plt.plot(log_alphas_lassocv,coefs_lassocv.T)

plt.vlines(x=np.log10(lassocv.alpha_), ymin=np.min(coefs_lassocv), ymax=np.max(coefs_lassocv), 

           color='b',linestyle='-.',label = 'alpha chosen')

plt.axhline(y=0, color='black',linestyle='--')

plt.xlabel(r'$\log_{10}(\alpha)$', fontsize=12)

plt.ylabel(r'$\hat{\beta}$', fontsize=12, rotation=0)

plt.title('Solution Path',fontsize=12)

plt.legend()

plt.show()
# fit a multiple layer percepton (neural network)

from sklearn.neural_network import MLPClassifier

names=df_train.drop('target',axis=1).columns

names=list(names)

clf = MLPClassifier(max_iter=1000, random_state=0)

clf.fit(X, y)

# define a permutation importance object

perm = PermutationImportance(clf).fit(X, y)

# show the importance

eli5.show_weights(perm, feature_names=names)
x_train =df_train[['Seeddiff','Score_y','FGM_y','FGA_y','FTM_x','FTA_x']].values

x_test =df_test[['Seeddiff','Score_y','FGM_y','FGA_y','FTM_x','FTA_x']].values
names = ['Seeddiff','Score_y','FGM_y','FGA_y','FTM_x','FTA_x']

names
gam = LogisticGAM().fit(x_train,y_train)

# f: factor term

# some parameters combinations in grid search meet the error exception.

gam.gridsearch(x_train_W,y_train_W)
# plotting

fig, axs = plt.subplots(2,3,figsize=(20,8))

for i, ax in enumerate(axs.flatten()):

    XX = gam.generate_X_grid(term=i)

    plt.subplot(ax)

    plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))

    plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='grey', ls='--')

    plt.title(names[i])

plt.tight_layout()
y_pred_train = gam.predict(x_train)

y_pred_test = gam.predict(x_test)



print('The Acc on training set:',accuracy_score(y_train,y_pred_train))

print('The Acc on testing set:',accuracy_score(y_test,y_pred_test))
names=df_train.drop('target',axis=1).columns

names=list(names)
DIRC_MENS_PATH_player = '../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data'

DIRC_WOMENS_PATH_player = '../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Womens-Data'
M_players = pd.read_csv(f'{DIRC_MENS_PATH_player}/MPlayers.csv')

W_players = pd.read_csv(f'{DIRC_WOMENS_PATH_player}/WPlayers.csv')

W_players.head()
def high_order_stats(event_by_player):

    event_by_player['Points'] = event_by_player['made1']+2*event_by_player['made2']+3*event_by_player['made3']

    element_list = ['assist','block','foul','made1','made2','made3','miss1','miss2','miss3',

               'reb','steal','sub','timeout','turnover','Points']

    for item in element_list:

        event_by_player[item] = event_by_player[item]/event_by_player['count']

    event_by_player['Field_goal'] = (event_by_player['made2']+

                                 event_by_player['made3'])/(event_by_player['made2']+

                                                            event_by_player['made3']+

                                                           event_by_player['miss2']+

                                                            event_by_player['miss3'])

    event_by_player['FT_goal'] = event_by_player['made1']/(event_by_player['miss1']+event_by_player['made1'])

    event_by_player['3PT'] = event_by_player['made3']/(event_by_player['miss3']+event_by_player['made3'])

    # assist vs turnover

    event_by_player['AT'] = event_by_player['assist']/event_by_player['turnover']

    event_by_player['eFG'] = (event_by_player['made2']+

                          0.5*event_by_player['made3'])/(event_by_player['made2']+

                                                         event_by_player['made3']+

                                                         event_by_player['miss2']+

                                                         event_by_player['miss3'])

    event_by_player['TS'] = event_by_player['Points']/(2*((event_by_player['made2']+

                                                         event_by_player['made3']+

                                                         event_by_player['miss2']+

                                                         event_by_player['miss3'])

                                                      +0.44*(event_by_player['made1']+

                                                         event_by_player['made1'])))

    event_by_player['PER'] = (event_by_player['Points']+event_by_player['assist']+event_by_player['reb']+event_by_player['steal']+

                         event_by_player['block']-event_by_player['miss1']-event_by_player['turnover'])/event_by_player['count']

    return event_by_player

    



    
mkdir('mens_stats')

mkdir('womens_stats')
# year of stats



year = 2015

for year in range(2015,2020):

    M_events = pd.read_csv(f'{DIRC_MENS_PATH_player}/MEvents{year}.csv')

    W_events = pd.read_csv(f'{DIRC_WOMENS_PATH_player}/WEvents{year}.csv')

    M_game_made = M_events[['DayNum','EventPlayerID']]

    W_game_made = W_events[['DayNum','EventPlayerID']]

    M_game_made['count'] = 0

    W_game_made['count'] = 0

    M_game_made = M_game_made.drop_duplicates()

    W_game_made = W_game_made.drop_duplicates()

    M_game_made = M_game_made.groupby(['EventPlayerID']).count()

    M_game_made.reset_index(inplace = True)

    W_game_made = W_game_made.groupby(['EventPlayerID']).count()

    W_game_made.reset_index(inplace = True)

    M_game_made.drop(index = 0,inplace = True)

    W_game_made.drop(index = 0,inplace = True)

    M_game_made = M_game_made[['EventPlayerID','count']]

    W_game_made = W_game_made[['EventPlayerID','count']]

    M_events_useful = M_events[['EventPlayerID','EventType']]

    W_events_useful = W_events[['EventPlayerID','EventType']]

    M_events_useful['count'] = 0

    W_events_useful['count'] = 0

    M_events_useful = M_events_useful.groupby(['EventPlayerID','EventType']).count()

    W_events_useful = W_events_useful.groupby(['EventPlayerID','EventType']).count()

    M_events_reindex = M_events_useful.reset_index()

    W_events_reindex = W_events_useful.reset_index()

    M_events_pivoted=M_events_reindex.pivot('EventPlayerID', 'EventType', 'count')

    W_events_pivoted=W_events_reindex.pivot('EventPlayerID', 'EventType', 'count')

    M_event_by_player = M_events_pivoted.fillna(0)

    W_event_by_player = W_events_pivoted.fillna(0)

    M_event_by_player = M_event_by_player.drop(index = 0)

    W_event_by_player = W_event_by_player.drop(index = 0)

    M_event_by_player = pd.merge(M_event_by_player, M_game_made,on = 'EventPlayerID')

    W_event_by_player = pd.merge(W_event_by_player, W_game_made,on = 'EventPlayerID')

    M_event_by_player = high_order_stats(M_event_by_player)

    W_event_by_player = high_order_stats(W_event_by_player)

    M_players.rename(columns = {'PlayerID':'EventPlayerID'},inplace = True)

    W_players.rename(columns = {'PlayerID':'EventPlayerID'},inplace = True)



    M_player_stats = pd.merge(M_players,M_event_by_player,on = 'EventPlayerID',how = 'left')

    W_player_stats = pd.merge(W_players,W_event_by_player,on = 'EventPlayerID',how = 'left')

    M_player_stats = M_player_stats.fillna(0)

    W_player_stats = W_player_stats.fillna(0)

    

    

    M_player_stats.to_csv(f'mens_stats/M_player_stats_{year}.csv')

    W_player_stats.to_csv(f'womens_stats/W_player_stats_{year}.csv')
def pre_processer(df):

    player_information=df.iloc[:,0:4]

    EventPlayerID=player_information['EventPlayerID']

    df=df.drop(['EventPlayerID','LastName','FirstName'],axis=1)

    df[df==0]=np.nan

    pd.isnull(df)

    df=df.dropna(how='all')

    df.tail()

    df.insert(0,'EventPlayerID',EventPlayerID)

    df=pd.merge(player_information, df, on='EventPlayerID')

    return df
def Team_member(df):

    number_of_teamplayer=Counter(df['TeamID'])

    #number=list(number_of_tramplayer)

    number=number_of_teamplayer.values()

    number=list(number)

    np.shape(number)

    

    return number
def TeamID_made(df):

    TeamID=list(df.drop_duplicates(['TeamID']).TeamID)

    k=len(TeamID)

    

    return TeamID
def calculator_advanced(i,df,TeamID): 

    df1=df[df['TeamID']==TeamID[i]][['AT','eFG','TS','PER']].sum()

    #df_sum=df1.append([df2,df3,df4],ignore_index = False)

    df_sum=pd.DataFrame(df1,columns=[TeamID[i]])

    return df_sum
def sum_final(df,k,TeamID):

    df[df['AT']>10000]=0

    # k=len(TeamID)

    df_sum_final=calculator_advanced(0,df,TeamID)

    for i in range (1,k):

        df_temp=calculator_advanced(i,df,TeamID)

        df_sum_final=pd.concat([df_sum_final,df_temp],axis=1)

        #df_sum_final=df_sum_final.append([df_temp],ignore_index = False)

    return df_sum_final
from collections import Counter

import warnings

warnings.filterwarnings('ignore')



df_2015=pd.read_csv("mens_stats/M_player_stats_2015.csv")

df_2016=pd.read_csv("mens_stats/M_player_stats_2016.csv")

df_2017=pd.read_csv("mens_stats/M_player_stats_2017.csv")

df_2018=pd.read_csv("mens_stats/M_player_stats_2018.csv")

df_2019=pd.read_csv("mens_stats/M_player_stats_2019.csv")
def pre_processer(df):

    player_information=df.iloc[:,0:4]

    EventPlayerID=player_information['EventPlayerID']

    df=df.drop(['EventPlayerID','LastName','FirstName'],axis=1)

    df[df==0]=np.nan

    pd.isnull(df)

    df=df.dropna(how='all')

    df.tail()

    df.insert(0,'EventPlayerID',EventPlayerID)

    df=pd.merge(player_information, df, on='EventPlayerID')

    return df



def Team_member(df):

    number_of_teamplayer=Counter(df['TeamID'])

    #number=list(number_of_tramplayer)

    number=number_of_teamplayer.values()

    number=list(number)

    np.shape(number)

    return number





def TeamID_made(df):

    TeamID=list(df.drop_duplicates(['TeamID']).TeamID)

    k=len(TeamID)

    

    return TeamID



def calculator_advanced(i,df,TeamID): 

    df1=df[df['TeamID']==TeamID[i]][['AT','eFG','TS','PER']].sum()

    #df_sum=df1.append([df2,df3,df4],ignore_index = False)

    df_sum=pd.DataFrame(df1,columns=[TeamID[i]])

    return df_sum





def calculator_advanced(i,df,TeamID): 

    df1=df[df['TeamID']==TeamID[i]][['AT','eFG','TS','PER']].sum()

    #df_sum=df1.append([df2,df3,df4],ignore_index = False)

    df_sum=pd.DataFrame(df1,columns=[TeamID[i]])

    return df_sum



def sum_final(df,k,TeamID):

    df[df['AT']>10000]=0

    # k=len(TeamID)

    df_sum_final=calculator_advanced(0,df,TeamID)

    for i in range (1,k):

        df_temp=calculator_advanced(i,df,TeamID)

        df_sum_final=pd.concat([df_sum_final,df_temp],axis=1)

        #df_sum_final=df_sum_final.append([df_temp],ignore_index = False)

    return df_sum_final
df_2015=pre_processer(df_2015)

number_2015=Team_member(df_2015)

TeamID_2015=TeamID_made(df_2015)

k_2015=len(TeamID_2015)

df_sum_final_2015=sum_final(df_2015,k_2015,TeamID_2015)

for i in range(k_2015):

    df_sum_final_2015.iloc[:,i]=df_sum_final_2015.iloc[:,i]/number_2015[i]



df_2016=pre_processer(df_2016)

number_2016=Team_member(df_2016)

TeamID_2016=TeamID_made(df_2016)

k_2016=len(TeamID_2016)

df_sum_final_2016=sum_final(df_2016,k_2016,TeamID_2016)

for i in range(k_2016):

    df_sum_final_2016.iloc[:,i]=df_sum_final_2016.iloc[:,i]/number_2016[i]



df_2017=pre_processer(df_2017)

number_2017=Team_member(df_2017)

TeamID_2017=TeamID_made(df_2017)

k_2017=len(TeamID_2017)

df_sum_final_2017=sum_final(df_2017,k_2017,TeamID_2017)

for i in range(k_2017):

    df_sum_final_2017.iloc[:,i]=df_sum_final_2017.iloc[:,i]/number_2017[i]

    

df_2018=pre_processer(df_2018)

number_2018=Team_member(df_2018)

TeamID_2018=TeamID_made(df_2018)

k_2018=len(TeamID_2018)

df_sum_final_2018=sum_final(df_2018,k_2018,TeamID_2018)

for i in range(k_2018):

    df_sum_final_2018.iloc[:,i]=df_sum_final_2018.iloc[:,i]/number_2018[i]



df_2019=pre_processer(df_2019)

number_2019=Team_member(df_2019)

TeamID_2019=TeamID_made(df_2019)

k_2019=len(TeamID_2019)

df_sum_final_2019=sum_final(df_2019,k_2019,TeamID_2019)

for i in range(k_2019):

    df_sum_final_2019.iloc[:,i]=df_sum_final_2019.iloc[:,i]/number_2019[i]

df_sum_final_2019.tail()
mkdir('player_as')
df_sum_final_2015.to_csv('player_as/df_sum_final_2015.csv')

df_sum_final_2016.to_csv('player_as/df_sum_final_2016.csv')

df_sum_final_2017.to_csv('player_as/df_sum_final_2017.csv')

df_sum_final_2018.to_csv('player_as/df_sum_final_2018.csv')

df_sum_final_2019.to_csv('player_as/df_sum_final_2019.csv')
df_target=pd.read_csv("M_result_by_game_tourney_editored.csv")
df_target=df_target[['Season','WTeamID','LTeamID','target']]

df_target.tail()
def preprocess_win(df,i,TeamID):

    df_new=df.T

    df_new.columns=[['AT_win','eFG_win','TS_win','PER_win']]

    df_new['Season']=i

    df_new['WTeamID']=TeamID

    

    return df_new
def preprocess_lose(df,i,TeamID):

    df_new=df.T

    df_new.columns=[['AT_lose','eFG_lose','TS_lose','PER_lose']]

    df_new['Season']=i

    df_new['LTeamID']=TeamID

    

    return df_new
df_2015_new_win=preprocess_win(df_sum_final_2015,2015,TeamID_2015)

df_2016_new_win=preprocess_win(df_sum_final_2016,2016,TeamID_2016)

df_2017_new_win=preprocess_win(df_sum_final_2017,2017,TeamID_2017)

df_2018_new_win=preprocess_win(df_sum_final_2018,2018,TeamID_2018)

df_2019_new_win=preprocess_win(df_sum_final_2019,2019,TeamID_2019)

df_2015_new_lose=preprocess_lose(df_sum_final_2015,2015,TeamID_2015)

df_2016_new_lose=preprocess_lose(df_sum_final_2016,2016,TeamID_2016)

df_2017_new_lose=preprocess_lose(df_sum_final_2017,2017,TeamID_2017)

df_2018_new_lose=preprocess_lose(df_sum_final_2018,2018,TeamID_2018)

df_2019_new_lose=preprocess_lose(df_sum_final_2019,2019,TeamID_2019)
df1=df_2015_new_win.append(df_2016_new_win)

df2=df1.append(df_2017_new_win)

df3=df2.append(df_2018_new_win)

df4=df3.append(df_2019_new_win)

df_final_win=df4

df_final_win
df1=df_2015_new_lose.append(df_2016_new_lose)

df2=df1.append(df_2017_new_lose)

df3=df2.append(df_2018_new_lose)

df4=df3.append(df_2019_new_lose)

df_final_lose=df4

df_final_lose
df_final_win.columns=df_final_win.columns.get_level_values(0)

df_final_win.columns
#第一句话不被运行是第一次的时候才用：

df_target_new=pd.merge(df_target,df_final_win,on = ['Season','WTeamID'],how='left')

df_target_new.tail()
df_final_lose.columns=df_final_lose.columns.get_level_values(0)

df_final_lose.columns
df_target_new=pd.merge(df_target_new,df_final_lose,on = ['Season','LTeamID'],how='left')

df_target_new.tail()
df_target_new.to_csv('with_advanced_stat.csv')
df_2015=pd.read_csv("womens_stats/W_player_stats_2015.csv")

df_2016=pd.read_csv("womens_stats/W_player_stats_2016.csv")

df_2017=pd.read_csv("womens_stats/W_player_stats_2017.csv")

df_2018=pd.read_csv("womens_stats/W_player_stats_2018.csv")

df_2019=pd.read_csv("womens_stats/W_player_stats_2019.csv")
df_2015=pre_processer(df_2015)

number_2015=Team_member(df_2015)

TeamID_2015=TeamID_made(df_2015)

k_2015=len(TeamID_2015)

df_sum_final_2015=sum_final(df_2015,k_2015,TeamID_2015)

for i in range(k_2015):

    df_sum_final_2015.iloc[:,i]=df_sum_final_2015.iloc[:,i]/number_2015[i]



df_2016=pre_processer(df_2016)

number_2016=Team_member(df_2016)

TeamID_2016=TeamID_made(df_2016)

k_2016=len(TeamID_2016)

df_sum_final_2016=sum_final(df_2016,k_2016,TeamID_2016)

for i in range(k_2016):

    df_sum_final_2016.iloc[:,i]=df_sum_final_2016.iloc[:,i]/number_2016[i]



df_2017=pre_processer(df_2017)

number_2017=Team_member(df_2017)

TeamID_2017=TeamID_made(df_2017)

k_2017=len(TeamID_2017)

df_sum_final_2017=sum_final(df_2017,k_2017,TeamID_2017)

for i in range(k_2017):

    df_sum_final_2017.iloc[:,i]=df_sum_final_2017.iloc[:,i]/number_2017[i]

    

df_2018=pre_processer(df_2018)

number_2018=Team_member(df_2018)

TeamID_2018=TeamID_made(df_2018)

k_2018=len(TeamID_2018)

df_sum_final_2018=sum_final(df_2018,k_2018,TeamID_2018)

for i in range(k_2018):

    df_sum_final_2018.iloc[:,i]=df_sum_final_2018.iloc[:,i]/number_2018[i]



df_2019=pre_processer(df_2019)

number_2019=Team_member(df_2019)

TeamID_2019=TeamID_made(df_2019)

k_2019=len(TeamID_2019)

df_sum_final_2019=sum_final(df_2019,k_2019,TeamID_2019)

for i in range(k_2019):

    df_sum_final_2019.iloc[:,i]=df_sum_final_2019.iloc[:,i]/number_2019[i]

df_sum_final_2019.tail()
df_sum_final_2015.to_csv('player_as/W_df_sum_final_2015.csv')

df_sum_final_2016.to_csv('player_as/W_df_sum_final_2016.csv')

df_sum_final_2017.to_csv('player_as/W_df_sum_final_2017.csv')

df_sum_final_2018.to_csv('player_as/W_df_sum_final_2018.csv')

df_sum_final_2019.to_csv('player_as/W_df_sum_final_2019.csv')
df_target=pd.read_csv("W_result_by_game_tourney_editored.csv")
df_target=df_target[['Season','WTeamID','LTeamID','target']]

df_target.tail()
df_2015_new_win=preprocess_win(df_sum_final_2015,2015,TeamID_2015)

df_2016_new_win=preprocess_win(df_sum_final_2016,2016,TeamID_2016)

df_2017_new_win=preprocess_win(df_sum_final_2017,2017,TeamID_2017)

df_2018_new_win=preprocess_win(df_sum_final_2018,2018,TeamID_2018)

df_2019_new_win=preprocess_win(df_sum_final_2019,2019,TeamID_2019)

df_2015_new_lose=preprocess_lose(df_sum_final_2015,2015,TeamID_2015)

df_2016_new_lose=preprocess_lose(df_sum_final_2016,2016,TeamID_2016)

df_2017_new_lose=preprocess_lose(df_sum_final_2017,2017,TeamID_2017)

df_2018_new_lose=preprocess_lose(df_sum_final_2018,2018,TeamID_2018)

df_2019_new_lose=preprocess_lose(df_sum_final_2019,2019,TeamID_2019)
df1=df_2015_new_win.append(df_2016_new_win)

df2=df1.append(df_2017_new_win)

df3=df2.append(df_2018_new_win)

df4=df3.append(df_2019_new_win)

df_final_win=df4

df_final_win
df1=df_2015_new_lose.append(df_2016_new_lose)

df2=df1.append(df_2017_new_lose)

df3=df2.append(df_2018_new_lose)

df4=df3.append(df_2019_new_lose)

df_final_lose=df4

df_final_lose
df_final_win.columns=df_final_win.columns.get_level_values(0)

df_final_win.columns
#第一句话不被运行是第一次的时候才用：

df_target_new=pd.merge(df_target,df_final_win,on = ['Season','WTeamID'],how='left')

df_target_new.tail()
df_final_lose.columns=df_final_lose.columns.get_level_values(0)

df_final_lose.columns
df_target_new_W=pd.merge(df_target_new,df_final_lose,on = ['Season','LTeamID'],how='left')

df_target_new_W.tail()
df_target_new_W.to_csv('W_with_advanced_stat.csv')
df_target_new_W
df=df_target_new

#order=['Season','WTeamID','LTeamID','AT_win','eFG_win','TS_win','PER_win','AT_lose','eFG_lose','TS_lose','PER_lose','target']

target=df['target']

df=df.drop('target',axis=1)

df['target']=target

df.tail()
df_plot=df[['target','AT_win','eFG_win','TS_win','PER_win']]

sns.pairplot(df_plot, hue="target", size=3, diag_kind="kde")
# Split the data 

df_train=df[df['Season']<2019]

df_test=df[df['Season']>2018]



df_train=df_train.drop(['Season','WTeamID','LTeamID'],axis=1)

df_test=df_test.drop(['Season','WTeamID','LTeamID'],axis=1)



m,n=np.shape(df_train)



x_train = df_train.iloc[:,0:n-1].values

y_train = df_train.target.values



x_test = df_test.iloc[:,0:n-1].values

y_test = df_test.target.values
df=df_target_new_W

#order=['Season','WTeamID','LTeamID','AT_win','eFG_win','TS_win','PER_win','AT_lose','eFG_lose','TS_lose','PER_lose','target']

target=df['target']

df=df.drop('target',axis=1)

df['target']=target

df.tail()
df_plot=df[['target','AT_win','eFG_win','TS_win','PER_win']]

sns.pairplot(df_plot, hue="target", size=3, diag_kind="kde")
# Split the data 

df_train=df[df['Season']<2019]

df_test=df[df['Season']>2018]



df_train=df_train.drop(['Season','WTeamID','LTeamID'],axis=1)

df_test=df_test.drop(['Season','WTeamID','LTeamID'],axis=1)



m,n=np.shape(df_train)



x_train = df_train.iloc[:,0:n-1].values

y_train = df_train.target.values



x_test = df_test.iloc[:,0:n-1].values

y_test = df_test.target.values
df=df_target_new_W

#order=['Season','WTeamID','LTeamID','AT_win','eFG_win','TS_win','PER_win','AT_lose','eFG_lose','TS_lose','PER_lose','target']

target=df['target']

df=df.drop('target',axis=1)

df['target']=target
df_plot=df[['target','AT_win','eFG_win','TS_win','PER_win']]

sns.pairplot(df_plot, hue="target", size=3, diag_kind="kde")
# Split the data 

df_train=df[df['Season']<2019]

df_test=df[df['Season']>2018]



df_train=df_train.drop(['Season','WTeamID','LTeamID'],axis=1)

df_test=df_test.drop(['Season','WTeamID','LTeamID'],axis=1)



m,n=np.shape(df_train)



x_train = df_train.iloc[:,0:n-1].values

y_train = df_train.target.values



x_test = df_test.iloc[:,0:n-1].values

y_test = df_test.target.values
from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import accuracy_score

logreg = LogisticRegressionCV(cv=5,random_state=0, solver='newton-cg')

logreg.fit(x_train, y_train)



y_pred_train = logreg.predict(x_train)

y_pred_test = logreg.predict(x_test)



print("Coefficients :", np.round(logreg.intercept_,4), np.round(logreg.coef_,4))
y_pred_train = logreg.predict(x_train)

y_pred_test = logreg.predict(x_test)



accuracy_train = accuracy_score(y_train, y_pred_train)

accuracy_test = accuracy_score(y_test, y_pred_test)

print('Accuracy on the training set =', np.round(accuracy_train,4))

print('Accuracy on the test set =', np.round(accuracy_test,4))
df_2015=df_2015[['AT','eFG','TS','PER','TeamID']]

df_2016=df_2016[['AT','eFG','TS','PER','TeamID']]

df_2017=df_2017[['AT','eFG','TS','PER','TeamID']]

df_2018=df_2018[['AT','eFG','TS','PER','TeamID']]

df_2019=df_2019[['AT','eFG','TS','PER','TeamID']]
coef=np.round(logreg.coef_,4)

coef[0][0:4]
def project(df):

    df_player_influence=0

    for i in range(4):  

        df_player_influence=coef[0][i]*df.iloc[:,i]+df_player_influence

    df['player_influence']=df_player_influence

    return df
df_2015_new=project(df_2015)

df_2016_new=project(df_2016)

df_2017_new=project(df_2017)

df_2018_new=project(df_2018)

df_2019_new=project(df_2019)



df_2015_new.dropna(axis=0,how='any',inplace=True)

df_2016_new.dropna(axis=0,how='any',inplace=True)

df_2017_new.dropna(axis=0,how='any',inplace=True)

df_2018_new.dropna(axis=0,how='any',inplace=True)

df_2019_new.dropna(axis=0,how='any',inplace=True)
def sum_influence(df,k,TeamID,q,number):

    # k=len(TeamID)

    df_sum_final=[]

    df_sum_final.append(df[df['TeamID']==TeamID[0]]['player_influence'].sum())

    lists=['influence_2015','influence_2016','influence_2017','influence_2018','influence_2019']

    for i in range (1,k):

        df_temp=df[df['TeamID']==TeamID[i]]['player_influence'].sum()

        df_sum_final.append(df_temp)

        #df_sum_final=df_sum_final.append([df_temp],ignore_index = False)

    df_sum_final=pd.DataFrame(df_sum_final,columns=[lists[q]])

    for i in range(k):

        df_sum_final.loc[i]=df_sum_final.loc[i]/number[i]

    return df_sum_final
team_2015_influence=sum_influence(df_2015_new,k_2015,TeamID_2015,0,number_2015)

team_2015_influence['Season_2015']=2015

team_2015_influence['TeamID']=TeamID_2015



team_2016_influence=sum_influence(df_2016_new,k_2016,TeamID_2016,1,number_2016)

team_2016_influence['Season_2016']=2016

team_2016_influence['TeamID']=TeamID_2016



team_2017_influence=sum_influence(df_2017_new,k_2017,TeamID_2017,2,number_2017)

team_2017_influence['Season_2017']=2017

team_2017_influence['TeamID']=TeamID_2017



team_2018_influence=sum_influence(df_2018_new,k_2018,TeamID_2018,3,number_2018)

team_2018_influence['Season_2018']=2018

team_2018_influence['TeamID']=TeamID_2018



team_2019_influence=sum_influence(df_2019_new,k_2019,TeamID_2019,4,number_2019)

team_2019_influence['Season_2019']=2019

team_2019_influence['TeamID']=TeamID_2019
df_1=pd.merge(team_2015_influence,team_2016_influence,on=['TeamID'])

df_2=pd.merge(df_1,team_2017_influence,on=['TeamID'])

df_3=pd.merge(df_2,team_2018_influence,on=['TeamID'])

df_final_influence=pd.merge(df_3,team_2019_influence,on=['TeamID'])

df_final_influence=df_final_influence.drop(['Season_2015','Season_2016','Season_2017','Season_2018','Season_2019'],axis=1)

TeamID=df_final_influence['TeamID']

df_final_influence.drop(['TeamID'],axis=1,inplace=True)

df_final_influence.insert(0,'TeamID',TeamID)

df_final_influence.tail()
df_final_influence.to_csv('team_influence_time_series.csv')
!pip install pmdarima
import pmdarima as pm

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import adfuller



df_input=df_final_influence.iloc[:,1:6]

df_input=df_input.T

df_input.tail()





def auto_arima(df,i):

    model = pm.auto_arima(df.iloc[:,i], trace=False, error_action='ignore', suppress_warnings=True)

    model.fit(df.iloc[:,i])

    forecast = model.predict(n_periods=1)

    

    return forecast
m,n=np.shape(df_input)

result=[]

for i in range(n):

    temp=auto_arima(df_input,i)

    result.append(temp)
result=pd.DataFrame(result,columns=['prediction_2020'])

result=result.T

result
df_output=df_input.append(result)

df_output.tail()
df_output.to_csv('team_influence_prediction.csv')
!pip install chart_studio

!pip install bubbly
df_output_new=df_output.T

df_1=df_output_new[['influence_2015']]

df_1.rename(columns={'influence_2015': 'influence'}, inplace=True)

df_1['teamid']=TeamID

df_1['season']='Real'



df_2=df_output_new[['influence_2016']]

df_2.rename(columns={'influence_2016': 'influence'}, inplace=True)

df_2['teamid']=TeamID

df_2['season']='Real'



df_3=df_output_new[['influence_2017']]

df_3.rename(columns={'influence_2017': 'influence'}, inplace=True)

df_3['teamid']=TeamID

df_3['season']='Real'



df_4=df_output_new[['influence_2018']]

df_4.rename(columns={'influence_2018': 'influence'}, inplace=True)

df_4['teamid']=TeamID

df_4['season']='Real'



df_5=df_output_new[['influence_2019']]

df_5.rename(columns={'influence_2019': 'influence'}, inplace=True)

df_5['teamid']=TeamID

df_5['season']='Real'



df_6=df_output_new[['prediction_2020']]

df_6.rename(columns={'prediction_2020': 'influence'}, inplace=True)

df_6['teamid']=TeamID

df_6['season']='Prediction'



df_bubble=pd.concat([df_1,df_2,df_3,df_4,df_5,df_6],axis=0)

df_bubble



#df_bubble.iloc[:,0]=df_bubble.iloc[:,0]-min(df_bubble.iloc[:,0])

df_bubble.reset_index(drop=True, inplace=True)

df_bubble.sort_values(by='influence',ascending=False,inplace=True)

df_bubble_plot=df_bubble.head(100)

df_bubble_plot.tail()

from bubbly.bubbly import bubbleplot 

from plotly.offline import iplot

import chart_studio.plotly as py

figure = bubbleplot(dataset=df_bubble_plot, x_column='teamid', y_column='influence', 

                    bubble_column='season', size_column='influence', color_column='season', 

                    x_logscale=True, scale_bubble=2, height=350)



iplot(figure)
import seaborn as sns

df_influence = df_bubble['influence']

plt.figure(figsize=(8,6))

sns.set_style("darkgrid")

sns.kdeplot(data=df_influence,label="Team_Competitiveness" ,shade=True)
logreg = LogisticRegressionCV(cv=5,random_state=0, solver='newton-cg')

logreg.fit(x_train_W, y_train_W)



y_pred_train = logreg.predict(x_train_W)

y_pred_test = logreg.predict(x_test_W)



print("Coefficients :", np.round(logreg.intercept_,4), np.round(logreg.coef_,4))
df_2015=df_2015[['AT','eFG','TS','PER','TeamID']]

df_2016=df_2016[['AT','eFG','TS','PER','TeamID']]

df_2017=df_2017[['AT','eFG','TS','PER','TeamID']]

df_2018=df_2018[['AT','eFG','TS','PER','TeamID']]

df_2019=df_2019[['AT','eFG','TS','PER','TeamID']]