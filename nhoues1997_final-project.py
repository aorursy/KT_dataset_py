import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from tqdm.notebook import tqdm



from yellowbrick.regressor import ResidualsPlot

from sklearn.decomposition  import FactorAnalysis

from sklearn.decomposition import PCA

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn import model_selection

#implement linear regression

from sklearn.linear_model import LinearRegression,Lasso,Ridge

from sklearn.model_selection import train_test_split





train = pd.read_csv('../input/data-analysis-2020/train.csv')

test = pd.read_csv('../input/data-analysis-2020/test.csv')
train =train[train['next_day_ret']>-1000]

train['next_day_ret'].hist()
train.date=pd.to_datetime(train.date,format='%Y-%m-%d')

train['currReturn']=train.groupby(['company'])['close'].pct_change().fillna(0)

train['nextReturn']=train.groupby(['company'])['currReturn'].shift(-1) # defined in the course 
corrMatrix = train[['currReturn','nextReturn','next_day_ret']].corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
train = train[train['company']=='ADWYA']

y = train['next_day_ret'].values.reshape(-1,1)

x = train['nextReturn']

plt.scatter(x, y, alpha=0.5)

plt.title('nextReturn vs next_day_ret')

plt.xlabel('nextReturn')

plt.ylabel('next_day_ret')

plt.show()
train= train[train['company']=='ADWYA']

train= train[train['nextReturn']==0]

train['value'] = train['value'].fillna(train['value'].mean())

corrmat = train.corr()

k = 10 #number of variables for heatmap

cols = abs(corrmat).nlargest(k, 'next_day_ret')['next_day_ret'].index

cm = np.corrcoef(train[cols].values.T)

plt.figure(figsize=(20,20))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train = pd.read_csv('../input/data-analysis-2020/train.csv')

test = pd.read_csv('../input/data-analysis-2020/test.csv')

train =train[train['next_day_ret']>-1000] # drop outliers 

train_len = len(train)

all_data = pd.concat([train,test], ignore_index=True)
all_data.date=pd.to_datetime(all_data.date,format='%Y-%m-%d')

all_data['currReturn']=all_data.groupby(['company'])['close'].pct_change().fillna(0)

all_data['nextReturn']=all_data.groupby(['company'])['currReturn'].shift(-1) # defined in the course 
all_data['nextReturn'].hist()
all_data['currReturn_square'] = all_data['currReturn']**2
#Resduial analysis 

# Instantiate the linear model and visualizer

model = LinearRegression()

visualizer = ResidualsPlot(model)

data = all_data.iloc[:train_len]

data = data[data['company']=='ADWYA']

data = data[data['nextReturn']>-0.01]

X_train = data[['currReturn','currReturn_square']]

y_train = data['nextReturn']

visualizer.fit(X_train, y_train) 

visualizer.score(X_train, y_train)  # Fit the training data to the visualizer

AGRO_AlIMENTAIRE = ['LAND OR', 'STE TUN. DU SUCRE', 'SFBT', 'DELICE HOLDING', 'CEREALIS', 'POULINA GP HOLDING', 'SOPAT', 'ELBENE INDUSTRIE']

ATUO = ['ASSAD', 'STEQ', 'GIF-FILTER', 'STIP']

BANQUES =['BT', 'ATTIJARI BANK', 'BH', 'BIAT', 'STB', 'AMEN BANK', 'ATB', 'WIFACK INT BANK', 'UBCI', 'UIB', 'BNA', 'BTE (ADP)']

BATIMENT = ['SITS', 'SIMPAR', 'SANIMED', 'ESSOUKNA', 'SOTEMAIL', 'SOMOCER']

TECHNOLOGIE = ['SERVICOM', 'ONE TECH HOLDING', 'TELNET HOLDING', 'SOTETEL', 'EURO-CYCLES', 'AETECH', 'HEXABYTE']

SERVICES_FINANCIERS = ['MAGHREB INTERN PUB', 'HANNIBAL LEASE', 'ASS MULTI ITTIHAD', 'BEST LEASE', 'TAWASOL GP HOLDING', 'CIL', 'ATTIJARI LEASING', 'TUNINVEST-SICAR', 'MODERN LEASING', 'ATL', 'SPDIT - SICAF', 'PLAC. TSIE-SICAF', 'TUNISAIR']

ASSURANCES = ['STAR', 'TUNIS RE', 'ASTREE']

SERVICE_INDUS = ['MPBS', 'SOTUVER', 'NEW BODY LINE', 'ELECTROSTAR', 'SOTIPAPIER', 'SOTRAPIL', 'SIAME', 'ATELIER MEUBLE INT', 'AMS', 'OFFICEPLAST']

PHARMACEUTIC =['SIPHAT', 'ADWYA', 'ICF', 'UNIMED', 'ALKIMIA', 'AIR LIQUDE TSIE']

MAT_PREMIERES = ['TPR', 'CIMENTS DE BIZERTE', 'CARTHAGE CEMENT']

DISTRIBUTION =['UADH', 'SAH', 'CELLCOM', 'SOTUMAG', 'CITY CARS', 'MAGASIN GENERAL', 'MONOPRIX', 'ARTES', 'ENNAKL AUTOMOBILES', 'SITEX']
all_company = {'AGRO_AlIMENTAIRE+DISTRIBUTION+MAT_PREMIERES+SERVICE_INDUS':AGRO_AlIMENTAIRE+DISTRIBUTION+MAT_PREMIERES+SERVICE_INDUS

               ,'SERVICES_FINANCIERS+ASSURANCES+BANQUES':SERVICES_FINANCIERS+ASSURANCES+BANQUES

               ,'PHARMACEUTIC+TECHNOLOGIE+ATUO+BATIMENT':PHARMACEUTIC+TECHNOLOGIE+ATUO+BATIMENT}
company_map = dict()

for sector in all_company.keys() : 

  for company in all_company[sector] : 

    company_map[company] = sector

all_data['sector'] = all_data['company'].map(company_map)
## all_company current return 

return_data=all_data.pivot(index='date',columns='company',values='currReturn').replace([np.inf, -np.inf],0).fillna(0)

return_data.head()
all_data['date']=all_data['date'].astype('str')

# correlated sector 

sector_features = []

for se in all_company.keys() : 

  companies = all_company[se]

  for n_compent in range(1,6) :

    pca = PCA(n_components=n_compent)

    pca.fit(return_data[companies].values)

    if sum(pca.explained_variance_ratio_)> 0.70 : 

      break 

  print(f'{se}  : number of PC is {n_compent} and explain {100*sum(pca.explained_variance_ratio_)} % of the total  Data variance')

  components = pca.transform(return_data[companies].values)

  col = [se+'_pc_'+str(i) for i in range(n_compent)]

  sector_features += col

  new_compontes = pd.DataFrame(data = components , columns=col  )

  new_compontes['date'] = return_data.index.tolist()

  new_compontes['date'] = new_compontes['date'].astype(str)

  all_data =pd.merge(all_data,new_compontes,how = 'left' , on= 'date')

  all_data = all_data.fillna(0)
for stock in tqdm(all_data['company'].unique()) :

  index = all_data[all_data['company']==stock].index

  all_data.loc[index, 'Momentum_1D'] = (all_data.loc[index, 'currReturn'] - all_data.loc[index, 'currReturn'].shift(1)).fillna(0)

  all_data.loc[index, 'Momentum'] = (all_data.loc[index, 'currReturn'] - all_data.loc[index, 'currReturn'].shift(5)).fillna(0)
for company in all_data['company'].unique() : 

  sub_set = all_data[all_data['company']==company][['date','currReturn']]

  index = sub_set.index

  sub_set = sub_set.sort_values('date')

  for i in range(1,20) : 

    col = 'return( t-'+str(i)+' )'

    sub_set[col] = sub_set['currReturn'].shift(i)

    all_data.loc[index,col] = sub_set[col]

col = ['return( t-'+str(i)+' )' for i in range(1,6) ] + ['currReturn']

all_data['mean_last_week'] = all_data[col].mean(axis=1) 

all_data['std_last_week'] = all_data[col].std(axis=1)

all_data['max_last_week'] = all_data['mean_last_week'] +all_data['std_last_week']

all_data['min_last_week'] = all_data['mean_last_week'] -all_data['std_last_week']

col = ['return( t-'+str(i)+' )' for i in range(1,12) ] + ['currReturn']

all_data['mean_last_2week'] = all_data[col].mean(axis=1) 

all_data['std_last_2week'] = all_data[col].std(axis=1)

all_data['max_last_2week'] = all_data['mean_last_2week'] +all_data['std_last_2week']

all_data['min_last_2week'] = all_data['mean_last_2week'] -all_data['std_last_2week']

col = ['return( t-'+str(i)+' )' for i in range(1,20) ] + ['currReturn']

all_data['mean_last_month'] = all_data[col].mean(axis=1) 

all_data['std_last_month'] = all_data[col].std(axis=1)

all_data['max_last_month'] = all_data['mean_last_month'] +all_data['std_last_month']

all_data['min_last_month'] = all_data['mean_last_month'] -all_data['std_last_month']
last_mean = ['mean_last_week','max_last_week','min_last_week','mean_last_2week','max_last_2week','min_last_2week','mean_last_month','max_last_month','min_last_month']
for stock in tqdm(all_data['company'].unique()):

  index = all_data[all_data['company']==stock].index

  all_data.loc[index,'EMA'] = all_data.loc[index,'currReturn'].ewm(span=5,min_periods=1,adjust=True,ignore_na=False).mean()

  all_data.loc[index,'TEMA'] = (3 * all_data.loc[index,'EMA'] - 3 * all_data.loc[index,'EMA']) + (all_data.loc[index,'EMA']*all_data.loc[index,'EMA']*all_data.loc[index,'EMA'])

all_data['TEMA_square'] = all_data['TEMA']**2
for stock in tqdm(all_data['company'].unique()):

  index = all_data[all_data['company']==stock].index     

  all_data.loc[index,'26_ema'] =  all_data.loc[index,'currReturn'].ewm(span=26,min_periods=0,adjust=True,ignore_na=False).mean()

  all_data.loc[index,'12_ema'] =   all_data.loc[index,'currReturn'].ewm(span=12,min_periods=0,adjust=True,ignore_na=False).mean()

  all_data.loc[index,'MACD'] = all_data.loc[index,'12_ema'] -  all_data.loc[index,'26_ema']

def bbands(price, length=30, numsd=2):

    """ returns average, upper band, and lower band"""

    #ave = pd.stats.moments.rolling_mean(price,length)

    ave = price.rolling(window = length, center = False).mean()

    #sd = pd.stats.moments.rolling_std(price,length)

    sd = price.rolling(window = length, center = False).std()

    upband = ave + (sd*numsd)

    dnband = ave - (sd*numsd)

    return np.round(ave,3), np.round(upband,3), np.round(dnband,3)

for stock in tqdm(all_data['company'].unique()) :

    index = all_data[all_data['company']==stock].index

    all_data.loc[index,'BB_Middle_Band'],all_data.loc[index,'BB_Upper_Band'], all_data.loc[index,'BB_Lower_Band'] = bbands(all_data.loc[index,'currReturn'], length=14, numsd=1)

    all_data.loc[index,'BB_Middle_Band'] = all_data.loc[index,'BB_Middle_Band'].fillna(0)

    all_data.loc[index,'BB_Upper_Band'] = all_data.loc[index,'BB_Upper_Band'].fillna(0)

    all_data.loc[index,'BB_Lower_Band'] = all_data.loc[index,'BB_Lower_Band'].fillna(0)
def aroon(df, tf=25):

  aroonup = []

  aroondown = []

  x = tf

  while x< len(df['date']):

    aroon_up = ((df['currReturn'][x-tf:x].tolist().index(max(df['currReturn'][x-tf:x])))/float(tf))*100

    aroon_down = ((df['currReturn'][x-tf:x].tolist().index(min(df['currReturn'][x-tf:x])))/float(tf))*100

    aroonup.append(aroon_up)

    aroondown.append(aroon_down)

    x+=1

  return aroonup, aroondown
for stock in tqdm(all_data['company'].unique()) :

  index = all_data[all_data['company']==stock].index

  listofzeros = [0] * 25

  up, down = aroon(all_data.loc[index])

  aroon_list = [x - y for x, y in zip(up,down)]

  if len(aroon_list)==0:

    aroon_list = [0] *     all_data.loc[index].shape[0]

    all_data.loc[index,'Aroon_Oscillator'] = aroon_list

  else:

    all_data.loc[index,'Aroon_Oscillator'] = listofzeros+aroon_list
all_data['Aroon_current'] = (all_data['Aroon_Oscillator'])+all_data['currReturn']
def CCI(df, n, constant):

    TP = df['currReturn'] 

    CCI = pd.Series((TP - TP.rolling(window=n, center=False).mean()) / (constant * TP.rolling(window=n, center=False).std()))

    return CCI

for stock in tqdm(all_data['company'].unique()):

  index = all_data[all_data['company']==stock].index    

  all_data.loc[index,'CCI'] = CCI( all_data.loc[index], 20, 0.015)
all_data['CCI'] = all_data['CCI'].replace([np.inf],500)

all_data['CCI'] = all_data['CCI'].replace([-np.inf],-500)

all_data['CCI'] = all_data['CCI'] /50

all_data['CCI_current']  = all_data['currReturn'] * all_data['CCI']
all_data.date=pd.to_datetime(all_data.date,format='%Y-%m-%d')

all_data['year']=all_data['date'].dt.year 

all_data['month']=all_data['date'].dt.month 

all_data['dayofweek_num']=all_data['date'].dt.dayofweek  

all_data['quarter']=all_data['date'].dt.quarter

all_data['day']=all_data['date'].dt.day
new = all_data.groupby(['company','year','month'])['currReturn'].agg({'mean'})

new = new.reset_index()

name = 'Month_mean'

last_mean.append(name)

new.columns = ['company','year','month',name] 

all_data =pd.merge(all_data,new,how = 'left',on = ['company','year','month'])
get_time = pd.get_dummies(all_data[['quarter','month','dayofweek_num']].astype('str'))

all_data =pd.concat([all_data,get_time],axis=1)

time = get_time.columns.tolist()
train = all_data.iloc[:train_len].fillna(0)

test =  all_data.iloc[train_len:].copy()
num_features= ['MACD', 'BB_Middle_Band', 'BB_Upper_Band','EMA','Momentum']+last_mean+[ '12_ema', '26_ema']+['currReturn','currReturn_square','TEMA','Aroon_current','CCI_current','currReturn_square','TEMA_square']

for num in num_features : 

  scale = preprocessing.MinMaxScaler()

  scale.fit(train[num].values.reshape(-1,1))

  train[num] = scale.transform(train[num].values.reshape(-1,1))

  test[num] = scale.transform(test[num].values.reshape(-1,1))
def heatmap(train,target='next_day_ret',k=20) : 

  corrmat = train.corr()

  cols = abs(corrmat).nlargest(k, target)[target].index.tolist()

  if 'Month_mean' in cols : 

    cols.remove('Month_mean')

  cm = np.corrcoef(train[cols].values.T)

  plt.figure(figsize=(20,20))

  hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols, xticklabels=cols)

  plt.show()

  return cols



cols = heatmap(train)
feature_selected = [ 'next_day_ret','currReturn','currReturn_square','TEMA','Aroon_current','CCI_current','currReturn_square','TEMA_square']

tech = ['MACD', 'BB_Middle_Band', 'BB_Upper_Band','EMA','Momentum']

other = [ '12_ema', 'mean_last_week', 'max_last_week', '26_ema']

to_reduce = {'tech':tech , 'other':other}
def pca_function(train,test=[],to_reduce=to_reduce , deb=True ) : 



  pca_col = []

  train_data = train.copy()

  test_data = test.copy()

  for to_red in to_reduce.keys() : 

    features = to_reduce[to_red]

    df = train_data[features]

    df_test = test_data[features]

    for n_compent in range(len(features)-1) :

        pca = PCA(n_components=n_compent)

        pca.fit(df.values)

        if sum(pca.explained_variance_ratio_)> 0.95 : 

          break 

    if deb : 

      print(f'{to_red} : number of PC is {n_compent} and explain {100*sum(pca.explained_variance_ratio_)} % of the total  Data variance')

    components = pca.transform(df.values)

    col = [to_red+'_pc_'+str(i) for i in range(n_compent)]

    new_compontes = pd.DataFrame(data = components , columns=col  )

    new_compontes['index'] = df.index

    pca_col += col

    new_compontes = new_compontes.set_index('index')

    train_data[col] = new_compontes[col]



    if len(test) != 0: 

      components = pca.transform(df_test.values)

      new_compontes = pd.DataFrame(data = components , columns=col  )

      new_compontes['index'] = df_test.index

      new_compontes = new_compontes.set_index('index')

      test_data[col] = new_compontes[col]

    else : 

      test_data=-1

  return train_data,test_data,pca_col
def factor_analysis_function(train,test=[],to_reduce=to_reduce ) : 

  factor_col = []

  train_data = train.copy()

  test_data = test.copy()

  for to_red in to_reduce.keys() : 

    features = to_reduce[to_red]

    df = train_data[features]

    df_test = test_data[features]

    n_compent = len(features)//2 + 1 



    factor = FactorAnalysis(n_components=n_compent, random_state=0)

    factor = factor.fit(df.values)

    components = factor.transform(df.values)

    col = [to_red+'_factor_'+str(i) for i in range(n_compent)]

    new_compontes = pd.DataFrame(data = components , columns=col  )

    new_compontes['index'] = df.index

    factor_col += col

    new_compontes = new_compontes.set_index('index')

    train_data[col] = new_compontes[col]

    train_data.drop(features,1,inplace=True)



    if len(test) != 0: 

      components = factor.transform(df_test.values)

      new_compontes = pd.DataFrame(data = components , columns=col  )

      new_compontes['index'] = df_test.index

      new_compontes = new_compontes.set_index('index')

      test_data[col] = new_compontes[col]

      test_data.drop(features,1,inplace=True)

    else : 

      test_data=-1

  return train_data,test_data,factor_col
def simplelinear(train,valid): 

  "this function create a linear Model that predict nextReturn"

  x_train = train.drop('nextReturn',1)

  y_train = train['nextReturn']

  x_valid = valid.drop('nextReturn',1)

  y_valid = valid['nextReturn']

  model = LinearRegression()

  model.fit(x_train,y_train)

  preds = model.predict(x_valid)

  rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))

  return model , rms
def simplelinear_next_day(train,valid): 

  "this function create a linear Model that predict next_day_ret"

  x_train = train.drop('next_day_ret',1)

  y_train = train['next_day_ret']

  x_valid = valid.drop('next_day_ret',1)

  y_valid = valid['next_day_ret']

  model = LinearRegression()

  model.fit(x_train,y_train)

  preds = model.predict(x_valid)

  rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))

  return model , rms
feature_selected = [ 'nextReturn','currReturn','currReturn_square','TEMA','TEMA_square'] # features selected to predict the nextReturn 

tech = ['MACD', 'BB_Middle_Band', 'BB_Upper_Band','EMA','Momentum','Aroon_current'] 

to_reduce = {'tech':tech}
df_train,df_test,col=pca_function(train,test,to_reduce,deb=False) # dimensionality reduction phase 
f = feature_selected+col
f = feature_selected+col

train1 , valid = train_test_split(df_train[f],test_size=0.2,random_state=70)

model,rms= simplelinear(train1,valid)

print(f'RMSE = {rms}')
cofficents = dict()

for i in range(1,len(f)) : 

  cofficents[f[i]] = model.coef_[i-1]

cofficents
folds = 5

df_train =df_train.reset_index(drop=True)

kf = model_selection.KFold(n_splits=folds)

vaild_rmse = np.zeros(folds)

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(df_train)),total=folds):

    train1 = df_train.iloc[train_idx].copy()

    valid = df_train.iloc[val_idx].copy()

    model,rms= simplelinear(train1[f],valid[f])

    vaild_rmse[fold] = rms

    preds = model.predict(df_test[f[1:]])

    df_test.loc[:,f'fold_{fold}'] = preds

    preds = model.predict(df_train[f[1:]])

    df_train.loc[:,f'fold_{fold}'] = preds

print(f'RMSE_MEAN = {vaild_rmse.mean()} std = {vaild_rmse.std()}')
fold = [f'fold_{fold}' for fold in range(folds)]

df_test['NextReturn_pred'] = df_test[fold].mean(axis=1)

df_train['NextReturn_pred'] = df_train[fold].mean(axis=1)
to_reduce = {'last_mean':last_mean}

df_train,df_test,col=pca_function(df_train,df_test,to_reduce,deb=False) # dimensionality reduction phase 
featues_selected_2 = ['next_day_ret','NextReturn_pred'] + col 

train1 , valid = train_test_split(df_train[featues_selected_2],test_size=0.2,random_state=70)

model,rms= simplelinear_next_day(train1,valid)

print(f'RMSE = {rms}')
cofficents = dict()

for i in range(1,len(featues_selected_2)) : 

  cofficents[featues_selected_2[i]] = model.coef_[i-1]

cofficents
folds = 5

df_train =df_train.reset_index(drop=True)

kf = model_selection.KFold(n_splits=folds)

vaild_rmse = np.zeros(folds)

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(df_train)),total=folds):

    train1 = df_train.iloc[train_idx].copy()

    valid = df_train.iloc[val_idx].copy()

    model,rms= simplelinear_next_day(train1[featues_selected_2],valid[featues_selected_2])

    vaild_rmse[fold] = rms

    preds = model.predict(df_test[featues_selected_2[1:]])

    df_test.loc[:,f'fold_{fold}'] = preds

    preds = model.predict(df_train[featues_selected_2[1:]])

    df_train.loc[:,f'fold_{fold}'] = preds

 

print(f'RMSE_MEAN = {vaild_rmse.mean()} std = {vaild_rmse.std()}')
fold = [f'fold_{fold}' for fold in range(folds)]

df_test['next_day_ret'] = df_test[fold].mean(axis=1)

df_train['next_day_ret_pred'] = df_train[fold].mean(axis=1)
sample = pd.read_csv('../input/data-analysis-2020/sample_submission.csv')

test_copy = df_test.set_index('Id')

sample =sample.set_index('Id')

sample['next_day_ret'] =test_copy['next_day_ret'] # Make a prediction with all companies model 
company = df_train[df_train['company']=='ADWYA']

Y = df_train['next_day_ret'].iloc[150:250]

y = df_train['next_day_ret_pred'].iloc[150:250]

plt.figure(figsize=(10,10))

plt.plot(Y, label = 'real')

plt.plot(y, label = 'predicted')

plt.legend()

plt.title('next_day_ret')

plt.xlabel('date')

plt.ylabel('next_day_ret')

plt.show()
sector = 'SERVICES_FINANCIERS+ASSURANCES+BANQUES'

train_sector = train[train['sector']==sector].copy()

test_sector = test[test['sector']==sector].copy()
feature_selected = [ 'nextReturn','TEMA','Aroon_current']

tech = ['MACD', 'BB_Middle_Band', 'BB_Upper_Band','EMA','Momentum']+['currReturn','currReturn_square']

to_reduce = {'tech':tech }
train_sector,test_sector ,col = pca_function(train_sector,test_sector,to_reduce,deb=False)

len_sector = len(train_sector)



data = pd.concat([train_sector,test_sector],ignore_index=True)

company = pd.get_dummies(data['company'].astype('str'))

data = pd.concat([data,company],axis=1)



train_sector = data.iloc[:len_sector].copy()

test_sector = data.iloc[len_sector:].copy()
corrmat = train_sector[sector_features+['nextReturn']].corr()

selected = [sector+'_pc_'+str(i) for i in range(5)] # sector factors

f = feature_selected + selected + col
folds = 5

train_sector =train_sector.reset_index(drop=True)

kf = model_selection.KFold(n_splits=folds)

vaild_rmse = np.zeros(folds)

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(train_sector)),total=folds):

    train1 = train_sector.iloc[train_idx].copy()

    valid = train_sector.iloc[val_idx].copy()

    model,rms= simplelinear(train1[f],valid[f])

    vaild_rmse[fold] = rms

    preds = model.predict(test_sector[f[1:]])

    test_sector.loc[:,f'fold_{fold}'] = preds

    preds = model.predict(train_sector[f[1:]])

    train_sector.loc[:,f'fold_{fold}'] = preds

print(f'RMSE_MEAN = {vaild_rmse.mean()} std = {vaild_rmse.std()}')
fold = [f'fold_{fold}' for fold in range(folds)]

test_sector['NextReturn_pred'] = test_sector[fold].mean(axis=1)

train_sector['NextReturn_pred'] = train_sector[fold].mean(axis=1)
featues_selected_2 = ['next_day_ret','NextReturn_pred','last']+last_mean
folds = 5

train_sector =train_sector.reset_index(drop=True)

kf = model_selection.KFold(n_splits=folds)

vaild_rmse = np.zeros(folds)

for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(train_sector)),total=folds):

    

    train1 = train_sector.iloc[train_idx].copy()

    valid = train_sector.iloc[val_idx].copy()

    model,rms= simplelinear_next_day(train1[featues_selected_2],valid[featues_selected_2])

    vaild_rmse[fold] = rms

    preds = model.predict(test_sector[featues_selected_2[1:]])

    test_sector.loc[:,f'fold_{fold}'] = preds

 

print(f'RMSE_MEAN = {vaild_rmse.mean()} std = {vaild_rmse.std()}')
class LinearModelsectors() : 

  

  def __init__(self,train,test, feature_selected ,  to_reduce) : 

  

    self.train = train

    self.test = test 

    self.feature_selected = feature_selected 

    self.to_reduce = to_reduce

    self.subission_file='not_yet_genrated'

  

  def prep_data(self,train_sector,test_sector,to_reduce,sector) : 

    

    train_sector,test_sector ,col = pca_function(train_sector,test_sector,to_reduce,deb=False)

    len_sector = len(train_sector)



    data = pd.concat([train_sector,test_sector],ignore_index=True)

    company = pd.get_dummies(data['company'].astype('str'))

    data = pd.concat([data,company],axis=1)



    train_sector = data.iloc[:len_sector].copy()

    test_sector = data.iloc[len_sector:].copy()

    corrmat = train_sector[sector_features+['nextReturn']].corr()

    selected = [sector+'_pc_'+str(i) for i in range(5)]



    selected = selected+col

    

    return train_sector,test_sector,selected



  def predict_one_sector(self,train_sector,test_sector,folds,sector) : 

   

    train_sector , test_sector,selected = self.prep_data(train_sector,test_sector,self.to_reduce,sector)

    train_sector =train_sector.reset_index(drop=True)

    kf = model_selection.KFold(n_splits=folds)

    vaild_rmse = np.zeros(folds)



    f = self.feature_selected + selected



    for fold, (train_idx, val_idx) in enumerate(kf.split(train_sector)):

      train1 = train_sector.iloc[train_idx].copy()

      valid = train_sector.iloc[val_idx].copy()

      model,rms= simplelinear(train1[f],valid[f])

      vaild_rmse[fold] = rms

      preds = model.predict(test_sector[f[1:]])

      test_sector.loc[:,f'fold_{fold}'] = preds

      preds = model.predict(train_sector[f[1:]])

      train_sector.loc[:,f'fold_{fold}'] = preds



    fold = [f'fold_{fold}' for fold in range(folds)]

    test_sector['NextReturn_pred'] = test_sector[fold].mean(axis=1)

    train_sector['NextReturn_pred'] = train_sector[fold].mean(axis=1)

    featues_selected_2 = ['next_day_ret','NextReturn_pred','last']+last_mean



    train_sector =train_sector.reset_index(drop=True)

    kf = model_selection.KFold(n_splits=folds)

    vaild_rmse = np.zeros(folds)



    for fold, (train_idx, val_idx) in enumerate(kf.split(train_sector)):

        

      train1 = train_sector.iloc[train_idx].copy()

      valid = train_sector.iloc[val_idx].copy()

      model,rms= simplelinear_next_day(train1[featues_selected_2],valid[featues_selected_2])

      vaild_rmse[fold] = rms

      preds = model.predict(test_sector[featues_selected_2[1:]])

      test_sector.loc[:,f'fold_{fold}'] = preds



    fold = [f'fold_{fold}' for fold in range(folds)]

    df_test['next_day_ret'] = df_test[fold].mean(axis=1)



    return df_test , vaild_rmse.mean() , vaild_rmse.std()

  def kflod_predict(self,folds=5) : 

  

    self.sector_rmse = dict()

    self.sector_std = dict()

    

    sector_rmse = np.zeros(11)

    index = 0  

    test_copy = self.test.copy().set_index('Id')



    for sector in tqdm(all_company.keys(),total=len(all_company.keys())) :



      train_sector = self.train[self.train['sector']==sector].copy()

      test_sector  = self.test[self.test['sector']==sector].copy()

      test_sector , mean_sector , std_sector = self.predict_one_sector(train_sector,test_sector,folds,sector)

      test_sector = test_sector.set_index('Id')

      test_copy.loc[test_sector.index,'next_day_ret'] = test_sector['next_day_ret']

      sector_rmse[index] = mean_sector

      self.sector_rmse[sector] =mean_sector

      self.sector_std[sector] = std_sector

      index +=1 

    

      sample = pd.read_csv('../input/data-analysis-2020/sample_submission.csv')

      sample.loc[:,'next_day_ret'] =test_copy['next_day_ret'].values

      self.submission_file = sample

    
feature_selected = [ 'nextReturn','TEMA','Aroon_current']

tech = ['MACD', 'BB_Middle_Band', 'BB_Upper_Band','EMA','Momentum']+['currReturn','currReturn_square']

to_reduce = {'tech':tech }
sector_model = LinearModelsectors(train,test,feature_selected,to_reduce)

sector_model.kflod_predict(5)
sector_model.sector_rmse
sector_model.sector_std
sector_sub= sector_model.submission_file.set_index('Id')
sample.loc[sample[sample['next_day_ret']<-1000].index,'next_day_ret'] =-600 # remove outliers from prediction 

sector_sub.loc[sector_sub[sector_sub['next_day_ret']<-1000].index,'next_day_ret'] =-600# remove outliers from prediction 
sample['next_day_ret'] = sector_sub['next_day_ret']*0.3 + sample['next_day_ret']*0.7
sample.to_csv('submission.csv') 
sample['next_day_ret'].hist()