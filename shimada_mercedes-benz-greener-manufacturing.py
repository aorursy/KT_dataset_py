import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFromModel

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVR, SVR

from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.linear_model import SGDRegressor, stochastic_gradient, LogisticRegression

from sklearn.neural_network import BernoulliRBM



%matplotlib inline
df_train = pd.read_csv('../input/train.csv',index_col=0)
from sklearn.preprocessing import OneHotEncoder, add_dummy_feature, LabelEncoder 
tmp=df_train.iloc[:,:9]
tmp.groupby('X0').describe()
tmp.groupby('X0').describe().y['mean'].plot(kind='bar')

tmp.groupby('X0').describe().y['count'].plot(kind='bar')
df_train
train_list = [

'X10',

 'X11',

 'X12',

 'X13',

 'X14',

 'X15',

 'X16',

 'X17',

 'X18',

 'X19',

 'X20',

 'X21',

 'X22',

 'X23',

 'X24',

 'X26',

 'X27',

 'X28',

 'X29',

 'X30',

 'X31',

 'X32',

 'X33',

 'X34',

 'X35',

 'X36',

 'X37',

 'X38',

 'X39',

 'X40',

 'X41',

 'X42',

 'X43',

 'X44',

 'X45',

 'X46',

 'X47',

 'X48',

 'X49',

 'X50',

 'X51',

 'X52',

 'X53',

 'X54',

 'X55',

 'X56',

 'X57',

 'X58',

 'X59',

 'X60',

 'X61',

 'X62',

 'X63',

 'X64',

 'X65',

 'X66',

 'X67',

 'X68',

 'X69',

 'X70',

 'X71',

 'X73',

 'X74',

 'X75',

 'X76',

 'X77',

 'X78',

 'X79',

 'X80',

 'X81',

 'X82',

 'X83',

 'X84',

 'X85',

 'X86',

 'X87',

 'X88',

 'X89',

 'X90',

 'X91',

 'X92',

 'X93',

 'X94',

 'X95',

 'X96',

 'X97',

 'X98',

 'X99',

 'X100',

 'X101',

 'X102',

 'X103',

 'X104',

 'X105',

 'X106',

 'X107',

 'X108',

 'X109',

 'X110',

 'X111',

 'X112',

 'X113',

 'X114',

 'X115',

 'X116',

 'X117',

 'X118',

 'X119',

 'X120',

 'X122',

 'X123',

 'X124',

 'X125',

 'X126',

 'X127',

 'X128',

 'X129',

 'X130',

 'X131',

 'X132',

 'X133',

 'X134',

 'X135',

 'X136',

 'X137',

 'X138',

 'X139',

 'X140',

 'X141',

 'X142',

 'X143',

 'X144',

 'X145',

 'X146',

 'X147',

 'X148',

 'X150',

 'X151',

 'X152',

 'X153',

 'X154',

 'X155',

 'X156',

 'X157',

 'X158',

 'X159',

 'X160',

 'X161',

 'X162',

 'X163',

 'X164',

 'X165',

 'X166',

 'X167',

 'X168',

 'X169',

 'X170',

 'X171',

 'X172',

 'X173',

 'X174',

 'X175',

 'X176',

 'X177',

 'X178',

 'X179',

 'X180',

 'X181',

 'X182',

 'X183',

 'X184',

 'X185',

 'X186',

 'X187',

 'X189',

 'X190',

 'X191',

 'X192',

 'X194',

 'X195',

 'X196',

 'X197',

 'X198',

 'X199',

 'X200',

 'X201',

 'X202',

 'X203',

 'X204',

 'X205',

 'X206',

 'X207',

 'X208',

 'X209',

 'X210',

 'X211',

 'X212',

 'X213',

 'X214',

 'X215',

 'X216',

 'X217',

 'X218',

 'X219',

 'X220',

 'X221',

 'X222',

 'X223',

 'X224',

 'X225',

 'X226',

 'X227',

 'X228',

 'X229',

 'X230',

 'X231',

 'X232',

 'X233',

 'X234',

 'X235',

 'X236',

 'X237',

 'X238',

 'X239',

 'X240',

 'X241',

 'X242',

 'X243',

 'X244',

 'X245',

 'X246',

 'X247',

 'X248',

 'X249',

 'X250',

 'X251',

 'X252',

 'X253',

 'X254',

 'X255',

 'X256',

 'X257',

 'X258',

 'X259',

 'X260',

 'X261',

 'X262',

 'X263',

 'X264',

 'X265',

 'X266',

 'X267',

 'X268',

 'X269',

 'X270',

 'X271',

 'X272',

 'X273',

 'X274',

 'X275',

 'X276',

 'X277',

 'X278',

 'X279',

 'X280',

 'X281',

 'X282',

 'X283',

 'X284',

 'X285',

 'X286',

 'X287',

 'X288',

 'X289',

 'X290',

 'X291',

 'X292',

 'X293',

 'X294',

 'X295',

 'X296',

 'X297',

 'X298',

 'X299',

 'X300',

 'X301',

 'X302',

 'X304',

 'X305',

 'X306',

 'X307',

 'X308',

 'X309',

 'X310',

 'X311',

 'X312',

 'X313',

 'X314',

 'X315',

 'X316',

 'X317',

 'X318',

 'X319',

 'X320',

 'X321',

 'X322',

 'X323',

 'X324',

 'X325',

 'X326',

 'X327',

 'X328',

 'X329',

 'X330',

 'X331',

 'X332',

 'X333',

 'X334',

 'X335',

 'X336',

 'X337',

 'X338',

 'X339',

 'X340',

 'X341',

 'X342',

 'X343',

 'X344',

 'X345',

 'X346',

 'X347',

 'X348',

 'X349',

 'X350',

 'X351',

 'X352',

 'X353',

 'X354',

 'X355',

 'X356',

 'X357',

 'X358',

 'X359',

 'X360',

 'X361',

 'X362',

 'X363',

 'X364',

 'X365',

 'X366',

 'X367',

 'X368',

 'X369',

 'X370',

 'X371',

 'X372',

 'X373',

 'X374',

 'X375',

 'X376',

 'X377',

 'X378',

 'X379',

 'X380',

 'X382',

 'X383',

 'X384',

 'X385']
df_train = pd.read_csv('../input/train.csv',index_col=0)

df_train = df_train[['y']+train_list]

np_train = df_train.as_matrix()

np_train.astype(np.float32)

X = np_train[:,1:]

y = np_train[:,0]

df_test = pd.read_csv('../input/test.csv',index_col=0)

df_test = df_test[train_list]

np_test = df_test.as_matrix()

np_test = np_test.astype(np.float32)

X_val = np_test[:,:]
std = StandardScaler()

std.fit(X)

X = std.transform(X)

X_val = std.transform(X_val)

#lsvc = LinearSVR(C=0.01).fit(X, y)

#tmp = LinearSVR().fit(X,y)

#slt = SelectFromModel(tmp, prefit=True)
slt = SelectKBest(k=50)

#slt = SelectPercentile(percentile=30)

slt.fit(X,y)

X = slt.transform(X)

X_val = slt.transform(X_val)
rbm = BernoulliRBM(n_components=20, n_iter=30,verbose=True, random_state=42)

rbm.fit(X,y)
X = rbm.transform(X)

X_val = rbm.transform(X_val)
#X = slt.fit_transform(X,y)

#X_val = slt.transform(X_val)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1, random_state=42)
#model = RandomForestRegressor(n_jobs=-1, n_estimators=100)

model = GradientBoostingRegressor(n_estimators=100, max_depth=10)



model.fit(X_train,y_train)



pred = model.predict(X_test)

r2_score(y_test,pred)
y_val = model.predict(X_val)
X_syn = np.r_[X,X_val]

y_syn = np.r_[y,y_val]
model = GradientBoostingRegressor(n_estimators=100, max_depth=10)

model.fit(X_syn,y_syn)



pred = model.predict(X_syn)

r2_score(y_syn,pred)



y_val = model.predict(X_val)
df_val = pd.DataFrame(y_val,index=df_test.index,columns=['y'])
df_val.to_csv('submit.csv')