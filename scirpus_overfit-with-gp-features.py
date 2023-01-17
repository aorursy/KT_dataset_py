import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import r2_score

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
def LeaveOneOutByX261(data1, data2, columnName, useLOO=False):

    grpOutcomes = data1.groupby(list(['X261'])+list([columnName]))['y'].mean().reset_index()

    grpCount = data1.groupby(list(['X261'])+list([columnName]))['y'].count().reset_index()

    grpOutcomes['cnt'] = grpCount.y

    if(useLOO):

        grpOutcomes = grpOutcomes[grpOutcomes.cnt > 4]

    grpOutcomes.drop('cnt', inplace=True, axis=1)

    outcomes = data2['y'].values

    x = pd.merge(data2[[columnName,'X261', 'y']], grpOutcomes,

                 suffixes=('x_', ''),

                 how='left',

                 on=list(['X261'])+list([columnName]),

                 left_index=True)['y']

    if(useLOO):

        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)

    

    return x.values
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.y = np.log(train.y)

test.insert(1,'y',np.nan)

print(train.shape)

print(test.shape)

ytrain = train.y.ravel()

for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(train[c].values) + list(test[c].values))

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))
remove = []

c = train.columns

for i in range(len(c)):

    v = train[c[i]].values

    for j in range(i+1, len(c)):

        if np.array_equal(v, train[c[j]].values):

            remove.append(c[j])



train.drop(remove, axis=1, inplace=True)

test.drop(remove, axis=1, inplace=True)



remove = []

for col in train.columns:

    if train[col].std() == 0.0:

        remove.append(col)



train.drop(remove, axis=1, inplace=True)

test.drop(remove, axis=1, inplace=True)
feats = list(set(train.columns[2:]).difference(set(['X261'])))

trainids = train.ID.ravel()

testids = test.ID.ravel()

for c in feats:

    test['loo'+c] = LeaveOneOutByX261(train,

                                      test,c,False)

    test.loc[(test['loo'+c].isnull())&(test.X261==0),'loo'+c] = test.loc[test.X261==0,'loo'+c].mean()

    test.loc[(test['loo'+c].isnull())&(test.X261==1),'loo'+c] = test.loc[test.X261==1,'loo'+c].mean()

    train['loo'+c] = LeaveOneOutByX261(train,

                                       train,c,True)

    train.loc[(train['loo'+c].isnull())&(train.X261==0),'loo'+c] = train.loc[test.X261==0,'loo'+c].mean()

    train.loc[(train['loo'+c].isnull())&(train.X261==1),'loo'+c] = train.loc[test.X261==1,'loo'+c].mean()

   

train.drop(['ID','y'],inplace=True,axis=1)

test.drop(['ID','y'],inplace=True,axis=1)
train.drop(feats,inplace=True,axis=1)

test.drop(feats,inplace=True,axis=1)

train.drop('X261',inplace=True,axis=1)

test.drop('X261',inplace=True,axis=1)

train = train[train.columns]

test = test[train.columns]
ss = StandardScaler()

ss.fit(pd.concat([train,test]))

train[train.columns] = ss.transform(train[train.columns])

test[test.columns] = ss.transform(test[test.columns])

train['y'] = ytrain

def GP(data):

    return (4.596634 +

            0.020000*np.tanh((11.0 * np.maximum( (data["looX47"]),  (np.maximum( (data["looX348"]),  (data["looX315"])))))) +

            0.020000*np.tanh((11.0 * np.maximum( (np.maximum( (data["looX314"]),  (data["looX315"]))),  (data["looX47"])))) +

            0.020000*np.tanh((((data["looX0"] * 2.0) - np.tanh(np.tanh(data["looX143"]))) * 2.0)) +

            0.020000*np.tanh(((data["looX0"] + np.tanh(((data["looX38"] < data["looX0"]).astype(float)))) * 2.0)) +

            0.020000*np.tanh((data["looX0"] + np.maximum( (data["looX0"]),  (((data["looX153"] < data["looX95"]).astype(float)))))) +

            0.020000*np.tanh((data["looX5"] + (((data["looX0"] - data["looX163"]) * 2.0) * 2.0))) +

            0.020000*np.tanh((data["looX0"] - (data["looX308"] - (data["looX5"] / 2.0)))) +

            0.020000*np.tanh((data["looX118"] * (data["looX136"] * (data["looX238"] - data["looX273"])))) +

            0.020000*np.tanh((0.458824 - ((data["looX54"] > np.tanh(np.tanh(data["looX292"]))).astype(float)))) +

            0.020000*np.tanh(((data["looX263"] - np.tanh(((data["looX311"] + data["looX3"])/2.0))) * 2.0)) +

            0.020000*np.tanh((data["looX136"] * (np.maximum( (data["looX5"]),  (data["looX54"])) - data["looX263"]))) +

            0.020000*np.tanh(((data["looX263"] - np.maximum( (np.tanh(data["looX143"])),  (data["looX162"]))) * 2.0)) +

            0.020000*np.tanh((11.0 * ((data["looX45"] > (data["looX237"] * data["looX54"])).astype(float)))) +

            0.020000*np.tanh(((((data["looX54"] < np.tanh(np.tanh(data["looX16"]))).astype(float)) * 2.0) * 2.0)) +

            0.020000*np.tanh(np.minimum( ((data["looX29"] - data["looX161"])),  ((data["looX5"] - data["looX0"])))) +

            0.020000*np.tanh(((data["looX251"] > (data["looX91"] * ((data["looX62"] + data["looX255"])/2.0))).astype(float))) +

            0.020000*np.tanh((np.minimum( (data["looX8"]),  (np.minimum( (data["looX1"]),  (data["looX29"])))) - data["looX220"])) +

            0.020000*np.tanh((((data["looX358"] < np.tanh(data["looX240"])).astype(float)) + (data["looX29"] / 2.0))) +

            0.016736*np.tanh((-(((data["looX358"] > np.tanh(data["looX240"])).astype(float))))) +

            0.020000*np.tanh(((((data["looX54"] < np.tanh(np.tanh(data["looX353"]))).astype(float)) * 2.0) * 2.0)) +

            0.000300*np.tanh((np.minimum( (data["looX246"]),  (data["looX263"])) - np.tanh(data["looX0"]))) +

            0.020000*np.tanh((np.minimum( (data["looX276"]),  (data["looX246"])) - np.tanh(data["looX0"]))) +

            0.020000*np.tanh(((((data["looX175"] > (data["looX306"] * data["looX306"])).astype(float)) * 2.0) * 2.0)) +

            0.009148*np.tanh(np.tanh((np.minimum( (data["looX358"]),  (np.tanh(data["looX58"]))) - data["looX314"]))) +

            0.020000*np.tanh(((np.maximum( (data["looX68"]),  ((data["looX315"] * 2.0))) < data["looX315"]).astype(float))) +

            0.020000*np.tanh(((9.0) * ((data["looX145"] > (data["looX315"] * data["looX68"])).astype(float)))) +

            0.004044*np.tanh((np.minimum( (data["looX358"]),  (np.tanh(data["looX137"]))) - data["looX315"])) +

            0.020000*np.tanh((((data["looX194"] / 2.0) > (data["looX319"] * (data["looX183"] / 2.0))).astype(float))) +

            0.020000*np.tanh((11.0 * ((data["looX137"] > (data["looX10"] * data["looX350"])).astype(float)))) +

            0.020000*np.tanh((-(np.tanh(((1.375000 < (data["looX128"] * 1.288460)).astype(float)))))) +

            0.020000*np.tanh(((((data["looX47"] > (data["looX85"] * data["looX292"])).astype(float)) * 2.0) * 2.0)) +

            0.020000*np.tanh((data["looX358"] - (data["looX315"] - ((data["looX68"] < data["looX315"]).astype(float))))) +

            0.019998*np.tanh((np.maximum( (data["looX196"]),  (data["looX383"])) - np.maximum( (data["looX362"]),  (data["looX115"])))) +

            0.020000*np.tanh(((((np.tanh(1.288460) < (-(data["looX66"]))).astype(float)) * 2.0) * 2.0)) +

            0.020000*np.tanh(((data["looX285"] < np.tanh((np.tanh(data["looX195"]) * 2.0))).astype(float))) +

            0.020000*np.tanh((data["looX198"] - np.maximum( (data["looX0"]),  (data["looX304"])))) +

            0.020000*np.tanh((((((data["looX280"] > data["looX357"]).astype(float)) * 2.0) * 2.0) * 2.0)) +

            0.016542*np.tanh((data["looX125"] - np.maximum( (data["looX267"]),  (((data["looX0"] + data["looX31"])/2.0))))) +

            0.020000*np.tanh((((((data["looX295"] < data["looX257"]).astype(float)) * 2.0) * 2.0) * 2.0)) +

            0.020000*np.tanh(((((data["looX362"] > (data["looX127"] * data["looX383"])).astype(float)) * 2.0) * 2.0)) +

            0.020000*np.tanh(((((data["looX40"] > (data["looX264"] * data["looX75"])).astype(float)) * 2.0) * 2.0)) +

            0.020000*np.tanh(((-(np.tanh(((data["looX301"] > np.tanh(data["looX171"])).astype(float))))) / 2.0)) +

            0.020000*np.tanh((((1.740000 < ((-(data["looX257"])) - data["looX267"])).astype(float)) * 2.0)) +

            0.019998*np.tanh(((((np.tanh(data["looX118"]) + data["looX238"])/2.0) - data["looX0"]) / 2.0)) +

            0.020000*np.tanh((data["looX315"] * ((data["looX135"] > (data["looX47"] * data["looX64"])).astype(float)))) +

            0.020000*np.tanh(((((data["looX54"] < np.tanh(np.tanh(data["looX338"]))).astype(float)) * 2.0) * 2.0)) +

            0.020000*np.tanh((np.minimum( (np.minimum( (data["looX178"]),  (data["looX162"]))),  (data["looX265"])) - data["looX315"])) +

            0.020000*np.tanh(((7.0) * (((data["looX52"] * data["looX52"]) < data["looX192"]).astype(float)))) +

            0.020000*np.tanh((((data["looX68"] < np.minimum( (data["looX315"]),  ((-(data["looX315"]))))).astype(float)) * 2.0)) +

            0.020000*np.tanh((11.0 * ((data["looX148"] > (data["looX194"] * data["looX183"])).astype(float)))) +

            0.020000*np.tanh((11.0 * ((1.740000 < (-((data["looX349"] * 2.0)))).astype(float)))) +

            0.020000*np.tanh((np.minimum( (np.minimum( (data["looX220"]),  (data["looX58"]))),  (data["looX223"])) - data["looX211"])) +

            0.020000*np.tanh(((14.91481018066406250) * ((data["looX145"] > (data["looX59"] * data["looX70"])).astype(float)))) +

            0.020000*np.tanh((((1.740000 < ((-(data["looX206"])) - data["looX203"])).astype(float)) * 2.0)) +

            0.020000*np.tanh((((((data["looX295"] < data["looX257"]).astype(float)) * 2.0) * 2.0) * 2.0)) +

            0.020000*np.tanh((data["looX38"] - np.maximum( (((data["looX0"] + data["looX5"])/2.0)),  (data["looX56"])))) +

            0.020000*np.tanh(((((data["looX190"] < ((data["looX301"] + data["looX61"])/2.0)).astype(float)) * 2.0) * 2.0)) +

            0.005224*np.tanh((np.minimum( (data["looX125"]),  (((data["looX286"] < data["looX292"]).astype(float)))) - data["looX61"])) +

            0.020000*np.tanh((np.maximum( (data["looX236"]),  (data["looX105"])) - np.maximum( (data["looX61"]),  (data["looX206"])))) +

            0.020000*np.tanh((((((data["looX280"] > data["looX357"]).astype(float)) * 2.0) * 2.0) * 2.0)) +

            0.020000*np.tanh(((((data["looX194"] > (data["looX65"] * data["looX236"])).astype(float)) * 2.0) * 2.0)) +

            0.020000*np.tanh((((((data["looX357"] < data["looX280"]).astype(float)) * 2.0) * 2.0) * 2.0)) +

            0.020000*np.tanh(np.tanh(((1.740000 < ((-(data["looX240"])) * 2.0)).astype(float)))) +

            0.020000*np.tanh((data["looX306"] - np.maximum( (np.maximum( (data["looX306"]),  (data["looX315"]))),  (data["looX0"])))) +

            0.020000*np.tanh((data["looX1"] * (data["looX47"] - ((data["looX273"] + data["looX125"])/2.0)))) +

            0.020000*np.tanh(((((data["looX354"] < np.tanh((data["looX90"] * 2.0))).astype(float)) * 2.0) * 2.0)) +

            0.003702*np.tanh(((data["looX354"] - np.maximum( (data["looX314"]),  (np.tanh(data["looX142"])))) / 2.0)) +

            0.020000*np.tanh((data["looX292"] - np.maximum( (data["looX314"]),  (((data["looX5"] + data["looX142"])/2.0))))) +

            0.020000*np.tanh(((((data["looX362"] > (data["looX270"] * data["looX105"])).astype(float)) * 2.0) * 2.0)) +

            0.020000*np.tanh(((((data["looX354"] < np.tanh((data["looX105"] * 2.0))).astype(float)) * 2.0) * 2.0)) +

            0.016372*np.tanh((data["looX2"] * ((data["looX267"] > (data["looX246"] * data["looX163"])).astype(float)))) +

            0.019996*np.tanh((data["looX85"] - ((np.maximum( (data["looX115"]),  (data["looX362"])) + data["looX117"])/2.0))) +

            0.020000*np.tanh(((((np.tanh((data["looX357"] * 2.0)) > data["looX354"]).astype(float)) * 2.0) * 2.0)) +

            0.020000*np.tanh((11.0 * ((data["looX192"] > (data["looX312"] * data["looX358"])).astype(float)))) +

            0.020000*np.tanh(((((data["looX26"] > (data["looX154"] * data["looX276"])).astype(float)) * 2.0) * 2.0)) +

            0.020000*np.tanh((((1.740000 < ((-(data["looX206"])) - data["looX16"])).astype(float)) * 2.0)) +

            0.020000*np.tanh(((((data["looX58"] > (data["looX343"] * data["looX92"])).astype(float)) * 2.0) * 2.0)) +

            0.019998*np.tanh((data["looX163"] - ((data["looX5"] + np.maximum( (data["looX191"]),  (data["looX237"])))/2.0))) +

            0.020000*np.tanh(((((1.740000 < (-((data["looX135"] * 2.0)))).astype(float)) * 2.0) * 2.0)) +

            0.020000*np.tanh(((data["looX230"] > np.maximum( ((data["looX179"] * data["looX49"])),  (data["looX166"]))).astype(float))) +

            0.020000*np.tanh(((data["looX128"] < ((data["looX366"] + np.tanh((data["looX203"] * 2.0)))/2.0)).astype(float))) +

            0.020000*np.tanh((((data["looX362"] > (data["looX153"] * data["looX200"])).astype(float)) * (6.0))) +

            0.019974*np.tanh(((np.minimum( (data["looX301"]),  (data["looX170"])) - data["looX135"]) * 2.0)) +

            0.020000*np.tanh((data["looX304"] * (np.maximum( (data["looX383"]),  (data["looX55"])) - data["looX38"]))) +

            0.020000*np.tanh(((9.94399166107177734) * ((data["looX269"] > (data["looX98"] * data["looX98"])).astype(float)))) +

            0.020000*np.tanh((((data["looX339"] - np.minimum( (data["looX40"]),  (data["looX59"]))) * 2.0) * 2.0)) +

            0.017742*np.tanh(((np.tanh(((data["looX0"] < np.tanh(data["looX258"])).astype(float))) / 2.0) / 2.0)) +

            0.020000*np.tanh(((((data["looX104"] > (data["looX126"] * data["looX48"])).astype(float)) / 2.0) / 2.0)) +

            0.004384*np.tanh((data["looX118"] * ((np.tanh(data["looX315"]) - data["looX5"]) / 2.0))) +

            0.019850*np.tanh((data["looX68"] * ((np.tanh(data["looX315"]) + (-(data["looX57"])))/2.0))) +

            0.020000*np.tanh((((((-(data["looX173"])) - data["looX282"]) > 1.740000).astype(float)) * 2.0)) +

            0.020000*np.tanh((((data["looX332"] - np.maximum( (data["looX62"]),  (data["looX280"]))) * 2.0) * 2.0)) +

            0.019992*np.tanh(((data["looX354"] + (data["looX70"] - (data["looX152"] + data["looX322"])))/2.0)) +

            0.020000*np.tanh((((data["looX28"] > data["looX306"]).astype(float)) * ((data["looX339"] > data["looX28"]).astype(float)))) +

            0.020000*np.tanh((((((data["looX236"] > data["looX40"]).astype(float)) * 2.0) * 2.0) * 2.0)) +

            0.019988*np.tanh((((data["looX105"] - ((data["looX271"] + data["looX280"])/2.0)) * 2.0) * 2.0)) +

            0.020000*np.tanh(np.minimum( ((data["looX292"] - data["looX383"])),  ((data["looX145"] - data["looX153"])))) +

            0.020000*np.tanh((data["looX70"] - ((data["looX145"] + ((data["looX191"] + data["looX58"])/2.0))/2.0))) +

            0.019102*np.tanh((data["looX63"] - np.minimum( (np.minimum( (data["looX308"]),  (data["looX59"]))),  (data["looX287"])))) +

            0.019986*np.tanh(((data["looX301"] + (data["looX339"] - (data["looX280"] * 2.0))) * 2.0)))
plt.scatter(np.exp(GP(train)),np.exp(train.y))
print(r2_score(np.exp(train.y),np.exp(GP(train))))
def Munger(data):

    munge = pd.DataFrame()

    munge["i0"] = np.tanh((((data["looX0"] * 2.0) - np.tanh(np.tanh(data["looX143"]))) * 2.0))

    munge["i1"] = np.tanh((data["looX118"] * (data["looX136"] * (data["looX238"] - data["looX273"]))))

    munge["i2"] = np.tanh((data["looX136"] * (np.maximum( (data["looX5"]),  (data["looX54"])) - data["looX263"])))

    munge["i3"] = np.tanh(((data["looX263"] - np.maximum( (np.tanh(data["looX143"])),  (data["looX162"]))) * 2.0))

    munge["i4"] = np.tanh((((data["looX358"] < np.tanh(data["looX240"])).astype(float)) + (data["looX29"] / 2.0)))

    munge["i5"] = np.tanh((np.minimum( (data["looX125"]),  (((data["looX286"] < data["looX292"]).astype(float)))) - data["looX61"]))

    munge["i6"] = np.tanh((np.maximum( (data["looX236"]),  (data["looX105"])) - np.maximum( (data["looX61"]),  (data["looX206"]))))

    munge["i7"] = np.tanh((((data["looX339"] - np.minimum( (data["looX40"]),  (data["looX59"]))) * 2.0) * 2.0))

    munge["i8"] = np.tanh((((data["looX332"] - np.maximum( (data["looX62"]),  (data["looX280"]))) * 2.0) * 2.0))

    munge["i9"] = np.tanh((data["looX63"] - np.minimum( (np.minimum( (data["looX308"]),  (data["looX59"]))),  (data["looX287"]))))

    return munge
xtrain = Munger(train.copy())

y = train.y.ravel()
import xgboost as xgb

xgb_params = {}

xgb_params['objective'] = 'reg:linear'

xgb_params['eta'] = 0.05

xgb_params['max_depth'] = 4

xgb_params['basescore'] = np.median(y)
import operator

def create_feature_map(features):

    outfile = open('xgb.fmap', 'w')

    i = 0

    for feat in features:

        outfile.write('{0}\t{1}\tq\n'.format(i, feat))

        i = i + 1



    outfile.close()

create_feature_map(xtrain.columns)
dtrain = xgb.DMatrix(xtrain, y)

model = xgb.train(dict(xgb_params, silent=0),

                  dtrain,

                  num_boost_round=275)
importance = model.get_fscore(fmap='xgb.fmap')

importance = sorted(importance.items(), key=operator.itemgetter(1))



df = pd.DataFrame(importance, columns=['feature', 'fscore'])

df['fscore'] = df['fscore'] / df['fscore'].sum()
df.plot(figsize=(12, 12))

df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(12, 12))
print(r2_score(np.exp(y),

               np.exp(model.predict(dtrain))))
plt.scatter(np.exp(model.predict(dtrain)),np.exp(y))