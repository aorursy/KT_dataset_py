import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test['Target'] = np.nan
train['istrain'] = 1
test['istrain'] = 0
print ("Train Dataset: Rows, Columns: ", train.shape)
print ("Test Dataset: Rows, Columns: ", test.shape)
def preprocess(df):
    """
    Main feature engineering function.
    """
    def mk_categoricals(df, prefixes=None, subsets=None):
        """
        Converts one-hot-encoded categorical to true categorical.
        prefixes: list of prefixes of one-hot-encoded categorical variables
                  e.g. for variables
                      abastaguadentro, =1 if water provision inside the dwelling
                      abastaguafuera, =1 if water provision outside the dwelling
                      abastaguano, =1 if no water provision
                  we provide prefix "abastagua"
        subsets: dictionary {name_of_feature: [columns], ...}
                 e.g. for variables
                     public, "=1 electricity from CNFL,  ICE,  ESPH/JASEC"
                     planpri, =1 electricity from private plant
                     noelec, =1 no electricity in the dwelling
                     coopele, =1 electricity from cooperative
                 we provide {"electricity": ['public', 'planpri', 'noelec', 'coopele']}
        """
        def mk_category(dummies):
            assert (dummies.sum(axis=1) <= 1).all()
            nans = dummies.sum(axis=1) != 1
            if nans.any():
                dummies = dummies.assign(_na=nans.astype(int))
            return dummies.idxmax(axis=1).astype('category')

        categoricals = pd.DataFrame()

        if prefixes:
            for prefix in prefixes:
                columns = df.columns[df.columns.str.startswith(prefix)]
                categoricals[prefix] = mk_category(df[columns])
        if subsets:
            for feature_name, subset in subsets.items():
                categoricals[feature_name] = mk_category(df[subset])

        return categoricals
    groupper = df.groupby('idhogar')
    
    interactions = (pd.DataFrame(dict(
                    head_escolari=df.parentesco1 * df.escolari,
                    head_female=df.parentesco1 * df.female,
                    head_partner_escolari=df.parentesco2 * df.escolari))
                    .groupby(df.idhogar)
                    .max())
    # basic interaction features
    
    my_features = (groupper.mean()[['istrain',
                                    'escolari', 'age', 'hogar_nin', 
                                    'hogar_total', 'epared3', 'epared1',
                                    'etecho3', 'etecho1', 'eviv3', 'eviv1',
                                    'male',
                                    'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 
                                    'r4m3',
                                    'r4t1', 'r4t2', 'r4t3', 'v2a1', 'rooms', 
                                    'bedrooms',
                                    'meaneduc', 
                                    'SQBdependency', 'rez_esc', 'refrig', 
                                    'tamviv', 'overcrowding']]
                   .join(groupper.std()[['escolari', 'age']], 
                         rsuffix='_std')
                   .join(groupper[['escolari', 'age']].min(), rsuffix="_min")
                   .join(groupper[['escolari', 'age']].max(), rsuffix="_max")
                   .join(groupper[['dis']].sum(), rsuffix="_sum")
                   # partially based on
                   # https://www.kaggle.com/taindow/predicting-poverty-levels-with-r
                   .assign(child_rate=lambda x: x.hogar_nin / x.hogar_total,
                           wrf=lambda x: x.epared3 - x.epared1 +
                                         x.etecho3 - x.etecho1 +
                                         x.eviv3 - x.eviv1,
                           # wrf is an integral feature that measure
                           # quality of the house
                           escolari_range=lambda x: x.escolari_max - x.escolari_min,
                           age_range=lambda x: x.age_max - x.age_min,
                           rent_per_individual=lambda x: x.v2a1 / x.r4t3,
                           rent_per_child=lambda x: x.v2a1 / x.r4t1,
                           rent_per_over65=lambda x: x.v2a1 / x.r4t3,
                           rent_per_room=lambda x: x.v2a1 / x.rooms,
                           rent_per_bedroom=lambda x: x.v2a1 / x.bedrooms,
                           rooms_per_individual=lambda x: x.rooms / x.r4t3,
                           rooms_per_child=lambda x: x.rooms / x.r4t1,
                           bedrooms_per_individual=lambda x: x.bedrooms / x.r4t3,
                           bedrooms_per_child=lambda x: x.bedrooms / x.r4t1,
                           years_schooling_per_individual=lambda x: x.escolari / x.r4t3,
                           years_schooling_per_adult=lambda x: x.escolari / (x.r4t3 - x.r4t1),
                           years_schooling_per_child=lambda x: x.escolari / x.r4t3
                          )
                   .drop(['hogar_nin', 'hogar_total', 'epared3', 'epared1',
                                   'etecho3', 'etecho1', 'eviv3', 'eviv1'], 
                         axis=1)
                   .join(interactions)
                   .join(groupper[['computer', 'television', 
                                   'qmobilephone', 'v18q1']]
                         .mean().sum(axis=1).rename('technics'))
                   # we provide integral technical level as a new feature 
                   .assign(technics_per_individual=lambda x: x.technics / x.r4t3,
                           technics_per_child=lambda x: x.technics / x.r4t1)
                   .join(mk_categoricals(groupper.mean(), 
                                prefixes=['lugar', 'area', 'tipovivi', 
                                          'energcocinar', 
                                          'sanitario', 'pared', 'piso',
                                          'abastagua'],
                                subsets={'electricity': ['public', 
                                                         'planpri', 
                                                         'noelec', 
                                                         'coopele']}))
                  )
    return my_features
alldata = pd.concat([train,test])
y = alldata.groupby('idhogar').Target.median()
alldata = preprocess(alldata)
alldata = alldata.replace(np.inf,np.nan)
alldata.head()
catfeatures = []
for c in alldata.columns[1:]:
    if((alldata[c].dtype != 'float64')&(alldata[c].dtype != 'int64')):
        catfeatures.append(c)
        
print(catfeatures)
for c in catfeatures:
    alldata =  pd.concat([alldata,pd.get_dummies(alldata[c],prefix=c)],axis=1)
    
alldata.drop(catfeatures,inplace=True,axis=1)

for c in alldata.columns[1:]:
    ss = StandardScaler()
    alldata.loc[~alldata[c].isnull(),c] = ss.fit_transform(alldata.loc[~alldata[c].isnull(),c].values.astype('float32').reshape(-1,1))
    alldata[c].fillna(-999,inplace=True)
    
alldata['Target'] = y
mungedtrain = alldata[alldata.istrain==1].copy()
mungedtrain = mungedtrain[mungedtrain.Target.isin(np.array([1,2,3,4]))] 
mungedtest = alldata[alldata.istrain==0].copy()
mungedtest.drop('Target',inplace=True,axis=1)
mungedtrain = pd.concat([mungedtrain,pd.get_dummies(mungedtrain['Target'],prefix='Target')],axis=1)
mungedtrain.drop('Target',inplace=True,axis=1)

def Output(p):
    return 1./(1.+np.exp(-p))

def GP1(data):
    return Output(  0.100000*np.tanh(((((-3.0) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((-3.0) + (-3.0))) +
                    0.100000*np.tanh(((((-3.0) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((-3.0) + (-3.0))) + (((-3.0) * 2.0)))) * 2.0)) +
                    0.100000*np.tanh(((((((((-3.0) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((np.minimum(((((((((-3.0) * 2.0)) * 2.0)) * 2.0))), ((3.0)))) * 2.0)) +
                    0.100000*np.tanh(((((((((((-3.0) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((np.minimum(((((-3.0) + (-3.0)))), ((-3.0)))) * 2.0)) + (-3.0))) * 2.0)) +
                    0.100000*np.tanh(((((((-3.0) * 2.0)) + (((((((-3.0) + (-3.0))) * 2.0)) * 2.0)))) * 2.0)) +
                    0.100000*np.tanh(((((-3.0) + (np.minimum(((((-2.0) * 2.0))), ((np.minimum(((-3.0)), ((np.minimum(((-3.0)), ((-3.0)))))))))))) * 2.0)) +
                    0.100000*np.tanh(np.minimum(((np.minimum(((-3.0)), ((((((-3.0) * 2.0)) * 2.0)))))), ((-3.0)))) +
                    0.100000*np.tanh(((np.minimum(((((((((-3.0) * 2.0)) * 2.0)) * 2.0))), ((((((np.minimum(((-3.0)), ((-2.0)))) * 2.0)) * 2.0))))) * 2.0)) +
                    0.100000*np.tanh(np.minimum(((((np.minimum(((np.minimum(((-3.0)), ((-3.0))))), ((((((-3.0) * 2.0)) * 2.0))))) * 2.0))), ((-3.0)))) +
                    0.100000*np.tanh(((((-3.0) + (np.minimum(((((-3.0) * 2.0))), ((np.minimum(((-3.0)), ((np.minimum(((-2.0)), ((-3.0)))))))))))) * 2.0)) +
                    0.100000*np.tanh(((((np.minimum(((data["escolari_min"])), ((np.minimum(((np.minimum(((((-3.0) * 2.0))), ((-3.0))))), ((-2.0))))))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(np.minimum((((11.24597263336181641))), ((np.minimum(((-3.0)), ((np.minimum(((((-3.0) * 2.0))), ((-3.0)))))))))) +
                    0.100000*np.tanh(((((((np.minimum(((((-3.0) - (data["wrf"])))), ((data["escolari"])))) - (((data["escolari"]) * 2.0)))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(np.where(-3.0 < -998, ((np.minimum(((data["technics_per_individual"])), ((data["escolari"])))) / 2.0), (((14.09721565246582031)) * (((-1.0) - (data["escolari"])))) )) +
                    0.100000*np.tanh(np.minimum(((np.minimum(((-3.0)), ((-3.0))))), ((np.maximum(((-3.0)), ((-3.0))))))) +
                    0.100000*np.tanh(((((-3.0) + (((-3.0) - (((data["escolari"]) * 2.0)))))) - (((((data["escolari"]) * 2.0)) * 2.0)))) +
                    0.100000*np.tanh(((((((((((-2.0) - (((data["technics_per_individual"]) * 2.0)))) * 2.0)) - (((data["escolari"]) * 2.0)))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((np.where(data["wrf"] < -998, -1.0, -3.0 )) - (((data["escolari"]) * 2.0)))) - (((data["wrf"]) * 2.0)))) * 2.0)) +
                    0.100000*np.tanh(((np.minimum(((data["wrf"])), ((-3.0)))) + (((np.minimum(((-3.0)), ((-3.0)))) * (data["wrf"]))))) +
                    0.100000*np.tanh(((np.minimum(((np.maximum(((((-3.0) + (-3.0)))), ((-3.0))))), ((data["technics_per_individual"])))) * 2.0)) +
                    0.100000*np.tanh(((-3.0) + (-3.0))) +
                    0.100000*np.tanh(((((data["r4t1"]) + (((np.where(data["head_escolari"]<0, np.where(data["escolari_max"]>0, data["head_escolari"], data["r4t1"] ), -2.0 )) * 2.0)))) * 2.0)) +
                    0.100000*np.tanh(((((((((((-2.0) - (data["wrf"]))) - (data["wrf"]))) - (data["escolari"]))) - (data["escolari"]))) - (data["wrf"]))) +
                    0.100000*np.tanh(np.where((((data["technics_per_individual"]) + (data["escolari_max"]))/2.0)>0, -3.0, ((data["r4t1"]) + (np.where(data["escolari_max"] < -998, data["escolari_max"], data["r4t1"] ))) )) +
                    0.100000*np.tanh(((((((((((-1.0) - (data["technics_per_individual"]))) - (data["escolari"]))) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((((((((data["r4t1"]) - (data["piso_pisomoscer"]))) * 2.0)) - (data["head_escolari"]))) - (data["piso_pisomoscer"]))) - (data["piso_pisomoscer"]))) * 2.0)) +
                    0.100000*np.tanh(((((((((((-2.0) - (data["technics_per_individual"]))) - (data["technics_per_individual"]))) - (data["escolari"]))) * 2.0)) - (data["escolari"]))) +
                    0.100000*np.tanh(((((data["r4t1"]) + (-2.0))) + (((-2.0) + (data["r4t1"]))))) +
                    0.100000*np.tanh(((((-1.0) - (data["escolari_max"]))) - (((((((data["wrf"]) + (data["piso_pisomoscer"]))) + (data["technics_per_individual"]))) + (data["technics_per_individual"]))))) +
                    0.100000*np.tanh(((np.where(2.0 < -998, ((np.where(data["r4t1"] < -998, 2.0, -2.0 )) + (data["r4t1"])), -3.0 )) + (data["r4t1"]))) +
                    0.100000*np.tanh((-1.0*((((((data["head_escolari"]) + ((0.33609518408775330)))) + (((data["wrf"]) + (data["escolari"])))))))) +
                    0.100000*np.tanh((-1.0*((np.where(((data["technics_per_individual"]) + (data["wrf"]))>0, (5.0), data["technics_per_individual"] ))))) +
                    0.100000*np.tanh(np.where(data["escolari_min"]>0, ((-3.0) * (data["piso_pisomoscer"])), np.minimum(((((data["escolari_max"]) * (-3.0)))), ((data["piso_pisomoscer"]))) )) +
                    0.100000*np.tanh(np.where(data["piso_pisomoscer"]<0, np.where(data["piso_pisomoscer"]<0, data["head_escolari"], (-1.0*((data["head_escolari"]))) ), np.where(data["head_escolari"]>0, -3.0, data["piso_pisomoscer"] ) )) +
                    0.100000*np.tanh(((np.where(data["escolari_max"]>0, (((-1.0*((((data["technics_per_individual"]) + (data["escolari_max"])))))) - (data["technics_per_individual"])), data["technics_per_individual"] )) - (data["wrf"]))) +
                    0.100000*np.tanh(np.where(((data["escolari_max"]) + (data["technics_per_individual"]))>0, -3.0, (((((data["escolari_max"]) * 2.0)) > (data["escolari_max"]))*1.) )) +
                    0.100000*np.tanh(((((np.where(data["wrf"]>0, (-1.0*((np.where(data["head_escolari"]>0, data["piso_pisomoscer"], data["technics_per_individual"] )))), data["head_escolari"] )) * 2.0)) - (data["technics_per_individual"]))) +
                    0.100000*np.tanh(((((data["escolari_min"]) - (np.maximum(((np.maximum(((data["escolari"])), ((data["wrf"]))))), ((data["escolari_max"])))))) * ((((5.0)) - (data["wrf"]))))) +
                    0.100000*np.tanh(np.where(data["head_escolari"]>0, -2.0, np.maximum(((data["r4t1"])), ((data["escolari_min"]))) )) +
                    0.100000*np.tanh((-1.0*((np.where((((data["escolari_max"]) < (data["escolari"]))*1.)>0, data["piso_pisomoscer"], data["escolari_max"] ))))) +
                    0.100000*np.tanh(np.where(data["wrf"]<0, data["head_escolari"], np.where(data["head_escolari"]<0, data["wrf"], data["head_escolari"] ) )) +
                    0.100000*np.tanh((((((((data["escolari_max"]) > (data["head_escolari"]))*1.)) - (((data["piso_pisomoscer"]) * (data["head_escolari"]))))) - (np.maximum(((data["wrf"])), ((data["escolari_max"])))))) +
                    0.100000*np.tanh(((np.where((((np.tanh((data["head_escolari"]))) < (data["escolari_min"]))*1.)>0, -2.0, (((data["head_escolari"]) < (np.tanh((data["head_escolari"]))))*1.) )) * 2.0)) +
                    0.099980*np.tanh(((np.where(data["technics_per_individual"]>0, ((data["technics_per_individual"]) - ((3.0))), np.where(data["escolari_min"]>0, data["escolari_max"], data["technics_per_individual"] ) )) * 2.0)) +
                    0.100000*np.tanh((((data["escolari"]) < ((((data["overcrowding"]) > (-1.0))*1.)))*1.)) +
                    0.100000*np.tanh(np.where(data["escolari_max"]<0, np.where(data["escolari_min"]>0, np.where(data["technics_per_individual"]>0, -3.0, data["escolari_min"] ), data["technics_per_individual"] ), (-1.0*((data["technics_per_individual"]))) )))

def GP2(data):
    return Output(  0.100000*np.tanh(((((-3.0) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((-3.0) + (((-3.0) + (-3.0))))) +
                    0.100000*np.tanh(((((-3.0) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((np.where(-3.0 < -998, (14.51777839660644531), ((-3.0) * 2.0) )) * 2.0)) +
                    0.100000*np.tanh(((((np.minimum(((((-3.0) + (-3.0)))), ((((-3.0) * 2.0))))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((np.minimum(((((((np.minimum(((((-3.0) * 2.0))), ((-3.0)))) * 2.0)) * 2.0))), ((-3.0)))) * 2.0)) +
                    0.100000*np.tanh(((((((((((np.minimum((((7.25457954406738281))), ((-3.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(np.minimum(((((np.minimum(((((-3.0) * 2.0))), ((np.minimum(((np.minimum(((-3.0)), ((-3.0))))), ((-3.0))))))) * 2.0))), ((-3.0)))) +
                    0.100000*np.tanh(((((((((((((((((-1.0) - (data["technics_per_individual"]))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(np.minimum(((np.minimum((((((-1.0*(((7.0))))) + (-3.0)))), ((((((-3.0) * 2.0)) + (-3.0))))))), ((-3.0)))) +
                    0.100000*np.tanh(((((((((-3.0) - (((data["technics_per_individual"]) + (((data["technics_per_individual"]) * 2.0)))))) * 2.0)) - (data["wrf"]))) - (data["years_schooling_per_adult"]))) +
                    0.100000*np.tanh(((((((-3.0) + (np.where(data["wrf"]>0, -3.0, ((((data["escolari"]) * 2.0)) * (-3.0)) )))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((((((-2.0) - (data["wrf"]))) - (((data["escolari"]) + (data["escolari_min"]))))) * 2.0)) - (data["escolari"]))) * 2.0)) +
                    0.100000*np.tanh(np.minimum(((((-3.0) * 2.0))), ((((np.minimum(((np.minimum(((-3.0)), ((np.minimum(((-2.0)), ((-2.0)))))))), ((-3.0)))) * 2.0))))) +
                    0.100000*np.tanh(((np.where(((data["wrf"]) + (data["escolari"]))<0, np.where(data["escolari_min"]<0, (-1.0*((data["technics_per_individual"]))), -3.0 ), -3.0 )) * 2.0)) +
                    0.100000*np.tanh((-1.0*((np.where(np.where(data["wrf"]>0, 3.0, data["escolari"] )<0, np.where(data["escolari"]<0, data["wrf"], (4.0) ), (7.51970005035400391) ))))) +
                    0.100000*np.tanh(((-3.0) - (((((((3.0) * 2.0)) * 2.0)) * (np.maximum(((((-3.0) * (data["escolari"])))), ((data["escolari"])))))))) +
                    0.100000*np.tanh(((((np.where(((data["escolari_max"]) + (data["technics_per_individual"]))>0, (4.0), data["escolari_min"] )) * (-3.0))) * 2.0)) +
                    0.100000*np.tanh(np.where(np.where(data["years_schooling_per_child"]>0, (((data["escolari_max"]) > (data["years_schooling_per_child"]))*1.), data["escolari_max"] )<0, (((data["head_escolari"]) > (data["escolari"]))*1.), -3.0 )) +
                    0.100000*np.tanh(np.where(data["escolari"]<0, (((data["escolari"]) < (data["escolari_max"]))*1.), np.where(0.0<0, -3.0, -3.0 ) )) +
                    0.100000*np.tanh(((((((np.minimum(((-3.0)), ((-2.0)))) - (((((data["technics_per_individual"]) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(np.where(((((data["escolari"]) + (data["escolari"]))) + (data["wrf"]))>0, -3.0, (((data["wrf"]) < (data["escolari"]))*1.) )) +
                    0.100000*np.tanh(np.where(data["escolari"]<0, (0.0), np.minimum(((np.minimum(((-3.0)), ((((data["escolari"]) * 2.0)))))), ((-3.0))) )) +
                    0.100000*np.tanh((((((((((((-1.0*((((data["escolari_max"]) * 2.0))))) * (data["escolari_max"]))) - (data["escolari"]))) - (data["technics_per_individual"]))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(np.where(data["escolari_max"]<0, (((data["technics_per_individual"]) < (data["escolari_max"]))*1.), -3.0 )) +
                    0.100000*np.tanh(((-3.0) * (data["piso_pisomoscer"]))) +
                    0.100000*np.tanh((-1.0*((((np.where(data["escolari_max"] < -998, data["escolari_max"], ((data["wrf"]) * (data["escolari"])) )) + (data["escolari_max"])))))) +
                    0.100000*np.tanh(np.where(data["escolari_max"]>0, ((data["escolari_max"]) * (-2.0)), ((data["r4t1"]) + (np.where(-1.0<0, data["escolari_max"], data["escolari_max"] ))) )) +
                    0.100000*np.tanh((-1.0*((((((((((data["wrf"]) + (data["escolari"]))) + (data["escolari"]))) + (((data["technics_per_individual"]) + (data["head_escolari"]))))) * 2.0))))) +
                    0.100000*np.tanh(np.minimum(((np.minimum(((-3.0)), ((-3.0))))), ((np.minimum(((data["escolari"])), ((-3.0))))))) +
                    0.100000*np.tanh(((((((1.0) - (((data["escolari"]) + (data["technics_per_individual"]))))) - (((data["technics_per_individual"]) + (data["head_escolari"]))))) - (data["technics_per_individual"]))) +
                    0.100000*np.tanh((-1.0*((np.where(data["head_escolari"]>0, data["escolari"], np.where(data["technics_per_individual"]>0, data["escolari"], ((data["r4t1"]) - (((data["escolari"]) * 2.0))) ) ))))) +
                    0.100000*np.tanh(np.where(np.maximum(((data["wrf"])), ((data["technics_per_individual"])))<0, data["escolari_min"], (-1.0*((((((data["escolari"]) + (data["technics_per_individual"]))) + (data["escolari_max"]))))) )) +
                    0.100000*np.tanh(((np.where(data["piso_pisomoscer"] < -998, data["wrf"], (-1.0*((data["wrf"]))) )) - (data["piso_pisomoscer"]))) +
                    0.100000*np.tanh(np.where((-1.0*((((data["escolari_max"]) * (data["head_escolari"])))))<0, (-1.0*((((data["head_escolari"]) * (data["escolari_max"]))))), (-1.0*((data["head_escolari"]))) )) +
                    0.100000*np.tanh((((((data["escolari_max"]) > (data["technics_per_individual"]))*1.)) - (np.maximum(((((data["escolari_max"]) * 2.0))), ((((data["wrf"]) * (((data["technics_per_individual"]) * 2.0))))))))) +
                    0.100000*np.tanh(((np.where(data["technics_per_individual"]>0, (((-1.0*((data["escolari"])))) - (data["technics_per_individual"])), (((-1.0) < (data["escolari_min"]))*1.) )) * 2.0)) +
                    0.100000*np.tanh((-1.0*((np.maximum(((data["escolari_max"])), ((np.maximum(((data["escolari_max"])), ((((data["head_escolari"]) / 2.0))))))))))) +
                    0.100000*np.tanh(np.where(data["piso_pisomoscer"]>0, np.where(data["technics_per_individual"]>0, np.where(data["technics_per_individual"]>0, (-1.0*((data["piso_pisomoscer"]))), data["technics_per_individual"] ), data["escolari_max"] ), data["technics_per_individual"] )) +
                    0.100000*np.tanh(np.where(data["technics_per_individual"]>0, (-1.0*((((data["head_escolari"]) + (data["technics_per_individual"]))))), (((data["escolari"]) > (data["technics_per_individual"]))*1.) )) +
                    0.100000*np.tanh((-1.0*((np.maximum(((data["technics_per_individual"])), ((data["escolari_min"]))))))) +
                    0.100000*np.tanh(np.where(data["head_escolari"]<0, np.minimum(((data["escolari_min"])), ((0.0))), ((((data["head_escolari"]) * (data["escolari_min"]))) - (data["wrf"])) )) +
                    0.100000*np.tanh(np.where(data["escolari"] < -998, ((data["technics_per_individual"]) - (data["technics_per_individual"])), ((data["technics_per_individual"]) - (data["head_escolari"])) )) +
                    0.100000*np.tanh((-1.0*(((((np.where(data["escolari_max"] < -998, data["escolari_max"], 1.0 )) < (data["escolari_max"]))*1.))))) +
                    0.100000*np.tanh(np.where(np.tanh((data["escolari_max"]))>0, (-1.0*((data["r4t1"]))), ((data["r4t1"]) + (np.where(data["escolari"]>0, data["escolari"], -1.0 ))) )) +
                    0.100000*np.tanh(np.where(data["escolari"]<0, (((data["escolari_max"]) < (data["wrf"]))*1.), (-1.0*((((((data["escolari"]) + (data["wrf"]))) + (data["wrf"]))))) )) +
                    0.100000*np.tanh((-1.0*(((((((data["head_escolari"]) * (((data["head_escolari"]) * (((data["head_escolari"]) * (data["head_escolari"]))))))) + (data["wrf"]))/2.0))))) +
                    0.099980*np.tanh((((np.where(data["technics_per_individual"]>0, data["escolari_max"], data["technics_per_individual"] )) < (np.where(data["wrf"] < -998, data["wrf"], data["wrf"] )))*1.)) +
                    0.100000*np.tanh(((((data["wrf"]) - (-1.0))) * (np.where(data["overcrowding"]<0, -3.0, (6.0) )))) +
                    0.100000*np.tanh(np.where(((data["wrf"]) * (data["head_escolari"]))<0, (((((data["wrf"]) < (data["head_escolari"]))*1.)) - (data["head_escolari"])), data["escolari_min"] )))

def GP3(data):
    return Output(  0.100000*np.tanh(((np.minimum(((np.minimum(((((-1.0) * 2.0))), ((-2.0))))), ((((((-3.0) * 2.0)) * 2.0))))) * 2.0)) +
                    0.100000*np.tanh(((-2.0) + (-3.0))) +
                    0.100000*np.tanh(((((((((((-3.0) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((((((-3.0) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((-3.0) + (((np.minimum(((((-3.0) * 2.0))), ((((-3.0) * 2.0))))) * 2.0)))) +
                    0.100000*np.tanh(((np.minimum(((3.0)), ((-3.0)))) + (((np.minimum(((-3.0)), ((-3.0)))) * 2.0)))) +
                    0.100000*np.tanh(((((-3.0) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((np.minimum(((data["technics_per_individual"])), ((((((((-3.0) - (((data["head_escolari"]) / 2.0)))) * 2.0)) * 2.0))))) * 2.0)) + (-3.0))) +
                    0.100000*np.tanh(((-3.0) + (np.minimum(((-3.0)), ((-3.0)))))) +
                    0.100000*np.tanh(((np.minimum(((-3.0)), ((-3.0)))) * 2.0)) +
                    0.100000*np.tanh(np.minimum(((np.minimum(((((-3.0) * 2.0))), ((-3.0))))), ((np.minimum(((((-3.0) * (-3.0)))), ((-3.0))))))) +
                    0.100000*np.tanh(np.minimum(((((((((-3.0) * 2.0)) * 2.0)) * 2.0))), ((((-3.0) * 2.0))))) +
                    0.100000*np.tanh(np.minimum(((((-3.0) * 2.0))), ((np.minimum(((-3.0)), ((-2.0))))))) +
                    0.100000*np.tanh(np.where(-3.0 < -998, data["head_escolari"], (((9.86491298675537109)) * (((((-3.0) / 2.0)) - (data["head_escolari"])))) )) +
                    0.100000*np.tanh(((((-3.0) + (-3.0))) + (-3.0))) +
                    0.100000*np.tanh(((((((-2.0) - (data["head_escolari"]))) - (data["escolari_max"]))) - (data["head_escolari"]))) +
                    0.100000*np.tanh(((((np.where(data["head_escolari"] < -998, data["years_schooling_per_child"], ((-3.0) - (data["head_escolari"])) )) - (data["head_escolari"]))) - (data["head_escolari"]))) +
                    0.100000*np.tanh(np.where(data["technics_per_individual"]<0, np.where(data["escolari"]<0, ((data["escolari"]) * (-3.0)), -3.0 ), -3.0 )) +
                    0.100000*np.tanh(np.minimum(((-3.0)), ((((np.minimum(((-3.0)), ((-3.0)))) * 2.0))))) +
                    0.100000*np.tanh(np.where(np.maximum(((data["escolari"])), ((np.maximum(((data["head_escolari"])), ((data["head_escolari"]))))))>0, -3.0, (((data["head_escolari"]) > (-3.0))*1.) )) +
                    0.100000*np.tanh(np.where(((data["technics_per_individual"]) + (data["escolari_max"]))<0, ((((9.63199329376220703)) > (data["escolari_max"]))*1.), -3.0 )) +
                    0.100000*np.tanh(np.where(np.minimum(((-3.0)), ((-3.0)))>0, (4.66225624084472656), np.minimum(((-2.0)), ((np.minimum(((-1.0)), ((-3.0)))))) )) +
                    0.100000*np.tanh((-1.0*((np.where(((data["escolari_max"]) + (data["technics_per_individual"]))<0, np.minimum(((data["technics_per_individual"])), ((-3.0))), (11.90677547454833984) ))))) +
                    0.100000*np.tanh(np.where(data["escolari"]<0, data["escolari"], np.minimum(((np.minimum(((((-1.0) / 2.0))), ((data["escolari"]))))), ((np.minimum(((data["technics_per_individual"])), ((-3.0)))))) )) +
                    0.100000*np.tanh(np.where((((((data["escolari_max"]) / 2.0)) + (np.where(-3.0 < -998, data["years_schooling_per_individual"], data["years_schooling_per_individual"] )))/2.0)>0, -3.0, 1.0 )) +
                    0.100000*np.tanh(np.where(data["head_escolari"]<0, data["r4t1"], np.minimum((((5.71194410324096680))), ((-3.0))) )) +
                    0.100000*np.tanh(np.where(data["escolari"]>0, -3.0, np.where(data["escolari"]>0, data["escolari"], data["escolari_min"] ) )) +
                    0.100000*np.tanh(((((((((data["r4t1"]) - (data["technics_per_individual"]))) * 2.0)) * 2.0)) - (data["technics_per_individual"]))) +
                    0.100000*np.tanh(np.where(data["escolari"]<0, data["technics_per_individual"], np.where(data["piso_pisomoscer"] < -998, -3.0, np.where(data["technics_per_individual"]<0, -3.0, -3.0 ) ) )) +
                    0.100000*np.tanh(np.where(data["escolari_max"]<0, data["piso_pisomoscer"], (((-1.0*((data["head_escolari"])))) - (data["technics_per_individual"])) )) +
                    0.100000*np.tanh((-1.0*((((np.where(data["escolari_max"]>0, data["technics_per_individual"], ((((data["technics_per_individual"]) + (data["head_escolari"]))) * (data["escolari_min"])) )) + (data["escolari_max"])))))) +
                    0.100000*np.tanh((((-1.0*((data["technics_per_individual"])))) - (np.where(data["head_escolari"]<0, ((data["technics_per_individual"]) * (data["escolari"])), ((data["escolari"]) + (data["escolari_max"])) )))) +
                    0.100000*np.tanh((-1.0*((np.where(data["escolari_min"]<0, np.where(data["escolari_min"]<0, ((data["technics_per_individual"]) * (data["wrf"])), data["wrf"] ), ((data["technics_per_individual"]) * 2.0) ))))) +
                    0.100000*np.tanh(((((((((data["wrf"]) + (data["wrf"]))) * (data["r4t1"]))) - (((data["wrf"]) + (data["escolari_max"]))))) - (data["wrf"]))) +
                    0.100000*np.tanh((-1.0*((((data["escolari_max"]) * (np.where(data["escolari_min"] < -998, ((data["escolari_max"]) * (data["escolari_max"])), data["wrf"] ))))))) +
                    0.100000*np.tanh(((((((((data["r4t1"]) - (data["wrf"]))) - (data["technics_per_individual"]))) - (data["wrf"]))) - (((data["wrf"]) - (data["wrf"]))))) +
                    0.100000*np.tanh(np.where(data["escolari_min"]<0, np.where(data["escolari_min"]<0, data["technics_per_individual"], (-1.0*((data["escolari_min"]))) ), (-1.0*((data["technics_per_individual"]))) )) +
                    0.100000*np.tanh(((((2.0) - (((((data["escolari_max"]) * 2.0)) * (data["escolari_max"]))))) - (((data["wrf"]) * (((data["escolari_max"]) * 2.0)))))) +
                    0.100000*np.tanh(np.where(data["technics_per_individual"] < -998, 0.0, (((-1.0*((data["technics_per_individual"])))) * (data["escolari_min"])) )) +
                    0.100000*np.tanh((((((data["head_escolari"]) > ((((data["head_escolari"]) > (data["head_escolari"]))*1.)))*1.)) - (((data["head_escolari"]) + ((((data["escolari"]) > (data["head_escolari"]))*1.)))))) +
                    0.100000*np.tanh(np.where(data["escolari_min"]<0, np.where(data["escolari_min"]<0, (-1.0*((0.0))), data["head_escolari"] ), (-1.0*((data["head_escolari"]))) )) +
                    0.100000*np.tanh(np.where(np.where(data["escolari"]>0, data["escolari_min"], data["escolari_min"] )>0, (-1.0*((data["head_escolari"]))), data["escolari_min"] )) +
                    0.100000*np.tanh(((np.where(((data["head_escolari"]) - (data["escolari"]))>0, (((data["r4t1"]) < ((((data["r4t1"]) > (data["escolari"]))*1.)))*1.), data["r4t1"] )) * 2.0)) +
                    0.100000*np.tanh(np.where(((data["wrf"]) * (np.where(data["escolari_min"]<0, data["escolari_max"], data["escolari_min"] )))<0, data["piso_pisomoscer"], (((data["piso_pisomoscer"]) < (data["escolari_min"]))*1.) )) +
                    0.100000*np.tanh(np.where(((data["escolari_max"]) - (((data["escolari_min"]) + (-1.0))))<0, 3.0, np.where((8.0)<0, -1.0, data["escolari_max"] ) )) +
                    0.100000*np.tanh((-1.0*((((((((0.0)) < (0.0))*1.)) / 2.0))))) +
                    0.100000*np.tanh(np.tanh((((((((((((0.0) / 2.0)) / 2.0)) * (0.0))) / 2.0)) / 2.0)))) +
                    0.099980*np.tanh(((np.minimum(((0.0)), ((((np.minimum(((0.0)), ((0.0)))) / 2.0))))) * (-3.0))) +
                    0.100000*np.tanh(np.where(data["overcrowding"]>0, data["escolari"], ((data["overcrowding"]) - (data["escolari"])) )) +
                    0.100000*np.tanh(np.where(data["piso_pisomoscer"]<0, (((data["escolari_min"]) < ((((data["escolari_max"]) > ((((data["technics_per_individual"]) > (np.tanh((data["escolari_min"]))))*1.)))*1.)))*1.), data["escolari_min"] )))

def GP4(data):
    return Output(  0.100000*np.tanh(((((((2.0) + (((((data["escolari"]) * 2.0)) + (((data["wrf"]) + (((data["technics_per_individual"]) * 2.0)))))))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((np.where(data["wrf"]<0, np.where(data["escolari_max"]<0, (6.32442140579223633), data["escolari"] ), (11.70862483978271484) )) + ((((14.73191165924072266)) * (data["escolari"]))))) +
                    0.100000*np.tanh(((((((((((((((5.01557493209838867)) + (data["wrf"]))/2.0)) + (((((data["escolari"]) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((((((((((data["technics_per_individual"]) + (np.maximum(((data["escolari_max"])), ((data["escolari_min"])))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((np.maximum(((((((data["wrf"]) + (np.tanh((1.0))))) + (data["escolari"])))), ((data["escolari"])))) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((data["wrf"]) + (((((((data["years_schooling_per_adult"]) + (((data["technics_per_individual"]) + (1.0))))) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((((data["escolari"]) + (np.where(data["technics_per_individual"]>0, data["escolari_max"], 2.0 )))) + (((data["technics_per_individual"]) * 2.0)))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh((((((14.43354988098144531)) * (((data["technics_per_individual"]) + (np.maximum(((data["escolari"])), ((np.maximum(((data["wrf"])), ((((data["escolari_max"]) * 2.0)))))))))))) * 2.0)) +
                    0.100000*np.tanh((((((((((((12.09874343872070312)) * ((((data["technics_per_individual"]) + (data["escolari"]))/2.0)))) + ((7.0)))) * 2.0)) * 2.0)) + (data["escolari"]))) +
                    0.100000*np.tanh((((14.48213768005371094)) * ((((14.48213768005371094)) * (((data["technics_per_individual"]) + (np.maximum(((((data["escolari_min"]) + (data["wrf"])))), ((data["escolari_max"])))))))))) +
                    0.100000*np.tanh(((np.where(((data["escolari_max"]) + (data["technics_per_individual"]))<0, ((data["wrf"]) + (data["years_schooling_per_child"])), (8.23322868347167969) )) * 2.0)) +
                    0.100000*np.tanh(((((data["head_escolari"]) * 2.0)) + (np.where(data["escolari"]<0, data["escolari"], np.where(data["escolari"]>0, 3.0, data["escolari"] ) )))) +
                    0.100000*np.tanh(((data["escolari"]) + ((((10.0)) * (((((2.0) + (data["technics_per_individual"]))) + (np.minimum(((data["technics_per_individual"])), ((data["escolari"])))))))))) +
                    0.100000*np.tanh(np.where(((((data["escolari_max"]) * (data["technics_per_individual"]))) + (data["technics_per_individual"]))>0, (12.16580867767333984), ((data["head_escolari"]) + (data["escolari_max"])) )) +
                    0.100000*np.tanh(np.where(((data["technics_per_individual"]) + (data["escolari"]))<0, data["wrf"], np.maximum((((11.57767200469970703))), ((data["wrf"]))) )) +
                    0.100000*np.tanh(((np.where(np.where(data["escolari"]<0, data["escolari"], data["escolari"] )>0, 3.0, data["head_escolari"] )) * (((data["wrf"]) + (data["escolari_max"]))))) +
                    0.100000*np.tanh(((((data["escolari"]) + (((((((np.tanh((data["escolari"]))) + (data["escolari"]))) + (data["head_escolari"]))) * 2.0)))) * 2.0)) +
                    0.100000*np.tanh(np.maximum(((3.0)), (((9.0))))) +
                    0.100000*np.tanh(((((((((data["years_schooling_per_individual"]) + (((data["wrf"]) - ((((data["escolari_max"]) < (data["wrf"]))*1.)))))) * 2.0)) * 2.0)) + (data["wrf"]))) +
                    0.100000*np.tanh(((np.where(((data["escolari_max"]) * 2.0)>0, data["escolari_max"], data["years_schooling_per_adult"] )) * (((data["escolari_max"]) + (data["escolari_max"]))))) +
                    0.100000*np.tanh(((data["technics_per_individual"]) + (((((data["escolari_max"]) + (np.maximum(((data["escolari_max"])), ((data["escolari_max"])))))) * (np.maximum(((data["technics_per_individual"])), ((data["escolari_max"])))))))) +
                    0.100000*np.tanh(np.where(data["escolari"]>0, ((((data["technics_per_individual"]) + (data["head_escolari"]))) + (np.where(data["escolari_max"]>0, data["technics_per_individual"], data["escolari"] ))), data["escolari"] )) +
                    0.100000*np.tanh(((((data["wrf"]) + (data["escolari_max"]))) * (((((np.maximum(((data["technics_per_individual"])), ((data["escolari_max"])))) + (data["technics_per_individual"]))) + (data["escolari"]))))) +
                    0.100000*np.tanh(np.where((((1.0)) - (data["escolari_max"]))<0, ((data["escolari_max"]) * 2.0), np.maximum(((data["technics_per_individual"])), ((data["technics_per_individual"]))) )) +
                    0.100000*np.tanh(((np.where(data["years_schooling_per_adult"]>0, data["wrf"], np.where(data["years_schooling_per_adult"]>0, data["wrf"], (((data["wrf"]) < (data["technics_per_individual"]))*1.) ) )) * 2.0)) +
                    0.100000*np.tanh(((((((data["piso_pisomoscer"]) + (((data["head_escolari"]) - (data["r4t1"]))))) + (data["piso_pisomoscer"]))) + (((data["piso_pisomoscer"]) + (data["piso_pisomoscer"]))))) +
                    0.100000*np.tanh(np.where(data["wrf"]<0, np.where(data["escolari_min"]>0, data["head_escolari"], data["piso_pisomoscer"] ), np.where(data["piso_pisomoscer"]>0, data["head_escolari"], data["piso_pisomoscer"] ) )) +
                    0.100000*np.tanh(np.where(data["technics_per_individual"]>0, np.where(data["escolari"]<0, data["escolari"], data["technics_per_individual"] ), ((((data["r4t1"]) * (data["escolari"]))) - (data["escolari_min"])) )) +
                    0.100000*np.tanh(((np.maximum(((np.maximum(((data["escolari_max"])), ((0.0))))), ((data["escolari_max"])))) * (np.maximum(((np.maximum(((data["technics_per_individual"])), ((data["head_escolari"]))))), ((data["technics_per_individual"])))))) +
                    0.100000*np.tanh(np.where(data["technics_per_individual"] < -998, data["escolari"], np.where(data["escolari"]>0, ((((data["technics_per_individual"]) * 2.0)) * 2.0), (-1.0*((data["r4t1"]))) ) )) +
                    0.100000*np.tanh(((((data["head_escolari"]) * (data["escolari_max"]))) * (data["escolari_max"]))) +
                    0.100000*np.tanh(((((data["technics_per_individual"]) * 2.0)) * (((((((((data["escolari"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)))) +
                    0.100000*np.tanh(np.where((((1.0) > ((((((data["escolari_max"]) + (data["escolari_max"]))/2.0)) / 2.0)))*1.) < -998, data["escolari_max"], (((1.0) < (data["escolari_max"]))*1.) )) +
                    0.100000*np.tanh(np.where(data["escolari_max"]>0, ((((-1.0) + (data["r4t1"]))) + (data["escolari_max"])), (-1.0*((data["r4t1"]))) )) +
                    0.100000*np.tanh(np.where(((data["head_escolari"]) - (data["escolari_min"]))>0, ((((((data["escolari_min"]) / 2.0)) - (data["escolari_max"]))) - (data["head_escolari"])), data["escolari_max"] )) +
                    0.100000*np.tanh(np.where(data["wrf"]>0, np.where(data["escolari_max"] < -998, data["wrf"], data["escolari_max"] ), (-1.0*((data["r4t1"]))) )) +
                    0.100000*np.tanh(np.where((((data["escolari"]) > ((-1.0*((data["escolari_min"])))))*1.) < -998, data["escolari_min"], ((data["wrf"]) - (data["escolari_min"])) )) +
                    0.100000*np.tanh((((((data["wrf"]) < (data["escolari_max"]))*1.)) - ((((data["escolari_max"]) < (np.tanh((data["wrf"]))))*1.)))) +
                    0.100000*np.tanh(((((data["technics_per_individual"]) * 2.0)) * (((((np.maximum(((data["technics_per_individual"])), ((data["escolari_min"])))) * 2.0)) * 2.0)))) +
                    0.100000*np.tanh(np.where((((data["technics_per_individual"]) < (data["piso_pisomoscer"]))*1.)>0, np.where(data["escolari_max"]>0, -3.0, (((data["technics_per_individual"]) < (data["escolari_max"]))*1.) ), data["piso_pisomoscer"] )) +
                    0.100000*np.tanh(np.where(((((((data["technics_per_individual"]) > (data["escolari_min"]))*1.)) > (data["wrf"]))*1.)>0, (((data["piso_pisomoscer"]) < (data["technics_per_individual"]))*1.), data["piso_pisomoscer"] )) +
                    0.100000*np.tanh((((data["escolari"]) > (((((((0.37607082724571228)) * 2.0)) > (((((((0.37607082724571228)) * 2.0)) > (((data["escolari"]) * 2.0)))*1.)))*1.)))*1.)) +
                    0.100000*np.tanh((-1.0*((np.where(data["escolari"] < -998, data["escolari"], np.where(((data["r4t1"]) + (data["escolari"]))>0, (1.35394728183746338), data["escolari"] ) ))))) +
                    0.100000*np.tanh(((((data["head_escolari"]) + (((data["head_escolari"]) + ((((data["head_escolari"]) > (data["wrf"]))*1.)))))) * ((((data["escolari_min"]) > (data["head_escolari"]))*1.)))) +
                    0.100000*np.tanh((-1.0*((np.where(((data["escolari"]) - (data["escolari_min"]))<0, data["r4t1"], (-1.0*((np.where(data["escolari_max"]<0, data["r4t1"], data["wrf"] )))) ))))) +
                    0.100000*np.tanh(np.where((((data["escolari_max"]) < (-2.0))*1.) < -998, (((1.0) < (data["escolari_max"]))*1.), (((1.0) < (data["escolari_max"]))*1.) )) +
                    0.100000*np.tanh(np.where(((data["escolari"]) - (-1.0))<0, data["wrf"], np.where(data["wrf"]>0, -1.0, ((-1.0) * (data["wrf"])) ) )) +
                    0.099980*np.tanh(((((data["wrf"]) - (data["piso_pisomoscer"]))) + (((data["wrf"]) - ((((data["piso_pisomoscer"]) > (((data["piso_pisomoscer"]) - (data["wrf"]))))*1.)))))) +
                    0.100000*np.tanh(np.where(data["overcrowding"]>0, ((((((data["overcrowding"]) + (data["escolari_min"]))) * (-3.0))) - (data["wrf"])), data["wrf"] )) +
                    0.100000*np.tanh(((((np.minimum(((data["wrf"])), ((data["piso_pisomoscer"])))) * ((((data["head_escolari"]) + (np.where(data["escolari_min"]>0, data["wrf"], data["technics_per_individual"] )))/2.0)))) * 2.0)))
gp = pd.DataFrame()
gp['isone'] = GP1(mungedtrain)
gp['istwo'] = GP2(mungedtrain)
gp['isthree'] = GP3(mungedtrain)
gp['isfour'] = GP4(mungedtrain)
gp = gp.div(gp.sum(axis=1), axis=0)

actual_labels = np.argmax(mungedtrain[mungedtrain.columns[-4:]].values, axis=1)+1
pred_labels = np.argmax(gp.values, axis=1)+1
f1_score(actual_labels,pred_labels,average='micro')
X_embedded = TSNE(n_components=2).fit_transform(gp)
cm = plt.cm.get_cmap('RdYlBu')
fig, axes = plt.subplots(1, 1, figsize=(15, 15))
sc = axes.scatter(X_embedded[:,0], X_embedded[:,1], alpha=1., c=(actual_labels), cmap=cm, s=50)
cbar = fig.colorbar(sc, ax=axes)
cbar.set_label('Target')
_ = axes.set_title("Clustering colored by target")