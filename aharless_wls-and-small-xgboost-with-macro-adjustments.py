# Parameters

use_pipe = True  

weight_base = "2011-08-19"

xgb_lr = .02 #  Learning rate.  I use .007, but needs to be larger to run on Kaggle.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv')

macro = pd.read_csv('../input/macro.csv')

test = pd.read_csv('../input/test.csv')
dfa = pd.concat([train, test])  # "dfa" stands for "data frame all"

# Eliminate spaces and special characters in area names

dfa.loc[:,"sub_area"] = dfa.sub_area.str.replace(" ","").str.replace("\'","").str.replace("-","")

dfa = dfa.merge(macro, 

                on='timestamp', suffixes=['','_macro'])
dfa["fullzero"] = (dfa.full_sq==0)

dfa["fulltiny"] = (dfa.full_sq<4)

dfa["fullhuge"] = (dfa.full_sq>2000)

dfa["lnfull"] = np.log(dfa.full_sq+1)



dfa["nolife"] = dfa.life_sq.isnull()

dfa.life_sq = dfa.life_sq.fillna(dfa.life_sq.median())

dfa["lifezero"] = (dfa.life_sq==0)

dfa["lifetiny"] = (dfa.life_sq<4)

dfa["lifehuge"] = (dfa.life_sq>2000)

dfa["lnlife"] = np.log( dfa.life_sq + 1 )



dfa["nofloor"] = dfa.floor.isnull()

dfa.floor = dfa.floor.fillna(dfa.floor.median())

dfa["floor1"] = (dfa.floor==1)

dfa["floor0"] = (dfa.floor==0)

dfa["floorhuge"] = (dfa.floor>50)

dfa["lnfloor"] = np.log(dfa.floor+1)



dfa["nomax"] = dfa.max_floor.isnull()

dfa.max_floor = dfa.max_floor.fillna(dfa.max_floor.median())

dfa["max1"] = (dfa.max_floor==1)

dfa["max0"] = (dfa.max_floor==0)

dfa["maxhuge"] = (dfa.max_floor>80)

dfa["lnmax"] = np.log(dfa.max_floor+1)



dfa["norooms"] = dfa.num_room.isnull()

dfa.num_room = dfa.num_room.fillna(dfa.num_room.median())

dfa["zerorooms"] = (dfa.num_room==0)

dfa["lnrooms"] = np.log( dfa.num_room + 1 )



dfa["nokitch"] = dfa.kitch_sq.isnull()

dfa.kitch_sq = dfa.kitch_sq.fillna(dfa.kitch_sq.median())

dfa["kitch1"] = (dfa.kitch_sq==1)

dfa["kitch0"] = (dfa.kitch_sq==0)

dfa["kitchhuge"] = (dfa.kitch_sq>400)

dfa["lnkitch"] = np.log(dfa.kitch_sq+1)
dfa["material0"] = dfa.material.isnull()

dfa["material1"] = (dfa.material==1)

dfa["material2"] = (dfa.material==2)

dfa["material3"] = (dfa.material==3)

dfa["material4"] = (dfa.material==4)

dfa["material5"] = (dfa.material==5)

dfa["material6"] = (dfa.material==6)



# "state" isn't explained but it looks like an ordinal number, so for now keep numeric

dfa.loc[dfa.state>5,"state"] = np.NaN  # Value 33 seems to be invalid; others all 1-4

dfa.state = dfa.state.fillna(dfa.state.median())



# product_type gonna be ugly because there are missing values in the test set but not training

# Check for the same problem with other variables

dfa["owner_occ"] = (dfa.product_type=='OwnerOccupier')

dfa.owner_occ.fillna(dfa.owner_occ.mean())



dfa = pd.get_dummies(dfa, columns=['sub_area'], drop_first=True)
# Build year is ugly

# Can be missing

# Can be zero

# Can be one

# Can be some ridiculous pre-Medieval number

# Can be some invalid huge number like 20052009

# Can be some other invalid huge number like 4965

# Can be a reasonable number but later than purchase year

# Can be equal to purchase year

# Can be a reasonable nubmer before purchase year



dfa.loc[dfa.build_year>2030,"build_year"] = np.NaN

dfa["nobuild"] = dfa.build_year.isnull()

dfa["sincebuild"] = pd.to_datetime(dfa.timestamp).dt.year - dfa.build_year

dfa.sincebuild.fillna(dfa.sincebuild.median(),inplace=True)

dfa["futurebuild"] = (dfa.sincebuild < 0)

dfa["newhouse"] = (dfa.sincebuild==0)

dfa["tooold"] = (dfa.sincebuild>1000)

dfa["build0"] = (dfa.build_year==0)

dfa["build1"] = (dfa.build_year==1)

dfa["untilbuild"] = -dfa.sincebuild.apply(np.min, args=[0]) # How many years until planned build

dfa["lnsince"] = dfa.sincebuild.mul(dfa.sincebuild>0).add(1).apply(np.log)
# Interaction terms

dfa["fullzero_Xowner"] = dfa.fullzero.astype("float64") * dfa.owner_occ

dfa["fulltiny_Xowner"] = dfa.fulltiny.astype("float64") * dfa.owner_occ

dfa["fullhuge_Xowner"] = dfa.fullhuge.astype("float64") * dfa.owner_occ

dfa["lnfull_Xowner"] = dfa.lnfull * dfa.owner_occ

dfa["nofloor_Xowner"] = dfa.nofloor.astype("float64") * dfa.owner_occ

dfa["floor0_Xowner"] = dfa.floor0.astype("float64") * dfa.owner_occ

dfa["floor1_Xowner"] = dfa.floor1.astype("float64") * dfa.owner_occ

dfa["lnfloor_Xowner"] = dfa.lnfloor * dfa.owner_occ

dfa["max1_Xowner"] = dfa.max1.astype("float64") * dfa.owner_occ

dfa["max0_Xowner"] = dfa.max0.astype("float64") * dfa.owner_occ

dfa["maxhuge_Xowner"] = dfa.maxhuge.astype("float64") * dfa.owner_occ

dfa["lnmax_Xowner"] = dfa.lnmax * dfa.owner_occ

dfa["kitch1_Xowner"] = dfa.kitch1.astype("float64") * dfa.owner_occ

dfa["kitch0_Xowner"] = dfa.kitch0.astype("float64") * dfa.owner_occ

dfa["kitchhuge_Xowner"] = dfa.kitchhuge.astype("float64") * dfa.owner_occ

dfa["lnkitch_Xowner"] = dfa.lnkitch * dfa.owner_occ

dfa["nobuild_Xowner"] = dfa.nobuild.astype("float64") * dfa.owner_occ

dfa["newhouse_Xowner"] = dfa.newhouse.astype("float64") * dfa.owner_occ

dfa["tooold_Xowner"] = dfa.tooold.astype("float64") * dfa.owner_occ

dfa["build0_Xowner"] = dfa.build0.astype("float64") * dfa.owner_occ

dfa["build1_Xowner"] = dfa.build1.astype("float64") * dfa.owner_occ

dfa["lnsince_Xowner"] = dfa.lnsince * dfa.owner_occ

dfa["state_Xowner"] = dfa.state * dfa.owner_occ
dfa["lnruboil"] = np.log( dfa.oil_urals * dfa.usdrub )
# Sets of features that go together



# Features derived from full_sq

fullvars = ["fullzero", "fulltiny",

           # For now I'm going to drop the one "fullhuge" case. Later use dummy, maybe.

           #"fullhuge",

           "lnfull" ]



# Features derived from floor

floorvars = ["nofloor", "floor1", "floor0",

             # floorhuge isn't very important, and it's causing problems, so drop it

             #"floorhuge", 

             "lnfloor"]



# Features derived from max_floor

maxvars = ["max1", "max0", "maxhuge", "lnmax"]



# Features derived from kitch_sq

kitchvars = ["kitch1", "kitch0", "kitchhuge", "lnkitch"]



# Features derived from bulid_year

buildvars = ["nobuild", "futurebuild", "newhouse", "tooold", 

             "build0", "build1", "untilbuild", "lnsince"]



# Features (dummy set) derived from material

matervars = ["material1", "material2",  # material3 is rare, so lumped in with missing 

             "material4", "material5", "material6"]



# Features derived from interaction of floor and product_type

floorXvars = ["nofloor_Xowner", "floor1_Xowner", "lnfloor_Xowner"]



# Features derived from interaction of kitch_sq and product_type

kitchXvars = ["kitch1_Xowner", "kitch0_Xowner", "lnkitch_Xowner"]



# Features (dummy set) derived from sub_area

subarvars = [

       'sub_area_Akademicheskoe',

       'sub_area_Altufevskoe', 'sub_area_Arbat',

       'sub_area_Babushkinskoe', 'sub_area_Basmannoe', 'sub_area_Begovoe',

       'sub_area_Beskudnikovskoe', 'sub_area_Bibirevo',

       'sub_area_BirjulevoVostochnoe', 'sub_area_BirjulevoZapadnoe',

       'sub_area_Bogorodskoe', 'sub_area_Brateevo', 'sub_area_Butyrskoe',

       'sub_area_Caricyno', 'sub_area_Cheremushki',

       'sub_area_ChertanovoCentralnoe', 'sub_area_ChertanovoJuzhnoe',

       'sub_area_ChertanovoSevernoe', 'sub_area_Danilovskoe',

       'sub_area_Dmitrovskoe', 'sub_area_Donskoe', 'sub_area_Dorogomilovo',

       'sub_area_FilevskijPark', 'sub_area_FiliDavydkovo',

       'sub_area_Gagarinskoe', 'sub_area_Goljanovo',

       'sub_area_Golovinskoe', 'sub_area_Hamovniki',

       'sub_area_HoroshevoMnevniki', 'sub_area_Horoshevskoe',

       'sub_area_Hovrino', 'sub_area_Ivanovskoe', 'sub_area_Izmajlovo',

       'sub_area_Jakimanka', 'sub_area_Jaroslavskoe', 'sub_area_Jasenevo',

       'sub_area_JuzhnoeButovo', 'sub_area_JuzhnoeMedvedkovo',

       'sub_area_JuzhnoeTushino', 'sub_area_Juzhnoportovoe',

       'sub_area_Kapotnja', 'sub_area_Konkovo', 'sub_area_Koptevo',

       'sub_area_KosinoUhtomskoe', 'sub_area_Kotlovka',

       'sub_area_Krasnoselskoe', 'sub_area_Krjukovo',

       'sub_area_Krylatskoe', 'sub_area_Kuncevo', 

       'sub_area_Kuzminki', 'sub_area_Lefortovo', 'sub_area_Levoberezhnoe',

       'sub_area_Lianozovo', 'sub_area_Ljublino', 'sub_area_Lomonosovskoe',

       'sub_area_Losinoostrovskoe', 'sub_area_Marfino',

       'sub_area_MarinaRoshha', 'sub_area_Marino', 'sub_area_Matushkino',

       'sub_area_Meshhanskoe', 'sub_area_Metrogorodok', 'sub_area_Mitino',

       'sub_area_MoskvorecheSaburovo',

       'sub_area_Mozhajskoe', 'sub_area_NagatinoSadovniki',

       'sub_area_NagatinskijZaton', 'sub_area_Nagornoe',

       'sub_area_Nekrasovka', 'sub_area_Nizhegorodskoe',

       'sub_area_NovoPeredelkino', 'sub_area_Novogireevo',

       'sub_area_Novokosino', 'sub_area_Obruchevskoe',

       'sub_area_OchakovoMatveevskoe', 'sub_area_OrehovoBorisovoJuzhnoe',

       'sub_area_OrehovoBorisovoSevernoe', 'sub_area_Ostankinskoe',

       'sub_area_Otradnoe', 'sub_area_Pechatniki', 'sub_area_Perovo',

       'sub_area_PokrovskoeStreshnevo', 'sub_area_PoselenieDesjonovskoe',

       'sub_area_PoselenieFilimonkovskoe', 

       'sub_area_PoselenieKrasnopahorskoe',

       'sub_area_PoselenieMoskovskij', 'sub_area_PoselenieMosrentgen',

       'sub_area_PoselenieNovofedorovskoe',

       'sub_area_PoseleniePervomajskoe', 'sub_area_PoselenieRjazanovskoe',

       'sub_area_PoselenieRogovskoe', 

       'sub_area_PoselenieShherbinka', 'sub_area_PoselenieSosenskoe',

       'sub_area_PoselenieVnukovskoe',  

       'sub_area_PoselenieVoskresenskoe', 'sub_area_Preobrazhenskoe',

       'sub_area_Presnenskoe', 'sub_area_ProspektVernadskogo',

       'sub_area_Ramenki', 'sub_area_Rjazanskij', 'sub_area_Rostokino',

       'sub_area_Savelki', 'sub_area_Savelovskoe', 'sub_area_Severnoe',

       'sub_area_SevernoeButovo', 'sub_area_SevernoeIzmajlovo',

       'sub_area_SevernoeMedvedkovo', 'sub_area_SevernoeTushino',

       'sub_area_Shhukino', 'sub_area_Silino', 'sub_area_Sokol',

       'sub_area_SokolinajaGora', 'sub_area_Sokolniki',

       'sub_area_Solncevo', 'sub_area_StaroeKrjukovo', 'sub_area_Strogino',

       'sub_area_Sviblovo', 'sub_area_Taganskoe', 'sub_area_Tekstilshhiki',

       'sub_area_TeplyjStan', 'sub_area_Timirjazevskoe',

       'sub_area_Troickijokrug', 'sub_area_TroparevoNikulino',

       'sub_area_Tverskoe', 'sub_area_Veshnjaki', 

       'sub_area_Vojkovskoe', 

       'sub_area_VostochnoeDegunino', 'sub_area_VostochnoeIzmajlovo',

       'sub_area_VyhinoZhulebino', 'sub_area_Zamoskvoreche',

       'sub_area_ZapadnoeDegunino', 'sub_area_Zjablikovo', 'sub_area_Zjuzino'

       ]





# Lump together small sub_areas



dfa = dfa.assign( sub_area_SmallSW =

   dfa.sub_area_PoselenieMihajlovoJarcevskoe + 

   dfa.sub_area_PoselenieKievskij +

   dfa.sub_area_PoselenieKlenovskoe +

   dfa.sub_area_PoselenieVoronovskoe +

   dfa.sub_area_PoselenieShhapovskoe )



dfa = dfa.assign( sub_area_SmallNW =

   dfa.sub_area_Molzhaninovskoe +

   dfa.sub_area_Kurkino )



dfa = dfa.assign( sub_area_SmallW =

   dfa.sub_area_PoselenieMarushkinskoe +

   dfa.sub_area_Vnukovo +

   dfa.sub_area_PoselenieKokoshkino )



dfa = dfa.assign( sub_area_SmallN =

   dfa.sub_area_Vostochnoe +

   dfa.sub_area_Alekseevskoe )



subarvars += ["sub_area_SmallSW", "sub_area_SmallNW", "sub_area_SmallW", "sub_area_SmallN"]

                 





# For now eliminate case with ridiculous value of full_sq

dfa = dfa[~dfa.fullhuge]



    

# Independent features



indievars = ["owner_occ", "state", "state_Xowner"]





# Complete list of features to use for fit



allvars = fullvars + floorvars + maxvars + kitchvars + buildvars + matervars

allvars += floorXvars + kitchXvars + subarvars + indievars
# The normalized target variable:  log real sale price

training = dfa[dfa.price_doc.notnull()]

training.lnrp = training.price_doc.div(training.cpi).apply(np.log)

y = training.lnrp



# Features to use in heteroskedasticity model if I go back to that

million1 = (training.price_doc==1e6)

million2 = (training.price_doc==2e6)

million3 = (training.price_doc==3e6)



# Create X matrix for fitting

keep = allvars + ['timestamp']  # Need to keep timestamp to calculate weights

X = training[keep] 
def get_weights(df):

    # Weight cases linearly on time

    # with later cases (more like test data) weighted more heavily

    basedate = pd.to_datetime(weight_base).toordinal() # Basedate gets a weight of zero

    wtd = pd.to_datetime(df.timestamp).apply(lambda x: x.toordinal()) - basedate

    wts = np.array(wtd)/1e3 # The denominator here shouldn't matter, just gives nice numbers.

    return wts
wts = get_weights(X)

X = X.drop("timestamp", axis=1)
from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.pipeline import make_pipeline



# Make a pipeline that transforms X

pipe = make_pipeline(Imputer(), StandardScaler())

pipe.fit(X)
from sklearn.linear_model import LinearRegression



lr = LinearRegression(fit_intercept=True)

lr.fit(pipe.transform(X), y, sample_weight=wts)

lr.score( pipe.transform(X), y ) # Unweighted R^2, just to see what it looks like
# Function to create an indicator array that selects positions

#   corresponding to a set of features from the regression



def get_selector( df, varnames ):

    selector = np.zeros( df.shape[1] )

    selector[[df.columns.get_loc(x) for x in varnames]] = 1

    return( selector )
def append_composite( df, varnames, name, X, Xuse, estimator ):

    selector = get_selector(X, varnames)

    v = pd.Series( np.matmul( Xuse, selector*estimator.coef_ ), 

                   name=name, index=df.index )

    return( pd.concat( [df, v], axis=1 ) )
Xuse = pipe.transform(X)



vars = {"fullv":fullvars,     "floorv":floorvars,   "maxv":maxvars, 

        "kitchv":kitchvars,   "buildv":buildvars,   "materv":matervars, 

        "floorxv":floorXvars, "kitchxv":kitchXvars, "subarv":subarvars}

for v in vars:

    training = append_composite( training, vars[v], v, X, Xuse, lr )



shortvarlist = list(vars.keys())

shortvarlist += indievars



Xshort = training[shortvarlist]



pipe1 = make_pipeline(Imputer(), StandardScaler())

pipe1.fit(Xshort)
# Fit again to make sure result is same

lr1 = LinearRegression(fit_intercept=True)

lr1.fit(pipe1.transform(Xshort), y, sample_weight=wts)

lr1.score( pipe1.transform(Xshort), y )
# Set up test data



testing = dfa[dfa.price_doc.isnull()]



df_test_full = pd.DataFrame(columns=X.columns)

for column in df_test_full.columns:

        df_test_full[column] = testing[column]        



Xuse = pipe.transform(df_test_full)

for v in vars:

    df_test_full = append_composite( df_test_full, vars[v], v, X, Xuse, lr )



df_test = pd.DataFrame(columns=Xshort.columns)

for column in df_test.columns:

        df_test[column] = df_test_full[column]  
def append_series( X_train, X_test, train_input, test_input, sername ):

    vtrain = pd.Series( train_input[sername], name=sername, index=X_train.index )

    X_train_out = pd.concat( [X_train, vtrain], axis=1 )

    vtest = pd.Series( test_input[sername], name=sername, index=X_test.index )

    X_test_out = pd.concat( [X_test, vtest], axis=1 )

    return( X_train_out, X_test_out )
# Arbitrary down-weighting of ridiculous prices

wts *= (1 - .2*million1 - .1*million2 - .05*million3)
vars_to_add = [

    "kindergarten_km", 

    "railroad_km", 

    "swim_pool_km", 

    "public_transport_station_km",

    "big_road1_km",

    "big_road2_km",

    "church_synagogue_km",

    "ttk_km",

    "metro_min_walk",

    "kremlin_km",

    "mosque_km",

]

Xdata_train = Xshort

Xdata_test = df_test

print( Xdata_train.shape )

print( Xdata_test.shape )

for v in vars_to_add:

    Xdata_train, Xdata_test = append_series( Xdata_train, Xdata_test, training, testing, v )

print( Xdata_train.shape )

print( Xdata_test.shape )
y.shape
import xgboost as xgb

import joblib

xgb_params = {

    'eta': xgb_lr,

    'max_depth': 6,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'nthread': 4,

    'silent': 1,

}





dtrain = xgb.DMatrix(Xdata_train, y, weight=wts)

dtest = xgb.DMatrix(Xdata_test)
%matplotlib inline

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=2000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)

# cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

cv_output["test-rmse-mean"][len(cv_output)-1]
num_boost_rounds = len(cv_output)

print( num_boost_rounds )

model = xgb.train(xgb_params, dtrain, num_boost_round= num_boost_rounds)
fig, ax = plt.subplots(1, 1, figsize=(8, 13))

xgb.plot_importance(model, height=0.5, ax=ax)
y_predict = model.predict(dtest)

predictions = np.exp(y_predict)*testing.cpi



# And put this in a dataframe

predxgb_df = pd.DataFrame()

predxgb_df['id'] = testing['id']

predxgb_df['price_doc'] = predictions

predxgb_df.head()
predxgb_df.to_csv('xgb_predicitons.csv', index=False)
test["timestamp"] = pd.to_datetime(test["timestamp"])

test["year"]  = test["timestamp"].dt.year

test["month"] = test["timestamp"].dt.month

test["yearmonth"] = 100*test.year + test.month

test_ids = test[["yearmonth","id"]]

test_data = test_ids.merge(predxgb_df,on="id")



test_prices = test_data[["yearmonth","price_doc"]]

test_p = test_prices.groupby("yearmonth").median()

test_p.head()
import statsmodels.api as sm
macro["timestamp"] = pd.to_datetime(macro["timestamp"])

macro["year"]  = macro["timestamp"].dt.year

macro["month"] = macro["timestamp"].dt.month

macro["yearmonth"] = 100*macro.year + macro.month

macmeds = macro.groupby("yearmonth").median()



train["timestamp"] = pd.to_datetime(train["timestamp"])

train["year"]  = train["timestamp"].dt.year

train["month"] = train["timestamp"].dt.month

train["yearmonth"] = 100*train.year + train.month

prices = train[["yearmonth","price_doc"]]

p = prices.groupby("yearmonth").median()



df = macmeds.join(p)
#  Adapted from code at http://adorio-research.org/wordpress/?p=7595

#  Original post was dated May 31st, 2010

#    but was unreachable last time I tried



import numpy.matlib as ml

 

def almonZmatrix(X, maxlag, maxdeg):

    """

    Creates the Z matrix corresponding to vector X.

    """

    n = len(X)

    Z = ml.zeros((len(X)-maxlag, maxdeg+1))

    for t in range(maxlag,  n):

       #Solve for Z[t][0].

       Z[t-maxlag,0] = sum([X[t-lag] for lag in range(maxlag+1)])

       for j in range(1, maxdeg+1):

             s = 0.0

             for i in range(1, maxlag+1):       

                s += (i)**j * X[t-i]

             Z[t-maxlag,j] = s

    return Z
y_macro = df.price_doc.div(df.cpi).apply(np.log).loc[201108:201506]

nobs = 47  # August 2011 through June 2015, months with price_doc data

tblags = 5    # Number of lags used on PDL for Trade Balance

mrlags = 5    # Number of lags used on PDL for Mortgage Rate

ztb = almonZmatrix(df.balance_trade.loc[201103:201506].as_matrix(), tblags, 1)

zmr = almonZmatrix(df.mortgage_rate.loc[201103:201506].as_matrix(), mrlags, 1)

columns = ['tb0', 'tb1', 'mr0', 'mr1']

z = pd.DataFrame( np.concatenate( (ztb, zmr), axis=1), y_macro.index.values, columns )

X_macro = sm.add_constant( z )
macro_fit = sm.OLS(y_macro, X_macro).fit()
test_cpi = df.cpi.loc[201507:201605]

test_index = test_cpi.index

ztb_test = almonZmatrix(df.balance_trade.loc[201502:201605].as_matrix(), tblags, 1)

zmr_test = almonZmatrix(df.mortgage_rate.loc[201502:201605].as_matrix(), mrlags, 1)

z_test = pd.DataFrame( np.concatenate( (ztb_test, zmr_test), axis=1), test_index, columns )

X_macro_test = sm.add_constant( z_test )

pred_lnrp = macro_fit.predict( X_macro_test )

pred_p = np.exp(pred_lnrp) * test_cpi
adjust = pd.DataFrame( pred_p/test_p.price_doc, columns=["adjustment"] )

adjust
combo = test_data.join(adjust, on='yearmonth')

combo['adjusted'] = combo.price_doc * combo.adjustment

adjxgb_df = pd.DataFrame()

adjxgb_df['id'] = combo.id

adjxgb_df['price_doc'] = combo.adjusted

adjxgb_df.head()
adjxgb_df.to_csv('adjusted_xgb_predicitons.csv', index=False)