# Parameters

use_pipe = True

weight_base = "2010-07-01"  # Used for the initial analysis, but later I try other values
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv')

macro = pd.read_csv('../input/macro.csv')

test = pd.read_csv('../input/test.csv')
dfa = pd.concat([train, test])  # "dfa" stands for "data frame all"

# Eliminate spaces and special characters in area names

dfa.loc[:,"sub_area"] = dfa.sub_area.str.replace(" ","").str.replace("\'","").str.replace("-","")

dfa = dfa.merge(macro, on='timestamp', suffixes=['','_macro'])
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
# Note for later:

# Want to check for valididty of relationships, e.g. kitch_sq < life_sq < full_sq

# But this interacts with how variables are already processed, so that may have to be changed

# For example, if kitch_sq is sometimes huge and there is a dummy to identify those huge cases,

#  do we want a separate dummy to identify which of those cases are internally consistent?
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

       'sub_area_Alekseevskoe', 'sub_area_Altufevskoe', 'sub_area_Arbat',

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

       'sub_area_Krylatskoe', 'sub_area_Kuncevo', 'sub_area_Kurkino',

       'sub_area_Kuzminki', 'sub_area_Lefortovo', 'sub_area_Levoberezhnoe',

       'sub_area_Lianozovo', 'sub_area_Ljublino', 'sub_area_Lomonosovskoe',

       'sub_area_Losinoostrovskoe', 'sub_area_Marfino',

       'sub_area_MarinaRoshha', 'sub_area_Marino', 'sub_area_Matushkino',

       'sub_area_Meshhanskoe', 'sub_area_Metrogorodok', 'sub_area_Mitino',

       'sub_area_Molzhaninovskoe', 'sub_area_MoskvorecheSaburovo',

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

        # This one is almost empty.  Will lump in with another category.

        #'sub_area_PoselenieKievskij',

        # This one is almost empty.  Will lump in with another category.

        #'sub_area_PoselenieKlenovskoe', 

       'sub_area_PoselenieKokoshkino',

       'sub_area_PoselenieKrasnopahorskoe',

       'sub_area_PoselenieMarushkinskoe',

        # This one is almost empty.  Will lump in with another category.

        #'sub_area_PoselenieMihajlovoJarcevskoe',

       'sub_area_PoselenieMoskovskij', 'sub_area_PoselenieMosrentgen',

       'sub_area_PoselenieNovofedorovskoe',

       'sub_area_PoseleniePervomajskoe', 'sub_area_PoselenieRjazanovskoe',

       'sub_area_PoselenieRogovskoe', 

        # This one is almost empty.  Will lump in with another category.

        #'sub_area_PoselenieShhapovskoe',

       'sub_area_PoselenieShherbinka', 'sub_area_PoselenieSosenskoe',

       'sub_area_PoselenieVnukovskoe', 'sub_area_PoselenieVoronovskoe',

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

       'sub_area_Tverskoe', 'sub_area_Veshnjaki', 'sub_area_Vnukovo',

       'sub_area_Vojkovskoe', 'sub_area_Vostochnoe',

       'sub_area_VostochnoeDegunino', 'sub_area_VostochnoeIzmajlovo',

       'sub_area_VyhinoZhulebino', 'sub_area_Zamoskvoreche',

       'sub_area_ZapadnoeDegunino', 'sub_area_Zjablikovo', 'sub_area_Zjuzino',   

       ]





# Lump together small sub_areas

dfa = dfa.assign( sub_area_PoselenieSmall =

   dfa.sub_area_PoselenieMihajlovoJarcevskoe +

   dfa.sub_area_PoselenieKievskij +

   dfa.sub_area_PoselenieKlenovskoe +

   dfa.sub_area_PoselenieShhapovskoe )



# For now eliminate case with ridiculous value of full_sq

dfa = dfa[~dfa.fullhuge]



    

# Independent features



indievars = ["owner_occ", "state", "state_Xowner", "lnfull_Xowner", "mortgage_growth"]





# Complete list of features to use for fit



allvars = fullvars + floorvars + maxvars + kitchvars + buildvars + matervars

allvars += floorXvars + kitchXvars + subarvars + indievars



# The normalized target variable:  log real sale price

training = dfa[dfa.price_doc.notnull()]

training.lnrp = training.price_doc.div(training.cpi).apply(np.log)

y = training.lnrp



# Features to use later in heteroskedasticity model

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
if use_pipe:

    from sklearn.preprocessing import Imputer, StandardScaler

    from sklearn.pipeline import make_pipeline



    # Make a pipeline that transforms X

    pipe = make_pipeline(Imputer(), StandardScaler())

    pipe.fit(X)

    pipe.transform(X)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import BaggingRegressor

import random



random.seed = 100

lr = LinearRegression(fit_intercept=True)

br = BaggingRegressor(lr)

if use_pipe:

    br.fit(pipe.transform(X), y, sample_weight=wts)

else:

    br.fit(X, y, sample_weight=wts)
# Look for collinearity problems

for e in br.estimators_:

    co = e.coef_

    mask = np.abs(co)>1e4

    print( X.columns[mask].values )
# Predict on the training set and take the residuals

if use_pipe:

    pred = br.predict( pipe.transform(X) )

else:

    pred = br.predict(X)

resids = y - pred

resids2 = resids * resids
# Add heteroskedasticity-related features

Xhetero = X.assign(million1=million1, million2=million2, million3=million3, lnrp=y)
lnres2 = np.log(resids2)

plt.hist(resids2, bins=20)

plt.show()

plt.hist(lnres2, bins=20)

plt.show()
# For a visual look at the results of a rough model, use statsmodels.WLS

from statsmodels.regression.linear_model import WLS

xdat = Xhetero.copy().astype(np.float64)

xdat["constant"] = 1

ydat = lnres2.copy().astype(np.float64)

result = WLS(ydat, xdat, weights=wts).fit()

result.summary()
from statsmodels.regression.linear_model import OLS

result = OLS(ydat, xdat).fit()

result.summary()
from sklearn.linear_model import Lasso

la_hetero = Lasso(alpha=5e-2)

if use_pipe:

    newpipe = make_pipeline(Imputer(), StandardScaler())

    newpipe.fit(Xhetero)

    la_hetero.fit(newpipe.transform(Xhetero), lnres2)

    pred_res2 = np.exp( la_hetero.predict( newpipe.transform(Xhetero) ) )

else:

    la_hetero.fit(Xhetero, lnres2)

    pred_res2 = np.exp( la_hetero.predict(Xhetero) )

print( np.min(pred_res2) )

print( np.max(pred_res2) )

pd.DataFrame(Xhetero.columns, la_hetero.coef_)[np.abs(la_hetero.coef_)>1e-5]
la_hetero = Lasso(alpha=1e-1)

if use_pipe:

    la_hetero.fit(newpipe.transform(Xhetero), lnres2)

    pred_res2 = np.exp( la_hetero.predict( newpipe.transform(Xhetero) ) )

else:

    la_hetero.fit(Xhetero, lnres2)

    pred_res2 = np.exp( la_hetero.predict(Xhetero) )

print( np.min(pred_res2) )

print( np.max(pred_res2) )

pd.DataFrame(Xhetero.columns, la_hetero.coef_)[np.abs(la_hetero.coef_)>1e-5]
la_hetero = Lasso(alpha=2e-1)

if use_pipe:

    la_hetero.fit(newpipe.transform(Xhetero), lnres2)

    pred_res2 = np.exp( la_hetero.predict( newpipe.transform(Xhetero) ) )

else:

    la_hetero.fit(Xhetero, lnres2)

    pred_res2 = np.exp( la_hetero.predict(Xhetero) )

print( np.min(pred_res2) )

print( np.max(pred_res2) )

pd.DataFrame(Xhetero.columns, la_hetero.coef_)[np.abs(la_hetero.coef_)>1e-5]
la_hetero = Lasso(alpha=4e-1)

if use_pipe:

    la_hetero.fit(newpipe.transform(Xhetero), lnres2)

    pred_res2 = np.exp( la_hetero.predict( newpipe.transform(Xhetero) ) )

else:

    la_hetero.fit(Xhetero, lnres2)

    pred_res2 = np.exp( la_hetero.predict(Xhetero) )

print( np.min(pred_res2) )

print( np.max(pred_res2) )

pd.DataFrame(Xhetero.columns, la_hetero.coef_)[np.abs(la_hetero.coef_)>1e-5]
ls_hetero = LinearRegression(fit_intercept=True)

xh = Xhetero[["lnfloor_Xowner", "sub_area_Nekrasovka", "owner_occ",

              "state_Xowner", "million1", "million2", "million3"]]

if use_pipe:

    newerpipe = make_pipeline(Imputer(), StandardScaler())

    newerpipe.fit(xh)

    ls_hetero.fit(newerpipe.transform(xh), lnres2)

    pred_res2 = np.exp( ls_hetero.predict( newerpipe.transform(xh) ) )

else:

    ls_hetero.fit(Xhetero, lnres2)

    pred_res2 = np.exp( ls_hetero.predict(Xhetero) )

print( np.min(pred_res2) )

print( np.max(pred_res2) )

print( np.mean(pred_res2) )

print( np.std(resids2) )

print( np.min(resids2) )

print( np.max(resids2) )

print( np.mean(resids2) )

print( np.std(resids2) )



pd.DataFrame(xh.columns, ls_hetero.coef_)[np.abs(ls_hetero.coef_)>1e-5]
xdat = xh.copy().astype(np.float64)

xdat["constant"] = 1

ydat = lnres2.copy().astype(np.float64)

result = WLS(ydat, xdat, weights=wts).fit()

result.summary()
random.seed = 200

xin = xh

lr = LinearRegression(fit_intercept=True)

br = BaggingRegressor(lr, max_samples=0.2)

br.fit(xin, lnres2, sample_weight=wts)

pd.DataFrame(data=[e.coef_ for e in br.estimators_], columns=xin.columns)
sums = [sum(s) for s in br.estimators_samples_]

print( sums )

print (np.mean(sums))
print( xin.shape[0] )

br
sum(br.estimators_samples_[0]*br.estimators_samples_[1])
br.estimators_samples_
xin_top = xin[0:15235]

r_top = lnres2[0:15235]

xin_bot = xin[15236:30470]

r_bot = lnres2[15236:30470]

xdat = xin_top.copy().astype(np.float64)

xdat["constant"]=1

rdat = r_top.copy().astype(np.float64)

print( OLS(rdat, xdat).fit().summary() )

xdat = xin_bot.copy().astype(np.float64)

xdat["constant"]=1

rdat = r_bot.copy().astype(np.float64)

print( OLS(rdat, xdat).fit().summary() )
xh = Xhetero[["sub_area_Nekrasovka", "owner_occ",

              "million1", "million2", "million3"]]

xdat = xh.copy().astype(np.float64)

xdat["constant"] = 1

ydat = lnres2.copy().astype(np.float64)

result = WLS(ydat, xdat, weights=wts).fit()

result.summary()
random.seed = 200

xin = xh

lr = LinearRegression(fit_intercept=True)

br = BaggingRegressor(lr, max_samples=0.5)

br.fit(xin, lnres2, sample_weight=wts)

pd.DataFrame(data=[e.coef_ for e in br.estimators_], columns=xin.columns)
pred_res2 = np.exp( br.predict(xin) )

print()

print( np.min(pred_res2) )

print( np.min(resids2) )

print()

print( np.max(pred_res2) )

print( np.max(resids2) )

print()

print( np.mean(pred_res2) )

print( np.mean(resids2) )

print()

print( np.std(pred_res2) )

print( np.std(resids2) )
predvals = pd.Series(pred_res2).value_counts().sort_index()

predvals
print( predvals.index.values[3]/predvals.index.values[0] )

print( predvals.index.values[8]/predvals.index.values[3] )
from sklearn.metrics import make_scorer



def rmsle_exp(y_true_log, y_pred_log):

    y_true = np.exp(y_true_log)

    y_pred = np.exp(y_pred_log)

    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))



def score_model(model, pipe=None):

    if (pipe==None):

        train_error = rmsle_exp(y_train, model.predict(X_train))

        test_error = rmsle_exp(y_test, model.predict(X_test))

    else:

        train_error = rmsle_exp(y_train, model.predict(pipe.transform(X_train)))

        test_error = rmsle_exp(y_test, model.predict(pipe.transform(X_test)))

    return train_error, test_error
len(X)
split = int(.75*len(X))

X_train = X[0:split]

X_test = X[split:len(X)]

y_train = y[0:split]

y_test = y[split:len(y)]

wts_train = wts[0:split]
from sklearn.linear_model import Lasso

newpipe = make_pipeline(Imputer(), StandardScaler())

newpipe.fit(Xhetero)



minval = 100.

where_min = ["Never",0.,0.]

results = pd.DataFrame()

alphas = [5e-1,2e-1,1e-1,5e-2,2e-2,1e-2,5e-3,2e-3]

# DELETE NEXT LINE TO RUN FULL VERSION

alphas = [5e-1,5e-2,2e-3]

for alpha in alphas:

    print( "alpha=", alpha)

    la_hetero = Lasso(alpha=alpha)

    la_hetero.fit(newpipe.transform(Xhetero), lnres2)

    pred_res2 = np.exp( la_hetero.predict( newpipe.transform(Xhetero) ) )

    la_hetero.fit(Xhetero, lnres2)

    pred_res2 = np.exp( la_hetero.predict(Xhetero) )

    wbvalues = ["2010-01-01", "2010-04-01", "2010-07-01","2010-10-01",

                "2011-01-01", "2011-04-01", "2011-07-01"]

# DELETE NEXT LINE TO RUN FULL VERSION

    wbvalues = ["2010-01-01", "2011-01-01", "2011-07-01"]

    for wbase in wbvalues:

        print( "    wbase=", wbase)

        basedate = pd.to_datetime(wbase).toordinal() # Basedate gets a weight of zero

        wtd = pd.to_datetime(training.timestamp).apply(lambda x: x.toordinal()) - basedate

        wts = np.array(wtd)/1e3 # The denominator here shouldn't matter, just gives nice numbers.

        row = []

        wdvalues = [0,1,2,4,8,16,32,64,128,256,512]

# DELETE NEXT LINE TO RUN FULL VERSION

        wdvalues = [1,32,512]

        for waterdown in wdvalues:

            pred_wd = pred_res2 + waterdown

            wts_train = (wts * (pred_wd)) [0:split]

            lr.fit(X_train, y_train, sample_weight=wts_train)

            test_error = rmsle_exp(y_test, lr.predict(X_test))

            if test_error < minval:

                minval = test_error

                where_min = [wbase, alpha, waterdown]

            row = row + [test_error]

        index = pd.MultiIndex.from_tuples([(wbase,alpha)], names=['wbase', 'alpha'])

        dfrow = pd.DataFrame( index=index, data=[row], columns=wdvalues)

        results = results.append( dfrow )

print( where_min )

print( minval )

results
minval = 100.

where_min = ["Never",0.,0.]

results = pd.DataFrame()

alphas = [2e-1,1e-1,5e-2,2e-2,1e-2]

# DELETE NEXT LINE TO RUN FULL VERSION

alphas = [2e-1,1e-2]

for alpha in alphas:

    print( "alpha=", alpha)

    la_hetero = Lasso(alpha=alpha)

    la_hetero.fit(newpipe.transform(Xhetero), lnres2)

    pred_res2 = np.exp( la_hetero.predict( newpipe.transform(Xhetero) ) )

    la_hetero.fit(Xhetero, lnres2)

    pred_res2 = np.exp( la_hetero.predict(Xhetero) )

    wbvalues = ["2011-06-01", "2011-06-15", "2011-07-01","2010-07-15",

                "2011-08-01", "2011-08-15"]

# DELETE NEXT LINE TO RUN FULL VERSION

    wbvalues = ["2011-06-15", "2011-08-15"]

    for wbase in wbvalues:

        print( "    wbase=", wbase)

        basedate = pd.to_datetime(wbase).toordinal() # Basedate gets a weight of zero

        wtd = pd.to_datetime(training.timestamp).apply(lambda x: x.toordinal()) - basedate

        wts = np.array(wtd)/1e3 # The denominator here shouldn't matter, just gives nice numbers.

        row = []

        wdvalues = [0,2,8,32,128,512]

# DELETE NEXT LINE TO RUN FULL VERSION

        wdvalues = [8,512]

        for waterdown in wdvalues:

            pred_wd = pred_res2 + waterdown

            wts_train = (wts * (pred_wd)) [0:split]

            lr.fit(X_train, y_train, sample_weight=wts_train)

            test_error = rmsle_exp(y_test, lr.predict(X_test))

            if test_error < minval:

                minval = test_error

                where_min = [wbase, alpha, waterdown]

            row = row + [test_error]

        index = pd.MultiIndex.from_tuples([(wbase,alpha)], names=['wbase', 'alpha'])

        dfrow = pd.DataFrame( index=index, data=[row], columns=wdvalues)

        results = results.append( dfrow )

print( where_min )

print( minval )

results            
weight_base = "2010-08-19"

wts = get_weights(training)

newwts = wts

from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=True)

if use_pipe:

    lr.fit(pipe.transform(X), y, sample_weight=newwts)

else:

    lr.fit(X, y, sample_weight=newwts)
testing = dfa[dfa.price_doc.isnull()]
df_test = pd.DataFrame(columns=X.columns)

for column in df_test.columns:

        df_test[column] = testing[column]        
# Make the predictions

if use_pipe:

    pred = lr.predict( pipe.transform(df_test) )

else:

    pred = lr.predict(df_test)

predictions = np.exp(pred)*testing.cpi



# And put this in a dataframe

predictions_df = pd.DataFrame()

predictions_df['id'] = testing['id']

predictions_df['price_doc'] = predictions

predictions_df.head()



predictions_df.to_csv('wls_predictions.csv', index=False)
# Check for ridiculous coefficients, likely indicating collinearity

co = lr.coef_

ra = range(len(co))

mask = np.abs(co)>1e4

X.columns[mask].values

from statsmodels.regression.linear_model import WLS

xdat = X.copy().astype(np.float64)

xdat["constant"] = 1

ydat = y.copy().astype(np.float64)

result = WLS(ydat, xdat, weights=newwts).fit()

result.summary()
# Note that, if the model is run without the pipe transform, the coefficients below

#  should be the same as those above.  Sometimes they are, sometimes not.

#  If they're not the same, probably numerical instability due to collinearity.

pd.DataFrame(X.columns, co)
train.sub_area.value_counts().tail(20)
print(train.sub_area.sort_values().unique()[1:170])