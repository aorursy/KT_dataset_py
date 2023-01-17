# Parameters

use_pipe = True  # Standardize. (Shouldn't matter for OLS, but there are compuatation issues.)

weight_base = "2011-08-19"  # Linear weights start from zero
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
# Interaction terms (many not used, ultimately, but I haven't whittled it down yet).

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
# Just a tiny bit of feature engineering:  (log) price of oil in rubles

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

        # Aggregate with neighboring districts

        #'sub_area_Alekseevskoe', 

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

        # Aggregate with small neighbor

        #'sub_area_Kurkino',

       'sub_area_Kuzminki', 'sub_area_Lefortovo', 'sub_area_Levoberezhnoe',

       'sub_area_Lianozovo', 'sub_area_Ljublino', 'sub_area_Lomonosovskoe',

       'sub_area_Losinoostrovskoe', 'sub_area_Marfino',

       'sub_area_MarinaRoshha', 'sub_area_Marino', 'sub_area_Matushkino',

       'sub_area_Meshhanskoe', 'sub_area_Metrogorodok', 'sub_area_Mitino',

        # Aggregate with neighboring districts

        #'sub_area_Molzhaninovskoe', 

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

        # This one is almost empty.  Will lump in with another category.

        #'sub_area_PoselenieKievskij',

        # This one is almost empty.  Will lump in with another category.

        #'sub_area_PoselenieKlenovskoe', 

        # Aggregate with neighboring districts

        #'sub_area_PoselenieKokoshkino',

       'sub_area_PoselenieKrasnopahorskoe',

        # Aggregate with neighboring districts

        #'sub_area_PoselenieMarushkinskoe',

        # This one is almost empty.  Will lump in with another category.

        #'sub_area_PoselenieMihajlovoJarcevskoe',

       'sub_area_PoselenieMoskovskij', 'sub_area_PoselenieMosrentgen',

       'sub_area_PoselenieNovofedorovskoe',

       'sub_area_PoseleniePervomajskoe', 'sub_area_PoselenieRjazanovskoe',

       'sub_area_PoselenieRogovskoe', 

        # This one is almost empty.  Will lump in with another category.

        #'sub_area_PoselenieShhapovskoe',

       'sub_area_PoselenieShherbinka', 'sub_area_PoselenieSosenskoe',

       'sub_area_PoselenieVnukovskoe',  

        # Aggregate with neighboring districts

        #'sub_area_PoselenieVoronovskoe',

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

        # Aggregate with neighboring districts

        #'sub_area_Vnukovo',

       'sub_area_Vojkovskoe', 

        # Aggregate with neighboring districts

        #'sub_area_Vostochnoe',

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



indievars = ["owner_occ", "state", "state_Xowner",

             # Dropping due to "visiual regularizaiton" and unclear relationship to fullv

             #"lnfull_Xowner",

             #"lnruboil",

             "mortgage_growth" ]

          



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
if use_pipe:

    from sklearn.preprocessing import Imputer, StandardScaler

    from sklearn.pipeline import make_pipeline



    # Make a pipeline that transforms X

    pipe = make_pipeline(Imputer(), StandardScaler())

    pipe.fit(X)

    pipe.transform(X)


from sklearn.linear_model import LinearRegression



lr = LinearRegression(fit_intercept=True)

if use_pipe:

    lr.fit(pipe.transform(X), y, sample_weight=wts)

else:

    lr.fit(X, y, sample_weight=wts)



# At home I have version 0.18, in which LinearRegression knows its sum of squared residuals

# WTF, Scikit-learn developers, why did you deprecate this??

# I needed it to check that my code is working



# lr.residues_  # Show SSR to check it will be same as verison with composite features
# Hey, Scikit-learn developers, I want my residues_ back!

# Dang ML types think you can just use statistical methods as a tool

#   and ignore the actual statistics.  That bad science IMO, even bad data science.

import sklearn

sklearn.__version__
# Function to create an indicator array that selects positions

#   corresponding to a set of variables from the regression



def get_selector( df, varnames ):

    selector = np.zeros( df.shape[1] )

    selector[[df.columns.get_loc(x) for x in varnames]] = 1

    return( selector )
# Function to calculate a composite feature and append it to the data frame



def append_composite( df, varnames, name, X, Xuse, estimator ):

    selector = get_selector(X, varnames)

    v = pd.Series( np.matmul( Xuse, selector*estimator.coef_ ), 

                   name=name, index=df.index )

    return( pd.concat( [df, v], axis=1 ) )
# Generate composite features for groups of input variables using WLS coefficeints



if use_pipe:

    Xuse = pipe.transform(X)

else:

    Xuse = X



vars = {"fullv":fullvars,     "floorv":floorvars,   "maxv":maxvars, 

        "kitchv":kitchvars,   "buildv":buildvars,   "materv":matervars, 

        "floorxv":floorXvars, "kitchxv":kitchXvars, "subarv":subarvars}

for v in vars:

    training = append_composite( training, vars[v], v, X, Xuse, lr )



shortvarlist = list(vars.keys())

shortvarlist += indievars



Xshort = training[shortvarlist]



if use_pipe:

    pipe1 = make_pipeline(Imputer(), StandardScaler())

    pipe1.fit(Xshort)

    pipe1.transform(Xshort)
# Fit again to make sure result is same

lr1 = LinearRegression(fit_intercept=True)

if use_pipe:

    lr1.fit(pipe1.transform(Xshort), y, sample_weight=wts)

else:

    lr1.fit(Xshort, y, sample_weight=wts)



# Sorry, can't do that in version 0.19,

#   apparently unless you do the extra step of predicting on the training data

# Never mind, I've already debugged this.



# lr1.residues_
testing = dfa[dfa.price_doc.isnull()]
df_test_full = pd.DataFrame(columns=X.columns)

for column in df_test_full.columns:

        df_test_full[column] = testing[column]        

if use_pipe:

    Xuse = pipe.transform(df_test_full)

else:

    Xuse = df_test_full



for v in vars:

    df_test_full = append_composite( df_test_full, vars[v], v, X, Xuse, lr )
df_test = pd.DataFrame(columns=Xshort.columns)

for column in df_test.columns:

        df_test[column] = df_test_full[column]        
# Make the predictions

if use_pipe:

    pred = lr1.predict( pipe1.transform(df_test) )

else:

    pred = lr1.predict(df_test)

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

xdat = Xshort.copy().astype(np.float64)

xdat["constant"] = 1

ydat = y.copy().astype(np.float64)

result = WLS(ydat, xdat, weights=wts).fit()

result.summary()
# Note that, if the model is run without the pipe transform, the coefficients below

#  should be the same as those above.  Sometimes they have been, sometimes not.

#  If they're not the same, probably numerical instability due to collinearity.

#  In any case, if the pipe transform is used they are different, because

#  they apply to standardized variables, not raw data.

pd.DataFrame(Xshort.columns, lr1.coef_)
# Function to add another features to the training set for XGBoost



def append_series( X_train, X_test, train_input, test_input, sername ):

    vtrain = pd.Series( train_input[sername], name=sername, index=X_train.index )

    X_train_out = pd.concat( [X_train, vtrain], axis=1 )

    vtest = pd.Series( test_input[sername], name=sername, index=X_test.index )

    X_test_out = pd.concat( [X_test, vtest], axis=1 )

    return( X_train_out, X_test_out )
wts *= (1 - .2*million1 + .1*million2 + .05*million3)
vars_to_add = [

    "kindergarten_km", 

    "railroad_km", 

    "swim_pool_km", 

#    "fitness_km",

#    "workplaces_km",

#    "radiation_km",

    "public_transport_station_km",

    "big_road1_km",

    "big_road2_km",

#    "university_km",

#    "big_church_km",

#    "park_km",

#    "power_transmission_line_km",

#    "green_zone_km",

#    "public_healthcare_km",

#    "additional_education_km",

#    "catering_km",

    "church_synagogue_km",

#    "school_km",

#    "theater_km",

#    "water_km",

#    "stadium_km",

#    "nuclear_reactor_km",

#    "lnlife",

#    "lnrooms",

#    "hospice_morgue_km",

    "ttk_km",

#    "metro_min_avto",

#    "metro_km_avto",

    "metro_min_walk",

#    "metro_km_walk",

#    "cemetery_km",

#    "incineration_km",

#    "railroad_station_walk_min",

#    "railroad_station_walk_km",

#    "ID_railroad_station_walk",

#    "railroad_station_avto_km",

#    "railroad_station_avto_min",

#    "ID_railroad_station_avto",

#    "public_transport_station_min_walk",

#    "water_1line", # PROBLEM WITH DATA TYPE

#    "mkad_km",

#    "sadovoe_km"

#    "bulvar_ring_km"

    "kremlin_km",

#    "ID_big_road1",

#    "big_road1_1line", # PROBLEM WITH DATA TYPE

#    "ID_big_road2",

#    "railroad_1line", # PROBLEM WITH DATA TYPE

#    "zd_vokzaly_avto_km",

#    "ID_railroad_terminal",

#    "bus_terminal_avto_km",

#    "ID_bus_terminal",

#    "oil_chemistry_km",

#    "thermal_power_plant_km",

#    "ts_km".

#    "big_market_km",

#    "market_shop_km",

#    "ice_rink_km",

#    "basketball_km",

#    "detention_facility_km",

#    "shopping_centers_km",

#    "office_km",

#    "preschool_km",

    "mosque_km",

#    "museum_km",

#    "exhibition_km",

#    "cafe_count_500_price_high",

#    "cafe_count_1000_price_high",

#    "cafe_count_1000_price_4000",

#    "cafe_count_500_price_4000",

#    "cafe_avg_price_1500",

#    "cafe_avg_price_3000",

#    "raion_popul",

#    "oil_urals",

#    "gdp_annual",

#    "mortgage_value",

    "rent_price_3room_eco",

#    "gdp_quart_growth",

    "mortgage_rate",

    "lnruboil"

]

Xdata_train = Xshort

Xdata_test = df_test

print( Xdata_train.shape )

print( Xdata_test.shape )

for v in vars_to_add:

    Xdata_train, Xdata_test = append_series( Xdata_train, Xdata_test, training, testing, v )

print( Xdata_train.shape )

print( Xdata_test.shape )
import xgboost as xgb

xgb_params = {

    'eta': 0.05,   # Try .01 or .005, but for now...

    'max_depth': 6,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(Xdata_train, y, weight=wts)

dtest = xgb.DMatrix(Xdata_test)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=2000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)

cv_output["test-rmse-mean"][len(cv_output)-1]
num_boost_rounds = len(cv_output)

print( num_boost_rounds )

model = xgb.train(xgb_params, dtrain, num_boost_round= num_boost_rounds)
%matplotlib inline

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