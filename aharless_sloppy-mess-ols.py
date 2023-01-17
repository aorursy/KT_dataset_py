import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv')

macro = pd.read_csv('../input/macro.csv')

test = pd.read_csv('../input/test.csv')
dfa = pd.concat([train, test])

dfa.loc[:,"sub_area"] = dfa.sub_area.str.replace(" ","").str.replace("\'","").str.replace("-","")

dfa = dfa.merge(macro, on='timestamp', suffixes=['','_macro'])
dfa.lnfloor = np.log(dfa.floor+1)

dfa.build_year.isnull().mean()

dfa.build_year[dfa.build_year<4965].max()

dfa.num_room.isnull().mean()

print( train.product_type.isnull().mean() )

print( test.product_type.isnull().mean() )
var = (dfa.product_type=='OwnerOccupier')

var[dfa.product_type.isnull()] = var.mean()

print( var.isnull().mean() )

print( var.unique() )

var.value_counts()
dfa.full_all.head()

dfa.sub_area.head()

b = dfa[dfa.sub_area=="Bibirevo"]

b.green_part_3000.unique()

# b.product_type.unique()
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

dfa["lnmax"] = np.log(dfa.floor+1)



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





# Perhaps drop full_sq, life_sq, floor, max_floor, num_rooms, kitch_sq
dfa["material0"] = dfa.material.isnull()

dfa["material1"] = (dfa.material==1)

dfa["material2"] = (dfa.material==2)

dfa["material3"] = (dfa.material==3)

dfa["material4"] = (dfa.material==4)

dfa["material5"] = (dfa.material==5)

dfa["material6"] = (dfa.material==6)



# "state" isn't explained but it looks like an ordinal number, so for now keep numberic

dfa.loc[dfa.build_year>5,"build_year"] = np.NaN  # Value 33 seems to be invalid; others all 1-4

dfa.state = dfa.state.fillna(dfa.state.median())



# product_type gonna be ugly because there are missing values in the test set but not training

# Check for the same problem with other variables

dfa["owner_occ"] = (dfa.product_type=='OwnerOccupier')

dfa.owner_occ.fillna(dfa.owner_occ.mean())



dfa = pd.get_dummies(dfa, dummy_na=True, columns=['sub_area'], drop_first=True)



# Perhaps drop material, product_type

# Perhaps drop all neighborhood variables
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

dfa["lnsince"] = dfa.sincebuild.apply( np.max, args=[0] ).add(1).apply(np.log)



# Perhaps drop build_year, sincebuild
print('build0' in dfa.columns.values)

dfa.build0.head()

dfa.material.unique()
# Want to check for valididty of relationships, e.g. kitch_sq < life_sq < full_sq

# But this interacts with how variables are already processed, so that may have to be changed

# For example, if kitch_sq is sometimes huge and there is a dummy to identify those huge cases,

#  do we want a separate dummy to identify which of those cases are internally consistent?
training = dfa[dfa.price_doc.notnull()]

training.lnrp = training.price_doc.div(training.cpi).apply(np.log)

y = training.lnrp

X = training[["state", "owner_occ",

              "fullzero", "fulltiny", "fullhuge", "lnfull", 

              "nolife", "lifezero", "lifetiny", "lifehuge", "lnlife",

              "nofloor", "floor1", "floor0", "floorhuge", "lnfloor",

              "nomax", "max1", "max0", "maxhuge", "lnmax", 

              "norooms", "zerorooms", "lnrooms",

              "nokitch", "kitch1", "kitch0", "kitchhuge", "lnkitch",

#              "material1", "material2", "material3", "material4", "material5", "material6",

              "nobuild", "futurebuild", "newhouse", "tooold", 

              "build0", "build1", "untilbuild", "lnsince",

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

       'sub_area_PoselenieFilimonkovskoe', 'sub_area_PoselenieKievskij',

       'sub_area_PoselenieKlenovskoe', 'sub_area_PoselenieKokoshkino',

       'sub_area_PoselenieKrasnopahorskoe',

       'sub_area_PoselenieMarushkinskoe',

       'sub_area_PoselenieMihajlovoJarcevskoe',

       'sub_area_PoselenieMoskovskij', 'sub_area_PoselenieMosrentgen',

       'sub_area_PoselenieNovofedorovskoe',

       'sub_area_PoseleniePervomajskoe', 'sub_area_PoselenieRjazanovskoe',

       'sub_area_PoselenieRogovskoe', 'sub_area_PoselenieShhapovskoe',

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

       'sub_area_ZapadnoeDegunino', 'sub_area_Zjablikovo',

       'sub_area_Zjuzino', 'sub_area_nan'

             ]]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.pipeline import make_pipeline



# Make a pipeline that transforms X

pipe = make_pipeline(Imputer(), StandardScaler())

pipe.fit(X_train)

pipe.transform(X_train)
from sklearn.metrics import make_scorer



def rmsle_exp(y_true_log, y_pred_log):

    y_true = np.exp(y_true_log)

    y_pred = np.exp(y_pred_log)

    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))



def score_model(model, pipe):

    train_error = rmsle_exp(y_train, model.predict(pipe.transform(X_train)))

    test_error = rmsle_exp(y_test, model.predict(pipe.transform(X_test)))

    return train_error, test_error
from sklearn.linear_model import LinearRegression



lr = LinearRegression(fit_intercept=True)

lr.fit(pipe.transform(X_train), y_train)



print("Train error: {:.4f}, Test error: {:.4f}".format(*score_model(lr, pipe)))
pipe.fit(X)

pipe.transform(X)
lr.fit(pipe.transform(X), y)
testing = dfa[dfa.price_doc.isnull()]
df_test = pd.DataFrame(columns=X.columns)

for column in df_test.columns:

        df_test[column] = testing[column]

        
# Make the predictions

pi = pipe.transform(df_test)

pred = lr.predict(pi)

predictions = np.exp(pred)*testing.cpi



# And put this in a dataframe

predictions_df = pd.DataFrame()

predictions_df['id'] = testing['id']

predictions_df['price_doc'] = predictions

predictions_df.head()

 
predictions_df.to_csv('dumb_ols_predictions.csv', index=False)
np.std(pipe.transform(X)[:,15])

print( X.iloc[:,15].mean() )

print( X.iloc[:,15].std() )

X.columns.values[15]
co = lr.coef_

ra = range(len(co))

mask = np.abs(co)>1e9

ra*mask

X.columns[mask].values
fo = "y ~ "

i = 0

for col in X.columns.values:

    fo += col

    i += 1

    if (i<len(X.columns.values)):

        fo += "+"

fo
import statsmodels.formula.api as sm

dfols = X.copy()

dfols["y"] = y 

# How did y get to be a column of X?

# And now it's gone again. WTF?

result = sm.ols(formula=fo, data=dfols).fit()

#dfo = X.drop('y',axis=1)

#sm.ols(y, X).fit()

#result.summary()
result.summary()
"y" in X.columns
mask = (X.isnull().sum().values>0)

X.columns[mask]
print( X.shape )

print( y.shape )

print( y.isnull().sum() )

print( np.sum(X.isnull().sum().values) )
X.columns.values