# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt



sns.set(font_scale=2.2)

plt.style.use('seaborn')



from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder

from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit

from sklearn.metrics import f1_score

import itertools

import lightgbm as lgb

from xgboost import XGBClassifier

import shap 

from tqdm import tqdm

import featuretools as ft

import warnings

warnings.filterwarnings('ignore')

import time

df_train = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/train.csv')

df_test = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/test.csv')
print('df_train shape:', df_train.shape, '  ', 'df_test shape: ', df_test.shape)
df_train.head()
df_test.head()
description = [

("v2a1"," Monthly rent payment"),

("hacdor"," =1 Overcrowding by bedrooms"),

("rooms","  number of all rooms in the house"),

("hacapo"," =1 Overcrowding by rooms"),

("v14a"," =1 has toilet in the household"),

("refrig"," =1 if the household has refrigerator"),

("v18q"," owns a tablet"),

("v18q1"," number of tablets household owns"),

("r4h1"," Males younger than 12 years of age"),

("r4h2"," Males 12 years of age and older"),

("r4h3"," Total males in the household"),

("r4m1"," Females younger than 12 years of age"),

("r4m2"," Females 12 years of age and older"),

("r4m3"," Total females in the household"),

("r4t1"," persons younger than 12 years of age"),

("r4t2"," persons 12 years of age and older"),

("r4t3"," Total persons in the household"),

("tamhog"," size of the household"),

("tamviv"," number of persons living in the household"),

("escolari"," years of schooling"),

("rez_esc"," Years behind in school"),

("hhsize"," household size"),

("paredblolad"," =1 if predominant material on the outside wall is block or brick"),

("paredzocalo"," =1 if predominant material on the outside wall is socket (wood, zinc or absbesto"),

("paredpreb"," =1 if predominant material on the outside wall is prefabricated or cement"),

("pareddes"," =1 if predominant material on the outside wall is waste material"),

("paredmad"," =1 if predominant material on the outside wall is wood"),

("paredzinc"," =1 if predominant material on the outside wall is zink"),

("paredfibras"," =1 if predominant material on the outside wall is natural fibers"),

("paredother"," =1 if predominant material on the outside wall is other"),

("pisomoscer"," =1 if predominant material on the floor is mosaic ceramic   terrazo"),

("pisocemento"," =1 if predominant material on the floor is cement"),

("pisoother"," =1 if predominant material on the floor is other"),

("pisonatur"," =1 if predominant material on the floor is  natural material"),

("pisonotiene"," =1 if no floor at the household"),

("pisomadera"," =1 if predominant material on the floor is wood"),

("techozinc"," =1 if predominant material on the roof is metal foil or zink"),

("techoentrepiso"," =1 if predominant material on the roof is fiber cement,   mezzanine "),

("techocane"," =1 if predominant material on the roof is natural fibers"),

("techootro"," =1 if predominant material on the roof is other"),

("cielorazo"," =1 if the house has ceiling"),

("abastaguadentro"," =1 if water provision inside the dwelling"),

("abastaguafuera"," =1 if water provision outside the dwelling"),

("abastaguano"," =1 if no water provision"),

("public"," =1 electricity from CNFL,  ICE, ESPH/JASEC"),

("planpri"," =1 electricity from private plant"),

("noelec"," =1 no electricity in the dwelling"),

("coopele"," =1 electricity from cooperative"),

("sanitario1"," =1 no toilet in the dwelling"),

("sanitario2"," =1 toilet connected to sewer or cesspool"),

("sanitario3"," =1 toilet connected to  septic tank"),

("sanitario5"," =1 toilet connected to black hole or letrine"),

("sanitario6"," =1 toilet connected to other system"),

("energcocinar1"," =1 no main source of energy used for cooking (no kitchen)"),

("energcocinar2"," =1 main source of energy used for cooking electricity"),

("energcocinar3"," =1 main source of energy used for cooking gas"),

("energcocinar4"," =1 main source of energy used for cooking wood charcoal"),

("elimbasu1"," =1 if rubbish disposal mainly by tanker truck"),

("elimbasu2"," =1 if rubbish disposal mainly by botan hollow or buried"),

("elimbasu3"," =1 if rubbish disposal mainly by burning"),

("elimbasu4"," =1 if rubbish disposal mainly by throwing in an unoccupied space"),

("elimbasu5"," =1 if rubbish disposal mainly by throwing in river,   creek or sea"),

("elimbasu6"," =1 if rubbish disposal mainly other"),

("epared1"," =1 if walls are bad"),

("epared2"," =1 if walls are regular"),

("epared3"," =1 if walls are good"),

("etecho1"," =1 if roof are bad"),

("etecho2"," =1 if roof are regular"),

("etecho3"," =1 if roof are good"),

("eviv1"," =1 if floor are bad"),

("eviv2"," =1 if floor are regular"),

("eviv3"," =1 if floor are good"),

("dis"," =1 if disable person"),

("male"," =1 if male"),

("female"," =1 if female"),

("estadocivil1"," =1 if less than 10 years old"),

("estadocivil2"," =1 if free or coupled uunion"),

("estadocivil3"," =1 if married"),

("estadocivil4"," =1 if divorced"),

("estadocivil5"," =1 if separated"),

("estadocivil6"," =1 if widow/er"),

("estadocivil7"," =1 if single"),

("parentesco1"," =1 if household head"),

("parentesco2"," =1 if spouse/partner"),

("parentesco3"," =1 if son/doughter"),

("parentesco4"," =1 if stepson/doughter"),

("parentesco5"," =1 if son/doughter in law"),

("parentesco6"," =1 if grandson/doughter"),

("parentesco7"," =1 if mother/father"),

("parentesco8"," =1 if father/mother in law"),

("parentesco9"," =1 if brother/sister"),

("parentesco10"," =1 if brother/sister in law"),

("parentesco11"," =1 if other family member"),

("parentesco12"," =1 if other non family member"),

("idhogar"," Household level identifier"),

("hogar_nin"," Number of children 0 to 19 in household"),

("hogar_adul"," Number of adults in household"),

("hogar_mayor"," # of individuals 65+ in the household"),

("hogar_total"," # of total individuals in the household"),

("dependency"," Dependency rate"),

("edjefe"," years of education of male head of household"),

("edjefa"," years of education of female head of household"),

("meaneduc","average years of education for adults (18+)"),

("instlevel1"," =1 no level of education"),

("instlevel2"," =1 incomplete primary"),

("instlevel3"," =1 complete primary"),

("instlevel4"," =1 incomplete academic secondary level"),

("instlevel5"," =1 complete academic secondary level"),

("instlevel6"," =1 incomplete technical secondary level"),

("instlevel7"," =1 complete technical secondary level"),

("instlevel8"," =1 undergraduate and higher education"),

("instlevel9"," =1 postgraduate higher education"),

("bedrooms"," number of bedrooms"),

("overcrowding"," # persons per room"),

("tipovivi1"," =1 own and fully paid house"),

("tipovivi2"," =1 own,   paying in installments"),

("tipovivi3"," =1 rented"),

("tipovivi4"," =1 precarious"),

("tipovivi5"," =1 other(assigned"),

("computer"," =1 if the household has notebook or desktop computer,   borrowed)"),

("television"," =1 if the household has TV"),

("mobilephone"," =1 if mobile phone"),

("qmobilephone"," # of mobile phones"),

("lugar1"," =1 region Central"),

("lugar2"," =1 region Chorotega"),

("lugar3"," =1 region PacÃƒÂ­fico central"),

("lugar4"," =1 region Brunca"),

("lugar5"," =1 region Huetar AtlÃƒÂ¡ntica"),

("lugar6"," =1 region Huetar Norte"),

("area1"," =1 zona urbana"),

("area2"," =2 zona rural"),

("age"," Age in years"),

("SQBescolari"," escolari squared"),

("SQBage"," age squared"),

("SQBhogar_total"," hogar_total squared"),

("SQBedjefe"," edjefe squared"),

("SQBhogar_nin"," hogar_nin squared"),

("SQBovercrowding"," overcrowding squared"),

("SQBdependency"," dependency squared"),

("SQBmeaned"," meaned squared"),

("agesq"," Age squared"),]



description = pd.DataFrame(description, columns=['varname', 'description'])
description
total = df_train.isnull().sum().sort_values(ascending=False)

print(total)

percent = 100 * (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)

missing_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



missing_df.head(20)
df_train['edjefa'].head(20)
#df_train.drop(['edjef'], axis=1)

#df_test.drop(['edjef'], axis=1)
# edjefa if education is "yes" and person is head of household <== fill with escolari

# 

df_train.loc[(df_train['edjefa'] == "yes") & (df_train['parentesco1'] == 1), "edjefa"] = df_train.loc[(df_train['edjefa'] == "yes") & (df_train['parentesco1'] == 1), "escolari"]

df_train.loc[(df_train['edjefe'] == "yes") & (df_train['parentesco1'] == 1), "edjefe"] = df_train.loc[(df_train['edjefe'] == "yes") & (df_train['parentesco1'] == 1), "escolari"]



df_test.loc[(df_test['edjefa'] == "yes") & (df_test['parentesco1'] == 1), "edjefa"] = df_test.loc[(df_test['edjefa'] == "yes") & (df_test['parentesco1'] == 1), "escolari"]

df_test.loc[(df_test['edjefe'] == "yes") & (df_test['parentesco1'] == 1), "edjefe"] = df_test.loc[(df_test['edjefe'] == "yes") & (df_test['parentesco1'] == 1), "escolari"]



# this field is supposed to be interaction between gender and escolari, 

# but it isn't clear what "yes" means, let's fill it with 4

df_train.loc[df_train['edjefa'] == "yes", "edjefa"] = 4

df_train.loc[df_train['edjefe'] == "yes", "edjefe"] = 4



df_test.loc[df_test['edjefa'] == "yes", "edjefa"] = 4

df_test.loc[df_test['edjefe'] == "yes", "edjefe"] = 4



# create feature with max education of either head of household

#("edjefe"," years of education of male head of household"),

#("edjefa"," years of education of female head of household"),

df_train['edjef'] = df_train[['edjefa','edjefe']].max(axis=1) # axis=1, max of every row

df_test['edjef'] = df_test[['edjefa','edjefe']].max(axis=1)



# "v14a"," =1 has toilet in the household"),

# "sanitario1"," =1 no toilet in the dwelling"), sanitario=위생적인

# "abastaguano"," =1 if no water provision"),

# fix some inconsistencies in the data - some rows indicate both that the household does and does not have a toilet, 

# if there is no water we'll assume they do not

df_train.loc[(df_train.v14a ==  1) & (df_train.sanitario1 ==  1) & (df_train.abastaguano == 0), "v14a"] = 0

df_train.loc[(df_train.v14a ==  1) & (df_train.sanitario1 ==  1) & (df_train.abastaguano == 0), "sanitario1"] = 0



df_test.loc[(df_test.v14a ==  1) & (df_test.sanitario1 ==  1) & (df_test.abastaguano == 0), "v14a"] = 0

df_test.loc[(df_test.v14a ==  1) & (df_test.sanitario1 ==  1) & (df_test.abastaguano == 0), "sanitario1"] = 0
df_train[['edjefa','edjefe']]
df_train['rez_esc']
df_train['rez_esc'].fillna(0, inplace=True)

df_test['rez_esc'].fillna(0, inplace=True)
df_train['rez_esc'].head(20)
# SQBmeaned : Found 5 NaN, and fill 0

df_train['SQBmeaned'].fillna(0, inplace=True)

df_test['SQBmeaned'].fillna(0, inplace=True)
# number of count the every value

#

df_train['v18q'].value_counts()
df_train.loc[df_train['v18q'] == 1, 'v18q1'].value_counts()
df_train.loc[df_train['v18q'] == 0, 'v18q1'].value_counts()
df_train['v18q1'].fillna(0, inplace=True)

df_test['v18q1'].fillna(0, inplace=True)
df_train['v18q1'].head(20)
# ("tipovivi3"," =1 rented"),

df_train['tipovivi3'].value_counts()
sns.kdeplot(df_train.loc[df_train['tipovivi3'] == 1, 'v2a1'], label = 'Monthly rent payment of household(rented=1)')

sns.kdeplot(df_train.loc[df_train['tipovivi3'] == 0, 'v2a1'], label = 'Monthly rent payment of household(rented=0)')

plt.xscale('log')

plt.show()
df_train['v2a1'].fillna(0, inplace=True)

df_test['v2a1'].fillna(0, inplace=True)
sns.kdeplot(df_train.loc[df_train['tipovivi3'] == 1, 'v2a1'], label = 'Monthly rent payment of household(rented=1)')

sns.kdeplot(df_train.loc[df_train['tipovivi3'] == 0, 'v2a1'], label = 'Monthly rent payment of household(rented=0)')

plt.xscale('log')

plt.show()
total = df_train.isnull().sum().sort_values(ascending=False)

percent = 100 * (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)

missing_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



missing_df.head(20)
total = df_test.isnull().sum().sort_values(ascending=False)

percent = 100 * (df_test.isnull().sum() / df_test.isnull().count()).sort_values(ascending=False)

missing_df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



missing_df.head(20)
# object type check ( mixed types )

#

features_object = [col for col in df_train.columns if df_train[col].dtype == 'object']
features_object
# some dependencies are Na, fill those with "the square root" of the square ( Na = NaN = Null 모두 같은 의미 )

df_train['dependency'] = np.sqrt(df_train['SQBdependency'])

df_test['dependency'] = np.sqrt(df_test['SQBdependency'])
df_train['dependency']
# df_train['dependency'] = df_train['dependency'].replace({np.inf: 0})

# df_test['dependency'] = df_test['dependency'].replace({np.inf: 0})



# def replace_dependency(x):

#     if x == 'yes':

#         return 10

#     elif x == 'no':

#         return 0

#     else:

#         return x



# df_train['dependency'] = df_train['dependency'].apply(replace_dependency).astype(float)

# df_test['dependency'] = df_test['dependency'].apply(replace_dependency).astype(float)



# - As you can see, setting yes -> 10 and no -> 0 is good choice.

# - At first, fill inf value with 0.
df_train['edjefe'].head(100)
def replace_edjefe(x):

    if x == 'yes':

        return 1

    elif x == 'no':

        return 0

    else:

        return x

    

df_train['edjefe'] = df_train['edjefe'].apply(replace_edjefe).astype(float)

df_test['edjefe'] = df_test['edjefe'].apply(replace_edjefe).astype(float)
df_train['edjefe'].head(100)
def replace_edjefa(x):

    if x == 'yes':

        return 1

    elif x == 'no':

        return 0

    else:

        return x

    

df_train['edjefa'] = df_train['edjefa'].apply(replace_edjefa).astype(float)

df_test['edjefa'] = df_test['edjefa'].apply(replace_edjefa).astype(float)
# create feature with max education of either head of household

df_train['edjef'] = np.max(df_train[['edjefa', 'edjefe']], axis=1)

df_test['edjef'] = np.max(df_test[['edjefa', 'edjefe']], axis=1)
df_train['edjef']
df_train.shape
# new feature

df_train['roof_waste_material'] = np.nan

df_test['roof_waste_material'] = np.nan

df_train['electricity_other'] = np.nan

df_test['electricity_other'] = np.nan



def fill_roof_exception(x):

    if ( x['techozinc'] == 0 ) and (x['techoentrepiso'] == 0) and (x['techocane'] == 0) and (x['techootro'] == 0):

        return 1

    else:

        return 0

    

def fill_no_electricity(x):

    if (x['public'] == 0) and (x['planpri'] == 0) and (x['noelec'] == 0) and (x['coopele'] == 0):

        return 1

    else:

        return 0

    

df_train['roof_waste_material'] = df_train.apply(lambda x : fill_roof_exception(x), axis=1)

df_test['roof_waste_material'] = df_test.apply(lambda x : fill_roof_exception(x), axis=1)

df_train['electricity_other'] = df_train.apply(lambda x : fill_no_electricity(x), axis=1)

df_test['electricity_other'] = df_test.apply(lambda x : fill_no_electricity(x), axis=1)
binary_cat_features = [col for col in df_train.columns if df_train[col].value_counts().shape[0] == 2]
binary_cat_features
continuous_features = [col for col in df_train.columns if col not in binary_cat_features]

continuous_features = [col for col in continuous_features if col not in features_object]

continuous_features = [col for col in continuous_features if col not in ['Id', 'Target', 'idhogar']]
print('There are {} continuous features'.format(len(continuous_features)))

for col in continuous_features:

    print('{}: {}'.format(col, description.loc[description['varname'] == col, 'description'].values))
df_train['tamhog'].value_counts()
df_train.drop('tamhog', axis=1, inplace=True)

df_test.drop('tamhog', axis=1, inplace=True)
# ("hogar_adul"," Number of adults in household"),

# ("hogar_mayor"," # of individuals 65+ in the household"),

df_train['adult'] = df_train['hogar_adul'] - df_train['hogar_mayor']



# ("hogar_nin"," Number of children 0 to 19 in household"),

# ("hogar_mayor"," # of individuals 65+ in the household"),

df_train['dependency_count'] = df_train['hogar_nin'] + df_train['hogar_mayor']



# ("dependency"," Dependency rate"),

df_train['dependency'] = df_train['dependency_count'] / df_train['adult']



# ("hogar_nin"," Number of children 0 to 19 in household"),

# ("hogar_total"," # of total individuals in the household"),

df_train['child_percent'] = df_train['hogar_nin'] / df_train['hogar_total']



# ("hogar_mayor"," # of individuals 65+ in the household"),

# ("hogar_total"," # of total individuals in the household"),

df_train['elder_percent'] = df_train['hogar_mayor'] / df_train['hogar_total']



# ("hogar_adul"," Number of adults in household"),

# ("hogar_total"," # of total individuals in the household"),

df_train['adult_percent'] = df_train['hogar_adul'] / df_train['hogar_total']



# ("r4h2"," Males 12 years of age and older"),

# ("hogar_total"," # of total individuals in the household"),

df_train['males_older_12_years_percent'] = df_train['r4h2'] / df_train['hogar_total']



# ("r4h3"," Total males in the household"),

# ("hogar_total"," # of total individuals in the household"),

df_train['males_percent'] = df_train['r4h3'] / df_train['hogar_total']



# ("r4m1"," Females younger than 12 years of age"),

# ("hogar_total"," # of total individuals in the household"),

df_train['females_older_12_years_percent'] = df_train['r4m2'] / df_train['hogar_total']



# ("r4m3"," Total females in the household"),

# ("hogar_total"," # of total individuals in the household"),

df_train['females_percent'] = df_train['r4m3'] / df_train['hogar_total']



# ("r4t1"," persons younger than 12 years of age"),

# ("hogar_total"," # of total individuals in the household"),

df_train['persons_younger_12_years_percent'] = df_train['r4t1'] / df_train['hogar_total']



# ("r4t2"," persons 12 years of age and older"),

# ("hogar_total"," # of total individuals in the household"),

df_train['persons_older_12_years_percent'] = df_train['r4t2'] / df_train['hogar_total']



# ("r4t3"," Total persons in the household"),

# ("hogar_total"," # of total individuals in the household"),

df_train['persons_percent'] = df_train['r4t3'] / df_train['hogar_total']
df_test['adult'] = df_test['hogar_adul'] - df_test['hogar_mayor']

df_test['dependency_count'] = df_test['hogar_nin'] + df_test['hogar_mayor']

df_test['dependency'] = df_test['dependency_count'] / df_test['adult']

df_test['child_percent'] = df_test['hogar_nin'] / df_test['hogar_total']

df_test['elder_percent'] = df_test['hogar_mayor'] / df_test['hogar_total']

df_test['adult_percent'] = df_test['hogar_adul'] / df_test['hogar_total']

df_test['males_younger_12_years_percent'] = df_test['r4h1'] / df_test['hogar_total']

df_test['males_older_12_years_percent'] = df_test['r4h2'] / df_test['hogar_total']

df_test['males_percent'] = df_test['r4h3'] / df_test['hogar_total']

df_test['females_younger_12_years_percent'] = df_test['r4m1'] / df_test['hogar_total']

df_test['females_older_12_years_percent'] = df_test['r4m2'] / df_test['hogar_total']

df_test['females_percent'] = df_test['r4m3'] / df_test['hogar_total']

df_test['persons_younger_12_years_percent'] = df_test['r4t1'] / df_test['hogar_total']

df_test['persons_older_12_years_percent'] = df_test['r4t2'] / df_test['hogar_total']

df_test['persons_percent'] = df_test['r4t3'] / df_test['hogar_total']
# ("r4h1"," Males younger than 12 years of age"),

# ("hhsize"," household size"),

df_train['males_younger_12_years_in_household_size'] = df_train['r4h1'] / df_train['hhsize']



# ("r4h2"," Males 12 years of age and older"),

# ("hhsize"," household size"),

df_train['males_older_12_years_in_household_size'] = df_train['r4h2'] / df_train['hhsize']



# ("r4h3"," Total males in the household"),

# ("hhsize"," household size"),

df_train['males_in_household_size'] = df_train['r4h3'] / df_train['hhsize']



# ("r4m1"," Females younger than 12 years of age"),

# ("hhsize"," household size"),

df_train['females_younger_12_years_in_household_size'] = df_train['r4m1'] / df_train['hhsize']



# ("r4m2"," Females 12 years of age and older"),

# ("hhsize"," household size"),

df_train['females_older_12_years_in_household_size'] = df_train['r4m2'] / df_train['hhsize']



# ("r4m3"," Total females in the household"),

# ("hogar_total"," # of total individuals in the household"),

df_train['females_in_household_size'] = df_train['r4m3'] / df_train['hogar_total']



# ("r4t1"," persons younger than 12 years of age"),

# ("hhsize"," household size"),

df_train['persons_younger_12_years_in_household_size'] = df_train['r4t1'] / df_train['hhsize']



# ("r4t2"," persons 12 years of age and older"),

# ("hhsize"," household size"),

df_train['persons_older_12_years_in_household_size'] = df_train['r4t2'] / df_train['hhsize']



# ("r4t3"," Total persons in the household"),

# ("hhsize"," household size"),

df_train['persons_in_household_size'] = df_train['r4t3'] / df_train['hhsize']
df_test['males_younger_12_years_in_household_size'] = df_test['r4h1'] / df_test['hhsize']

df_test['males_older_12_years_in_household_size'] = df_test['r4h2'] / df_test['hhsize']

df_test['males_in_household_size'] = df_test['r4h3'] / df_test['hhsize']

df_test['females_younger_12_years_in_household_size'] = df_test['r4m1'] / df_test['hhsize']

df_test['females_older_12_years_in_household_size'] = df_test['r4m2'] / df_test['hhsize']

df_test['females_in_household_size'] = df_test['r4m3'] / df_test['hogar_total']

df_test['persons_younger_12_years_in_household_size'] = df_test['r4t1'] / df_test['hhsize']

df_test['persons_older_12_years_in_household_size'] = df_test['r4t2'] / df_test['hhsize']

df_test['persons_in_household_size'] = df_test['r4t3'] / df_test['hhsize']

# ("hacdor"," =1 Overcrowding by bedrooms"),

# ("hacapo"," =1 Overcrowding by rooms"),

df_train['overcrowding_room_and_bedroom'] = ( df_train['hacdor'] + df_train['hacapo'] ) / 2

df_test['overcrowding_room_and_bedroom'] = ( df_test['hacdor'] + df_test['hacapo'] ) / 2
# ("escolari"," years of schooling"),

# ("age"," Age in years"),

df_train['escolari_age'] = df_train['escolari'] / df_train['age']

df_test['escolari_age'] = df_test['escolari'] / df_test['age']



# ("hogar_nin"," Number of children 0 to 19 in household"),

# ("r4t1"," persons younger than 12 years of age"),

df_train['age_12_19'] = df_train['hogar_nin'] - df_train['r4t1']

df_test['age_12_19'] = df_test['hogar_nin'] - df_test['r4t1']
# ("qmobilephone"," # of mobile phones"),

# ("tamviv"," number of persons living in the household"),

df_train['phones-per-capita'] = df_train['qmobilephone'] / df_train['tamviv']



# ("v18q1"," number of tablets household owns"),

# ("tamviv"," number of persons living in the household"),

df_train['tablets-per-capita'] = df_train['v18q1'] / df_train['tamviv']



# ("rooms","  number of all rooms in the house"),

# ("tamviv"," number of persons living in the household"),

df_train['rooms-per-capita'] = df_train['rooms'] / df_train['tamviv']



# ("v2a1"," Monthly rent payment"),

# ("tamviv"," number of persons living in the household"),

df_train['rent-per-capita'] = df_train['v2a1'] / df_train['tamviv']

df_test['phones-per-capita'] = df_test['qmobilephone'] / df_test['tamviv']

df_test['tablets-per-capita'] = df_test['v18q1'] / df_test['tamviv']

df_test['rooms-per-capita'] = df_test['rooms'] / df_test['tamviv']

df_test['rent-per-capita'] = df_test['v2a1'] / df_test['tamviv']
df_train.shape

( df_train['hogar_total'] == df_train['r4t3'] ).sum()
# ("v2a1"," Monthly rent payment"),



family_size_features = ['adult', 'hogar_adul', 'hogar_mayor', 'hogar_nin', 'hogar_total', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3', 'hhsize']

new_feats = []

for col in family_size_features:

    new_col_name = 'new_{}_per_{}'.format('v2a1', col)

    new_feats.append(new_col_name)

    df_train[new_col_name] = df_train['v2a1'] / df_train[col]

    df_test[new_col_name] = df_test['v2a1'] / df_train[col]
new_feats
for col in new_feats:

    df_train[col].replace([np.inf], np.nan, inplace=True)

    df_train[col].fillna(0, inplace=True)

    

    df_test[col].replace([np.inf], np.nan, inplace=True)

    df_test[col].fillna(0, inplace=True)    
# ("rooms","  number of all rooms in the house"),

new_feats = []

for col in family_size_features:

    new_col_name = 'new_{}_{}'.format('rooms', col)

    new_feats.append(new_col_name)

    df_train[new_col_name] = df_train['rooms'] / df_train[col]

    df_test[new_col_name] = df_test['rooms'] / df_test[col]

    

for col in new_feats:

    df_train[col].replace([np.inf], np.nan, inplace=True)

    df_train[col].fillna(0, inplace=True)

    

    df_test[col].replace([np.inf], np.nan, inplace=True)

    df_test[col].fillna(0, inplace=True)
# ("bedrooms"," number of bedrooms"),

new_feats = []

for col in family_size_features:

    new_col_name = 'new_{}_per_{}'.format('bedrooms', col)

    new_feats.append(new_col_name)

    df_train[new_col_name] = df_train['bedrooms'] / df_train[col]

    df_test[new_col_name] = df_test['bedrooms'] / df_test[col]

    

for col in new_feats:

    df_train[col].replace([np.inf], np.nan, inplace=True)

    df_train[col].fillna(0, inplace=True)

    

    df_test[col].replace([np.inf], np.nan, inplace=True)

    df_test[col].fillna(0, inplace=True)



# number of features, target is in train

print(df_train.shape, df_test.shape)
# ("v18q1"," number of tablets household owns"),



new_feats = []

for col in family_size_features:

    new_col_name = 'new_{}_per_{}'.format('v18q1', col)

    new_feats.append(new_col_name)

    df_train[new_col_name] = df_train['v18q1'] / df_train[col]

    df_test[new_col_name] = df_test['v18q1'] / df_test[col]

    

for col in new_feats:

    df_train[col].replace([np.inf], np.nan, inplace=True)

    df_train[col].fillna(0, inplace=True)

    

    df_test[col].replace([np.inf], np.nan, inplace=True)

    df_test[col].fillna(0, inplace=True)
# ("qmobilephone"," # of mobile phones"),

new_feats = []

for col in family_size_features:

    new_col_name = 'new_{}_per_{}'.format('qmobilephone', col)

    new_feats.append(new_col_name)

    df_train[new_col_name] = df_train['qmobilephone'] / df_train[col]

    df_test[new_col_name] = df_test['qmobilephone'] / df_test[col]



for col in new_feats:

    df_train[col].replace([np.inf], np.nan, inplace=True)

    df_train[col].fillna(0, inplace=True)

    

    df_test[col].replace([np.inf], np.nan, inplace=True)

    df_test[col].fillna(0, inplace=True)
# ("rez_esc"," Years behind in school"),

new_feats = []

for col in family_size_features:

    new_col_name = 'new_{}_per_{}'.format('rez_esc', col)

    new_feats.append(new_col_name)

    df_train[new_col_name] = df_train['rez_esc'] / df_train[col]

    df_test[new_col_name] = df_test['rez_esc'] / df_test[col]



for col in new_feats:

    df_train[col].replace([np.inf], np.nan, inplace=True)

    df_train[col].fillna(0, inplace=True)

    

    df_test[col].replace([np.inf], np.nan, inplace=True)

    df_test[col].fillna(0, inplace=True)
# ("age"," Age in years"),

# ("escolari"," years of schooling"),

df_train['rez_esc_age'] = df_train['rez_esc'] / df_train['age']

df_train['rez_esc_escolari'] = df_train['rez_esc'] / df_train['escolari']



df_test['rez_esc_age'] = df_test['rez_esc'] / df_test['age']

df_test['rez_esc_escolari'] = df_test['rez_esc'] / df_test['escolari']

# ("v18q1"," number of tablets household owns"),

# ("qmobilephone"," # of mobile phones"),

df_train['tabulet_x_qmobilephone'] = df_train['v18q1'] * df_train['qmobilephone']

df_test['tabulet_x_qmobilephone'] = df_test['v18q1'] * df_test['qmobilephone']
("epared1"," =1 if walls are bad"),

("epared2"," =1 if walls are regular"),

("epared3"," =1 if walls are good"),



("etecho1"," =1 if roof are bad"),

("etecho2"," =1 if roof are regular"),

("etecho3"," =1 if roof are good"),



# wall and roof

for col1 in ['epared1', 'epared2', 'epared3']:

    for col2 in ['etecho1', 'etecho2', 'etecho3']:

        new_col_name = 'new_{}_x_{}'.format(col1, col2)

        df_train[new_col_name] = df_train[col1] * df_train[col2]

        df_test[new_col_name] = df_test[col1] * df_test[col2]



# ("eviv1"," =1 if floor are bad"),

# ("eviv2"," =1 if floor are regular"),

# ("eviv3"," =1 if floor are good"),



# wall and floor

for col1 in ['epared1', 'epared2', 'epared3']:

    for col2 in ['eviv1', 'eviv2', 'eviv3']:

        new_col_name = 'new_{}_x_{}'.format(col1, col2)

        df_train[new_col_name] = df_train[col1] * df_train[col2]

        df_test[new_col_name] = df_test[col1] * df_test[col2]





# roof and floor

for col1 in ['etecho1', 'etecho2', 'etecho3']:

    for col2 in ['eviv1', 'eviv2', 'eviv3']:

        new_col_name = 'new_{}_x_{}'.format(col1, col2)

        df_train[new_col_name] = df_train[col1] * df_train[col2]

        df_test[new_col_name] = df_test[col1] * df_test[col2]
for col1 in ['epared1', 'epared2', 'epared3']:

    for col2 in ['etecho1', 'etecho2', 'etecho3']:

        for col3 in ['eviv1', 'eviv2', 'eviv3']:

            new_col_name = 'new_{}_x_{}_x_{}'.format(col1, col2, col3)

            df_train[new_col_name] = df_train[col1] * df_train[col2] * df_train[col3]

            df_test[new_col_name] = df_test[col1] * df_test[col2] * df_train[col3]
print(df_train.shape, df_test.shape)
# ("public"," =1 electricity from CNFL,  ICE, ESPH/JASEC"),

# ("planpri"," =1 electricity from private plant"),

# ("noelec"," =1 no electricity in the dwelling"),

# ("coopele"," =1 electricity from cooperative"),



# ("energcocinar1"," =1 no main source of energy used for cooking (no kitchen)"),

# ("energcocinar2"," =1 main source of energy used for cooking electricity"),

# ("energcocinar3"," =1 main source of energy used for cooking gas"),

# ("energcocinar4"," =1 main source of energy used for cooking wood charcoal"),



for col1 in ['public', 'planpri', 'noelec', 'coopele']:

    for col2 in ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4']:

        new_col_name = 'new_{}_x_{}'.format(col1, col2)

        df_train[new_col_name] = df_train[col1] * df_train[col2]

        df_test[new_col_name] = df_test[col1] * df_test[col2]
# ("sanitario1"," =1 no toilet in the dwelling"),

# ("sanitario2"," =1 toilet connected to sewer or cesspool"),

# ("sanitario3"," =1 toilet connected to  septic tank"),

# ("sanitario5"," =1 toilet connected to black hole or letrine"),

# ("sanitario6"," =1 toilet connected to other system"),



# ("elimbasu1"," =1 if rubbish disposal mainly by tanker truck"),

# ("elimbasu2"," =1 if rubbish disposal mainly by botan hollow or buried"),

# ("elimbasu3"," =1 if rubbish disposal mainly by burning"),

# ("elimbasu4"," =1 if rubbish disposal mainly by throwing in an unoccupied space"),

# ("elimbasu5"," =1 if rubbish disposal mainly by throwing in river,   creek or sea"),

# ("elimbasu6"," =1 if rubbish disposal mainly other"),



for col1 in ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']:

    for col2 in ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']:

        new_col_name = 'new_{}_x_{}'.format(col1, col2)

        df_train[new_col_name] = df_train[col1] * df_train[col2]

        df_test[new_col_name] = df_test[col1] * df_test[col2]
# ("abastaguadentro"," =1 if water provision inside the dwelling"),

# ("abastaguafuera"," =1 if water provision outside the dwelling"),

# ("abastaguano"," =1 if no water provision"),



# ("sanitario1"," =1 no toilet in the dwelling"),

# ("sanitario2"," =1 toilet connected to sewer or cesspool"), 하수구

# ("sanitario3"," =1 toilet connected to  septic tank"), 정화조

# ("sanitario5"," =1 toilet connected to black hole or letrine"), 변소

# ("sanitario6"," =1 toilet connected to other system"),





for col1 in ['abastaguadentro', 'abastaguafuera', 'abastaguano']:

    for col2 in ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']:

        new_col_name = 'new_{}_x_{}'.format(col1, col2)

        df_train[new_col_name] = df_train[col1] * df_train[col2]

        df_test[new_col_name] = df_test[col1] * df_test[col2]
print(df_train.shape, df_test.shape)
# ("area1"," =1 zona urbana"),

# ("area2"," =2 zona rural"),



# ("instlevel1"," =1 no level of education"),

# ("instlevel2"," =1 incomplete primary"),

# ("instlevel3"," =1 complete primary"),

# ("instlevel4"," =1 incomplete academic secondary level"),

# ("instlevel5"," =1 complete academic secondary level"),

# ("instlevel6"," =1 incomplete technical secondary level"),

# ("instlevel7"," =1 complete technical secondary level"),

# ("instlevel8"," =1 undergraduate and higher education"),

# ("instlevel9"," =1 postgraduate higher education"),



for col1 in ['area1', 'area2']:

    for col2 in ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']:

        new_col_name = 'new_{}_x_{}'.format(col1, col2)

        df_train[new_col_name] = df_train[col1] * df_train[col2]

        df_test[new_col_name] = df_test[col1] * df_test[col2]
# ("lugar1"," =1 region Central"),

# ("lugar2"," =1 region Chorotega"),

# ("lugar3"," =1 region PacÃƒÂ­fico central"),

# ("lugar4"," =1 region Brunca"),

# ("lugar5"," =1 region Huetar AtlÃƒÂ¡ntica"),

# ("lugar6"," =1 region Huetar Norte"),



for col1 in ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']:

    for col2 in ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']:

        new_col_name = 'new_{}_x_{}'.format(col1, col2)

        df_train[new_col_name] = df_train[col1] * df_train[col2]

        df_test[new_col_name] = df_test[col1] * df_test[col2]

print(df_train.shape, df_test.shape)
# ("computer"," =1 if the household has notebook or desktop computer,   borrowed)"),

# ("television"," =1 if the household has TV"),

# ("mobilephone"," =1 if mobile phone"),

# ("v18q"," owns a tablet"),

# ("refrig"," =1 if the household has refrigerator"),



df_train['electronics'] = df_train['computer'] * df_train['mobilephone'] * df_train['television'] * df_train['v18q'] * df_train['refrig']

df_test['electronics'] = df_test['computer'] * df_test['mobilephone'] * df_test['television'] * df_test['v18q'] * df_test['refrig']



df_train['no_appliances'] = df_train['refrig'] + df_train['computer'] + df_train['television'] + df_train['mobilephone']

df_test['no_appliances'] = df_test['refrig'] + df_test['computer'] + df_test['television'] + df_test['mobilephone']
# ("paredblolad"," =1 if predominant material(주 재료) on the outside wall is block or brick"), 블록/벽돌

# ("paredzocalo"," =1 if predominant material on the outside wall is socket (wood, zinc or absbesto"), 나무, 아연, 

# ("paredpreb"," =1 if predominant material on the outside wall is prefabricated or cement"), 조립식(?) 시멘트

# ("pareddes"," =1 if predominant material on the outside wall is waste material"), 폐기물

# ("paredmad"," =1 if predominant material on the outside wall is wood"),

# ("paredzinc"," =1 if predominant material on the outside wall is zink"),

# ("paredfibras"," =1 if predominant material on the outside wall is natural fibers"), 천연 섬유

# ("paredother"," =1 if predominant material on the outside wall is other"),



# ("pisomoscer"," =1 if predominant material on the floor is mosaic ceramic   terrazo"),

# ("pisocemento"," =1 if predominant material on the floor is cement"),

# ("pisoother"," =1 if predominant material on the floor is other"),

# ("pisonatur"," =1 if predominant material on the floor is  natural material"),

# ("pisonotiene"," =1 if no floor at the household"),

# ("pisomadera"," =1 if predominant material on the floor is wood"),





for col1 in ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother']:

    for col2 in ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']:

        new_col_name = 'new_{}_x_{}'.format(col1, col2)

        df_train[new_col_name] = df_train[col1] * df_train[col2]

        df_test[new_col_name] = df_test[col1] * df_test[col2]



# ("techozinc"," =1 if predominant material on the roof is metal foil or zink"), 금속 호일, 나무

# ("techoentrepiso"," =1 if predominant material on the roof is fiber cement,   mezzanine "), 섬유 시멘트(?),  메자닌(?)

# ("techocane"," =1 if predominant material on the roof is natural fibers"),

# ("techootro"," =1 if predominant material on the roof is other"),



for col1 in ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']:

    for col1 in ['techozinc', 'techoentrepiso', 'techocane', 'techootro']:

        new_col_name = 'new_{}_x_{}'.format(col1, col2)

        df_train[new_col_name] = df_train[col1] * df_train[col2]

        df_test[new_col_name] = df_test[col1] * df_test[col2]

        

for col1 in ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother']:

    for col2 in ['techozinc', 'techoentrepiso', 'techocane', 'techootro']:

        new_col_name = 'new_{}_x_{}'.format(col1, col2)

        df_train[new_col_name] = df_train[col1] * df_train[col2]

        df_test[new_col_name] = df_test[col1] * df_test[col2]        

        

for col1 in ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother']:

    for col2 in ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']:

        for col3 in ['techozinc', 'techoentrepiso', 'techocane', 'techootro']:

            new_col_name = 'new_{}_x_{}_x_{}'.format(col1, col2, col3)

            df_train[new_col_name] = df_train[col1] * df_train[col2] * df_train[col3]

            df_test[new_col_name] = df_test[col1] * df_test[col2] * df_train[col3]

print(df_train.shape, df_test.shape)
cols_with_only_one_value = []

for col in df_train.columns:

    if col == 'Target':

        continue

    if df_train[col].value_counts().shape[0] == 1 or df_test[col].value_counts().shape[0] == 1:

        print(col)

        cols_with_only_one_value.append(col)

df_train.drop(cols_with_only_one_value, axis=1, inplace=True)

df_test.drop(cols_with_only_one_value, axis=1, inplace=True)

df_train.shape
cols_train = np.array(sorted([col for col in df_train.columns if col != 'Target']))

cols_test = np.array(sorted(df_test.columns))
(cols_train == cols_test).sum() == len(cols_train)



# ??? ValueError: shape mismatch: objects cannot be broadcast to a single shape
cols_train.shape
cols_test.shape
def max_min(x):

    return x.max() - x.min()
# ("idhogar"," Household level identifier"),

# tqdm : for 문의 상태바를 나타내줌.

# family_size_features = ['adult', 'hogar_adul', 'hogar_mayor', 'hogar_nin', 'hogar_total', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3', 'hhsize']



agg_train = pd.DataFrame()

agg_test = pd.DataFrame()



for item in tqdm(family_size_features):

    for i, function in enumerate(['mean','std','min','max','sum', 'count', max_min]):

        group_train = df_train[item].groupby(df_train['idhogar']).agg(function)

        group_test = df_test[item].groupby(df_test['idhogar']).agg(function)

        if i == 6:

            new_col = item + '_new_' + 'max_min'

        else:

            new_col = item + '_new_' + function

        agg_train[new_col] = group_train

        agg_test[new_col] = group_test



print('new aggregate train set has {} rows, and {} features'.format(agg_train.shape[0], agg_train.shape[1]))

print('new aggregate test set has {} rows, and {} features'.format(agg_test.shape[0], agg_test.shape[1]))
agg_train.shape
aggr_list = ['rez_esc', 'dis', 'male', 'female', 

                  'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 

                  'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 

                  'parentesco11', 'parentesco12',

                  'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9',

                 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'refrig', 'television', 'mobilephone',

            'area1', 'area2', 'v18q', 'edjef']







for item in tqdm(aggr_list):

    for function in ['count', 'sum']:

        group_train = df_train[item].groupby(df_train['idhogar']).agg(function)

        group_test = df_test[item].groupby(df_test['idhogar']).agg(function)

        new_col = item + '_new1_' + function

        agg_train[new_col] = group_train

        agg_test[new_col] = group_test

print('new aggregate train set has {} rows, and {} features'.format(agg_train.shape[0], agg_train.shape[1]))

print('new aggregate test set has {} rows, and {} features'.format(agg_test.shape[0], agg_test.shape[1]))

aggr_list = ['escolari', 'age', 'escolari_age', 'dependency', 'bedrooms', 'overcrowding', 'rooms', 'qmobilephone', 'v18q1']



for item in tqdm(aggr_list):

    for function in ['mean','std','min','max','sum', 'count', max_min]:

        group_train = df_train[item].groupby(df_train['idhogar']).agg(function)

        group_test = df_test[item].groupby(df_test['idhogar']).agg(function)

        if i == 6:

            new_col = item + '_new2_' + 'max_min'

        else:

            new_col = item + '_new2_' + function

        agg_train[new_col] = group_train

        agg_test[new_col] = group_test



print('new aggregate train set has {} rows, and {} features'.format(agg_train.shape[0], agg_train.shape[1]))

print('new aggregate test set has {} rows, and {} features'.format(agg_test.shape[0], agg_test.shape[1]))

agg_test = agg_test.reset_index()

agg_train = agg_train.reset_index()



train_agg = pd.merge(df_train, agg_train, on='idhogar')

test = pd.merge(df_test, agg_test, on='idhogar')



#fill all na as 0

train_agg.fillna(value=0, inplace=True)

test.fillna(value=0, inplace=True)



print('train shape:', train_agg.shape, 'test shape:', test.shape)

aggr_list = ['rez_esc', 'dis', 'male', 'female', 

                  'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 

                  'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 

                  'parentesco11', 'parentesco12',

                  'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9',

                 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'refrig', 'television', 'mobilephone',

            'area1', 'area2', 'v18q', 'edjef']

    

for lugar in ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']:

    group_train = df_train[[lugar, 'idhogar'] + aggr_list].groupby([lugar, 'idhogar']).sum().reset_index()

    group_train.columns = [lugar, 'idhogar'] + ['new3_{}_idhogar_{}'.format(lugar, col) for col in group_train][2:]



    group_test = df_test[[lugar, 'idhogar'] + aggr_list].groupby([lugar, 'idhogar']).sum().reset_index()

    group_test.columns = [lugar, 'idhogar'] + ['new3_{}_idhogar_{}'.format(lugar, col) for col in group_test][2:]



    train_agg = pd.merge(train_agg, group_train, on=[lugar, 'idhogar'])

    test = pd.merge(test, group_test, on=[lugar, 'idhogar'])

    

print('train shape:', train_agg.shape, 'test shape:', test.shape)

aggr_list = ['rez_esc', 'dis', 'male', 'female', 

                  'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 

                  'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 

                  'parentesco11', 'parentesco12',

                  'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9',

                 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'refrig', 'television', 'mobilephone',

            'area1', 'area2', 'v18q', 'edjef']

    

for lugar in ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']:

    group_train = df_train[[lugar, 'idhogar'] + aggr_list].groupby([lugar, 'idhogar']).sum().reset_index()

    group_train.columns = [lugar, 'idhogar'] + ['new4_{}_idhogar_{}'.format(lugar, col) for col in group_train][2:]



    group_test = df_test[[lugar, 'idhogar'] + aggr_list].groupby([lugar, 'idhogar']).sum().reset_index()

    group_test.columns = [lugar, 'idhogar'] + ['new4_{}_idhogar_{}'.format(lugar, col) for col in group_test][2:]



    train_agg = pd.merge(train_agg, group_train, on=[lugar, 'idhogar'])

    test = pd.merge(test, group_test, on=[lugar, 'idhogar'])

    

print('train shape:', train_agg.shape, 'test shape:', test.shape)

# 이 로직은 시간이 많이 소요..  over 10min...

cols_nums = ['age', 'meaneduc', 'dependency', 

             'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total',

             'bedrooms', 'overcrowding']



for function in tqdm(['mean','std','min','max','sum', 'count', max_min]):

    for lugar in ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']:

        group_train = df_train[[lugar, 'idhogar'] + aggr_list].groupby([lugar, 'idhogar']).agg(function).reset_index()

        group_train.columns = [lugar, 'idhogar'] + ['new5_{}_idhogar_{}_{}'.format(lugar, col, function) for col in group_train][2:]



        group_test = df_test[[lugar, 'idhogar'] + aggr_list].groupby([lugar, 'idhogar']).agg(function).reset_index()

        group_test.columns = [lugar, 'idhogar'] + ['new5_{}_idhogar_{}_{}'.format(lugar, col, function) for col in group_test][2:]



        train_agg = pd.merge(train_agg, group_train, on=[lugar, 'idhogar'])

        test = pd.merge(test, group_test, on=[lugar, 'idhogar'])

        

print('train shape:', train_agg.shape, 'test shape:', test.shape)

train = train_agg.query('parentesco1==1')
train['dependency'].replace(np.inf, 0, inplace=True)

test['dependency'].replace(np.inf, 0, inplace=True)
submission = test[['Id']]



#Remove useless feature to reduce dimension

train.drop(columns=['idhogar','Id', 'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace=True)

test.drop(columns=['idhogar','Id',  'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace=True)



correlation = train.corr()

correlation = correlation['Target'].sort_values(ascending=False)

print('final_data size', train.shape, test.shape)

print(f'The most 20 positive feature: \n{correlation.head(40)}')
print(f'The most 20 negative feature: \n{correlation.tail(20)}')
# shape[0] == 2, 두 가지 값을 가진 feature는 binary category

#

binary_cat_features = [col for col in train.columns if train[col].value_counts().shape[0] == 2]

object_features = ['edjefe', 'edjefa']



# category 는 binary와 object를 합침.

categorical_feats = binary_cat_features + object_features
categorical_feats
def evaluate_macroF1_lgb(truth, predictions):  

    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483

    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)

    f1 = f1_score(truth, pred_labels, average='macro')

    return ('macroF1', f1, True) 

y = train['Target']

train.drop(columns=['Target'], inplace=True)
def print_execution_time(start):

    end = time.time()

    hours, rem = divmod(end-start, 3600)

    minutes, seconds = divmod(rem, 60)

    print('*'*20, "Execution ended in {:0>2}h {:0>2}m {:05.2f}s".format(int(hours),int(minutes),seconds), '*'*20)

def extract_good_features_using_shap_LGB(params, SEED):

    clf = lgb.LGBMClassifier(objective='multiclass',

                             random_state=1989,

                             max_depth=params['max_depth'], 

                             learning_rate=params['learning_rate'],  

                             silent=True, 

                             metric='multi_logloss',

                             n_jobs=-1, n_estimators=10000, 

                             class_weight='balanced',

                             colsample_bytree = params['colsample_bytree'], 

                             min_split_gain= params['min_split_gain'], 

                             bagging_freq = params['bagging_freq'],

                             min_child_weight=params['min_child_weight'],

                             num_leaves = params['num_leaves'], 

                             subsample = params['subsample'],

                             reg_alpha= params['reg_alpha'],

                             reg_lambda= params['reg_lambda'],

                             num_class=len(np.unique(y)),

                             bagging_seed=SEED,

                             seed=SEED,

                            )



    kfold = 5

    kf = StratifiedKFold(n_splits=kfold, shuffle=True) 

           # shuffle before divide data

           # n_splits : 5 divide

            

    feat_importance_df  = pd.DataFrame()



    for i, (train_index, test_index) in enumerate(kf.split(train, y)):

        print('='*30, '{} of {} folds'.format(i+1, kfold), '='*30)

        start = time.time()

        X_train, X_val = train.iloc[train_index], train.iloc[test_index]

        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric=evaluate_macroF1_lgb, 

                categorical_feature=categorical_feats,

                early_stopping_rounds=500, verbose=500)

        shap_values = shap.TreeExplainer(clf.booster_).shap_values(X_train)

        fold_importance_df  = pd.DataFrame()

        fold_importance_df['feature'] = X_train.columns

        fold_importance_df['shap_values'] = abs(np.array(shap_values)[:, :].mean(1).mean(0))

        fold_importance_df['feat_imp'] = clf.feature_importances_

        feat_importance_df = pd.concat([feat_importance_df, fold_importance_df])

        print_execution_time(start)



    feat_importance_df_shap = feat_importance_df.groupby('feature').mean().sort_values('shap_values', ascending=False).reset_index()

#     feat_importance_df_shap['shap_cumsum'] = feat_importance_df_shap['shap_values'].cumsum() / feat_importance_df_shap['shap_values'].sum()

#     good_features = feat_importance_df_shap.loc[feat_importance_df_shap['shap_cumsum'] < 0.999].feature

    return feat_importance_df_shap

# run time : 10???min

total_shap_df  = pd.DataFrame()

NUM_ITERATIONS = 3

for SEED in range(NUM_ITERATIONS):

    print('#'*40, '{} of {} iterations'.format(SEED+1, NUM_ITERATIONS), '#' * 40)

    params = {'max_depth': np.random.choice([5, 6, 7, 8, 10, 12, -1]),

             'learning_rate': np.random.rand() * 0.02,

              'colsample_bytree': np.random.rand() * (1 - 0.5) + 0.5,

              'subsample': np.random.rand() * (1 - 0.5) + 0.5,

              'min_split_gain': np.random.rand() * 0.2,

              'num_leaves': np.random.choice([32, 48, 64]),

              'reg_alpha': np.random.rand() * 2,

              'reg_lambda': np.random.rand() *2,

              'bagging_freq': np.random.randint(4) +1,

              'min_child_weight': np.random.randint(100) + 20

             }

    temp_shap_df = extract_good_features_using_shap_LGB(params, SEED)

    total_shap_df = pd.concat([total_shap_df, temp_shap_df])

# 1) SHAP value 높은 순으로 정렬해서 shap_sorted_df 에 저장.

shap_sorted_df = total_shap_df.groupby('feature').mean().sort_values('shap_values', ascending=False).reset_index()



# 2) LGBM feature importances feature를 feat_imp_sorted_df에 저장



# LGBMClassifier : 각 독립 변수의 중요도(feature importance)를 계산할 수 있다는 점이다.

# 모든 노드에 대해 어떤 독립 변수를 사용하였고 그 노드에서 얻은 information gain을 구할 수 있으므로 

# 각각의 독립 변수들이 얻어낸 information gain의 평균을 비교하면 어떤 독립 변수가 중요한지를 비교할 수 있다.

# feat_imp은 LGBMClassifier.feature_importances_ feature임.

feat_imp_sorted_df = total_shap_df.groupby('feature').mean().sort_values('feat_imp', ascending=False).reset_index()



# 3) 상위 500개만 가져옴.

features_top_shap = shap_sorted_df['feature'][:500]

features_top_feat_imp = feat_imp_sorted_df['feature'][:500]



# 4) 위 두 feature를 합쳐서 top_features를 만듬.

top_features = pd.Series(features_top_shap.tolist() + features_top_feat_imp.tolist())

top_features = top_features.unique()

new_train = train[top_features].copy()

new_test = test[top_features].copy()
print('new_train shape:', new_train.shape, 'new_test shape:', new_test.shape)
# top_features 중에서, categorical_feats 만 뽑아냄.

new_categorical_feats = [col for col in top_features if col in categorical_feats]
def LGB_OOF(params, categorical_feats, N_FOLDs, SEED=1989):

    clf = lgb.LGBMClassifier(objective='multiclass',

                             random_state=1989,

                             max_depth=params['max_depth'], 

                             learning_rate=params['learning_rate'],  

                             silent=True, 

                             metric='multi_logloss',

                             n_jobs=-1, n_estimators=10000, 

                             class_weight='balanced',

                             colsample_bytree = params['colsample_bytree'], 

                             min_split_gain= params['min_split_gain'], 

                             bagging_freq = params['bagging_freq'],

                             min_child_weight=params['min_child_weight'],

                             num_leaves = params['num_leaves'], 

                             subsample = params['subsample'],

                             reg_alpha= params['reg_alpha'],

                             reg_lambda= params['reg_lambda'],

                             num_class=len(np.unique(y)),

                             bagging_seed=SEED,

                             seed=SEED,

                            )



    kfold = N_FOLDs

    kf = StratifiedKFold(n_splits=kfold, shuffle=True)

    feat_importance_df  = pd.DataFrame()

    predicts_result = []



    for i, (train_index, test_index) in enumerate(kf.split(new_train, y)):

        print('='*30, '{} of {} folds'.format(i+1, kfold), '='*30)

        start = time.time()

        X_train, X_val = new_train.iloc[train_index], new_train.iloc[test_index]

        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric=evaluate_macroF1_lgb,categorical_feature=new_categorical_feats,

                early_stopping_rounds=500, verbose=500)

        shap_values = shap.TreeExplainer(clf.booster_).shap_values(X_train)

        fold_importance_df  = pd.DataFrame()

        fold_importance_df['feature'] = X_train.columns

        fold_importance_df['shap_values'] = abs(np.array(shap_values)[:, :].mean(1).mean(0))

        fold_importance_df['feat_imp'] = clf.feature_importances_

        feat_importance_df = pd.concat([feat_importance_df, fold_importance_df])

        predicts_result.append(clf.predict(new_test))

        print_execution_time(start)

    return predicts_result, feat_importance_df

params = {'max_depth': 6,

         'learning_rate': 0.002,

          'colsample_bytree': 0.8,

          'subsample': 0.8,

          'min_split_gain': 0.02,

          'num_leaves': 48,

          'reg_alpha': 0.04,

          'reg_lambda': 0.073,

          'bagging_freq': 2,

          'min_child_weight': 40

         }



N_Folds = 20

SEED = 1989



# predit결과와 모델(SHAP value포함) 에서 중요한 feat 

predicts_result, feat_importance_df = LGB_OOF(params, new_categorical_feats, N_Folds, SEED=1989)

fig, ax = plt.subplots(1, 2, figsize=(20, 20))

feat_importance_df_shap = feat_importance_df.groupby('feature').mean().sort_values('shap_values', ascending=False).reset_index()



num_features = 50

sns.barplot(x=feat_importance_df_shap.shap_values[:num_features], y=feat_importance_df_shap.feature[:num_features], ax=ax[0])

ax[0].set_title('Feature importance based on shap values')



feat_importance_df = feat_importance_df.groupby('feature').mean().sort_values('feat_imp', ascending=False).reset_index()



num_features = 50

sns.barplot(x=feat_importance_df.shap_values[:num_features], y=feat_importance_df.feature[:num_features], ax=ax[1])

ax[1].set_title('Feaure importance based on feature importance from lgbm')

plt.show()



# shap value 기준으로 중요한 feature

#     => phones-per-capital, new5_lugar1_idhogar_instlevel8_mean..

# LGBM에서 중요한 feature

#     => measureduc, escolari_age ..
submission['Target'] = np.array(predicts_result).mean(axis=0).round().astype(int)

submission.to_csv('submission_with_new_feature_set.csv', index = False)

optimized_param = None

lowest_cv = 1000

total_iteration = 100

for i in range(total_iteration):

    print('-'*20, 'For {} of {} iterations'.format(i+1, total_iteration), '-'*20)

    learning_rate = np.random.rand() * 0.02

    n_folds = 3



    num_class = len(np.unique(y))



    params = {}

    params['application'] = 'multiclass'

    params['metric'] = 'multi_logloss'

    params['num_class'] = num_class

    params['class_weight'] = 'balanced'

    params['num_leaves'] = np.random.randint(24, 48)

    params['max_depth'] = np.random.randint(5, 8)

    params['min_child_weight'] = np.random.randint(5, 50)

    params['min_split_gain'] = np.random.rand() * 0.09

    params['colsample_bytree'] = np.random.rand() * (0.9 - 0.1) + 0.1

    params['subsample'] = np.random.rand() * (1 - 0.8) + 0.8

    params['bagging_freq'] = np.random.randint(1, 5)

    params['bagging_seed'] = np.random.randint(1, 5)

    params['reg_alpha'] = np.random.rand() * 2

    params['reg_lambda'] = np.random.rand() * 2

    params['learning_rate'] = np.random.rand() * 0.02

    params['seed']  =1989



    d_train = lgb.Dataset(data=new_train, label=y.values-1, categorical_feature=new_categorical_feats, free_raw_data=False)

    cv_results = lgb.cv(params=params, train_set=d_train, num_boost_round=10000, categorical_feature=new_categorical_feats,

                        nfold=n_folds, stratified=True, shuffle=True, early_stopping_rounds=1, verbose_eval=1000)



    min_cv_results = min(cv_results['multi_logloss-mean'])



    if min_cv_results < lowest_cv:

        lowest_cv = min_cv_results

        optimized_param = params

N_Folds = 20

SEED = 1989

predicts_result, feat_importance_df = LGB_OOF(optimized_param, new_categorical_feats, N_Folds, SEED=1989)

submission['Target'] = np.array(predicts_result).mean(axis=0).round().astype(int)

submission.to_csv('submission_shap_randomized_search.csv', index = False)