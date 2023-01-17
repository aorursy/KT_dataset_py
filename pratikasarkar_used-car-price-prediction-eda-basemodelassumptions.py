import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('../input/used-cars-database/autos.csv',encoding='latin_1')

zip_codes = pd.read_csv('../input/zipcodes/Zipcodes.csv')

zip_codes = zip_codes.loc[:,['zipcode','state']]
df = df.merge(zip_codes,how='left',left_on='postalCode',right_on='zipcode',validate = 'm:1')

df.drop(columns='zipcode',inplace=True)

df.head()
df['seller'].value_counts()
df['seller'].replace({'privat':'private',

                      'gewerblich':'commercial'},

                     inplace = True)

df['seller'].head()
df['offerType'].value_counts()
df['offerType'].replace({'Angebot':'offer',

                      'Gesuch':'application'},

                     inplace = True)

df['offerType'].head()
df['abtest'].value_counts()
df['vehicleType'].value_counts()
df['vehicleType'].replace({'kleinwagen':'small car',

                      'kombi':'microbus',

                      'cabrio':'convertible',

                      'andere' : 'other'},

                     inplace = True)

df['vehicleType'].head()
df['gearbox'].value_counts()
df['gearbox'].replace({'manuell':'manual',

                      'automatik':'automatic'},

                     inplace = True)

df['gearbox'].head()
orig_list = list(df['model'].value_counts().index)

converted_list = ['golf','other','3s','polo','corsa','astra','passat','a4','cclass','5s','eclass','a3','a6','focus','fiesta','transporter','twingo','2series','fortwo','aclass','vectra','1s','mondeo','clio','touran','3series','punto','zafira','megane','ibiza','ka','lupo','xseries','octavia','cooper','fabia','clk','micra','caddy','80','sharan','scenic','omega','slk','leon','laguna','civic','tt','1stseries','6series','iseries','galaxy','mclass','7s','meriva','yaris','great','mxseries','a5','kangoo','911','bclass','500','tiguan','vito','escort','one','arosa','zseries','bora','colt','beetle','berlingo','sprinter','tigra','v40','transit','touareg','fox','swift','insignia','c_max','corolla','panda','seicento','sl','v70','4series','scirocco','156','a1','primera','espace','grand','stilo','almera','a8','147','avensis','qashqai','eos','c3','c5','signum','Beetle','s_max','5series','q5','c4','matiz','ducato','agila','aygo','viano','getz','601','combo','100','carisma','cayenne','boxster','alhambra','cordoba','c2','superb','c1','kuga','forfour','rio','jetta','cuore','a2','altea','cadet','rav','picanto','sorento','mseries','accord','crseries','up','q7','vivaro','toledo','voyager','xcseries','Bravo','santa','doblo','logan','mode','verso','ptcruiser','cl','sportage','jazz','fusion','sandero','mustang','roomster','carnival','6s','ceed','gallant','v50','q3','tucson','lancer','auris','impreza','phaeton','freelander','glk','calibra','pajero','x_trail','850','159','jimny','ypsilon','spider','duster','clubman','yeti','cseries','cc','roadster','cherokee','x_type','gclass','captiva','vclass','wrangler','legacy','s60','300c','rxseries','defender','justy','sirion','forester','outlander','grade','niva','s_type','spark','r19','navara','cxseries','aveo','900','antara','90','juke','discovery','exeo','range_rover_sport','kalos','range_rover','citigo','lanos','mii','crossfire','range_rover_evoque','gl','nubira','move','lybra','145','v60','croma','amarok','delta','terios','lodgy','9000','charade','b_max','musa','materia','200','kappa','samara','elefantino','i3','kalina','serie_2','rangerover','serie_3','serie_1','discovery_sport']
df['model'].replace(dict(zip(orig_list,converted_list)),inplace = True)

df.head()
df['monthOfRegistration'].value_counts()
df['fuelType'].value_counts()
df['fuelType'].replace({'benzin':'petrol',

                      'andere':'other',

                      'elektro':'electric'},

                     inplace = True)

df['fuelType'].head()
df['brand'].value_counts()
df['notRepairedDamage'].value_counts()
df['notRepairedDamage'].replace({'nein':'No',

                      'ja':'Yes'},

                     inplace = True)

df['notRepairedDamage'].head()
df.head(3)
df.shape
df.info()
df['dateCrawled'] = pd.to_datetime(df['dateCrawled'])

df['dateCreated'] = pd.to_datetime(df['dateCreated'])

df['lastSeen'] = pd.to_datetime(df['lastSeen'])
df.info()
df.head()
def get_missing_val_count_df(df):

  missing_count_list = []

  for col in df.columns:

    missing_count_list.append(df[col].isnull().sum())

  missing_count_df = pd.DataFrame(missing_count_list,columns=['count'],index = df.columns)

  return missing_count_df.sort_values('count',ascending=False)

get_missing_val_count_df(df)
df['price'].describe()
plt.figure(figsize = (20,5))

sns.boxplot(df['price'])
plt.figure(figsize = (20,5))

sns.boxplot(df[(df['price']>=100) & (df['price']<=100000)]['price'])
dfprice = df[(df['price'].isnull() == False)]['price']
#Import necessary libraries

from sklearn.ensemble import IsolationForest

#The required columns

isolation_forest = IsolationForest(contamination='auto')

isolation_forest.fit(dfprice.values.reshape(-1,1))



xx = np.linspace(dfprice.min(), dfprice.max(), len(df)).reshape(-1,1)

print(xx)

anomaly_score = isolation_forest.decision_function(xx)

outlier = isolation_forest.predict(xx)



plt.figure(figsize=(25,5))

plt.plot(xx, anomaly_score, label='anomaly score')

plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 

                     where=outlier==-1, color='r', 

                     alpha=.4, label='outlier region')

plt.legend()

plt.xlim(0, 100000)

# plt.axis([0,100000,0,0])

plt.title('price')
sns.distplot(df[(df['price']>=100) & (df['price']<=100000)]['price'])
df[(df['price']>=100) & (df['price']<=100000)].shape
df2 = df[(df['price']>=100) & (df['price']<=100000)]
get_missing_val_count_df(df2)
from scipy.stats import shapiro

shapiro(df2['price'])
df2['kilometer'].describe()
df2['kilometer'].value_counts()
df2['kilometer'].value_counts().plot(kind = 'bar')
from scipy.stats import f_oneway

def oneway_posthoc(i):

    cat = {}

    for j in range(df2[i].nunique()):

        if pd.isna(df2[i].unique()[j]) == False:

            cat[df2[i].unique()[j]] = df2[df2[i] == df2[i].unique()[j]]['price']

    tstat,pval = f_oneway(*list(cat.values()))

    print(f'pvalue for {i} is {pval}')

    print()

    for k in cat:

        print(f'Avg price of car for {i} = {k} is ',cat[k].mean())
oneway_posthoc('kilometer')
df2.boxplot(column='price',by = 'kilometer')
df2['notRepairedDamage'].describe()
df2['notRepairedDamage'].value_counts(dropna = False).plot(kind = 'bar')
df2 = df2[df2['notRepairedDamage'].isnull() == False]
price_repaired = df2.loc[df2['notRepairedDamage'] == 'No','price']

price_notRepaired = df2.loc[df2['notRepairedDamage'] == 'Yes','price']
print(shapiro(price_repaired))

print(shapiro(price_notRepaired))
from scipy.stats import bartlett

print(bartlett(price_repaired,price_notRepaired))
from scipy.stats import mannwhitneyu

print(mannwhitneyu(price_repaired,price_notRepaired))
print('Avg price for car which has been repaired : ',price_repaired.mean())

print('Avg price for car which has not been repaired : ',price_notRepaired.mean())
df2.boxplot(column='price',by = 'notRepairedDamage')
df2['vehicleType'].describe()
df2['vehicleType'].value_counts(dropna = False).plot(kind = 'bar')
oneway_posthoc('vehicleType')
df2.boxplot(column='price',by = 'vehicleType')
df2['fuelType'].describe()
print(df2['fuelType'].value_counts(dropna = False))

df2['fuelType'].value_counts(dropna = False).plot(kind = 'bar')
df2 = df2[df2['fuelType'].isnull() == False]
oneway_posthoc('fuelType')
df2.boxplot(column='price',by = 'fuelType')
df2['model'].describe()
plt.figure(figsize=(40,8))

df2['model'].value_counts(dropna = False).plot(kind = 'bar')
df2 = df2[df2['model'].isnull() == False]
oneway_posthoc('model')
df2['gearbox'].describe()
df2['gearbox'].value_counts(dropna = False).plot(kind = 'bar')
df2 = df2[df2['gearbox'].isnull() == False]
price_manual = df2.loc[df2['gearbox'] == 'manual','price']

price_automatic = df2.loc[df2['gearbox'] == 'automatic','price']
print(shapiro(price_manual))

print(shapiro(price_automatic))
from scipy.stats import bartlett

print(bartlett(price_manual,price_automatic))
from scipy.stats import mannwhitneyu

print(mannwhitneyu(price_manual,price_automatic))
print('Avg price for car which has manual gearbox : ',price_manual.mean())

print('Avg price for car which has automatic gearbox : ',price_automatic.mean())
df2.boxplot(column='price',by = 'gearbox')
df2['seller'].describe()
print(df2['seller'].value_counts(dropna = False))

df2['seller'].value_counts(dropna = False).plot(kind = 'bar')
df2['offerType'].describe()
print(df2['offerType'].value_counts(dropna = False))

df2['offerType'].value_counts(dropna = False).plot(kind = 'bar')
df2['abtest'].describe()
df2['abtest'].value_counts(dropna = False).plot(kind = 'bar')
price_test = df2.loc[df2['abtest'] == 'test','price']

price_control = df2.loc[df2['abtest'] == 'control','price']
print(shapiro(price_test))

print(shapiro(price_control))
from scipy.stats import bartlett

print(bartlett(price_test,price_control))
from scipy.stats import mannwhitneyu

print(mannwhitneyu(price_test,price_control))
df2['yearOfRegistration'].describe()
df2.loc[(df2['yearOfRegistration']<1923) | (df2['yearOfRegistration']>2020),'yearOfRegistration'] = np.nan
plt.figure(figsize=(25,5))

df2['yearOfRegistration'].value_counts(dropna = False).plot(kind = 'bar')
df2 = df2[df2['yearOfRegistration'].isnull() == False]
df2['powerPS'].describe()
isolation_forest = IsolationForest(contamination='auto')

isolation_forest.fit(df2['powerPS'].values.reshape(-1,1))



xx = np.linspace(df2['powerPS'].min(), df2['powerPS'].max(), len(df)).reshape(-1,1)

print(xx)

anomaly_score = isolation_forest.decision_function(xx)

outlier = isolation_forest.predict(xx)



plt.figure(figsize=(25,5))

plt.plot(xx, anomaly_score, label='anomaly score')

plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 

                     where=outlier==-1, color='r', 

                     alpha=.4, label='outlier region')

plt.legend()

plt.xlim(40, 800)

plt.title('Power PS')
df2 = df2[(df2['powerPS']<800) & (df2['powerPS']>40)]
sns.scatterplot(x = 'price', y = 'powerPS',data = df2)
df2[['price','powerPS']].corr()
df2['state'].describe()
plt.figure(figsize=(15,5))

df2['state'].value_counts(dropna = False).plot(kind = 'bar')
df2 = df2[df2['state'].isnull() == False]
oneway_posthoc('state')
df2.boxplot(column='price',by = 'state',figsize = (15,5))

plt.xticks(rotation = 90)
df2['CountryOfManufacture'] = df2['brand']



df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace(['ford','chevrolet','chrysler','jeep'], 'USA')

df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace(['jaguar', 'land_rover', 'rover', 'mini'], 'UK')

df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace(['toyota', 'nissan', 'honda', 'subaru', 'mazda', 

                                                                   'mitsubishi','suzuki','daihatsu'], 'Japan')

df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace(['volkswagen', 'bmw', 'audi', 'mercedes_benz','opel',

                                                                   'porsche', 'smart','trabant'], 'Germany')

df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace(['hyundai', 'kia', 'daewoo'], 'Korea')

df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace(['volvo','saab'], 'Sweden')

df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace(['fiat', 'alfa_romeo', 'lancia', 'Ferrari'], 'Italy')

df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace(['skoda'],'Czech')

df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace(['peugeot','renault','citroen'],'France')

df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace('seat','Spain')

df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace('dacia','Romania')

df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace('lada','Russia')

df2['CountryOfManufacture'] = df2['CountryOfManufacture'].replace(['sonstige_autos'],'Others')
df2['CountryOfManufacture'].describe()
plt.figure(figsize=(15,5))

df2['CountryOfManufacture'].value_counts(dropna = False).plot(kind = 'bar')
oneway_posthoc('CountryOfManufacture')
df2.boxplot(column='price',by = 'CountryOfManufacture',figsize = (15,5))

plt.xticks(rotation = 90)
df2['brand'].describe()
plt.figure(figsize=(15,5))

df2['brand'].value_counts(dropna = False).plot(kind = 'bar')
oneway_posthoc('brand')
df2.boxplot(column='price',by = 'brand',figsize = (15,5))

plt.xticks(rotation = 90)
df2['monthOfRegistration'].describe()
plt.figure(figsize=(25,7))

df2['monthOfRegistration'].value_counts(dropna = False).plot(kind = 'bar')
df2 = df2[df2['monthOfRegistration'] != 0]
val = list(df2['lastSeen'] - df2['dateCreated'])

days = [obj.days for obj in val]

for i in range(len(days)):

  if days[i] < 0:

    days[i] = np.nan

df2['No_of_days_online'] = days
sns.scatterplot(x = 'price', y = 'No_of_days_online',data = df2)
df2[['price','No_of_days_online']].corr()
import datetime



def calculateAge(yr,mnth):

    today_date = datetime.datetime.today()

    years = today_date.year - yr

    month = today_date.month - mnth

    ageindecimal = years + month/12

    return round(ageindecimal,2)
df2['ageOfVehicle'] = list(map(calculateAge,df2['yearOfRegistration'],df2['monthOfRegistration']))
sns.scatterplot(y = 'price', x = 'ageOfVehicle',data = df2)
df2[['price','ageOfVehicle']].corr()
df3 = df2[['kilometer','notRepairedDamage', 'vehicleType', 'fuelType', 'gearbox', 'ageOfVehicle', 'model', 'brand', 'powerPS', 'No_of_days_online', 'state','CountryOfManufacture','price']]
df3.head()
get_missing_val_count_df(df3)
# cat_cols = ['notRepairedDamage', 'vehicleType', 'fuelType', 'gearbox', 'model', 'brand']

cat_cols = ['vehicleType', 'model', 'brand']
from sklearn.preprocessing import OrdinalEncoder

ordinal_enc_dict = {}

for col_name in cat_cols:

  ordinal_enc_dict[col_name] = OrdinalEncoder()

  col = df3[col_name]

  col_not_null = col[col.notnull()]

  reshaped_vals = col_not_null.values.reshape(-1, 1)

  encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)

  df3.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
from sklearn.impute import KNNImputer

KNN_imputer = KNNImputer(n_neighbors=20)

df3_KNN = df3[['vehicleType', 'model', 'brand','price']].copy(deep=True)

df3_KNN.iloc[:, :] = np.round(KNN_imputer.fit_transform(df3_KNN))
for col in cat_cols:

    reshaped_col = df3_KNN[col].values.reshape(-1, 1)

    df3_KNN[col] = ordinal_enc_dict[col].inverse_transform(reshaped_col)
df_final = df3.copy(deep=True)
df_final.loc[:,cat_cols] = df3_KNN[cat_cols]
df_final.reset_index(drop=True)
get_missing_val_count_df(df_final)
df_final.head()
# pip install feature-engine
X = df_final.drop(columns=['price'])

y = df_final.price
X['kilometer'] = X['kilometer'].astype('O')

#X['postalCode'] = X['postalCode'].astype('O')
X['kilometer'].nunique()
cat_cols = ['kilometer','notRepairedDamage','vehicleType','fuelType','gearbox','model', 'brand','state','CountryOfManufacture']

num_cols = ['powerPS' ,'ageOfVehicle','No_of_days_online']
cat_cols_high_crdnlty = []

cat_cols_low_crdnlty = []



for i in cat_cols:

    if X[i].nunique()>5:

        cat_cols_high_crdnlty.append(i)

    else:

        cat_cols_low_crdnlty.append(i)

print(cat_cols_high_crdnlty)

print(cat_cols_low_crdnlty)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.25 , random_state = 123)
# def OHE(df):

#     df_OHE = pd.concat([df[num_cols], pd.get_dummies(df[cat_cols], drop_first=True)],axis=1)

#     return df_OHE
# X_train_OHE = OHE(X_train)

# X_test_OHE = OHE(X_test)

# #X_test_OHE.head()
# def categorical_to_counts(df_train, df_test):

#     #df copy

#     df_train_temp = df_train.copy()

#     df_test_temp = df_test.copy()

#     for col in cat_cols_high_crdnlty:

#         counts_map = df_train_temp[col].value_counts().to_dict()

#         df_train_temp[col] = df_train_temp[col].map(counts_map)

#         df_test_temp[col] = df_test_temp[col].map(counts_map)

#     df_train_temp = pd.concat([df_train_temp, pd.get_dummies(df_train_temp[cat_cols_low_crdnlty], drop_first=True)],axis=1)

#     df_test_temp = pd.concat([df_test_temp, pd.get_dummies(df_test_temp[cat_cols_low_crdnlty], drop_first=True)],axis=1)

#     df_train_temp.drop(cat_cols_low_crdnlty,axis = 1, inplace = True)

#     df_test_temp.drop(cat_cols_low_crdnlty,axis = 1, inplace = True)

# #     df_train_temp[cat_cols_low_crdnlty] = pd.get_dummies(df_train_temp[cat_cols_low_crdnlty], drop_first=True)

# #     df_test_temp[cat_cols_low_crdnlty] = pd.get_dummies(df_test_temp[cat_cols_low_crdnlty], drop_first=True)

#     return df_train_temp, df_test_temp



# X_train_count, X_test_count = categorical_to_counts(X_train, X_test)

# X_test_count.head()
# from feature_engine.categorical_encoders import CountFrequencyCategoricalEncoder



# def counts(df_train, df_test):

#     count_enc = CountFrequencyCategoricalEncoder(encoding_method='count',variables=cat_cols)

#     count_enc.fit(df_train)

#     return count_enc.transform(df_train) , count_enc.transform(df_test)
# X_train_count, X_test_count = counts(X_train, X_test)

# #X_test_count.head()
# def categorical_to_freq(df_train, df_test):

#     #df copy

#     df_train_temp = df_train.copy()

#     df_test_temp = df_test.copy()

#     for col in cat_cols_high_crdnlty:

#         freq_map = (df_train_temp.groupby([col]).size()/len(df_train_temp)).to_dict()

#         df_train_temp[col] = df_train_temp[col].map(freq_map)

#         df_test_temp[col] = df_test_temp[col].map(freq_map)

#     df_train_temp = pd.concat([df_train_temp, pd.get_dummies(df_train_temp[cat_cols_low_crdnlty], drop_first=True)],axis=1)

#     df_test_temp = pd.concat([df_test_temp, pd.get_dummies(df_test_temp[cat_cols_low_crdnlty], drop_first=True)],axis=1)

#     df_train_temp.drop(cat_cols_low_crdnlty,axis = 1, inplace = True)

#     df_test_temp.drop(cat_cols_low_crdnlty,axis = 1, inplace = True)

#     return df_train_temp, df_test_temp



# X_train_freq, X_test_freq = categorical_to_freq(X_train, X_test)

# X_test_freq.head()
# def frequency(df_train, df_test):

#     freq_enc = CountFrequencyCategoricalEncoder(encoding_method='frequency',variables=cat_cols)

#     freq_enc.fit(df_train)

#     return freq_enc.transform(df_train) , freq_enc.transform(df_test)
# X_train_freq, X_test_freq = frequency(X_train, X_test)

# #X_test_freq.head()
# def cat_to_mean(df_train, df_test, y_train, y_test):

#     df_train_temp = pd.concat([df_train, y_train], axis=1).copy()

#     df_test_temp = df_test



#     for col in cat_cols_high_crdnlty:

#         ordered_labels = df_train_temp.groupby([col])['price'].mean().to_dict()

#         # Mapping

#         df_train_temp[col] = df_train[col].map(ordered_labels)

#         df_test_temp[col] = df_test[col].map(ordered_labels)



#     # remove the target

#     df_train_temp.drop(['price'], axis=1, inplace=True)

#     df_train_temp = pd.concat([df_train_temp, pd.get_dummies(df_train_temp[cat_cols_low_crdnlty], drop_first=True)],axis=1)

#     df_test_temp = pd.concat([df_test_temp, pd.get_dummies(df_test_temp[cat_cols_low_crdnlty], drop_first=True)],axis=1)

#     df_train_temp.drop(cat_cols_low_crdnlty,axis = 1, inplace = True)

#     df_test_temp.drop(cat_cols_low_crdnlty,axis = 1, inplace = True)

#     return df_train_temp, df_test_temp





# X_train_mean, X_test_mean = cat_to_mean(

#     X_train, X_test, y_train, y_test)



# X_test_mean.head()
# from feature_engine.categorical_encoders import MeanCategoricalEncoder



# def mean_enc(df_train,df_target,df_test):

#     mean_enc = MeanCategoricalEncoder(variables=cat_cols)

#     mean_enc.fit(df_train,df_target)

#     return mean_enc.transform(df_train) , mean_enc.transform(df_test)
# X_train_mean, X_test_mean = frequency(X_train, X_test)

# #X_test_mean.head()
from sklearn.model_selection import KFold



#train1,test1=train_test_split(df_final,test_size=0.2,stratify = df_final.model,random_state=321)

train1,test1=train_test_split(df_final,test_size=0.2,random_state=321)



def k_fold_target(train,test,columns,target,folds=5):

    train2 = train

    for column in columns:

        for i,j in KFold(n_splits=folds).split(train):

            mean=train.loc[train.index[i]].groupby(column)[target].mean()

            train.loc[train.index[j],column+'_Enc']=train.loc[train.index[j],column].map(mean)

        test[column+'_Enc']=test[column].map(train.groupby(column)[target].mean())

        train2[column+'_Enc']=train2[column].map(train.groupby(column)[target].mean())

        train = train.drop(column,axis=1)

        train2  = train2.drop(column,axis=1) 

        test  = test.drop(column,axis=1) 

    return train2,test
# cat_cols.append('postalCode')

train_kte , test_kte = k_fold_target(train1,test1,cat_cols_high_crdnlty,'price')

# train_kte.drop('postalCode',axis = 1,inplace = True)

# test_kte.drop('postalCode',axis = 1,inplace = True)

train_kte = pd.concat([train_kte, pd.get_dummies(train_kte[cat_cols_low_crdnlty], drop_first=True)],axis=1)

test_kte = pd.concat([test_kte, pd.get_dummies(test_kte[cat_cols_low_crdnlty], drop_first=True)],axis=1)

train_kte.drop(cat_cols_low_crdnlty,axis = 1, inplace = True)

test_kte.drop(cat_cols_low_crdnlty,axis = 1, inplace = True)

train_kte.head()

test_kte.head()
y_train = np.log1p(y_train)
train_kte.dropna(inplace=True)
from sklearn.linear_model import LinearRegression , LassoCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import KFold , cross_val_score , cross_validate

from sklearn.metrics import make_scorer

from sklearn.preprocessing import StandardScaler
# res = []

# def eval(Enc,X,y):

#     for i in (LinearRegression() , LassoCV() , RandomForestRegressor(max_depth=4,n_jobs=-1) , KNeighborsRegressor()):

#         scoring = {'r2':'r2' , 

#                    'Neg_RMSE' : 'neg_root_mean_squared_error'}

#         kfold = KFold(n_splits=5, random_state=3)

#         n = len(list(kfold.split(X))[0][1])

#         p = len(X.columns)

#         if isinstance(i,KNeighborsRegressor):

#             ss = StandardScaler()

#             X_scaled = ss.fit_transform(X)

#             results = cross_validate(i, X_scaled, y, cv=kfold, scoring=scoring)

#         else:

#             results = cross_validate(i, X, y, cv=kfold, scoring=scoring)

#         res1 = results['test_r2']

#         res1 = list(map(lambda x : 1 - (1-x) * (n-1)/(n-p-1),res1))

#         #res.append((Enc,str(i).split('(')[0],'r2',np.mean(results['test_r2']),np.std(results['test_r2'])))

#         res.append((Enc,str(i).split('(')[0],'Adj_r2',np.mean(res1),np.std(res1)))

#         res.append((Enc,str(i).split('(')[0],'Neg_RMSE',np.mean(results['test_Neg_RMSE']),np.std(results['test_Neg_RMSE'])))
# from tqdm import tqdm_notebook as tqdm

# for i,j,k in tqdm((('OHE',X_train_OHE,y_train),('Count Encoding',X_train_count,y_train),('Frequency Encoding',X_train_freq,y_train),

#              ('Mean Encoding',X_train_mean,y_train),('K Fold Target Encoding',train_kte.drop(columns='price'),np.log1p(train_kte['price'])))):

#     eval(i,j,k)



# score = pd.DataFrame(res,columns=['Encoding','Algorithm','Scoring','Bias','Var'])

# score.to_csv('scores_encoding.csv',index=False)



# s1 = score

# s1['Bias'] = s1['Bias'].round(4)

# s1['Var']  = s1['Var'].round(4)

# s1['Scores'] = s1['Bias'].astype('str') +' (+/-'+ s1['Var'].astype('str')+')'

# s1.drop(columns=['Bias','Var'],inplace=True)

# s1.set_index(['Algorithm','Encoding','Scoring']).unstack('Algorithm')
import statsmodels.api as sm



X = train_kte.drop(columns='price')

y = np.log1p(train_kte['price'])



X_constant = sm.add_constant(X)

lr = sm.OLS(y,X_constant).fit()

lr.summary()
from statsmodels.stats.stattools import durbin_watson

print('Durbin Watson value : ',durbin_watson(lr.resid))
import statsmodels.tsa.api as smt



fig , ax = plt.subplots(figsize=(25, 10))

ax.set_ylim([-0.007,0.007])

acf = smt.graphics.plot_acf(lr.resid,lags=150, alpha=0.05 , ax = ax)

plt.title('ACF Plot')

acf.show()
from scipy import stats

jb_val , pval = stats.jarque_bera(lr.resid)

print('P value for Jarque Bera test is ',pval)

print('Test statistic value for Jarque Bera test is',jb_val)
#test critical

stats.chi2.isf(0.05, df=2)
fig = sm.qqplot(lr.resid,fit=True,line='45')
fstat , pval = sm.stats.diagnostic.linear_rainbow(res=lr, frac=0.5)

print('Pvalue for Linear rainbow test is ',np.round(pval,3))
sns.set_style('darkgrid')

fig, ax = plt.subplots(1,2,figsize = (30,10))



predicted = lr.predict()[:5000]

resid = lr.resid[:5000]



sns.regplot(x=predicted, y=y[:5000], lowess=True, ax=ax[0], line_kws={'color': 'red'})

ax[0].set_title('Observed vs. Predicted Values', fontsize=16)

ax[0].set(xlabel='Predicted', ylabel='Observed')



sns.regplot(x=predicted, y=resid, lowess=True, ax=ax[1], line_kws={'color': 'red'})

ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)

ax[1].set(xlabel='Predicted', ylabel='Residuals')
import statsmodels.stats.api as sms

fval , pval , ordering = sms.het_goldfeldquandt(lr.resid, lr.model.exog)

fval , pval , ordering
print('P value for Goldfeld quandt is ',np.round(pval,4))
fitted_vals = lr.predict()[:5000]

resids = lr.resid[:5000]

resids_standardized = lr.get_influence().resid_studentized_internal[:5000]

fig, ax = plt.subplots(1,2,figsize = (30,10))



sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})

ax[0].set_title('Residuals vs Fitted', fontsize=16)

ax[0].set(xlabel='Fitted Values', ylabel='Residuals')

sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})

ax[1].set_title('Scale-Location', fontsize=16)

ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]

pd.DataFrame({'vif': vif[1:]}, index=X.columns).T
X1 = X

X1['price'] = y

plt.figure(figsize = (20,15))

sns.heatmap(X1.corr() , annot = True , cmap = 'viridis')
