# Import packages

import numpy as np 

import pandas as pd 

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt

from scipy import stats

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
# Load data

train_data = pd.read_csv('../input/dataset_treino.csv')

test_data = pd.read_csv('../input/dataset_teste.csv')
# Preview data

train_data.head(2)
# Preview data

test_data.head(2)
# Check the number of registers and features

print("Train shape with Id : {} ".format(train_data.shape))

print("Test shape with Id : {} ".format(test_data.shape))



#id_name='OrderId'



#Save the 'Id' column

train_ID = train_data['Order']  # fel! behöver fixas

test_ID = test_data['OrderId']  # fel! behöver fixas

test_ID2 = test_data['Property Id'] # rätt 



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train_data.drop('Order', axis = 1, inplace = True)

test_data.drop('OrderId', axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nTrain shape without Id : {} ".format(train_data.shape)) 

print("Test shape without Id : {} ".format(test_data.shape))
target = 'ENERGY STAR Score'

source = 'Source'



y_train = train_data[target]

features = train_data.columns.tolist()

features.remove(target)



train_data[source] = 'Train'

test_data[source] = 'Test'



df = pd.concat((train_data, test_data)).reset_index(drop=True)

#df.drop([target], axis=1, inplace=True)



train_data.drop([source], axis = 1, inplace = True)

test_data.drop([source], axis = 1, inplace = True)



print("Total dataframe size is : {}".format(df.shape))



#X = df[features]



X_train = train_data[features]

y_train = train_data[target]

X = train_data

y = y_train
# Preview data

df.head(2)
# Obtaining general info about the dataset

df.dtypes
# Obtain an overview of the variables

import pandas_profiling as pf

pf.ProfileReport(df)
df_nan = (df.isnull().sum() / len(df)) * 100

missing_data = pd.DataFrame({'Missing n':df.isnull().sum(),'% Missing' :df_nan})

missing_data.sort_values('% Missing', ascending=False).head(15)
df.replace('Not Available',np.nan,inplace=True)
df_nan = (df.isnull().sum() / len(df)) * 100

missing_data = pd.DataFrame({'Missing n':df.isnull().sum(),'% Missing' :df_nan})

missing_data.sort_values('% Missing', ascending=False).head(45)
numeric_terms=['ft²','kBtu','(therms)','(kWh)','(Metric Tons CO2e)','(kgal)']



for col in list(df.columns):

    for term in numeric_terms:

        if (term in col):

            df[col] = df[col].astype(float)
# Descriptive statistics

df.describe()
#df.loc[df['BBL - 10 digits'].isnull()]

#Shows that id 6656 and 6690 are the ones that have no info on Borough

#df.iloc[6656,:] # 'BBL - 10 digits' was NAN. Used Address to locate BBL online ()

#df.iloc[6690,:] # 'BBL - 10 digits' was NAN. Used Address to locate BBL online ()

#df.iloc[8270,:] #  Has same Postal Code as 6690 but divergent information when looking at website databases. 

                 # 'BBL - 10 digits' and Postal Code.
df.ix[6656,'BBL - 10 digits'] ='2036420001'

df.ix[6690,'BBL - 10 digits'] = '3067130006'

df.ix[8270,'BBL - 10 digits'] = '3044240012'

df.ix[8270,'Postal Code'] = '10467'
df['Largest Property Use Rate'] = df['Largest Property Use Type - Gross Floor Area (ft²)']/df['Property GFA - Self-Reported (ft²)']

df['2nd Property Use Rate'] = df['2nd Largest Property Use - Gross Floor Area (ft²)']/df['Property GFA - Self-Reported (ft²)']

df['3rd Property Use Rate'] = df['3rd Largest Property Use Type - Gross Floor Area (ft²)']/df['Property GFA - Self-Reported (ft²)']

df['Direct GHG Emissions Rate'] = df['Direct GHG Emissions (Metric Tons CO2e)']/df['Total GHG Emissions (Metric Tons CO2e)']

df['BBL - 10 digits'] = df['BBL - 10 digits'].str.extract('(\d+)', expand=False)

df['Borough'] = df['BBL - 10 digits'].str[0]

df['Tax Block'] = df['BBL - 10 digits'].str[1:6]

df['Tax Lot'] = df['BBL - 10 digits'].str[6:10]

df['Postal Code'] = df['Postal Code'].astype(str)

df['Year Built'] = df['Year Built'].astype(str)

df['Borough'] = df['Borough'].astype(str)

df['Tax Block'] = df['Tax Block'].astype(str)

df['Tax Lot'] = df['Tax Lot'].astype(str)

df['Property Id'] = df['Property Id'].astype(str)
df_nan = (df.isnull().sum() / len(df)) * 100

missing_data = pd.DataFrame({'Missing n':df.isnull().sum(),'% Missing' :df_nan})

missing_data.sort_values('% Missing', ascending=False).head(45)
zero_col = ['2nd Property Use Rate','3rd Property Use Rate',

            'Water Intensity (All Water Sources) (gal/ft²)','Weather Normalized Site Natural Gas Intensity (therms/ft²)',

            'Total GHG Emissions (Metric Tons CO2e)','Weather Normalized Site Electricity Intensity (kWh/ft²)',

            'Direct GHG Emissions Rate','Total GHG Emissions (Metric Tons CO2e)'

            ]



for col in zero_col:

    df[col].replace(np.nan, 0, inplace = True)
prop_col = ['2nd Largest Property Use Type','3rd Largest Property Use Type']



for col in prop_col:

    df[col].replace(np.nan, 'none', inplace = True)
drop_items = ['NYC Borough, Block and Lot (BBL) self-reported',

            'NYC Building Identification Number (BIN)',

            'BBL - 10 digits',

            'Parent Property Name',

            'Property Name',

            'Address 1 (self-reported)',

            'Address 2',

            'Street Number',

            'Street Name',

            'Latitude',

            'Longitude',

            'DOF Gross Floor Area',

            'DOF Benchmarking Submission Status',

            'List of All Property Use Types at Property',

            'Largest Property Use Type - Gross Floor Area (ft²)',

            '2nd Largest Property Use - Gross Floor Area (ft²)',

            '3rd Largest Property Use Type - Gross Floor Area (ft²)',                     

            'Fuel Oil #1 Use (kBtu)',                                         

            'Fuel Oil #2 Use (kBtu)',                                         

            'Fuel Oil #4 Use (kBtu)',                                         

            'Fuel Oil #5 & 6 Use (kBtu)',                                     

            'Diesel #2 Use (kBtu)',                                           

            'District Steam Use (kBtu)',                                      

            'Natural Gas Use (kBtu)',                                         

            'Weather Normalized Site Natural Gas Use (therms)',               

            'Electricity Use - Grid Purchase (kBtu)',                         

            'Weather Normalized Site Electricity (kWh)',                      

            'Direct GHG Emissions (Metric Tons CO2e)',                        

            'Indirect GHG Emissions (Metric Tons CO2e)',                      

            'Property GFA - Self-Reported (ft²)',                              

            'Water Use (All Water Sources) (kgal)', 

            'Weather Normalized Site EUI (kBtu/ft²)',

            'Weather Normalized Source EUI (kBtu/ft²)'

           ]



df.drop(drop_items, axis = 1, inplace = True)
# Calculates the correlation and plot the data in s heatmap

sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm', linewidths = 0.1, annot_kws = {'size':18}, fmt = '.2f')

fig = plt.gcf()

fig.set_size_inches(24,24)

plt.xticks(fontsize = 16)

plt.yticks(fontsize = 16)

plt.show()
property_type = {'Multifamily Housing':'Multifamily Housing',  

            'Residence Hall/Dormitory':'Residence Hall/Dormitory',

            'Other - Lodging/Residential':'Residence Hall/Dormitory',

            'Hotel':'Hotel',

            'Adult Education':'College/University',

            'College/University':'College/University',

            'K-12 School':'College/University',

            'Library':'College/University',

            'Vocational School':'College/University',

            'Other - Education':'College/University',

            'Office':'Office',

            'Medical Office':'Office',

            'Financial Office':'Office',

            'Bank Branch':'Office',

            'Distribution Center':'Distribution Center',

            'Self-Storage Facility':'Distribution Center',

            'Wholesale Club/Supercenter':'Distribution Center',

            'Non-Refrigerated Warehouse':'Distribution Center',

            'Fast Food Restaurant':'Food Service',

            'Food Sales':'Food Service',

            'Food Service':'Food Service',

            'Restaurant':'Food Service',

            'Supermarket/Grocery Store':'Food Service',

            'Convenience Store without Gas Station':'Food Service',

            'Other - Restaurant/Bar':'Food Service',

            'Hospital (General Medical & Surgical)':'Senior Care Community',

            'Urgent Care/Clinic/Other Outpatient':'Senior Care Community',

            'Ambulatory Surgical Center':'Senior Care Community',

            'Laboratory':'Senior Care Community',

            'Pre-school/Daycare':'Senior Care Community',

            'Senior Care Community':'Senior Care Community',

            'Outpatient Rehabilitation/Physical Therapy':'Senior Care Community',

            'Retail Store':'Retail Store',

            'Repair Services (Vehicle, Shoe, Locksmith, etc.)':'Retail Store',

            'Mailing Center/Post Office':'Retail Store',

            'Automobile Dealership':'Retail Store',

            'Mailing Center/Post Office':'Retail Store',

            'Personal Services (Health/Beauty, Dry Cleaning...':'Retail Store',

            'Enclosed Mall':'Retail Store',

            'Other - Mall':'Retail Store',

            'Other - Services':'Retail Store',

            'Other - Utility':'Retail Store',

            'Bar/Nightclub':'Recreation',

            'Bowling Alley':'Recreation',

            'Fitness Center/Health Club/Gym':'Recreation',

            'Other - Recreation':'Recreation',

            'Other - Entertainment/Public Assembly':'Recreation',

            'Performing Arts':'Recreation',

            'Social/Meeting Hall':'Recreation',

            'Museum':'Recreation',

            'Worship Facility':'Recreation',

            'Other':'Other',

            'Courthouse':'Other',

            'Other - Public Services':'Other',

            'Swimming Pool':'Other',

            'Parking':'Other',

            'Refrigerated Warehouse':'Other',

            'Data Center':'Other',

            'none':'none'

              }
df['Largest Property Use Type'] = df['Largest Property Use Type'].map(property_type).astype(str)

df['2nd Largest Property Use Type'] = df['2nd Largest Property Use Type'].map(property_type).astype(str)

df['3rd Largest Property Use Type'] = df['3rd Largest Property Use Type'].map(property_type).astype(str)
sns.boxplot(train_data['ENERGY STAR Score'],train_data['Largest Property Use Type'])
new_droplist = ['Metered Areas (Energy)','Metered Areas  (Water)','Release Date','Water Required?','Community Board','Council District','Census Tract','NTA']



df.drop(new_droplist, axis = 1, inplace = True)
# Select the numeric columns

df_numeric_col = df.select_dtypes('number')

df_numeric_feat_col = df_numeric_col.drop('ENERGY STAR Score', axis = 1)

ycol = df['ENERGY STAR Score']



df_feat_col = df.select_dtypes('object')

df_feat_col2 = df_feat_col.drop(['Borough', 'Largest Property Use Type'], axis = 1)



# Select the categorical columns

df_cat_col = df[['Borough', 'Largest Property Use Type']]



# One hot encode

df_cat_col = pd.get_dummies(df_cat_col)



# Join the two dataframes using concat

df_v1_full = pd.concat([ycol, df_feat_col2, df_numeric_feat_col, df_cat_col], axis = 1)
df_v1_full.dtypes
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder



le = LabelEncoder()

df_v1_full['n_Postal Code'] = le.fit_transform(df_v1_full['Postal Code'])

df_v1_full['n_Parent Property Id'] = le.fit_transform(df_v1_full['Parent Property Id'])

df_v1_full['n_Property Id'] = le.fit_transform(df_v1_full['Property Id'])

df_v1_full['n_Tax Lot'] = le.fit_transform(df_v1_full['Tax Lot'])

df_v1_full['n_Tax Block'] = le.fit_transform(df_v1_full['Tax Block'])

df_v1_full['n_3rd Largest Property Use Type'] = le.fit_transform(df_v1_full['3rd Largest Property Use Type'])

df_v1_full['n_2nd Largest Property Use Type'] = le.fit_transform(df_v1_full['2nd Largest Property Use Type'])

df_v1_full['n_Primary Property Type - Self Selected'] = le.fit_transform(df_v1_full['Primary Property Type - Self Selected'])



#oe = OrdinalEncoder() -- discretize!!

df_v1_full['n_Year Built'] = le.fit_transform(df_v1_full['Year Built'])
selection = ['Postal Code','Parent Property Id','Property Id','Tax Lot','Tax Block',

        '3rd Largest Property Use Type','2nd Largest Property Use Type',

        'Primary Property Type - Self Selected','Year Built']



df_v1_full = df_v1_full.drop(selection, axis = 1)
from sklearn.preprocessing import StandardScaler,Normalizer,FunctionTransformer,QuantileTransformer,PowerTransformer



df_v2 = df_v1_full.copy()



y_val = df_v2.copy()



select = ['ENERGY STAR Score',

        'Number of Buildings - Self-reported','Occupancy',

        'Site EUI (kBtu/ft²)','Source EUI (kBtu/ft²)','Total GHG Emissions (Metric Tons CO2e)',

        'Water Intensity (All Water Sources) (gal/ft²)','Weather Normalized Site Electricity Intensity (kWh/ft²)',

        'Weather Normalized Site Natural Gas Intensity (therms/ft²)','Largest Property Use Rate',

        '2nd Property Use Rate','3rd Property Use Rate','Direct GHG Emissions Rate'

        ]



df_v2_part = df_v2[select]



scaler = StandardScaler()

normal = Normalizer()

log1p = FunctionTransformer(np.log1p),

qtnormal = QuantileTransformer(output_distribution='normal')

jtrans =PowerTransformer(method='yeo-johnson')

#boxcox = PowerTransformer(method='box-cox')



df_v2_part_normalized = pd.DataFrame(scaler.fit_transform(df_v2_part))

df_v2_part_transformed = pd.DataFrame(jtrans.fit_transform(df_v2_part_normalized))

df_v2_part_transformed.columns = select

df_v2[select] = df_v2_part_transformed
df_test_y = y_val.loc[y_val['Source'] == 'Train']

df_test_y.drop('Source', axis = 1, inplace = True)

y_val_fin = df_test_y['ENERGY STAR Score']
df_base = df_v2 ### <-

df_train = df_base.loc[df_base['Source'] == 'Train']

df_test = df_base.loc[df_base['Source'] == 'Test']

df_train.drop('Source', axis = 1, inplace = True)

df_test.drop('Source', axis = 1, inplace = True)
y = df_train['ENERGY STAR Score']

X = df_train.drop('ENERGY STAR Score', axis=1)



X_TEST = df_test.drop('ENERGY STAR Score', axis=1)
from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(X, y_val_fin, test_size = 0.3, random_state = 100)

y_not_norm = y_test



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)



#X_test=X_TEST #  <---  
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error



#gradient_boosted = GradientBoostingRegressor() #  <---  

gradient_boosted=GradientBoostingRegressor(loss='lad', max_depth=5, #  <---  

                          max_features=None,

                          min_samples_leaf=6,

                          min_samples_split=6,

                          n_estimators=500)



gradient_boosted.fit(X_train, y_train)

predictions = gradient_boosted.predict(X_test)

mae = mean_absolute_error(y_test, predictions)



print('Gradient Boosted Performance on the test set: MAE = %0.4f' % mae)

#Gradient Boosted Performance on the test set: MAE = 8.3002 df_v1_full

#Gradient Boosted Performance on the test set: MAE = 9.6863 df_v2_full

#Gradient Boosted Performance on the test set: MAE = 9.9390 df_V2_red

#Gradient Boosted Performance on the test set: MAE = 0.2812 df_v2_full  jtrans + scaler
#gradient_boosted=GradientBoostingRegressor(loss='lad', max_depth=5,

#                          max_features=None,

#                          min_samples_leaf=6,

#                          min_samples_split=6,

#                          n_estimators=500)



#gradient_boosted.fit(X_train, y_train)



#predictions = gradient_boosted.predict(X_TEST)
#X_TEST['ENERGY STAR Score'] = predictions  #  <---  

#new_x = X_TEST[select]                     #  <---  



X_test['ENERGY STAR Score'] = predictions #  <---  

new_x = X_test[select]                    #  <---  



new_transformed = pd.DataFrame(jtrans.inverse_transform(new_x))

new2 = pd.DataFrame(scaler.inverse_transform(new_transformed))

predictions2 = new2[0]

predictions3 = [0 if item < 0 else 100 if item > 100 else round(item,0) for item in predictions2]

#predictions2 = [0 if item < 0 else 100 if item > 100 else round(item,0) for item in predictions]



result = predictions3
mae_1 = mean_absolute_error(y_test, predictions)

mae_2 = mean_absolute_error(y_not_norm, result)

print('Gradient Boosted Performance with output transformed and normalized on the test set: MAE = %0.4f' % mae_1)

print('Gradient Boosted Performance with output converted back to original format on the test set: MAE = %0.4f' % mae_2)
#sub_file = []

#sub_file = pd.DataFrame(test_ID2)

#sub_file['score'] = result

#sub_file.to_csv('submission_v2.csv', index=False)