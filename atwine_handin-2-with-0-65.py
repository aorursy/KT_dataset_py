import matplotlib.pyplot as plt #this is for visualization
import numpy as np #this is for matrix calculation and maths
import pandas as pd #this is for data manipulation
import seaborn as sns #this is also to help in visualization 
# import featuretools as ft #used for auto feature generation where we have many datasets
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split #to help in spliting the data
from sklearn.metrics import mean_squared_error #this is to test the accuracy of the model
from sklearn import cross_validation #to help in feature selection

# I need to install xgboost as my ML method and other parameters I may have forgotten
# I am going to need feature tools since I have many dataframes so that I can create more features

# Suppress warnings 
import warnings 
warnings.filterwarnings('ignore')

df_test = pd.read_csv('../input/application_test.csv')
df_train = pd.read_csv('../input/application_train.csv')
df_bureau = pd.read_csv('../input/bureau_balance.csv')
credit_card_bal = pd.read_csv('../input/credit_card_balance.csv')
instalments = pd.read_csv('../input/installments_payments.csv')
POS_cash = pd.read_csv('../input/POS_CASH_balance.csv')
#how big are our data sets?
df_bureau.shape,df_test.shape,df_train.shape,credit_card_bal.shape,instalments.shape,POS_cash.shape
#I want to look at the variables in the train dataset because it is the most important dataset
df_train.describe()
#how many empty values do we have
df_train.isnull().any().sum()
df_train['TARGET'].value_counts()
#let us look at the distribution of the target
df_train['TARGET'].plot.hist()
#so we see here that there are more defaults than those that have paid their loans back
#check for missing values:
def miss_val(df):
    #the number of missing values
    val_miss = df.isnull().sum()
    
    #percentage of the missing values
    perc_miss = 100* df.isnull().sum()/len(df)
    
    #put the two together to form a table
    miss_table = pd.concat([val_miss,perc_miss], axis= 1)
    
    #rename the columns
    fin_mis_table = miss_table.rename(columns = {0:'Missing Values', 1: 'Percentage'})
    
    #sort the values in descending order
    final_table = fin_mis_table[fin_mis_table.iloc[:,1]!=0].sort_values('Percentage', ascending = False ).round(2)
    
    return final_table
miss_val(df_train).head(20)
#how many data types do we have
df_train.dtypes.value_counts()
#we have 16 data types that are non numeric
#lets see them
df_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
#since we have many values that are categorical we need to encode them because they are not easily handled

#create the encoder object
# labE = LabelEncoder()
# le_count = 0 #to keep track of the encoded

# for col in df_train:
#     if df_train[col].dtype == 'object':
#         #we want to encode the labels with fewer labels
#         if len(list(df_train[col].unique())) <= 2:
#             #train on the data
#             labE.fit(df_train[col])
#             #transform all the dataframes.
#             df_test[col] = labE.transform(df_test[col])
#             df_train[col]= labE.transform(df_train[col])
            
#             le_count += 1
            
# print('Were transformed', le_count)

#one hot encoding
df_test = pd.get_dummies(df_test)
df_train = pd.get_dummies(df_train)

print('Shape', df_test.shape)
print('Shape', df_train.shape)
#one hot encoding creates many more columns and so the dataframes are not aligned 
#let's align the datasets

#first we take out the target column
target_label = df_train['TARGET']

df_train,df_test = df_train.align(df_test,axis=1,join='inner')

#replace the target column
df_train['TARGET'] = target_label

print('Test Shape',df_test.shape)
print('Train Train', df_train.shape)
#there is an anomaly in the number of days worked, the days are so many so we will replace them
#CREATE A FLAG COLUMN FOR THE ANOMALY DAYS
df_train['DAYS_EMPLOYED_ANOMALY'] = df_train['DAYS_EMPLOYED']== 365243

#replace the days which are abnormal
df_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

df_train['DAYS_EMPLOYED'].plot.hist()
#from numpy we have a correlation function, using pearson's correlation
correlation = df_train.corr()['TARGET'].sort_values()

print('Least correlated', correlation.tail(10))
print('Most correlated', correlation.head(10))
#there is a high correlation between target and the days birth
df_train['DAYS_BIRTH'] = abs(df_train['DAYS_BIRTH'])
df_train['DAYS_BIRTH'].corr(df_train['TARGET'])
# let's plot the fig and see how it looks like
plt.style.use('fivethirtyeight')

plt.hist(df_train['DAYS_BIRTH']/365, bins=10, color= 'blue', edgecolor = 'k')
#first we create a list of the variables
poly_train = df_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_test = df_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

#to handle missing values we use the imputer to fill them in based on the mean
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')

#we take the target column out because we don't want to add anything
poly_target = poly_train['TARGET']

#lets drop it from the poly_train df
poly_train = poly_train.drop(columns = ['TARGET'])

#now lets impute the values into the dataframes
poly_train = imputer.fit_transform(poly_train)
poly_test = imputer.transform(poly_test)

#lets now bring in the polynomial feature creator
#we will create the features to the 4 degree to prevent over fitting

from sklearn.preprocessing import PolynomialFeatures

#we create the polynomial feature object
poly_creator = PolynomialFeatures(degree=4)

#we fit the poly features/ train the features on the training data
poly_creator.fit(poly_train)


#now we need to  in put the data frames created
poly_train_ft = poly_creator.transform(poly_train)
poly_test_ft = poly_creator.transform(poly_test)

print('Poly Features Shape:', poly_train_ft.shape)
print('Poly Features Shape:', poly_test_ft.shape)
poly_creator.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:20]
poly_df = pd.DataFrame(poly_train_ft)
#return the target column into the data created
poly_df['TARGET']= poly_target

#lets now look at the features correlation
poly_corr = poly_df.corr()['TARGET'].sort_values()

#display the top 10 and bottom 10
print(poly_corr.head(10))
print('-'*20)
print(poly_corr.tail(10))
#we want a data frame from the info above
poly_df_ft = pd.DataFrame(poly_test_ft, columns = poly_creator.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
'SK_ID_CURR' in df_train.columns
type(poly_train)

#lets merge the training data frame with the features
poly_df_ft['SK_ID_CURR'] = df_train['SK_ID_CURR']
train_poly = df_train.merge(poly_df_ft, on = 'SK_ID_CURR', how = 'left')

#lets join the testing data witht he poly features
poly_df_ft['SK_ID_CURR']= df_test['SK_ID_CURR']
test_poly = df_test.merge(poly_df_ft, on = 'SK_ID_CURR', how = 'left')

train_poly,test_poly = train_poly.align(test_poly, join = 'inner', axis = 1)

print('Shape of the training data:',train_poly.shape)
print('Shape of the testing data:',test_poly.shape)
#lets get the polynomial feature names
poly_feature_names = list(train_poly.columns)

#impute the polynomial features
train_poly = imputer.fit_transform(train_poly)
test_poly = imputer.transform(test_poly)

#scale the features so they can be in the range of 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range= (0,1))

train_poly = scaler.fit_transform(train_poly)
test_poly = scaler.transform(test_poly)

from sklearn.ensemble import RandomForestClassifier

poly_classiffier = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)


poly_classiffier.fit(train_poly,target_label)
predictions = poly_classiffier.predict_proba(test_poly)[:,1]
submit = df_test[['SK_ID_CURR']]
submit['TARGET']= predictions
submit.shape
# submit.to_csv('Random_Forest_HandIn.csv',index= False)
