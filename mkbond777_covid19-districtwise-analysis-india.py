# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', 100)
pd.set_option('max_info_columns',200)
covid19_district_data = pd.read_csv("https://api.covid19india.org/csv/latest/districts.csv")
covid19_district_data.info()
covid19_district_data['District'].nunique()
covid19_district_data.head()
from datetime import datetime, timedelta
today = datetime.today().strftime("%Y-%m-%d")
covid_19_district_agg = covid19_district_data[(covid19_district_data['Date'] == today)][['District','Confirmed','Recovered','Deceased','Other','Tested']]
covid_19_district_agg.head()
covid_19_district_agg['Confirmed'].sum()
census_2011 = pd.read_csv("../input/india-census/india-districts-census-2011.csv",engine='python')
census_2011.head()
old_dict = {"Ahmedabad":"Ahmadabad","Ahmednagar":"Ahmadnagar","Amroha":"Jyotiba Phule Nagar","Angul":"Anugul","Ayodhya":"Faizabad","Bagalkote" :"Bagalkot","Balasore":"Baleshwar","Ballari":"Bellary","Banaskantha":"Banas Kantha","Bandipora":"Bandipore","Barabanki":"Bara Banki","Baramulla":"Baramula","Beed":"Bid","Belagavi":"Belgaum","Bengaluru Rural":"Bangalore Rural","Bengaluru Urban":"Bangalore","Bhadohi":"Sant Ravidas Nagar (Bhadohi)","Boudh":"Baudh","Budgam":"Badgam","Buldhana":"Buldana","Chamarajanagara":"Chamarajanagar","Dadra and Nagar Haveli":"Dadra AND Nagar Haveli","Dahod":"Dohad","Dang":"The Dangs","Darjeeling":"Darjiling","Delhi":"New Delhi","Deogarh":"Deoghar","Dholpur":"Dhaulpur","East Champaran":"Purba Champaran","East Sikkim":"East District","East Singhbhum":"Purbi Singhbhum","Ferozepur":"Firozpur","Gondia":"Gondiya","Gurugram":"Gurgaon","Haridwar":"Hardwar","Hathras":"Mahamaya Nagar","Hooghly":"Hugli","Howrah":"Haora","Jagatsinghpur":"Jagatsinghapur","Jajpur":"Jajapur","Jalore":"Jalor","Janjgir Champa":"Janjgir - Champa","Jhunjhunu":"Jhunjhunun","Kaimur":"Kaimur (Bhabua)","Kalaburagi":"Gulbarga","Kanyakumari":"Kanniyakumari","Kasganj":"Kanshiram Nagar","Khandwa":"Khandwa (East Nimar),","Khargone":"Khargone (West Nimar)","Koderma":"Kodarma","Kutch":"Kachchh","Lahaul and Spiti":"Lahul AND Spiti","Lakhimpur Kheri":"Kheri","Leh":"Leh(Ladakh)","Maharajganj":"Mahrajganj","Malda":"Maldah","Mehsana":"Mahesana","Mysuru":"Mysore","Narsinghpur":"Narsimhapur","Nilgiris":"The Nilgiris","North 24 Parganas":"North Twenty Four Parganas","North Sikkim":"North District","Nuh":"Mewat","Panchmahal":"Panch Mahals","Pauri Garhwal":"Garhwal","Prayagraj":"Allahabad","Puducherry":"PONDICHERRY","Purulia":"Puruliya","Raigad":"Raigarh","S.A.S. Nagar":"Sahibzada Ajit Singh Nagar","S.P.S. Nellore":"Sri Potti Sriramulu Nellore","Sabarkantha":"Sabar Kantha","Shivamogga":"Shimoga","Shopiyan":"Shupiyan","South 24 Parganas":"South Twenty Four Parganas","South Sikkim":"South District","Sri Muktsar Sahib":"Muktsar","Tengnoupal":"Chandel","Tumakuru":"Tumkur","Vijayapura":"Bijapur","West Champaran":"Pashchim Champaran","West Sikkim":"West District","West Singhbhum":"Pashchimi Singhbhum","Y.S.R. Kadap":"Y.S.R"}
new_dict = dict([(value, key) for key, value in old_dict.items()]) 
census_2011['District name'].replace(new_dict,inplace=True)
final_df = pd.merge(census_2011, covid_19_district_agg, left_on=['District name'], right_on=['District'],how='inner')
final_df.drop(['District code','State name','District'],axis=1,inplace=True)
final_df.drop(['Recovered','Deceased','Other','Tested'],axis=1,inplace=True)
final_df.head()
final_df.shape
sns.distplot(final_df['Confirmed'],)
pd.Series(final_df['Confirmed']).skew()
transform_confirmed_case = np.log(final_df['Confirmed']+1)

pd.Series(transform_confirmed_case).skew()
sns.distplot(transform_confirmed_case)
final_df['Confirmed'] = np.log(final_df['Confirmed']+1)
df = final_df.drop('District name',axis=1)
final_df.head()
for col in df.columns[:-1]:
    df[col] = np.log(df[col]+1)
np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
df_train.shape
scale_cols = [col for col in df_train.columns if ((df_train[col].max()) + (df_train[col].min())) > 1]
len(scale_cols)
df_train[scale_cols].describe()
scaler = MinMaxScaler()
df_train[scale_cols] = scaler.fit_transform(df_train[scale_cols])
df_train.describe()
y_train = df_train.pop('Confirmed')
X_train = df_train
X_train.shape
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 10)             # running RFE
rfe = rfe.fit(X_train, y_train)
col = X_train.columns[rfe.support_]
col
X_train_rfe = X_train[col]
# linear regression
lm = LinearRegression()
lm.fit(X_train_rfe, y_train)

# predict
y_train_pred = lm.predict(X_train_rfe)
round(metrics.r2_score(y_true=y_train, y_pred=y_train_pred),2)
df_test[scale_cols] = scaler.transform(df_test[scale_cols])
y_test = df_test.pop('Confirmed')
X_test = df_test[col]
X_test.describe()
y_pred = lm.predict(X_test)
r_squared = metrics.r2_score(y_test, y_pred)
round(r_squared,2)

model_parameters = list(lm.coef_)
model_parameters.insert(0, lm.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X_train_rfe.columns
cols = cols.insert(0, "constant")
final_list = [i for i in list(zip(cols, model_parameters)) if i[1] != 0]
sorted(final_list, key = lambda x: x[1]) 
X_train_rfe.shape
X_test.shape
X_train_rfe.describe()
# set up cross validation scheme
l_folds = KFold(n_splits = 5, shuffle = True, random_state = 101)

# specify range of hyperparameters
l_params = {'alpha': [0.0001,0.0004,0.0005,0.0008,0.001,0.01, 1.0, 5.0, 10.0]}

# grid search
# lasso model
l_model = Lasso(max_iter=1000000)
l_model_cv = GridSearchCV(estimator = l_model, param_grid = l_params, 
                        scoring= 'r2', 
                        cv = l_folds, 
                        return_train_score=True,
                          verbose = 1)            
l_model_cv.fit(X_train_rfe, y_train) 
l_cv_results = pd.DataFrame(l_model_cv.cv_results_)
l_cv_results['test_train_diff'] = l_cv_results['mean_train_score'] - l_cv_results['mean_test_score']

l_cv_results[['param_alpha','mean_test_score','mean_train_score','test_train_diff']]
# plot
l_cv_results['param_alpha'] = l_cv_results['param_alpha'].astype('float32')
plt.plot(l_cv_results['param_alpha'], l_cv_results['mean_train_score'])
plt.plot(l_cv_results['param_alpha'], l_cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('r2 score')
plt.xscale('log')
plt.show()
l_model_cv.best_estimator_.alpha
from time import time
lm_lasso = Lasso(alpha=l_model_cv.best_estimator_.alpha,max_iter=1000000)
t0=time()
lm_lasso.fit(X_train_rfe, y_train)
print ("training time:", round(time()-t0, 3), "s")
# predict
y_train_pred = lm_lasso.predict(X_train_rfe)
print("train accuracy:",round(metrics.r2_score(y_true=y_train, y_pred=y_train_pred),2))
t1=time()
y_test_pred = lm_lasso.predict(X_test)
print ("predict time:", round(time()-t1, 3), "s")
print("test accuracy:",round(metrics.r2_score(y_true=y_test, y_pred=y_test_pred),2))
# lasso model parameters
model_parameters = list(lm_lasso.coef_)
model_parameters.insert(0, lm_lasso.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X_train_rfe.columns
cols = cols.insert(0, "constant")
final_list = [i for i in list(zip(cols, model_parameters)) if i[1] != 0]
len(final_list)
sorted(final_list, key = lambda x: x[1])
# set up cross validation scheme
folds = KFold(n_splits = 5, shuffle = True, random_state = 105)

# specify range of hyperparameters
params = {'alpha': [0.0001,0.0004,0.0005,0.0008,0.001,0.01, 1.0, 5.0, 10.0,50.0,100.0]}

# grid search
# lasso model
model = Ridge()
model_cv = GridSearchCV(estimator = model, param_grid = params, 
                        scoring= 'r2', 
                        cv = folds, 
                        return_train_score=True, verbose = 1)            
model_cv.fit(X_train_rfe, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results['test_train_diff'] = cv_results['mean_train_score'] - cv_results['mean_test_score']

cv_results[['param_alpha','mean_test_score','mean_train_score','test_train_diff']]
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('r2 score')
plt.title("r2 score and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()
model_cv.best_estimator_.alpha
#alpha = 1000
ridge = Ridge(alpha=model_cv.best_estimator_.alpha)

ridge_lm = ridge.fit(X_train_rfe, y_train)
# predict
y_train_pred = ridge_lm.predict(X_train_rfe)
print(round(metrics.r2_score(y_true=y_train, y_pred=y_train_pred),2))
y_test_pred = ridge_lm.predict(X_test)
print(round(metrics.r2_score(y_true=y_test, y_pred=y_test_pred),2))