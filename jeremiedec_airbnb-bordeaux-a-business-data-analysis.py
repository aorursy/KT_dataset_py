from IPython.core.display import display, HTML

import IPython.display

import pandas as pd

import numpy as np

from sklearn.preprocessing import *

listings = pd.read_csv('../input/listings_detail.csv')

listings_raw = pd.read_csv('../input/listings_detail.csv')
listings.shape

listings.columns
# output of the mean avaibility over 30 days of all the listings by neighourhood. It can be computed for avaibility_60, avaibility_90 in the same fashion.

le = LabelEncoder()

neigh_encoded = le.fit_transform(listings_raw['neighbourhood'].astype(str))

neigh_encoded  = pd.DataFrame(neigh_encoded)

neigh_encoded.columns = ['neigh_encoded']

listings_raw_enc = pd.merge(listings_raw, neigh_encoded, left_index=True, right_index=True)

for i in range(0,13,1): # I don't include index 13, these are NaNs encoded values

    avaib_mean = [None]*15

    avaibility = [None]*15

    avab = [None]*15

    avaibility[i] = listings_raw_enc[listings_raw_enc.neigh_encoded==i]

    avaib_mean[i] = avaibility[i].availability_30

    avab[i] = np.around(np.nanmean(avaib_mean[i]), 0)

    number = i

    labels_integers = {l: i for i, l in enumerate(le.classes_)}

    lab = list(labels_integers)

    print(avab[i],"days over the next 30 days are available for rent at", lab[i])

    

import numpy as np
a = listings_raw.availability_30.describe()# 75% of properties are rented for a minimum of 21 days in the next 30 >>> 70% occupancy

print(a.index[6],"of properties are  rented for a maximum of", 30-a[6],"days in the next 30. That makes an (optimistic) assumption of", np.round((30-a[6])/0.3, 1),"% occupancy")
a = listings_raw.availability_60.describe() # 58.3%

print(a.index[6],"of properties are  rented for a maximum of", 60-a[6],"days in the next 60. That makes an (optimistic) assumption of", np.round((60-a[6])/0.6, 1),"% occupancy")
a =listings_raw.availability_90.describe() # 48.9%

print(a.index[6],"of properties are  rented for a maximum of", 90-a[6],"days in the next 90. That makes an (optimistic) assumption of", np.round((90-a[6])/0.9, 1),"% occupancy")
a = listings_raw.availability_365.describe()  # 51%

print(a.index[6],"of properties are  rented for a maximum of", 365-a[6],"days in the next 365. That makes an (optimistic) assumption of", np.round((365-a[6])/3.65, 1),"%")
listings_raw.reviews_per_month.describe()

rev_excell = listings_raw[listings_raw.reviews_per_month==15]

print("There are", rev_excell.shape[0], "host(s) with at least 15 reviews per month")


print("The studio is  rented for a maximum of", 30-rev_excell.availability_30.mean(),"days in the next 30. That makes an (optimistic) assumption of", np.round((30-rev_excell.availability_30.mean())/0.3, 1),"% occupancy rate")

print("The studio is  rented for a maximum of", 60-rev_excell.availability_60.mean(),"days in the next 60. That makes an (optimistic) assumption of", np.round((60-rev_excell.availability_60.mean())/0.6, 1),"% occupancy rate")

print("The studio is  rented for a maximum of", 90-rev_excell.availability_90.mean(),"days in the next 60. That makes an (optimistic) assumption of", np.round((90-rev_excell.availability_90.mean())/0.9, 1),"% occupancy rate")

print("The studio is  rented for a maximum of", 365-rev_excell.availability_365.mean(),"days in the next 365. That makes an (optimistic) assumption of", np.round((365-rev_excell.availability_365.mean())/3.65, 1),"% occupancy rate")
rev_excell.listing_url
%matplotlib inline

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
listings.columns

listings.price.head()

print(listings.shape)
def isnull(X):

    listings_na = pd.isnull(X)

    listings_na = listings_na*1

    listings_na = (listings_na==1).sum()/listings_na.shape[0]*100

    listings_na = pd.DataFrame(listings_na.T.sort_index(ascending=True))

    listings_na = listings_na[listings_na>0.0001] # je garde uniquement les colonnes dont le % de NaNs (non-attribuées) est défini 

    listings_na = listings_na.dropna()

    return listings_na



isnull(listings)
import missingno as msno



missingdata_df = listings_raw.columns[listings_raw.isnull().any()].tolist()

msno.heatmap(listings_raw[missingdata_df], figsize=(20,20))
listings.reviews_per_month  = listings.reviews_per_month.fillna(0)
listings.price.head()
a = []

for i in listings.price:

    if pd.isnull(i):

        a.append(i)

    else:

        b = str(i)[1:]

        b = b.replace(',','')

        b = float(b)

        c = b*0.88 

        a.append(c)

        

listings.price = a



a = []

for i in listings.weekly_price:

    if pd.isnull(i):

        a.append(i)

    else:

        b = str(i)[1:] #removes the first character '$'

        b = b.replace(',','') # replaces ',' value as 135,00 to 'empty' in order to be able to convert it then 

        b = float(b)

        c = b*0.88

        a.append(b)

listings.weekly_price = a



a = []



for i in listings.monthly_price:

    if pd.isnull(i):

        a.append(i)

    else:

        b = str(i)[1:]

        b = b.replace(',','')

        b = float(b)

        a.append(b)

        c = b*0.88

listings.monthly_price = a



a = []

for i in listings.cleaning_fee:

    if pd.isnull(i):

        a.append(i)

    else:

        b = str(i)[1:]

        b = b.replace(',','')

        b = float(b)

        c = b*0.88

        a.append(b)

        

listings.cleaning_fee = a



a = []

for i in listings.extra_people:

    if pd.isnull(i):

        a.append(i)

    else:

        b = str(i)[1:]

        b = b.replace(',','')

        b = float(b)

        c = b*0.88

        a.append(b)



listings.extra_people = a



a = []

for i in listings.security_deposit:

    if pd.isnull(i):

        a.append(i)

    else:

        b = str(i)[1:]

        b = b.replace(',','')

        b = float(b)

        c = b*0.88

        a.append(b)

listings.security_deposit = a
categorical_cols = ['room_type', 'host_since', 'property_type','cancellation_policy'] 



le = LabelEncoder()



listings_encoded = listings[categorical_cols].apply(lambda col: le.fit_transform(col))

neigh_encoded = le.fit_transform(listings['neighbourhood'].astype(str))

neigh_encoded  = pd.DataFrame(neigh_encoded)

neigh_encoded.columns = ['neighbourhood']



listings_num = listings.select_dtypes(include = ['float64', 'int64']) 



listings = pd.merge(listings_encoded, listings_num, left_index=True, right_index=True) 

listings = pd.merge(listings, neigh_encoded, left_index=True, right_index=True)





listings  = listings.drop(columns = ['longitude', 'latitude','id', 'host_id'])

li = listings.index[listings.price>800].tolist()

listings = listings.drop(index = li)

listings = listings.reset_index(drop=True)

listings = listings.fillna(listings.mean())


# I temporarily enter cleaning_fee and security_deposit by their respective mean

listings = listings.fillna(listings.mean())
listings_for_knn_imputation = listings

listings = listings.dropna(axis=0, subset = ['review_scores_location'])

listings = listings.dropna(axis=0, subset = ['review_scores_accuracy'])

listings = listings.dropna(axis=0, subset = ['review_scores_checkin'])

listings = listings.dropna(axis=0, subset = ['review_scores_communication'])

listings = listings.dropna(axis=0, subset = ['review_scores_value'])

listings = listings.dropna(axis=0, subset = ['review_scores_cleanliness'])

listings = listings.dropna(axis=0, subset = ['review_scores_rating'])

listings = listings.reset_index()

#del listings['index', 'thumbnail_url', 'xl_picture_url', 'host_acceptance_rate', 'medium_url', 'scrape_id', 'square_feet']

#More than 95% of the surfaces in m2 are missing. After verification, some hosts mention the area of the property in the descriptions (str variables): these could be determined.
listings = listings.dropna(axis=0, subset = ['bathrooms'])

listings = listings.dropna(axis=0, subset = ['bedrooms'])

listings = listings.dropna(axis=0, subset = ['beds'])

listings = listings[listings['beds'] != 0] 

listings = listings[listings['bedrooms'] != 0]

listings = listings[listings['price'] != 0]

del listings['thumbnail_url'] 

del listings['xl_picture_url']

del listings['host_acceptance_rate']

del listings['square_feet'] #  more than 95% missing

del listings['medium_url']

del listings['scrape_id']

listings = listings.drop(index = 5843) #outlier removal

listings = listings.reset_index(drop=True)

%matplotlib inline

import os

import numpy as np

import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt

from scipy.stats import norm, skew

from sklearn.feature_selection import f_regression

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression as Lin_Reg

from sklearn import metrics

import xgboost as xgb

import warnings

def ignore_warn(*args, **kwargs):

    pass
listings = listings.drop(index = [64, 743, 5842])

listings = listings.reset_index(drop=True)



listings = listings.fillna(listings.mean())

Y = listings.price

print(Y.shape)

del(listings['weekly_price']) # removal of the leaked variable 'price': It would not normally be possessed in a real situation

del(listings['monthly_price']) # removal of the leaked variable 'price': It would not normally be possessed in a real situation
columns = listings.columns.tolist()



p_values = f_regression(listings, Y)[1]

p_valuesdf = pd.DataFrame(p_values, index = listings.columns)



p_valuesdf.sort_values(by = 0, ascending=True)
import seaborn as sns

sns.kdeplot(listings['price'] , clip= (0.0, 800))



fig = plt.figure()

res = stats.probplot(listings['price'], plot=plt)

plt.show()
sns.kdeplot(np.log(listings['price']) , clip= (0.0, 800))



fig = plt.figure()

res = stats.probplot(np.log(listings['price']), plot=plt)

plt.show()
class model:



    def __init__(self, model):

        self.model = model

        self.x_train = None

        self.y_train = None

        self.x_test = None

        self.y_test = None

        self.y_pred_train = None

        self.y_pred_test = None

        self.train_score = None

        self.test_score = None

        self.train_score_log = None

        self.test_score_log = None

        self.train_score_mae = None

        self.test_score_mae = None

        self.train_score_mae1 = None

        self.test_score_mae1 = None

        self.train_score_m_ae_unlog = None 

        self.test_score_m_ae_unlog = None

        self.train_score_mae_unlog = None

        self.test_score_mae_unlog = None



    def data_split(self, x, y, test_size):

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size)



    def score_reg(self):

        return self.train_score, self.test_score

        

    def score_mean_abs_err(self):

        self.train_score_mae = metrics.mean_absolute_error(self.y_pred_train, self.y_train)

        self.test_score_mae = metrics.mean_absolute_error(self.y_test, self.y_pred_test)

        return self.train_score_mae, self.test_score_mae



    def score_median_abs_err(self):

      self.train_score_mae1 = metrics.median_absolute_error(self.y_pred_train, self.y_train)  

      self.test_score_mae1 = metrics.median_absolute_error(self.y_test, self.y_pred_test)

      return self.train_score_mae1, self.test_score_mae1



    def score_log(self):

        self.train_score_log = metrics.r2_score(np.exp(self.y_train), np.exp(self.y_pred_train))

        self.test_score_log = metrics.r2_score(np.exp(self.y_test), np.exp(self.y_pred_test))

        return self.train_score_log, self.test_score_log



    def score_mean_log(self):

       self.train_score_m_ae_unlog = metrics.mean_absolute_error(np.exp(self.y_pred_train), np.exp(self.y_train))

       self.test_score_m_ae_unlog = metrics.mean_absolute_error(np.exp(self.y_test), np.exp(self.y_pred_test))

       return self.train_score_m_ae_unlog, self.test_score_m_ae_unlog



    def score_median_log(self):

        self.train_score_mae_unlog = metrics.median_absolute_error(np.exp(self.y_pred_train), np.exp(self.y_train))

        self.test_score_mae_unlog = metrics.median_absolute_error(np.exp(self.y_test), np.exp(self.y_pred_test))

        return self.train_score_mae_unlog, self.test_score_mae_unlog



    def data_frame_convert(self):

        df_train = pd.DataFrame({'y_pred': self.y_pred_train, 'y_real': self.y_train})

        df_test = pd.DataFrame({'y_pred_test': self.y_pred_test, 'y_real_test': self.y_test})

        return self.train_score, self.test_score, df_train, df_test



    def data_frame_convert_log(self):

        df_train = pd.DataFrame({'y_pred': np.exp(self.y_pred_train), 'y_real': np.exp(self.y_train)})

        df_test = pd.DataFrame({'y_pred_test': np.exp(self.y_pred_test), 'y_real_test': np.exp(self.y_test)})

        return self.train_score_log, self.test_score_log, df_train, df_test



    def fit_model(self, x, y, test_size):

        self.data_split(x, y, test_size)

        self.model = self.model.fit(self.x_train, self.y_train)

        self.train_score = self.model.score(self.x_train, self.y_train)

        self.test_score = self.model.score(self.x_test, self.y_test)

        self.y_pred_train = self.model.predict(self.x_train)

        self.y_pred_test = self.model.predict(self.x_test)



def model_iterations(n, x, y, model_arg, log_bool=False):

    training_scores = [None]*n

    testing_scores = [None]*n

    training_scores_mean_ae = [None]*n

    testing_scores_mean_ae = [None]*n

    training_scores_mae = [None]*n

    testing_scores_mae = [None]*n





    for i in range(n):

        new_model = model(model_arg)

        new_model.fit_model(x, y, 0.7)

        training_scores[i], testing_scores[i] = new_model.score_reg() if not log_bool else new_model.score_log()

        training_scores_mean_ae[i], testing_scores_mean_ae[i]  = new_model.score_mean_abs_err() if not log_bool else new_model.score_mean_log()

        training_scores_mae[i], testing_scores_mae[i] = new_model.score_median_abs_err() if not log_bool else new_model.score_median_log()



    print('-Best R2 training', np.max(training_scores))

    print('-Best R2 testing', np.max(testing_scores))

    print('-Avg R2 training', np.mean(training_scores))

    print ('-Avg R2 testing',np.mean(testing_scores))

    print ('Training mean score (_mean absolute error)', np.mean(training_scores_mean_ae))

    print ('Testing mean score (_mean absolute error)', np.mean(testing_scores_mean_ae))



    print ('Training best score (_mean absolute error)', np.min(training_scores_mean_ae))

    print ('Training best score (median absolute error)', np.min(training_scores_mae))



    print ('Testing best score (_mean absolute error)', np.min(testing_scores_mean_ae))

    print ('Testing best score (median absolute error)', np.min(testing_scores_mae))

    print ('std -ecarts moyens des perfs testing', np.std(testing_scores_mae))

    print ('std des perfs training', np.std(training_scores_mae))



    return new_model
def plot_residual(ax1, ax2, ax3, y_pred, y_real, line_label, title):

    ax1.scatter(y_pred,

                y_real,

                color='blue',

                alpha=0.6,

                label=line_label)

    ax1.set_xlabel('Predicted Y')

    ax1.set_ylabel('Real Y')

    ax1.legend(loc='best')

    ax1.set_title(title)



    ax2.scatter(y_pred,

                y_real - y_pred,

                color='green',

                marker='x',

                alpha=0.6,

                label='Residual')

    ax2.set_xlabel('Y Prédit')

    ax2.set_ylabel('Residual')



    ax2.axhline(y=0, color='black', linewidth=2.0, alpha=0.7, label='y=0')



    ax2.legend(loc='best')

    ax2.set_title('Residual Graph')



    ax3.hist(y_real - y_pred, bins=30, color='green', alpha=0.7)

    ax3.set_title('Histogram of residual values')



    return ax1, ax2, ax3



def plots(model):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    data_vals = model.data_frame_convert()

    plot_residual(axes[0][0], axes[0][1], axes[0][2], data_vals[2]['y_pred'], data_vals[2]['y_real'], 'model: {}'.format(data_vals[0]), 'Scatter Plot: Y_Predit vs. Y')

    plot_residual(axes[1][0], axes[1][1], axes[1][2], data_vals[3]['y_pred_test'], data_vals[3]['y_real_test'], 'model: {}'.format(data_vals[1]), 'Residual Plot for Test Data')

plt.show()

del(listings['price'])

import xgboost as xgb

gpu_params = {'tree_method':'gpu_hist',

              'predictor':'gpu_predictor',

              'gamma': 1,  

              'learning_rate': 0.01,

              'max_depth': 3,

              'n_estimators': 10000,                                                           

              'random_state': 98,

               }



mod_gpu = xgb.XGBRegressor(**gpu_params)



Xgb_model = model_iterations(1, listings, Y, mod_gpu, log_bool=False)

listings['price'] = Y

listings_prices_inf200 = listings[listings.price<200]

Y_log_prices_inf200 = np.log(listings_prices_inf200.price)

del(listings_prices_inf200['price'])

Xgb_model2 = model_iterations(1, listings_prices_inf200, Y_log_prices_inf200, mod_gpu, log_bool=True)
plots(Xgb_model2)