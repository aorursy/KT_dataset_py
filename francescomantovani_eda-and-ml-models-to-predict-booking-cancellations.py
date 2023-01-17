import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

import seaborn as sns



data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

data.head()

data.shape

data.dtypes

data = data.rename({'is_canceled':'y'}, axis=1)

pd.set_option('display.max_columns', None)

data.describe()

nans = pd.DataFrame(data.isna().sum(), columns=['count'])    #check nans

nans.reset_index(inplace=True)

nans.loc[nans['count'] != 0]
data = data.drop(columns=['company', 

                'agent', 'reservation_status', 'reservation_status_date']) #company, agent for too many nans. Reservation to prevent leakage

data['country'] = data['country'].fillna(value = 'no info')

data['children'] = data['children'].fillna(0)

nans = pd.DataFrame(data.isna().sum(), columns=['count'])                  #check nans

nans.reset_index(inplace=True)

nans.loc[nans['count'] != 0]                                        
#modify variable hotel

data.replace(['Resort Hotel', 'City Hotel'], [1, 0], inplace=True)

data.rename(columns={'hotel':'resort'}, inplace=True)



#delete outliers and errors (adults: it's a series of 12 mistakes, all with common characteristics)

data = data.loc[data['adults'] <= 10]

data = data.loc[data['babies'] <= 3]

data = data.loc[data['adr'] <= 510]

data.reset_index(drop=True, inplace=True)



#remove negative 'adr' (must be a mistake)

data = data.loc[data['adr'] >= 0 ]



#remove obs with 0 adults and 0 children (180 observations)

data = data.loc[(data['adults'] != 0) | (data['children'] != 0)]

data.reset_index(drop=True, inplace=True)



#separate numerical non-bin, numerical and bin vars, and categorical variables (count arrival dates as categorical)

data.arrival_date_year = data.arrival_date_year.astype('object')

data.arrival_date_week_number = data.arrival_date_week_number.astype('object')

data.arrival_date_day_of_month = data.arrival_date_day_of_month.astype('object')

cont_var = data.drop(columns=['y', 'resort', 'is_repeated_guest']).select_dtypes(include='number').columns

num_var = data.drop(columns=['y']).select_dtypes(include='number').columns

cat_var = data.drop(columns=['y']).select_dtypes(include='object').columns



data = data[['y', 'resort', 'lead_time', 'arrival_date_year', 'arrival_date_month',

       'arrival_date_week_number', 'arrival_date_day_of_month',

       'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',

       'babies', 'meal', 'country', 'market_segment', 'distribution_channel',

       'is_repeated_guest', 'previous_cancellations',

       'previous_bookings_not_canceled', 'booking_changes', 'deposit_type',

       'days_in_waiting_list', 'customer_type', 'adr',

       'required_car_parking_spaces', 'total_of_special_requests',

       'reserved_room_type']]
# EXPLORATORY DATA ANALYSIS #

#create a function that labels objects with some specific information

def labeller(kind, values, axis, tot=None, only_minmax=False): 

    if kind == 'bar':

        ax=axis

        all_heights=[]

        for rect in values:

            all_heights.append(rect.get_height())

        all_heights = pd.Series(all_heights)

        for rect in values:

            height = rect.get_height()

            perc = height / all_heights.sum() *100

            per = str(perc.round(1))

            if only_minmax == True:

                if height == all_heights.max():

                    ax.annotate('max:\n' + str(per) + '%', 

                                xy=(rect.get_x() + rect.get_width() / 2, height), 

                                xytext=(0, 5),

                                textcoords="offset points", 

                                ha='center', 

                                va='bottom', 

                                size=11, 

                                bbox=dict(boxstyle="round4", fc="w"))

                elif height == all_heights.min():

                    ax.annotate('min:\n' + str(per) + '%', 

                                xy=(rect.get_x() + rect.get_width() / 2, height), 

                                xytext=(0, 5),

                                textcoords="offset points", 

                                ha='center', 

                                va='bottom', 

                                size=11, 

                                bbox=dict(boxstyle="round4", fc="w"))

            else:    

                ax.annotate(str(per) + '%', 

                            xy=(rect.get_x() + rect.get_width() / 2, height), 

                            xytext=(0, 3),

                            textcoords="offset points", 

                            ha='center', 

                            va='bottom', 

                            size=11)

    elif kind == 'scatter':

        for i, txt in enumerate(values):

            ax=axis

            if only_minmax == True:

                values = pd.Series(values)

                values.reset_index(drop=True, inplace=True)

                if txt == values.max():

                    ax.annotate('high: ' + str(txt) + '€', 

                                (x[i], values[i]), 

                                size=11, 

                                xytext=(-85, -3), 

                                textcoords="offset points", 

                                bbox=dict(boxstyle="rarrow", fc="w"))

                elif txt == values.min():

                    ax.annotate('low: ' + str(txt) + '€', 

                                (x[i], values[i]), 

                                size=11, 

                                xytext=(17, -3), 

                                textcoords="offset points", 

                                bbox=dict(boxstyle="larrow", fc="w"))

            else: ax.annotate(str(txt), 

                              (x[i], values[i]), 

                              size=9, 

                              xytext=(-10, 7), 

                              textcoords="offset points", 

                              bbox=dict(boxstyle="round4",fc="w"))
#How many cancellations??

#separate resort data & count bookings

df = data.copy()

dr = df[['resort', 'y']].loc[df['resort'] == 1]                      

dr = pd.DataFrame(dr['y'].value_counts())                            

dr.reset_index(inplace=True)

dr.rename({'index':'y', 'y':'y_count'},axis=1,inplace=True)

#add rows to identify it's Resort data, separate city data

dr['hotel'] = ['res', 'res']                                             

dc = df[['resort', 'y']].loc[df['resort'] == 0]                       

dc.reset_index(drop=True, inplace=True)                               

dc = pd.DataFrame(dc['y'].value_counts())

dc.reset_index(inplace=True)

dc.rename({'index':'y', 'y':'y_count'},axis=1,inplace=True)

#add rows to identify it's city data

dc['hotel'] = ['city','city']                                         

d = dr.append(dc)       #merge again



labels = ['Beach Resort', 'City Hotel']

cancellations = d['y_count'].loc[d['y'] == 1]             

non_cancellations = d['y_count'].loc[d['y'] == 0]    

city_count = dc['y_count'].sum()                     

resort_count = dr['y_count'].sum()                   



width = 0.35                    #width of each rectangle

x = np.arange(len(labels))      #list for positions



fig, ax = plt.subplots(figsize=(6,6))



rect1 = ax.bar(x - width/2, cancellations, width)         

rect2 = ax.bar(x + width/2, non_cancellations, width)     



#create a function to put labels. It's different from the other

def labeller2(rects, city_tot, res_tot):                  

    for rect in rects:                                    

        height = rect.get_height()                        

        if height == rects[0].get_height():             

            tot = res_tot

        else:

            tot = city_tot

        percentage = height / tot * 100                   

        percentage = str(percentage.round(2)) + '%'

        ax.annotate(percentage, 

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),

                    textcoords="offset points",

                    ha='center', va='bottom', size=13)



labeller2(rect1, city_count, resort_count)              

labeller2(rect2, city_count, resort_count)  



ax.set_xticks(x)

ax.set_xticklabels(labels, fontsize=13)

ax.set_ylim([0,52000])

ax.set_ylabel('Bookings')

ax.set_title('Number of bookings cancelled per Hotel')

ax.legend(['Cancelled', 'Not Cancelled'], fontsize= 12)

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

plt.show()
# repeated guests

d1 = data.copy()

d1 = data[['resort', 'y', 'is_repeated_guest']]

d1r = d1[['is_repeated_guest', 'y']].loc[data['resort'] == 1].groupby('is_repeated_guest').count() 

d1c = d1[['is_repeated_guest', 'y']].loc[data['resort'] == 0].groupby('is_repeated_guest').count()

d1r = d1r['y'] / d1r['y'].sum() * 100

d1r = d1r.round(2)

d1c = d1c['y'] / d1c['y'].sum() * 100

d1c = d1c.round(2)

print('City Hotel:', d1c, '\nBeach Resort:', d1r)
#Where do guests come from?? 

d2 = data.copy()

d2 = d2.loc[d2['y'] == 0]  #only actual guests



#City Hotel data

d2c = d2.loc[d2['resort'] == 0]                  

d2c = d2c.country.value_counts()

d2c_top = d2c[0:12]

d2c_bottom = d2c[13:].sum()

d2c_bottom = pd.Series({'Others': d2c_bottom})

d2c = d2c_top.append(d2c_bottom)

#Beach resort data

d2r = d2.loc[d2['resort'] == 1]                   

d2r = d2r.country.value_counts()

d2r_top = d2r[0:12]

d2r_bottom = d2r[13:].sum()

d2r_bottom = pd.Series({'Others': d2r_bottom})

d2r = d2r_top.append(d2r_bottom)



width = 0.6

x1=np.arange(len(d2r))

x2=np.arange(len(d2c))



fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))



rect1 = ax1.bar(x1, d2r.values, width)

rect2 = ax2.bar(x2, d2c.values, width)



labeller(kind='bar', values=rect1, axis=ax1)

labeller(kind='bar', values=rect2, axis=ax2)

        

ax1.set_title('Beach Resort', fontsize=14)

ax1.set_xticks(x1)

ax1.set_xticklabels(d2r.index, rotation=30, size=13)

ax1.spines["top"].set_visible(False)

ax1.spines["right"].set_visible(False)

ax1.yaxis.grid(True)

ax2.set_title('City Hotel', fontsize= 14)

ax2.set_xticks(x2)

ax2.set_xticklabels(d2c.index, rotation=30, size= 13)

ax2.spines["top"].set_visible(False)

ax2.spines["right"].set_visible(False)

ax2.yaxis.grid(True)

fig.suptitle('Where do guests come from?', fontsize=17)

plt.show()
dco = data.copy()

dco = dco.loc[dco['y'] == 0]    #only actual guests

#remove adr = 0 (must be guests that got a free stay)

dco = dco.loc[dco['adr'] != 0]              

dco.reset_index(drop=True, inplace=True)

dco['adr_pp'] = dco['adr'] / (dco['adults'] + dco['children'])

dco_r = dco[['adr_pp', 'arrival_date_month']].loc[dco['resort'] == 1].groupby('arrival_date_month').mean()        

dco_c = dco[['adr_pp', 'arrival_date_month']].loc[dco['resort'] == 0].groupby('arrival_date_month').mean()

dco_r = dco_r.round(2)

dco_c = dco_c.round(2)



months_ordered = ['January', 'February', 'March', 'April', 'May', 'June', 

                  'July', 'August', 'September', 'October', 'November', 'December'] 

dco_r = dco_r.reindex(months_ordered)

dco_c = dco_c.reindex(months_ordered)

dco_r = dco_r.reset_index()

dco_r = list(dco_r['adr_pp'])

dco_c = dco_c.reset_index()

dco_c = list(dco_c['adr_pp'])



x=np.arange(len(months_ordered))



fig, ax = plt.subplots(figsize=(9,6))

points_r = ax.scatter(x=x, y=dco_r, s=100) #create scatters

points_c = ax.scatter(x=x, y=dco_c, s=100)

ax.plot(x, dco_r)                          #add lines

ax.plot(x, dco_c)



labeller(kind='scatter', values=dco_r, axis=ax, only_minmax=True)

labeller(kind='scatter', values=dco_c, axis=ax, only_minmax=True)

    

ax.set_xticks(x)

ax.set_xticklabels(months_ordered, rotation=35)

ax.set_ylim([10,105])

ax.legend(['Beach Resort', 'City Hotel'], fontsize= 12)

ax.spines["right"].set_visible(False)

ax.spines["top"].set_visible(False)

ax.yaxis.grid(True)

ax.set_title('Average Rate per person', fontsize= 14)

plt.show()
#how long do guests stay at the hotels?

data1= data.copy()

data1 = data1.loc[data1['y'] == 0]

data1['total_nights'] = data1['stays_in_weekend_nights'] + data1['stays_in_week_nights']

data1 = data1.loc[(data1['total_nights'] != 0) & (data1['total_nights'] < 14)]

data1 = data1[['resort', 'total_nights']]

res = data1['total_nights'].loc[data['resort'] == 1 ]

cit = data1['total_nights'].loc[data['resort'] == 0]

res.reset_index(drop=True, inplace=True)

cit.reset_index(drop=True, inplace=True)

res = list(res)

cit = list(cit)



fig, ax = plt.subplots(figsize=(6,6))

ax.boxplot(res, positions=[1], widths=0.5, patch_artist = True)

ax.boxplot(cit, positions=[2], widths=0.5, patch_artist = True)

ax.set_xticks([1,2])

ax.set_xticklabels(['Beach Resort', 'City Hotel'], fontsize=13)

ax.spines["right"].set_visible(False)

ax.spines["top"].set_visible(False)

ax.yaxis.grid(True)

plt.title('Staying nights per booking')

plt.show()
#Bookings by market segment

data2 = data.copy()

data2 = data2[['resort', 'market_segment']]

data2c = data2['market_segment'].loc[data2['resort'] == 0].value_counts()

data2r = data2['market_segment'].loc[data2['resort'] == 1].value_counts()



fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

fig.suptitle('How was the booking made?', fontsize=14)

ax1.pie(data2r[0:5].values, startangle=90, autopct='%1.1f%%', pctdistance=1.15)

ax1.set_xlabel('Beach Resort', fontsize=13)

ax2.pie(data2c[0:5].values, startangle=90, autopct='%1.1f%%', pctdistance=1.15)

ax2.set_xlabel('City Hotel', fontsize=13)



plt.legend(data2r.index, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
#What months get the most cancellations? 

#(divide 2 datasets and plot the difference between Resort and City)

months_ordered = ['January', 'February', 'March', 'April', 'May', 'June', 

                  'July', 'August', 'September', 'October', 'November', 'December']

data3 = data.copy()

data3 = data3.loc[data3['y'] == 1]

data3 = data3[['y','resort', 'arrival_date_month']]



data3res = data3.loc[data['resort'] == 1]

data3res = data3res.drop(columns={'resort'})

data3res = data3res.groupby('arrival_date_month').count()

data3res.reset_index(inplace=True)

data3res.loc[((data3res["arrival_date_month"] == "July") | (data3res["arrival_date_month"] == "August")),

                    "y"] /= 3

data3res.loc[~((data3res["arrival_date_month"] == "July") | (data3res["arrival_date_month"] == "August")),

                    "y"] /= 2

data3res.set_index('arrival_date_month', inplace=True)

data3res = data3res.reindex(months_ordered)



data3cit = data3.loc[data['resort'] == 0]

data3cit = data3cit.drop(columns={'resort'})

data3cit = data3cit.groupby('arrival_date_month').count()

data3cit.reset_index(inplace=True)

data3cit.loc[((data3cit["arrival_date_month"] == "July") | (data3cit["arrival_date_month"] == "August")),

                    "y"] /= 3

data3cit.loc[~((data3cit["arrival_date_month"] == "July") | (data3cit["arrival_date_month"] == "August")),

                    "y"] /= 2

data3cit.set_index('arrival_date_month', inplace=True)

data3cit = data3cit.reindex(months_ordered)



d3c=data3cit['y'].astype(int)

d3r=data3res['y'].astype(int)



width = 0.5

x=np.arange(len(months_ordered))

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))



rect3 = ax1.bar(x, d3r.values, width)

rect4 = ax2.bar(x, d3c.values, width)



labeller(kind='bar', values=rect3, axis=ax1, only_minmax=True)

labeller(kind='bar', values=rect4, axis=ax2, only_minmax=True)



ax1.set_xticks(x)

ax1.set_xticklabels(months_ordered, rotation=30)

ax1.spines["right"].set_visible(False)

ax1.spines["top"].set_visible(False)

ax1.yaxis.grid(True)

ax1.set_title('Beach Resort')

ax1.set_ylim([0,650])

ax2.set_xticks(x)

ax2.set_xticklabels(months_ordered, rotation=30)

ax2.spines["right"].set_visible(False)

ax2.spines["top"].set_visible(False)

ax2.yaxis.grid(True)

ax2.set_title('City Hotel')

ax2.set_ylim([0,2100])



plt.suptitle('Cancellations per month (%)', fontsize=16)

plt.show()
#create dummies and standardize

data_dummy=pd.get_dummies(data.copy(), dummy_na=False, drop_first=True)

data_dummy[cont_var]=pd.DataFrame(preprocessing.StandardScaler().fit_transform(data[cont_var].values),columns = cont_var)
#Correlation of top 20 variables

cancel_corr = data_dummy.corr()

cancel_corr['y'].abs().sort_values(ascending=False)[1:20]
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant

VIF_set = data_dummy.copy().drop(columns=['y'])

cols=VIF_set.columns

VIF_set = add_constant(VIF_set.values)

VIF_series = pd.Series(["{0:.2f}".format(variance_inflation_factor(VIF_set, i)) for i in range(VIF_set.shape[1])], 

                       index=['constant'] + list(cols))

#Keep only variables with VIF < 5

VIF_df = pd.DataFrame(VIF_series.rename('vif'))

VIF_df['vif'] = VIF_df['vif'].astype(float)

cols_to_keep = VIF_df.loc[VIF_df['vif'] < 5].index
#import all modules necessary for CV, fitting and measurements

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegressionCV

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import RidgeClassifierCV

from sklearn.linear_model import RidgeClassifier 

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from statistics import mean



#Divide training and testing

x = data_dummy[cols_to_keep]

y = data_dummy['y'].values 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
# CV and models: Logistic regression, Decision Tree and Random Forest. For Log. Reg., we have to specify penalty='none', otherwise a penalty is automatically added

acc_LogReg = mean(cross_val_score(LogisticRegression(solver='newton-cg', penalty='none'), X_train, y_train, cv=10))

acc_DecTree = mean(cross_val_score(DecisionTreeClassifier(), X_train, y_train, cv=10))

acc_RanFor = mean(cross_val_score(RandomForestClassifier(), X_train, y_train, cv=10))
#RidgeRegression

alphas = 10**np.linspace(10,-2,100)*0.5

ridge_cv = RidgeClassifierCV(alphas = alphas)    #select the best lambda with CV

ridge_cv.fit(X_train, y_train)

ridge_final = RidgeClassifier(alpha=ridge_cv.alpha_)

ridge_final.fit(X_train, y_train)



#LassoRegression

lasso_cv = LogisticRegressionCV(solver='liblinear', penalty='l1', cv=5)    #select the best lamba with CV. Penalty = l1 men

lasso_cv.fit(X_train, y_train)

lasso_final = LogisticRegression(solver='liblinear', penalty='l1', C=lasso_cv.C_[0])

lasso_final.fit(X_train, y_train)
print('Mean Accuracy for Logistic Regression:       ', acc_LogReg)

print('Mean Accuracy for Ridge Regression:          ', ridge_final.score(X_test, y_test))

print('Mean Accuracy for Lasso Regression:          ', accuracy_score(y_test, lasso_final.predict(X_test)))

print('Mean Accuracy for Decision Tree:             ', acc_DecTree)

print('Mean Accuracy for Random Forest:             ', acc_RanFor)
#Best performing model is Random Forest, even without Hyperparameter tuning

#Let's discover what are the best hyperparameters



param_grid = {

    'max_features': ['sqrt', 'auto', 'log2'],

    'min_samples_leaf': [1, 3, 5],

    'n_estimators': [250, 500, 750, 1000, 1500]

}

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)
grid_search.best_params_
final_model = RandomForestClassifier(n_estimators=750, max_features='sqrt', min_samples_leaf=1)

final_model.fit(X_train, y_train)

y_pred_final_model = final_model.predict(X_test)

print(classification_report(y_test, y_pred_final_model))
roc_auc = roc_auc_score(y_test, y_pred_final_model)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_final_model)



plt.figure()

plt.plot(fpr, tpr, label='Random Forest (area = %0.4f)' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
#Feature importance: create a list ('feature_importances') where each element is a tuple containing feature name and relative importance



importances = list(final_model.feature_importances_)

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(data_dummy[cols_to_keep].columns), importances)]

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

feature_importances[:20]