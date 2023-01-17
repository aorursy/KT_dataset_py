import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

pd.set_option('display.max_columns', None)  

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)

%matplotlib inline

plt.rcParams['figure.figsize'] = (20.0, 15.0)



#using the cleaned marketing funnel data to merge with sellers

funnel_df = pd.read_csv("../input/adnuolist-analysis/cleaned_marketing_funnel.csv")
#Lets see the business segments and lead types and compare them over SRs in closed deals

segment_df = funnel_df.groupby(["sr_id","business_segment"]).count().seller_id.reset_index(name="count")

lead_df = funnel_df.groupby(["sr_id","lead_type"]).count().seller_id.reset_index(name="count")



#ordering and cleaning data of NaN/NA values

segment_df = segment_df.pivot(index= "sr_id", columns= "business_segment", values= "count").fillna(0)

lead_df= lead_df.pivot(index= "sr_id", columns= "lead_type", values= "count").fillna(0)



#summarising groupby totals for better stack graph visualization

segment_df["total"] = segment_df.sum(axis=1)

segment_df = segment_df.sort_values("total",ascending= True).drop(["total"],axis=1)



lead_df["total"] = lead_df.sum(axis=1)

lead_df = lead_df.sort_values("total",ascending= True).drop(["total"],axis=1)

sns.set()

sns.palplot(sns.color_palette("Paired"))
#for segments plot



plt.rcParams['figure.figsize'] = (15.0, 20.0)



segment_df.plot.barh(stacked= True).legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.title("SR and Segment Closed Deals",fontsize=25)

plt.xlabel("Closed deal count")

plt.tight_layout()



#plt.savefig("sr_segment_analysis.png")
#for lead type plot

plt.rcParams['figure.figsize'] = (15.0, 20.0)

lead_df.plot.barh(stacked= True).legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.title("SR and Lead Type - Closed Deals",fontsize=25)

plt.xlabel("Closed deal count")

plt.tight_layout()



#plt.savefig("charts/sr_lead_type_analysis.png")
orders_df = pd.read_csv("../input/brazilian-ecommerce/olist_order_items_dataset.csv")

orders_df.describe(include="all")
quantity_df = orders_df.groupby(['order_id', 'product_id','seller_id','price'])['order_item_id'].agg({"quantity":"max"}).reset_index()

quantity_df['order_price'] = quantity_df['price']*quantity_df['quantity']

quantity_df
#multiple product quantity summary

tdf= quantity_df.quantity.value_counts().sort_values()

tdf = pd.DataFrame({'quantity':tdf.index, 'order_counts':tdf.values}).sort_values("order_counts",ascending=False) 

tdf.head()
quantity_df[quantity_df["quantity"]>1]
#Aggregating total revenue per seller

total_revenue_df = quantity_df.groupby(['seller_id'])['order_price'].agg('sum').reset_index()

total_revenue_df
#merging funnel df and total revenue df (379 out of 841 leads have seller data)

funnel_df.first_contact_date = pd.to_datetime(funnel_df.first_contact_date) #normalizing contact date

funnel_df["contact_month"] = funnel_df.first_contact_date.dt.month

final_df_left = funnel_df.merge(total_revenue_df, on='seller_id', how="left")

final_df_inner = funnel_df.merge(total_revenue_df, on='seller_id', how="inner")





final_df_inner
print("Declared monthly revenue of seller(count) in all closed leads dataset - ",len(final_df_left[final_df_left.declared_monthly_revenue>0]))

print("Declared monthly revenue of seller(count) in closed leads dataset merged with calculated revenue from orders data - ",len(final_df_inner[final_df_inner.declared_monthly_revenue>0]))

#converting timedelta to numeric day count for input to classifier

final_df_inner["close_duration"]= pd.to_timedelta(final_df_inner.close_duration)

final_df_inner["closing_days"] = final_df_inner.close_duration.values/np.timedelta64(1, 'D')
from sklearn import preprocessing

column_list = ['business_segment','lead_type','lead_behaviour_profile','business_type','landing_page_id','origin']



for column in column_list:

    encoder = preprocessing.LabelEncoder()

    encoder.fit(final_df_inner[column])

    final_df_inner[column] = encoder.transform(final_df_inner[column])

from sklearn.model_selection import train_test_split



features = column_list + ['contact_month','closing_days']

target = ['order_price']



X = final_df_inner[features].values

y = final_df_inner[target].values

split_test_size = 0.10



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, random_state=8)



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', X_test.shape)

print('Testing Features Shape:', y_train.shape)

print('Testing Labels Shape:', y_test.shape)

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

rf_model = RandomForestRegressor(n_estimators=16, min_samples_split=2, random_state = 8)

rf_model.fit(X_train, np.ravel(y_train,order='C'))
from sklearn.metrics import mean_absolute_error

rf_predict_train = rf_model.predict(X_train)

rf_predict_test = rf_model.predict(X_test)



print("Mean Absolute error in predicting revenue amount (Training):-",mean_absolute_error(y_train,rf_predict_train))

print("Mean Absolute error in predicting revenue amount (Testing):-",mean_absolute_error(y_test,rf_predict_test))
feature_imp = pd.Series(rf_model.feature_importances_,index=features).sort_values(ascending=False)

feature_imp
#Trying TPOT automl

# from tpot import TPOTRegressor

# tpot = TPOTRegressor(generations=10, population_size=100, verbosity=2, random_state=8,n_jobs=10)

# tpot.fit(X_train, np.ravel(y_train,order='C'))

# print(tpot.score(X_test, np.ravel(y_test,order='C')))

# tpot.export('tpot_revenue_pipeline.py')
leads_df = pd.read_csv("../input/marketing-funnel-olist/olist_marketing_qualified_leads_dataset.csv")

leads_df
#converting timedelta to numeric day count for input to classifier

final_df_left["close_duration"]= pd.to_timedelta(final_df_left.close_duration)

final_df_left["closing_days"] = final_df_left.close_duration.values/np.timedelta64(1, 'D')

final_df_left["contact_year"] = final_df_left.first_contact_date.dt.year
final_df_left.head()
from sklearn import preprocessing

column_list = ['landing_page_id','origin'] #'contact_month','contact_year']



for column in column_list:

    encoder = preprocessing.LabelEncoder()

    encoder.fit(final_df_left[column])

    final_df_left[column] = encoder.transform(final_df_left[column])

    

    
from sklearn.model_selection import train_test_split



features = column_list + ['contact_month','contact_year']

target = ['closing_days']



X = final_df_left[features].values

y = final_df_left[target].values

split_test_size = 0.15



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, random_state=8)



print('Training Features Shape:', X_train.shape)

print('Training Labels Shape:', X_test.shape)

print('Testing Features Shape:', y_train.shape)

print('Testing Labels Shape:', y_test.shape)
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

rf_model_leadclose = RandomForestRegressor(n_estimators=50, min_samples_split=2, random_state = 8)

rf_model_leadclose.fit(X_train, np.ravel(y_train,order='C'))
from sklearn.metrics import mean_absolute_error

rf_predict_train = rf_model_leadclose.predict(X_train)

rf_predict_test = rf_model_leadclose.predict(X_test)



print("Mean Absolute error in predicting closing days (Training):-",mean_absolute_error(y_train,rf_predict_train))

print("Mean Absolute error in predicting closing days (Testing):-",mean_absolute_error(y_test,rf_predict_test))
feature_imp = pd.Series(rf_model_leadclose.feature_importances_,index=features).sort_values(ascending=False)

feature_imp
import xgboost as xgb

clf = xgb.XGBRegressor()

clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))

# import os

# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



xgb.to_graphviz(clf, num_trees=4)