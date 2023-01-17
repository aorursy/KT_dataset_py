# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from datetime import datetime
import datetime
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy import genfromtxt
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
calendar_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
sell_price_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
sales_train_validation.shape
sales_train_validation.head()
date_col = [col for col in sales_train_validation if col.startswith('d_')]
len(date_col)
sales_train_validation.state_id.value_counts()
sales_train_validation['total_sales'] = sales_train_validation[date_col].sum(axis=1)
sales_train_validation['total_sales'].head()
sales_train_validation.groupby('state_id').agg({"total_sales":"sum"}).reset_index()
print(sales_train_validation.state_id.value_counts())
#Tilføj ny kolonne til dataset med total salg (summen af alle dato kolonner)
sales_train_validation['total_sales'] = sales_train_validation[date_col].sum(axis=1)
#Calculating the sales ratio
total_salg_per_stat = sales_train_validation.groupby('state_id').agg({"total_sales":"sum"})/sales_train_validation.total_sales.sum() * 100
total_salg_per_stat = total_salg_per_stat.reset_index()
#Plotting the sales ratio
fig1, ax1 = plt.subplots()
#Opret et nyt pie chart vha. matplotlib
ax1.pie(total_salg_per_stat['total_sales'],labels= total_salg_per_stat['state_id'] , autopct='%1.1f%%',
        shadow=True, startangle=90)# Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()
plt.title("Total salg per stat",fontweight = "bold")
plt.show()
print(sales_train_validation.groupby('store_id').agg({"total_sales":"sum"}).reset_index())

#Finder den totale salgsrate fordelt på de enkelte butikker
total_salg_per_butik=sales_train_validation.groupby('store_id').agg({"total_sales":"sum"})/sales_train_validation.total_sales.sum() * 100
#Lav et piechart som viser fordelingen i procent
total_salg_per_butik = total_salg_per_butik.reset_index()
fig1, ax1 = plt.subplots()
ax1.pie(total_salg_per_butik['total_sales'],labels= total_salg_per_butik['store_id'] , autopct='%1.1f%%',
        shadow=True, startangle=90)# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.title("Totale solgte varer i procent per butik",fontweight = "bold")
plt.show()
print(sales_train_validation.groupby('cat_id').agg({"total_sales":"sum"}).reset_index())

#Finder den totale salgsrate fordelt på de enkelte kategorier
total_salg_per_kategori = sales_train_validation.groupby('cat_id').agg({"total_sales":"sum"})/sales_train_validation.total_sales.sum() * 100
total_salg_per_kategori = total_salg_per_kategori.reset_index()
#Lav et piechart som viser fordelingen i procent
fig1, ax1 = plt.subplots()
ax1.pie(total_salg_per_kategori['total_sales'],labels= total_salg_per_kategori['cat_id'] , autopct='%1.1f%%',
        shadow=True, startangle=90)# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.title("Totale solgte varer i procent per kategori",fontweight = "bold")
plt.show()
sales_train_validation.groupby(['state_id','cat_id']).agg({"total_sales":"sum"}).reset_index()
dept_sales = sales_train_validation.groupby('dept_id').agg({"total_sales":"sum"}).reset_index()
print(dept_sales)
#Finder den totale salgsrate fordelt på de enkelte afdelinger
total_salg_per_afdeling = sales_train_validation.groupby('dept_id').agg({"total_sales":"sum"})/sales_train_validation.total_sales.sum() * 100
#Lav et piechart som viser fordelingen i procent
total_salg_per_afdeling = total_salg_per_afdeling.reset_index()
fig1, ax1 = plt.subplots()
ax1.pie(total_salg_per_afdeling['total_sales'],labels= total_salg_per_afdeling['dept_id'] , autopct='%1.1f%%',
        shadow=True, startangle=90)# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.title("Totale solgte varer i procent per afdeling",fontweight = "bold")
plt.show()
sell_price_data.head()
sell_price_data.groupby(['store_id', 'item_id']).agg({"sell_price": ["max", "min"]}).reset_index()
#Undersøgelse af prisændringer for de enkelte items over tid, med min og max værdier
#Aggregerer min og max prisværdier for de enkelte items i hver butik
item_store_prices = sell_price_data.groupby(["item_id","store_id"]).agg({"sell_price":["max","min"]})
#print(item_store_prices.head())
#Ændre navn på kolonner for min og max værdier
item_store_prices.columns = [f'{i}_{j}' if j != '' else f'{i}' for i,j in item_store_prices.columns]
#Tilføjer en ny 'price change' kolonne med prisændring for hver item per butik
item_store_prices["price_change"] = item_store_prices["sell_price_max"] - item_store_prices["sell_price_min"]
#print(item_store_prices.head())
#Laver en ny data frame med værdierne sorteret efter prisændring
item_store_prices_sorted = item_store_prices.sort_values(["price_change","item_id"],ascending=False).reset_index()
#Tilføjer en ny kolonne 'category' med navnet for kategorien per item
item_store_prices_sorted["category"] = item_store_prices_sorted["item_id"].str.split("_",expand = True)[0]
#print(item_store_prices_sorted.head())
#Boxplot af prisændringer ved brug af seaborn
sns.boxplot(x="price_change", y="category", data=item_store_prices_sorted)
title = plt.title("Boxplot over prisændringer for alle items opdelt efter kategori")
calendar_data.head()
calendar_data.shape
snap_days = calendar_data.groupby(['year','month'])['snap_CA','snap_TX','snap_WI'].sum().reset_index()
snap_days.pivot(index="month",columns = "year",values = ["snap_CA","snap_TX","snap_WI"])
start_date = datetime.datetime(2011,1,29)
sales_sum = pd.DataFrame(sales_train_validation[date_col].sum(axis =0),columns = ["sales"])

#tilføjer dato kolonne vha. for in range loop
sales_sum['date'] = [start_date + datetime.timedelta(days=x) for x in range(1913)]
sales_sum.set_index('date', drop=True, inplace=True)
print(sales_sum)
sales_sum.plot()
clndr = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
clndr['date'] = pd.to_datetime(clndr.date)
clndr['days'] = clndr['date'].dt.day

df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

data = pd.DataFrame(df.groupby(by= ['cat_id','dept_id','item_id','store_id']).sum())

# Daily Sales data for each category

food = pd.DataFrame(data.xs('FOODS').sum(axis = 0))
hobbies = pd.DataFrame(data.xs('HOBBIES').sum(axis = 0))
house = pd.DataFrame(data.xs('HOUSEHOLD').sum(axis = 0))

clndr = pd.merge(clndr,food,how = 'left',left_on=clndr['d'],right_on = food.index)
del clndr['key_0']
clndr.rename(columns = {0:'food'},inplace = True)
clndr = pd.merge(clndr,hobbies,how = 'left',left_on=clndr['d'],right_on = hobbies.index)
del clndr['key_0']
clndr.rename(columns = {0:'hobby'},inplace = True)
clndr = pd.merge(clndr,house,how = 'left',left_on=clndr['d'],right_on = house.index)
del clndr['key_0']
clndr.rename(columns = {0:'house'},inplace = True)

cln = clndr[0:1913]

l1 = ['FOODS','HOBBIES','HOUSEHOLD']
l2 = list(df['store_id'].unique())
for cat in l1:
    for store in l2:
        tmp = pd.DataFrame(data.xs(cat).xs(store,level = 2 ).sum(axis = 0))
        tmp.reset_index(inplace = True)
        tmp.rename(columns = {0:(cat+'_'+store).lower()},inplace = True)
        cln = pd.concat([cln,tmp[(cat+'_'+store).lower()]],axis = 1)

grps = cln.groupby(by=['year','month'])
def plot_trend(factor,subplots):
    if subplots == True:
        f, a = plt.subplots(3,2,figsize = (14,10))
        if type(factor) == list:
            for i,fact in enumerate(factor):
                check = grps.agg(fact=(fact,'sum'))
                check.rename(columns = {'fact':factor[i]},inplace=True)
                check.xs(2011).plot(ax=a[0,0])
                check.xs(2012).plot(ax=a[0,1])
                check.xs(2013).plot(ax=a[1,0])
                check.xs(2014).plot(ax=a[1,1])
                check.xs(2015).plot(ax=a[2,0])
                check.xs(2016).plot(ax=a[2,1])
                
        else:
            check = grps.agg({factor:'sum'})
            check.xs(2011).plot(ax=a[0,0])
            check.xs(2012).plot(ax=a[0,1])
            check.xs(2013).plot(ax=a[1,0])
            check.xs(2014).plot(ax=a[1,1])
            check.xs(2015).plot(ax=a[2,0])
            check.xs(2016).plot(ax=a[2,1])
        a[0,0].title.set_text('2011')
        a[0,1].title.set_text('2012')
        a[1,0].title.set_text('2013')
        a[1,1].title.set_text('2014')
        a[2,0].title.set_text('2015')
        a[2,1].title.set_text('2016')
        f.tight_layout()
        f.suptitle('Monthly Sales Trends')
    else:
        fig,ax = plt.subplots(figsize = (20,5))
        for fact in factor:
            cln.set_index('date')[fact].rolling(30).mean().plot(label = fact)
            plt.legend()
            fig.suptitle('30 Day Moving Average')
plot_trend(['food','hobby','house'],subplots = True)
plot_trend(['food'],subplots = False)
plot_trend(['hobby'],subplots = False)
plot_trend(['house'],subplots = False)
result = seasonal_decompose(sales_sum, model='additive')
result.plot()
plt.show()
calendar_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'], usecols=['date','d'])
calendar_stv = calendar_df[:1913] 

sales_train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv', index_col='id')

store_dept = sales_train_df.groupby(by= ['cat_id'], axis=0).mean()
store_dept.columns = calendar_stv['date']
store_trans = store_dept.transpose()
weekends = ['01-03-2015','01-04-2015','01-10-2015','01-11-2015','01-17-2015', '01-18-2015','01-24-2015', '01-25-2015', '01-31-2015', 
            '02-01-2015', '02-07-2015', '02-08-2015', '02-14-2015', '02-15-2015', '02-21-2015', '02-22-2015', '02-28-2015', 
            '03-01-2015', '03-07-2015', '03-08-2015', '03-14-2015', '03-15-2015', '03-21-2015', '03-22-2015', '03-28-2015',  '03-29-2015']
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [25, 5]
ax = store_trans['01-01-2015':'04-02-2015'].plot(title="Gns. salg 3 måneder jan-mar 2015")
ax.set_ylabel('# enheder')
ax.vlines(weekends, 0, 2.5, colors=['y','c'])
plt.show()
sales_train_validation.dtypes
train_dataset = sales_train_validation[date_col[-100:-30]]
val_dataset = sales_train_validation[date_col[-30:]]
predictions = []
summary = []
##An ARMA model is an ARIMA model where the d parameter in the order is 0
for row in (train_dataset[train_dataset.columns[-100:]].values[:3]):
    print(row)
    fit = SARIMAX(row, seasonal_order=(1, 1, 1, 7)).fit()
    predictions.append(fit.forecast(30))
    summary.append(fit.summary())
    fit.plot_diagnostics()
predictions = np.array(predictions).reshape((-1, 30))
error_arima = np.linalg.norm(predictions[:3] - val_dataset.values[:3])/len(predictions[0])
print(predictions)
print(summary)
pred_1 = predictions[0]
pred_2 = predictions[1]
pred_3 = predictions[2]

fig = make_subplots(rows=3, cols=1)

fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[0].values, mode='lines', marker=dict(color="darkorange"),
               name="Val"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines', marker=dict(color="seagreen"),
               name="Pred"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[1].values, mode='lines', marker=dict(color="darkorange"), showlegend=False),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=pred_2, mode='lines', marker=dict(color="seagreen"), showlegend=False,
               name="Denoised signal"),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=val_dataset.loc[2].values, mode='lines', marker=dict(color="darkorange"), showlegend=False),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(70, 100), y=pred_3, mode='lines', marker=dict(color="seagreen"), showlegend=False,
               name="Denoised signal"),
    row=3, col=1
)

fig.update_layout(height=1200, width=800, title_text="ARIMA")
fig.show()
predict_dataset = sales_train_validation[sales_train_validation.columns[-500:]].values[:5]

print(predict_dataset)

#results = []
#for row in (sales_train_validation[sales_train_validation.columns[-100:]].values):
#    fit = SARIMAX(row, seasonal_order=(1, 1, 1, 7)).fit()
#    results.append(fit.forecast(28))
#print(results)
results = genfromtxt("sarimax-predictions.csv", delimiter=',')
column_index = [1,2,3,4,5]
for i in range(6 , len(sales_train_validation.columns)):
    column_index.append(i)

clus_hobbies = sales_train_validation.iloc[:,column_index].query("cat_id == 'HOBBIES'")
clus_household = sales_train_validation.iloc[:,column_index].query("cat_id == 'HOUSEHOLD'")
clus_foods = sales_train_validation.iloc[:,column_index].query("cat_id == 'FOODS'")
clus_ca = sales_train_validation.iloc[:,column_index].query("state_id == 'CA'")
clus_tx = sales_train_validation.iloc[:,column_index].query("state_id == 'TX'")
clus_wi = sales_train_validation.iloc[:,column_index].query("state_id == 'WI'")
clus = sales_train_validation.iloc[:,column_index]
calendar_data["event_type_1_snap"] = pd.notna(calendar_data["event_type_1"]) 
calendar_data["event_type_2_snap"] = pd.notna(calendar_data["event_type_2"]) 
calendar_data["date"] =  pd.to_datetime(calendar_data["date"])
calendar_data["d_month"] = calendar_data["date"].dt.day
calendar_data["year"] = pd.to_numeric(calendar_data["year"])
calendar_data["wday"] = pd.to_numeric(calendar_data["wday"])
print(calendar_data.shape)
calendar_data.head()
#Bucket columns by calander days of month
columnsets = []
for i in range(1,32):      
    d = calendar_data[:1913].query("d_month == "+ str(i))["d"]
    columnsets.append([d.values])

# Label encoding for catagorical data
def label_encoding(data_preap,cat_features):
    categorical_names = {}
    data = []
    encoders = []
    
    data = data_preap[:]
    for feature in cat_features:
        le = LabelEncoder()
        le.fit(data.iloc[:,feature])
        data.iloc[:, feature] = le.transform(data.iloc[:, feature])
        categorical_names[feature] = le.classes_
        encoders.append(le)
    X_data = data.astype(float)
    return X_data, encoders
# Training random forest model
def train_model(X_train, X_test, Y_train, Y_test):
    # Random forest regressor model with Training dataset
    start_time = datetime.today()
    regressor = RandomForestRegressor(n_estimators = 350, random_state = 50)
    regressor.fit(X_train,Y_train)

    print("Time taken to Train Model: " + str(datetime.today() - start_time))

    # Running Regession model score check
    Y_score = regressor.score(X_test,Y_test)
    return regressor,Y_score
# Predict function from model
def model_predict(regressor,X_data):
    # Predicting model model result
    Y_pred = regressor.predict(X_data)
    return Y_pred
# Validating model with last year data & generating rmse value for the model predection
def validate_model(regressor,X_validation, Y_validation):
   
    Y_validation_pred = model_predict(regressor, X_validation)
    mse = mean_squared_error(Y_validation, Y_validation_pred)
    rmse = np.sqrt(mse)
    return rmse, Y_validation_pred
# Basic function for geting data from pandas based on range
def get_data_range(Inital_Range,start_index,end_index):
    result = []
    [result.append(a) for a in Inital_Range]
    for i in range(max(Inital_Range) +1 + start_index, end_index):
        result.append(i)
    return result
# main function to run predictions
def run_predictions(orig_data):
    process_data = orig_data[:]
    results = pd.DataFrame()
    for s in range(1,29):
        categorical_features = [0,1]
        data = []
        data_range = []
        for i in range(0,s):
            [data_range.append(a) for a in columnsets[i]]
        data_list = [process_data[a] for a in data_range]
        data  = pd.concat(data_list,axis = 1)


        data.insert(loc=0, column='item_id', value=process_data["item_id"])
        data.insert(loc=1, column='store_id', value=process_data["store_id"])
        X_data_preap = data[:]

        d = get_data_range(categorical_features,0,len(X_data_preap.columns)-1)   
        X,label_encoders = label_encoding(X_data_preap.iloc[:,d],categorical_features)
        Y = X.iloc[:,-1]

        d_validation = get_data_range(categorical_features,1,len(X_data_preap.columns))   
        X_validation,label_encoders_validation = label_encoding(X_data_preap.iloc[:,d_validation],categorical_features)
        Y_validation = X_validation.iloc[:,-1]

        print("Running Model for Day " + str(s))
        # Sampling data for train & split
        X_train, X_test, Y_train, Y_test = train_test_split(X.iloc[:,0:len(X.columns)-1],Y,test_size = 0.2, random_state = 0)
        model, score = train_model(X_train, X_test, Y_train, Y_test)
        print("Model Score: " + str(score))
        
       # Uncomment for inital model
        rmse,validation_predictions = validate_model(model,X_validation.iloc[:,0:len(X_validation.columns)-1], Y_validation)
        print("RMSE Result: " + str(rmse))
        
        if (len(results.columns) == 0):
            for feature in categorical_features:
                results[feature] = label_encoders_validation[feature].inverse_transform(X_validation.iloc[:,feature].astype(int))

        results["d_" + str(s)] = validation_predictions.astype(int)
        print(results)
        results.to_csv('pd_predictions_' + str(s) +'.csv')
    return results
#pd_predictions = run_predictions(clus)