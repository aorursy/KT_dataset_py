# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sales = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")

sales.columns
li=[1]

for i in range(6,sales.columns.shape[0] -1):

    li.append(i)

li
#sales['total']=sales.iloc[:,6:].sum(axis=1)

#sales['total'].head()

sales_total=sales.iloc[:,[1,-1]]

sales_total.head()
###sort values about total sales φθινουσα

sales_total.sort_values(by='total',ascending=False).tail(10)
#ΣΥΝΟΛΙΚΕΣ ΠΩΛΗΣΕΙΣ ΑΝΑ ΚΑΤΗΓΟΡΙΑ ΤΑΞΙΝΟΜΗΜΕΝΕΣ ΚΑΤΑ ΦΘΙΝΟΥΣΑ ΣΕΙΡΑ

hobbies=sales[sales['item_id'].str.split('_').str[0]=='HOBBIES']

#ΣΥΝΟΛΙΚΕΣ ΠΩΛΗΣΕΙΣ ΑΝΑ ΚΑΤΗΓΟΡΙΑ ΤΑΞΙΝΟΜΗΜΕΝΕΣ ΚΑΤΑ ΦΘΙΝΟΥΣΑ ΣΕΙΡΑ

hobbies_sort=hobbies.iloc[:,[1,-1]].sort_values(by='total',ascending=False)

hobbies_sort.head(10)
#ΣΥΝΟΛΙΚΕΣ ΠΩΛΗΣΕΙΣ ΑΝΑ ΚΑΤΗΓΟΡΙΑ ΤΑΞΙΝΟΜΗΜΕΝΕΣ ΚΑΤΑ ΦΘΙΝΟΥΣΑ ΣΕΙΡΑ

household=sales[sales['item_id'].str.split('_').str[0]=='HOUSEHOLD']

household_sort=household.iloc[:,[1,-1]].sort_values(by='total',ascending=False)

household_sort.head(10)
#ΣΥΝΟΛΙΚΕΣ ΠΩΛΗΣΕΙΣ ΑΝΑ ΚΑΤΗΓΟΡΙΑ ΤΑΞΙΝΟΜΗΜΕΝΕΣ ΚΑΤΑ ΦΘΙΝΟΥΣΑ ΣΕΙΡΑ

foods=sales[sales['item_id'].str.split('_').str[0]=='FOODS']

foods_sort=foods.iloc[:,[1,-1]].sort_values(by='total',ascending=False)

foods_sort.head(10)



##TREXEI OPWS THELOUME ALLA ARGEI VERSION 1

from ipywidgets import interact, Dropdown

import ipywidgets as wg

range_slider2=wg.IntRangeSlider(value=[1000,10000],min=sales_total['total'].min() - 5,max=sales_total['total'].max() + 5,readout_format='d')

categories={'HOOBBIES_SALES':hobbies_sort,

           'FOODS_SALES':foods_sort,

           'HOUSEHOLD_SALES':household_sort,

           'TOTAL_SALES':sales_total}

categoriesW=Dropdown(options=categories.keys())

idW=Dropdown()



@interact(category=categoriesW,Range=range_slider2,item_id=idW)

def upd_idw(category,Range,item_id):

    idW.options=categories[categoriesW.value][(categories[categoriesW.value]['total']>=Range[0]) & (categories[categoriesW.value]['total']<=Range[1])]['item_id']

    
##DEN ALLAZEI TO 2o DROP DOWN MENU 

from ipywidgets import interact, Dropdown

import ipywidgets as wg



categories={'HOOBBIES_SALES':hobbies_sort,

           'FOODS_SALES':foods_sort,

           'HOUSEHOLD_SALES':household_sort,

           'TOTAL_SALES':sales_total}

categoriesW=Dropdown(options=categories.keys())

idW=Dropdown()

range_slider100=wg.IntSlider(min=0,max=255000,value=1000)

range_slider200=wg.IntSlider(min=0,max=255000,value=1000)

@interact(category=categoriesW,minmin=range_slider100.value,maxmax=range_slider200.value,item_id=idW)

def upd_idw(category,minmin,maxmax,item_id):

    idW.options=categories[categoriesW.value][(categories[categoriesW.value]['total']>=minmin) & (categories[categoriesW.value]['total']<=maxmax)]['item_id']
from ipywidgets import interact, Dropdown

import ipywidgets as wg



categories100={'HOOBBIES_SALES':hobbies_sort,

           'FOODS_SALES':foods_sort,

           'HOUSEHOLD_SALES':household_sort,

           'TOTAL_SALES':sales_total}

categories100W=Dropdown(options=categories100.keys())



range_slider100=wg.IntSlider(min=0,max=255000,value=1000)

range_slider200=wg.IntSlider(min=0,max=255000,value=1000)

display(categories100W,range_slider100,range_slider200)
id100W=Dropdown()





def up100(category,range100,range200):

    id100W.options=categories100[categories100W.value][(categories100[categories100W.value]['total']>= range100) & (categories100[categories100W.value]['total']<=range200)]['item_id']

    display(id100W)

interact(up100,category=categories100W,range100=range_slider100.value,range200=range_slider200.value)

###KWDIKAS MARIOS

from ipywidgets import interact, interactive, fixed, interact_manual, Dropdown

import ipywidgets as widgets

from IPython.display import display

cat={'HOBBIES':hobbies_sort,'HOUSEHOLD':household_sort,'FOODS':foods_sort}



catW = Dropdown(options = cat.keys())

idW=Dropdown()

range_slider0 = widgets.IntRangeSlider(description='Αριθμός Πωλήσεων', min=0, max=252000, step=1,value=[0,255000])

range_slider1 = widgets.IntRangeSlider(description='Αριθμός Πωλήσεων', min=0, max=252000, step=1,value=[0,255000])





@interact(Category=catW,Min_Sales=range_slider0.value,Max_Sales=range_slider1.value,Product=idW)

def print_cat(Category,Min_Sales,Max_Sales,Product):

    idW.options = cat[catW.value][(cat[catW.value]['total']>=Min_Sales) & (cat[catW.value]['total']<=Max_Sales)]['item_id']

##TREXEI KANONIKA-VERSION2

from ipywidgets import interact, Dropdown

import ipywidgets as wg



range_slider2=wg.IntRangeSlider(value=[1000,10000],min=sales_total['total'].min() - 5,max=sales_total['total'].max() + 5,description='Test:',readout_format='d')

categories={'HOOBBIES_SALES':[],

           'FOODS_SALES':[],

           'HOUSEHOLD_SALES':[],

           'TOTAL_SALES':[]}

categoriesW=Dropdown(options=categories.keys())



display(categoriesW,range_slider2)
###TREXEI KANONIKA-VERSION2

if categoriesW.value== 'HOUSEHOLD_SALES':

    categories['HOUSEHOLD_SALES']=household_sort[(household_sort['total']<= range_slider2.value[1]) & (household_sort['total']>=range_slider2.value[0])]['item_id']

elif categoriesW.value=='FOODS_SALES':

    categories['FOODS_SALES']=foods_sort[(foods_sort['total']<=range_slider2.value[1]) & (foods_sort['total']>=range_slider2.value[0])]['item_id']

elif categoriesW.value=='HOBBIES_SALES':

    categories['HOBBIES_SALES']=hobbies_sort[(hobbies_sort['total']<=range_slider2.value[1]) & (hobbies_sort['total']<=range_slider2.value[0])]['item_id']

elif categoriesW.value=='TOTAL_SALES':

    categories['TOTAL_SALES']=sales_total[(sales_total['total']<=range_slider2.value[1]) & (sales_total['total']>=range_slider2.value[0])]['item_id']



idW=Dropdown(options=categories[categoriesW.value])

display(idW)
from ipywidgets import interact, Dropdown

import ipywidgets as wg

range_slider=wg.IntRangeSlider(value=[1000,10000],min=sales_total['total'].min() - 5,max=sales_total['total'].max() + 5,description='Test:',readout_format='d')



cat={'HOBBIES_SALES':lambda : hobbies_sort[(hobbies_sort['total']>= range_slider.value[0]) & (hobbies_sort['total']<=range_slider.value[1])]['item_id'],

     'HOUSEHOLD_SALES':lambda : household_sort[(household_sort['total']>= range_slider.value[0]) & (household_sort['total']<=range_slider.value[1])]['item_id'],

     'FOODS_SALES':lambda : foods_sort[(foods_sort['total']>= range_slider.value[0]) & (foods_sort['total']<=range_slider.value[1])]['item_id'],

     'TOTAL_SALES':lambda : sales_total[(sales_total['total']>=range_slider.value[0]) & (sales_total['total']<=range_slider.value[1])]['item_id']}

catW=Dropdown(options=cat.keys())

idW=Dropdown(options=cat[catW.value]())



# Creating event handler

def ddl_event_handler(event):

    idW.options=cat[event['new']]() #updating the options to the selected value



# Add an observer

catW.observe(ddl_event_handler,names="value")

#optionally specify a `display_id` to update the same area

display(catW,range_slider,idW,display_id="options_area")


a = widgets.IntSlider(description="Delayed", continuous_update=False)

b = widgets.IntText(description="Delayed", continuous_update=False)

c = widgets.IntSlider(description="Continuous", continuous_update=True)

d = widgets.IntText(description="Continuous", continuous_update=True)



widgets.link((a, 'value'), (b, 'value'))

widgets.link((a, 'value'), (c, 'value'))

widgets.link((a, 'value'), (d, 'value'))

widgets.VBox([a,b,c,d])
import xgboost as xgb

from sklearn import model_selection, preprocessing

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error
for i in range(1,29):

    sales[i]=np.nan

sales.columns    

sales.drop(['i'],axis=1)

sales.shape
X=sales.iloc[:,6:-29]

X.columns

y=sales.iloc[:,-28]

y
X=sales.iloc[:,6:-29]

y=sales.iloc[:,-28]

data_dmatrix = xgb.DMatrix(data=X,label=y)
sales.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg =xgb.XGBRegressor()

xg_reg.fit(X_train,y_train)



preds = xg_reg.predict(X_test)

preds
rmse = np.sqrt(mean_squared_error(y_test, preds))
sales2 = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")

sales2.columns
import xgboost as xgb

from sklearn.metrics import mean_squared_error

import pandas as pd

import numpy as np
X, y = sales2.iloc[:100,6:-28],sales2.iloc[:100,-28:-27]
X.columns
y.columns
data_dmatrix = xgb.DMatrix(data=X,label=y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror' ,colsample_bytree = 0.5, learning_rate = 0.1,

               alpha = 5, n_estimators = 100)

xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
preds
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse))

params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}



cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5, num_boost_round=28,early_stopping_rounds=28,metrics="rmse", as_pandas=True, seed=123)