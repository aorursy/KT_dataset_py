import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import pickle

from sklearn import preprocessing

from sklearn import model_selection

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import roc_auc_score,r2_score,mean_absolute_error,mean_squared_error,accuracy_score,classification_report,confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/dataco-smart-supply-chain-for-big-data-analysis/DataCoSupplyChainDataset.csv',header= 0,encoding='unicode_escape')

pd.set_option('display.max_columns',None)

data.head()
def data_info(data):

    print('1) Number of columns are : ',data.shape[1])

    print('2) Number of rows are : ',data.shape[0])

    print('3) Total number of data-points :',data.size)

    numerical_features = [f for f in data.columns if data[f].dtypes!='O']

    print('4) Count of Numerical Features :',len(numerical_features))

    cat_features = [c for c in data.columns if data[c].dtypes=='O']

    print('5) Count of Categorical Features :',len(cat_features))

data_info(data)
def features_with_missing_values(data):

    x = data.isnull().sum().sum()/(data.shape[0]*data.shape[1])*100

    print('Percentage of Total Missing Values is ' ,round(x,2) ,'%')

    print('Missing Value Estimation :')

    for i in data.columns:

        if data[i].isna().sum()>0:

            print('The Feature ',i,' has '+ str(data[i].isna().sum()) + ' missing values')

            

features_with_missing_values(data)
# Target value analysis

sns.set()

sns.scatterplot(x="Days for shipment (scheduled)",y="Days for shipping (real)",hue="Late_delivery_risk",data=data)
corrmap = data.corr()

top=corrmap.index

plt.figure(figsize=(30,20))

g=sns.heatmap(data[top].corr(),annot=True,cmap="RdYlGn")
shipment_features = ['Type','Days for shipping (real)','Days for shipment (scheduled)','Late_delivery_risk','Benefit per order',

                        'Sales per customer','Latitude','Longitude','Shipping Mode','Order Status','Order Region',

                        'Order Country','Order City','Market','Delivery Status']

shipment = data[shipment_features]

shipment.head()
finance_features=['Benefit per order','Sales per customer','Order Item Discount','Order Item Discount Rate',

                  'Order Item Product Price','Order Item Profit Ratio']

finance = data[finance_features]

finance.head()
#Converting categorical features that represent date and time to datetime datatype.

data['order_date'] = pd.to_datetime(data['order date (DateOrders)'])

data['shipping_date']=pd.to_datetime(data['shipping date (DateOrders)'])
# Handling Time and date variables

data['order_year'] = pd.DatetimeIndex(data['order_date']).year

data['order_month'] = pd.DatetimeIndex(data['order_date']).month

data['order_day'] = pd.DatetimeIndex(data['order_date']).day

data['shipping_year'] = pd.DatetimeIndex(data['shipping_date']).year

data['shipping_month'] = pd.DatetimeIndex(data['shipping_date']).month

data['shipping_day'] = pd.DatetimeIndex(data['shipping_date']).day
new_dataset_features = ['Type','Days for shipping (real)','Days for shipment (scheduled)','Late_delivery_risk','Benefit per order',

                        'Sales per customer','Latitude','Longitude','Shipping Mode','Order Status','Order Region',

                        'Order Country','Order City','Market','Delivery Status','order_day','order_month','order_year',

                        'shipping_day','shipping_month','shipping_year']

len(new_dataset_features)
new_data = data[new_dataset_features]

model_data = new_data

new_data.head()
#One-Hot encoding categotical variables in the data

model_data = pd.get_dummies(model_data)

model_data.shape
x = model_data.drop(['Days for shipping (real)','Days for shipment (scheduled)'],axis=1)

y = model_data[['Days for shipping (real)','Days for shipment (scheduled)']]

x.shape,y.shape
# train-test_split

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.20)
#sc=StandardScaler() 

#No standard Scaling is not required for Decision Trees are tree-based algorithms and do not need normalization or standard scaling

#pc=PCA()

# The hyper-parameters used are default hyper-parameters

model=DecisionTreeRegressor()

model.fit(x_train,y_train)

pred=model.predict(x_test)
#Function for estimating r2_score, mean_squared_error, mean_absolute_error

def metrics(y_test,pred):

    a =r2_score(y_test,pred)

    b =mean_squared_error(y_test,pred)

    c =mean_absolute_error(y_test,pred)

    print('The r-squared score of the model is ',a)

    print('The mean squared error is',b)

    print('The mean accuracy score is',c)
metrics(y_test,pred)

#from sklearn.model_selection import cross_validate

#cv_results = cross_validate(clf,x,y,cv=4)

#cv_results {NO sufficient RAM space in Kaggle Kernel to run the CROSS-VALIDATION TESTS}
#Converting the predicted output array to dataframe

Prediction = pd.DataFrame(pred)

prediction = Prediction.rename(columns={0:'Fastest_shipment',1:'Avg_shipment'})

prediction.head()
# Statiscal modelling

prediction['risk'] = np.where(prediction['Avg_shipment'] >= prediction['Fastest_shipment'],0,1)
prediction.head()
l = prediction['risk']

m = x_test['Late_delivery_risk']

l.shape,m.shape
# Defining a function to evaluate our statiscal model for Late_delivery_risk_prediction

def evaluation_risk_factor(l,m):

  print('1) The accuracy of the risk predictor model is :',accuracy_score(l,m))

  print('2) The AUROC score is :',roc_auc_score(l,m))

  print('3) Some of the key classification metrics are :')

  print(classification_report(l,m))

  ax=plt.subplot()

  sns.heatmap(confusion_matrix(l,m),annot=True,ax=ax);

  ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')

  ax.set_title('Confusion matrix for Risk Delivery classfication');
evaluation_risk_factor(l,m)
sns.set(style="darkgrid")

ax=sns.countplot(x="risk",data=prediction).set_title('Predicted Late delivery risks')
bx=sns.countplot(x='Late_delivery_risk',data=x_test).set_title('Actual Late_delivery_risk')
filename = 'Shipping_duration_estimator.pkl'

pickle.dump(model,open(filename,'wb'))