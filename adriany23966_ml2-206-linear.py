import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy.stats import norm
import datetime
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

kc_house_data=pd.read_csv(r'/kaggle/input/housesalesprediction/kc_house_data.csv')

#Set up Real Estate Age Calculation method
def estate_age_calculation(built,renov):
    if renov==0:
        result=datetime.date.today().year-built
        return result
    else:
        result=datetime.date.today().year-renov
        return result
kc_house_data['renov_state']= kc_house_data.yr_renovated.apply(lambda x: 1 if str(x)!='0' else 0)
kc_house_data['modified_asset_age']=kc_house_data.apply(lambda x: estate_age_calculation(x.yr_built, x.yr_renovated), axis = 1)
#Set Real Estate Age Calculation method

#Set dummy variables for month factor
kc_house_data['basement']=kc_house_data.sqft_basement.apply(lambda x: 1 if x!=0 else 0)
kc_house_data['month']=kc_house_data.date.apply(lambda x: datetime.datetime.strptime(x[:8],'%Y%m%d').strftime('%b'))
dummy_field=['basement','floors','waterfront','view','condition','grade','month','renov_state']
for each in dummy_field:
    dummies_tmp = pd.get_dummies( kc_house_data.loc[:, each], prefix=each) 
    kc_house_data = pd.concat( [kc_house_data, dummies_tmp], axis = 1 )
for cols in dummy_field:
    try:
        kc_house_data=kc_house_data.drop(cols,axis=1)
    except:
        pass
#Set dummy variables for month factor

#Exam data type
kc_house_data.info()
#Exam data type


#Correlation
# 'I figured that sqft_living equals sum of sqft_above + sqft_basement \
# thus causing linear realtionship between variables, so sqft_living is \
# deprecated and also i imported basement as a dummy variable.'
plt.figure(figsize = (20,10))
house_info =['bedrooms', 'bathrooms',
       'sqft_lot', 'sqft_above', 'sqft_basement', 
       'lat', 'long', 'sqft_living15', 'sqft_lot15','modified_asset_age']
var_correlation = kc_house_data[house_info].corr()  
mask = np.zeros_like(var_correlation)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    sns.heatmap(var_correlation,linewidths=0, annot=True,vmax=1,mask=mask,cmap= "RdBu_r", square=True)
#Correlation
#Split train & test
kc_house_data_input,kc_house_data_test = train_test_split(kc_house_data,train_size = 0.8,random_state=3)
#Split train & test
#Regression
variables = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
        'lat', 'long', 'sqft_living15', 'sqft_lot15',
       'modified_asset_age', 'basement_0', 'basement_1', 'floors_1.0',
       'floors_1.5', 'floors_2.0', 'floors_2.5', 'floors_3.0', 'floors_3.5',
       'waterfront_0', 'waterfront_1', 'view_0', 'view_1', 'view_2', 'view_3',
       'view_4', 'condition_1', 'condition_2', 'condition_3', 'condition_4',
       'condition_5', 'grade_1', 'grade_3', 'grade_4', 'grade_5', 'grade_6',
       'grade_7', 'grade_8', 'grade_9', 'grade_10', 'grade_11', 'grade_12',
       'grade_13', 'month_Apr', 'month_Aug', 'month_Dec', 'month_Feb',
       'month_Jan', 'month_Jul', 'month_Jun', 'month_Mar', 'month_May',
       'month_Nov', 'month_Oct', 'month_Sep', 'renov_state_0',
       'renov_state_1']
lr_model = linear_model.LinearRegression()
lr_model.fit(kc_house_data_input[variables],kc_house_data_input['price'])
#Regression
#evaluation
print('Intercept: {}'.format(lr_model.intercept_))
print('Coefficients: {}'.format(lr_model.coef_))
evaluation = pd.DataFrame({'Root Mean Squared Error (RMSE)':[],
                           'R-squared (training)':[],
                           'R-squared (testing)':[]})
pred = lr_model.predict(kc_house_data_test[variables])
rmse = float(format(np.sqrt(metrics.mean_squared_error(kc_house_data_test['price'],pred)),'.3f'))
r2train = float(format(lr_model.score(kc_house_data_input[variables],kc_house_data_input['price']),'.3f'))

r2test = float(format(lr_model.score(kc_house_data_test[variables],kc_house_data_test['price']),'.3f'))


r = evaluation.shape[0]

evaluation.loc[r] = [rmse,r2train,r2test]
#evaluation
evaluation