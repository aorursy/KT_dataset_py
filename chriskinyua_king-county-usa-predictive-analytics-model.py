import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.linear_model import LinearRegression 

from sklearn.model_selection import train_test_split  

from math import sqrt

from sklearn.metrics import mean_squared_error

import mpl_toolkits

from sklearn.ensemble import GradientBoostingRegressor
data=pd.read_csv('../input/kc_house_data.csv')
#Display the first 10 records in the dataset

data.head(10)
#Display a concise summary of the dataset

#There are 21613 records in the dataset. Each column has exactly 21613 entries indicating there are no null entries.



data.info()
#Generate descriptive statistics that summarize the central tendency, dispersion and shape of the dataset’s distribution.



data.describe()
corrmat = data.corr()

cols = corrmat.nlargest(21, 'price')['price'].index #specify number of columns to display i.e 21

f, ax = plt.subplots(figsize=(18, 10)) #size of matrix

cm = np.corrcoef(data[cols].values.T)

sb.set(font_scale=1.25)

hm = sb.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':12}, yticklabels=cols.values,

                 xticklabels=cols.values)

plt.yticks(rotation=0, size=15)

plt.xticks(rotation=90, size=15)

plt.title("Correlation Matrix",style='oblique', size= 20)



plt.show()
plt.figure(figsize=(18,10))



plt.scatter(data['grade'],data['price'])

plt.xlabel("Grade")

plt.ylabel("Price")

plt.title("Price v. Grade")

plt.figure(figsize=(18,10))



plt.scatter(data['yr_built'],data['price'])

plt.xlabel("Year Built")

plt.ylabel("Price")

plt.title("Price v. Year Built")
plt.figure(figsize=(18,10))

plt.title("Price v. Bathrooms")



plt.scatter(data['price'],data['bathrooms'])

plt.ylabel("Bathrooms")

plt.xlabel("Price")

plt.plot(np.unique(data['price']), np.poly1d(np.polyfit(data['price'], data['bathrooms'], 1))(np.unique(data['price'])), color='green') #line of best fit



plt.figure(figsize=(18,10))



plt.scatter(data['bedrooms'],data['price'])

plt.ylabel('Price')

plt.xlabel('No. of bedrooms')

plt.title('Bedrooms v. Price')
plt.figure(figsize=(18,10))



plt.scatter(data['sqft_living'],data['price'])

plt.title('Price v. Square footage (living area)')

plt.xlabel('Square Footage')

plt.ylabel("Price")
plt.figure(figsize=(18,10))



plt.scatter(data['price'],data['sqft_above'])

plt.title('Price v. Square footage (above)')

plt.ylabel('Square Footage')

plt.xlabel("Price")

plt.plot(np.unique(data['price']), np.poly1d(np.polyfit(data['price'], data['sqft_above'], 1))(np.unique(data['price'])), color='green') #line of best fit

plt.figure(figsize=(18,10))



sb.jointplot(x=data.lat.values, y=data.long.values, size=12,color='brown')

plt.ylabel('Longitude', fontsize=12)

plt.xlabel('Latitude', fontsize=12)

plt.title("Concentration of Houses by Location")

sb.despine

plt.figure(figsize=(18,10))



data["bedrooms"].value_counts().plot(kind='bar')

plt.title('Count vs bedrooms Bar Graph')

plt.ylabel("Count")

plt.xlabel('Number of bedrooms')
plt.figure(figsize=(18,10))



plt.boxplot(data['bedrooms'],1,'gD')
#count number of houses with more than ten bedrooms

data[data['bedrooms']>10].count()
#locate house with 33 bedrooms

data.loc[data['bedrooms'] == 33]
plt.figure(figsize=(18,10))



plt.boxplot(data['price'],1,'gD')
#count number of houses with prices above 7,000,000

data[data['price']>7000000].count()
#locate houses with a value above 7,000,000

data.loc[data['price'] > 7000000]
data = data[data.bedrooms != 33]

data = data[data.price < 6000000]
plt.figure(figsize=(18,10))



from scipy import stats

from scipy.stats import skew,norm

from scipy.stats.stats import pearsonr

# kernel density plot

sb.distplot(data.condition,fit=norm);

plt.ylabel =('Frequency')

plt.title = ('Condition Distribution');

(mu,sigma)= norm.fit(data['condition']);



#QQ plot

plt.figure(figsize=(18,10))

res = stats.probplot(data['condition'], plot=plt)

plt.show()



print("skewness: %f" % data['condition'].skew())

print("kurtosis: %f" % data ['condition'].kurt())
plt.figure(figsize=(18,10))



from scipy import stats

from scipy.stats import skew,norm

from scipy.stats.stats import pearsonr

# kernel density plot

sb.distplot(data.sqft_above,fit=norm);

plt.ylabel =('Frequency')

plt.title = ('Square Foot Above Distribution');

(mu,sigma)= norm.fit(data['sqft_above']);



#QQ plot

plt.figure(figsize=(18,10))

res = stats.probplot(data['sqft_above'], plot=plt)

plt.show()



print("skewness: %f" % data['sqft_above'].skew())

print("kurtosis: %f" % data ['sqft_above'].kurt())
plt.figure(figsize=(18,10))



from scipy import stats

from scipy.stats import skew,norm

from scipy.stats.stats import pearsonr

# kernel density plot

sb.distplot(data.sqft_living15,fit=norm);

plt.ylabel =('Frequency')

plt.title = ('Square Foot Living(2015) Distribution');

(mu,sigma)= norm.fit(data['sqft_living15']);



#QQ plot

plt.figure(figsize=(18,10))

res = stats.probplot(data['sqft_living15'], plot=plt)

plt.show()



print("skewness: %f" % data['sqft_living15'].skew())

print("kurtosis: %f" % data ['sqft_living15'].kurt())
plt.figure(figsize=(18,10))



from scipy import stats

from scipy.stats import skew,norm

from scipy.stats.stats import pearsonr

# kernel density plot

sb.distplot(data.sqft_living,fit=norm);

plt.ylabel =('Frequency')

plt.title = ('Square Foot Living Distribution');

(mu,sigma)= norm.fit(data['sqft_living']);



#QQ plot

plt.figure(figsize=(18,10))

res = stats.probplot(data['sqft_living'], plot=plt)

plt.show()



print("skewness: %f" % data['sqft_living'].skew())

print("kurtosis: %f" % data ['sqft_living'].kurt())
from scipy import stats

from scipy.stats import skew,norm

from scipy.stats.stats import pearsonr

# kernel density plot

plt.figure(figsize=(18,10))



sb.distplot(data.sqft_lot15,fit=norm);

plt.ylabel =('Frequency')

plt.title = ('Square Foot Lot(2015) Distribution');

(mu,sigma)= norm.fit(data['sqft_lot15']);



#QQ plot



plt.figure(figsize=(18,10))

res = stats.probplot(data['sqft_lot15'], plot=plt)

plt.show()



print("skewness: %f" % data['sqft_lot15'].skew())

print("kurtosis: %f" % data ['sqft_lot15'].kurt())


from scipy import stats

from scipy.stats import skew,norm

from scipy.stats.stats import pearsonr

# kernel density plot

plt.figure(figsize=(18,10))



sb.distplot(data.price,fit=norm);

plt.ylabel =('Frequency')

plt.title = ('Price Distribution');

(mu,sigma)= norm.fit(data['price']);



#QQ plot

plt.figure(figsize=(18,10))

res = stats.probplot(data['price'], plot=plt)

plt.show()





print("skewness: %f" % data['price'].skew())

print("kurtosis: %f" % data ['price'].kurt())
plt.figure(figsize=(18,10))



#log transform the target 

data["sqft_lot15"] = np.log1p(data["sqft_lot15"])



#Kernel Density plot

sb.distplot(data.sqft_lot15,fit=norm);

plt.ylabel=('Frequency')

plt.title=('Square Foot Lot(2015) distribution');

#Get the fitted parameters used by the function

(mu,sigma)= norm.fit(data['sqft_lot15']);







#QQ plot

plt.figure(figsize=(18,10))



res =stats. probplot(data['sqft_lot15'], plot=plt)

plt.show()

print("skewness: %f" % data['sqft_lot15'].skew())

print("kurtosis: %f" % data['sqft_lot15'].kurt())
plt.figure(figsize=(18,10))



#log transform the target 

data["price"] = np.log1p(data["price"])



#Kernel Density plot

sb.distplot(data.price,fit=norm);

plt.ylabel=('Frequency')

plt.title=('Price distribution');

#Get the fitted parameters used by the function

(mu,sigma)= norm.fit(data['price']);

plt.savefig('dist.png')





#QQ plot

plt.figure(figsize=(18,10))

res =stats. probplot(data['price'], plot=plt)

plt.show()

print("skewness: %f" % data['price'].skew())

print("kurtosis: %f" % data['price'].kurt())
#Initialize Linear Regression to a variable reg



reg=LinearRegression()
#Initialize the value to be predicted(label) as price



labels=data['price']
#convert date into a readable data-type by the algorithm

#since the date variable had only 2014 and 2015, the date column can be trasformed into a nominal category with 1 representing 2014 and 0 representing 2015.



conv_dates = [1 if values == 2014 else 0 for values in data.date ]

data['date']=conv_dates
#drop columns not used in training.

#id, yr_built, condition and long (longitute) are droped because the have low corelation/significance on the target.

#price is also droped since it is not used as part of the independent variables.



train1 = data.drop(['id', 'price','condition','yr_built','long'],axis=1)

#70%, 30% train, test split



x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.3,random_state =5)

#Fitting the regression algorithm with data from the train set.

#x_train represents the predictors (independent variables) and y_train represents the target.



reg.fit(x_train,y_train)
#Testing our accuracy.

acc1=reg.score(x_test,y_test)

print(str("The accuracy of the model is: "+str("%.2f" %(acc1*100))+"%"))

from sklearn.metrics import mean_squared_error

from math import sqrt
y_prediction1 = reg.predict(x_test)
RMSE_lin = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction1))
print(RMSE_lin)
from sklearn.ensemble import GradientBoostingRegressor 
gbr=GradientBoostingRegressor(n_estimators= 400, max_depth = 5, min_samples_split = 2, learning_rate = 0.08, loss = 'ls')
train2 = data.drop(['id', 'price','condition','yr_built','long'],axis=1)



#70%, 30% train, test split



x_train1 , x_test1 , y_train1 , y_test1 = train_test_split(train2 , labels , test_size = 0.3,random_state =5)

gbr.fit(x_train1,y_train1)
acc=gbr.score(x_test1,y_test1)

acc
acc2=("%.2f" % (acc*100))

acc2
print(str("The acccuracy of the model is: "+str(acc2)+"%"))
feature_importance = gbr.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5



plt.figure(figsize=(12,6))

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, x_train.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Feature Importance')

plt.show()
from sklearn.metrics import mean_squared_error

from math import sqrt
y_prediction = gbr.predict(x_test)

RMSE_gbr = sqrt(mean_squared_error(y_true = y_test1, y_pred = y_prediction))
print(RMSE_gbr)