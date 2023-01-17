#import some necessary librairies



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style='darkgrid', context='notebook', palette='viridis')

sns.despine(top=True,right=True)



from scipy import stats

from scipy.stats import norm, skew

from scipy.special import boxcox1p



from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler,StandardScaler,LabelEncoder

from sklearn import metrics

from sklearn.model_selection import train_test_split
# import the train and test datasets 



train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')
#check the numbers of samples and features for train and test

print("Train data set: %i " % train.shape[0],"samples ",train.shape[1],"features")

print("Test data set: %i " % test.shape[0],"samples ",test.shape[1],"features")
# check first 5 rows of train 

train.head()
# check first 5 rows of test 

test.head()
train.info()
test.info()
#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it is unnecessary for the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
#check again the numbers of samples and features

print("train: %i " % train.shape[0],"samples ",train.shape[1],"features")

print("test: %i " % test.shape[0],"samples ",test.shape[1],"features")
#seaborn boxplot depciting target variable

fig = sns.boxplot(data=train[['SalePrice']],width = 0.2)

fig.axis(ymax=800000,ymin=0);



print("Median: %f" % train['SalePrice'].median())

train['SalePrice'].describe()
fig = plt.figure(figsize=(9,4))

sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

fig = plt.figure(figsize=(9,4))

res = stats.probplot(train['SalePrice'], plot=plt)



print("kurtosis: %f" % train['SalePrice'].kurtosis())

fig = plt.figure(figsize=(9,4))

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])



#Check the new distribution 

sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

fig = plt.figure(figsize=(9, 4))

prob = stats.probplot(train['SalePrice'], plot=plt)



print("kurtosis: %f" % train['SalePrice'].kurtosis())

#ntrain = train.shape[0]

#ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
#Correlation map to see how features are correlated with SalePrice

plt.figure(figsize=[25,12])



corrmat = round(train.corr(),2)

sns.heatmap(corrmat, annot=True,linewidths=0.5,cmap ="YlGnBu" )


k=7

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

plt.figure(figsize=(8,8))

sns.set(font_scale=1.25)

sns.heatmap(cm, cbar=True,linewidths=1.5, annot=True, square=True,cmap ="YlGnBu", fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#Drop columns 'GarageArea','1stFlrSF','Utilities'

train.drop(['GarageArea','1stFlrSF','Utilities'],axis =1, inplace = True)

test.drop(['GarageArea','1stFlrSF','Utilities'],axis =1, inplace = True)
sns.set(style = 'darkgrid', context = 'notebook',palette='viridis')

cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF']

sns.pairplot(train[cols], height =2.5)

plt.show()
correlTable = pd.DataFrame(train.corr().abs()['SalePrice'].sort_values(ascending=False))

correlTable[correlTable['SalePrice'] > 0.6]
fig = plt.figure(figsize=(12, 6))

cmap = sns.color_palette("husl", n_colors=10)



sns.scatterplot(x=train['GrLivArea'], y='SalePrice', hue='OverallQual', palette=cmap, data=train)



plt.xlabel('GrLivArea', size=15)

plt.ylabel('SalePrice', size=15)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12) 

    

plt.title('GrLivArea & OverallQual vs SalePrice', size=15, y=1.05)



plt.show()
plt.subplots(figsize=(15, 5))



plt.subplot(1, 2, 1)

g = sns.regplot(x=train['GrLivArea'], y=train["SalePrice"], fit_reg=False).set_title("Before")





#Delete outliers

plt.subplot(1, 2, 2) 

train = train.drop(train[(train['GrLivArea']>4650)].index)

g = sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], fit_reg=False).set_title("After")
plt.subplots(figsize=(15, 5))



plt.subplot(1, 2, 1)

g = sns.regplot(x=train['BsmtFinSF1'], y=train["SalePrice"], fit_reg=False).set_title("Before")

plt.subplots(figsize=(15, 5))



plt.subplot(1, 2, 1)

g = sns.regplot(x=train['GarageCars'], y=train["SalePrice"], fit_reg=False).set_title("Before")
plt.subplots(figsize=(15, 5))



plt.subplot(1, 2, 1)

g = sns.regplot(x=train['TotalBsmtSF'], y=train["SalePrice"], fit_reg=False).set_title("Before")
train = train.drop(train[(train['SalePrice']>=13.5)].index)
#Update the target variable dataframe before concatenating the train and test data sets.

y_train = train.SalePrice.values



#Recreate the concatenated All Data variable with the updated train data set

ntrain = train.shape[0]

ntest = test.shape[0]

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)



#percentage missing Data 

total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



#show missing values

missing_data[missing_data['Total']>0]



#Barplot of missing data

missing_data = missing_data[missing_data['Total']>0]



#Bar charts for missing data

f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'],palette='plasma')



plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
#dealing with missing data

all_data = all_data.drop((missing_data[missing_data['Total'] > 2348]).index,1)

all_data.isnull().sum().max() #just checking that there's no missing data missing...

# checking the new percentage of missing Data 

total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(30)
none = ['Fence','FireplaceQu','GarageType','MasVnrType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MSSubClass']

zero = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea','GarageYrBlt', 'GarageCars']

mode = ['MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType']



#imputing the remaining missing values with most applicable values

for col in none:

    all_data[col] = all_data[col].fillna('None')

for col in zero:

    all_data[col] = all_data[col].fillna(0)

for col in mode:

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    

#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

    

#Data description says NA means typical 

all_data["Functional"] = all_data["Functional"].fillna("Typ")

print("Train:", train.shape)

print("Sales:", y_train.shape)
#Check remaining missing values if any 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing percentage' :all_data_na})

print(missing_data.count())
#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)



#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)


cat_cols = ('FireplaceQu', 'MoSold' ,'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street','CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'ExterCond')



# process columns, apply LabelEncoder to categorical features

for c in cat_cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))
# Adding total sqfootage feature ,excluding 1stFlrSF since it was removed due to the correlation 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['2ndFlrSF']
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(15)
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam)

    

#all_data[skewed_features] = np.log1p(all_data[skewed_features])
#Get dummy variables while avoiding the dummy trap

all_data = pd.get_dummies(all_data,drop_first=True)

print(all_data.shape)
total=all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()*100/all_data.isnull().count()).sort_values(ascending=False)

#types=train.dtypes

DataMissing=pd.concat([total,percent],axis=1,keys=['total','percentage'])



#show missing values

DataMissing[DataMissing['total']>0]
train = all_data[:ntrain]

test = all_data[ntrain:]





print("Train data set: %i " % train.shape[0],"samples ",train.shape[1],"features")

print("Test data set: %i " % test.shape[0],"samples ",test.shape[1],"features\n")



X = train

y = y_train



print("X:", X.shape)

print("y:", y.shape)



print("\nTrain data set: %i " % X.shape[0],"samples ",X.shape[1],"features")

print("Test data set: %i " % test.shape[0],"samples ",test.shape[1],"features\n")
X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,shuffle=False)
#Create regression object

lm = LinearRegression()



#Fit model

lm.fit(X_train, y_train)



#Y intercept of regression model

b = float(lm.intercept_)



#Coefficients of the features

coeff = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
# Plot important coefficients

fig = plt.figure(figsize=(8,6))

coefs = pd.Series(lm.coef_, index = X_train.columns)

print("Multivariate OLS Regression picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")



plt.title("Coefficients in the Multivariate OLS")

plt.show()



print("y-intercept:", b)
from sklearn import metrics



predictedTrainPrices = lm.predict(X_train)

trainR2 = metrics.r2_score(y_train, predictedTrainPrices)

predictedTestPrices = lm.predict(X_test)

testR2 = metrics.r2_score(y_test, predictedTestPrices)



print("Trained R-squared: ",trainR2)

print("Test R-squared: ",testR2)
# Plot residuals

sns.scatterplot(x=predictedTrainPrices, y=predictedTrainPrices - y_train, palette=cmap, data=train,label = "Training data")

sns.scatterplot(x=predictedTestPrices, y=predictedTestPrices - y_test, palette=cmap, data=test,label = "Test data")

plt.title("Multivariate Regression")

plt.xlabel("Predicted SalePrice values")

plt.ylabel("Residuals")



plt.legend(loc = "best")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X.values),columns = X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,test_size=0.20,shuffle=False)

X_train = pd.DataFrame(X_train)
#Create ridge object

ridge = Ridge()



#Fit Ridge model

ridge.fit(X_train, y_train)





#Y intercept of regression model

b = float(lm.intercept_)



#Coefficients of the features

coeff = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
type(X_train)
# Plot important coefficients

fig = plt.figure(figsize=(8,6))

coefs = pd.Series(ridge.coef_, index = X_train.columns)

print("Ridge Regression picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the Ridge Regression")

plt.show()



print("y-intercept:", b)
predictedTrainPrices = ridge.predict(X_train)

trainR2 = metrics.r2_score(y_train, predictedTrainPrices)

predictedTestPrices = ridge.predict(X_test)

testR2 = metrics.r2_score(y_test, predictedTestPrices)



print("Trained R-squared: ",trainR2)

print("Test R-squared: ",testR2)
# Plot residuals

sns.scatterplot(x=predictedTrainPrices, y=predictedTrainPrices - y_train, palette=cmap, data=train,label = "Training data")

sns.scatterplot(x=predictedTestPrices, y=predictedTestPrices - y_test, palette=cmap, data=test,label = "Test data")

plt.title("Ridge Regression")

plt.xlabel("Predicted SalePrice values")

plt.ylabel("Residuals")



plt.legend(loc = "best")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()


#Create ridge object

lasso = Lasso(alpha=0.01)



#Fit Lasso model

lasso.fit(X_train, y_train)



intercept = float(lasso.intercept_)



coeff = pd.DataFrame(lasso.coef_, X.columns, columns=['Coefficient'])
# Plot important coefficients

fig = plt.figure(figsize=(8,6))

coefs = pd.Series(lasso.coef_, index = X_train.columns)

print("Ridge Regression picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the Ridge Regression")

plt.show()



print("y-intercept:", b)
predictedTrainPrices = lasso.predict(X_train)

trainR2 = metrics.r2_score(y_train, predictedTrainPrices)

predictedTestPrices = lasso.predict(X_test)

testR2 = metrics.r2_score(y_test, predictedTestPrices)



print("Trained R-squared: ",trainR2)

print("Test R-squared: ",testR2)
# Plot residuals

sns.scatterplot(x=predictedTrainPrices, y=predictedTrainPrices - y_train, palette=cmap, data=train,label = "Training data")

sns.scatterplot(x=predictedTestPrices, y=predictedTestPrices - y_test, palette=cmap, data=test,label = "Test data")

plt.title("Lasso Regression")

plt.xlabel("Predicted SalePrice values")

plt.ylabel("Residuals")



plt.legend(loc = "best")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
#Dataframe which will be used to store resulsts of model fits

LassoResultTable = pd.DataFrame(columns=['alphaVar', 'TrainR', 'TestR'])



for alphaV in np.arange(0.001, 0.010, 0.001):



    lasso = Lasso(alpha=alphaV)

    lasso.fit(X_train, y_train)



    predictedTrainPrices = lasso.predict(X_train)

    trainR2 = metrics.r2_score(y_train, predictedTrainPrices)

    predictedTestPrices = lasso.predict(X_test)

    testR2 = metrics.r2_score(y_test, predictedTestPrices)

    LassoResultTable = LassoResultTable.append(pd.Series([alphaV, round(trainR2,6), round(testR2,6)], index=LassoResultTable.columns ), ignore_index=True)



LassoResultTable
#Dataframe which will be used to store resulsts of model fits

RandomForestResultTable = pd.DataFrame(columns=['alphaVar', 'TrainR', 'TestR'])

estimatorsList = [10,50,100]



for var in estimatorsList:

    

    RFregressor = RandomForestRegressor(n_estimators = var,criterion = 'mse', random_state =0)

    #Whether features are regularized or not does not impact Random Forests

    RFregressor.fit(X_train,y_train)

    Trainpredict = RFregressor.predict(X_test)



    predictedTrainPrices = RFregressor.predict(X_train)

    trainR2 = metrics.r2_score(y_train, predictedTrainPrices)

    predictedTestPrices = RFregressor.predict(X_test)

    testR2 = metrics.r2_score(y_test, predictedTestPrices)

    RandomForestResultTable = RandomForestResultTable.append(pd.Series([var, trainR2, testR2], index=RandomForestResultTable.columns ), ignore_index=True)
RandomForestResultTable
# Plot residuals

sns.scatterplot(x=predictedTrainPrices, y=predictedTrainPrices - y_train, palette=cmap, data=train,label = "Training data")

sns.scatterplot(x=predictedTestPrices, y=predictedTestPrices - y_test, palette=cmap, data=test,label = "Test data")

plt.title("Random Forest Regression")

plt.xlabel("Predicted SalePrice values (n_estimators = 100)")

plt.ylabel("Residuals")



plt.legend(loc = "best")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
test_scaled = scaler.fit_transform(test)
test_lasso = lasso.predict(test_scaled)

test_lasso
predict = np.exp(test_lasso)
#Prepare Submission File

sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = predict

sub.to_csv('submit.csv',index=False)
sub