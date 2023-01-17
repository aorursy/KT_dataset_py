import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
gapp = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

gapp.head()
print("Column names before clean-up:: ", gapp.columns)

gapp.columns = [colName.replace(" ","_") if(len(colName.split())>1) else colName for colName in gapp.columns]

print("======================================================================================================")

print("Column names after clean-up:: ", gapp.columns)
# Find the columns wise NaN counts

gapp.isnull().sum(axis = 0)
gappC = gapp.dropna()

print("Number of rows after cleaning:: ", gappC.shape[0])

gappC.isnull().sum(axis = 0)
#sns.distplot(gapp["Rating"]) # this will through an error because there are NaN

print("Unique values of ratings across apps:: ",gapp["Rating"].unique())

print("==================================================")

print("Number of rows before cleaning:: ", gapp.shape[0])
gappC = gappC[gappC.Rating != 19]

gappC["Rating"].unique()
#So apparently there are no NaN values 

#From now on we will use the cleaned data frame for our anlysis

#Let us c how the vlaues are distributed in ratings

plt.figure(figsize=(26, 10))

sns.distplot(gappC["Rating"])
print("Unique values in Price column :: ", gappC["Price"].unique()) # There are  $ signs so we can not plot them So let us remove them 

gappC["Price"] = gappC["Price"].replace({'\$':''}, regex = True).astype(float) # remove the $ sign and make it a numeric column

print("=============================================================================")

print("Unique values in price column post processing::", gappC["Price"].unique())
gappC["Price"].describe()
gappC.loc[gappC['Price'] >=100, ['App', "Price"]]
gappC["Type"].unique()
plt.figure(figsize=(26, 10))

sns.scatterplot(x = gappC.Price, y = gappC.Rating, s = 80)
cvrF = gappC[gappC.Price == 0] # Category Vs rating for free apps

plt.figure(figsize=(26, 10))

plt.xticks(rotation=90, horizontalalignment='right')

sns.scatterplot(x = cvrF.Category, y = cvrF.Rating, hue=cvrF.Content_Rating, s = 80)
cvrP = gappC[gappC.Price != 0] # Category Vs rating for free apps

plt.figure(figsize=(26, 10))

plt.xticks(rotation=90, horizontalalignment='right')

sns.scatterplot(x = cvrP.Category, y = cvrP.Rating, hue=cvrP.Content_Rating, s = 80)
plsf = gappC[(gappC.Price > 0) & (gappC.Price < 75) ] # Paid apps less than 75

plt.figure(figsize=(26, 10))

plt.xticks(rotation=90, horizontalalignment='right')

sns.scatterplot(x = plsf.Category, y = plsf.Rating, hue=plsf.Content_Rating, s = 80)
plt.figure(figsize=(26, 10))

plt.xticks(rotation=90, horizontalalignment='right')

sns.boxplot(x = plsf.Category, y = plsf.Rating, hue=plsf.Content_Rating)
pgtf = gappC[gappC.Price > 250] # Paid apps greater than 250

plt.figure(figsize=(26, 10))

plt.xticks(rotation=90, horizontalalignment='right')

sns.scatterplot(x = pgtf.Category, y = pgtf.Rating, hue=pgtf.Content_Rating, s = 80)
gappC[gappC.Price >100].shape[0]

# so there are only 15 Applications above 100$.
fig, axs = plt.subplots(2,2,figsize=(15, 10))

sns.distplot(gappC[(gappC.Price >= 0.0)&(gappC.Price <= 0.99)].Price, ax=axs[0,0])

sns.distplot(gappC[(gappC.Price >= 1.00)&(gappC.Price <= 10)].Price, ax=axs[0,1])

sns.distplot(gappC[(gappC.Price >= 11.00)&(gappC.Price <= 100)].Price, ax=axs[1,0])

sns.distplot(gappC[gappC.Price >= 101.00].Price, ax=axs[1,1])





fig.suptitle("App distribution by Price segments", fontsize=16)

axs[0, 0].set_title('Free apps and below 1$')

axs[0, 1].set_title('Apps Priced below 10 $ excluding free')

axs[1, 0].set_title('Apps Priced below 100$ and above 11')

axs[1, 1].set_title('Apps Priced above 100$')

gappC.groupby(['Category'])['Category'].count().sort_values(ascending = False).head(10)
plt.figure(figsize=(26, 10))

plt.xticks(rotation=90, horizontalalignment='right')

plt.title('Count of app in each category',size = 20)

g = sns.countplot(x="Category",data=gappC, palette = "Set1")

g 
gappC.groupby(['Category'])['Price'].max().sort_values(ascending = False).head(10)
#Ratings by Category

plt.figure(figsize=(26, 10))

plt.xticks(rotation=90, horizontalalignment='right')

sns.boxplot(x="Category", y="Rating", data=gappC)
gappC.info()
gappC.head(5)
gappC['Size'] = gappC['Size'].map(lambda x: x.rstrip('M'))

gappC['Size'] = gappC['Size'].map(lambda x: str(round((float(x.rstrip('k'))/1024), 1)) if x[-1]=='k' else x)

gappC['Size'] = gappC['Size'].map(lambda x: np.nan if x.startswith('Varies') else x)
gappC['Installs'] = gappC['Installs'].map(lambda x: x.rstrip('+'))

gappC['Installs'] = gappC['Installs'].map(lambda x: ''.join(x.split(',')))
gappC.info()
gappC['Reviews'] = gappC['Reviews'].apply(lambda x: float(x))

gappC['Size'] = gappC['Size'].apply(lambda x: float(x))

gappC['Installs'] = gappC['Installs'].apply(lambda x: float(x))
gappC.info()
gappC.isnull().sum(axis = 0)
gappDF = gappC.dropna()
gappDFM = gappDF.drop(['App', 'Category', 'Type', 'Content_Rating', 'Genres', 'Last_Updated', 'Current_Ver', 'Android_Ver'], axis = 1)

gappDFM.head()
plt.figure(figsize=(15, 5))

sns.regplot(x = gappDFM.Rating, y = gappDFM.Reviews)
plt.figure(figsize=(15, 5))

sns.regplot(x = gappDFM.Rating, y = gappDFM.Installs)
plt.figure(figsize=(15, 5))

sns.regplot(x = gappDFM.Rating, y =gappDFM.Size )
from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split
#for evaluation of error term  

def evalMat(y_true, y_predict):

    print ('Mean Squared Error: '+ str(metrics.mean_squared_error(y_true,y_predict)))

    print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(y_true,y_predict)))

    print ('Mean squared Log Error: '+ str(metrics.mean_squared_log_error(y_true,y_predict)))
def evalMat_dict(y_true, y_predict, name = 'Linear Regression'):

    dict_matrix = {}

    dict_matrix['Regression Method'] = name

    dict_matrix['Mean Squared Error'] = metrics.mean_squared_error(y_true,y_predict)

    dict_matrix['Mean Absolute Error'] = metrics.mean_absolute_error(y_true,y_predict)

    dict_matrix['Mean Squared Log Error'] = metrics.mean_squared_log_error(y_true,y_predict)

    return dict_matrix
x = gappDFM.drop(["Rating"], axis = 1)

y = gappDFM.Rating
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
linMod = LinearRegression()

linMod.fit(x_train, y_train)

linRes = linMod.predict(x_test)
resDF = pd.DataFrame()

resDF = resDF.from_dict(evalMat_dict(y_test,linRes),orient = 'index')

resDF = resDF.transpose()
resDF.head()
print('Actual mean of Population:: ', y.mean())

print('Predicted mean:: ', linRes.mean())
plt.figure(figsize = (12, 6))

sns.regplot(linRes, y_test, marker = 'x')

plt.title('Linear model')

plt.xlabel('Predicted Ratings')

plt.ylabel('Actual Ratings')

plt.show()
linMod.coef_
linMod.intercept_
from sklearn import svm



svrMod = svm.SVR(gamma='auto')

svrMod.fit(x_train, y_train)

svrRes = svrMod.predict(x_test)
print('Actual mean of Population:: ', y.mean())

print('Predicted mean:: ', svrRes.mean())
resDF = resDF.append(evalMat_dict(y_test, svrRes, name = "SVR"), ignore_index = True)
resDF
plt.figure(figsize = (12, 6))

sns.regplot(svrRes, y_test, marker = 'x')

plt.title('SVR model')

plt.xlabel('Predicted Ratings')

plt.ylabel('Actual Ratings')

plt.show()
from sklearn.ensemble import RandomForestRegressor

rfrMod = RandomForestRegressor()

rfrMod.fit(x_train, y_train)

rfrRes = rfrMod.predict(x_test)
print('Actual mean of Population:: ', y.mean())

print('Predicted mean:: ', rfrRes.mean())
resDF = resDF.append(evalMat_dict(y_test, rfrRes, name = "RFR"), ignore_index = True)

resDF
plt.figure(figsize = (12, 6))

sns.regplot(rfrRes, y_test, marker = 'x')

plt.title('RFR model')

plt.xlabel('Predicted Ratings')

plt.ylabel('Actual Ratings')

plt.show()