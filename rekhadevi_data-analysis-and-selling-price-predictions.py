import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Reading dataset

car_file= pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")

car_file.head()
# Rows and columns in dataset

car_file.shape

rw,clm= car_file.shape

print(f'There are {rw} rows and {clm} columns in our cars dataset.')
# Checking any null entry in dataset

nullvalue = car_file.isnull().any()

nullvalue
# checking datatypes

car_file.dtypes
car_file['Fuel_Type'].value_counts()
car_file['Seller_Type'].value_counts()
car_file['Transmission'].value_counts()
car_file['Seller_Type'].dtypes
# Converting Fuel_Type,Seller Type and Transmission Columns from Object to numeric datatypes to make it machne readable form.

car_file["Fuel_Type"]=car_file["Fuel_Type"].replace({'CNG':0,'Diesel':1,'Petrol':2})

car_file['Seller_Type']=car_file['Seller_Type'].replace({'Dealer':0,'Individual':1})

car_file["Transmission"]=car_file["Transmission"].replace({'Manual':0,'Automatic':1})
car_file.dtypes
car_file.columns
# Correlation between various columns

ax=car_file.corr()

plt.figure(figsize=(10,8))

sns.heatmap(ax,cmap="Blues",vmin=0, vmax=1)
# Let us ckeck relationship of some selected columns with the Selling Price

df=car_file[['Year','Selling_Price','Present_Price','Kms_Driven']]

sns.pairplot(df)
# There are five preset seaborn themes: darkgrid, whitegrid, dark, white, and ticks

#The four preset contexts, in order of relative size, are paper, notebook, talk, and poster.By default noetbook is used.

sns.set_style("ticks")

sns.set_context("talk")
#Plotting relationship between Selling Price vs. Present Price

plt.figure(figsize=(8,5))

x=car_file['Selling_Price']

y=car_file['Present_Price']



plt.title('Selling Price vs. Present Price')

sns.scatterplot(x,y,color='c')
#Plotting relationship between Selling Price vs. Year

sns.set_context("talk")

plt.figure(figsize=(8,5))

x=car_file['Selling_Price']

y=car_file['Year']



plt.title('Selling Price vs Year')

sns.scatterplot(x,y,color='r')
#Plotting relationship between Selling Price vs. Kilometers Driven

sns.set_context("talk")

plt.figure(figsize=(8,5))

y=car_file['Selling_Price']

x=car_file['Kms_Driven']



plt.title('Selling Price vs. Kilometers Driven')

sns.scatterplot(x,y,color='y')
#Plotting relationship between Selling Price vs. Kilometers Driven

sns.set_context("talk")

plt.figure(figsize=(8,5))

x=car_file['Selling_Price']

y=car_file['Fuel_Type']



plt.title('Selling Price vs. Kilometers Driven')

sns.scatterplot(x,y,color='c')
car_file.dtypes
# Selling Price vs Fuel Type and Transmission Type

sns.set_context("talk")

plt.figure(figsize=(10,10))

x= car_file['Selling_Price']

y= car_file['Fuel_Type']

x1= car_file['Selling_Price']

y1= car_file['Transmission']

sns.jointplot(x,y,data=car_file,kind='hex',color='yellow')

sns.jointplot(x1,y1,data=car_file,kind='hex')
# Selling Price vs Seller_Type

sns.set_context("talk")

plt.figure(figsize=(10,10))

x= car_file['Selling_Price']

y= car_file['Seller_Type']

sns.jointplot(x,y,data=car_file,kind='reg',color='yellow')
#Plotting relationship between Selling Price vs Number of Owners

sns.set_context("talk")

plt.figure(figsize=(8,5))

x=car_file['Selling_Price']

y=car_file['Owner']



plt.title('Selling Price vs Owner')

sns.scatterplot(x,y,color='g')
most_sell_cars= car_file.groupby(['Car_Name']) ['Selling_Price'].nunique().sort_values(ascending=False).head(10)

most_sell_cars
top_sellpric_cars= car_file.groupby(['Car_Name']) ['Selling_Price'].sum().sort_values(ascending=False).head(10)

top_sellpric_cars
# Plotting relationship between Various cars w.r.t their selling counts and selling price. 

sns.set_style("darkgrid")

sns.set_context("notebook")

plt.figure(figsize=(15,12))



x=most_sell_cars.index

y=most_sell_cars.values



plt.subplot(2,1,1)

plt.bar(x,y,color='r')

plt.xlabel('Car Name')

plt.ylabel('No. of Cars sold')

plt.title('Mostly Sold Cars')



x1=top_sellpric_cars.index

y1=top_sellpric_cars.values



plt.subplot(2,1,2)

plt.bar(x1,y1,color='b')

plt.xlabel('Car Name')

plt.ylabel('Selling Price earned(Lacs)')

plt.title('Top Selling Price Cars')
car_file.columns
car_file['Price Reduction']= car_file['Present_Price']-car_file['Selling_Price'] # Inserting a new column of reduced price

top_reduc= car_file.groupby('Car_Name')['Price Reduction'].mean().sort_values(ascending=False).head(20)

top_reduc
car_file['Car_Name'].value_counts()
fort=car_file.query("Car_Name=='fortuner'")

fort.head()
corolla_alt=car_file.query("Car_Name == 'corolla altis'")

corolla_alt.head()
# Selling Price vs. Year



#sns.set_style("ticks")

#sns.set_context("notebook")

plt.figure(figsize=(15,10))



# For Fortuner

fort=car_file.query("Car_Name=='fortuner'")

x2= fort['Year']

y2= fort['Price Reduction']

plt.subplot(211)

plt.bar(x2,y2,color='g')

plt.xlabel('Year')

plt.ylabel('Price Reduction')

plt.title('Price reduction of Fortuner over year')



# For Corolla Altis

corolla_alt=car_file.query("Car_Name == 'corolla altis'")

corolla_alt

x4=corolla_alt['Year']

y4=corolla_alt['Price Reduction']

plt.subplot(212)

plt.bar(x4,y4,color='c')

plt.xlabel('Year')

plt.ylabel('Price Reduction')

plt.title('Price reduction of Corolla Altis over year')
#plt.style.available
sns.set_style("darkgrid")

sns.set_context("talk")

plt.figure(figsize=(8,5))

x=car_file['Price Reduction']

y=car_file['Kms_Driven']

plt.title('Price Reduction vs Kilometers Driven')

sns.scatterplot(x,y,color='g')
#Price Depreciation vs Other Columns



#sns.set_style("ticks")

#sns.set_context("talk")

plt.figure(figsize=(15,10))

y1=car_file['Price Reduction']

x1=car_file['Owner']

plt.subplot(221)

plt.title('Price Reduction vs No. of Car Owners')

sns.barplot(x1,y1,color='y')



y2=car_file['Price Reduction']

x2=car_file['Fuel_Type']

plt.subplot(222)

sns.barplot(x2,y2,color='c')

plt.title('Price Reduction vs Fuel Type')



y3=car_file['Price Reduction']

x3=car_file['Seller_Type']

plt.subplot(223)

plt.title('Price Reduction vs Seller Type')

sns.barplot(x3,y3,color='r')





y4=car_file['Price Reduction']

x4=car_file['Transmission']

plt.subplot(224)

sns.barplot(x4,y4)

plt.title('Price Reduction vs Transmission')
# Plotting  a 3d relationship of Selling Price w.r.t Year, Present Price and Kms_Driven

sns.set_style("white")

sns.set_context("notebook")

from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(15,9))

ax  = fig.gca(projection = "3d")



plot =  ax.scatter(car_file["Year"],car_file["Present_Price"],car_file["Kms_Driven"],linewidth=1,edgecolor ="k",c=car_file["Selling_Price"],s=100,cmap='twilight_r')

ax.set_xlabel("Year")

ax.set_ylabel("Present_Price")

ax.set_zlabel("Kms_Driven")



lab = fig.colorbar(plot,shrink=.5,aspect=5)

lab.set_label("Selling_Price",fontsize = 15)



plt.title("Relationship between Selling Price,Year,Present price and Kilometers driven",fontsize = 20)
car_file.head(5)
#Let us separate input and output variables.

x= car_file.loc[:,car_file.columns!='Selling_Price']

#x

y=car_file['Selling_Price']

#y
x.dtypes
#Changing Car_Name Column datatype to numeric

from sklearn.preprocessing import LabelEncoder

lb_enc= LabelEncoder()

x.loc[:,'Car_Name']= lb_enc.fit_transform(x.loc[:,'Car_Name'])

x.loc[:,'Fuel_Type']= lb_enc.fit_transform(x.loc[:,'Fuel_Type'])

x.loc[:,'Seller_Type']= lb_enc.fit_transform(x.loc[:,'Seller_Type'])

x.loc[:,'Transmission']= lb_enc.fit_transform(x.loc[:,'Transmission'])
#Checking for nan Values

x.isnull().any()
y.isnull().any() #no null value in both x and y.
# Further splitting our data in training and test datsets

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.20) #Setting 20% data for test purpose

print("x_train",x_train)

print("\n x_test",x_test)



print("y_train",y_train)

print("\n y_test",y_test)
#Linear Regression Algorithm

from sklearn.linear_model import LinearRegression

lnr= LinearRegression()

lnr.fit(x_train,y_train)



#Prediction of output y on the basis of x training data

y_predic= lnr.predict(x_test)

y_predic #y_test values predicted by machine
# Checking Accuracy

from sklearn.metrics import accuracy_score

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt



RMSE_LinearAlgo=np.sqrt(metrics.mean_squared_error(y_test,y_predic))

print("Root mean squared error of Linear Regression is : ",RMSE_LinearAlgo)
#Decision Tree with Deafault parameters

from sklearn.tree import DecisionTreeRegressor

dt= DecisionTreeRegressor() # By default takes gini criterion

dt.fit(x_train,y_train)



#Predicting Output

y_predic2= dt.predict(x_test)

y_predic2
#Accuracy Check  for Decision_Tree with no parameters

RMSE_DecisionTreeAlgo= np.sqrt(metrics.mean_squared_error(y_test,y_predic2))

#RMSE_DecisionTreeAlgo

print("Root mean squared error of Decision Tree Algorithm with default parameters is : ",RMSE_DecisionTreeAlgo)
#Decision Tree with parameters

dt1= DecisionTreeRegressor(criterion='mse',max_depth=8,random_state=3)

print("Decision Tree with parameters: Criteria-mse\n",dt1.fit(x_train,y_train))



#Predicting Output

y_predic3= dt1.predict(x_test)

y_predic3
#Accuracy Check

RMSE_DecisionTreeAlgo1=np.sqrt(metrics.mean_squared_error(y_test,y_predic3))

#RMSE_DecisionTreeAlgo1

print("Root mean squared error of Decision Tree Algorithm with parameters is : ",RMSE_DecisionTreeAlgo1)
car_file.columns
#Plotting tree

from sklearn import tree

from sklearn.tree import export_graphviz 

data_feature= ['Car_Name','Year','Present_Price','Kms_Driven',

               'Fuel_Type','Seller_Type', 'Transmission','Owner',"Price Reduction"]

from graphviz import Source

from IPython.display import SVG

from IPython.display import display



graph = Source(tree.export_graphviz(dt,out_file=None, feature_names=data_feature, filled = True,rounded=True))

display(SVG(graph.pipe(format='svg')))
from sklearn.ensemble import RandomForestRegressor

rnd_clf= RandomForestRegressor(n_estimators=9, random_state=3, max_depth=8, criterion = 'mse')

rnd_clf.fit(x_train,y_train)



#Output Predicting

y_predic4= rnd_clf.predict(x_test)

y_predic4



#Accuracy Check for Random Forest Algorithm

RMSE_RndmForAlgo=np.sqrt(metrics.mean_squared_error(y_test,y_predic4))

RMSE_RndmForAlgo
estimators=rnd_clf.estimators_[7] # gives 5 decision trees

data_feature= ['Car_Name','Year','Present_Price','Kms_Driven',

               'Fuel_Type','Seller_Type', 'Transmission','Owner','Price Reduction']

from sklearn import tree

from graphviz import Source

from IPython.display import SVG  #SVG format

from IPython.display import display 



graph = Source(tree.export_graphviz(estimators,out_file=None,feature_names=data_feature,filled = True))

display(SVG(graph.pipe(format='svg')))
Algorithms = ["Linear Regression","DecisionTree Regressor(no parameters)","DecisionTree Regresssor(with parameters)","Random Forest Regressor"]

print(Algorithms)

RMSE = [RMSE_LinearAlgo,RMSE_DecisionTreeAlgo,RMSE_DecisionTreeAlgo1,RMSE_RndmForAlgo]

print(RMSE)



#Creating dataframe of Algorithms vs their RMSE values

Algo_vs_RMSE= pd.DataFrame({'Algorithm':Algorithms,'RMSE Values(without Scaling)':RMSE})

Algo_vs_RMSE
from sklearn.preprocessing import MinMaxScaler 

scalar = MinMaxScaler()

x_train1 = scalar.fit_transform(x_train)

x_test1 = scalar.transform(x_test)

#print(x_train1)

#print(x_test1)
# Applying Linear Regression

lnr1= LinearRegression()

lnr1.fit(x_train1,y_train)



#Output Prediction

y_predict_L1=lnr1.predict(x_test1)

y_predict_L1



#Accuracy Check

RMSE_LinearAlgo1= np.sqrt(metrics.mean_squared_error(y_test,y_predict_L1))

RMSE_LinearAlgo1
#Decision Tree Algorithm with Default Parameters

dt_min= DecisionTreeRegressor()

dt_min.fit(x_train1,y_train)



#Output Prediction

y_predict_DT1=dt_min.predict(x_test1)

y_predict_DT1



#Accuracy Check

RMSE_DecisionTreeAlgo2= np.sqrt(metrics.mean_squared_error(y_test,y_predict_DT1))

RMSE_DecisionTreeAlgo2
#Decision Tree Algorithm with Parameters : criterion='mse',max_depth=8,random_state=3)

dt_min1= DecisionTreeRegressor(criterion='mse',max_depth=7,random_state=3)

dt_min1.fit(x_train1,y_train)



#Output Prediction

y_predict_DT2=dt_min1.predict(x_test1)

y_predict_DT2



#Accuracy Check

RMSE_DecisionTreeAlgo3= np.sqrt(metrics.mean_squared_error(y_test,y_predict_DT2))

RMSE_DecisionTreeAlgo3
#RandomForest Algorithm

rnd_clf_min= RandomForestRegressor(n_estimators=9, random_state=3, max_depth=7, criterion = 'mse')

rnd_clf_min.fit(x_train1,y_train)



#Output Prediction

y_predict_RF1=rnd_clf_min.predict(x_test1)

y_predict_RF1



#Accuracy Check

RMSE_RndmForAlgo1= np.sqrt(metrics.mean_squared_error(y_test,y_predict_RF1))

RMSE_RndmForAlgo1
estimators=rnd_clf.estimators_[7] # gives 5 decision trees

data_feature= ['Car_Name','Year','Present_Price','Kms_Driven',

               'Fuel_Type','Seller_Type', 'Transmission','Owner','Price Reduction']

from sklearn import tree

from graphviz import Source

from IPython.display import SVG  #SVG format

from IPython.display import display 



graph = Source(tree.export_graphviz(estimators,out_file=None,feature_names=data_feature,filled = True))

display(SVG(graph.pipe(format='svg')))
#Checking RMSE errors with different algorithms using Min-Max Scalar

Algorithms_MinMaxScalar = ["Linear Regression","DecisionTree Regressor(with no parameters)","DecisionTree Regresssor(with parameters)","Random Forest Regressor"]

RMSE1 = [RMSE_LinearAlgo1,RMSE_DecisionTreeAlgo2,RMSE_DecisionTreeAlgo3,RMSE_RndmForAlgo1]

print(Algorithms_MinMaxScalar)

print(RMSE1)



#Creating dataframe of Algorithms vs their RMSE values(Min-Max Scalar)

Algo_vs_RMSE1= pd.DataFrame({'Algorithm':Algorithms,'RMSE Values(with Min-Max Scaling)':RMSE1})

Algo_vs_RMSE1
from sklearn.preprocessing import StandardScaler

independent_scalar = StandardScaler()

x_train2 = independent_scalar.fit_transform(x_train)

x_test2 = independent_scalar.transform(x_test)

#print(x_train2)

#print(x_test2)



#Dimenstionality Reduction Using PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=4)

xtrainPCA=pca.fit_transform(x_train2)

xtestPCA= pca.transform(x_test2)

print("Old dimension of training dataset:",x_train2.shape)

print("Reduced Dimension of training datset:",xtrainPCA.shape)
print(pca.explained_variance_ratio_)
# Applying Algorithms



#Linear Regression

# Applying Linear Regression

lnr3= LinearRegression()

lnr3.fit(xtrainPCA,y_train)



#Output Prediction

y_predict_L2=lnr3.predict(xtestPCA)

y_predict_L2



#Accuracy Check

RMSE_LinearAlgo2= np.sqrt(metrics.mean_squared_error(y_test,y_predict_L2))

RMSE_LinearAlgo2
#Decision Tree Algorithm with Default Parameters

dt_std= DecisionTreeRegressor()

dt_std.fit(xtrainPCA,y_train)



#Output Prediction

y_predict_DT3=dt_std.predict(xtestPCA)

y_predict_DT3



#Accuracy Check

RMSE_DecisionTreeAlgo4= np.sqrt(metrics.mean_squared_error(y_test,y_predict_DT2))

RMSE_DecisionTreeAlgo4
#Decision Tree Algorithm with Parameters

dt_std1= DecisionTreeRegressor(criterion='mse',max_depth=7,random_state=3)

dt_std1.fit(xtrainPCA,y_train)



#Output Prediction

y_predict_DT4=dt_std1.predict(xtestPCA)

y_predict_DT4



#Accuracy Check

RMSE_DecisionTreeAlgo5= np.sqrt(metrics.mean_squared_error(y_test,y_predict_DT4))

RMSE_DecisionTreeAlgo5
#Random Forest Algorthm

rnd_clf_std= RandomForestRegressor(n_estimators=8, random_state=3, max_depth=7, criterion = 'mse')

rnd_clf_std.fit(xtrainPCA,y_train)



#Output Prediction

y_predict_RF2=rnd_clf_std.predict(xtestPCA)

y_predict_RF2



#Accuracy Check

RMSE_RndmForAlgo2= np.sqrt(metrics.mean_squared_error(y_test,y_predict_RF2))

RMSE_RndmForAlgo2
#Checking RMSE errors with different algorithms using Standard Scalar

Algorithms_Standard_scalar = ["Linear Regression","DecisionTree Regressor(with no parameters)","DecisionTree Regresssor(with parameters)","Random Forest Regressor"]

RMSE2 = [RMSE_LinearAlgo2,RMSE_DecisionTreeAlgo4,RMSE_DecisionTreeAlgo5,RMSE_RndmForAlgo2]

print(Algorithms_Standard_scalar)

print(RMSE2)



#Creating dataframe of Algorithms vs their RMSE values(Min-Max Scalar)

Algo_vs_RMSE2= pd.DataFrame({'Algorithm':Algorithms,'RMSE Values(Dimension reduction with PCA)':RMSE2})

Algo_vs_RMSE2
#Lets plot all RMSE values in one table

Algorithms_RMSEs= pd.DataFrame({'Algorithm':Algorithms, 'RMSE values(without Scaling)':RMSE,'RMSE values(Min-Max Scaling)':RMSE1,'RMSE values(Using PCA)':RMSE2})

Algorithms_RMSEs
#plt.style.available
#Plotting

from matplotlib import style

style.use('seaborn-darkgrid')

sns.set_context("talk")

plt.figure(figsize=(20,10))

plt.plot(Algorithms,RMSE,'yo--',label='Without Scaling',linewidth=3, markersize=12)

plt.plot(Algorithms,RMSE1,'co-',label='Min-Max Scalin',linewidth=4, markersize=12)

plt.plot(Algorithms,RMSE2,'ro-',label='Using PCA',linewidth=4, markersize=12)

plt.legend()

plt.title("Algorithms vs RMSE Values",fontsize=25)