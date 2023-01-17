#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import plotly.express as pe
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.graphics.regressionplots import influence_plot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

%matplotlib inline
#Reading the CSV file

data = pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')

#Creating a DataFrame for the CarPrice data
carprice_df = pd.DataFrame(data)
carprice_df.head()
#Undetanding the shape and data types of the data
carprice_df.info()
# Number of Unique Values in each categorical column
carprice_df.select_dtypes(exclude = np.number).nunique()
#Splitting the carName column into Company and Model name
car_name_df = carprice_df.CarName.str.split(n = 1, expand = True)
carprice_df.insert(loc = 2,column = "car_name", value = car_name_df[1])
carprice_df.insert(loc = 2,column = "car_company", value = car_name_df[0])
carprice_df.drop(columns = "CarName", inplace = True)
carprice_df.head()
# Number of Unique Values in each categorical column
carprice_df.select_dtypes(exclude = np.number).nunique()
#Dropping the car_name or model name column due to high cardinality

carprice_df.drop(columns = ["car_name"], inplace = True)
print("car_name column dropped from the dataframe")
#Unique values of each categorical column
columns = list(carprice_df.select_dtypes(exclude =  np.number).columns)
for i in columns:
    print(f"{i.upper()}: {' , '.join(carprice_df[i].unique())} \n")
#Removing error from names of car companies
print(f"Unique values before cleaning: {carprice_df.car_company.nunique()}")
carprice_df.car_company.replace(to_replace = ["alfa-romero","Nissan","porcshce", "toyouta","maxda", "vokswagen", "vw"], 
                                value = ["alfa-romeo","nissan", "porsche","toyota","mazda", "volkswagen","volkswagen"], 
                                inplace = True)
print(f"Unique values after cleaning : {carprice_df.car_company.nunique()}")
#Checking the numeric columns or features
carprice_df.describe(percentiles= [0.25, 0.50, 0.75, 0.95])
# Number of unique values in each numeric column

carprice_df.select_dtypes(include = np.number).nunique()
#Setting the index of the dataframe as car_id as it has all unique values.

carprice_df = carprice_df.set_index("car_ID")
carprice_df.head()
#Understanding the Symboling feature
carprice_df.symboling.unique()
#Using Z-Score to detect outliers
from scipy.stats import zscore
index = carprice_df[abs(zscore(carprice_df["price"]))>=3].index
index
#Removing Outliers
carprice_df.drop(index, inplace = True)
print("Outliers Removed Successfully")
#Plot 1
fig = make_subplots(rows =2, cols = 2, subplot_titles = ["Distribution of Price W.R.T Car compnay",
                                                        "Distribution of Car Company",
                                                        "Distribution of Price W.R.T Fuel System",
                                                        "Distribution of Fuel System"])
fig.add_trace(go.Box(x = carprice_df["car_company"], y = carprice_df["price"], name = "Car Company"),
              row = 1, col =1)
fig.add_trace(go.Histogram(x = carprice_df["car_company"], histnorm = "percent", name = "Car Company"),
              row = 1, col =2)
fig.add_trace(go.Box(x = carprice_df["fuelsystem"], y = carprice_df["price"], name = "Fuel System"),
              row = 2, col =1)
fig.add_trace(go.Histogram(x = carprice_df["fuelsystem"], histnorm = "percent", name = "Fuel System"),
              row = 2, col =2)
fig.update_layout(height = 700,
                 xaxis1 = dict(title = "Car company"),
                 xaxis2 = dict(title = "Car compnay"),
                 xaxis3 = dict(title = "Fuel System"),
                 xaxis4 = dict(title = "Fuel System"),
                 yaxis1 = dict(title = "Price"),
                 yaxis2 = dict(title = "Percentage"),
                 yaxis3 = dict(title = "Price"),
                 yaxis4 = dict(title = "Percentage"))
fig.show()
#Plot 2
fig = make_subplots(rows =2, cols = 2, subplot_titles = ["Distribution of Price W.R.T Fuel Type",
                                                        "Distribution of Fuel Type",
                                                        "Distribution of Price W.R.T Car Body",
                                                        "Distribution of Car Body"])
fig.add_trace(go.Box(x = carprice_df["fueltype"], y = carprice_df["price"], name = "Fuel Type" ),
              row = 1, col =1)
fig.add_trace(go.Histogram(x = carprice_df["fueltype"], histnorm = "percent", name = "Fuel Type"),
              row = 1, col =2)
fig.add_trace(go.Box(x = carprice_df["carbody"], y = carprice_df["price"], name = "Car Body"),
              row = 2, col =1)
fig.add_trace(go.Histogram(x = carprice_df["carbody"], histnorm = "percent", name = "Car Body"),
              row = 2, col =2)
fig.update_layout(height = 700,
                 xaxis1 = dict(title = "Fuel Type"),
                 xaxis2 = dict(title = "Fuel Type"),
                 xaxis3 = dict(title = "Car Body"),
                 xaxis4 = dict(title = "Car Body"),
                 yaxis1 = dict(title = "Price"),
                 yaxis2 = dict(title = "Percentage"),
                 yaxis3 = dict(title = "Price"),
                 yaxis4 = dict(title = "Percentage"))
fig.show()
#Plot 3
fig = make_subplots(rows =2, cols = 2, subplot_titles = ["Distribution of Price W.R.T Drive Wheel",
                                                        "Distribution of Driv Wheel",
                                                        "Distribution of Price W.R.T Cylinder Count",
                                                        "Distribution of Cylinder Count"])
fig.add_trace(go.Box(x = carprice_df["drivewheel"], y = carprice_df["price"], name = "Drive Wheel" ),
              row = 1, col =1)
fig.add_trace(go.Histogram(x = carprice_df["drivewheel"], histnorm = "percent", name = "Drive Wheel"),
              row = 1, col =2)
fig.add_trace(go.Box(x = carprice_df["cylindernumber"], y = carprice_df["price"], name = "Cylinder Count"),
              row = 2, col =1)
fig.add_trace(go.Histogram(x = carprice_df["cylindernumber"], histnorm = "percent", name = "Cylinder Count"),
              row = 2, col =2)
fig.update_layout(height = 700,
                 xaxis1 = dict(title = "Drive Wheel"),
                 xaxis2 = dict(title = "Drive Wheel"),
                 xaxis3 = dict(title = "Cylinder Count"),
                 xaxis4 = dict(title = "Cylinder Count"),
                 yaxis1 = dict(title = "Price"),
                 yaxis2 = dict(title = "Percentage"),
                 yaxis3 = dict(title = "Price"),
                 yaxis4 = dict(title = "Percentage"))
fig.show()
#Boxplot to study variation in price of cars with respect to other features
fig = make_subplots(rows =2, cols = 2, subplot_titles = ["Distribution of Price W.R.T Engine Location",
                                                        "Distribution of Engine Location",
                                                        "Distribution of Price W.R.T Engine Type",
                                                        "Distribution of Engine Type"])
fig.add_trace(go.Box(x = carprice_df["enginelocation"], y = carprice_df["price"], name = "Engine Location" ),
              row = 1, col =1)
fig.add_trace(go.Histogram(x = carprice_df["enginelocation"], histnorm = "percent", name = "Engine Location"),
              row = 1, col =2)
fig.add_trace(go.Box(x = carprice_df["enginetype"], y = carprice_df["price"], name = "Engine Type"),
              row = 2, col =1)
fig.add_trace(go.Histogram(x = carprice_df["enginetype"], histnorm = "percent", name = "Engine Type"),
              row = 2, col =2)
fig.update_layout(height = 700,
                 xaxis1 = dict(title = "Engine Location"),
                 xaxis2 = dict(title = "Engine Location"),
                 xaxis3 = dict(title = "Engine Type"),
                 xaxis4 = dict(title = "Engine Type"),
                 yaxis1 = dict(title = "Price"),
                 yaxis2 = dict(title = "Percentage"),
                 yaxis3 = dict(title = "Price"),
                 yaxis4 = dict(title = "Percentage"))
fig.show()
#Correlation between various numeric features depicted using a heatmap
plt.figure(figsize = (10,8))
fig = sns.heatmap(carprice_df.corr(), annot = True)
plt.show()
#Label encoding the column number of cylinders

carprice_df["cylindernumber"] = carprice_df.cylindernumber.map({"two": 0, "three": 1, "four": 2, "five" : 3, "six" : 4, "eight": 5, "twelve": 6})
print(f"Cylinder Number column encoded as {carprice_df['cylindernumber'].unique()}")
#function to encode variables with high cardinality

def encoder(column_name):    
    #Storing the median price values for each category in a datframe 
    median_price = carprice_df.groupby(column_name)["price"].median().to_frame().reset_index()
    median_price.sort_values(by = "price", inplace = True) #Sorting the frame
    median_price["price"]  = median_price["price"].map({i: index for index, i in enumerate(median_price["price"])}) #Mapping with non-negative integers
    median_price = median_price.set_index(column_name) #Again set the index to category names
    median_price = median_price.to_dict() #Convert the frame back to dictionary
    median_price  = median_price["price"] #Storing the dictionary with price values as the mapped dictionary is the value of price key
    return median_price #Returning the map or dict
#Encoding Engine type variable
enginetype_dict = encoder("enginetype") #Storing enginetype_map in enginetype_dict for reference
carprice_df["enginetype"] = carprice_df.enginetype.map(enginetype_dict) #mapping the labels
#Encoding Fuel System Variable
fuelsystem_dict = encoder("fuelsystem") #Storing dict for fuelsystem in fuelsystem_dict for reference
carprice_df["fuelsystem"] = carprice_df.fuelsystem.map(fuelsystem_dict) #mapping the labels
#Encoding Carbody varibale
carbody_dict = encoder("carbody") #Storing dict for carbody in carbody_dict for reference
carprice_df["carbody"] = carprice_df.carbody.map(carbody_dict) #Mapping the labels
print(f"{enginetype_dict}\n")
print(f"{fuelsystem_dict}\n")
print(carbody_dict)
#Encoding Car_company variable using similar technique as used above.

#Storing the median prices of cars for respective companies in a dataframe
median_price = carprice_df.groupby("car_company")["price"].median().to_frame().reset_index()
#Using the cut function to create three price segements low, mid and high range
median_price["price"] = pd.cut(median_price["price"],bins = [0,10000,20000,50000], labels = ["low range", "mid range", "high range"])
print(f"Use this encoding for reference: {median_price}")
median_price = median_price.set_index("car_company") #Resetting the index to car_company
median_price  = median_price.to_dict() #Storing the data frame a dictionary
median_price = median_price["price"] #Storing the dictionary with price values as the mapped dictionary is the value of price key
carprice_df["car_company"] = carprice_df.car_company.map(median_price) #Mapping th values to car_company column

#Finally creating dummy variable for all the categorical variables and dropping the first column of all to avoid multicolinearity
print(f"Number of columns before Dummy encoding - {carprice_df.shape[1]}")
carprice_df = pd.get_dummies(carprice_df, drop_first = True)
print(f"Number of columns after Dummy encoding - {carprice_df.shape[1]}")
carprice_df.head()
#Splitting the dataset in train and test dataset in a ratio of 70-30
train_df, test_df = train_test_split(carprice_df, train_size = 0.7, random_state = 42)
#Scaling the training dataset
scaler = MinMaxScaler() #Creating a MinMaxScaler object
train_df.loc[:,:] = scaler.fit_transform(train_df.loc[:,:]) #Fitting and transforming the dataset
#Creating x_train and y_train
y_train = train_df.pop("price") #Storing the response variable price in y_train
x_train = train_df
#Creating a LinearRegression model using sklearn as we can find the features using RFE only after fitting a model
lm = LinearRegression() #Creating Linear Regression Object
lm_model = lm.fit(x_train, y_train) #Fitting the model
#Using Recursive Feature Elimination to pick top 10-15 features and ranking them
rfe = RFE(lm_model, 15).fit(x_train,y_train) #Creating a RFE object and then fitting it on training data
list(zip(train_df.columns, rfe.support_, rfe.ranking_)) #Name of feature along with the ranking give to it by the RFE object
#Storing the selected features in a variable
cols = list(train_df.columns[rfe.support_])
x_train = sm.add_constant(train_df[cols]) #Adding a constant for the intercept term in the x_train
ols_model = sm.OLS(y_train, x_train).fit() #Creating and fitting the OLS model on the training set

ols_model.summary2() #Viewing the summary statistics
#Check for multicolinearity
vif = pd.DataFrame() #Creating a new dataframe
vif["Features"] = cols #Storing the column names as a column in vif dataframe
vif["VIF"] = [variance_inflation_factor(x_train[cols].values, i) for i in range(x_train.shape[1]-1)] #Computing VIF for each variable
vif
cols.remove("carlength")
x_train = sm.add_constant(train_df[cols])
#Model fitting
ols_model = sm.OLS(y_train, x_train).fit()
ols_model.summary2()
vif = pd.DataFrame()
vif["Features"] = cols
vif["VIF"] = [variance_inflation_factor(x_train[cols].values, i) for i in range(x_train.shape[1]-1)]
vif
cols.remove("cylindernumber")
x_train = sm.add_constant(train_df[cols])

ols_model = sm.OLS(y_train, x_train).fit()
ols_model.summary2()
vif = pd.DataFrame()
vif["Features"] = cols
vif["VIF"] = [variance_inflation_factor(x_train[cols].values, i) for i in range(x_train.shape[1]-1)]
vif
cols.remove("highwaympg")
x_train = sm.add_constant(train_df[cols])

ols_model = sm.OLS(y_train, x_train).fit()
ols_model.summary2()
vif = pd.DataFrame()
vif["Features"] = cols
vif["VIF"] = [variance_inflation_factor(x_train[cols].values, i) for i in range(x_train.shape[1]-1)]
vif
cols.remove("boreratio")
x_train = sm.add_constant(train_df[cols])

ols_model = sm.OLS(y_train, x_train).fit()
ols_model.summary2()
vif = pd.DataFrame()
vif["Features"] = cols
vif["VIF"] = [variance_inflation_factor(x_train[cols].values, i) for i in range(x_train.shape[1]-1)]
vif
cols.remove("horsepower")
x_train = sm.add_constant(train_df[cols])

ols_model = sm.OLS(y_train, x_train).fit()
ols_model.summary2()
vif = pd.DataFrame()
vif["Features"] = cols
vif["VIF"] = [variance_inflation_factor(x_train[cols].values, i) for i in range(x_train.shape[1]-1)]
vif
cols.remove("carheight")
x_train = sm.add_constant(train_df[cols])

ols_model = sm.OLS(y_train, x_train).fit()
ols_model.summary2()
vif = pd.DataFrame()
vif["Features"] = cols
vif["VIF"] = [variance_inflation_factor(x_train[cols].values, i) for i in range(x_train.shape[1]-1)]
vif
cols.remove("carwidth")
x_train = sm.add_constant(train_df[cols])

ols_model = sm.OLS(y_train, x_train).fit()
ols_model.summary2()
vif = pd.DataFrame()
vif["Features"] = cols
vif["VIF"] = [variance_inflation_factor(x_train[cols].values, i) for i in range(x_train.shape[1]-1)]
vif
cols.remove("fueltype_gas")
x_train = sm.add_constant(train_df[cols])

ols_model = sm.OLS(y_train, x_train).fit()
ols_model.summary2()
vif = pd.DataFrame()
vif["Features"] = cols
vif["VIF"] = [variance_inflation_factor(x_train[cols].values, i) for i in range(x_train.shape[1]-1)]
vif
cols.remove("carbody")
x_train = sm.add_constant(train_df[cols])

ols_model = sm.OLS(y_train, x_train).fit()
ols_model.summary2()
vif = pd.DataFrame()
vif["Features"] = cols
vif["VIF"] = [variance_inflation_factor(x_train[cols].values, i) for i in range(x_train.shape[1]-1)]
vif
cols.remove("aspiration_turbo")
x_train = sm.add_constant(train_df[cols])

ols_model = sm.OLS(y_train, x_train).fit()
ols_model.summary2()
vif = pd.DataFrame()
vif["Features"] = cols
vif["VIF"] = [variance_inflation_factor(x_train[cols].values, i) for i in range(x_train.shape[1]-1)]
vif
#Check for Normality- Whether the residual follow a normal distribution or not
sns.distplot(ols_model.resid)
plt.title("P-P Plot")
plt.show()
#Test for Homoscedasticity - Whether the variance of errors is constant or not.
plt.scatter(x = ols_model.fittedvalues, y = ols_model.resid) #Scatter Plot between Fitted Values and Residuals
plt.title("Fitted Price values VS Errors")
plt.xlabel("Normalized Fitted Values")
plt.ylabel("Normalized Residuals")
plt.show()
#Tranforming the test dataframe using the scaler object. We do not fit the frame again as we already fitted it on the train set
test_df.loc[:,:] = scaler.transform(test_df)
#Creating y_test and x_test
y_test = test_df.pop("price")
x_test = test_df[cols]
x_test = sm.add_constant(x_test)
#Storing predicted values for y in y_pred
y_pred = ols_model.predict(x_test)
#Checking model performance using r2_score method
print(f"The Coefficient of determination for test set is {round(r2_score(y_test, y_pred),3)}")
#Checking the mean squared error
print(f"The mean squared error for the test set is {round(mean_squared_error(y_test,y_pred),3)}")
