#load relevant libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#print paths

import os



training_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

testing_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#separate the data into (training) X and y, and X for testing

y=training_data["SalePrice"]

X=training_data.iloc[:,:80]



X_test=testing_data.iloc[:,:80]
def porchsum(pt):

    #This function merges all of the porch data into a single point

    summation = pt.iloc[:,1]+pt.iloc[:,2]+pt.iloc[:,3]+pt.iloc[:,4]+pt.iloc[:,0]

    return summation



def quality(series):

    #this function takes a series of the quality format and turns it into fun numerical data

    qual_dict = {"Ex":6,"Gd":5,"TA":4,"Fa":3,"Po":2,"Na":1,np.nan:4}

    return series.map(qual_dict)
def cleanOld(data_table,ignore = []):

    #turns the x value into usable data

    #ignore allows us to ignore some of the values for a more robust model

    base_numerical = data_table[["LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","1stFlrSF",'2ndFlrSF',"LowQualFinSF","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces",'GarageYrBlt','GarageCars','GarageArea',"PoolArea","MiscVal","MoSold","YrSold"]]

    porch = data_table[["WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"]]

    porchreal = porchsum(porch)

    #ignored is Class

    Class = pd.get_dummies(data_table["MSSubClass"])

    Zone = pd.get_dummies(data_table["MSZoning"])

    LotConfig = pd.get_dummies(data_table["LotConfig"])

    Slope= pd.get_dummies(data_table["LandSlope"])

    Hood = pd.get_dummies(data_table["Neighborhood"])

    Type = pd.get_dummies(data_table["BldgType"])

    Electrical = pd.get_dummies(data_table["Electrical"])

        

    Reno = data_table["YearBuilt"]-data_table["YearRemodAdd"]



    ExterQual = quality(data_table["ExterQual"])

    ExterCond = quality(data_table["ExterCond"])

    BsmtQual = quality(data_table["BsmtQual"])

    BsmtCond = quality(data_table["BsmtCond"])

    HeatingQC = quality(data_table["HeatingQC"])



    AirConBool = data_table["CentralAir"].map({"Y":1,"N":0,np.nan:0.5})

    Kitchen = quality(data_table["KitchenQual"])

    #fireplace is not included

    Fireplace = data_table["FireplaceQu"].map({"Ex":6,"Gd":5,"TA":4,"Fa":3,"Po": 2,np.nan:2.5})

    #not included

    Garage = quality(data_table["GarageQual"])

    #again,not included

    RdType = data_table["Street"].map({"Grvl":1,"Pave":0,np.nan:0.5})

    #ignored

    Shape= data_table["LotShape"].map({"Reg":1,"IR1":2,"IR2":3,"IR3":4,np.nan:1})

    dataInitial = [Reno,Hood,Slope,LotConfig,Zone,RdType,AirConBool,Kitchen, base_numerical,porchreal,ExterQual,ExterCond,BsmtQual,BsmtCond,HeatingQC]

    dataFinal = [data_table["Id"]]

    for i in range(len(dataInitial)):

        if not i in ignore:

            dataFinal.append(dataInitial[i])

    return pd.concat(dataFinal,axis = 1)
def clean(data_table,ignore = []):

    #turns the x value into usable data

    #ignore allows us to ignore some of the values for a more robust model

    Labels = ["LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","1stFlrSF",'2ndFlrSF',"LowQualFinSF","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces",'GarageYrBlt','GarageCars','GarageArea',"PoolArea","MiscVal","MoSold","YrSold"]

    

    base_num = [data_table[i] for i in Labels]

    

    

    porch = data_table[["WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"]]

    porchreal = porchsum(porch)

    

    #Class = pd.get_dummies(pd.Categorical(data_table["MSSubClass"],categories = pd.unique(X["MSSubClass"])))

    #Class = Class.set_index(data_table.index)

    Zone = pd.get_dummies(pd.Categorical(data_table["MSZoning"],categories = pd.unique(X["MSZoning"])))

    Zone = Zone.set_index(data_table.index)

    LotConfig = pd.get_dummies(pd.Categorical(data_table["LotConfig"],categories = pd.unique(X["LotConfig"])))

    LotConfig = LotConfig.set_index(data_table.index)

    Slope = pd.get_dummies(pd.Categorical(data_table["LandSlope"],categories = pd.unique(X["LandSlope"])))

    Slope = Slope.set_index(data_table.index)

    Hood = pd.get_dummies(pd.Categorical(data_table["Neighborhood"],categories = pd.unique(X["Neighborhood"])))

    Hood = Hood.set_index(data_table.index)

    Type = pd.get_dummies(pd.Categorical(data_table["BldgType"],categories = pd.unique(X["BldgType"])))

    Type = Type.set_index(data_table.index)

    



    #Electrical = pd.get_dummies(pd.Categorical(data_table["Electrical"].fillna(value = "Electrical NaN"),categories = pd.unique(X["Electrical"])))

    #Electrical = Electrical.set_index(data_table.index)



    #BsmtCond = pd.get_dummies(pd.Categorical(data_table["BsmtCond"].fillna(value = "BsmtCond NaN"),categories = pd.unique(X["BsmtCond"])))

    #BsmtCond = BsmtCond.set_index(data_table.index)

    #Electrical has Nan values that are difficult to deal with

    

    Categorical = [Zone,LotConfig,Slope,Hood,Type]

        

    Reno = data_table["YearBuilt"]-data_table["YearRemodAdd"]

    QSquare = data_table["OverallQual"]*data_table["OverallQual"]

    QSquare.name = "QSquare"

    

    ExterQual = quality(data_table["ExterQual"])

    ExterCond = quality(data_table["ExterCond"])

    BsmtQual = quality(data_table["BsmtQual"])

    HeatingQC = quality(data_table["HeatingQC"])

    Kitchen = quality(data_table["KitchenQual"])

    Garage = quality(data_table["GarageQual"])

    

    qualities = [ ExterQual,ExterCond,BsmtQual,HeatingQC,Kitchen,Garage]

    

    AirConBool = data_table["CentralAir"].map({"Y":1,"N":0,np.nan:1})

    Functional = data_table["Functional"].map({"Typ":0,"Min1":1,"Min2":1,"Mod":1,"Maj1": 1,"Maj2":1,"Sev": 1,"Sal":1,np.nan:0})

    Fireplace = data_table["FireplaceQu"].map({"Ex":6,"Gd":5,"TA":4,"Fa":3,"Po": 2,np.nan:2.5})

    RdType = data_table["Street"].map({"Grvl":1,"Pave":0,np.nan:0.5})

    Shape= data_table["LotShape"].map({"Reg":1,"IR1":2,"IR2":3,"IR3":4,np.nan:1})

    BsmtCond =data_table["BsmtCond"].map({"Ex":6,"Gd":5,"TA":4,"Fa":3,"Po": 2,np.nan:1})

    

    maps = [AirConBool,Functional,Fireplace,RdType,Shape,BsmtCond]

    

    #print(Junk.columns)

    dataInitial = Categorical+base_num + qualities + maps +[Reno]+[QSquare]

    dataFinal = [data_table["Id"]]

    for i in range(len(dataInitial)):

        if not i in ignore:

            dataFinal.append(dataInitial[i])

    result =  pd.concat(dataFinal,axis = 1)

    return result
from sklearn.impute import SimpleImputer

#Format training data



# Imputation

my_imputer = SimpleImputer(strategy = "median")

X_train = clean(X)

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))



#put columns back

imputed_X_train.columns = X_train.columns



imputed_X_train["Id"]=imputed_X_train["Id"].apply(lambda x: int(x) if x == x else "")

imputed_X_train=imputed_X_train.set_index("Id")



imputed_X_train.head()
#Format testing data

test_imputer = SimpleImputer(strategy = "median")

X_test = clean(X_test)

imputed_X_test = pd.DataFrame(test_imputer.fit_transform(X_test))



#put columns back

imputed_X_test.columns = X_test.columns



imputed_X_test["Id"]=imputed_X_test["Id"].apply(lambda x: int(x) if x == x else "")

imputed_X_test=imputed_X_test.set_index("Id")



imputed_X_test.head()
from sklearn.linear_model import LinearRegression as LineBae





model1 = LineBae().fit(imputed_X_train,y)

Linear_Test_Predictions = pd.Series(model1.predict(imputed_X_test))

Linear_Test_Predictions = pd.concat([pd.Series(imputed_X_test.index),Linear_Test_Predictions],axis = 1)

Linear_Test_Predictions = Linear_Test_Predictions.set_index("Id")



print(Linear_Test_Predictions)
Linear_Test_Estimates = pd.Series(model1.predict(imputed_X_train))

Linear_Test_Estimates = pd.concat([pd.Series(X_train["Id"]),Linear_Test_Estimates],axis=1)

Linear_Test_Estimates.columns = ["Id","Predictions"]

Linear_Test_Estimates= Linear_Test_Estimates.set_index("Id")

Linear_Test_Estimates= Linear_Test_Estimates["Predictions"]



print(Linear_Test_Estimates)
#make Id the axis of y

y = pd.concat([pd.Series(X_train["Id"]),y],axis=1)

Linear_Test_Estimates.columns = ["Id","SalePrice"]

y= y.set_index("Id")

y= y["SalePrice"]

print(y)
from xgboost import XGBRegressor



error_data = y - Linear_Test_Estimates



print(y)

print(Linear_Test_Estimates)



print(error_data)



# Define the model

boost_model = XGBRegressor(learning_rate = 0.08,n_estimators = 100,random_state = 66,subsample = 0.9)



# Fit the model

boost_model.fit(imputed_X_train,error_data)



# Get predictions

boost_preds = boost_model.predict(imputed_X_test)

boost_preds = pd.concat([pd.Series(X_test["Id"]),pd.Series(boost_preds)],axis = 1)

boost_preds.columns = ["Id","Predictions"]

boost_preds = boost_preds.set_index("Id")

print(boost_preds.iloc[:,0])
res = boost_preds.iloc[:,0]+Linear_Test_Predictions.iloc[:,0]

ind = pd.Series(X.index)

res = pd.DataFrame(res).reset_index()

res.columns =["Id","SalePrice"]



print(res)

res.to_csv('submission.csv', index=False)