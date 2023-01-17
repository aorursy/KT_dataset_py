import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn import feature_selection

from sklearn import tree
#read data

data = pd.read_csv("../input/brasilian-houses-to-rent/houses_to_rent.csv")
#primary selection of parameters

#parameters 'hoa', 'rent amount', 'property tax' and 'fire insurance' at sum give parameter 'total', so they are not nesessary



data = data[

    [ "area", "rooms", "bathroom", "parking spaces", "floor", "animal", "furniture", "total"]

]



print(data)
#converting data into needed form



i = 0

for i in data.index:

    if( data["floor"][i]=="-" ):

        data["floor"][i]=0

    else:

        data["floor"][i]=int(data["floor"][i])

    

    if( data["animal"][i]=="acept" ):

        data["animal"][i]=1

    else:

        data["animal"][i]=0

    

    if( data["furniture"][i]=="furnished" ):

        data["furniture"][i]=1

    else:

        data["furniture"][i]=0

    

    data["total"][i] = int(data["total"][i].split("$")[1].replace(",", ""))



print(data)
#the most interesting parameter is parameter 'total'

#firstry we shall calculate mean and standart deviation for this parameter



from math import sqrt



print( "Mean: "+str(data["total"].mean()) )

print( "Standart deviation: "+str(sqrt(data["total"].std())) )
#Parameter 'area' is interesting too



print( "Mean: "+str(data["area"].mean()) )

print( "Standart deviation: "+str(sqrt(data["area"].std())) )
#here we shall start to build linear regression model



reg_model = LinearRegression()

X = []

X_data = data[

    ["area", "rooms", "bathroom", "parking spaces", "floor", "animal", "furniture"]

]

y = data["total"]



for i in data.index:

    X.append( X_data.loc[i].to_list() )



reg_model.fit(X, y)

print(reg_model.coef_)
print( "R^2="+str(reg_model.score(X, y)) )
#lets see a value of correlation parameter between parameters 'rooms' and 'bathroom'



print( data["rooms"].corr(data["bathroom"]) )
#we see good corrrelation

#so lets to try to build refression model without parameter 'bathroom' and see, is new model better or no



test_reg_model1 = LinearRegression()

X_for_test1 = []

X_data_for_test1 = data[

    ["area", "rooms", "parking spaces", "floor", "animal", "furniture"]

]



for i in data.index:

    X_for_test1.append( X_data_for_test1.loc[i].to_list() )



test_reg_model1.fit(X_for_test1, y)

print( "Coeffitients: "+str(reg_model.coef_) )

print( "R^2="+str(test_reg_model1.score(X_for_test1, y)) )
#so without parameter 'bathroom' linear model is worse

#now lets to add new parameters to build quadratic regression model

#lets next things: x1 is 'area' variable, x2 is 'rooms' variable, x3 is 'bathroom' variable, x4 is 'parking spaces' variable,

# x5 is 'floor' varible, x6 is 'animal' variable, x7 is 'furniture' variable, x8 is x1^2, x9 is x2^2, x10 is x3^2, x11 is x4^2

# x11 is x5^2



quadr_reg_model = LinearRegression()

X_quadr = []

X_quadr_data = X_data

X_add = []



for i in data.index:

    X_add = [ pow(X_quadr_data.loc[i]["area"], 2) ]

    X_add+=[ pow(X_quadr_data.loc[i]["rooms"], 2), pow(X_quadr_data.loc[i]["bathroom"], 2) ]

    X_add+=[ pow(X_quadr_data.loc[i]["parking spaces"], 2), pow(X_quadr_data.loc[i]["floor"], 2) ]

    X_quadr.append( X_quadr_data.loc[i].to_list()+X_add )



quadr_reg_model.fit(X_quadr, y)

print( "Coeffitients: "+str(quadr_reg_model.coef_) )

print( "R^2="+str(quadr_reg_model.score(X_quadr, y)) )
#now lets try to build cubic regression model

#lets next things: x1 is 'area' variable, x2 is 'rooms' variable, x3 is 'bathroom' variable, x4 is 'parking spaces' variable,

# x5 is 'floor' varible, x6 is 'animal' variable, x7 is 'furniture' variable, x8 is x1^2, x9 is x2^2, x10 is x3^2, x11 is x4^2

# x11 is x5^2, x12 is x2^3, x13 is x3^3, x14 is x4^3, x15 is x5^3



cubic_reg_model = LinearRegression()

X_cubic = []

X_cubic_data = X_data

X_add = []



for i in data.index:

    X_add = [ pow(X_cubic_data.loc[i]["area"], 2) ]

    X_add+=[ pow(X_cubic_data.loc[i]["rooms"], 2), pow(X_cubic_data.loc[i]["bathroom"], 2) ]

    X_add+=[ pow(X_cubic_data.loc[i]["parking spaces"], 2), pow(X_cubic_data.loc[i]["floor"], 2) ]

    X_add+=[ pow(X_cubic_data.loc[i]["rooms"], 3), pow(X_cubic_data.loc[i]["bathroom"], 3) ]

    X_add+=[ pow(X_cubic_data.loc[i]["parking spaces"], 3), pow(X_cubic_data.loc[i]["floor"], 3) ]

    X_cubic.append( X_cubic_data.loc[i].to_list()+X_add )



cubic_reg_model.fit( X_cubic, y )

print( "Coeffitients: "+str(cubic_reg_model.coef_) )

print( "R^2="+str(cubic_reg_model.score(X_cubic, y)) )
#now try to build non-polynomial regression model

#and see will oue models better or no



X = []

X_data = data[

    ["area", "rooms", "bathroom", "parking spaces", "floor", "animal", "furniture"]

]



for i in data.index:

    X.append( np.cbrt(X_data.loc[i].to_list()) )



reg_model.fit(X, y)

print(reg_model.coef_)

print( "R^2="+str(reg_model.score(X, y)) )