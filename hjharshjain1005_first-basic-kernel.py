import numpy as np
import pandas as pd
data=pd.read_csv("../input/first-model-chennai-house/chennai_house_price_prediction.csv")
data.head()  
data.describe()    # gives a description of continous variables
data.describe(include="all")   # by including all we get details about categorical values also
data.drop_duplicates()

data.drop_duplicates().shape
data.shape
data.isnull().sum()      # it gives the count of missing values present in column
data.dropna(axis=0)
data["N_BEDROOM"].fillna( value=(data["N_BEDROOM"].mode()[0]) , inplace=True)   
#inplace=True made the changes in orignal data

data["N_BEDROOM"].value_counts()/len(data)
data["N_BATHROOM"].value_counts()/len(data)
for i in range(0,len(data)):
    if pd.isnull(data["N_BATHROOM"][i])==True:
        if data["N_BEDROOM"][i]==1.0 or 2.0:
            data["N_BATHROOM"][i]=1.0
        else:
            data["N_BATHROOM"][i]=2.0
            
    
for i in range(0,len(data)):
    if pd.isnull(data["QS_OVERALL"][i])==True:
        x = (data["QS_ROOMS"][i]+data["QS_BEDROOM"][i]+data["QS_BATHROOM"][i])/3
        data["QS_OVERALL"][i]=x
data.isnull().sum()   # all missing values filled, now we have complete data
data.dtypes
# N_BEDROOM , N_BATHROOM must be in integers , as they can not have float values
data=data.astype({"N_BEDROOM": int, "N_BATHROOM": int })
data.dtypes
temp = ["AREA","N_BEDROOM","N_BATHROOM","N_ROOM","SALE_COND","PARK_FACIL","BUILDTYPE","UTILITY_AVAIL","STREET","MZZONE"]
for i in temp:
    print("********** Value count for category",i,"********** ")
    print(data[i].value_counts())
    print(" ")
data["PARK_FACIL"].replace({"Noo":"No"}, inplace=True )
data["AREA"].replace({"Chrmpet":"Chrompet","Chormpet":"Chrompet","Chrompt":"Chrompet","Karapakam" : "Karapakkam",
                      "KKNagar":"KK Nagar", "TNagar": "T Nagar", "Adyr": "Adyar","Ana Nagar":"Anna Nagar",
                      "Ann Nagar":"Anna Nagar", "Velchery":"Velachery",}, inplace=True )
data["AREA"].value_counts()
data["UTILITY_AVAIL"].replace({"All Pub":"AllPub"}, inplace=True )
data["STREET"].replace({"Pavd":"Paved", "No Access": "No Access"}, inplace=True )
data["SALE_COND"].replace({"Partiall":"Partial","PartiaLl":"Partial","Ab Normal":"AbNormal","Adj Land":"AdjLand"}, inplace=True )
data["BUILDTYPE"].replace({"Comercial":"Commercial", "Other": "Others"}, inplace=True )
temp = ["SALE_COND","PARK_FACIL","UTILITY_AVAIL","STREET","BUILDTYPE"]
for i in temp:
    print("********** Value count for category",i,"********** ")
    print(data[i].value_counts())
    print(" ")
