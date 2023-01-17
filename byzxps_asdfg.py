import pandas as pd
v_dict1 = { "COUNTRY" : ["TURKEY","U.K.","GERMANY","FRANCE","U.S.A","AZERBAIJAN","IRAN"],

            "CAPITAL":["ISTANBUL","LONDON","BERLIN","PARIS","NEW YORK","BAKU","TAHRAN"],

            "POPULATION":[15.07,8.13,3.57,2.12,8.62,4.3,8.69]}



v_dataFrame1 = pd.DataFrame(v_dict1)



print(v_dataFrame1)

print()

print("Type of v_dataFrame1 is : " , type(v_dataFrame1))
v_head1 = v_dataFrame1.head()

print(v_head1)

print()

print("Type of v_head1 is :" ,type(v_head1))
print(v_dataFrame1.head(100))
v_tail1 = v_dataFrame1.tail()

print(v_tail1)

print()

print("Type of v_tail1 is :" ,type(v_tail1))
v_columns1 = v_dataFrame1.columns

print(v_columns1)

print()

print("Type of v_columns is : " , type(v_columns1))
v_info1 = v_dataFrame1.info()

print(v_info1)

print()

print("Type of v_info1 is : " , type(v_info1))
v_dtypes1 = v_dataFrame1.dtypes

print(v_dtypes1)

print()

print("Type of v_dtypes1 is : " , type(v_dtypes1))
v_descr1 = v_dataFrame1.describe()

print(v_descr1)

print()

print("Type of v_descr1 is : " , type(v_descr1))
v_country1 = v_dataFrame1["COUNTRY"]

print(v_country1)

print()

print("Type of v_country1 is : " , type(v_country1))
v_currenyList1 = ["TRY","GBP","EUR","EUR","USD","AZN","IRR"]

v_dataFrame1["CURRENCY"] = v_currenyList1



print(v_dataFrame1.head())
v_AllCapital = v_dataFrame1.loc[:,"CAPITAL"]

print(v_AllCapital)

print()

print("Type of v_AllCapital is : " , type(v_AllCapital))
v_top3Currency = v_dataFrame1.loc[0:3,"CURRENCY"]

print(v_top3Currency)

v_CityCountry = v_dataFrame1.loc[:,["CAPITAL","COUNTRY","BLABLA"]] #--> BLABLA not defined !!!

print(v_CityCountry)
v_Reverse1 = v_dataFrame1.loc[::-1,:]

print(v_Reverse1)
print(v_dataFrame1.loc[:,:"POPULATION"])

print()

print(v_dataFrame1.loc[:,"POPULATION":])
print(v_dataFrame1.iloc[:,2])
v_filter1 = v_dataFrame1.POPULATION > 4

print(v_filter1)
v_filter2 = v_dataFrame1["POPULATION"] < 9

print(v_filter2)
print(v_dataFrame1[v_filter1 & v_filter2])
print(v_dataFrame1[v_dataFrame1["CURRENCY"] == "EUR"])
v_meanPop =v_dataFrame1["POPULATION"].mean()

print(v_meanPop)



v_meanPopNP = np.mean(v_dataFrame1["POPULATION"])

print(v_meanPopNP)
for a in v_dataFrame1["POPULATION"]:

    print(a)
v_dataFrame1["POP LEVEL"] = ["Low" if v_meanPop > a else "HIGH" for a in v_dataFrame1["POPULATION"]]

print(v_dataFrame1)
print(v_dataFrame1.columns)



v_dataFrame1.columns = [a.lower() for a in v_dataFrame1.columns]



print(v_dataFrame1.columns)
v_dataFrame1.columns = [a.split()[0]+"_"+a.split()[1] if (len(a.split())>1) else a for a in v_dataFrame1.columns]

print(v_dataFrame1.columns)
v_dataFrame1["test1"] = [-1,-2,-3,-4,-5,-6,-7]

print(v_dataFrame1)
v_dataFrame1.drop(["test1"],axis=1,inplace = True) #--> inplace = True must be written

print(v_dataFrame1)
v_data1 = v_dataFrame1.head()

v_data2 = v_dataFrame1.tail()



print(v_data1)

print()

print(v_data2)
v_dataConcat1 = pd.concat([v_data1,v_data2],axis=0) # axis = 0 --> VERTICAL CONCAT

v_dataConcat2 = pd.concat([v_data2,v_data1],axis=0) # axis = 0 --> VERTICAL CONCAT



print(v_dataConcat1)

print()

print(v_dataConcat2)
v_CAPITAL = v_dataFrame1["capital"]

v_POPULATION = v_dataFrame1["population"]



v_dataConcat3 = pd.concat([v_CAPITAL,v_POPULATION],axis=1) #axis = 1 --> HORIZONTAL CONCAT

v_dataConcat4 = pd.concat([v_POPULATION,v_CAPITAL],axis=1) #axis = 1 --> HORIZONTAL CONCAT

print(v_dataConcat3)

print()

print(v_dataConcat4)
v_dataFrame1["test1"] = [a*2 for a in v_dataFrame1["population"]]

print(v_dataFrame1)
def f_multiply(v_population):

    return v_population*3



v_dataFrame1["test2"] = v_dataFrame1["population"].apply(f_multiply)

print(v_dataFrame1)