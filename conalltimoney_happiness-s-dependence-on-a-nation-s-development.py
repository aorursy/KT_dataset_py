import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

import csv

import sqlite3 



from tqdm import tqdm_notebook as bar 

from tqdm import tqdm_notebook as tqdm



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
print(os.listdir("../input/world-happiness"))
#reading the happiness data into a data frame

HappinessDF=pd.read_csv("../input/world-happiness/2015.csv")

print(HappinessDF.dtypes)

HappinessDF.head()
connection = sqlite3.connect("../input/world-development-indicators/database.sqlite")

cursor=connection.cursor()



def DataFrameTable(query):

    return pd.read_sql_query(query,connection)
#initiallty need to read country names in a Data Frame 

query="""

    SELECT ShortName  

    FROM Country

"""

DevelCountryNames=DataFrameTable(query)

DevelCountryNames
# Now list the non matching countries

NonMatches=[]

for country in HappinessDF["Country"]:

    if country not in DevelCountryNames["ShortName"].tolist():

        NonMatches.append(country)

print(NonMatches)
SearchTerms=["Congo","Ivoire","Kyrgyzstan","Kyrgyz","Syria","Lao","Slovak","Palestine","Somaliland","Cyprus","China","Korea"]

SearchQuery="SELECT ShortName FROM Country WHERE"

for SearchTerm in tqdm(SearchTerms):

    SearchQuery+=" ShortName LIKE '%"+SearchTerm+"%' OR"

SearchQuery=SearchQuery[:-3]

print(SearchQuery)

    



DataFrameTable(SearchQuery)
#first create the list of countries 

CountrysList = HappinessDF["Country"].tolist()



CountrysToRemove=["Taiwan","North Cyprus","Somaliland region","Palestinian Territories"]



CountrysToReplace={

    "Slovakia":"Slovak Republic",

    "South Korea":"Korea",

    "Hong Kong":"Hong Kong SAR, China",

    "Kyrgyzstan":"Kyrgyz Republic",

    "Laos":"Lao PDR",

    "Congo (Kinshasa)":"Dem. Rep. Congo",

    "Congo (Brazzaville)":"Congo",

    "Ivory Coast":"Côte d''Ivoire",

    "Syria":"Syrian Arab Republic"} 

    #This can used to convert when needed





for Country in CountrysToRemove:

    CountrysList.remove(Country)

        

for Country in CountrysToReplace.keys():

    CountrysList[CountrysList.index(Country)]=CountrysToReplace[Country]



print(CountrysList)


IndicatorQuery="""  

SELECT IndicatorName,max(Value),min(Value)

FROM Indicators

GROUP BY IndicatorName

""" #max and min values is to investigate values

Indicators=DataFrameTable(IndicatorQuery)

Indicators.head()

pd.set_option('max_colwidth', 120)

pd.set_option("max_rows",1400)



#used to pick the indecators of interest uncomment to print them all 

#print(Indicators)
IndicatorsList=[

"Access to electricity (% of population)"

,"Adjusted net enrolment rate, primary, both sexes (%)"

,"Adolescent fertility rate (births per 1,000 women ages 15-19)"

,"Adult literacy rate, population 15+ years, both sexes (%)"

,"Arable land (hectares per person)"

,"Average precipitation in depth (mm per year)"

,"Bribery incidence (% of firms experiencing at least one bribe payment request)"

,"Central government debt, total (% of GDP)"

,"Community health workers (per 1,000 people)"

,"Currency composition of PPG debt, U.S. dollars (%)"

,"Death rate, crude (per 1,000 people)"

,"Droughts, floods, extreme temperatures (% of population, average 1990-2009)"

,"Emigration rate of tertiary educated (% of total tertiary educated population)"

,"Expenditure on education as % of total government expenditure (%)"

,"Fixed broadband subscriptions (per 100 people)"

,"GDP per capita (constant 2005 US$)"

,"GDP per capita growth (annual %)"

,"Improved sanitation facilities (% of population with access)"

,"Income share held by highest 10%"

,"Income share held by highest 20%"

,"Internet users (per 100 people)"

,"Life expectancy at birth, total (years)"

,"Long-term unemployment (% of total unemployment)"

,"Mobile cellular subscriptions (per 100 people)"

,"Net enrolment rate, secondary, both sexes (%)"

,"Net migration"

,"Percentage of students in secondary education who are female (%)"

,"Population density (people per sq. km of land area)"

,"Population, total"

,"Poverty gap at $3.10 a day (2011 PPP) (%)"

,"Refugee population by country or territory of origin"

,"Tax revenue (% of GDP)"

,"Urban population (% of total)"

]

def ListToString(List):

    tup="("

    for x in List:

        tup+="'"+str(x)+"',"

    tup=tup[:-1]+")"

    return tup

    

# takes a long time to run, there is no pivot function it in SQL lite so instead I will pivot with pandas

query = """

SELECT data.CountryName,data.IndicatorName,data.Year,I.Value

FROM

(SELECT Country.ShortName AS CountryName,Indicators.CountryCode,IndicatorName,IndicatorCode,MAX(Year) AS Year

FROM Indicators,Country

WHERE Indicators.CountryCode = Country.CountryCode

AND Country.ShortName IN """+ListToString(CountrysList)+"""

AND Indicators.IndicatorName IN """+ListToString(IndicatorsList)+"""

GROUP BY Country.ShortName,Indicators.CountryCode,IndicatorName,IndicatorCode) AS data

LEFT JOIN Indicators I ON data.CountryCode = I.CountryCode AND data.IndicatorCode = I.IndicatorCode and I.Year = data.Year

;

"""



IndicatorData=DataFrameTable(query)

IndicatorData.head()
IndicatorData=IndicatorData.pivot("CountryName","IndicatorName","Value")

IndicatorData.columns.name = None 

IndicatorData.head()
print("There are",len(IndicatorData)-len(IndicatorData.dropna()),"countries with missing data out of",len(IndicatorData))

IndicatorData.describe()
query = """

SELECT AVG(data.Year) AS AverageMostRecentNonNullYearForEachFeature

FROM

(SELECT MAX(Year) AS Year

FROM Indicators,Country

WHERE Indicators.CountryCode = Country.CountryCode

AND Country.ShortName IN """+ListToString(CountrysList)+"""

AND Indicators.IndicatorName IN """+ListToString(IndicatorsList)+"""

AND Indicators.Value IS NOT NULL

GROUP BY Country.ShortName,Indicators.CountryCode,IndicatorName,IndicatorCode) AS data

;

"""



AverageMostRecentNonNullYear=DataFrameTable(query)

display(AverageMostRecentNonNullYear.head())



query = """

SELECT Count(data.Year) AS NumberOfNonNullValues

FROM

(SELECT MAX(Year) AS Year

FROM Indicators,Country

WHERE Indicators.CountryCode = Country.CountryCode

AND Country.ShortName IN """+ListToString(CountrysList)+"""

AND Indicators.IndicatorName IN """+ListToString(IndicatorsList)+"""

AND Indicators.Value IS NOT NULL

GROUP BY Country.ShortName,Indicators.CountryCode,IndicatorName,IndicatorCode) AS data

;

"""

NumberOfNonNulls=DataFrameTable(query)

display(NumberOfNonNulls.head())



print("There should be",len(IndicatorData.index)*len(IndicatorData.columns),"values.")
query = """

SELECT data.CountryName,data.IndicatorName,data.Year,I.Value

FROM

(SELECT Country.ShortName AS CountryName,Indicators.CountryCode,IndicatorName,IndicatorCode,MAX(Year) AS Year

FROM Indicators,Country

WHERE Indicators.CountryCode = Country.CountryCode

AND Country.ShortName IN """+ListToString(CountrysList)+"""

AND Indicators.IndicatorName IN """+ListToString(IndicatorsList)+"""

AND Indicators.Value IS NOT NULL

GROUP BY Country.ShortName,Indicators.CountryCode,IndicatorName,IndicatorCode) AS data

LEFT JOIN Indicators I ON data.CountryCode = I.CountryCode AND data.IndicatorCode = I.IndicatorCode and I.Year = data.Year

;

"""



IndicatorData=DataFrameTable(query)

IndicatorData=IndicatorData.pivot("CountryName","IndicatorName","Value")

IndicatorData.columns.name = None 

display(IndicatorData.head(5))

display(IndicatorData.describe())
IndicatorData.to_csv("IndicatorData.csv")
import pandas as pd

import numpy as np 





IndicatorData=pd.read_csv("../input/development-data/IndicatorData.csv")

IndicatorData=IndicatorData.set_index(["CountryName"])

display(IndicatorData.head())

IndicatorData.describe()
Data=IndicatorData[[col for col in IndicatorData.columns if IndicatorData[col].count()>=147]]

print("We now have",len(Data.columns),"features instead of",len(IndicatorData.columns))

print("If we drop counries with missing data we have",len(Data.dropna()),"countries out of",len(Data))
Data=Data.dropna()

IndicatorDF=Data

display(Data.columns)
IndicatorDF["Refugee Rate"]=IndicatorDF["Refugee population by country or territory of origin"]/IndicatorDF["Population, total"]

IndicatorDF=IndicatorDF.drop(columns=["Refugee population by country or territory of origin"])
HappinessFileString="../input/world-happiness/2015.csv"

HappinessDF=pd.read_csv(HappinessFileString)
IndicatorDF.head()
CountrysToReplace={

    "Slovakia":"Slovak Republic",

    "South Korea":"Korea",

    "Hong Kong":"Hong Kong SAR, China",

    "Kyrgyzstan":"Kyrgyz Republic",

    "Laos":"Lao PDR",

    "Congo (Kinshasa)":"Dem. Rep. Congo",

    "Congo (Brazzaville)":"Congo",

    "Ivory Coast":"Côte d''Ivoire",

    "Syria":"Syrian Arab Republic"} 



Index=HappinessDF["Country"].tolist()

Index=[country if country not in CountrysToReplace.keys() else CountrysToReplace[country] for country in Index ]

HappinessDF.index=Index

HappinessDF=HappinessDF[["Happiness Score"]]

HappinessDF.head()
DataDF=IndicatorDF.join(HappinessDF,how="inner")

DataDF.head()
DataDF.info()
DataDF.describe()
DataDF.hist(figsize=(40,40),bins=50, xlabelsize=10, ylabelsize=10)
import seaborn as sns



for i in range(0, len(DataDF.columns), 5):

    sns.pairplot(data=DataDF,x_vars=DataDF.columns[i:i+5],y_vars=["Happiness Score"],height=5)

CorMatrix=DataDF.corr()

CorMatrix.style.background_gradient(cmap='coolwarm')
X=DataDF.drop(columns=["Happiness Score"]).copy()

Y=DataDF["Happiness Score"].copy()
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression



model=LinearRegression(normalize=True)

params={}



clf = GridSearchCV(model,params,cv=5,n_jobs=-1,iid=True)

clf.fit(X,Y)

results=pd.DataFrame(data=clf.cv_results_)

BestScore = results[results["mean_test_score"]==results["mean_test_score"].max()]["mean_test_score"].values[0]

print("The score on the training set is",BestScore)

print("The models parametres are",results[results["mean_test_score"]==BestScore]["params"].values[0])
from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt



model=Lasso(normalize=True)

params = {

    "alpha":np.logspace(-5,1)

}



clf = GridSearchCV(model,params,cv=5,n_jobs=-1,iid=True)

clf.fit(X,Y)

results=pd.DataFrame(data=clf.cv_results_)

BestScore = results[results["mean_test_score"]==results["mean_test_score"].max()]["mean_test_score"].values[0]

print("The score on the training set is",BestScore)

print("The models parametres are",results[results["mean_test_score"]==BestScore]["params"].values[0])





for param,values in params.items():

    plt.plot( results["param_"+param],results["mean_test_score"])

    plt.xscale("log")

    plt.ylabel(r"$R^2$")

    plt.xlabel(param)

    plt.gca().spines['top'].set_visible(False)

    plt.gca().spines['right'].set_visible(False)

    plt.show()
from sklearn.linear_model import Ridge



model=Ridge(normalize=True)

params = {

    "alpha":np.logspace(-6,1)

}



clf = GridSearchCV(model,params,cv=5,n_jobs=-1,iid=True)

clf.fit(X,Y)

results=pd.DataFrame(data=clf.cv_results_)

BestScore = results[results["mean_test_score"]==results["mean_test_score"].max()]["mean_test_score"].values[0]

print("The score on the training set is",BestScore)

print("The models parametres are",results[results["mean_test_score"]==BestScore]["params"].values[0])



for param,values in params.items():

    plt.plot( results["param_"+param],results["mean_test_score"])

    plt.xscale("log")

    plt.ylabel(r"$R^2$")

    plt.xlabel(param)

    plt.gca().spines['top'].set_visible(False)

    plt.gca().spines['right'].set_visible(False)

    plt.show()
from sklearn.preprocessing import PolynomialFeatures

transformer= PolynomialFeatures(degree=2)

PolyX = transformer.fit_transform(X.copy())
from sklearn.model_selection import cross_validate



model=LinearRegression(normalize=True)



results=pd.DataFrame(data=cross_validate(model,PolyX,Y,cv=5,return_train_score=True),index=range(1,6))

results.index=results.index.rename("Fold")

display(results)
from sklearn.linear_model import Lasso



model=Lasso(normalize=True)

params = {

    "alpha":np.logspace(-3.5,0)

}



clf = GridSearchCV(model,params,cv=5,n_jobs=-1,iid=True)

clf.fit(PolyX,Y)

results=pd.DataFrame(data=clf.cv_results_)

BestScore = results[results["mean_test_score"]==results["mean_test_score"].max()]["mean_test_score"].values[0]

print("The score on the training set is",BestScore)

print("The models parametres are",results[results["mean_test_score"]==BestScore]["params"].values[0])





for param,values in params.items():

    plt.plot( results["param_"+param],results["mean_test_score"])

    plt.xscale("log")

    plt.ylabel(r"$R^2$")

    plt.xlabel(param)

    plt.gca().spines['top'].set_visible(False)

    plt.gca().spines['right'].set_visible(False)

    plt.show()





model=Ridge(normalize=True)

params = {

    "alpha":np.logspace(-1,2)

}



clf = GridSearchCV(model,params,cv=5,n_jobs=-1,iid=True)

clf.fit(PolyX,Y)

results=pd.DataFrame(data=clf.cv_results_)

BestScore = results[results["mean_test_score"]==results["mean_test_score"].max()]["mean_test_score"].values[0]

print("The score on the training set is",BestScore)

print("The models parametres are",results[results["mean_test_score"]==BestScore]["params"].values[0])



for param,values in params.items():

    plt.plot( results["param_"+param],results["mean_test_score"])

    plt.xscale("log")

    plt.ylabel(r"$R^2$")

    plt.xlabel(param)

    plt.gca().spines['top'].set_visible(False)

    plt.gca().spines['right'].set_visible(False)

    plt.show()
# first we need to normalize the data 

from sklearn.preprocessing import StandardScaler

transformer=StandardScaler()

NormX = transformer.fit_transform(X)
from sklearn.svm import SVR 



model=SVR()

params={

    "C"       : np.logspace(-2,2),

    "epsilon" : np.logspace(-2,2),

    "kernel"  : ["rbf","poly","sigmoid"]

}



clf = GridSearchCV(model,params,cv=5,n_jobs=-1,iid=True,verbose=1)

clf.fit(NormX,Y)

results=pd.DataFrame(data=clf.cv_results_)

BestScore = results[results["mean_test_score"]==results["mean_test_score"].max()]["mean_test_score"].values[0]

print("The score on the training set is",BestScore)



BestParams = results[results["mean_test_score"]==BestScore]["params"].values[0]

print("The models parametres are",BestParams)
from sklearn.ensemble import RandomForestRegressor



model= RandomForestRegressor()

params={

    "n_estimators" : [int(x) for x in np.linspace(1,1000,8)],

    "max_depth"    : [x for x in np.logspace(0,3,8)]+[None]

}



clf = GridSearchCV(model,params,cv=5,n_jobs=-1,iid=True,verbose=1)

clf.fit(NormX,Y)

results=pd.DataFrame(data=clf.cv_results_)

BestScore = results[results["mean_test_score"]==results["mean_test_score"].max()]["mean_test_score"].values[0]

print("The score on the training set is",BestScore)



BestParams = results[results["mean_test_score"]==BestScore]["params"].values[0]

print("The models parametres are",BestParams)
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV



model=Ridge(normalize=True)

params = {

    "alpha":np.logspace(-1.6,-0.2)

}



clf = GridSearchCV(model,params,cv=5,n_jobs=-1,iid=True)

clf.fit(X,Y)

results=pd.DataFrame(data=clf.cv_results_)

BestScore = results[results["mean_test_score"]==results["mean_test_score"].max()]["mean_test_score"].values[0]

print("The score on the training set is",BestScore)

print("The models parametres are",results[results["mean_test_score"]==BestScore]["params"].values[0])



for param,values in params.items():

    plt.plot( results["param_"+param],results["mean_test_score"])

    plt.xscale("log")

    plt.ylabel(r"$R^2$")

    plt.xlabel(param)

    plt.gca().spines['top'].set_visible(False)

    plt.gca().spines['right'].set_visible(False)

    plt.show()
model=Ridge(normalize=True,alpha=0.11406249238513208)

model.fit(X,Y)

Predictions=X.copy()

Predictions["Prediction"]=Predictions.apply(

    lambda row: model.predict([[(row[col]-Predictions[col].mean())/Predictions[col].std() for col in Predictions.columns]])[0],

    axis=1

)



Predictions = Predictions.join(pd.DataFrame(Y),how="inner")

pd.options.display.max_columns=200

display(Predictions[["Prediction","Happiness Score"]].T)

from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold

from sklearn.metrics import r2_score



# Leave one out

folds = KFold(n_splits=len(X))



predictions=[]

true=[]

for train,test in folds.split(NormX):

    model=Ridge(normalize=True,alpha=0.11406249238513208)

    X_train, X_test = NormX[train], NormX[test]

    y_train, y_test = Y.values[train], Y.values[test]

    model.fit(X_train,y_train)

    predictions.append(model.predict(X_test))

    true.append(y_test)

    

predictions=[x[0] if x>0 else 0 for x in predictions]

predictions=[x if 10>x else 10 for x in predictions]

true= [x[0] for x in true]

print("The new R squared value is",r2_score(true,predictions))



display(pd.DataFrame({"Predication":predictions,"Happiness Score":true},index=X.index).T)
Coeficients=[(col,coef) for col,coef in zip(X.columns,model.coef_)]

Coeficients = sorted(Coeficients,key=lambda x:np.abs(x[1]),reverse=True)

Coeficients={col:coef for col,coef in Coeficients}

pd.DataFrame(index=Coeficients.keys(),data={"Coefficient":list(Coeficients.values())}).T