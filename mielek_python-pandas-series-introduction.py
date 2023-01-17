import pandas as pd
d =pd.read_csv("../input/Death_United_States.csv")

d
d["State"].tolist()
state=d["State"].to_frame()
state.to_csv("state.csv",index=False)
pd.read_csv("state.csv")
c = pd.read_excel("../input/excel.xlsx",sheet_name=["2006","2010"])
c["2010"]
four=pd.read_csv("../input/2014 wc.csv")
four.to_excel("2014.xlsx",sheet_name="2014")
years = [2015,2016,2017,2018,2019]

income = [10000,20000,30000,40000,50000]



pd.Series(data = income ,index = years)
number = [1,2,4,6,8,3,34]

number=pd.Series(number)
number
number.sum()
number.max()
number.min()
number.product()
number.mean()
import pandas as pd
x=[1,2,3,5,6]
x=pd.Series(x)
country=["usa","china","turkey","france"]
country=pd.Series(country)
d=[True,True,False]
pd.Series(d)
income={"usa":80000,"germany":50000,"turkey":15000}
pd.Series(income)
country.values
country.index
country.dtype
x.shape
country.name="Countries"
country.head()
country=pd.read_csv("../input/ulke.csv",squeeze=True)

income=pd.read_csv("../input/milligelir.csv",squeeze=True)

country
country.sort_values().head()
income.sort_values(ascending=False).tail()
income.sort_values(ascending=True,inplace=True)
income
income.sort_index()
1 in [1,2,3,4]
"USA" in country.values
country=pd.read_csv("../input/ulke.csv",squeeze=True)

income=pd.read_csv("../input/milligelir.csv",squeeze=True)
len(income)
type(country)
sorted(income)
list(income)
dict(income)
max(income)
min(income)
country=pd.read_csv("../input/ulke.csv",squeeze=True)

country
country[:10]
income=pd.read_csv("../input/milligelir.csv",squeeze=True)

income
income.count()
len(income)
income.sum()
income.mean()
income.std()
income.max()
income.min()
income.median()
income.describe()
country=pd.read_csv("../input/ulke.csv")

country
country=pd.read_csv("../input/ulke.csv",squeeze=True)

country
income=pd.read_csv("../input/milligelir.csv")

income
income.head(10)
income.tail(10)
continent=pd.read_csv("../input/kta.csv",squeeze=True)
continent.value_counts()
continent.value_counts(ascending=True)
nat_income=pd.read_csv("../input/milligelir.csv",squeeze=True)
nat_income.idxmax()
nat_income[0]
nat_income.idxmin()
nat_income[19]
def classs(gel):

    if gel < 2000000:

        return "medium"

    elif gel >=2000000 and gel <=5000000:

        return "high"

    else:

        return "too high"
nat_income.apply(classs)
nat_income.apply(lambda nat_income:nat_income*2)