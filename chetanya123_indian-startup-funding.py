import pandas as pd 

from matplotlib import pyplot as plt

%matplotlib inline

df = pd.read_csv("../input/indian-startup-funding/startup_funding.csv")

df
df.head()
df.columns
df.shape
df.info()
df["Industry Vertical"].value_counts()
filter = df["Industry Vertical"].isin(["Consumer Internet", "Technology" , "eCommerce"])

df[filter]
filter.shape
New = df.drop(columns=["Remarks"])

New
New.columns
New=New.dropna(subset=["Amount in USD","City  Location","Industry Vertical","SubVertical","InvestmentnType"])

New
New['Amount in USD'] = New['Amount in USD'].str.replace(',', '')

New

New["Amount in USD"].max()
New["Amount in USD"].min()
New['InvestmentnType'].value_counts()
f=New['InvestmentnType'].unique()

y=","

S7=y.join(f)+" are the different types of funding in india"

print(S7)
fig,ax=plt.subplots()

New['InvestmentnType'].value_counts().plot(kind='bar',legend=True)

ax.grid()

ax.set_axisbelow(True)

ax.grid(linestyle='-',linewidth='0.2',color='grey')
it=New['InvestmentnType']

it.value_counts().plot(kind='pie',legend=True)

S3=list(it.mode())

y=""

S3=y.join(S3)+" type of companies got more easily funding"

print(S3)

print(New['InvestmentnType'].value_counts().max(),"is the count of funding")
New['Industry Vertical'].value_counts()
New['Industry Vertical'].value_counts().plot(kind='barh',legend=True)
b=New['Industry Vertical']

b.value_counts().plot(kind='pie',legend=True)

S4=list(b.mode())

y=""

S4=y.join(S4)+" type of companies got more easily funding"

print(S4)
New['City  Location'].value_counts()
fig,ax=plt.subplots()

New['City  Location'].value_counts().plot(kind='bar',legend=True)

ax.grid()

ax.set_axisbelow(True)

ax.grid(linestyle='-',linewidth='0.2',color='grey')
a=New['City  Location'].head()

a.value_counts().plot(kind='pie',legend=True)

S5=list(a.mode())

x=""

S5=x.join(S5)+" has the highest no of startups"

print(S5)
d=New['Investorsxe2x80x99 Name'].head()

d.value_counts().plot(kind='pie',legend=True)

X=list(d.mode())

v=" "

S6=v.join(X)+" are the important investors in the indian ecosystem"

print(S6)
print("CONCLUSION")

print("Now after performing data analysis on the obtained dataset we have arrived on the following conclusions:")

#print("1.",S1)

#print("2.",S2)

print("3.",S3)

print("4.",S4)

print("5.",S5)

print("6.",S6)

print("7.",S7)