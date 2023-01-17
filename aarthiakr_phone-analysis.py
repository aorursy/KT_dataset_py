import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/MobilePhones.csv')
len(df)
df.info()
df.describe()
df.head()
df['MobileName'][0].split("(")[1].split(",")
df['Color'] = df['MobileName'].apply(lambda x : x.split(",")[0].split("(")[1] 
                                        if len(x.split(",")[0].split("(")) > 1 else 'No Color')

df.head()
df['Brand'] = df['MobileName'].apply(lambda x : x.split()[0])
df['Brand'] = df['Brand'].apply(lambda x : 'I Kall' if x == 'I' else x)
df.head()
df['MobileName'] = df['MobileName'].apply(lambda x : x.split("(")[0])
df.head()
df['Discount'] = df['ListPrice'] - df['SalesPrice']
df.head()
print(df['Brand'].value_counts())

plt.figure(figsize=(10,5))
sns.countplot('Brand', data=df)
print(df['MobileName'].value_counts()[:20])

plt.figure(figsize=(20,5))
sns.countplot('MobileName', data=df)
plt.xticks(rotation=60)
print(df['Stars'].value_counts())

sns.distplot(df['Stars'])
print("Phones with lowest stars")
print("\n".join(df[df['Stars']==3.0]['MobileName'].unique()))

print("\nPhones with highest stars")
print("\n".join(df[df['Stars']==4.6]['MobileName'].unique()))
plt.figure(figsize=(10,5))
sns.barplot(df['Brand'],df['Stars'],data=df)
discount = df['Discount'].value_counts()[:5]
discount.plot(kind='bar',title='Top 5 Discount Rate (In Rupees)')
print("*** RAM *** ")
print(df['RAM_GB'].value_counts())
print("\n*** ROM *** ")
print(df['ROM_GB'].value_counts())
print(df[df['RAM_GB'] == 32].index)
print(df[df['ROM_GB'] == 4].index)
df.drop([115,118,80], inplace=True,axis=0)
print("*** RAM *** ")
print(df['RAM_GB'].value_counts())
print("\n*** ROM *** ")
print(df['ROM_GB'].value_counts())

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.title("RAM Space in GB")
sns.countplot('RAM_GB', data=df)
plt.xlabel("GB")

plt.subplot(1,2,2)
plt.title("ROM Space in GB")
sns.countplot('ROM_GB', data=df)
plt.xlabel("GB")
print(df['Color'].value_counts()[:10])

popcol = df['Color'].value_counts()[:10]

plt.figure(figsize=(10,5))
popcol.plot(kind='bar')
plt.figure(figsize=(10,5))

plt.subplot(2,2,1)
plt.title("Ratings")
sns.boxplot('Ratings', data=df)

plt.subplot(2,2,2)
plt.title("Reviews")
sns.boxplot('Reviews', data=df)

plt.subplot(2,2,3)
plt.title("List Price")
sns.boxplot('ListPrice', data=df)

plt.subplot(2,2,4)
plt.title("Sales Price")
sns.boxplot('SalesPrice', data=df)

plt.tight_layout(pad=2.0)
df = df[df['Reviews'] < 5500]
df = df[df['Ratings'] < 60000]
plt.figure(figsize=(10,5))

plt.subplot(2,2,1)
plt.title("Ratings")
sns.boxplot('Ratings', data=df)

plt.subplot(2,2,2)
plt.title("Reviews")
sns.boxplot('Reviews', data=df)

plt.subplot(2,2,3)
plt.title("List Price")
sns.boxplot('ListPrice', data=df)

plt.subplot(2,2,4)
plt.title("Sales Price")
sns.boxplot('SalesPrice', data=df)

plt.tight_layout(pad=2.0)
df = df[df['ListPrice'] < 30000]
plt.figure(figsize=(10,5))

plt.subplot(2,2,1)
plt.title("Ratings")
sns.boxplot('Ratings', data=df)

plt.subplot(2,2,2)
plt.title("Reviews")
sns.boxplot('Reviews', data=df)

plt.subplot(2,2,3)
plt.title("List Price")
sns.boxplot('ListPrice', data=df)

plt.subplot(2,2,4)
plt.title("Sales Price")
sns.boxplot('SalesPrice', data=df)

plt.tight_layout(pad=2.0)
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True)
plt.figure(figsize=(10,5))

plt.suptitle("Correlation Between Attributes")

plt.subplot(1,2,1)
plt.title("Postive Correlation")
plt.scatter(df['Ratings'],df['Reviews'], marker='v')
plt.xlabel("Ratings")
plt.ylabel("Review")

plt.subplot(1,2,2)
plt.title("Negative Correlation")
plt.scatter(df['Discount'],df['Reviews'], marker='v')
plt.xlabel("Discount in Rupees")
plt.ylabel("Ratings")

plt.tight_layout(pad=3.5)
table = pd.pivot_table(df, index='Brand', values=['SalesPrice','Discount','Ratings'])
table
table.plot(kind='bar',figsize=(10,5))
plt.figure(figsize=(10,5))
sns.countplot('Brand', data=df)
df.head()
df.Brand.value_counts().index
df['Brand'] = df['Brand'].map({'Realme':0,'Vivo':1,'OPPO':2,'I Kall':3,'Redmi':4,
                               'Infinix':5,'POCO':7,'Motorola':8,'Tecno':9})   
df.head(10)
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
dfnumberic = df.select_dtypes(include=[np.number])
dfnumberic.head()
print("Shape of the numberic data frame")
print(dfnumberic.shape)
X = dfnumberic.drop('SalesPrice',axis=1)
y = dfnumberic['SalesPrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LinearRegression
lrm = LinearRegression()
lrm.fit(X_train,y_train)
predictionslrm = lrm.predict(X_test)
scorelrm = round((lrm.score(X_test, y_test)*100),2)
print ("Model Score:",scorelrm,"%")
from sklearn.linear_model import Ridge
rrm = Ridge(alpha=100)
rrm.fit(X_train,y_train)
predictionrrm = rrm.predict(X_test)
scorerrm = round((rrm.score(X_test, y_test)*100),2)
print ("Model Score:",scorerrm,"%")
data = [['Linear Regression',scorelrm],['Ridge Regression',scorerrm]]
final = pd.DataFrame(data,columns=['Algorithm','Precision'],index=[1,2])
final
df.head()
df['UserType'] = 'Teen'
high = df[(df['RAM_GB'] > 4) & (df['ROM_GB'] > 32)].index
low =  df[(df['SalesPrice'] < 12000) & (df['ROM_GB'] < 64)].index
for i in high:
    df['UserType'].loc[i] = 'High'
for i in low:
    if i not in high:
        df['UserType'].loc[i] = 'Low'
df['UserType'].value_counts()
df.head()
df['UserType'] = df['UserType'].map({'High':0,'Teen':1,'Low':2})
dfnumberic = df.select_dtypes(include=[np.number]).drop('ListPrice', axis=1)
dfnumberic.head()
dfnumberic[['Ratings','Reviews','Stars','SalesPrice','Discount']].describe()
def ratings(num):
    if num < 10000:
        return 1
    elif num >= 10000 & num < 20000:
        return 2
    elif num >= 20000 & num < 30000:
        return 3
    elif num >= 30000 & num < 40000:
        return 4
    elif num >= 40000 & num < 50000:
        return 5
    else:
        return 6
    
    
def reviews(num):
    if num < 1000:
        return 1
    elif num >= 1000 & num < 2000:
        return 2
    elif num >= 2000 & num < 3000:
        return 3
    elif num >= 3000 & num < 4000:
        return 4
    else:
        return 5

    

def salesprice(num):
    if num < 5000:
        return 1
    elif num >= 5000 & num < 10000:
        return 2
    elif num >= 10000 & num < 15000:
        return 3
    elif num >= 15000 & num < 20000:
        return 4
    else:
        return 5
    
def stars(num):
    if num < 3.0:
        return 1
    elif num >= 3 and num < 3.5:
        return 2
    elif num >= 3.5 and num < 4.0:
        return 3
    elif num >= 4.0 and num < 4.5:
        return 4
    else:
        return 5


def discount(num):
    if num == 0:
        return 0
    elif num < 1200:
        return 1
    elif num >= 1200 & num < 2400:
        return 2
    elif num >= 2400 & num < 3600:
        return 3
    elif num >= 4800 & num < 6000:
        return 4
    else:
        return 5
dfnumberic['Ratings'] = dfnumberic['Ratings'].apply(ratings)
dfnumberic['Reviews'] = dfnumberic['Reviews'].apply(reviews)
dfnumberic['Stars'] = dfnumberic['Stars'].apply(stars)
dfnumberic['SalesPrice'] = dfnumberic['SalesPrice'].apply(salesprice)
dfnumberic['Discount'] = dfnumberic['Discount'].apply(discount)
dfnumberic.head()
X = dfnumberic.drop(['UserType'],axis=1)
y = dfnumberic['UserType']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.ensemble import RandomForestClassifier
rmodel = RandomForestClassifier(n_estimators=100)
rmodel.fit(X_train,y_train)
rprediction = rmodel.predict(X_test)
print("Confusion Matrix")
print(confusion_matrix(y_test,rprediction))

rscore = round((rmodel.score(X_test, y_test)*100),2)
print ("\nModel Score:",rscore,"%")