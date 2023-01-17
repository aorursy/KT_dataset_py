import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
app1=pd.read_csv("../input/playstore-analysis/googleplaystore.csv");app1.head()
print(app1.dtypes,app1.isnull().sum())
app1.shape
app1.dropna(inplace=True);print(app1.shape)

print(app1.Size.value_counts())

def change(Size):
    if 'M'in Size:
        x=Size[:-1]
        x=float(x)*1000
        return x

    elif 'k'in Size:
        x=Size[:-1]
        x=float(x)
        return x
    
    else: return None

    
app1.Size=app1.Size.map(change);app1.Size.value_counts()

    
print(app1.Size.isnull().sum())
app1.Size.fillna(method='pad',inplace=True)
print(app1.Size.isnull().sum())
app1.Reviews=app1.Reviews.astype('float')

print(app1.Installs.value_counts()[:5])
app1.Installs=app1.Installs.map(lambda x:x.replace(',','').replace('+',''))
print(app1.Installs.value_counts()[:5])
app1.Installs=app1.Installs.astype('float')

print(app1.Price.value_counts()[:5])
app1.Price=app1.Price.map(lambda x:x.replace('$',''))
print(app1.Price.value_counts()[:5])
app1.Price=app1.Price.astype('float')

print(app1.dtypes)
                          
print(len(app1[app1.Rating>5]))
print(len(app1[app1.Reviews>app1.Installs]))
print(len(app1[(app1.Type=='free')&(app1.Price>0)]))

app1=app1[app1.Reviews<app1.Installs].copy();print(app1.shape)

print(len(app1[app1.Price>200]))
app1=app1[app1.Price<200].copy();print(app1.shape)

print(len(app1[app1.Reviews>=2000000]))
app1=app1[app1.Reviews<=2000000].copy();print(app1.shape)

print(app1.Installs.quantile([.25,.50,.75,.90,.99]))
print(len(app1[app1.Installs>= 10000000]))
app1=app1[app1.Installs<=10000000].copy();print(app1.shape)
print(app1.hist(['Rating','Reviews','Size','Installs','Price'],figsize=(12,8),xlabelsize=12,ylabelsize=12))

app1.boxplot(fontsize=15)

app1.Reviews=app1.Reviews.apply(func=np.log1p)
app1.Installs=app1.Installs.apply(func=np.log1p)

app1.hist(column=['Reviews','Installs'])

plt.figure(figsize=(11,8))
sns.set_style(style='whitegrid',)
sns.set(font_scale=1.2)
sns.scatterplot(app1.Price,app1.Rating,hue=app1.Rating)

plt.show()
plt.figure(figsize=(11,8))
sns.scatterplot(app1.Size,app1.Rating,hue=app1.Rating)
plt.figure(figsize=(11,8))
sns.scatterplot(app1.Reviews,app1.Rating,hue=app1.Rating)
plt.figure(figsize=(12,8.27))
sns.boxplot(app1['Content Rating'],app1.Rating)
plt.figure(figsize=(25,8.27))
sns.boxplot(app1.Category,app1.Rating)
plt.xticks(fontsize=18,rotation='vertical')
plt.yticks(fontsize=18)
plt.xlabel("Category",fontsize=20)
plt.ylabel("Rating",fontsize=20)
app1.drop(["App", "Last Updated", "Current Ver", "Android Ver"], axis=1, inplace=True);print(app1.shape)
app1=pd.get_dummies(app1,drop_first=True);print(app1.columns)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
from statsmodels.api import OLS
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as ms
X=app1.iloc[:,1:]
y=app1.iloc[:,:1]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=1)
X_train.shape,X_test.shape
Model=linreg.fit(X_train, y_train)
predict=linreg.predict(X_test)

y_test=np.array(y_test)
predict=np.array(predict)

a=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':predict.flatten()});a.head(10)

fig=a.head(25)
fig.plot(kind='bar',figsize=(10,8))
results=OLS( y_train,X_train).fit()
results.summary()
print('R2_Score=',r2_score(y_test,predict))
print('Root Mean Squared Error=',np.sqrt(ms(y_test,predict)))
print('Prediction Error Percentage is',round((0.50/np.mean(y_test))*100))