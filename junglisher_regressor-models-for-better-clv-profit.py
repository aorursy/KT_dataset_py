import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv(r'../input/vehicle-insurance-data/VehicleInsuranceData.csv')
df.head(5)
sns.heatmap(df.isnull(), yticklabels=False,cbar=False, cmap='viridis')
sns.boxplot(df.clv)
df= df[(df.clv>2500) & (df.clv < 15000)]     
# according to boxplot any data below or above, Q1 or Q3 respectively are outliers.
df.shape
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()

for i in df.columns:
    if isinstance(df[i][0], str):
            df[i] = encoder.fit_transform(df[i])
df.head(2)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_vif = add_constant(df)

pd.Series([variance_inflation_factor(X_vif.values, i) 
               for i in range(X_vif.shape[1])], 
              index=X_vif.columns)
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

X = df[['Coverage','Monthly.Premium.Auto','Number.of.Policies','Renew.Offer.Type','Total.Claim.Amount','Vehicle.Class']]

y = df['clv']
drake= np.log(X+1)
from sklearn.preprocessing import StandardScaler 
  
scalar = StandardScaler() 
  
scalar.fit(drake) 
scaled_data = scalar.transform(drake) 
kiki = np.log(y)
scaled_data = pd.DataFrame(data=scaled_data, columns=['Coverage', 'Monthly.Premium.Auto', 'Number.of.Policies',
       'Renew.Offer.Type', 'Total.Claim.Amount', 'Vehicle.Class'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_data , kiki, test_size=0.3, random_state=200)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Training accuracy=',lm.score(X_train,y_train)*100)
pred = lm.predict(X_test)
from sklearn import metrics
from sklearn.metrics import accuracy_score
print('Prediction accuracy =',metrics.explained_variance_score(y_test, pred)*100)
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, pred)))
fig=plt.figure(figsize=(10,6))
plt.scatter(np.arange(1,100,10),pred[0:100:10],color='blue')
plt.scatter(np.arange(1,100,10),y_test[0:100:10],color='yellow')

plt.legend(['prediction','test'])
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])
cdf
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 3) 
X_poly = poly.fit_transform(scaled_data.iloc[:,0].values.reshape(-1,1)) 
  

lin2 = LinearRegression() 
lin2.fit(X_poly, y)
X_poly.shape
from sklearn.preprocessing import PolynomialFeatures 

def check_exp(inp,degree,out):
    
    poly = PolynomialFeatures(degree = degree) 
    X_poly = poly.fit_transform(inp) 


    lin2 = LinearRegression() 
    lin2.fit(X_poly, out)
    
    return lin2.score(X_poly, out)

for a in range(X.shape[1]):
    acc= []
    for i in range(10):
        acc.append(check_exp(X.iloc[:,a].values.reshape(-1,1), i, y))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,X.shape[1]*2))
    sb = (X.shape[1]*10+1)*10+(a+1)
    plt.subplot(sb)
    plt.title('column : '+str(a))
    plt.xlabel('degrees')
    plt.ylabel('accuracy')
    plt.plot(acc)
poly=PolynomialFeatures(degree=1)
X_poly= poly.fit_transform(X.iloc[:,0].values.reshape(-1,1))  #0
poly=PolynomialFeatures(degree=1)
X_poly1= poly.fit_transform(X.iloc[:,1].values.reshape(-1,1))  #1
poly=PolynomialFeatures(degree=4)
X_poly2= poly.fit_transform(X.iloc[:,2].values.reshape(-1,1))  #2
poly=PolynomialFeatures(degree=2)
X_poly3= poly.fit_transform(X.iloc[:,3].values.reshape(-1,1))  #3
poly=PolynomialFeatures(degree=1)
X_poly4= poly.fit_transform(X.iloc[:,4].values.reshape(-1,1))  #4
poly=PolynomialFeatures(degree=2)
X_poly5= poly.fit_transform(X.iloc[:,5].values.reshape(-1,1))  #5
Xo = np.concatenate((X_poly,X_poly1,X_poly2,X_poly3,X_poly4,X_poly5), axis=1)
Xo.shape
Xo
X_train, X_test, y_train, y_test = train_test_split(Xo, kiki, test_size=0.33, random_state=42)
lm.fit(X_train,y_train)
print('Training score =',lm.score(X_train,y_train)*100,'%')
pred = lm.predict(X_test)
from sklearn.metrics import accuracy_score
print ('Prediction accuracy =',metrics.explained_variance_score(y_test, pred)*100,'%')
cdf = pd.DataFrame(lm.coef_,columns=['coeff'])
cdf
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, pred)))
fig=plt.figure(figsize=(10,6))
plt.scatter(np.arange(1,100,10),pred[0:100:10],color='blue')
plt.scatter(np.arange(1,100,10),y_test[0:100:10],color='yellow')

plt.legend(['prediction','test'])
X_train, X_test, y_train, y_test = train_test_split(scaled_data, kiki, test_size=0.3, random_state=42)
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.1)
my_model.fit(X_train, y_train, early_stopping_rounds = 5,
             eval_set=[(X_train, y_train)], verbose=False)
my_model.score(X_train, y_train)*100
pred = my_model.predict(X_test)

print('Prediction accuracy =',metrics.explained_variance_score(y_test, pred)*100)
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, pred)))
fig=plt.figure(figsize=(10,6))
plt.scatter(np.arange(1,100,10),pred[0:100:10],color='blue')
plt.scatter(np.arange(1,100,10),y_test[0:100:10],color='yellow')

plt.legend(['prediction','test'])
