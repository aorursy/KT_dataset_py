import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
data = pd.read_csv('house_prediction.csv')
data.shape
#from text editor high have replced - with 0.
variation= data.groupby(['city']).mean()
plt.bar(['Belo Horizonte','Campinas','Porto Alegre city','Rio de janerio','SÃ£o Paulo'],variation['area'],color='green')
plt.xlabel('city')
plt.show()
variation= data.groupby(['city']).mean()
plt.bar(['Belo Horizonte','Campinas','Porto Alegre city','Rio de janerio','SÃ£o Paulo'],variation['rooms'],color='black')
plt.xlabel('city')
plt.show()


variation= data.groupby(['city']).mean()
plt.bar(['Belo Horizonte','Campinas','Porto Alegre city','Rio de janerio','SÃ£o Paulo'],variation['bathroom'],color='blue')
plt.xlabel('city')
plt.show()
variation= data.groupby(['city']).mean()
plt.bar(['Belo Horizonte','Campinas','Porto Alegre city','Rio de janerio','SÃ£o Paulo'],variation['parking spaces'],color='grey')
plt.xlabel('city')
plt.show()
variation= data.groupby(['city']).mean()
plt.bar(['Belo Horizonte','Campinas','Porto Alegre city','Rio de janerio','SÃ£o Paulo'],variation['floor'],color='pink')
plt.xlabel('city')
plt.show()
variation= data.groupby(['city']).mean()
plt.bar(['Belo Horizonte','Campinas','Porto Alegre city','Rio de janerio','SÃ£o Paulo'],variation['rent amount (R$)'],color='yellow')
plt.xlabel('city')
plt.show()
variation= data.groupby(['city']).mean()
plt.bar(['Belo Horizonte','Campinas','Porto Alegre city','Rio de janerio','SÃ£o Paulo'],variation['property tax (R$)'],color='blue')
plt.xlabel('city')
plt.show()

variation= data.groupby(['city']).mean()
plt.bar(['Belo Horizonte','Campinas','Porto Alegre city','Rio de janerio','SÃ£o Paulo'],variation['fire insurance (R$)'],color='orange')
plt.xlabel('city')
plt.show()

variation= data.groupby(['city']).mean()
plt.bar(['Belo Horizonte','Campinas','Porto Alegre city','Rio de janerio','SÃ£o Paulo'],variation['hoa (R$)'],color='pink')
plt.xlabel('city')
plt.show()

#plotted mean of all features and taxes.
df = pd.DataFrame(data,columns=['area','rooms','bathroom','parking spaces','floor','hoa (R$)','property tax (R$)','fire insurance (R$)'])
corrMatrix =df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#plotted correlational matrix so if we want to know how fire insurance depend on area we have to go to corresponding 
#block which is rightmost top corner and if its highly dependent its colour is white and vice versa .
X=data[['area','rooms','bathroom','floor','parking spaces','hoa (R$)','property tax (R$)','fire insurance (R$)']].values
y=data['rent amount (R$)'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=(692/10692),random_state=0)
reg = LinearRegression()  
reg.fit(X_train, y_train)
reg.coef_
y_pred = reg.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df = df.head(25)
df.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='yellow')
plt.show()
#for finding accuracy
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))