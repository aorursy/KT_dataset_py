import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
df.head()
df.drop(['Lat','Long'],axis=1,inplace=True)
df['Date']=pd.to_datetime(df['Date'])
df.head()
byCountry=df.groupby(['Country/Region','Date']).sum().sort_values(by=['Country/Region','Date'])
byCountry.head()
byCountry.xs(level='Country/Region',key='India').plot()
byCountry['Mortality_rate']=byCountry['Deaths']/byCountry['Confirmed']
byCountry['Recovery_rate']=byCountry['Recovered']/byCountry['Confirmed']
byCountry.head()
def display(ct,ch):
    choice=[]
    if(1 in ch):
        choice.append('Confirmed')
    if(2 in ch):
        choice.append('Deaths')
    if(3 in ch):
        choice.append('Recovered')
    if(4 in ch):
        choice.append('Mortality_rate')
    if(5 in ch):
        choice.append('Recovery_rate')
    if(choice==[]):
        print('Invalid Input!')
    byCountry.xs(level='Country/Region',key=ct)[choice].plot(figsize=(12,6))
country=input('Enter country name:')
while(country not in np.array(df['Country/Region'])):
      country=input('Enter Valid country name:')
print('Note: Multiple entries mus be seperated with comma(,)\n1.Confirmed\n2.Deaths\n3.Recovered\n4.Mortality rate\n5.Recovery Rate\n:-->')
choice=list(map(int,input().split(',')))
display(country,choice)
df['Day_no']=df['Date'].apply(lambda x:int(str(pd.datetime.now()-x).split()[0])-82)
df['Day_no']=df['Day_no'].apply(lambda x:-(x))
df.head()
data=df.groupby(['Country/Region','Date','Day_no']).sum().reset_index('Day_no')
data.xs('India')['Day_no']
data.head()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#India=data.xs('India')[data.xs('India')['Confirmed']>0]
X_poly=PolynomialFeatures(degree=6).fit_transform(np.array(data.xs('India')['Day_no']).reshape(-1,1))
lm=LinearRegression()
lm.fit(X_poly,data.xs('India')['Confirmed'])
pred=lm.predict(X_poly)
plt.scatter(data.xs('India')['Confirmed'],pred)
sns.distplot(data.xs('India')['Confirmed']-pred,bins=30)
from sklearn.metrics import mean_squared_error,r2_score
rmse = np.sqrt(mean_squared_error(data.xs('India')['Confirmed'],pred))
r2 = r2_score(data.xs('India')['Confirmed'],pred)
print(rmse)
print(r2)
n=int(input())
n_poly=PolynomialFeatures(6).fit_transform(np.array([n]).reshape(-1,1))
print('No of Confimed Cases(Predicted) :',int(lm.predict(n_poly)[0]))
if(n<78):
    print(data.xs('India')[data.xs('India')['Day_no']==n]['Confirmed'])
def predictor(country,day):
    X_poly=PolynomialFeatures(6).fit_transform(np.array(data.xs(country)['Day_no']).reshape(-1,1))
    n_poly=PolynomialFeatures(6).fit_transform(np.array([day]).reshape(-1,1))
    lm=LinearRegression()
    lm.fit(X_poly,data.xs(country)['Confirmed'])
    pred=lm.predict(X_poly)
    
    
    sns.set_style('darkgrid')
    fig=plt.figure()
    
    ax1=fig.add_axes([0.1,0.1,0.9,0.9])
    ax1.set_title('Prediction Function')
    ax1.set_xlabel('Predicted Confirmed')
    ax1.set_ylabel('Actaul Confirmed')
    ax1.scatter(data.xs(country)['Confirmed'],pred,color='red')
    
    ax2=fig.add_axes([0.7,0.7,0.3,0.3])
    ax2.set_title('Error Distribution')
    ax2.set_xlabel('Actual-Predicted')
    ax2.set_ylabel('Error rate')
    ax2.hist(data.xs(country)['Confirmed']-pred,color='green',bins=30)
    
    
    rmse = np.sqrt(mean_squared_error(data.xs(country)['Confirmed'],pred))
    r2 = r2_score(data.xs(country)['Confirmed'],pred)
    print('Root Mead Square Evaluation:',rmse)
    print('R2 score of designed model:',r2)
    
    print('Prediction:',int(lm.predict(n_poly)[0]))
    if(day<78):
        print(data.xs(country)[data.xs(country)['Day_no']==day]['Confirmed'])
    
    
    
country=input()
while(country not in np.array(df['Country/Region'])):
      country=input()
n=int(input('Enter the Day Number:'))
predictor(country,n)

