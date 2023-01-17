import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
frame=pd.read_csv('../input/3-Airplane_Crashes_Since_1908.txt',sep=',')
matplotlib.rcParams['figure.figsize'] = (12.0, 8.0)

operator = frame[['Operator','Fatalities']].groupby('Operator').agg(['sum','count'])

fig_ops,(ax1,ax2)=plt.subplots(1,2)
accidents = operator['Fatalities','count'].sort_values(ascending=False)
accidents.head(10).plot(kind='bar',title='Accidents by Operator',ax=ax1,rot=90)
fatalities = operator['Fatalities','sum'].sort_values(ascending=False)
fatalities.head(10).plot(kind='bar',title='Fatalities by Operator',ax=ax2,rot=90)

operator.head()
operator.dropna(inplace=True)
X = operator['Fatalities','count']
Y = operator['Fatalities','sum']
model = LinearRegression()
model.fit(X.values.reshape(-1,1),Y.values.reshape(-1,1))
m = model.coef_[0][0]
c = model.intercept_[0]

fig_fit,axd=plt.subplots()
axd.scatter(X,Y,label='Operators')
axd.set_title('Linear Model: Predicting Fatalities given Accidents')
axd.plot(X,model.predict(X.values.reshape(-1,1)),label='Linear Fit: y = %2.2fx %2.2f' % (m,c))
axd.grid(True)
axd.set_xlabel('Accidents')
axd.set_ylabel('Fatalities')
axd.legend(loc=2)
types = frame[['Type','Fatalities']].groupby('Type').agg(['sum','count'])

fig_type,(ax1,ax2)=plt.subplots(1,2)
acctype = types['Fatalities','count'].sort_values(ascending=False)
acctype.head(10).plot(kind='bar',title='Accidents by Type',grid=True,ax=ax1,rot=90)
fataltype = types['Fatalities','sum'].sort_values(ascending=False)
fataltype.head(10).plot(kind='bar',title='Fatalities by Type',grid=True,ax=ax2,rot=90)
frame['Year'] = frame['Date'].apply(lambda x: int(str(x)[-4:]))
yearly = frame[['Year','Fatalities']].groupby('Year').agg(['sum','count'])

fig_yearly,(axy1,axy2)=plt.subplots(2,1,figsize=(15,10))
yearly['Fatalities','sum'].plot(kind='bar',title='Fatalities by Year',grid=True,ax=axy1,rot=90)
yearly['Fatalities','count'].plot(kind='bar',title='Accidents by Year',grid=True,ax=axy2,rot=90)
plt.tight_layout()
#Just having a look at the Accident and Fatality trend by specific operator.
#The accident index is sorted from highest to lowest and can be used to select some of the more
#interesting operators.

#KLM for example have not had an accident since the 60's!

interestingOps = accidents.index.values.tolist()[0:5]
optrend = frame[['Operator','Year','Fatalities']].groupby(['Operator','Year']).agg(['sum','count'])
ops = optrend['Fatalities'].reset_index()
fig,axtrend = plt.subplots(2,1)
for op in interestingOps:
    ops[ops['Operator']==op].plot(x='Year',y='sum',ax=axtrend[0],grid=True,linewidth=2)
    ops[ops['Operator']==op].plot(x='Year',y='count',ax=axtrend[1],grid=True,linewidth=2)

axtrend[0].set_title('Fatality Trend by Operator')
axtrend[1].set_title('Accident Trend by Operator')
linesF, labelsF = axtrend[0].get_legend_handles_labels()
linesA, labelsA = axtrend[1].get_legend_handles_labels()
axtrend[0].legend(linesF,interestingOps)
axtrend[1].legend(linesA,interestingOps)
plt.tight_layout()
#Splitting out the country from the location to see if we can find some interesting
#statistics about where the most crashes have taken place.
s = frame['Location'].str[0:].str.split(',', expand=True)
country = pd.concat([s[3][~pd.isnull(s[3])],s[2][~pd.isnull(s[2])],s[1][~pd.isnull(s[1])]])
country.rename('Country',inplace=True)
result = frame.join(country)
result['Country'] = result['Country'].str.strip()

fatalcountries = result[result['Operator'] == 'Aeroflot'][['Operator','Country','Fatalities']].groupby(['Country']).agg('sum')
fatalcountries['Fatalities'].reset_index().sort_values(by='Fatalities',ascending=False)
fatalcountries.plot(kind='bar',title='Aeroflot Fatalities by Country')

#One can add the operator to the groupby and then pivot to create fatalities by country colour
#coded by operator... 
#a = a.pivot(index='Country', columns='Operator', values='Fatalities')

