import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset1 = pd.read_csv('../input/covid19-corona-virus-india-dataset/complete.csv')
df = pd.DataFrame(data = dataset1)
cases_by_date = df.groupby(['Date'])['Cured/Discharged/Migrated','Death','Total Confirmed cases'].sum()
x_date = df.groupby(['Date'])
x_date = (pd.DataFrame(x_date)).iloc[:,0].values
cases_by_date = pd.DataFrame(cases_by_date)
conf_cases_india = cases_by_date.iloc[:,-1].values
recv_cases_india = cases_by_date.iloc[:,-3].values
death_cases_india = cases_by_date.iloc[:,-2].values
daily_conf_cases_india=[1,]
daily_recv_cases_india=[0,]
daily_death_cases_india=[0,]
for i in range(1,len(conf_cases_india)):
  daily_conf_cases_india.append(conf_cases_india[i] - conf_cases_india[i-1])
  daily_recv_cases_india.append(recv_cases_india[i] - recv_cases_india[i-1])
  daily_death_cases_india.append(death_cases_india[i] - death_cases_india[i-1])

from sklearn.preprocessing import LabelEncoder
le_india = LabelEncoder()
x_date = le_india.fit_transform(x_date) + 1
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import  LinearRegression
poly_reg_conf_india = PolynomialFeatures(degree = 6)
X_poly_conf_india = poly_reg_conf_india.fit_transform(x_date.reshape(-1,1))
regressor_conf_india = LinearRegression()
regressor_conf_india.fit(X_poly_conf_india, conf_cases_india)
poly_reg_recv_india = PolynomialFeatures(degree = 9)
X_poly_recv_india = poly_reg_recv_india.fit_transform(x_date.reshape(-1,1))
regressor_recv_india = LinearRegression()
regressor_recv_india.fit(X_poly_recv_india, recv_cases_india)
poly_reg_d_india = PolynomialFeatures(degree = 6)
X_poly_d_india = poly_reg_d_india.fit_transform(x_date.reshape(-1,1))
regressor_d_india = LinearRegression()
regressor_d_india.fit(X_poly_d_india, death_cases_india)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(le_india.inverse_transform(x_date - 1), conf_cases_india, color = 'orange',label = 'Confirmed Cases')
ax.plot(le_india.inverse_transform(x_date - 1), recv_cases_india, color = 'blue',label = 'Recovered Cases')
ax.plot(le_india.inverse_transform(x_date - 1), death_cases_india, color = 'grey',label = 'Casualities')

#ax.bar(x_date, conf_cases_india, color = 'orange',label = 'Original Value')
plt.xlabel('Dates')
plt.ylabel('Count')
ax.set_xticks(range(0, len(x_date), 4))
#ax.set_yticks(range(0, conf_cases_india[len(conf_cases_india)-1], 3000))
plt.title('Confirmed Cases, Recovered cases, Casualities till date in India')
plt.xticks(rotation = 'vertical')
ax.legend(loc='best')
plt.show()
fig = plt.figure()
ax = fig.add_axes([0,0,2,1])
ax.plot(le_india.inverse_transform(x_date - 1), daily_conf_cases_india, color = 'red',label = 'Daily Confirmed Cases')
ax.plot(le_india.inverse_transform(x_date - 1), daily_recv_cases_india, color = 'blue',label = 'Daily RecoveredCases')
ax.plot(le_india.inverse_transform(x_date - 1), daily_death_cases_india, color = 'black',label = 'Daily Death Cases')
#ax.bar(x_date, conf_cases_india, color = 'orange',label = 'Original Value')
plt.xlabel('Dates')
plt.ylabel('Count')
ax.set_xticks(range(0, len(x_date), 4))
#ax.set_yticks(range(0, conf_cases_india[len(conf_cases_india)-1], 3000))
plt.title('Confirmed Cases, Recovered cases, Casualities Daily India')
plt.xticks(rotation = 'vertical')
ax.legend(loc='best')
plt.show()
fig = plt.figure()
ax = fig.add_axes([0,0,1.5,0.7])
ax.plot(le_india.inverse_transform(x_date - 1), regressor_conf_india.predict(X_poly_conf_india),color = 'blue',label = 'Predicted Value')
ax.bar(x_date, conf_cases_india, color = 'red',label = 'Original Value',log=True)
plt.xlabel('Dates')
plt.ylabel('Count')
ax.set_xticks(range(0, len(x_date), 4))
#ax.set_yticks(range(0, conf_cases_india[len(conf_cases_india)-1], 3000))
plt.title('Confirmed Cases till date in India')
plt.xticks(rotation = 'vertical')
ax.legend(loc='best')
plt.show()
fig = plt.figure()
ax = fig.add_axes([0,0,1.5,0.7])
ax.plot(le_india.inverse_transform(x_date - 1), regressor_recv_india.predict(X_poly_recv_india),color = 'red',label = 'Predicted Value')
ax.bar(x_date, recv_cases_india, color = 'orange',label = 'Original Value',log=True)
plt.xlabel('Dates')
plt.ylabel('Count')
ax.set_xticks(range(0, len(x_date), 4))
#ax.set_yticks(range(0, recv_cases_india[len(recv_cases_india)-1], ))
plt.title('Confirmed Cases till date in India')
plt.xticks(rotation = 'vertical')
plt.xlim(x_date[30],x_date[len(x_date)-1])
ax.legend(loc='best')
plt.show()
fig = plt.figure()
ax = fig.add_axes([0,0,1.5,0.7])
ax.plot(le_india.inverse_transform(x_date - 1), regressor_d_india.predict(X_poly_d_india),color = 'black',label = 'Predicted Value')
ax.bar(x_date, death_cases_india, color = 'grey',label = 'Original Value',log = True)
plt.xlabel('Dates')
plt.ylabel('Count')
plt.xlim(x_date[45],x_date[len(x_date)-1])
ax.set_xticks(range(x_date[50],x_date[len(x_date)-1], 4))
#ax.set_yticks(range(0, death_cases_india[len(death_cases_india)-1], 100))
plt.title('Confirmed Cases till date in India')
plt.xticks(rotation = 'vertical')
ax.legend(loc='best')
ax.bar
plt.show()
print('Total Confirmed cases till tomorrow: ', regressor_conf_india.predict(poly_reg_conf_india.transform([[len(x_date)+ 1]])))
print('Total Recovered cases till tomorrow: ', regressor_recv_india.predict(poly_reg_recv_india.transform([[len(x_date)+ 1]])))
print('Total Casualities till tomorrow: ', regressor_d_india.predict(poly_reg_d_india.transform([[len(x_date)+ 1]])))
labels = 'Active Cases', 'Cured Cases', 'Casualties'
sections = [conf_cases_india[len(conf_cases_india)-1] - recv_cases_india[len(recv_cases_india)-1] -  death_cases_india[len(death_cases_india)-1], recv_cases_india[len(recv_cases_india)-1], death_cases_india[len(death_cases_india)-1]]
colors = ['c', 'g', 'y']
plt.pie(sections, labels=labels, colors=colors,
        startangle=90,
        explode = (0, 0.1, 0.05),
        autopct = '%1.2f%%')

plt.axis('equal') # Try commenting this out.
plt.title('Covid-19 India Analysis')
plt.legend(loc='upper left',fontsize=8)
plt.show()