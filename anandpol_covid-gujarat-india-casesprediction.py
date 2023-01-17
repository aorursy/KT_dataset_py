import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/covid19-corona-virus-india-dataset/complete.csv')
is_true = dataset['Name of State / UT'] == 'Gujarat'
dataset = dataset[is_true]
X = dataset.iloc[:, 0].values
confcases = dataset.iloc[:, -1].values
recvcases = dataset.iloc[:, -5].values
death = dataset.iloc[:, -2].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X = le.fit_transform(X) + 1
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg_conf = PolynomialFeatures(degree = 7)
X_poly_conf = poly_reg_conf.fit_transform(X.reshape(-1,1))
regressor_conf = LinearRegression()
regressor_conf.fit(X_poly_conf, confcases)
poly_reg_recv = PolynomialFeatures(degree =8)
X_poly_recv = poly_reg_recv.fit_transform(X.reshape(-1,1))
regressor_recv = LinearRegression()
regressor_recv.fit(X_poly_recv, recvcases)
poly_reg_d = PolynomialFeatures(degree = 6)
X_poly_d = poly_reg_d.fit_transform(X.reshape(-1,1))
regressor_d = LinearRegression()
regressor_d.fit(X_poly_d, death)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.plot(le.inverse_transform(X - 1), confcases,color = 'orange',label = 'Confirmed Cases')
plt.plot(le.inverse_transform(X - 1), recvcases,color = 'blue',label = 'Recovered Cases')
plt.plot(le.inverse_transform(X - 1), death,color = 'grey',label = 'Casualities')

#ax.bar(x_date, conf_cases_india, color = 'orange',label = 'Original Value')
plt.xlabel('Dates')
plt.ylabel('Count')
ax.set_xticks(range(0, len(X), 4))
#ax.set_yticks(range(0, conf_cases_india[len(conf_cases_india)-1], 3000))
plt.title('Confirmed Cases, Recovered cases, Casualities till date in Gujarat-India')
plt.xticks(rotation = 'vertical')
ax.legend(loc='best')
plt.show()
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1.5,0.7])
ax.plot(le.inverse_transform(X - 1), regressor_conf.predict(X_poly_conf),color = 'blue',label = 'Predicted Value')
ax.bar(X, confcases, color = 'red',label = 'Original Value',log=True)
plt.xlabel('Dates')
plt.ylabel('Count')
#plt.ylim(0,6000)
#ax.set_yticks(range(0, confcases[len(confcases)-1], 2000))
ax.set_xticks(range(0, len(X), 3))
plt.title('Confirmed Cases till date in Gujarat-India')
plt.xticks(rotation = 'vertical')
ax.legend(loc='best')
plt.show()
fig = plt.figure()
ax = fig.add_axes([0,0,1.5,0.7])
ax.plot(le.inverse_transform(X-1), regressor_recv.predict(X_poly_recv),color = 'red',label = 'Predicted Value')
ax.bar(X-1, recvcases, color = 'orange',label = 'Original Value',log=True)
plt.xlabel('Dates')
plt.ylabel('Count')
ax.set_xticks(range(0, len(X), 3))
plt.title('Recovered Cases till date in Gujarat-India( LOGARITHMIC )')
plt.xticks(rotation = 'vertical')
ax.legend(loc='best')
plt.show()
fig = plt.figure()
ax = fig.add_axes([0,0,1.5,0.7])
ax.plot(le.inverse_transform(X - 1), regressor_d.predict(X_poly_d),color = 'black',label = 'Predicted Value')
plt.bar(X-1, death,color = 'grey',label = 'Original Value',log=True)
plt.xlabel('Dates')
plt.ylabel('Count')
ax.set_xticks(range(0, len(X), 3))
plt.title('Casualities till date in Gujarat-India')
plt.xticks(rotation = 'vertical')
ax.legend(loc='best')
plt.show()
from datetime import date
dt1 = date(2020,3,20)
dt2 = date(2020,5,15)
delta = dt2 - dt1
print(delta.days)

print('Total Confirmed cases till tomorrow: ', regressor_conf.predict(poly_reg_conf.transform([[delta.days + 1]])))
print('Total Recovered cases till tomorrow: ', regressor_recv.predict(poly_reg_recv.transform([[delta.days + 1]])))
print('Total Casualities till tomorrow: ', regressor_d.predict(poly_reg_d.transform([[delta.days + 1]])))

labels = 'Active Cases', 'Cured Cases', 'Casualties'
sections = [confcases[len(confcases)-1] - recvcases[len(recvcases)-1] - death[len(death)-1], recvcases[len(recvcases)-1], death[len(death)-1]]
colors = ['c', 'g', 'y']
plt.pie(sections, labels=labels, colors=colors,
        startangle=90,
        explode = (0, 0.1, 0.05),
        autopct = '%1.2f%%')

plt.axis('equal') # Try commenting this out.
plt.title('Covid-19 Gujarat Analysis')
plt.legend(loc='upper left',fontsize=8)
plt.show()