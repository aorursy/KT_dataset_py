import pandas as pd
import numpy as np
from scipy import stats
from scipy import stats, special
from sklearn import model_selection, metrics, linear_model, datasets, feature_selection

import matplotlib.pyplot as plt
%pylab inline
pylab.rcParams['figure.figsize'] = (12.0, 10.0)
tt = pd.read_csv('../input/Travel_Times 16.csv',index_col=0,parse_dates=True)
tt[tt['Destination Movement ID']==3582]
ttd = pd.read_csv('../input/Travel_Times_Daily 16.csv',index_col=0,parse_dates=True)
ttd.sort_index(ascending=True)
#90 rows x 23 columns
ttda = pd.DataFrame(ttd)  
ttda['Above Mean'] = ttda['Daily Mean Travel Time (Seconds)'] > 924 
ttda1 = ttda.loc[:,('Daily Mean Travel Time (Seconds)','Above Mean')]
scatter(ttda1['Daily Mean Travel Time (Seconds)'],ttda1['Above Mean']);
ttda1['Daily Mean Travel Time (Seconds)'].values.reshape(-1,1)
logreg = linear_model.LogisticRegression(solver='newton-cg')
#X = cr['Hours Researched'].reset_index().values
#Create model
X = ttda1['Daily Mean Travel Time (Seconds)'].values.reshape(-1,1)
Y = ttda1['Above Mean']
#Fit data to model
logreg.fit(X,Y)
travel_time_slope = logreg.coef_[0][0]
logreg.coef_[0][0]
#Note positive correlation and rate
np.exp(logreg.coef_[0][0] - 1)
#Rate of value increase for one additional hour researched
travel_time_intercept = logreg.intercept_[0]
logreg.intercept_[0]
a = logreg.coef_[0][0]
b = logreg.intercept_[0]
x = x = np.arange(550,1447,0.01)
plt.plot(x,special.expit(a*x+b))
plt.grid()
plt.xlim(500,1700)
Daily_Average = ttda1['Daily Mean Travel Time (Seconds)']
pr = pd.DataFrame({'Daily Average':Daily_Average,'Pr Above':[special.expit(a*x+b) for x in Daily_Average]})
pr.sort_index(ascending=True)
s16 = scatter(pr['Daily Average'],pr['Pr Above'])
logreg.predict([[921]])[0]
logreg.predict([[922]])[0]
tt17 = pd.read_csv('../input/Travel_Times 17.csv',index_col=0,parse_dates=True)
tt17.head()
ttd17 = pd.read_csv('/Users/GLP/Desktop/DataScience/UberMovement/Travel_Times_Daily 17.csv',index_col=0,parse_dates=True)
ttd17.sort_index(ascending=True)
#90 Rows, 22 Rows
Daily_Average = ttd17['Daily Mean Travel Time (Seconds)']
pr17 = pd.DataFrame({'Daily Average':Daily_Average,'Pr Above':[special.expit(a*x+b) for x in Daily_Average]})
pr17.sort_index(ascending=True).head()
a = logreg.coef_[0][0]
b = logreg.intercept_[0]
x = arange(600,1550)
plt.plot(x,special.expit(a*x+b))
plt.grid()
plt.xlim(500,1400);
scatter(pr17['Daily Average'],pr17['Pr Above']);
a = logreg.coef_[0][0]
b = logreg.intercept_[0]
x = np.arange(550,1447,0.01)
plt.plot(x,special.expit(a*x+b),color='red')
plt.grid()
plt.xlim(729,1655);
s17 = scatter(pr17['Daily Average'],pr17['Pr Above'])
logreg.predict([[921]])[0]
logreg.predict([[922]])[0]
scatter(pr['Daily Average'],pr['Pr Above']);
scatter(pr17['Daily Average'],pr17['Pr Above']);
tt17 = pd.read_csv('../input/Travel_Times 17.csv',index_col=0,parse_dates=True)
tt17[tt17['Destination Movement ID']==3582]
ttdwd = ttd[ttd.index.weekday<5]
ttdwd.sort_index(ascending=True)
ttdw = pd.read_csv('../input/Travel_Times_time_of_day.csv')
ttdw
ttdw.loc[1:1,:]
ttdwd['Above AM Mean'] = ttdwd['AM Mean Travel Time (Seconds)'] > 1099
ttdwd1 = ttdwd.loc[:,('AM Mean Travel Time (Seconds)','Above AM Mean')]
ttdwd1.head()
a = logreg.coef_[0][0]
b = logreg.intercept_[0]
x = np.arange(550,1447,0.01)
plt.plot(x,special.expit(a*x+b),color='red')
plt.grid()
plt.xlim(729,1655);
scatter(ttdwd1['AM Mean Travel Time (Seconds)'],ttdwd1['Above AM Mean']);
ttdwd1['AM Mean Travel Time (Seconds)'].values.reshape(-1,1)
#ttdwd1.dropna()
ttda.sort_index(ascending=True)
ttda[ttda.index.weekday<5]
ttd2 = pd.read_csv('../input/Travel_Times_Daily 16.csv',index_col=0,parse_dates=True)
ttda2 = pd.DataFrame(ttd2)  
ttda2['Above Mean'] = ttda2['Daily Mean Travel Time (Seconds)'] > 924 
ttda2.head()
#ttda2['Date'] = pd.to_datetime(ttda2['Date'])
#ttda2['Day of Week'] = ttda2['Date'].dt.weekday_name
#ttda2.head()

ttda2['Weekend'] = ((pd.DatetimeIndex(ttda2.index).dayofweek) // 5 == 1).astype(float)
ttda2.sort_index(ascending=True)
ttda2[ttda2.index.weekday<5].sort_index(ascending=True).head()
ttda3 = ttda2.loc[:,('Daily Mean Travel Time (Seconds)','Above Mean','Weekend')]
ttda3.sort_index(ascending=True)
ttda3.corr()
ttda3['Daily Mean Travel Time (Seconds)'].values.reshape(-1,1)
ttda3['Weekend'].values.reshape(-1,1)
ttda3['Above Mean'].values.reshape(-1,1)
lgrg = linear_model.LogisticRegression(solver='newton-cg')
X = ttda3[['Daily Mean Travel Time (Seconds)','Weekend']]
#Two Variables on the X axis
Y = ttda3['Above Mean']
#One variable on the Y axis
lgrg.fit(X,Y)
TT_Wknd_coeff17 = lgrg.coef_[0][0] #First row, first column of matrix
TT_Wknd_coeff17
#X-Value
#Positive movement along the X-scale
TT_Wknd_coeff17 = lgrg.coef_[0][1]
TT_Wknd_coeff17
TT_Wknd__intercept17 = lgrg.intercept_[0]
TT_Wknd__intercept17
lgrg.intercept_
b1,b2 = lgrg.coef_[0]
b0 = lgrg.intercept_[0]
print(b1)
print(b2)
print(b0)
def TT_model(TT,Wknd): #building model
    return special.expit(TT*b1+Wknd*b2+b0)
#Takes trip time and Weekend values
#Returns value along sigmoid function (x*b1 + y*b2 + e)
TT_model(922,1)
lgrg.predict_proba([[922,1]])
lgrg.predict_proba(ttda3[["Daily Mean Travel Time (Seconds)","Weekend"]])
import mpl_toolkits.mplot3d as m3d
tt = np.arange(589,1447,1)
wk = np.arange(0,1,1)
xx, yy = np.meshgrid(tt, wk)
Z = TT_model(xx,yy)
fig3d = m3d.Axes3D(plt.figure())
fig3d.plot_wireframe(xx, yy, Z, rstride=10, cstride=10);
fig3d.set_xlabel('Daily Mean Travel Time (Seconds)')
fig3d.set_ylabel('Weekend')
fig3d.set_zlabel('Probability of Above Mean Trip')
fig3d.view_init(25, 140)
plt.show();
fig3d = m3d.Axes3D(plt.figure())
fig3d.plot_wireframe(xx, yy, Z, rstride=10, cstride=10);
fig3d.set_xlabel('Daily Mean Travel Time (Seconds)')
fig3d.set_ylabel('Loan Amount ($100s)')
fig3d.set_zlabel('Probability of Above Mean Trip')
fig3d.view_init(30, 80)
plt.show();
lgrg.predict([[922,0]])[0]
lgrg.predict([[922,1]])[0]
#Training data shows that 
TT_model(922,0)
#Predict probability of if trip is above Mean Travel Time
TT_model(922,1)
#Shows that if it is the weekend, there is a lower probability that a trip will be over time