import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/elsect_summary.csv')
df.dropna(axis=0, subset=['ENROLL'],inplace=True)
df.head()
X = np.array(df['TOTAL_EXPENDITURE'])
Y = np.array(df['CAPITAL_OUTLAY_EXPENDITURE'])
plt.scatter(X,Y)
X = np.vstack([np.ones(len(X)),X]).T
w_ml = np.linalg.solve(X.T.dot(X),X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
plt.scatter(X[:,1],Y)
plt.scatter(X[:,1],Yhat_ml)
def r2(Y,yhat):
    d1 = Y-yhat
    d2 = Y-Y.mean()
    r2 = 1 - d1.dot(d1)/d2.dot(d2)
    return r2
r_squared_list = []
for i in range(0,20):
    l2x = 10**i
    w_mapx = np.linalg.solve(l2x*np.eye(2) + X.T.dot(X),X.T.dot(Y))
    Yhat_mapx = X.dot(w_mapx)
    r_squared = (r2(Y,Yhat_mapx))
    r_squared_list.append(r_squared)
plt.plot(r_squared_list)
l2_16 = 10**16
w_map_16 = np.linalg.solve(l2_16*np.eye(2) + X.T.dot(X),X.T.dot(Y))
Yhat_map_16 = X.dot(w_map_16)

l2_17 = 10**17
w_map_17= np.linalg.solve(l2_17*np.eye(2) + X.T.dot(X),X.T.dot(Y))
Yhat_map_17 = X.dot(w_map_17)
plt.scatter(X[:,1],Y,marker=".")
plt.scatter(X[:,1],Yhat_ml,label='l2=0',marker="1")
plt.scatter(X[:,1],Yhat_map_16,label='l2=10**16',marker="2",color='purple')
plt.scatter(X[:,1],Yhat_map_17,label='l2=10**17',marker='3',color = 'red')
plt.legend()
plt.xlabel('Total Expenditure')
plt.ylabel('Capital Outlay Expenditure')
plt.title('Regression Analysis with L2 Regularization')
print('The linear equation for gamma = 10**16: y =',round(w_map_16[1],2),'x',round(w_map_16[0]),2)
print('The linear equation for the standard linear regression: y =',round(w_ml[1],2),'x',round(w_ml[0],2))
