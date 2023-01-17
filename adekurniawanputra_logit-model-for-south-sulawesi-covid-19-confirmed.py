# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
%matplotlib inline
import time
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
df=pd.read_csv('../input/comulative-confirmed-case-in-indonesia/Comulative Confirmed Case in Indonesia.csv',usecols=['Date','Sulsel'])
df['Kasus']=df['Sulsel'].diff()
df.dropna()
xs = df['Date']
ys = df['Kasus']
dfx = pd.DataFrame({'x': xs, 'y': ys})
nn_start='6/1/2020'
fig = go.Figure()

fig.add_scattergl(x=xs, y=dfx.y, name="Before New Normal", line={'color': 'black'})

fig.add_scattergl(x=dfx.x.where(dfx.x>=nn_start), y=ys, name="After New Normal", line={'color': 'red'})
fig.update_layout(title={'text':"Kasus Harian Sulsel",
                         'y':0.9,
                         'x':0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'})
iplot(fig)
df=df.rename(columns={'Sulsel':'Komulatif','Date':'Tanggal'})
df2 = df.loc[:,['Tanggal','Komulatif']]
FMT = '%m/%d/%Y'
date = df2['Tanggal']
df2['Tanggal'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("3/20/2020", FMT)).days)
df2[df2['Tanggal'] == 0] 
def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))
x = list(df2.iloc[18:90,0])
y = list(df2.iloc[18:90,1])
fit = curve_fit(logistic_model,x,y)
A,B=fit
A
errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]
errors
a=A[0]+errors[0]
b=A[1]+errors[1]
c=A[2]+errors[2]
sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))
sol
pred_x = list(range(max(x),sol))
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)# Real data
plt.scatter(x,y,label="Real data",color="red")

plt.plot(x+pred_x, [logistic_model(i,a,b,c) for i in x+pred_x], label="Logistic model" )

plt.legend()
plt.xlabel("Days since 20 March 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))
plt.show()
y_pred_logistic = [logistic_model(i,a,b,c) for i in x]

s1=(np.subtract(y,y_pred_logistic)**2).sum()
s2=(np.subtract(y,np.mean(y))**2).sum()
r=1-s1/s2
print("R^2 adalah {}".format(r))
from datetime import timedelta, date
from datetime import datetime  
from datetime import timedelta 

start_date = "20/03/20"

date_1 = datetime.strptime(start_date, "%d/%m/%y")

end_date = date_1 + timedelta(days=sol)

x=end_date.strftime("%d %B %Y")


print("Jumlah kasus maksimal di Sulawesi Selatan menurut prediksi adalah {:.0f}".format(A[2]+errors[2])) #Penambahan dengan error
print("Puncak wabah adalah {:.0f} hari setelah 20 Maret 2020 atau {}". format(sol,x))
xx = list(df2.iloc[18:128,0])
yy = list(df2.iloc[18:128,1])
fit = curve_fit(logistic_model,xx,yy)
yy
AA,BB=fit
AA
errors1 = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]
errors1
aa=AA[0]+errors1[0]
bb=AA[1]+errors1[1]
cc=AA[2]+errors1[2]
sol1 = int(fsolve(lambda xx : logistic_model(xx,aa,bb,cc) - int(cc),bb))
sol1
pred_xx = list(range(max(xx),sol1))
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
plt.scatter(xx,yy,label="Real data",color="green")

plt.plot(xx+pred_xx, [logistic_model(i,aa,bb,cc) for i in xx+pred_xx], label="Logistic model" )

plt.legend()
plt.xlabel("Days since 20 March 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(yy)*0.9,cc*1.1))
plt.show()
yy_pred_logistic = [logistic_model(i,aa,bb,cc) for i in xx]

ss1=(np.subtract(yy,yy_pred_logistic)**2).sum()
ss2=(np.subtract(yy,np.mean(yy))**2).sum()
rr=1-ss1/ss2
print("R^2 adalah {}".format(rr))
from datetime import timedelta, date
from datetime import datetime  
from datetime import timedelta 

start_date = "20/03/20"

date_1 = datetime.strptime(start_date, "%d/%m/%y")

end_date = date_1 + timedelta(days=sol1)

xxx=end_date.strftime("%d %B %Y")


print("Jumlah kasus maksimal di indonesia menurut prediksi adalah {:.0f}".format(AA[2]+errors1[2])) #Penambahan dengan error
print("Puncak wabah adalah {:.0f} hari setelah 20 Maret 2020 atau {}". format(sol1,xxx))

