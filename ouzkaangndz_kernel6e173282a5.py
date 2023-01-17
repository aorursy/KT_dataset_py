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
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('../input/indonesia-coronavirus-cases/confirmed_acc.csv')
df = df.loc[:,['date','cases']]
FMT = '%m/%d/%Y'
date = df['date']
df['date'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("3/1/2020", FMT)).days  )
df
#Data tambahan sejak 18 Maret 2020
df = df.append({'date' : 17 , 'cases' : 227} , ignore_index=True) #18 Maret
df= df.append({'date' : 18, 'cases' : 309}, ignore_index=True) #19 Maret
df= df.append({'date' : 19, 'cases' : 369}, ignore_index=True) #20 Maret
df=df.append({'date' : 20, 'cases' : 450}, ignore_index=True) #21 Maret
df=df.append({'date' : 21, 'cases' : 514}, ignore_index=True) #22 Maret
df=df.append({'date' : 22, 'cases' : 579}, ignore_index=True) #23 Maret
df=df.append({'date' : 23, 'cases' : 686}, ignore_index=True) #24 Maret
df=df.append({'date' : 24, 'cases' : 790}, ignore_index=True) #25 Maret
df=df.append({'date' : 25, 'cases' : 893}, ignore_index=True) #26 Maret
df=df.append({'date' : 26, 'cases' : 1046}, ignore_index=True) #27 Maret
df=df.append({'date' : 27, 'cases' : 1155}, ignore_index=True) #28 Maret
df=df.append({'date' : 28, 'cases' : 1285}, ignore_index=True) #29 Maret
df=df.append({'date' : 29, 'cases' : 1414}, ignore_index=True) #30 Maret
df=df.append({'date' : 30, 'cases' : 1528}, ignore_index=True) #31 Maret
df=df.append({'date' : 31, 'cases' : 1677}, ignore_index=True) #1 April
df=df.append({'date' : 32, 'cases' : 1790}, ignore_index=True) #2 April
df=df.append({'date' : 33, 'cases' : 1986}, ignore_index=True) #3 April
df=df.append({'date' : 34, 'cases' : 2092}, ignore_index=True) #4 April
df=df.append({'date' : 35, 'cases' : 2273}, ignore_index=True) #5 April
df

def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))
x = list(df.iloc[39:,0])
y = list(df.iloc[39:,1])
fit = curve_fit(logistic_model,x,y)
A,B=fit
#nilai a,b,
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

# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,a,b,c) for i in x+pred_x], label="Logistic model" )

plt.legend()
plt.xlabel("Days since 1 March 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))
plt.show()
y_pred_logistic = [logistic_model(i,a,b,c) for i in x]
p=mean_squared_error(y,y_pred_logistic)

s1=(np.subtract(y,y_pred_logistic)**2).sum()
s2=(np.subtract(y,np.mean(y))**2).sum()
r=1-s1/s2
print("R^2 adalah {}".format(r))
print("Mean square errornya adalah {}".format(p))
from datetime import timedelta, date
from datetime import datetime  
from datetime import timedelta 

start_date = "01/03/20"

date_1 = datetime.strptime(start_date, "%d/%m/%y")

end_date = date_1 + timedelta(days=sol)

x=end_date.strftime("%d %B %Y")
print("Jumlah kasus maksimal di indonesia menurut prediksi adalah {:f}".format(A[2]+errors[2])) #Penambahan dengan error
print("Puncak wabah adalah {:.0f} hari setelah 1 Maret 2020 atau {}". format(sol,x))