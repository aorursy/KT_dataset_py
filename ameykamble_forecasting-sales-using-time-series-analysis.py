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
import math
Data = pd.read_csv('../input/Time series Data.csv')
Data.head()
Data.count()
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
print("minimum date is ",Data.Date.min()," and maximum date is ",Data.Date.max())
y1 = Data[Data['Product name'] == 'NAN PRO 2 REFILL  400GM']
y2 = Data[Data['Product name'] == 'NAN PRO 1 REFILL  400GM']
y3 = Data[Data['Product name'] == 'ENO FRUITSALT LEMON 5GM']
y4 = Data[Data['Product name'] == 'ECOSPRIN 75MG 14TAB']
y5 = Data[Data['Product name'] == 'NAN PRO 3 REFILL  400GM']
y6 = Data[Data['Product name'] == 'LACTOGEN STAGE 1 REFILL  400GM']
y7 = Data[Data['Product name'] == 'DUPHASTON 10TAB']
y8 = Data[Data['Product name'] == 'ENFAMIL A+ POWDER STAGE 2 400GM']
y9 = Data[Data['Product name'] == 'CROCIN ADVANCE TABLET']
y10 = Data[Data['Product name'] == 'LACTOGEN STAGE 2 REFILL  400GM']
y11 = Data[Data['Product name'] == 'ENFAMIL A+ POWDER STAGE 1 400GM']
y12 = Data[Data['Product name'] == 'DETTOL LIQUID 550ML']
y13 = Data[Data['Product name'] == 'ECOSPRIN AV 75MG CAPSULE']
y14 = Data[Data['Product name'] == 'EVION 400MG 10CAP']

y_1 = y1.Qty
y_2 = y2.Qty
y_3 = y3.Qty
y_4 = y4.Qty
y_5 = y5.Qty
y_6 = y6.Qty
y_7 = y7.Qty
y_8 = y8.Qty
y_9 = y9.Qty
y_10 = y10.Qty
y_11 = y11.Qty
y_12 = y12.Qty
y_13 = y13.Qty
y_14 = y14.Qty
y = [y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14]
for i in y:
    i.plot(figsize = (10,3))
    plt.show()
y1
y1 = Data[Data['Product name'] == 'NAN PRO 2 REFILL  400GM']
y1["Year"] = pd.to_datetime(y1["Date"]).dt.year
y1["month"] = pd.to_datetime(y1["Date"]).dt.month
y1["month_v1"] = (y1["month"] - y1["month"].mean())/y1["month"].std()

X = y1["month_v1"] ## X usually means our input variables (or independent variables)
y = y1["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
model.summary()

y1 = y1.append({'Date' : '2020-04-01' , 'Product name' : 'NAN PRO 2 REFILL 400GM'} , ignore_index=True)
y1 = y1.append({'Date' : '2020-05-01' , 'Product name' : 'NAN PRO 2 REFILL 400GM'} , ignore_index=True)
y1 = y1.append({'Date' : '2020-06-01' , 'Product name' : 'NAN PRO 2 REFILL 400GM'} , ignore_index=True)
y1["month_v2"] = pd.to_datetime(y1["Date"]).dt.month
y1["month_v3"] = (y1["month_v2"] - y1["month_v2"].mean())/y1['month_v2'].std()
y1['Pred_Qty'] = round(173.8333+28.2436*y1['month_v3'],0)
y1 = y1.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_1 = round(y1.Qty.mean() + (1.96*y1.Qty.std()/math.sqrt(len(y1.axes[0])-1)),0)
Lower_limit_1 = round(y1.Qty.mean() - (1.96*y1.Qty.std()/math.sqrt(len(y1.axes[0])-1)),0)
##############################################

y2 = Data[Data['Product name'] == 'NAN PRO 1 REFILL  400GM']
y2["Year"] = pd.to_datetime(y2["Date"]).dt.year
y2["month"] = pd.to_datetime(y2["Date"]).dt.month
y2["month_v1"] = (y2["month"] - y2["month"].mean())/y2["month"].std()
X = y2["month_v1"] ## X usually means our input variables (or independent variables)
y = y2["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y2 = y2.append({'Date' : '2020-04-01' , 'Product name' : 'NAN PRO 1 REFILL  400GM'} , ignore_index=True)
y2 = y2.append({'Date' : '2020-05-01' , 'Product name' : 'NAN PRO 1 REFILL  400GM'} , ignore_index=True)
y2 = y2.append({'Date' : '2020-06-01' , 'Product name' : 'NAN PRO 1 REFILL  400GM'} , ignore_index=True)
y2["month_v2"] = pd.to_datetime(y2["Date"]).dt.month
y2["month_v3"] = (y2["month_v2"] - y2["month_v2"].mean())/y2['month_v2'].std()
y2['Pred_Qty'] = round(114.667+26.1078*y2['month_v3'],0)
y2 = y2.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_2 = round(y2.Qty.mean() + (1.96*y2.Qty.std()/math.sqrt(len(y2.axes[0])-1)),0)
Lower_limit_2 = round(y2.Qty.mean() - (1.96*y2.Qty.std()/math.sqrt(len(y2.axes[0])-1)),0)

################################################

y3 = Data[Data['Product name'] == 'ENO FRUITSALT LEMON 5GM']
y3["Year"] = pd.to_datetime(y3["Date"]).dt.year
y3["month"] = pd.to_datetime(y3["Date"]).dt.month
y3["month_v1"] = (y3["month"] - y3["month"].mean())/y3["month"].std()
X = y3["month_v1"] ## X usually means our input variables (or independent variables)
y = y3["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y3 = y3.append({'Date' : '2020-04-01' , 'Product name' : 'ENO FRUITSALT LEMON 5GM'} , ignore_index=True)
y3 = y3.append({'Date' : '2020-05-01' , 'Product name' : 'ENO FRUITSALT LEMON 5GM'} , ignore_index=True)
y3 = y3.append({'Date' : '2020-06-01' , 'Product name' : 'ENO FRUITSALT LEMON 5GM'} , ignore_index=True)
y3["month_v2"] = pd.to_datetime(y3["Date"]).dt.month
y3["month_v3"] = (y3["month_v2"] - y3["month_v2"].mean())/y3['month_v2'].std()
y3['Pred_Qty'] = round(97.667-10.45*y3['month_v3'],0)
y3 = y3.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_3 = round(y3.Qty.mean() + (1.96*y3.Qty.std()/math.sqrt(len(y3.axes[0])-1)),0)
Lower_limit_3 = round(y3.Qty.mean() - (1.96*y3.Qty.std()/math.sqrt(len(y3.axes[0])-1)),0)

####################################################

y4 = Data[Data['Product name'] == 'ECOSPRIN 75MG 14TAB']
y4["Year"] = pd.to_datetime(y4["Date"]).dt.year
y4["month"] = pd.to_datetime(y4["Date"]).dt.month
y4["month_v1"] = (y4["month"] - y4["month"].mean())/y4["month"].std()
X = y4["month_v1"] ## X usually means our input variables (or independent variables)
y = y4["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y4 = y4.append({'Date' : '2020-04-01' , 'Product name' : 'ECOSPRIN 75MG 14TAB'} , ignore_index=True)
y4 = y4.append({'Date' : '2020-05-01' , 'Product name' : 'ECOSPRIN 75MG 14TAB'} , ignore_index=True)
y4 = y4.append({'Date' : '2020-06-01' , 'Product name' : 'ECOSPRIN 75MG 14TAB'} , ignore_index=True)
y4["month_v2"] = pd.to_datetime(y4["Date"]).dt.month
y4["month_v3"] = (y4["month_v2"] - y4["month_v2"].mean())/y4['month_v2'].std()
y4['Pred_Qty'] = round(89.333-7.745*y4['month_v3'],0)
y4 = y4.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_4 = round(y4.Qty.mean() + (1.96*y4.Qty.std()/math.sqrt(len(y4.axes[0])-1)),0)
Lower_limit_4 = round(y4.Qty.mean() - (1.96*y4.Qty.std()/math.sqrt(len(y4.axes[0])-1)),0)

#######################################################

y5 = Data[Data['Product name'] == 'NAN PRO 3 REFILL  400GM']
y5["Year"] = pd.to_datetime(y5["Date"]).dt.year
y5["month"] = pd.to_datetime(y5["Date"]).dt.month
y5["month_v1"] = (y5["month"] - y5["month"].mean())/y5["month"].std()
X = y5["month_v1"] ## X usually means our input variables (or independent variables)
y = y5["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y5 = y5.append({'Date' : '2020-04-01' , 'Product name' : 'NAN PRO 3 REFILL  400GM'} , ignore_index=True)
y5 = y5.append({'Date' : '2020-05-01' , 'Product name' : 'NAN PRO 3 REFILL  400GM'} , ignore_index=True)
y5 = y5.append({'Date' : '2020-06-01' , 'Product name' : 'NAN PRO 3 REFILL  400GM'} , ignore_index=True)
y5["month_v2"] = pd.to_datetime(y5["Date"]).dt.month
y5["month_v3"] = (y5["month_v2"] - y5["month_v2"].mean())/y5['month_v2'].std()
y5['Pred_Qty'] = round(66.666+14.05*y5['month_v3'],0)
y5 = y5.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_5 = round(y5.Qty.mean() + (1.96*y5.Qty.std()/math.sqrt(len(y5.axes[0])-1)),0)
Lower_limit_5 = round(y5.Qty.mean() - (1.96*y5.Qty.std()/math.sqrt(len(y5.axes[0])-1)),0)

#######################################################

y6 = Data[Data['Product name'] == 'LACTOGEN STAGE 1 REFILL  400GM']
y6["Year"] = pd.to_datetime(y6["Date"]).dt.year
y6["month"] = pd.to_datetime(y6["Date"]).dt.month
y6["month_v1"] = (y6["month"] - y6["month"].mean())/y6["month"].std()
X = y6["month_v1"] ## X usually means our input variables (or independent variables)
y = y6["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y6 = y6.append({'Date' : '2020-04-01' , 'Product name' : 'LACTOGEN STAGE 1 REFILL  400GM'} , ignore_index=True)
y6 = y6.append({'Date' : '2020-05-01' , 'Product name' : 'LACTOGEN STAGE 1 REFILL  400GM'} , ignore_index=True)
y6 = y6.append({'Date' : '2020-06-01' , 'Product name' : 'LACTOGEN STAGE 1 REFILL  400GM'} , ignore_index=True)
y6["month_v2"] = pd.to_datetime(y6["Date"]).dt.month
y6["month_v3"] = (y6["month_v2"] - y6["month_v2"].mean())/y6['month_v2'].std()
y6['Pred_Qty'] = round(61.833+8.444*y6['month_v3'],0)
y6 = y6.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_6 = round(y6.Qty.mean() + (1.96*y6.Qty.std()/math.sqrt(len(y6.axes[0])-1)),0)
Lower_limit_6 = round(y6.Qty.mean() - (1.96*y6.Qty.std()/math.sqrt(len(y6.axes[0])-1)),0)

############################################################

y7 = Data[Data['Product name'] == 'DUPHASTON 10TAB']
y7["Year"] = pd.to_datetime(y7["Date"]).dt.year
y7["month"] = pd.to_datetime(y7["Date"]).dt.month
y7["month_v1"] = (y7["month"] - y7["month"].mean())/y7["month"].std()
X = y7["month_v1"] ## X usually means our input variables (or independent variables)
y = y7["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y7 = y7.append({'Date' : '2020-04-01' , 'Product name' : 'DUPHASTON 10TAB'} , ignore_index=True)
y7 = y7.append({'Date' : '2020-05-01' , 'Product name' : 'DUPHASTON 10TAB'} , ignore_index=True)
y7 = y7.append({'Date' : '2020-06-01' , 'Product name' : 'DUPHASTON 10TAB'} , ignore_index=True)
y7["month_v2"] = pd.to_datetime(y7["Date"]).dt.month
y7["month_v3"] = (y7["month_v2"] - y7["month_v2"].mean())/y7['month_v2'].std()
y7['Pred_Qty'] = round(58.666+3.91*y7['month_v3'],0)
y7 = y7.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_7 = round(y7.Qty.mean() + (1.96*y7.Qty.std()/math.sqrt(len(y7.axes[0])-1)),0)
Lower_limit_7 = round(y7.Qty.mean() - (1.96*y7.Qty.std()/math.sqrt(len(y7.axes[0])-1)),0)

###########################################################

y8 = Data[Data['Product name'] == 'ENFAMIL A+ POWDER STAGE 2 400GM']
y8["Year"] = pd.to_datetime(y8["Date"]).dt.year
y8["month"] = pd.to_datetime(y8["Date"]).dt.month
y8["month_v1"] = (y8["month"] - y8["month"].mean())/y8["month"].std()
X = y8["month_v1"] ## X usually means our input variables (or independent variables)
y = y8["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y8 = y8.append({'Date' : '2020-04-01' , 'Product name' : 'ENFAMIL A+ POWDER STAGE 2 400GM'} , ignore_index=True)
y8 = y8.append({'Date' : '2020-05-01' , 'Product name' : 'ENFAMIL A+ POWDER STAGE 2 400GM'} , ignore_index=True)
y8 = y8.append({'Date' : '2020-06-01' , 'Product name' : 'ENFAMIL A+ POWDER STAGE 2 400GM'} , ignore_index=True)
y8["month_v2"] = pd.to_datetime(y8["Date"]).dt.month
y8["month_v3"] = (y8["month_v2"] - y8["month_v2"].mean())/y8['month_v2'].std()
y8['Pred_Qty'] = round(58.333+1.996*y8['month_v3'],0)
y8 = y8.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_8 = round(y8.Qty.mean() + (1.96*y8.Qty.std()/math.sqrt(len(y8.axes[0])-1)),0)
Lower_limit_8 = round(y8.Qty.mean() - (1.96*y8.Qty.std()/math.sqrt(len(y8.axes[0])-1)),0)

######################################################

y9 = Data[Data['Product name'] == 'CROCIN ADVANCE TABLET']
y9["Year"] = pd.to_datetime(y9["Date"]).dt.year
y9["month"] = pd.to_datetime(y9["Date"]).dt.month
y9["month_v1"] = (y9["month"] - y9["month"].mean())/y9["month"].std()
X = y9["month_v1"] ## X usually means our input variables (or independent variables)
y = y9["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y9 = y9.append({'Date' : '2020-04-01' , 'Product name' : 'CROCIN ADVANCE TABLET'} , ignore_index=True)
y9 = y9.append({'Date' : '2020-05-01' , 'Product name' : 'CROCIN ADVANCE TABLET'} , ignore_index=True)
y9 = y9.append({'Date' : '2020-06-01' , 'Product name' : 'CROCIN ADVANCE TABLET'} , ignore_index=True)
y9["month_v2"] = pd.to_datetime(y9["Date"]).dt.month
y9["month_v3"] = (y9["month_v2"] - y9["month_v2"].mean())/y9['month_v2'].std()
y9['Pred_Qty'] = round(55.0+12.055*y9['month_v3'],0)
y9 = y9.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_9 = round(y9.Qty.mean() + (1.96*y9.Qty.std()/math.sqrt(len(y9.axes[0])-1)),0)
Lower_limit_9 = round(y9.Qty.mean() - (1.96*y9.Qty.std()/math.sqrt(len(y9.axes[0])-1)),0)

########################################################

y10 = Data[Data['Product name'] == 'LACTOGEN STAGE 2 REFILL  400GM']
y10["Year"] = pd.to_datetime(y10["Date"]).dt.year
y10["month"] = pd.to_datetime(y10["Date"]).dt.month
y10["month_v1"] = (y10["month"] - y10["month"].mean())/y10["month"].std()
X = y10["month_v1"] ## X usually means our input variables (or independent variables)
y = y10["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y10 = y10.append({'Date' : '2020-04-01' , 'Product name' : 'LACTOGEN STAGE 2 REFILL  400GM'} , ignore_index=True)
y10 = y10.append({'Date' : '2020-05-01' , 'Product name' : 'LACTOGEN STAGE 2 REFILL  400GM'} , ignore_index=True)
y10 = y10.append({'Date' : '2020-06-01' , 'Product name' : 'LACTOGEN STAGE 2 REFILL  400GM'} , ignore_index=True)
y10["month_v2"] = pd.to_datetime(y10["Date"]).dt.month
y10["month_v3"] = (y10["month_v2"] - y10["month_v2"].mean())/y10['month_v2'].std()
y10['Pred_Qty'] = round(54.666+3.7525*y10['month_v3'],0)
y10 = y10.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_10 = round(y10.Qty.mean() + (1.96*y10.Qty.std()/math.sqrt(len(y10.axes[0])-1)),0)
Lower_limit_10 = round(y10.Qty.mean() - (1.96*y10.Qty.std()/math.sqrt(len(y10.axes[0])-1)),0)

#######################################################

y11 = Data[Data['Product name'] == 'ENFAMIL A+ POWDER STAGE 1 400GM']
y11["Year"] = pd.to_datetime(y11["Date"]).dt.year
y11["month"] = pd.to_datetime(y11["Date"]).dt.month
y11["month_v1"] = (y11["month"] - y11["month"].mean())/y11["month"].std()
X = y11["month_v1"] ## X usually means our input variables (or independent variables)
y = y11["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y11 = y11.append({'Date' : '2020-04-01' , 'Product name' : 'ENFAMIL A+ POWDER STAGE 1 400GM'} , ignore_index=True)
y11 = y11.append({'Date' : '2020-05-01' , 'Product name' : 'ENFAMIL A+ POWDER STAGE 1 400GM'} , ignore_index=True)
y11 = y11.append({'Date' : '2020-06-01' , 'Product name' : 'ENFAMIL A+ POWDER STAGE 1 400GM'} , ignore_index=True)
y11["month_v2"] = pd.to_datetime(y11["Date"]).dt.month
y11["month_v3"] = (y11["month_v2"] - y11["month_v2"].mean())/y11['month_v2'].std()
y11['Pred_Qty'] = round(54.666+7.4252*y11['month_v3'],0)
y11 = y11.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_11 = round(y11.Qty.mean() + (1.96*y11.Qty.std()/math.sqrt(len(y11.axes[0])-1)),0)
Lower_limit_11 = round(y11.Qty.mean() - (1.96*y11.Qty.std()/math.sqrt(len(y11.axes[0])-1)),0)

##########################################################

y12 = Data[Data['Product name'] == 'DETTOL LIQUID 550ML']
y12["Year"] = pd.to_datetime(y12["Date"]).dt.year
y12["month"] = pd.to_datetime(y12["Date"]).dt.month
y12["month_v1"] = (y12["month"] - y12["month"].mean())/y12["month"].std()
X = y12["month_v1"] ## X usually means our input variables (or independent variables)
y = y12["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y12 = y12.append({'Date' : '2020-04-01' , 'Product name' : 'DETTOL LIQUID 550ML'} , ignore_index=True)
y12 = y12.append({'Date' : '2020-05-01' , 'Product name' : 'DETTOL LIQUID 550ML'} , ignore_index=True)
y12 = y12.append({'Date' : '2020-06-01' , 'Product name' : 'DETTOL LIQUID 550ML'} , ignore_index=True)
y12["month_v2"] = pd.to_datetime(y12["Date"]).dt.month
y12["month_v3"] = (y12["month_v2"] - y12["month_v2"].mean())/y12['month_v2'].std()
y12['Pred_Qty'] = round(51.666+11.1333*y12['month_v3'],0)
y12 = y12.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_12 = round(y12.Qty.mean() + (1.96*y12.Qty.std()/math.sqrt(len(y12.axes[0])-1)),0)
Lower_limit_12 = round(y12.Qty.mean() - (1.96*y12.Qty.std()/math.sqrt(len(y12.axes[0])-1)),0)

########################################################

y13 = Data[Data['Product name'] == 'ECOSPRIN AV 75MG CAPSULE']
y13["Year"] = pd.to_datetime(y13["Date"]).dt.year
y13["month"] = pd.to_datetime(y13["Date"]).dt.month
y13["month_v1"] = (y13["month"] - y13["month"].mean())/y13["month"].std()
X = y13["month_v1"] ## X usually means our input variables (or independent variables)
y = y13["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y13 = y13.append({'Date' : '2020-04-01' , 'Product name' : 'ECOSPRIN AV 75MG CAPSULE'} , ignore_index=True)
y13 = y13.append({'Date' : '2020-05-01' , 'Product name' : 'ECOSPRIN AV 75MG CAPSULE'} , ignore_index=True)
y13 = y13.append({'Date' : '2020-06-01' , 'Product name' : 'ECOSPRIN AV 75MG CAPSULE'} , ignore_index=True)
y13["month_v2"] = pd.to_datetime(y13["Date"]).dt.month
y13["month_v3"] = (y13["month_v2"] - y13["month_v2"].mean())/y13['month_v2'].std()
y13['Pred_Qty'] = round(51.666-4.8300*y13['month_v3'],0)
y13 = y13.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_13 = round(y13.Qty.mean() + (1.96*y13.Qty.std()/math.sqrt(len(y13.axes[0])-1)),0)
Lower_limit_13 = round(y13.Qty.mean() - (1.96*y13.Qty.std()/math.sqrt(len(y13.axes[0])-1)),0)

################################################

y14 = Data[Data['Product name'] == 'EVION 400MG 10CAP']
y14["Year"] = pd.to_datetime(y14["Date"]).dt.year
y14["month"] = pd.to_datetime(y14["Date"]).dt.month
y14["month_v1"] = (y14["month"] - y14["month"].mean())/y14["month"].std()
X = y14["month_v1"] ## X usually means our input variables (or independent variables)
y = y14["Qty"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)
model.summary()
y14 = y14.append({'Date' : '2020-04-01' , 'Product name' : 'EVION 400MG 10CAP'} , ignore_index=True)
y14 = y14.append({'Date' : '2020-05-01' , 'Product name' : 'EVION 400MG 10CAP'} , ignore_index=True)
y14 = y14.append({'Date' : '2020-06-01' , 'Product name' : 'EVION 400MG 10CAP'} , ignore_index=True)
y14["month_v2"] = pd.to_datetime(y14["Date"]).dt.month
y14["month_v3"] = (y14["month_v2"] - y14["month_v2"].mean())/y14['month_v2'].std()
y14['Pred_Qty'] = round(51.666-4.8300*y14['month_v3'],0)
y14 = y14.drop(columns=['Year','month','month_v1','month_v2','month_v2','month_v3'])
upper_limit_14 = round(y14.Qty.mean() + (1.96*y14.Qty.std()/math.sqrt(len(y14.axes[0])-1)),0)
Lower_limit_14 = round(y14.Qty.mean() - (1.96*y14.Qty.std()/math.sqrt(len(y14.axes[0])-1)),0)

y1['Seaonal_index'] = round((y1['Pred_Qty']/y1['Pred_Qty'].mean()),2)
y1['estimated_sales'] = y1['Pred_Qty']*y1['Seaonal_index']
y2['Seaonal_index'] = round((y2['Pred_Qty']/y2['Pred_Qty'].mean()),2)
y2['estimated_sales'] = y2['Pred_Qty']*y2['Seaonal_index']
y3['Seaonal_index'] = round((y3['Pred_Qty']/y3['Pred_Qty'].mean()),2)
y3['estimated_sales'] = y3['Pred_Qty']*y3['Seaonal_index']
y4['Seaonal_index'] = round((y4['Pred_Qty']/y4['Pred_Qty'].mean()),2)
y4['estimated_sales'] = y4['Pred_Qty']*y4['Seaonal_index']
y5['Seaonal_index'] = round((y5['Pred_Qty']/y5['Pred_Qty'].mean()),2)
y5['estimated_sales'] = y5['Pred_Qty']*y5['Seaonal_index']
y6['Seaonal_index'] = round((y6['Pred_Qty']/y6['Pred_Qty'].mean()),2)
y6['estimated_sales'] = y6['Pred_Qty']*y6['Seaonal_index']
y7['Seaonal_index'] = round((y7['Pred_Qty']/y7['Pred_Qty'].mean()),2)
y7['estimated_sales'] = y7['Pred_Qty']*y7['Seaonal_index']
y8['Seaonal_index'] = round((y8['Pred_Qty']/y8['Pred_Qty'].mean()),2)
y8['estimated_sales'] = y8['Pred_Qty']*y8['Seaonal_index']
y9['Seaonal_index'] = round((y9['Pred_Qty']/y9['Pred_Qty'].mean()),2)
y9['estimated_sales'] = y9['Pred_Qty']*y9['Seaonal_index']
y10['Seaonal_index'] = round((y10['Pred_Qty']/y10['Pred_Qty'].mean()),2)
y10['estimated_sales'] = y10['Pred_Qty']*y10['Seaonal_index']
y11['Seaonal_index'] = round((y11['Pred_Qty']/y11['Pred_Qty'].mean()),2)
y11['estimated_sales'] = y11['Pred_Qty']*y11['Seaonal_index']
y12['Seaonal_index'] = round((y12['Pred_Qty']/y12['Pred_Qty'].mean()),2)
y12['estimated_sales'] = y12['Pred_Qty']*y12['Seaonal_index']
y13['Seaonal_index'] = round((y13['Pred_Qty']/y13['Pred_Qty'].mean()),2)
y13['estimated_sales'] = y13['Pred_Qty']*y13['Seaonal_index']
y14['Seaonal_index'] = round((y14['Pred_Qty']/y14['Pred_Qty'].mean()),2)
y14['estimated_sales'] = y14['Pred_Qty']*y14['Seaonal_index']
y1 = y1[y1['Date'] >= '2020-04-01']
y2 = y2[y2['Date'] >= '2020-04-01']
y3 = y3[y3['Date'] >= '2020-04-01']
y4 = y4[y4['Date'] >= '2020-04-01']
y5 = y5[y5['Date'] >= '2020-04-01']
y6 = y6[y6['Date'] >= '2020-04-01']
y7 = y7[y7['Date'] >= '2020-04-01']
y8 = y8[y8['Date'] >= '2020-04-01']
y9 = y9[y9['Date'] >= '2020-04-01']
y10 = y10[y10['Date'] >= '2020-04-01']
y11 = y11[y11['Date'] >= '2020-04-01']
y12 = y12[y12['Date'] >= '2020-04-01']
y13 = y13[y13['Date'] >= '2020-04-01']
y14 = y14[y14['Date'] >= '2020-04-01']
y1 = y1.drop(columns=['Qty'])
y2 = y2.drop(columns=['Qty'])
y3 = y3.drop(columns=['Qty'])
y4 = y4.drop(columns=['Qty'])
y5 = y5.drop(columns=['Qty'])
y6 = y6.drop(columns=['Qty'])
y7 = y7.drop(columns=['Qty'])
y8 = y8.drop(columns=['Qty'])
y9 = y9.drop(columns=['Qty'])
y10 = y10.drop(columns=['Qty'])
y11 = y11.drop(columns=['Qty'])
y12 = y12.drop(columns=['Qty'])
y13 = y13.drop(columns=['Qty'])
y14 = y14.drop(columns=['Qty'])

frames = [y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14]
y = pd.concat(frames)
y = y[['Date','Product name','Pred_Qty','estimated_sales']]
y["Year"] = pd.to_datetime(y["Date"]).dt.year
y["month"] = pd.to_datetime(y["Date"]).dt.month
y = y[['Date','Product name','Year','month','Pred_Qty','estimated_sales']]

print('Prediction of product quantity on next three months are given as follow')
y
upper_limit = [upper_limit_1,upper_limit_2,upper_limit_3,upper_limit_4,upper_limit_5,upper_limit_6,upper_limit_7,upper_limit_8,upper_limit_9,upper_limit_10,upper_limit_11,upper_limit_12,upper_limit_13,upper_limit_14]
Lower_limit = [Lower_limit_1,Lower_limit_2,Lower_limit_3,Lower_limit_4,Lower_limit_5,Lower_limit_6,Lower_limit_7,Lower_limit_8,Lower_limit_9,Lower_limit_10,Lower_limit_11,Lower_limit_12,Lower_limit_13,Lower_limit_14]
Product_name =['NAN PRO 2 REFILL 400GM','NAN PRO 1 REFILL 400GM','ENO FRUITSALT LEMON 5GM','ECOSPRIN 75MG 14TAB','NAN PRO 3 REFILL 400GM','LACTOGEN STAGE 1 REFILL 400GM','DUPHASTON 10TAB','ENFAMIL A+ POWDER STAGE 2 400GM','CROCIN ADVANCE TABLET','LACTOGEN STAGE 2 REFILL 400GM','ENFAMIL A+ POWDER STAGE 1 400GM','DETTOL LIQUID 550ML','ECOSPRIN AV 75MG CAPSULE','EVION 400MG 10CAP'] 
print("Upper and lower limit for each product based on there actual quantity")
print('\n')
for i in range(len(upper_limit)):
    print("Upper and lower limit for product {} are {} and  {}  ".format(Product_name[i],Lower_limit[i],upper_limit[i]))