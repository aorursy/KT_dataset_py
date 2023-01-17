# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sbn
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import glob
import theano.tensor as T
import theano
from sklearn.metrics import mean_squared_error
print ("Helpful guide: https://github.com/parsing-science/pymc3_quickstart_guide")
df=pd.concat([pd.read_csv(f) for f in glob.glob('../input/*.csv')], ignore_index = True)
df.count()[0]
df=df.dropna()
df["start_day"]=pd.to_datetime(df["start_time"], errors='ignore')
df["start_day"]= df['start_day'].dt.floor("d")
df.head()
df=pd.get_dummies(columns=["user_type"],data=df)
df.head()
aggregations = {
    "user_type_Customer":"sum",
    "user_type_Subscriber":"sum",  
}
day=df.groupby("start_day").agg(aggregations)
day.head()
dayList = day.index.day_name()
dayList
day["total_riders"]=day["user_type_Customer"]+day["user_type_Subscriber"]
print (len(day["total_riders"]))
nextDay=list(day["total_riders"])
nextDay.pop(0)#Don't need this value anymore
nextDay.append(0.0)#Add a zero to the next one as a test
print (len(nextDay))
#print (nextDay)
from sklearn.preprocessing import RobustScaler
scaledDF = RobustScaler().fit_transform(day)
scaledDF = pd.DataFrame(data=scaledDF, columns = ["scaled_"+str(x) for x in day.columns])
scaledDF['day'] = dayList

dayLookup = pd.DataFrame({"day": scaledDF.day.unique(), "dayIndex": range(7)})
scaledDF["nextDay"]=nextDay
scaledDF=scaledDF[:len(nextDay)-1]
scaledDF.tail(5)
scaledDF = pd.merge(scaledDF, dayLookup, on=["day"],how='left'  )
scaledDF.tail(5)
import seaborn as sns
corr = scaledDF.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
"""
If we plot the distributions of these variables, we see 
significant co-linearity among some variables with respect 
to the next day predictions. Predictably, categorical features
that exist in the majority (male ridership and subscription riders)
have strong relationships.
"""

g = sbn.pairplot(scaledDF[scaledDF['day']=='Saturday']);
"""
When we plotted the entirety of distributions together (all days of the week),
we could see multiple, linear relationships in the data. Now we observe what 
look to be very tight correlations.
"""

g = sbn.pairplot(scaledDF[scaledDF['day']=='Wednesday']);
print ("Let us try some baseline predictions: Naive Average of all ridership")

naivePreds=np.ones(len(scaledDF['nextDay']))+np.mean(day['total_riders'])
np.sqrt(mean_squared_error(scaledDF["nextDay"], naivePreds))
print ("What if we simply look back one day and see if that is a good predictor?")
np.sqrt(mean_squared_error(scaledDF["nextDay"], day["total_riders"][:len(scaledDF["nextDay"])]))
import theano.tensor as T

y = scaledDF["nextDay"]
X = scaledDF['scaled_total_riders']
index = scaledDF.dayIndex

#Let's test our model on the last 30 days of data
month_split = len(y)-30
X_train , Y_train, Index_train =  X[:month_split], y[:month_split], index[:month_split]
X_test , Y_test, Index_test = X[month_split:], y[month_split:], index[month_split:]

model_index = theano.shared(np.array(Index_train))
model_input = theano.shared(np.array(X_train))
model_output = theano.shared(np.array(Y_train))
import pymc3 as pm

print('Running on PyMC3 v{}'.format(pm.__version__))
hierarchical_model = pm.Model()

nDays = len(scaledDF.dayIndex.unique())
with hierarchical_model:

    """
    Meta Priors: For each day of the week, we 
    should have a different set of distributions. Potentially
    we could have different priors even for the season,
    weather, etc.
    """
    day_alpha = pm.Normal('day_alpha', mu=0, sd=100)
    day_beta = pm.Normal('day_beta', mu=0, sd=100)
    
    """
    Model the uncertainty of our parent distributions 
    with a HalfCauchy with beta = 4.
    """
    sigma_day_alpha = pm.HalfCauchy('sigma_day_alpha', 4)
    sigma_day_beta = pm.HalfCauchy('sigma_day_beta', 4)
    
    """
    Now we draw distributions depending on the day from
    the above values
    """ 

    alpha = pm.Normal('alpha', mu = day_alpha, sd = sigma_day_alpha, shape = nDays )
    beta = pm.Normal('beta', mu = day_beta, sd = sigma_day_beta, shape = nDays )

    # Expected value of outcome  
    values = np.exp(alpha[model_index] + beta[model_index]*model_input )
    
    # Likelihood (samplYeah does noting distribution) of observations
    Y_obs = pm.Poisson('Y_obs', mu=values, observed=model_output)
with hierarchical_model:
    trace = pm.sample(8000)
pm.traceplot(trace[-1000:]);
pm.summary(trace[-1000:])
import pickle
fileObject = open("models/nuts_trace_hierarchical.pickle",'wb')  
pickle.dump(trace, fileObject)
fileObject.close()
#Set to training again
with hierarchical_model:
    inference = pm.ADVI()
    approx = pm.fit(n=100000, method=inference)
advi_trace = approx.sample(6000)
pm.traceplot(trace[-1000:]);
import pickle
fileObject = open("models/advi_trace_hierarchical.pickle",'wb')  
pickle.dump(trace, fileObject)
fileObject.close()
def scoreModel(trace,y,model_name):
    ppc = pm.sample_ppc(trace[1000:], model=model_name, samples=1000)
    pred = ppc['Y_obs'].mean(axis=0)
    return np.sqrt(mean_squared_error(y, pred))

scoreModel(trace, Y_train, hierarchical_model)
scoreModel(advi_trace, Y_train, hierarchical_model)
model_index.set_value(np.array(Index_test))
model_input.set_value(np.array(X_test))
model_output.set_value(np.array(Y_test))
scoreModel(advi_trace,Y_test,hierarchical_model)
#Now we need to test our holdout
scoreModel(trace,Y_test,hierarchical_model)
ppc = pm.sample_ppc(trace[1000:], model=hierarchical_model, samples=1000)
print (ppc['Y_obs'])
model_index.get_value()

