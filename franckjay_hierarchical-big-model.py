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
import pandas as pd
import numpy as np
import glob
import theano.tensor as T
import theano
from sklearn.metrics import mean_squared_error
print ("Helpful guide: https://github.com/parsing-science/pymc3_quickstart_guide")
df=pd.concat([pd.read_csv(f) for f in glob.glob('../input/*.csv')], ignore_index = True)
df.count()[0]
df=df.dropna()
df.head()
df["duration_hrs"]=df["duration_sec"]/3600.
df["age"]=2019-df["member_birth_year"]
df["start_day"]=pd.to_datetime(df["start_time"], errors='ignore')
df["start_day"]= df['start_day'].dt.floor("d")
df=pd.get_dummies(columns=["member_gender","user_type"],data=df)
df.head()
aggregations = {
    'duration_hrs':'mean',
    "age" :"mean",
    "member_gender_Female":"sum",
    "member_gender_Male":"sum",
    "member_gender_Other":"sum",
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
from sklearn.preprocessing import RobustScaler
scaledDF = RobustScaler().fit_transform(day)
scaledDF = pd.DataFrame(data=scaledDF, columns = ["scaled_"+str(x) for x in day.columns])
scaledDF['day'] = dayList
dayLookup = pd.DataFrame({"day": scaledDF.day.unique(), "dayIndex": range(7)})
scaledDF["nextDay"]=nextDay
scaledDF=scaledDF[:len(nextDay)-1]
scaledDF.tail(5)
scaledDF = pd.merge(scaledDF, dayLookup, on=["day"], how='left')
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
X = scaledDF.drop(['day','nextDay','dayIndex'],axis=1)
index = scaledDF.dayIndex

#Let's test our model on the last 30 days of data
month_split = len(y)-30
X_train , Y_train, Index_train =  X[:month_split], y[:month_split], index[:month_split]
X_test , Y_test, Index_test = X[month_split:], y[month_split:], index[month_split:]

"""
We need to build a shared tensor for the input, output, and now
the index value that corresponds to which day we are referencing.
"""
model_index = theano.shared(np.array(Index_train))
model_input = theano.shared(np.array(X_train))
model_output = theano.shared(np.array(Y_train))
print (X_train.shape,Y_train.shape,Index_train.shape)
print (X_test.shape,Y_test.shape,Index_test.shape)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
preds = lr.predict(X_test)
np.sqrt(mean_squared_error(Y_test, preds)) #1298.350508812364
np.sqrt(mean_squared_error(Y_test, naivePreds[:len(Y_test)]))#2117.4783114031584
import pymc3 as pm

print('Running on PyMC3 v{}'.format(pm.__version__))
hierarchical_big_model = pm.Model()

nDays, nFeatures = len(scaledDF.dayIndex.unique()), X.shape[1]
with hierarchical_big_model:

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
    the above values. We now have to grab betas from an array
    of shape 7,8 as we have 7 days and 8 weights to learn.
    """ 
    alpha = pm.Normal('alpha', mu = day_alpha, sd = sigma_day_alpha, shape = nDays )
    beta = pm.Normal('beta', mu = day_beta, sd = sigma_day_beta, shape = (nDays,nFeatures) )
    """
     We can do the dot product as long as we index the day using the model_index and
     then the 8 weights via beta[model_index,:].T .
    """
    values = np.exp(alpha[model_index] + T.dot(model_input, beta[model_index,:].T) )
    
    # Likelihood (samplYeah does noting distribution) of observations
    Y_obs = pm.Poisson('Y_obs', mu=values, observed=model_output)
#Set to training again
with hierarchical_big_model:
    inference = pm.ADVI()
    approx = pm.fit(n=100000, method=inference)
advi_trace = approx.sample(10000)
import pickle
fileObject = open("models/advi_tracehierarchical_big.pickle",'wb')  
pickle.dump(advi_trace, fileObject)
fileObject.close()
pm.traceplot(advi_trace[-1000:]);
def scoreModel(trace,y,model_name):
    ppc = pm.sample_ppc(trace[1000:], model=model_name, samples=1000)
    #We have to change the scoring model to grab the first element
    pred = ppc['Y_obs'][0].mean(axis=0)
    return np.sqrt(mean_squared_error(y, pred))

scoreModel(advi_trace,Y_train,hierarchical_big_model)
model_input.set_value(np.array(X_test))
model_index.set_value(np.array(Index_test))
model_output.set_value(np.array(Y_test))
scoreModel(advi_trace,Y_test,hierarchical_big_model)
ppc = pm.sample_ppc(advi_trace[1000:], model=hierarchical_big_model, samples=1000)
print (ppc['Y_obs'][0].mean(axis=0))
