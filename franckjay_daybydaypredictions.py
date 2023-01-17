# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sbn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import os
print(os.listdir("../input"))
import pickle
# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import glob
import theano.tensor as T
import theano
from sklearn.metrics import mean_squared_error
print ("Helpful guide: https://github.com/parsing-science/pymc3_quickstart_guide")
df=pd.concat([pd.read_csv(f) for f in glob.glob('../input/*.csv')], ignore_index = True)
df.count()[0]
"""
Here are a few takeaways from the statistics of our dataset:

Gender Breakdown: 19% Female, 67% Male
Subscribers: 79% Subscribers

There are a number of missing values in a variety of columns in the dataset.
The duration of the rides generally look to be for less than an hour, but 
some values are anomalously high.


"""

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
scaledDF["nextDay"]=nextDay
scaledDF=scaledDF[:len(nextDay)-1]
scaledDF.head(5)
"""
From the heatmap, we can see some simple relationships between variabes, 
both positive and negative with respect to our target variable (nextDay).

However, this only tells part of the story. We know that the variables 
are inter-related, and to corresponding to some magnitude of correlation,
but there could be deeper relationships as well.
"""
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

Our target value (next day's ridership) appears to be a binomial distribution. Many
of the features, when scaled, also share this type of distribution. The scatter plots
show that there exist multiple linear relationships for some of the features. These could
correspond to weekend/weekday differences.
"""
g = sbn.pairplot(scaledDF);
print ("Let us try some baseline predictions: Naive Average of all ridership")

naivePreds=np.ones(len(scaledDF['nextDay']))+np.mean(day['total_riders'])
np.sqrt(mean_squared_error(scaledDF["nextDay"], naivePreds))
print ("What if we simply look back one day and see if that is a good predictor?")
np.sqrt(mean_squared_error(scaledDF["nextDay"], day["total_riders"][:len(scaledDF["nextDay"])]))
import theano.tensor as T

y = scaledDF["nextDay"]
X = scaledDF.drop('nextDay',axis=1)

#Let's test our model on the last month of data
month_split = len(y)-30
X_train , Y_train =  X[:month_split], y[:month_split]
X_test , Y_test = X[month_split:], y[month_split:]

#We initiate a shared theano array for training and
#    testing splits for ease of use.
model_input = theano.shared(np.array(X_train))
model_output = theano.shared(np.array(Y_train))
"""
What would a general sklearn Linear Regression pick for our parameters,
and how would it score? We can use this as our ML baseline without 
any hyperparameter tuning.
"""

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
preds = lr.predict(X_test)
np.sqrt(mean_squared_error(Y_test, preds))
import pymc3 as pm

print('Running on PyMC3 v{}'.format(pm.__version__))

big_model = pm.Model()

with big_model:
    """
    Our simple, linear model requires an intercept (alpha) and a weight
    for each of our parameters (beta). I have no prior knowledge about what
    the weights could be, but I know the features are scaled by the 
    RobustScaler. Therefore, we can assume they might be of centered around
    0 (mu). In this example, we did not choose to scale our target value 
    (riders the next day), so the slope of our scaled parameters might be 
    quite large. Therefore, we set sd=100. If we set it too low, it may 
    never find the correct beta value for that parameter.
    """
    alpha = pm.Normal('alpha', mu=0, sd=100)
    beta = pm.Normal('beta', mu=0, sd=100, shape=8)

    """
    The values we are trying to predict will be a simple dot product of our 
    features with the weights of our model (transpose of beta) plus our 
    intercept. We take the exponential of this value because we will 
    be modelling our output as a Poisson distribution. If we modelled it
    as a Gaussian, we would remove the np.exp and introduce a value 
    for the noise (sigma).
    """ 
    values = np.exp(alpha + T.dot(model_input, beta.T) )
    
    """
    This is the final output of our model. Our target variable (model_output)
    is being modelled as a Poisson, as we are dealing with a simple counting
    statistic (the number of riders the next day). We could reasonably choose
    a different distribution in PyMC3 (Normal, DiscreteUniform, et al.) but 
    this seems intuitive.
    """
    Y_obs = pm.Poisson('Y_obs', mu=values, observed=model_output)
"""
Now that we have specified our model, we need to 
begin estimating our model parameters. We could specify 
a sampler (Metropilis, NUTS, etc.) or we could choose
.sample(), which chooses a sampler suited to our data. 
Generally it chooses NUTS.
"""
with big_model:
    start = pm.find_MAP() # Use the MAP estimate as a starting value
    nuts_trace = pm.sample(8000, scaling=start) # Begin to build our trace
"""
By plotting the trace of our model, we can examine 
the distribution of the parameters as well as how well
they are converged. 

Generally speaking, if we see a parameter that has a peaked 
posterior distribution, the model is fairly sure of itself. Likewise,
if we look at the evolution of the parameter (right-side of the plots)
and see that it is fairly flat, the sampler has had no reason to deviate
from what it thinks is correct.

Notably, this model is not all that well constrained. We see on the left side 
that some beta parameters (the weights) have wide distributions, and the corresponding
evolution of these values continues to change on the right side. 

If we remember back to the pairplot above, we saw what looked to be multiple linear 
relationships in the data. It is certainly plausible that the sampler cannot converge 
in these types of situations with our simple linear model.
"""
pm.traceplot(nuts_trace[-1000:]);
pm.summary(nuts_trace[-1000:])
"""
We can pickle our traces as a binary model, and then 
re-use them later if we want.
"""
fileObject = open("nuts_trace.pickle",'wb')  
pickle.dump(nuts_trace, fileObject)
fileObject.close()
"""
What if we have a large, complicated model that we can only
approximate the posterior distribution?  Use Variational Inference:
https://docs.pymc.io/api/inference.html?highlight=advi#pymc3.variational.inference.ADVI

It works extremely quickly compared to NUTS, and can offer increased performance in 
estimating the model parameters for very complex distributions. Simple distributions 
will be better fit by NUTS.
"""
with big_model:
    inference = pm.ADVI()
    approx = pm.fit(n=100000, method=inference)
#Now we sample from our approximation in order to get a similar trace
advi_trace = approx.sample(10000)
pm.traceplot(advi_trace[-1000:]);
pm.summary(advi_trace[-1000:])
#Save the advi model as well
fileObject = open("advi_trace.pickle",'wb')  
pickle.dump(advi_trace, fileObject)
fileObject.close()
"""
Let's get the RMSE of our model as understood by the NUTS and ADVI
sampler. We will test our training data error first.

The NUTS sampler does a pretty decent job on predictions, and has a 
better RMSE than our sklearn LR model.
"""

def scoreModel(trace,y,model_name):
    ppc = pm.sample_ppc(trace[1000:], model=model_name, samples=1000)
    pred = ppc['Y_obs'].mean(axis=0)
    return np.sqrt(mean_squared_error(y, pred))

scoreModel(nuts_trace,Y_train,big_model)
"""
Our ADVI predictions are very consistent with our NUTS sampler, 
and it took a lot less time to train. This may not always be the case 
with all models!
"""
scoreModel(advi_trace,Y_train,big_model)
"""
Now we simply switch out our Theano tensor with our 
testing data. Using the shared value, we do not have
to respecify our model first.
"""
model_input.set_value(np.array(X_test))
model_output.set_value(np.array(Y_test))

scoreModel(nuts_trace,Y_test,big_model)
"""
Once again, our PyMC3 models do better than our sklearn 
model, and we did not do any hyperparameter tuning for either.

Is the increased difficulty and time worth the better performance? 
That is for you to decide.
"""
scoreModel(advi_trace,Y_test,big_model)