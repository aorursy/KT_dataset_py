#import pyforest

import pandas as pd

import numpy as np
# let's consider the sample data set

df=pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c'],

                 'target':[1,0,1,0,1,1,0,1,0]

                })

df
import category_encoders as ce

ce.__version__



# Encoded values with out regularization

# a- 1/3 - 0.3333(mean)

# b- 2/3 - 0.6666(mean)

# c- 2/3 - 0.6666(mean)
X = df.drop('target', axis = 1)

y = df.target

type(X),type(y)
# Mean Encoding with smoothing regularization parameter. 

Mean_encoding=ce.m_estimate.MEstimateEncoder()

X_new=Mean_encoding.fit_transform(X,y)

X_new  #By default it returns pandas dataframe, can get numpy array also by changing the parameter values in the encoding method.



# Gives mean encoded values with regularization ('m' or 'alpha'= 1 default value)
df=pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c',np.nan],

                 'target':[1,0,1,0,1,1,0,1,0,0]

                })

df
X = df.drop('target', axis = 1).values

y = df.target.values 

type(X),type(y)

# Earlier we passed pandas objects, here we passes numpy arrays
Mean_encoding=ce.m_estimate.MEstimateEncoder(return_df=False)

X_new=Mean_encoding.fit_transform(X,y)

X_new  # Now the output is numpy array. we controlled this with 'return_df' argument 
# With handle_missing='error'

Mean_encoding=ce.m_estimate.MEstimateEncoder(handle_missing='error')

# X_new=Mean_encoding.fit_transform(X,y)



# Raises value error if missing values are encountered in training phase.

# For readability purpose I didn't shown the output here.
# With handle_missing='return_nan'

Mean_encoding=ce.m_estimate.MEstimateEncoder(handle_missing='return_nan')

X_new=Mean_encoding.fit_transform(X,y)

X_new  

# It returns Nan value for missing values 
# With handle_missing='value'

Mean_encoding=ce.m_estimate.MEstimateEncoder(handle_missing='value')

X_new=Mean_encoding.fit_transform(X,y)

X_new  



# The default value is for this argument is 'value'. It treats Nan's as one among the class levels and

# compute the encoded value as usual. But in documentation it is mentioned that it will return the global mean(prior prob)
Mean_encoding.transform(['d']) # 'd' is unseen data



# It returns the global mean value.(A reasonable metric to substitute for unknown category while performing mean encoding).

# One good thing in category_encoders package is it will nicely handle the unknown values while transforming.

# If we set the value of 'handle_unknown' to 'return_nan' it will return Nan

# If we set the value of 'handle_unknown' to 'error' it will raise an error
df=pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c'],

                 'B':[1,2,3,4,1,2,3,4,1],

                 'target':[1,0,1,0,1,1,0,1,0]

                })

df.head()
X = df.drop('target', axis = 1)

y = df.target

type(X),type(y)
Mean_encoding=ce.m_estimate.MEstimateEncoder()

X_new=Mean_encoding.fit_transform(X,y)

X_new
Mean_encoding=ce.m_estimate.MEstimateEncoder(cols=[0,1])

# X_new=Mean_encoding.fit_transform(X,y) #commented for readability purpose

# X_new



# raises Key error, So indices are not allowed. I tried with (cols=[0]) also
# let's try with column names

Mean_encoding=ce.m_estimate.MEstimateEncoder(cols=['A','B'])

X_new=Mean_encoding.fit_transform(X,y)

X_new



# So, we can encode the numeric values also, by specifying the column names(not indices)
X = df.drop('target', axis = 1).values

y = df.target.values

type(X),type(y)
Mean_encoding=ce.m_estimate.MEstimateEncoder(cols=[0,1])

X_new=Mean_encoding.fit_transform(X,y)

X_new
df=pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c'],

                 'target':[123.5,100,120.3,101,108.7,109,100,113,110]

                })

df.head()
X = df.drop('target', axis = 1)

y = df.target

type(X),type(y)
Mean_encoding=ce.m_estimate.MEstimateEncoder()

X_new=Mean_encoding.fit_transform(X,y)

X_new

#It works well as expected
# let's consider the sample data set

df=pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','a','b','c'],

                 'target':[1,0,1,0,1,1,0,1,0,2,0,2,2,1,2,0,1,0]

                })

df
X = df.drop('target', axis = 1)

y = df.target



Mean_encoding=ce.m_estimate.MEstimateEncoder()

X_new=Mean_encoding.fit_transform(X,y)

X_new
# let's consider the sample data set

df=pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','a','b','c'],

                 'target_0':[0,1,0,1,0,0,1,0,1,0,1,0,0,0,0,1,0,1],

                 'target_1':[1,0,1,0,1,1,0,1,0,0,0,0,0,1,0,0,1,0],

                 'target_2':[0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0,0]

                })

df
# Encoded value obtained from level '0' 

Mean_encoding=ce.m_estimate.MEstimateEncoder()

X_new=Mean_encoding.fit_transform(df['A'],df['target_0'])

df['target_0_encoded']=X_new

X_new
# Encoded value obtained from level '1'

Mean_encoding=ce.m_estimate.MEstimateEncoder()

X_new=Mean_encoding.fit_transform(df['A'],df['target_1'])

df['target_1_encoded']=X_new

X_new
# Encoded value obtained from level '2'

Mean_encoding=ce.m_estimate.MEstimateEncoder()

X_new=Mean_encoding.fit_transform(df['A'],df['target_2'])

df['target_2_encoded']=X_new

X_new
df
df['sum']=df['target_0_encoded']+df['target_1_encoded']+df['target_2_encoded']

df
# let's consider the sample data set

df=pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c'],

                 'target':[1,0,1,0,1,1,0,1,0]

                })

df
X = df.drop('target', axis = 1)

y = df.target.values 



Mean_encoding=ce.leave_one_out.LeaveOneOutEncoder()

Mean_encoding.fit_transform(X,y)
Mean_encoding.transform(pd.DataFrame({'A':['a','b','c']}))
#let's see by applying mean encoding

Mean_encoding=ce.m_estimate.MEstimateEncoder(m=0) #smoothing=0 : Target encoding without smoothing/regularization

Mean_encoding.fit_transform(df['A'],df['target'])
Mean_encoding.transform(pd.DataFrame({'A':['a','b','c']}))
# Introduced a level ('d') only occured once in the variable 

df=pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c','d'],

                 'target':[1,0,1,0,1,1,0,1,0,1]

                })

df
X = df.drop('target', axis = 1)

y = df.target.values 



Mean_encoding=ce.leave_one_out.LeaveOneOutEncoder()

Mean_encoding.fit_transform(X,y)
# Can we set it to be nan by changing the handle_missing=return_nan. let's see

Mean_encoding=ce.leave_one_out.LeaveOneOutEncoder(handle_missing='return_nan')

Mean_encoding.fit_transform(X,y)
# Another observation of a level ('d') added 

df=pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c','d','d'],

                 'target':[1,0,1,0,1,1,0,1,0,1,0]

                })

df
X = df.drop('target', axis = 1)

y = df.target.values 



Mean_encoding=ce.leave_one_out.LeaveOneOutEncoder()

Mean_encoding.fit_transform(X,y)
# let's consider the demo data set

df=pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c'],

                 'target':[1,0,1,0,1,1,0,1,0]

                })

df
X = df.drop('target', axis = 1).values

y = df.target.values



Mean_encoding=ce.target_encoder.TargetEncoder(min_samples_leaf=1)

X_new=Mean_encoding.fit_transform(X,y)

X_new  



# The default regularization parameter values are  smoothing=1 and min_samples_leaf=1
# Validation for level 'a'

import math

smove=1/(1+math.exp(-((3-1)/1)))

target_mean=df.target.mean()

smoothing=((target_mean)*(1-smove))+((1/3)*smove)

smoothing
# Validation for level 'b','c'

smove=1/(1+math.exp(-((3-1)/1)))

target_mean=df.target.mean()

smoothing=((target_mean)*(1-smove))+((2/3)*smove)

smoothing
df=pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','d','d'],

                 'target':[1,0,1,0,1,1,0,1,0,1,0,1,0,1,1,0,1,0,0,0]

                })

df['A'].value_counts()
X = df.drop('target', axis = 1).values

y = df.target.values



Mean_encoding=ce.target_encoder.TargetEncoder()

min_samples_leaf_as_1=Mean_encoding.fit_transform(X,y)

df['min_samples_leaf_as_1']=min_samples_leaf_as_1

df

Mean_encoding=ce.target_encoder.TargetEncoder(min_samples_leaf=5)

min_samples_leaf_as_5=Mean_encoding.fit_transform(X,y)

df['min_samples_leaf_as_5']=min_samples_leaf_as_5

df
Mean_encoding=ce.target_encoder.TargetEncoder(min_samples_leaf=10)

min_samples_leaf_as_10=Mean_encoding.fit_transform(X,y)

df['min_samples_leaf_as_10']=min_samples_leaf_as_10

df
df.drop(['target'],axis=1,inplace=True)

df.drop_duplicates()
df=pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','a','b','c','d','d'],

                 'target':[1,0,1,0,1,1,0,1,0,1,0,1,0,1,1,0,1,0,0,0]

                })

df['A'].value_counts()
df
Mean_encoding=ce.cat_boost.CatBoostEncoder()

X_new=Mean_encoding.fit_transform(df['A'],df['target'])

X_new['levels']=df['A']

X_new
df_test=pd.DataFrame({'A':['a','b','c','d','a','b','c','d']})

df_test
X_new_transform=Mean_encoding.transform(df_test['A'])

X_new_transform['levels']=df_test['A']

X_new_transform
Mean_encoding_me=ce.m_estimate.MEstimateEncoder() #smoothing=1

Data=Mean_encoding_me.fit_transform(df['A'],df['target'])
Mean_encoding_me.transform(pd.DataFrame({'A':['a','b','c','d','a','b','c','d']}))
# result obtained from catboost encoding

X_new_transform