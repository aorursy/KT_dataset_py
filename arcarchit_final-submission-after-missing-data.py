# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from sklearn.linear_model import Ridge, LinearRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
trainDF = pd.read_csv('../input/train.csv');

testDF = pd.read_csv('../input/test.csv');
all_data = pd.concat((trainDF.loc[:,'MSSubClass':'SaleCondition'],

                      testDF.loc[:,'MSSubClass':'SaleCondition']))



all_data.loc[all_data.PoolQC.isnull(), 'PoolQC']='NA'





# In[212]:



all_data = all_data.replace({'PoolQC': {'Ex': 4,

                                            'Gd': 3,

                                            'TA': 2,

                                            'Fa': 1,

                                           'NA':0

                                            }

        })





# ### Alley



# In[213]:



all_data.loc[all_data.Alley.isnull(), 'Alley']='NA'





# ### Fence



# In[214]:



all_data.loc[all_data.Fence.isnull(), 'Fence']='NA'





# In[215]:



all_data = all_data.replace({'Fence': {'GdPrv': 4,

                                            'MnPrv': 3,

                                            'GdWo': 2,

                                            'MnWw': 1,

                                           'NA':0

                                            }

        })





# ### FirePlaceQu



# In[216]:



all_data.loc[all_data.FireplaceQu.isnull(), 'FireplaceQu']='NA'





# In[217]:



all_data = all_data.replace({'FireplaceQu': {'Ex': 5,

                                            'Gd': 4,

                                            'TA': 3,

                                            'Fa': 2,

                                            'Po': 1,

                                           'NA':0

                                            }

        })





# ### Utilities



# In[218]:



all_data.loc[all_data.Utilities.isnull(), 'Utilities']='AllPub'





# ### Kitchen Qualitty



# In[221]:



all_data.loc[all_data.KitchenQual.isnull(), 'KitchenQual']='TA'





# In[222]:



all_data = all_data.replace({ 'KitchenQual': {'Ex': 5,

                                            'Gd': 4,

                                            'TA': 3,

                                            'Fa': 2,

                                            'Po': 1

                                            }

        })





# ### SaleType



# In[223]:



all_data.loc[all_data.SaleType.isnull(), 'SaleType']='WD'





# ### MasVnrArea



# In[224]:



all_data.loc[all_data.MasVnrArea.isnull(), 'MasVnrArea']=0





# ### MasVnrType



# In[225]:



all_data.loc[all_data.MasVnrType.isnull(), 'MasVnrType']='None'





# ### Basement



# In[226]:



for c in ['TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtHalfBath', 'BsmtFullBath']:

    all_data.loc[all_data[c].isnull(), c]=0





# In[227]:



indexes = all_data['BsmtFinType1'].isnull()

for c in ['BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtExposure']:

    all_data.loc[indexes, c]='NA'





# In[228]:



indexes = all_data['BsmtFinType1'].isnull()

for c in ['BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtExposure']:

    all_data.loc[indexes, c]='NA'

#We filtered on BsmtFinType1 because it had least no of Nulls. Now there would 2 more Nulls in BsmtCond, need to check it further



all_data.loc[all_data.BsmtCond.isnull(), 'BsmtCond']='TA'

all_data = all_data.replace({ 'BsmtCond': {'Ex': 5,

                                            'Gd': 4,

                                            'TA': 3,

                                            'Fa': 2,

                                            'Po': 1,

                                           'NA':0

                                            }

        })





# In[231]:





all_data.loc[all_data.BsmtQual.isnull(), 'BsmtQual']='TA'

all_data = all_data.replace({ 'BsmtQual': {'Ex': 5,

                                            'Gd': 4,

                                            'TA': 3,

                                            'Fa': 2,

                                            'Po': 1,

                                           'NA':0

                                            }

        })







# In[234]:



all_data.loc[all_data.BsmtExposure.isnull(), 'BsmtExposure']='No'

all_data = all_data.replace({ 'BsmtExposure': {'Gd': 4,

                                            'Av': 3,

                                            'Mn': 2,

                                            'No': 1,

                                              'NA':0

                                            }

        })





# In[235]:



all_data.loc[all_data.BsmtFinType2.isnull(), 'BsmtFinType2']='Unf'

all_data = all_data.replace({ 'BsmtFinType2': {'GLQ': 6,

                                            'ALQ': 5,

                                            'BLQ': 4,

                                            'Rec': 3,

                                            'LwQ': 2,

                                              'Unf':1,

                                              'NA':0

                                            }

        })





# ### MSZoning



# In[237]:



all_data.loc[all_data.MSZoning.isnull(), 'MSZoning']='RL'





# ### Functional



# In[238]:



all_data.loc[all_data.Functional.isnull(), 'Functional']='Typ'

all_data = all_data.replace({ 'Functional': {'Typ': 7,

                                            'Min1': 6,

                                            'Min2': 5,

                                            'Mod': 4,

                                            'Maj1': 3,

                                              'Maj2':2,

                                              'Sev':1,

                                             'Sal':0

                                            }

        })





# In[239]:



all_data.loc[all_data.Electrical.isnull(), 'Electrical']='SBrkr'





# ### LotFrontage



# In[240]:



nullIndex = all_data.LotFrontage.isnull()





# In[241]:



nonNullIndex = all_data.LotFrontage.notnull()





# In[242]:



X_Train = np.sqrt(all_data.LotArea[nonNullIndex])





# In[243]:



Y_Train = all_data.LotFrontage[nonNullIndex]





# In[244]:



y= np.array(Y_Train)





# In[245]:



b=np.array(X_Train)

x = b.reshape(len(b),1)





# In[246]:



model = LinearRegression()





# In[247]:



model.fit(x, y)





# In[248]:



X_Test = np.sqrt(all_data.LotArea[nullIndex])

b=np.array(X_Test)

xt = b.reshape(len(b),1)

yPred = model.predict(xt)





# In[249]:



all_data.loc[nullIndex, 'LotFrontage']=yPred





# In[250]:



all_data.LotFrontage.isnull().sum()





# ### Garage



# In[251]:



garage = all_data[['GarageCond', 'GarageQual', 'GarageYrBlt', 'GarageFinish', 'GarageType', 'GarageCars', 'GarageArea']]





# In[252]:



garage.head()





# In[253]:



garage.GarageYrBlt.isnull().sum()





# In[254]:



garage[garage.GarageYrBlt.isnull()].head()





# So when garage year blt is null, we can say that there is no garage.

# Looking at the data description it seems that GarageType, GarageFinish, GarageQuality, GarageCondition has NA values signifying No Garrage. Replacing them.



# In[255]:



all_data.loc[all_data.GarageYrBlt.isnull(), 'GarageCond']='NA'





# In[256]:



all_data.loc[all_data.GarageYrBlt.isnull(), 'GarageFinish']='NA'





# In[257]:



all_data.loc[all_data.GarageYrBlt.isnull(), 'GarageQual']='NA'





# In[258]:



all_data.loc[all_data.GarageYrBlt.isnull(), 'GarageType']='NA'





# Now how to impute garage year built, when garage is not built. I think we should check the relation between Garage year build and sale price



# In[259]:



# In[260]:



trainDF['HasGarage']=trainDF.GarageYrBlt.apply(lambda x:0 if math.isnan(x) else 1)







# In[262]:



all_data.loc[all_data.GarageYrBlt.isnull(), 'GarageYrBlt']=0

all_data['HasGarage']=all_data.GarageYrBlt.apply(lambda x:0 if math.isnan(x) else 1)





# We clearly see that house that garage has more price. Now how to impute this value.



# In[263]:



garage = all_data[['GarageCond', 'GarageQual', 'GarageYrBlt', 'GarageFinish', 'GarageType', 'GarageCars', 'GarageArea']]





# In[264]:



garage[garage.GarageCars.isnull()]





# We will make them zero. We did not do it earlier because there were some cases when garageYrBlt was Null but Garage Cars where not null. Probably they refered to detached garage type. Lets have a look



# In[265]:



ind = garage.GarageYrBlt==0





# In[266]:



(garage.GarageCars[ind]!=0).sum()





# In[267]:



(garage.GarageArea[ind]!=0).sum()





# Okay so these are just two cases, we will make them all zero.



# In[268]:



all_data.loc[all_data.GarageCars.isnull(), ('GarageCars', 'GarageArea')]=0



all_data = pd.get_dummies(all_data)
# Data Frames

X_train = all_data[:trainDF.shape[0]] #We are using raw selector operator

X_test = all_data[trainDF.shape[0]:]

yOriginal = trainDF.SalePrice

y = np.log(yOriginal)


from math import log
i = [1 for j in range(X_train.shape[0])]

i = np.array(i)

i = np.expand_dims(i, axis=1)

np_X_train = np.hstack((i, X_train.as_matrix()))



i = [1 for j in range(X_test.shape[0])]

i = np.array(i)

i = np.expand_dims(i, axis=1)

np_X_test = np.hstack((i, X_test.as_matrix()))
np_y = y.values
from numpy import linalg
I = np.identity(np_X_train.shape[1])

temp0 = np_X_train.T.dot(np_X_train) + 15*I

temp1 = linalg.inv(temp0)

parameter = temp1.dot(np_X_train.T).dot(np_y)

Ypred = np_X_test.dot(parameter)
y2 = np.exp(Ypred)
#Sample submission

submission = pd.DataFrame({ 'Id': testDF['Id'],

                           'SalePrice': y2 })

submission.to_csv("fnal_with_missing.csv", index=False)
y2