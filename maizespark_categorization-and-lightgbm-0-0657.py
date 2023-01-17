# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# data analysis and wrangling
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
df2016 = pd.read_csv('../input/train_2016_v2.csv')
df2016['merger']=df2016.parcelid*10000+2016
df2017 = pd.read_csv('../input/train_2017.csv')
df2017['merger']=df2016.parcelid*10000+2017
df=pd.concat([df2016,df2017])
del df2016
del df2017
prop2016 = pd.read_csv('../input/properties_2016.csv')
prop2016['merger']=prop2016.parcelid*10000+2016
prop2017 = pd.read_csv('../input/properties_2017.csv')
prop2017['merger']=prop2016.parcelid*10000+2017
prop=pd.concat([prop2016,prop2017])
del prop2016
del prop2017
sample = pd.read_csv('../input/sample_submission.csv')
sample=sample.rename(index=str,columns={'ParcelId':'parcelid'})
dates=sample.columns
test=sample[['parcelid','201611']]
test['year']=2016
test2=test[['parcelid','year']]
test2.year=2017
del test['201611']
test=pd.concat([test,test2])
del test2
test['merger']=test.parcelid*10000+test.year
test.year=test.year-2016
print(df.info())
print(prop.info())
df = df.merge(prop, how='left', on='merger')
test=test.merge(prop, how='left', on='merger')
del prop
df['parcelid']=df.merger/10000
test['parcelid']=test.merger/10000
del df['merger']
del test['merger']
del df['parcelid_x']
del df['parcelid_y']
del test['parcelid_x']
del test['parcelid_y']
for c, dtype in zip(df.columns, df.dtypes):
    if dtype == np.float64:
        df[c] = df[c].astype(np.float32) 
    elif dtype == np.int64:
        df[c] = df[c].astype(np.int32)
for c, dtype in zip(test.columns, test.dtypes):
    if dtype == np.float64:
        test[c] = test[c].astype(np.float32) 
    elif dtype == np.int64:
        test[c] = test[c].astype(np.int32) 
df.describe().transpose()
df.head().transpose()
#air conditioning type 3 and 5 have only one sample, so we may need to absorb them to type 1. Correlation is very low, but it simply means that 
#increasing type id doesn't mean greater logerror. They are still believed to be related. There is a significant difference of error whether
#this data is NaN or not. Therefore we will keep NaN as separate category. Also, because typeid 5 and typeid 13 have similar error with
#significant size, I would choose to merge 13 to 5. In this way, this feature will be proportional (exponential maybe) to logerror.
df.airconditioningtypeid=df.airconditioningtypeid.fillna(-1)
df.airconditioningtypeid[df.airconditioningtypeid==3.0]=1.0
df.airconditioningtypeid[df.airconditioningtypeid==9.0]=1.0
df.airconditioningtypeid[df.airconditioningtypeid==13.0]=5.0

test.airconditioningtypeid=test.airconditioningtypeid.fillna(-1)
test.airconditioningtypeid[test.airconditioningtypeid==3.0]=1.0
test.airconditioningtypeid[test.airconditioningtypeid==9.0]=1.0
test.airconditioningtypeid[test.airconditioningtypeid==13.0]=5.0

print(df.logerror[df.airconditioningtypeid==-1].mean(),df.logerror[df.airconditioningtypeid!=-1].mean())
df[['airconditioningtypeid','logerror']].groupby(['airconditioningtypeid'], as_index=False)['logerror'].agg(['mean','count'])

#architecture type 3, 10, and 21 has insignificant samples. I keep NaN as separate category because being NaN leads to smaller error.
#There are less than 300 data points for this feature, and vast majority has value 7. The rest types have very small number of data points.
#Therefore, rather than using typeid, I would rather use if the house has architecture type info or not.
df.architecturalstyletypeid=df.architecturalstyletypeid.fillna(-1)
df.architecturalstyletypeid[df.architecturalstyletypeid!=-1]=1
test.architecturalstyletypeid=test.architecturalstyletypeid.fillna(-1)
test.architecturalstyletypeid[test.architecturalstyletypeid!=-1]=1
print(df.logerror[df.architecturalstyletypeid==-1].mean(),df.logerror[df.architecturalstyletypeid!=-1].mean())

df[['architecturalstyletypeid','logerror']].groupby(['architecturalstyletypeid'], as_index=False)['logerror'].agg(['mean','count'])
#This data is meaningful even though data is only available for 43 houses. When this data exists, error gets reduced significantly. Also, this
#data has high positive correlation with the error when it's there. As the error for null is about one sixth of the error for nonnull, I am
#comfortable with giving -1 to null.
print(df.logerror[df.basementsqft.isnull()].mean(),df.logerror[df.basementsqft.notnull()].mean())
print(df.logerror[df.basementsqft.isnull()].count(),df.logerror[df.basementsqft.notnull()].count())
print(np.corrcoef(df.basementsqft[df.basementsqft.notnull()],df.logerror[df.basementsqft.notnull()]))
df.basementsqft=df.basementsqft.fillna(-1)
test.basementsqft=test.basementsqft.fillna(-1)
#This is nonull feasure. As you can imagine the number of bathrooms is proportional to the size of house. Therefore, number of bathroom is
#supposed to be proportional to the error. There are some bathroom counts which have very few samples. I would say the data is not significant
#when there are less than 50 samples. Therefore, I chose to make 7.5 to 8 and 8.5 and 8.5+ to 9.
print(df.logerror[df.bathroomcnt.isnull()].count(),df.logerror[df.bathroomcnt.notnull()].count())
print(np.corrcoef(df.bathroomcnt[df.bathroomcnt.notnull()],df.logerror[df.bathroomcnt.notnull()]))
print(df[['bathroomcnt','logerror']].groupby(['bathroomcnt'], as_index=False)['logerror'].agg(['mean','count']))
df.bathroomcnt[df.bathroomcnt==7.5]=8
df.bathroomcnt[df.bathroomcnt>=8.5]=9
test.bathroomcnt[test.bathroomcnt==7.5]=8
test.bathroomcnt[test.bathroomcnt>=8.5]=9
df[['bathroomcnt','logerror']].groupby(['bathroomcnt'], as_index=False)['logerror'].agg(['mean','count'])
#Bedroom works in a similar way with bathroom. Therefore, I approached bedroom in a similar way.
print(df.logerror[df.bedroomcnt.isnull()].count(),df.logerror[df.bedroomcnt.notnull()].count())
print(np.corrcoef(df.bedroomcnt[df.bedroomcnt.notnull()],df.logerror[df.bedroomcnt.notnull()]))
print(df[['bedroomcnt','logerror']].groupby(['bedroomcnt'], as_index=False)['logerror'].agg(['mean','count']))
df.bedroomcnt[df.bedroomcnt>9]=10
test.bedroomcnt[test.bedroomcnt>9]=10
df[['bedroomcnt','logerror']].groupby(['bedroomcnt'], as_index=False)['logerror'].agg(['mean','count'])
#There are significant amount of nonull data. Additionally, error for null data is almost as same as notnull data. Type 6,8,11 are insignificant
#Therefore, this will be treated as nan.
print(df.logerror[df.buildingqualitytypeid.isnull()].mean(),df.logerror[df.buildingqualitytypeid.notnull()].mean())
print(df.logerror[df.buildingqualitytypeid.isnull()].count(),df.logerror[df.buildingqualitytypeid.notnull()].count())
print(np.corrcoef(df.buildingqualitytypeid[df.buildingqualitytypeid.notnull()],df.logerror[df.buildingqualitytypeid.notnull()]))
df.buildingqualitytypeid=df.buildingqualitytypeid.fillna(-1)
df.buildingqualitytypeid[df.buildingqualitytypeid==6.0]=-1
df.buildingqualitytypeid[df.buildingqualitytypeid==8.0]=-1
df.buildingqualitytypeid[df.buildingqualitytypeid==11.0]=-1
test.buildingqualitytypeid=test.buildingqualitytypeid.fillna(-1)
test.buildingqualitytypeid[test.buildingqualitytypeid==6.0]=-1
test.buildingqualitytypeid[test.buildingqualitytypeid==8.0]=-1
test.buildingqualitytypeid[test.buildingqualitytypeid==11.0]=-1
df[['buildingqualitytypeid','logerror']].groupby(['buildingqualitytypeid'], as_index=False)['logerror'].agg(['mean','count'])
#This data is hard to be determined. Only 16 data points, and they are only one type. I may not use this data at all.
print(df.logerror[df.buildingclasstypeid.isnull()].mean(),df.logerror[df.buildingclasstypeid.notnull()].mean())
print(df.logerror[df.buildingclasstypeid.isnull()].count(),df.logerror[df.buildingclasstypeid.notnull()].count())
print(np.corrcoef(df.buildingclasstypeid[df.buildingclasstypeid.notnull()],df.logerror[df.buildingclasstypeid.notnull()]))
print(df[['buildingclasstypeid','logerror']].groupby(['buildingclasstypeid'], as_index=False)['logerror'].agg(['mean','count']))
del df['buildingclasstypeid']
del test['buildingclasstypeid']
#calculatebathnbr is duplicate data as bathroomcnt. This data won't be used.
print(df.logerror[df.calculatedbathnbr.isnull()].count(),df.logerror[df.calculatedbathnbr.notnull()].count())
print(np.corrcoef(df.calculatedbathnbr[df.calculatedbathnbr.notnull()],df.logerror[df.calculatedbathnbr.notnull()]))
print(df[['calculatedbathnbr','logerror']].groupby(['calculatedbathnbr'], as_index=False)['logerror'].agg(['mean','count']))
del df['calculatedbathnbr']
del test['calculatedbathnbr']
#decktypeid is in a similar situation with buildingclasstypeid.
print(df.logerror[df.decktypeid.isnull()].count(),df.logerror[df.decktypeid.notnull()].count())
print(np.corrcoef(df.decktypeid[df.decktypeid.notnull()],df.logerror[df.decktypeid.notnull()]))
print(df[['decktypeid','logerror']].groupby(['decktypeid'], as_index=False)['logerror'].agg(['mean','count']))
del df['decktypeid']
del test['decktypeid']
#This data is a little different from other bathroom data. Around 12k data points available, but majority are 1. Additionally, error from null
#is slightly lower than notnull. I am going to convert 3 and 4 to 2.
print(df.logerror[df.threequarterbathnbr.isnull()].mean(),df.logerror[df.threequarterbathnbr.notnull()].mean())
print(df.logerror[df.threequarterbathnbr.isnull()].count(),df.logerror[df.threequarterbathnbr.notnull()].count())
print(np.corrcoef(df.threequarterbathnbr[df.threequarterbathnbr.notnull()],df.logerror[df.threequarterbathnbr.notnull()]))
print(df[['threequarterbathnbr','logerror']].groupby(['threequarterbathnbr'], as_index=False)['logerror'].agg(['mean','count']))
df.threequarterbathnbr=df.threequarterbathnbr.fillna(-1)
df.threequarterbathnbr[df.threequarterbathnbr==3.0]=2.0
df.threequarterbathnbr[df.threequarterbathnbr==4.0]=2.0
test.threequarterbathnbr=test.threequarterbathnbr.fillna(-1)
test.threequarterbathnbr[test.threequarterbathnbr==3.0]=2.0
test.threequarterbathnbr[test.threequarterbathnbr==4.0]=2.0
df[['threequarterbathnbr','logerror']].groupby(['threequarterbathnbr'], as_index=False)['logerror'].agg(['mean','count'])
#1st floor size of course is proportional to error. As there are so many unique values for this measure, I chose to use intervals for this
#feature. Additionally, I changed group 6,7,8,9 to 5.
print(df.logerror[df.finishedfloor1squarefeet.isnull()].mean(),df.logerror[df.finishedfloor1squarefeet.notnull()].mean())
print(df.logerror[df.finishedfloor1squarefeet.isnull()].count(),df.logerror[df.finishedfloor1squarefeet.notnull()].count())
print(np.corrcoef(df.finishedfloor1squarefeet[df.finishedfloor1squarefeet.notnull()],df.logerror[df.finishedfloor1squarefeet.notnull()]))
df['first_floor']=pd.cut(df.finishedfloor1squarefeet, 10).cat.codes
test['first_floor']=pd.cut(df.finishedfloor1squarefeet, 10).cat.codes
print(df[['first_floor','logerror']].groupby(['first_floor'], as_index=False)['logerror'].agg(['mean','count']))
df.first_floor[df.first_floor>5]=5
test.first_floor[test.first_floor>5]=5
df[['first_floor','logerror']].groupby(['first_floor'], as_index=False)['logerror'].agg(['mean','count'])
del df['finishedfloor1squarefeet']
del test['finishedfloor1squarefeet']
#this is the size of the whole house. It looks like we can apply the samething we did for the first floor size.
print(df.logerror[df.calculatedfinishedsquarefeet.isnull()].mean(),df.logerror[df.calculatedfinishedsquarefeet.notnull()].mean())
print(df.logerror[df.calculatedfinishedsquarefeet.isnull()].count(),df.logerror[df.calculatedfinishedsquarefeet.notnull()].count())
print(np.corrcoef(df.calculatedfinishedsquarefeet[df.calculatedfinishedsquarefeet.notnull()],df.logerror[df.calculatedfinishedsquarefeet.notnull()]))
df['finished_sqr']=pd.cut(df.calculatedfinishedsquarefeet, 10).cat.codes
test['finished_sqr']=pd.cut(df.calculatedfinishedsquarefeet, 10).cat.codes
print(df[['finished_sqr','logerror']].groupby(['finished_sqr'], as_index=False)['logerror'].agg(['mean','count']))
df.finished_sqr[df.finished_sqr>4]=4
test.finished_sqr[test.finished_sqr>4]=4
df[['finished_sqr','logerror']].groupby(['finished_sqr'], as_index=False)['logerror'].agg(['mean','count'])
del df['calculatedfinishedsquarefeet']
del test['calculatedfinishedsquarefeet']
#when the total area is known, error reduces a lot. Similar approach with other size features.
print(df.logerror[df.finishedsquarefeet6.isnull()].mean(),df.logerror[df.finishedsquarefeet6.notnull()].mean())
print(df.logerror[df.finishedsquarefeet6.isnull()].count(),df.logerror[df.finishedsquarefeet6.notnull()].count())
print(np.corrcoef(df.finishedsquarefeet6[df.finishedsquarefeet6.notnull()],df.logerror[df.finishedsquarefeet6.notnull()]))
df['finished_sqr6']=pd.cut(df.finishedsquarefeet6, 10).cat.codes
test['finished_sqr6']=pd.cut(df.finishedsquarefeet6, 10).cat.codes
print(df[['finished_sqr6','logerror']].groupby(['finished_sqr6'], as_index=False)['logerror'].agg(['mean','count']))
df.finished_sqr6[df.finished_sqr6>6]=6
test.finished_sqr6[test.finished_sqr6>6]=6
df[['finished_sqr6','logerror']].groupby(['finished_sqr6'], as_index=False)['logerror'].agg(['mean','count'])
del df['finishedsquarefeet6']
del test['finishedsquarefeet6']
#when the total area is known, error reduces a lot. Similar approach with other size features.
print(df.logerror[df.finishedsquarefeet12.isnull()].mean(),df.logerror[df.finishedsquarefeet12.notnull()].mean())
print(df.logerror[df.finishedsquarefeet12.isnull()].count(),df.logerror[df.finishedsquarefeet12.notnull()].count())
print(np.corrcoef(df.finishedsquarefeet12[df.finishedsquarefeet12.notnull()],df.logerror[df.finishedsquarefeet12.notnull()]))
df['finished_sqr12']=pd.cut(df.finishedsquarefeet12, 10).cat.codes
test['finished_sqr12']=pd.cut(df.finishedsquarefeet12, 10).cat.codes
print(df[['finished_sqr12','logerror']].groupby(['finished_sqr12'], as_index=False)['logerror'].agg(['mean','count']))
df.finished_sqr12[df.finished_sqr12>5]=5
test.finished_sqr12[test.finished_sqr12>5]=5
df[['finished_sqr12','logerror']].groupby(['finished_sqr12'], as_index=False)['logerror'].agg(['mean','count'])
del df['finishedsquarefeet12']
del test['finishedsquarefeet12']
#Only 33 notnull values, and there are 17 data points for 1440ft2. Therefore, I categorized by less than 1440, equal to 1440, larger than 1440,
#and null.
print(df.logerror[df.finishedsquarefeet13.isnull()].mean(),df.logerror[df.finishedsquarefeet13.notnull()].mean())
print(df.logerror[df.finishedsquarefeet13.isnull()].count(),df.logerror[df.finishedsquarefeet13.notnull()].count())
print(np.corrcoef(df.finishedsquarefeet13[df.finishedsquarefeet13.notnull()],df.logerror[df.finishedsquarefeet13.notnull()]))
df.finishedsquarefeet13[df.finishedsquarefeet13<1440.0]=0
df.finishedsquarefeet13[df.finishedsquarefeet13==1440.0]=1
df.finishedsquarefeet13[df.finishedsquarefeet13>1440.0]=2
df.finishedsquarefeet13=df.finishedsquarefeet13.fillna(-1)
test.finishedsquarefeet13[test.finishedsquarefeet13<1440.0]=0
test.finishedsquarefeet13[test.finishedsquarefeet13==1440.0]=1
test.finishedsquarefeet13[test.finishedsquarefeet13>1440.0]=2
test.finishedsquarefeet13=test.finishedsquarefeet13.fillna(-1)
df[['finishedsquarefeet13','logerror']].groupby(['finishedsquarefeet13'], as_index=False)['logerror'].agg(['mean','count'])
#when the total area is known, error reduces a lot. Similar approach with other size features.
print(df.logerror[df.finishedsquarefeet15.isnull()].mean(),df.logerror[df.finishedsquarefeet15.notnull()].mean())
print(df.logerror[df.finishedsquarefeet15.isnull()].count(),df.logerror[df.finishedsquarefeet15.notnull()].count())
print(np.corrcoef(df.finishedsquarefeet15[df.finishedsquarefeet15.notnull()],df.logerror[df.finishedsquarefeet15.notnull()]))
print(np.min(df.finishedsquarefeet15))
print(np.max(df.finishedsquarefeet15))
df.finishedsquarefeet15[df.finishedsquarefeet15==np.max(df.finishedsquarefeet15)]=10000.0
df['finished_sqr15']=pd.cut(df.finishedsquarefeet15, 10).cat.codes
test.finishedsquarefeet15[test.finishedsquarefeet15==np.max(test.finishedsquarefeet15)]=10000.0
test['finished_sqr15']=pd.cut(df.finishedsquarefeet15, 10).cat.codes
print(df[['finished_sqr15','logerror']].groupby(['finished_sqr15'], as_index=False)['logerror'].agg(['mean','count']))
df.finished_sqr15[df.finished_sqr15>5]=5
test.finished_sqr15[test.finished_sqr15>5]=5
df[['finished_sqr15','logerror']].groupby(['finished_sqr15'], as_index=False)['logerror'].agg(['mean','count'])
del df['finishedsquarefeet15']
del test['finishedsquarefeet15']
#when the total area is known, error reduces a lot. Similar approach with other size features.
print(df.logerror[df.finishedsquarefeet50.isnull()].mean(),df.logerror[df.finishedsquarefeet50.notnull()].mean())
print(df.logerror[df.finishedsquarefeet50.isnull()].count(),df.logerror[df.finishedsquarefeet50.notnull()].count())
print(np.corrcoef(df.finishedsquarefeet50[df.finishedsquarefeet50.notnull()],df.logerror[df.finishedsquarefeet50.notnull()]))
print(np.min(df.finishedsquarefeet50))
print(np.max(df.finishedsquarefeet50))
df['finished_sqr50']=pd.cut(df.finishedsquarefeet50, 10).cat.codes
test['finished_sqr50']=pd.cut(df.finishedsquarefeet50, 10).cat.codes
print(df[['finished_sqr50','logerror']].groupby(['finished_sqr50'], as_index=False)['logerror'].agg(['mean','count']))
df.finished_sqr50[df.finished_sqr50>5]=5
test.finished_sqr50[test.finished_sqr50>5]=5
df[['finished_sqr50','logerror']].groupby(['finished_sqr50'], as_index=False)['logerror'].agg(['mean','count'])
del df['finishedsquarefeet50']
del test['finishedsquarefeet50']
#There's no null for fips. We do not know specific meaning of each code, so we just categorize it.
print(df.logerror[df.fips.isnull()].mean(),df.logerror[df.fips.notnull()].mean())
print(df.logerror[df.fips.isnull()].count(),df.logerror[df.fips.notnull()].count())
print(np.corrcoef(df.fips[df.fips.notnull()],df.logerror[df.fips.notnull()]))
df[['fips','logerror']].groupby(['fips'], as_index=False)['logerror'].agg(['mean','count'])
#number of fireplace is proportional to error. This measure is another indirect measure on size.
print(df.logerror[df.fireplacecnt.isnull()].mean(),df.logerror[df.fireplacecnt.notnull()].mean())
print(df.logerror[df.fireplacecnt.isnull()].count(),df.logerror[df.fireplacecnt.notnull()].count())
print(np.corrcoef(df.fireplacecnt[df.fireplacecnt.notnull()],df.logerror[df.fireplacecnt.notnull()]))
print(df[['fireplacecnt','logerror']].groupby(['fireplacecnt'], as_index=False)['logerror'].agg(['mean','count']))
df.fireplacecnt[df.fireplacecnt>3]=3
test.fireplacecnt[test.fireplacecnt>3]=3
df[['fireplacecnt','logerror']].groupby(['fireplacecnt'], as_index=False)['logerror'].agg(['mean','count'])
#This data is not giving good enough information. It only gives true when fire place is there. It means house without fireplace and no info are
#treated same. Another strange thing is that fireplacecnt gave info for more than 10k datapoints. I would rather choose not to use this data.
print(df.logerror[df.fireplaceflag.isnull()].mean(),df.logerror[df.fireplaceflag.notnull()].mean())
print(df.logerror[df.fireplaceflag.isnull()].count(),df.logerror[df.fireplaceflag.notnull()].count())
print(df[['fireplaceflag','logerror']].groupby(['fireplaceflag'], as_index=False)['logerror'].agg(['mean','count']))
del df['fireplaceflag']
del test['fireplaceflag']
#this is a good data with only few null data. Similar approach to other bathroom or room measure. Merged 8+ full baths into 8 baths.
print(df.logerror[df.fullbathcnt.isnull()].mean(),df.logerror[df.fullbathcnt.notnull()].mean())
print(df.logerror[df.fullbathcnt.isnull()].count(),df.logerror[df.fullbathcnt.notnull()].count())
print(np.corrcoef(df.fullbathcnt[df.fullbathcnt.notnull()],df.logerror[df.fullbathcnt.notnull()]))
print(df[['fullbathcnt','logerror']].groupby(['fullbathcnt'], as_index=False)['logerror'].agg(['mean','count']))
df.fullbathcnt[df.fullbathcnt>8]=8
test.fullbathcnt[test.fullbathcnt>8]=8
df[['fullbathcnt','logerror']].groupby(['fullbathcnt'], as_index=False)['logerror'].agg(['mean','count'])
#about 1/3 datapoints are notnull. Merged 6+ garages to 6 garages.
print(df.logerror[df.garagecarcnt.isnull()].mean(),df.logerror[df.garagecarcnt.notnull()].mean())
print(df.logerror[df.garagecarcnt.isnull()].count(),df.logerror[df.garagecarcnt.notnull()].count())
print(np.corrcoef(df.garagecarcnt[df.garagecarcnt.notnull()],df.logerror[df.garagecarcnt.notnull()]))
print(df[['garagecarcnt','logerror']].groupby(['garagecarcnt'], as_index=False)['logerror'].agg(['mean','count']))
df.garagecarcnt[df.garagecarcnt>6]=6
test.garagecarcnt[test.garagecarcnt>6]=6
df[['garagecarcnt','logerror']].groupby(['garagecarcnt'], as_index=False)['logerror'].agg(['mean','count'])
#when the total area is known, error reduces a lot. Similar approach with other size features.
print(df.logerror[df.garagetotalsqft.isnull()].mean(),df.logerror[df.garagetotalsqft.notnull()].mean())
print(df.logerror[df.garagetotalsqft.isnull()].count(),df.logerror[df.garagetotalsqft.notnull()].count())
print(np.corrcoef(df.garagetotalsqft[df.garagetotalsqft.notnull()],df.logerror[df.garagetotalsqft.notnull()]))
df.garagetotalsqft[df.garagetotalsqft==np.max(df.garagetotalsqft)]=4500
df['garage_size']=pd.cut(df.garagetotalsqft, 10).cat.codes
test.garagetotalsqft[test.garagetotalsqft==np.max(test.garagetotalsqft)]=4500
test['garage_size']=pd.cut(df.garagetotalsqft, 10).cat.codes
print(df[['garage_size','logerror']].groupby(['garage_size'], as_index=False)['logerror'].agg(['mean','count']))
df.garage_size[df.garage_size>5]=5
test.garage_size[test.garage_size>5]=5
print(df[['garage_size','logerror']].groupby(['garage_size'], as_index=False)['logerror'].agg(['mean','count']))
del df['garagetotalsqft']
del test['garagetotalsqft']
#Even though there are only few houses with hottub or spa, error is meaningfully low when it is true.
print(df.logerror[df.hashottuborspa.isnull()].mean(),df.logerror[df.hashottuborspa.notnull()].mean())
print(df.logerror[df.hashottuborspa.isnull()].count(),df.logerror[df.hashottuborspa.notnull()].count())
print(df[['hashottuborspa','logerror']].groupby(['hashottuborspa'], as_index=False)['logerror'].agg(['mean','count']))
df.hashottuborspa=df.hashottuborspa*1
test.hashottuborspa=test.hashottuborspa*1
df.hashottuborspa=df.hashottuborspa.fillna(-1)
test.hashottuborspa=test.hashottuborspa.fillna(-1)
df[['hashottuborspa','logerror']].groupby(['hashottuborspa'], as_index=False)['logerror'].agg(['mean','count'])
#heating system has decent amount of data. I chose to merge minority types into one
print(df.logerror[df.heatingorsystemtypeid.isnull()].mean(),df.logerror[df.heatingorsystemtypeid.notnull()].mean())
print(df.logerror[df.heatingorsystemtypeid.isnull()].count(),df.logerror[df.heatingorsystemtypeid.notnull()].count())
print(df[['heatingorsystemtypeid','logerror']].groupby(['heatingorsystemtypeid'], as_index=False)['logerror'].agg(['mean','count']))
df.heatingorsystemtypeid[df.heatingorsystemtypeid==11]=10
df.heatingorsystemtypeid[df.heatingorsystemtypeid==12]=10
df.heatingorsystemtypeid[df.heatingorsystemtypeid==14]=10
test.heatingorsystemtypeid[test.heatingorsystemtypeid==11]=10
test.heatingorsystemtypeid[test.heatingorsystemtypeid==12]=10
test.heatingorsystemtypeid[test.heatingorsystemtypeid==14]=10
df[['heatingorsystemtypeid','logerror']].groupby(['heatingorsystemtypeid'], as_index=False)['logerror'].agg(['mean','count'])
#latitude may not be proportional to the result, but it may mean something significant.
print(df.logerror[df.latitude.isnull()].mean(),df.logerror[df.latitude.notnull()].mean())
print(df.logerror[df.latitude.isnull()].count(),df.logerror[df.latitude.notnull()].count())
print(np.corrcoef(df.latitude[df.latitude.notnull()],df.logerror[df.latitude.notnull()]))
#df.latitude[df.latitude==np.max(df.latitude)]=4500
df.latitude=pd.cut(df.latitude, 10).cat.codes
test.latitude=pd.cut(df.latitude, 10).cat.codes
print(df[['latitude','logerror']].groupby(['latitude'], as_index=False)['logerror'].agg(['mean','count']))
#df.latitude[df.latitude>5]=5
#print(df[['latitude','logerror']].groupby(['latitude'], as_index=False)['logerror'].agg(['mean','count']))
#longitude is also straightforward
print(df.logerror[df.longitude.isnull()].mean(),df.logerror[df.longitude.notnull()].mean())
print(df.logerror[df.longitude.isnull()].count(),df.logerror[df.longitude.notnull()].count())
print(np.corrcoef(df.logerror[df.longitude.notnull()],df.longitude[df.longitude.notnull()]))
#df.latitude[df.latitude==np.max(df.latitude)]=4500
df.longitude=pd.cut(df.longitude, 10).cat.codes
test.longitude=pd.cut(df.longitude, 10).cat.codes
print(df[['longitude','logerror']].groupby(['longitude'], as_index=False)['logerror'].agg(['mean','count']))
#df.latitude[df.latitude>5]=5
#print(df[['latitude','logerror']].groupby(['latitude'], as_index=False)['logerror'].agg(['mean','count']))
#It looks like lot size doesn't have good relationship with the error. I may choose not to use this feature.
print(df.logerror[df.lotsizesquarefeet.isnull()].mean(),df.logerror[df.lotsizesquarefeet.notnull()].mean())
print(df.logerror[df.lotsizesquarefeet.isnull()].count(),df.logerror[df.lotsizesquarefeet.notnull()].count())
print(np.corrcoef(df.logerror[df.lotsizesquarefeet.notnull()],df.longitude[df.lotsizesquarefeet.notnull()]))
print(df[['lotsizesquarefeet','logerror']].groupby(['lotsizesquarefeet'], as_index=False)['logerror'].agg(['mean','count']))
del df['lotsizesquarefeet']
del test['lotsizesquarefeet']
#of course number of stories is a significant factor. What's strange is that correlation is negative while it looks like positive based on pivot.
print(df.logerror[df.numberofstories.isnull()].mean(),df.logerror[df.numberofstories.notnull()].mean())
print(df.logerror[df.numberofstories.isnull()].count(),df.logerror[df.numberofstories.notnull()].count())
print(np.corrcoef(df.logerror[df.numberofstories.notnull()],df.longitude[df.numberofstories.notnull()]))
print(df[['numberofstories','logerror']].groupby(['numberofstories'], as_index=False)['logerror'].agg(['mean','count']))
df.numberofstories=df.numberofstories.fillna(-1)
test.numberofstories=test.numberofstories.fillna(-1)
df.numberofstories[df.numberofstories>3]=3
test.numberofstories[test.numberofstories>3]=3
df[['numberofstories','logerror']].groupby(['numberofstories'], as_index=False)['logerror'].agg(['mean','count'])
#No house has multiple pools. This measure is more about true or false.
print(df.logerror[df.poolcnt.isnull()].mean(),df.logerror[df.poolcnt.notnull()].mean())
print(df.logerror[df.poolcnt.isnull()].count(),df.logerror[df.poolcnt.notnull()].count())
print(np.corrcoef(df.poolcnt[df.poolcnt.notnull()],df.logerror[df.poolcnt.notnull()]))
print(df[['poolcnt','logerror']].groupby(['poolcnt'], as_index=False)['logerror'].agg(['mean','count']))
df.poolcnt=pd.Categorical(df.poolcnt)
df.poolcnt=df.poolcnt.cat.codes
test.poolcnt=pd.Categorical(test.poolcnt)
test.poolcnt=test.poolcnt.cat.codes
df[['poolcnt','logerror']].groupby(['poolcnt'], as_index=False)['logerror'].agg(['mean','count'])
#different pool size gives different errors. 
print(df.logerror[df.poolsizesum.isnull()].mean(),df.logerror[df.poolsizesum.notnull()].mean())
print(df.logerror[df.poolsizesum.isnull()].count(),df.logerror[df.poolsizesum.notnull()].count())
print(np.corrcoef(df.logerror[df.poolsizesum.notnull()],df.poolsizesum[df.poolsizesum.notnull()]))
df.poolsizesum=pd.cut(df.poolsizesum, 10).cat.codes
test.poolsizesum=pd.cut(df.poolsizesum, 10).cat.codes
print(df[['poolsizesum','logerror']].groupby(['poolsizesum'], as_index=False)['logerror'].agg(['mean','count']))
df.poolsizesum[df.poolsizesum>4]=4
df.poolsizesum[df.poolsizesum==0]=1
test.poolsizesum[test.poolsizesum>4]=4
test.poolsizesum[test.poolsizesum==0]=1
print(df[['poolsizesum','logerror']].groupby(['poolsizesum'], as_index=False)['logerror'].agg(['mean','count']))
#I do not know what's really different between this feature and hottuborspa feature, but result is different, so I can't really remove this.
print(df.logerror[df.pooltypeid10.isnull()].mean(),df.logerror[df.pooltypeid10.notnull()].mean())
print(df.logerror[df.pooltypeid10.isnull()].count(),df.logerror[df.pooltypeid10.notnull()].count())
print(np.corrcoef(df.pooltypeid10[df.pooltypeid10.notnull()],df.logerror[df.pooltypeid10.notnull()]))
print(df[['pooltypeid10','logerror']].groupby(['pooltypeid10'], as_index=False)['logerror'].agg(['mean','count']))
df.pooltypeid10=df.pooltypeid10.fillna(-1)
test.pooltypeid10=test.pooltypeid10.fillna(-1)
df[['pooltypeid10','logerror']].groupby(['pooltypeid10'], as_index=False)['logerror'].agg(['mean','count'])
#this measure is true or null
print(df.logerror[df.pooltypeid2.isnull()].mean(),df.logerror[df.pooltypeid2.notnull()].mean())
print(df.logerror[df.pooltypeid2.isnull()].count(),df.logerror[df.pooltypeid2.notnull()].count())
print(np.corrcoef(df.pooltypeid2[df.pooltypeid2.notnull()],df.logerror[df.pooltypeid2.notnull()]))
print(df[['pooltypeid2','logerror']].groupby(['pooltypeid2'], as_index=False)['logerror'].agg(['mean','count']))
df.pooltypeid2=df.pooltypeid2.fillna(-1)
test.pooltypeid2=test.pooltypeid2.fillna(-1)
df[['pooltypeid2','logerror']].groupby(['pooltypeid2'], as_index=False)['logerror'].agg(['mean','count'])
#it's true or null
print(df.logerror[df.pooltypeid7.isnull()].mean(),df.logerror[df.pooltypeid7.notnull()].mean())
print(df.logerror[df.pooltypeid7.isnull()].count(),df.logerror[df.pooltypeid7.notnull()].count())
print(np.corrcoef(df.pooltypeid7[df.pooltypeid7.notnull()],df.logerror[df.pooltypeid7.notnull()]))
print(df[['pooltypeid7','logerror']].groupby(['pooltypeid7'], as_index=False)['logerror'].agg(['mean','count']))
df.pooltypeid7=df.pooltypeid7.fillna(-1)
test.pooltypeid7=test.pooltypeid7.fillna(-1)
df[['pooltypeid7','logerror']].groupby(['pooltypeid7'], as_index=False)['logerror'].agg(['mean','count'])
#This code has so many different types. I only considered the ones with more than 20 houses of the same code. Otherwise, I treated
#it as equivalent to null.
print(df.logerror[df.propertycountylandusecode.isnull()].mean(),df.logerror[df.propertycountylandusecode.notnull()].mean())
print(df.logerror[df.propertycountylandusecode.isnull()].count(),df.logerror[df.propertycountylandusecode.notnull()].count())
#print(np.corrcoef(df.propertycountylandusecode[df.propertycountylandusecode.notnull()],df.logerror[df.propertycountylandusecode.notnull()]))
print(df[['propertycountylandusecode','logerror']].groupby(['propertycountylandusecode'], as_index=False)['logerror'].agg(['mean','count']))
pivot=df[['propertycountylandusecode','logerror']].groupby(['propertycountylandusecode'], as_index=True)['logerror'].agg(['mean','count'])
pivot['codes']=np.arange(len(pivot))
code_dict=pivot.codes.to_dict()
sigs=pivot.index[pivot['count']>20].tolist()
df.propertycountylandusecode=df.propertycountylandusecode.apply(lambda x: x if x in sigs else -1)
test.propertycountylandusecode=test.propertycountylandusecode.apply(lambda x: x if x in sigs else -1)
df['propertycountylandusecode']=df.propertycountylandusecode.map(code_dict)
test['propertycountylandusecode']=test.propertycountylandusecode.map(code_dict)
df[['propertycountylandusecode','logerror']].groupby(['propertycountylandusecode'], as_index=True)['logerror'].agg(['mean','count'])

#This feature has so many different types. I only considered the ones with more than 50 houses of the same code. Otherwise, I 
#treated it as equivalent to null.
print(df.logerror[df.propertylandusetypeid.isnull()].mean(),df.logerror[df.propertylandusetypeid.notnull()].mean())
print(df.logerror[df.propertylandusetypeid.isnull()].count(),df.logerror[df.propertylandusetypeid.notnull()].count())
#print(np.corrcoef(df.propertycountylandusecode[df.propertycountylandusecode.notnull()],df.logerror[df.propertycountylandusecode.notnull()]))
print(df[['propertylandusetypeid','logerror']].groupby(['propertylandusetypeid'], as_index=False)['logerror'].agg(['mean','count']))
pivot=df[['propertylandusetypeid','logerror']].groupby(['propertylandusetypeid'], as_index=True)['logerror'].agg(['mean','count'])
pivot['codes']=np.arange(len(pivot))
code_dict=pivot.codes.to_dict()
sigs=pivot.index[pivot['count']>50].tolist()
df.propertylandusetypeid=df.propertylandusetypeid.apply(lambda x: x if x in sigs else -1)
test.propertylandusetypeid=test.propertylandusetypeid.apply(lambda x: x if x in sigs else -1)
df['propertylandusetypeid']=df.propertylandusetypeid.map(code_dict)
test['propertylandusetypeid']=test.propertylandusetypeid.map(code_dict)
df[['propertylandusetypeid','logerror']].groupby(['propertylandusetypeid'], as_index=True)['logerror'].agg(['mean','count'])

#propertyzoningdesc has some meaningful relationship with error. There are a few of them with significant difference from total
#average with decent sample size.
print(df.logerror[df.propertyzoningdesc.isnull()].mean(),df.logerror[df.propertyzoningdesc.notnull()].mean())
print(df.logerror[df.propertyzoningdesc.isnull()].count(),df.logerror[df.propertyzoningdesc.notnull()].count())
print(df[['propertyzoningdesc','logerror']].groupby(['propertyzoningdesc'], as_index=False)['logerror'].agg(['mean','count']))
pivot=df[['propertyzoningdesc','logerror']].groupby(['propertyzoningdesc'], as_index=True)['logerror'].agg(['mean','count'])
pivot['codes']=np.arange(len(pivot))
code_dict=pivot.codes.to_dict()
sigs=pivot.index[pivot['count']>10].tolist()
df.propertyzoningdesc=df.propertyzoningdesc.apply(lambda x: x if x in sigs else -1)
test.propertyzoningdesc=test.propertyzoningdesc.apply(lambda x: x if x in sigs else -1)
df.propertyzoningdesc=df.propertyzoningdesc.map(code_dict)
test.propertyzoningdesc=test.propertyzoningdesc.map(code_dict)
df[['propertyzoningdesc','logerror']].groupby(['propertyzoningdesc'], as_index=True)['logerror'].agg(['mean','count'])
#censustractandblock doesn't look like it's correlated with the error.
print(df.logerror[df.censustractandblock.isnull()].mean(),df.logerror[df.censustractandblock.notnull()].mean())
print(df.logerror[df.censustractandblock.isnull()].count(),df.logerror[df.censustractandblock.notnull()].count())
print(np.corrcoef(df.censustractandblock[df.censustractandblock.notnull()],df.logerror[df.censustractandblock.notnull()]))
print(df[['censustractandblock','logerror']].groupby(['censustractandblock'], as_index=False)['logerror'].agg(['mean','count']))
print(df[['rawcensustractandblock','logerror']].groupby(['rawcensustractandblock'], as_index=False)['logerror'].agg(['mean','count']))
del df['censustractandblock']
del df['rawcensustractandblock']
del test['censustractandblock']
del test['rawcensustractandblock']
#county
print(df.logerror[df.regionidcounty.isnull()].mean(),df.logerror[df.regionidcounty.notnull()].mean())
print(df.logerror[df.regionidcounty.isnull()].count(),df.logerror[df.regionidcounty.notnull()].count())
#print(np.corrcoef(df.propertycountylandusecode[df.propertycountylandusecode.notnull()],df.logerror[df.propertycountylandusecode.notnull()]))
print(df[['regionidcounty','logerror']].groupby(['regionidcounty'], as_index=False)['logerror'].agg(['mean','count']))
df[['regionidcounty','logerror']].groupby(['regionidcounty'], as_index=True)['logerror'].agg(['mean','count'])
#city. using only city with more than 50 transactions
print(df.logerror[df.regionidcity.isnull()].mean(),df.logerror[df.regionidcity.notnull()].mean())
print(df.logerror[df.regionidcity.isnull()].count(),df.logerror[df.regionidcity.notnull()].count())
#print(np.corrcoef(df.propertycountylandusecode[df.propertycountylandusecode.notnull()],df.logerror[df.propertycountylandusecode.notnull()]))
print(df[['regionidcity','logerror']].groupby(['regionidcity'], as_index=False)['logerror'].agg(['mean','count']))
pivot=df[['regionidcity','logerror']].groupby(['regionidcity'], as_index=True)['logerror'].agg(['mean','count'])
pivot['codes']=np.arange(len(pivot))
code_dict=pivot.codes.to_dict()
sigs=pivot.index[pivot['count']>50].tolist()
df.regionidcity=df.regionidcity.apply(lambda x: x if x in sigs else -1)
test.regionidcity=test.regionidcity.apply(lambda x: x if x in sigs else -1)
df['regionidcity']=df.regionidcity.map(code_dict)
test['regionidcity']=test.regionidcity.map(code_dict)
df[['regionidcity','logerror']].groupby(['regionidcity'], as_index=True)['logerror'].agg(['mean','count'])

#zipcode
print(df.logerror[df.regionidzip.isnull()].mean(),df.logerror[df.regionidzip.notnull()].mean())
print(df.logerror[df.regionidzip.isnull()].count(),df.logerror[df.regionidzip.notnull()].count())
print(np.corrcoef(df.regionidzip[df.regionidzip.notnull()],df.logerror[df.regionidzip.notnull()]))
print(df[['regionidzip','logerror']].groupby(['regionidzip'], as_index=False)['logerror'].agg(['mean','count']))
pivot=df[['regionidzip','logerror']].groupby(['regionidzip'], as_index=True)['logerror'].agg(['mean','count'])
pivot['codes']=np.arange(len(pivot))
code_dict=pivot.codes.to_dict()
sigs=pivot.index[pivot['count']>50].tolist()
df.regionidzip=df.regionidzip.apply(lambda x: x if x in sigs else -1)
test.regionidzip=test.regionidzip.apply(lambda x: x if x in sigs else -1)
df['regionidzip']=df.regionidzip.map(code_dict)
test['regionidzip']=test.regionidzip.map(code_dict)
df[['regionidzip','logerror']].groupby(['regionidzip'], as_index=True)['logerror'].agg(['mean','count'])
#neighborhood
print(df.logerror[df.regionidneighborhood.isnull()].mean(),df.logerror[df.regionidneighborhood.notnull()].mean())
print(df.logerror[df.regionidneighborhood.isnull()].count(),df.logerror[df.regionidneighborhood.notnull()].count())
#print(np.corrcoef(df.regionidneighborhood[df.regionidneighborhood.notnull()],df.logerror[df.regionidneighborhood.notnull()]))
print(df[['regionidneighborhood','logerror']].groupby(['regionidneighborhood'], as_index=False)['logerror'].agg(['mean','count']))
pivot=df[['regionidneighborhood','logerror']].groupby(['regionidneighborhood'], as_index=True)['logerror'].agg(['mean','count'])
pivot['codes']=np.arange(len(pivot))
code_dict=pivot.codes.to_dict()
sigs=pivot.index[pivot['count']>50].tolist()
df.regionidneighborhood=df.regionidneighborhood.apply(lambda x: x if x in sigs else -1)
test.regionidneighborhood=test.regionidneighborhood.apply(lambda x: x if x in sigs else -1)
df['regionidneighborhood']=df.regionidneighborhood.map(code_dict)
test['regionidneighborhood']=test.regionidneighborhood.map(code_dict)
df[['regionidneighborhood','logerror']].groupby(['regionidneighborhood'], as_index=True)['logerror'].agg(['mean','count'])
#Primary residence room count
print(df.logerror[df.roomcnt.isnull()].mean(),df.logerror[df.roomcnt.notnull()].mean())
print(df.logerror[df.roomcnt.isnull()].count(),df.logerror[df.roomcnt.notnull()].count())
print(np.corrcoef(df.logerror[df.roomcnt.notnull()],df.roomcnt[df.roomcnt.notnull()]))
print(df[['roomcnt','logerror']].groupby(['roomcnt'], as_index=False)['logerror'].agg(['mean','count']))
df.roomcnt[df.roomcnt<3]=0
df.roomcnt[df.roomcnt>11]=12
test.roomcnt[test.roomcnt<3]=0
test.roomcnt[test.roomcnt>11]=12
df[['roomcnt','logerror']].groupby(['roomcnt'], as_index=False)['logerror'].agg(['mean','count'])
#story type. Only one kind, and only 43 houses.
print(df.logerror[df.storytypeid.isnull()].mean(),df.logerror[df.storytypeid.notnull()].mean())
print(df.logerror[df.storytypeid.isnull()].count(),df.logerror[df.storytypeid.notnull()].count())
#print(np.corrcoef(df.propertycountylandusecode[df.propertycountylandusecode.notnull()],df.logerror[df.propertycountylandusecode.notnull()]))
print(df[['storytypeid','logerror']].groupby(['storytypeid'], as_index=False)['logerror'].agg(['mean','count']))
df[['storytypeid','logerror']].groupby(['storytypeid'], as_index=True)['logerror'].agg(['mean','count'])
#construction type. Chose to only care if it exists because there were three types but only one of them has majority.
print(df.logerror[df.typeconstructiontypeid.isnull()].mean(),df.logerror[df.typeconstructiontypeid.notnull()].mean())
print(df.logerror[df.typeconstructiontypeid.isnull()].count(),df.logerror[df.typeconstructiontypeid.notnull()].count())
#print(np.corrcoef(df.propertycountylandusecode[df.propertycountylandusecode.notnull()],df.logerror[df.propertycountylandusecode.notnull()]))
print(df[['typeconstructiontypeid','logerror']].groupby(['typeconstructiontypeid'], as_index=False)['logerror'].agg(['mean','count']))
#df.typeconstructiontypeid[df.typeconstructiontypeid==0]=1
#df.typeconstructiontypeid[df.typeconstructiontypeid==2]=1
df[['typeconstructiontypeid','logerror']].groupby(['typeconstructiontypeid'], as_index=True)['logerror'].agg(['mean','count'])
#unit count
print(df.logerror[df.unitcnt.isnull()].mean(),df.logerror[df.unitcnt.notnull()].mean())
print(df.logerror[df.unitcnt.isnull()].count(),df.logerror[df.unitcnt.notnull()].count())
print(np.corrcoef(df.logerror[df.unitcnt.notnull()],df.unitcnt[df.unitcnt.notnull()]))
print(df[['unitcnt','logerror']].groupby(['unitcnt'], as_index=False)['logerror'].agg(['mean','count']))
#df.unitcnt[df.unitcnt<3]=0
df.unitcnt[df.unitcnt>4]=5
test.unitcnt[test.unitcnt>4]=5
df[['unitcnt','logerror']].groupby(['unitcnt'], as_index=False)['logerror'].agg(['mean','count'])
#Patio in yard
print(df.logerror[df.yardbuildingsqft17.isnull()].mean(),df.logerror[df.yardbuildingsqft17.notnull()].mean())
print(df.logerror[df.yardbuildingsqft17.isnull()].count(),df.logerror[df.yardbuildingsqft17.notnull()].count())
print(np.corrcoef(df.logerror[df.yardbuildingsqft17.notnull()],df.yardbuildingsqft17[df.yardbuildingsqft17.notnull()]))
df.yardbuildingsqft17=pd.cut(df.yardbuildingsqft17, 10).cat.codes
test.yardbuildingsqft17=pd.cut(df.yardbuildingsqft17, 10).cat.codes
print(df[['yardbuildingsqft17','logerror']].groupby(['yardbuildingsqft17'], as_index=False)['logerror'].agg(['mean','count']))
df.yardbuildingsqft17[df.yardbuildingsqft17>3]=3
test.yardbuildingsqft17[test.yardbuildingsqft17>3]=3
#df.yardbuildingsqft17[df.yardbuildingsqft17==0]=1
df[['yardbuildingsqft17','logerror']].groupby(['yardbuildingsqft17'], as_index=False)['logerror'].agg(['mean','count'])
#Storage size
print(df.logerror[df.yardbuildingsqft26.isnull()].mean(),df.logerror[df.yardbuildingsqft26.notnull()].mean())
print(df.logerror[df.yardbuildingsqft26.isnull()].count(),df.logerror[df.yardbuildingsqft26.notnull()].count())
print(np.corrcoef(df.logerror[df.yardbuildingsqft26.notnull()],df.yardbuildingsqft26[df.yardbuildingsqft26.notnull()]))
df.yardbuildingsqft26=pd.cut(df.yardbuildingsqft26, 10).cat.codes
test.yardbuildingsqft26=pd.cut(df.yardbuildingsqft26, 10).cat.codes
print(df[['yardbuildingsqft26','logerror']].groupby(['yardbuildingsqft26'], as_index=False)['logerror'].agg(['mean','count']))
#built year
print(df.logerror[df.yearbuilt.isnull()].mean(),df.logerror[df.yearbuilt.notnull()].mean())
print(df.logerror[df.yearbuilt.isnull()].count(),df.logerror[df.yearbuilt.notnull()].count())
print(np.corrcoef(df.logerror[df.yearbuilt.notnull()],df.yearbuilt[df.yearbuilt.notnull()]))
df.yearbuilt=pd.cut(df.yearbuilt, 10).cat.codes
test.yearbuilt=pd.cut(df.yearbuilt, 10).cat.codes
print(df[['yearbuilt','logerror']].groupby(['yearbuilt'], as_index=False)['logerror'].agg(['mean','count']))
#df.yearbuilt[df.yearbuilt>4]=4
#df.poolsizesum[df.poolsizesum==0]=1
#print(df[['yearbuilt','logerror']].groupby(['yearbuilt'], as_index=False)['logerror'].agg(['mean','count']))
#Assessment value. As even intervals doesn't work well, I chose to categorize manually.
print(df.logerror[df.taxvaluedollarcnt.isnull()].mean(),df.logerror[df.taxvaluedollarcnt.notnull()].mean())
print(df.logerror[df.taxvaluedollarcnt.isnull()].count(),df.logerror[df.taxvaluedollarcnt.notnull()].count())
print(np.corrcoef(df.logerror[df.taxvaluedollarcnt.notnull()],df.taxvaluedollarcnt[df.taxvaluedollarcnt.notnull()]))
bins=np.multiply([1,2,3,4,5,6,8,10,12.5,15,17.5,20,25,30,40,50,100],100000)
df.taxvaluedollarcnt=pd.cut(df.taxvaluedollarcnt,bins=bins).cat.codes
test.taxvaluedollarcnt=pd.cut(test.taxvaluedollarcnt,bins=bins).cat.codes
#df.taxvaluedollarcnt=pd.cut(df.taxvaluedollarcnt, 50)#.cat.codes
print(df[['taxvaluedollarcnt','logerror']].groupby(['taxvaluedollarcnt'], as_index=False)['logerror'].agg(['mean','count']))
#df.yearbuilt[df.yearbuilt>4]=4
#df.poolsizesum[df.poolsizesum==0]=1
#print(df[['yearbuilt','logerror']].groupby(['yearbuilt'], as_index=False)['logerror'].agg(['mean','count']))
#structure tax value
print(df.logerror[df.structuretaxvaluedollarcnt.isnull()].mean(),df.logerror[df.structuretaxvaluedollarcnt.notnull()].mean())
print(df.logerror[df.structuretaxvaluedollarcnt.isnull()].count(),df.logerror[df.structuretaxvaluedollarcnt.notnull()].count())
print(np.corrcoef(df.logerror[df.structuretaxvaluedollarcnt.notnull()],df.structuretaxvaluedollarcnt[df.structuretaxvaluedollarcnt.notnull()]))
bins=np.multiply([1,2,3,4,5,6,8,10,12.5,15,17.5,20,25,30,40,50,100],100000/2)
df.structuretaxvaluedollarcnt=pd.cut(df.structuretaxvaluedollarcnt,bins=bins).cat.codes
test.structuretaxvaluedollarcnt=pd.cut(test.structuretaxvaluedollarcnt,bins=bins).cat.codes
print(df[['structuretaxvaluedollarcnt','logerror']].groupby(['structuretaxvaluedollarcnt'], as_index=False)['logerror'].agg(['mean','count']))

#land tax value
print(df.logerror[df.landtaxvaluedollarcnt.isnull()].mean(),df.logerror[df.landtaxvaluedollarcnt.notnull()].mean())
print(df.logerror[df.landtaxvaluedollarcnt.isnull()].count(),df.logerror[df.landtaxvaluedollarcnt.notnull()].count())
print(np.corrcoef(df.logerror[df.landtaxvaluedollarcnt.notnull()],df.landtaxvaluedollarcnt[df.landtaxvaluedollarcnt.notnull()]))
bins=np.multiply([1,2,3,4,5,6,8,10,12.5,15,17.5,20,25,30,40,50,100],100000/2)
df.landtaxvaluedollarcnt=pd.cut(df.landtaxvaluedollarcnt,bins=bins).cat.codes
test.landtaxvaluedollarcnt=pd.cut(test.landtaxvaluedollarcnt,bins=bins).cat.codes
print(df[['landtaxvaluedollarcnt','logerror']].groupby(['landtaxvaluedollarcnt'], as_index=False)['logerror'].agg(['mean','count']))

#tax amount
print(df.logerror[df.taxamount.isnull()].mean(),df.logerror[df.taxamount.notnull()].mean())
print(df.logerror[df.taxamount.isnull()].count(),df.logerror[df.taxamount.notnull()].count())
print(np.corrcoef(df.logerror[df.taxamount.notnull()],df.taxamount[df.taxamount.notnull()]))
bins=np.multiply([1,2,3,4,5,6,8,10,12.5,15,17.5,20,25,30,40,50,100],100000/40)
df.taxamount=pd.cut(df.taxamount,bins=bins).cat.codes
test.taxamount=pd.cut(test.taxamount,bins=bins).cat.codes
print(df[['taxamount','logerror']].groupby(['taxamount'], as_index=False)['logerror'].agg(['mean','count']))
#Assessment year is just one year before transaction year. This variable is unnecessary.
print(df.logerror[df.assessmentyear.isnull()].mean(),df.logerror[df.assessmentyear.notnull()].mean())
print(df.logerror[df.assessmentyear.isnull()].count(),df.logerror[df.assessmentyear.notnull()].count())
print(np.corrcoef(df.logerror[df.assessmentyear.notnull()],df.assessmentyear[df.assessmentyear.notnull()]))
#df.assessmentyear=pd.cut(df.assessmentyear, 10).cat.codes
print(df[['assessmentyear','logerror']].groupby(['assessmentyear'], as_index=False)['logerror'].agg(['mean','count']))
del df['assessmentyear']
del test['assessmentyear']
#tax delinquency
print(df.logerror[df.taxdelinquencyflag.isnull()].mean(),df.logerror[df.taxdelinquencyflag.notnull()].mean())
print(df.logerror[df.taxdelinquencyflag.isnull()].count(),df.logerror[df.taxdelinquencyflag.notnull()].count())
#print(np.corrcoef(df.propertycountylandusecode[df.propertycountylandusecode.notnull()],df.logerror[df.propertycountylandusecode.notnull()]))
print(df[['taxdelinquencyflag','logerror']].groupby(['taxdelinquencyflag'], as_index=False)['logerror'].agg(['mean','count']))
df.taxdelinquencyflag=pd.Categorical(df.taxdelinquencyflag)
df.taxdelinquencyflag=df.taxdelinquencyflag.cat.codes
test.taxdelinquencyflag=pd.Categorical(test.taxdelinquencyflag)
test.taxdelinquencyflag=test.taxdelinquencyflag.cat.codes
df[['taxdelinquencyflag','logerror']].groupby(['taxdelinquencyflag'], as_index=True)['logerror'].agg(['mean','count'])
#tax delinquency year
print(df.logerror[df.taxdelinquencyyear.isnull()].mean(),df.logerror[df.taxdelinquencyyear.notnull()].mean())
print(df.logerror[df.taxdelinquencyyear.isnull()].count(),df.logerror[df.taxdelinquencyyear.notnull()].count())
#print(np.corrcoef(df.propertycountylandusecode[df.propertycountylandusecode.notnull()],df.logerror[df.propertycountylandusecode.notnull()]))
print(df[['taxdelinquencyyear','logerror']].groupby(['taxdelinquencyyear'], as_index=False)['logerror'].agg(['mean','count']))
df[['taxdelinquencyyear','logerror']].groupby(['taxdelinquencyyear'], as_index=True)['logerror'].agg(['mean','count'])
#In real estate market, seasonality does matter. I saw manay people just choose not to use transactiondate as variable, but it should be. Also,
#we do give different results for different months.
df.transactiondate=df.transactiondate.str.replace('-','').str[:6]
df['month']=df.transactiondate.str[4:6]
df['year']=df.transactiondate.str[:4]
print(df[['year','month','logerror']].groupby(['year','month'], as_index=False)['logerror'].agg(['mean','count']))
df.year=pd.Categorical(df.year)
df.year=df.year.cat.codes
df.month=pd.Categorical(df.month)
df.month=df.month.cat.codes
del df['transactiondate']
df[df.isnull()]=-1
test[test.isnull()]=-1
for c, dtype in zip(df.columns, df.dtypes):
    if dtype == np.float64:
        df[c] = df[c].astype(np.int32) 
    elif dtype == np.float32:
        df[c] = df[c].astype(np.int32)
for c, dtype in zip(test.columns, test.dtypes):
    if dtype == np.float64:
        test[c] = test[c].astype(np.int32) 
    elif dtype == np.float32:
        test[c] = test[c].astype(np.int32)
print(df.info())
print(test.info())
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random

df=df.reindex(sorted(df.columns), axis=1)
label=df.logerror
df=df.drop(['parcelid','logerror'],axis=1)
d_train = lgb.Dataset(df, label=label)

# Parameters
XGB_WEIGHT = 0.6000
BASELINE_WEIGHT = 0.0000
OLS_WEIGHT = 0.0600

XGB1_WEIGHT = 0.8000  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

##### RUN LIGHTGBM

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3

np.random.seed(0)
random.seed(0)

print("\nFitting LightGBM model ...")
clf = lgb.train(params, d_train, 430)

del d_train; gc.collect()

test['month']=0
test=test.reindex(sorted(test.columns), axis=1)
for d in dates:
    if d!='parcelid':
        test.month=int(d[4:6])-1
        sample[d]=clf.predict(test[test.year==(int(d[:4])-2016)].drop(['parcelid'],axis=1))
sample.to_csv('lightGBM.csv',index=False)