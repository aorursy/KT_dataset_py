

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import math



import time

import os

import gc

import random

from scipy.stats import uniform

#  Processing data

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  OneHotEncoder as ohe

from sklearn.preprocessing import StandardScaler as ss

from sklearn.compose import ColumnTransformer as ct



# Data imputation

from sklearn.impute import SimpleImputer



from sklearn.decomposition import PCA

from sklearn import metrics



import scikitplot as skplt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics

from sklearn.metrics import classification_report



from sklearn.ensemble import RandomForestClassifier 

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import f1_score, make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV





from xgboost.sklearn import XGBClassifier

from bayes_opt import BayesianOptimization



from skopt import BayesSearchCV

#  Model pipelining

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline





# 1.7 Model evaluation metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve



# 1.8

import matplotlib.pyplot as plt

from xgboost import plot_importance



# 1.9 Needed for Bayes optimization

from sklearn.model_selection import cross_val_score



from bayes_opt import BayesianOptimization





from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb



# 3.6 Change ipython options to display all data columns

pd.options.display.max_columns = 400
# 2.1 Read cdata/test files

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print(train.shape)    #9557 X 143

print(test.shape)     # 23856 X 142



train.dtypes

test.dtypes

#train.columns

train_col_names = list(train.columns)

test_col_names = list(test.columns)

### adding Target column  to test with null value

test['Target'] = np.nan

test.dtypes

cdata = pd.concat([train,test],

                axis = 0,            # Stack one upon another (rbind)

                ignore_index = True

                )

print(cdata.shape)     #33413 X 143



print(train.dtypes.value_counts())
##############################################################################

## combine 8 binary features for level of education ( 9 levels)  into a single column instlevel

###############################################################################

del_columns =[]  #initializing list of columns to be combined

cat_columns = [] # initializing a list of Categorical columns that will be needed for PCA

#Checking if the columns are undefined for any row.

((cdata['instlevel1'] + cdata['instlevel2'] + cdata['instlevel3'] + cdata['instlevel4'] +

  cdata['instlevel5'] + cdata['instlevel6'] + cdata['instlevel7'] + cdata['instlevel8'] + 

  cdata['instlevel9']) == 0).sum() 

## 3 rows does not have instlevel defined for any rows

cdata.loc[((cdata['instlevel1'] + cdata['instlevel2'] + cdata['instlevel3'] + cdata['instlevel4'] +

            cdata['instlevel5'] + cdata['instlevel6'] + cdata['instlevel7'] + cdata['instlevel8'] + 

            cdata['instlevel9']) == 0)].rez_esc

##escloari - years of schooling is 0, rez_esc is nan, age is 44,45 & 97 years for all the 3 rows, Target is 3 & 4

### meaneduc is 4, 4 & 3 years for these 3 records,  Safe to classify these records as 2 - Incomplete Primary   

cdata.loc[((cdata['instlevel1'] + cdata['instlevel2'] + cdata['instlevel3'] + cdata['instlevel4'] +

            cdata['instlevel5'] + cdata['instlevel6'] + cdata['instlevel7'] + cdata['instlevel8'] + 

            cdata['instlevel9']) == 0),['age','escolari','rez_esc','meaneduc','Target']]

cdata.loc[((cdata['instlevel1'] + cdata['instlevel2'] + cdata['instlevel3'] + cdata['instlevel4'] +

            cdata['instlevel5'] + cdata['instlevel6'] + cdata['instlevel7'] + cdata['instlevel8'] + 

            cdata['instlevel9']) == 0),'instlevel2'] = 1

education_mapping = ({1:'No education', 2: 'Incomplete primary', 3 :'Complete Primary', 4 :'Incomplete secondary', 5: 'Complete secondary', 6: 'Incomplete technical Secondary' , 7: 'Complete technical Secondary',8 : 'Undergraduate and higher', 9: 'Postgraduate and higher'})

cdata.loc[(cdata['instlevel1'] == 1), 'instlevel'] = 1

cdata.loc[(cdata['instlevel2'] == 1), 'instlevel'] = 2

cdata.loc[(cdata['instlevel3'] == 1), 'instlevel'] = 3

cdata.loc[(cdata['instlevel4'] == 1), 'instlevel'] = 4

cdata.loc[(cdata['instlevel5'] == 1), 'instlevel'] = 5

cdata.loc[(cdata['instlevel6'] == 1), 'instlevel'] = 6

cdata.loc[(cdata['instlevel7'] == 1), 'instlevel'] = 7

cdata.loc[(cdata['instlevel8'] == 1), 'instlevel'] = 8

cdata.loc[(cdata['instlevel9'] == 1), 'instlevel'] = 9



del_columns = del_columns + ['instlevel1', 'instlevel2','instlevel3', 'instlevel4','instlevel5']

del_columns = del_columns + ['instlevel6', 'instlevel7','instlevel8', 'instlevel9']

cat_columns = ['instlevel']
###############################################################################

## combine 12 binary features for level of education ( 9 levels)  into a single column instlevel

###############################################################################

(cdata['parentesco1'] ==1).sum() - 2973

## Checking if values are defined for more than 1 of these columns

((cdata['parentesco1'] + cdata['parentesco2'] + cdata['parentesco3'] + cdata['parentesco4'] +

  cdata['parentesco5'] + cdata['parentesco6'] + cdata['parentesco7'] + cdata['parentesco8'] + 

  cdata['parentesco9'] + cdata['parentesco10'] + cdata['parentesco11'] + cdata['parentesco12'] ) > 1).sum() 



##define the mapping values 

family_mapping = ({1: 'Household head', 2 : 'Spouse/Partner', 3: 'Son/daughter', 4 : 'Stepson/daughter', 5: 'Son/Daughter in law', 6: 'Grandson/Daughter', 7:'Mother/Father', 8:'Father/Mother in law', 9: 'Brother/Sister', 10: 'Brother/Sister in law', 11: 'Other familymember', 12 :'Non family member'})

cdata.loc[(cdata['parentesco1'] == 1), 'parent_level'] = 1

cdata.loc[(cdata['parentesco2'] == 1), 'parent_level'] = 2

cdata.loc[(cdata['parentesco3'] == 1), 'parent_level'] = 3

cdata.loc[(cdata['parentesco4'] == 1), 'parent_level'] = 4

cdata.loc[(cdata['parentesco5'] == 1), 'parent_level'] = 5

cdata.loc[(cdata['parentesco6'] == 1), 'parent_level'] = 6

cdata.loc[(cdata['parentesco7'] == 1), 'parent_level'] = 7

cdata.loc[(cdata['parentesco8'] == 1), 'parent_level'] = 8

cdata.loc[(cdata['parentesco9'] == 1), 'parent_level'] = 9

cdata.loc[(cdata['parentesco10'] == 1), 'parent_level'] = 10

cdata.loc[(cdata['parentesco11'] == 1), 'parent_level'] = 11

cdata.loc[(cdata['parentesco12'] == 1), 'parent_level'] = 12



del_columns = del_columns + ['parentesco1', 'parentesco2','parentesco3', 'parentesco4','parentesco5']

del_columns = del_columns + ['parentesco6', 'parentesco7','parentesco8', 'parentesco9','parentesco10']

del_columns = del_columns + ['parentesco11', 'parentesco12']

cat_columns = cat_columns + ['parent_level']
######################################################################################

###########combine 7 binary features for Marital Status ( 7 levels)  into a single column mar_sts

#####################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['estadocivil1'] + cdata['estadocivil2'] + cdata['estadocivil3'] + cdata['estadocivil4'] +

  cdata['estadocivil5'] + cdata['estadocivil6'] + cdata['estadocivil7']  ) > 1).sum() 



Mar_sts_mapping = ({1: 'Less than 10', 2 : 'Free or Coupled', 3: 'Married', 4 : 'Divorced', 5: 'Separated', 6: 'Widow-er', 7:'Single'})

cdata.loc[(cdata['estadocivil1'] == 1), 'mar_sts'] = 1

cdata.loc[(cdata['estadocivil2'] == 1), 'mar_sts'] = 2

cdata.loc[(cdata['estadocivil3'] == 1), 'mar_sts'] = 3

cdata.loc[(cdata['estadocivil4'] == 1), 'mar_sts'] = 4

cdata.loc[(cdata['estadocivil5'] == 1), 'mar_sts'] = 5

cdata.loc[(cdata['estadocivil6'] == 1), 'mar_sts'] = 6

cdata.loc[(cdata['estadocivil7'] == 1), 'mar_sts'] = 7

del_columns = del_columns + ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4','estadocivil5','estadocivil6', 'estadocivil7']

cat_columns = cat_columns + ['mar_sts']
##############################################################################

## combine 6 binary features for Region ( lugar1 to lugar6)  into a single column lugar

##################################################################################

region_mapping = ({1: 'Central', 2: 'Chorotega', 3: 'PacÃ­fico central', 4: 'Brunca' , 5 : 'Huetar AtlÃ¡ntica', 6:'Huetar Norte' })

Region  = ['Central', 'Chorotega', 'PacÃ­fico central','Brunca', 'Huetar AtlÃ¡ntica', 'Huetar Norte']



cdata.loc[(cdata['lugar1'] == 1), 'lugar'] = 1

cdata.loc[(cdata['lugar2'] == 1), 'lugar'] = 2

cdata.loc[(cdata['lugar3'] == 1), 'lugar'] = 3

cdata.loc[(cdata['lugar4'] == 1), 'lugar'] = 4

cdata.loc[(cdata['lugar5'] == 1), 'lugar'] = 5

cdata.loc[(cdata['lugar6'] == 1), 'lugar'] = 6

del_columns = del_columns + ['lugar1', 'lugar2', 'lugar3', 'lugar4','lugar5','lugar6']

cat_columns = cat_columns + ['lugar']
##############################################################################

## combine 2 binary features for Area ( area1 & area2)  into a single column area

## Mapping for Urban and Rural Zones

###############################################################################

area_mapping = ({1: 'Urban', 2: 'Rural'})

Area  = ['Urban', 'Rural']

cdata.loc[(cdata['area1'] == 1), 'area'] = 1

cdata.loc[(cdata['area2'] == 1), 'area'] = 2

sns.countplot(cdata.area.map(area_mapping), data = cdata) ## distribution by Zone - urban / rural

del_columns = del_columns + ['area1', 'area2']

cat_columns = cat_columns + ['area']
##############################################################################

## combine 5 binary features for house ownership status  ( tipoviv11 to tipovivi5)  into a single column lugar

##################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['tipovivi1'] + cdata['tipovivi2'] + cdata['tipovivi3'] + cdata['tipovivi4'] +

  cdata['tipovivi5']  ) > 1).sum() 

house_mapping = ({1: 'Owned and paid', 2: 'Owned and paying install', 3: 'Rented', 4: 'Precarious' , 5 : 'Other' })

cdata.loc[(cdata['tipovivi1'] == 1), 'tipovivi'] = 1

cdata.loc[(cdata['tipovivi2'] == 1), 'tipovivi'] = 2

cdata.loc[(cdata['tipovivi3'] == 1), 'tipovivi'] = 3

cdata.loc[(cdata['tipovivi4'] == 1), 'tipovivi'] = 4

cdata.loc[(cdata['tipovivi5'] == 1), 'tipovivi'] = 5



del_columns = del_columns + ['tipovivi1', 'tipovivi2', 'tipovivi3','tipovivi4', 'tipovivi5']

cat_columns = cat_columns + ['tipovivi']
##############################################################################

## combine 5 binary features for sanitation status  ( sanitario1 to sanitario6)  into a single column lugar

##################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['sanitario1'] + cdata['sanitario2'] + cdata['sanitario3'] + cdata['sanitario5'] +

  cdata['sanitario6']  ) > 1).sum()

sanit_mapping = ({1: 'No Toilet', 2: 'Connected to Sewer', 3: 'Connected to Septic T', 4: 'Connected to BH/L' , 5 : 'Connected to Others' })

cdata.loc[(cdata['sanitario1'] == 1), 'sanitario'] = 1

cdata.loc[(cdata['sanitario2'] == 1), 'sanitario'] = 2

cdata.loc[(cdata['sanitario3'] == 1), 'sanitario'] = 3

cdata.loc[(cdata['sanitario5'] == 1), 'sanitario'] = 4

cdata.loc[(cdata['sanitario6'] == 1), 'sanitario'] = 5



del_columns = del_columns + ['sanitario1', 'sanitario2', 'sanitario3','sanitario5', 'sanitario6']

cat_columns = cat_columns + ['sanitario']
##############################################################################

## combine 5 binary features for cooking energy source  ( energcocinar1 to energcocinar4)  into a single column energy

##################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['energcocinar1'] + cdata['energcocinar2'] + cdata['energcocinar3'] + cdata['energcocinar4']) > 1).sum()

cook_energy_mapping = ({1: 'No Kitchen', 2: 'Electricity', 3: 'Cooking gas', 4: 'Charcoal'  })

cdata.loc[(cdata['energcocinar1'] == 1), 'energco'] = 1

cdata.loc[(cdata['energcocinar2'] == 1), 'energco'] = 2

cdata.loc[(cdata['energcocinar3'] == 1), 'energco'] = 3

cdata.loc[(cdata['energcocinar4'] == 1), 'energco'] = 4





del_columns = del_columns + ['energcocinar1', 'energcocinar2', 'energcocinar3','energcocinar4']

cat_columns = cat_columns + ['energco']
##############################################################################

## combine 6 binary features for rubbish disposal  ( elimbasu1 to elimbasu1)  into a single column elimbasu

##################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['elimbasu1'] + cdata['elimbasu2'] + cdata['elimbasu3'] + cdata['elimbasu4'] +

  cdata['elimbasu5'] + cdata['elimbasu6'] ) > 1).sum()

rubbish_mapping = ({1: 'tanker truck', 2: 'hollow or buried', 3: 'burning', 4: 'throwing in UOC space' , 5 : 'throwing in river' , 6 :'Other'})

cdata.loc[(cdata['elimbasu1'] == 1), 'elimbasu'] = 1

cdata.loc[(cdata['elimbasu2'] == 1), 'elimbasu'] = 2

cdata.loc[(cdata['elimbasu3'] == 1), 'elimbasu'] = 3

cdata.loc[(cdata['elimbasu4'] == 1), 'elimbasu'] = 4

cdata.loc[(cdata['elimbasu5'] == 1), 'elimbasu'] = 5

cdata.loc[(cdata['elimbasu6'] == 1), 'elimbasu'] = 6

del_columns = del_columns + ['elimbasu1', 'elimbasu2', 'elimbasu3','elimbasu4', 'elimbasu5', 'elimbasu6']

cat_columns = cat_columns + ['elimbasu']
##############################################################################

## combine 2 binary features for male/ female  ( into a single column sex

##################################################################################

sex_mapping = ({1: 'Male', 2: 'Female'})

cdata.loc[(cdata['male'] == 1), 'sex'] = 1

cdata.loc[(cdata['female'] == 1), 'sex'] = 2



del_columns = del_columns + ['male', 'female']

cat_columns = cat_columns + ['sex']
######################################################################################

###########combine 8 binary features for Predominant material on outside wall ( 8 levels)  into a single column out_wall

#####################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['paredblolad'] + cdata['paredzocalo'] + cdata['paredpreb'] + cdata['pareddes'] +

  cdata['paredmad'] + cdata['paredzinc'] + cdata['paredfibras'] + cdata['paredother'] ) > 1).sum() 



wall_mat_mapping = ({1: 'block or brick', 2 : 'Socket', 3: 'pre-fab or cement', 4 : 'waste material', 5: 'Wood', 6: 'Zinc', 7:'natural fiber',8:'Other'})

cdata.loc[(cdata['paredblolad'] == 1), 'wall_mat'] = 1

cdata.loc[(cdata['paredzocalo'] == 1), 'wall_mat'] = 2

cdata.loc[(cdata['paredpreb'] == 1), 'wall_mat'] = 3

cdata.loc[(cdata['pareddes'] == 1), 'wall_mat'] = 4

cdata.loc[(cdata['paredmad'] == 1), 'wall_mat'] = 5

cdata.loc[(cdata['paredzinc'] == 1), 'wall_mat'] = 6

cdata.loc[(cdata['paredfibras'] == 1), 'wall_mat'] = 7

cdata.loc[(cdata['paredother'] == 1), 'wall_mat'] = 8



del_columns = del_columns + ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes','paredmad','paredzinc', 'paredfibras','paredother']

cat_columns = cat_columns + ['wall_mat']
######################################################################################

###########combine 6 binary features for Floor material ( 6 levels)  into a single column out_wall

#####################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['pisomoscer'] + cdata['pisocemento'] + cdata['pisoother'] + cdata['pisonatur'] +

  cdata['pisonotiene'] + cdata['pisomadera']  ) > 1).sum() 



floor_mat_mapping = ({1: 'Mosaic/Ceramic', 2 : 'Cement', 3: 'Other', 4 : 'Natural material', 5: 'No Floor', 6: 'Wood'})

cdata.loc[(cdata['pisomoscer'] == 1), 'floor_mat'] = 1

cdata.loc[(cdata['pisocemento'] == 1), 'floor_mat'] = 2

cdata.loc[(cdata['pisoother'] == 1), 'floor_mat'] = 3

cdata.loc[(cdata['pisonatur'] == 1), 'floor_mat'] = 4

cdata.loc[(cdata['pisonotiene'] == 1), 'floor_mat'] = 5

cdata.loc[(cdata['pisomadera'] == 1), 'floor_mat'] = 6



del_columns = del_columns + ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur','pisonotiene','pisomadera']

cat_columns = cat_columns + ['floor_mat']
######################################################################################

###########combine 4 binary features for Roof material ( 6 levels)  into a single column out_wall

#####################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['techozinc'] + cdata['techoentrepiso'] + cdata['techocane'] + cdata['techootro']  ) > 1).sum() 



((cdata['techozinc'] + cdata['techoentrepiso'] + cdata['techocane'] + cdata['techootro']  ) == 0).sum() ## 66 rows with undefined



cdata.loc[(cdata['techozinc'] + cdata['techoentrepiso'] + cdata['techocane'] + cdata['techootro'] ==0  ),['tipovivi4','Target']]



cdata.loc[(cdata['techozinc'] + cdata['techoentrepiso'] + cdata['techocane'] + cdata['techootro'] ==0)].tipovivi.value_counts()  # 40 in precarious

cdata.loc[(cdata['techozinc'] + cdata['techoentrepiso'] + cdata['techocane'] + cdata['techootro'] ==0)].Target.value_counts()  # 50 in moderate/extreme poverty





cdata[((cdata.Target == 1) | (cdata.Target == 2))].techootro.value_counts().nunique()  ## Roof material value = 1 

## can update techootro to 1 for these 66 rows

cdata.loc[((cdata['techozinc'] + cdata['techoentrepiso'] + cdata['techocane'] + cdata['techootro']  ) == 0),'techootro'] = 1



roof_mat_mapping = ({1: 'Metal foil/Zinc', 2 : 'Fiber Cement', 3: 'Natural Fiber', 4 : 'Other'})

cdata.loc[(cdata['techozinc'] == 1), 'roof_mat'] = 1

cdata.loc[(cdata['techoentrepiso'] == 1), 'roof_mat'] = 2

cdata.loc[(cdata['techocane'] == 1), 'roof_mat'] = 3

cdata.loc[(cdata['techootro'] == 1), 'roof_mat'] = 4





del_columns = del_columns + ['techozinc', 'techoentrepiso', 'techocane', 'techootro']

cat_columns = cat_columns + ['roof_mat']
######################################################################################

###########combine 3 binary features for Water Provision ( 3 levels)  into a single column 

#####################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['abastaguadentro'] + cdata['abastaguafuera'] + cdata['abastaguano']  ) > 1).sum() 



water_sts_mapping = ({1: 'Inside', 2 : 'Outside', 3: 'no water provision'})

cdata.loc[(cdata['abastaguadentro'] == 1), 'water_sts'] = 1

cdata.loc[(cdata['abastaguafuera'] == 1), 'water_sts'] = 2

cdata.loc[(cdata['abastaguano'] == 1), 'water_sts'] = 3



del_columns = del_columns + ['abastaguadentro', 'abastaguafuera', 'abastaguano']

cat_columns = cat_columns + ['water_sts']
######################################################################################

###########combine 3 binary features for Electricity( 4 levels)  into a single column 

#####################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['public'] + cdata['planpri'] + cdata['noelec']  + cdata['coopele']) > 1).sum() 

##  Checking if these columns are undefined for any rows

((cdata['public'] + cdata['planpri'] + cdata['noelec'] + cdata['coopele']) == 0).sum()   #15 rows with undefined  

cdata.loc[((cdata['public'] + cdata['planpri'] + cdata['noelec'] + cdata['coopele']) == 0),['tipovivi1','Target']]

cdata.loc[((cdata['public'] + cdata['planpri'] + cdata['noelec'] + cdata['coopele']) == 0)].tipovivi1.value_counts()  ##10 rows Own house

cdata.loc[((cdata['public'] + cdata['planpri'] + cdata['noelec'] + cdata['coopele']) == 0)].Target.value_counts() ## 9 + 4 in 1 & 2

cdata.public.value_counts()  ## 8459 rows 88%  with public

cdata.planpri.value_counts()  ##  3 rows

cdata.noelec.value_counts()   ## 21 rows

cdata.coopele.value_counts()  ## 1059 rows

cdata[((cdata.Target == 1) | (cdata.Target == 2))].public.value_counts()

##updating with majority -public

cdata.loc[((cdata['public'] + cdata['planpri'] + cdata['noelec'] + cdata['coopele']) == 0),['public']] = 1

electricity_sts_mapping = ({1: 'CNFL,ICE,ESPH/JASEC', 2 : 'Private', 3: 'Cooperative' , 4: 'None'})

cdata.loc[(cdata['public'] == 1), 'elect_sts'] = 1

cdata.loc[(cdata['planpri'] == 1), 'elect_sts'] = 2

cdata.loc[(cdata['coopele'] == 1), 'elect_sts'] = 3

cdata.loc[(cdata['noelec'] == 1), 'elect_sts'] = 4





del_columns = del_columns + ['public', 'planpri', 'coopele', 'noelec']

cat_columns = cat_columns + ['elect_sts']
######################################################################################

###########combine 3 binary features for wall condition( 3 levels)  into a single column 

#####################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['epared1'] + cdata['epared2'] + cdata['epared3']  ) > 1).sum() 



wall_sts_mapping = ({1: 'Bad', 2 : 'Regular', 3: 'Good'})

cdata.loc[(cdata['epared1'] == 1), 'wall_sts'] = 1

cdata.loc[(cdata['epared2'] == 1), 'wall_sts'] = 2

cdata.loc[(cdata['epared3'] == 1), 'wall_sts'] = 3



del_columns = del_columns + ['epared1', 'epared2', 'epared3']

cat_columns = cat_columns + ['wall_sts']
######################################################################################

###########combine 3 binary features for roof  condition( 3 levels)  into a single column 

#####################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['etecho1'] + cdata['etecho2'] + cdata['etecho3']  ) > 1).sum() 

##  Checking if these columns are undefined for any rows

((cdata['etecho1'] + cdata['etecho2'] + cdata['etecho3']  ) == 0).sum() 

roof_sts_mapping = ({1: 'Bad', 2 : 'Regular', 3: 'Good'})

cdata.loc[(cdata['etecho1'] == 1), 'roof_sts'] = 1

cdata.loc[(cdata['etecho2'] == 1), 'roof_sts'] = 2

cdata.loc[(cdata['etecho3'] == 1), 'roof_sts'] = 3



del_columns = del_columns + ['etecho1', 'etecho2', 'etecho3']

cat_columns = cat_columns + ['roof_sts']
####################################################################################

######################################################################################

###########combine 3 binary features for Floor  condition( 3 levels)  into a single column 

#####################################################################################

## Checking if values are defined for more than 1 of these columns

((cdata['eviv1'] + cdata['eviv2'] + cdata['eviv3']  ) > 1).sum() 



floor_sts_mapping = ({1: 'Bad', 2 : 'Regular', 3: 'Good'})

cdata.loc[(cdata['eviv1'] == 1), 'floor_sts'] = 1

cdata.loc[(cdata['eviv2'] == 1), 'floor_sts'] = 2

cdata.loc[(cdata['eviv3'] == 1), 'floor_sts'] = 3



del_columns = del_columns + ['eviv1', 'eviv2', 'eviv3']

cat_columns = cat_columns + ['floor_sts']
print(del_columns)

print(len(del_columns))



#cdata.drop(del_columns, inplace = True, axis =1)



print(cdata.shape)   #9557 X 70

print(cdata.columns)

print(cat_columns)

len(cat_columns)
cdata[ ['r4t3','tamhog','tamviv','hhsize','hogar_total' ]]

((cdata['r4t3'] + cdata['tamhog'] + cdata['tamviv'] + cdata['hhsize'] + cdata['hogar_total' ] % 5) != 0).sum()  #9557

## these columns are identical as all these columns have the same values

## Retaining column 'hogar_total' and deleting the rest of the columns as we dont need them



del_columns = del_columns + ['r4t3', 'tamhog', 'tamviv', 'hhsize']

cdata.drop(del_columns, inplace = True, axis =1)



cdata.columns
###############################################################################################################

#Checking null values in data #################################################################################

###############################################################################################################

cdata.isnull().any() 

cdata.columns[cdata.isnull().any()].tolist()

# 5 columns has null values - v2a1, v18q1, rez_esc, meaneduc, SQBmeaned

###############################################################################
# v2a1 - Monthly Rent payment is null..check how does this relate to tipovivi3, =1 rented and 

#tipovivi : 1 : Owned and paid', 2: 'Owned and paying install', 3: 'Rented', 4: 'Precarious' , 5 : 'Other' 

cdata.v2a1.isnull().sum() # 6860 rows has null value

cdata.v2a1.isnull().value_counts()   #6860 row is True - Null

cdata[cdata.v2a1.isnull()].tipovivi.value_counts()   #rented is 0 where monthly rent payment is null ;1, 4 & 5

cdata['v2a1'] = cdata['v2a1'].fillna(0)  ##  update as 0
###############################################################################

# v18q1 - number of tablets household owns  has null values;check how does this relate to v18q  =1 owns tablet

cdata.v18q1.isnull().sum()  #7342 rows has null values

cdata.v18q1.isnull().value_counts()  #7342 rows with nulls

cdata[cdata['v18q'] ==0].v18q.value_counts()  ## all of them does not own a mobile, v18q = 0

cdata[cdata.v18q1.isnull()].v18q.value_counts()   # owns tablet is 0 where # of tablets  is null -7342 does not own tablet

cdata['v18q1'] = cdata['v18q1'].fillna(0)   # fill NaN with zeroes for # of mobiles
##########################################################################################

# meaneduc & SQBmeaned - Mean education has null values and check how does this feature relate to instlevel1 - no level of education

###########################################################################################

np.sum(cdata.meaneduc.isnull())  # 5 rows has null values for meaneduc

np.sum(cdata.SQBmeaned.isnull()) # 5 rows has null values

cdata.loc[(cdata.meaneduc.isnull()),['SQBmeaned','rez_esc', 'meaneduc']]  ## all of these columns have NAN values SQBmeaned, rez_esc, meaneduc

###check further for level of education --intlevel

cdata[cdata.meaneduc.isnull()].instlevel.value_counts()  #  4 rows 'No education, 1 row  'incomplete primary

## Can safely initialize meaneduc & SQBmeaned  to 0 with the above data 

cdata['meaneduc'] = cdata['meaneduc'].fillna(0) 

cdata['SQBmeaned'] = cdata['SQBmeaned'].fillna(0) 
###############################################################################################################

# rez_esc - years behind in school

################################################################################################################

np.sum(cdata.rez_esc.isnull())    #7928 rows has null values - 83%

np.sum((cdata.rez_esc.isnull()) & (cdata.age < 7) | (cdata.age > 19))   ## 7578 rows can be set to 0 



cdata.loc[(cdata.rez_esc.isnull()) & (cdata.age < 7) | (cdata.age > 19), 'rez_esc'] = 0

np.sum(cdata.rez_esc.isnull())    ## 350 rows are still null..
##################################################################################

######dealing with Object variables in the data

###############################################################################

cdata.select_dtypes(object).nunique()

#Dropping column Id as it does not have any significance for the modeling

#cdata.drop(['Id'], inplace = True, axis =1)

cdata.dependency.value_counts() ### 2192 - yes  1747 no ; should be numerical

cdata.edjefe.value_counts()   ### 3762 no  123 yes ;; should be numerical

cdata.edjefa.value_counts()   ### 6230 no   69 yes

X= cdata.loc[(cdata['dependency'] == 'yes')]

X1= cdata.loc[(cdata['dependency'] == 'no')]

##  Replacing yes / no  with sqrt of SQBdependency as per Kaggle Data description

cdata.loc[(cdata['dependency'] == 'yes'), 'dependency'] = cdata.SQBdependency.map(lambda SQBdependency: math.sqrt(SQBdependency))

cdata.loc[(cdata['dependency'] == 'no'), 'dependency'] = cdata.SQBdependency.map(lambda SQBdependency: math.sqrt(SQBdependency))

X2= cdata.loc[(cdata['edjefe'] == 'no')]

##  Replacing yes with 1 and no with 0 as per Kaggle Data description

cdata.loc[(cdata['edjefe'] == 'no'), 'edjefe'] = 0

cdata.loc[(cdata['edjefe'] == 'yes'), 'edjefe'] = 1



cdata.loc[(cdata['edjefa'] == 'no'), 'edjefa'] = 0

cdata.loc[(cdata['edjefa'] == 'yes'), 'edjefa'] = 1

##  converting to Float              

cdata['dependency'] = cdata['dependency'].astype(np.float64)

cdata['edjefe'] = cdata['edjefe'].astype(np.float64)

cdata['edjefa'] = cdata['edjefa'].astype(np.float64)

cdata.dtypes

cdata.columns
################################################################################

## refrig, v18q1, computer, television are household level variables

cdata['devices_owned'] = cdata['refrig'] + cdata['computer'] + cdata['television'] + cdata['v18q1']

## total ownership of devices will have a better correlation with the target 

cdata.loc[(cdata['v18q1'] > 1), ['v18q1', 'Target','parent_level']]

cdata.groupby('area')['Target'].apply(lambda x: x.isnull().sum())   #Total count of Target with null value is 23,856
# Consolidating

## Skipping the Squared columns as they are all square of the existing features and 

## are getting picked up in the feature importance of the baseline and training models

cat_columns

data_num_columns = ['v2a1','rooms','v18q1','r4h1','r4h2','r4h3','r4m1','r4m2','r4m3','r4t1','r4t2','escolari','rez_esc',

                    'hogar_nin','hogar_adul','hogar_mayor','hogar_total','dependency','edjefe','edjefa',

                    'meaneduc','bedrooms','overcrowding','qmobilephone','age','devices_owned']

      

object_columns = ['Id','idhogar','Target']                  

data_cat_columns = ['hacdor','hacapo','v14a','refrig','v18q','cielorazo','dis','computer','television','mobilephone']

total_cat_columns = data_cat_columns + cat_columns

Total_columns = data_num_columns + total_cat_columns + object_columns

print('Total :',len(Total_columns), 'Numerical :', len(data_num_columns), 'Categorical :', len(total_cat_columns))



Float64_columns = cat_columns + ['devices_owned']

## Changing the categorical variables from  float64 columns to int64

cdata[Float64_columns] = cdata[Float64_columns].astype(np.int64)

len(total_cat_columns)
###############################################################################

# 8.2  Imputing data. 350 rows have null value for rez_esc    

# Updating with mean  

imp = SimpleImputer(strategy="mean")     

cdata['rez_esc'] = imp.fit_transform(cdata[['rez_esc']])

cdata.columns[cdata.isnull().any()].tolist()
################################################################################

#### Visualizations

################################################################################

poverty_mapping = ({1: 'Extreme-1', 2: 'Moderate-2', 3: 'Vulnerable-3', 4: 'Non vulnerable-4'})

ax1 = sns.countplot(cdata.Target.map(poverty_mapping), data = cdata)

ax1.set_ylabel("Count", fontname="Arial", fontsize=18)

# Set the title to Comic Sans

ax1.set_title("Distribution of Poverty - Combined", fontname='Arial', fontsize=18)

# Set the font name for axis tick labels to be Comic Sans

for tick in ax1.get_xticklabels():

    tick.set_fontname("Arial")

    tick.set_fontsize(18)



ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 45)

plt.show()

## Extreme Poverty is about 10% of the training data set. Target values are null for Test and 

## hence does not reflect in the data
fig = plt.figure(figsize = (5,5))

ax = fig.add_subplot(111)

Owns  = ['Tablet', 'Refrigerator', 'Computer','Television', 'Mobile']

names = ['v18q', 'refrig', 'computer', 'television','mobilephone']

for i,j in enumerate(names):

     ax = plt.subplot(2,3,i+1) 

     sns.countplot(cdata.Target.map(poverty_mapping),        

              hue= cdata[j],    # Target Distribution by ownership 

              data = cdata,

              ax = ax)

     ax.set_ylabel("Count", fontname="Arial", fontsize=18)

     ax.set_title("Owns " + Owns[i], fontname='Arial', fontsize=18)

# Set the title to Comic Sans

plt.show()

## Ownership of Tablet, Computer and Television is very minimal for the Extreme  poverty class

## Suprisingly ownership of Refreigerator and mobile is relatively higher for the extreme poverty class

## Almost everyone owns mobile regardless of target class mapping
fig = plt.figure(figsize = (5,5))

ax = fig.add_subplot(111)

explode = [0.2,0,0,0,0,0]

sns.countplot(cdata.lugar.map(region_mapping), data = cdata) # distribution by Region

fig = plt.figure(figsize = (5,5))

ax = fig.add_subplot(111)

plt.pie(cdata.lugar.value_counts(), labels = Region, data = cdata, explode = explode, autopct='%1.1f%%',shadow=True, startangle=90)  # distribution by Region

fig = plt.figure(figsize = (5,5))

ax = fig.add_subplot(111)

sns.countplot(cdata.Target.map(poverty_mapping),        

              hue= cdata.lugar.map(region_mapping),    # Target Distribution by Region

              data = cdata)
fig = plt.figure(figsize = (5,5))

ax = fig.add_subplot(111)

sns.countplot(cdata.area.map(area_mapping), data = cdata) #
f, axes = plt.subplots(2, 2, figsize=(7, 7))

sns.boxplot(x="Target", y="age", data=cdata, ax=axes[0, 0])

sns.boxplot(x="Target", y="meaneduc", data=cdata, ax=axes[0, 1])

sns.boxplot(x="Target", y="devices_owned", data=cdata, ax=axes[1, 0])

sns.boxplot(x="Target", y="hogar_total", data=cdata, ax=axes[1, 1])

plt.show()

## age is skewed to the right for all target classes

## meaneduc is slightly skewed to the right for all target classes

## Total Devices/ appliances owned only by the vulnerable and non-vulnerable households. Also 50% or 

## more of the data equal the maximum , hence there is no median line.Almost all of the households 

## in the target classes owns 2 or more 
### Plotting a matrix scatterplot of the numeric variables 



pd.plotting.scatter_matrix(cdata.loc[:, ["rez_esc", "meaneduc", "age",'devices_owned','hogar_total']], diagonal="kde")



plt.show()

### As expected, these features doesnt appear to have any correlation between them
fig = plt.figure(figsize = (5,5))

ax = fig.add_subplot(111)

sns.countplot(cdata.instlevel.map(education_mapping),        # Variable whose distribution is of interest

              hue= "Target",    # Distribution will be gender-wise

              data = cdata)

## Target class of Extreme and moderate poverty is directly linked to the level of education and concentrated 

## at levels of 'No education', ' Incomplete primary', 'Incomplete secondary, 'Completed primary' 
# split age into age categories

print(cdata.age.min())

print(cdata.age.max())

cdata['age_cat'] = pd.cut( cdata.age,

                          [0,7,19,30,50,80,100], 

                          include_lowest=True,

                          labels= ["Toddler","Student","Adult","Mature Adult","Senior","VSenior"]

                          )



cdata['age_cat'].value_counts()

fig = plt.figure(figsize = (5,5))

ax = fig.add_subplot(111)

cdata['age_cat'].value_counts().plot.bar()
fig = plt.figure(figsize = (5,5))

ax = fig.add_subplot(111)

sns.countplot("age_cat",        # Variable whose distribution is of interest

              hue= "Target",    # Distribution will be gender-wise

              data = cdata)

## uniform distribution of age across all Target classes
target_values = [1,2,3,4]

f, axes = plt.subplots(4, 2, figsize=(7, 7))

for j,Target in enumerate(target_values,0):

    subset = cdata[cdata['Target'] == Target]

   

    # Draw the density plot

    sns.distplot(subset['meaneduc'], hist = False, kde = True,

                  label = Target, ax = axes[j,0])

    sns.distplot(subset['rez_esc'], hist = False, kde = True,

                  label = Target, ax = axes[j,1])

plt.show()

## Mean education peaks around 6-7 years for extreme & moderate poverty levels

## Mean education peaks around 10 years for non-vulnerable
f, ax = plt.subplots(figsize=(20, 10))

sns.boxplot(x='Target', y = 'r4h3',ax = ax, data = cdata, hue = 'Target' )

ax.set_title('Number of men in households', size = 25)

plt.show()

## Median is around 2 men per household  for the 3 target classes - Extreme, moderate & Vulnerable

## Max & Median converges at 2 men per household for the non-vulnerable class
f, ax = plt.subplots(figsize=(20, 10))

sns.boxplot(x='Target', y = 'hogar_total',ax = ax, data = cdata, hue = 'Target' )

ax.set_title('Total Heads in households', size = 25)



## ## Median is around 4 men per household  for all the  target classes 

## Extracting data for the household

## Extracting columns as per Total columns

cdata.loc[(cdata['parent_level'] == 1)].Target.value_counts()

heads = cdata.loc[cdata['parent_level'] == 1, Total_columns]

print(heads.shape)  # 10307 X 57

heads.columns

heads.groupby('area')['Target'].apply(lambda x: x.isnull().sum())  ## total count of Target with null value is 7334

heads.groupby('area')['Target'].apply(lambda x: x.notnull().sum())  ## total count of Target without  null value is 2973
### separating Target for Train 

################################################################################

y_train = np.array(list(heads[heads['Target'].notnull()]['Target'].astype(np.uint8)))

print(y_train.shape )  # (2973,)
################################################################################

### separating Train & Test

################################################################################

X_train = heads.loc[heads['Target'].notnull(), :]



X_test =  heads.loc[heads['Target'].isnull(), :]

X_train.drop(object_columns, inplace = True, axis =1)

X_test.drop(object_columns, inplace = True, axis =1)





print(type(X_train))

print(type(X_test))

print(X_train.shape)  ## (2973 X 54)

print(X_test.shape)   ## (7334 X 54)
#################################################################################

##Standard Scaler for numeric and one hot encoding for categorical

##################################################################################

num = ("numtrans", ss() , data_num_columns)

colTrans = ct([num])

X_trans_num = colTrans.fit_transform(X_train)

X_trans_num.shape              # 2973 X 26

type(X_trans_num)    #numpy array

X_num = pd.DataFrame(X_trans_num, columns = data_num_columns)

X_num.shape                    #2973 X 26

#################################################################################

## One Hot encoding for the categorical columns

#############################################################################

cat = ("cattrans", ohe(), total_cat_columns)

#  Instantiate column transformer object

colTrans = ct([cat])



X_trans_cat = colTrans.fit_transform(X_train)

print(X_trans_cat.shape )             # 2973 X 99

print(type(X_trans_cat))

X_cat = pd.DataFrame(X_trans_cat.todense()).add_prefix('cat_')

## storing random projection column names

rp_columns = list(X_cat.columns)

print(X_cat.shape)
#################################################################################

#Combine the numerical & categorical arrays after transformation

#################################



X_trans = np.array(np.hstack([X_trans_num,X_trans_cat.todense()]))

print(X_trans.shape )                      #2973 X 125

print(type(X_trans))
########################################################################################

X_mtrain, X_mtest, y_mtrain, y_mtest = train_test_split(X_trans,

                                                    y_train,

                                                    test_size=0.25,

                                                    stratify = y_train,

                                                    shuffle = True

                                                    )
#######################################################################################

print(X_mtrain.shape)      # 2229 X 125

print(y_mtrain.shape)      # 2229

print(X_mtest.shape)       # 744 X 125

print(y_mtest.shape)       # 744

type(X_mtrain)

type(X_mtest)

######################################################################################
######################################################################################

### Modelling

#### Using Random Forest to baseline our 1st model

#####################################################################################

scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')

rf_model = RandomForestClassifier(n_estimators=500, random_state=10, 

                               n_jobs = -1)

# 10 fold cross validation

cv_score = cross_val_score(rf_model, X_mtrain, y_mtrain, cv = 10, scoring = scorer)



print(f'10 Fold Cross Validation F1 Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}')

rf_model.fit(X_mtrain, y_mtrain)

#feature_importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})

features = data_num_columns + rp_columns

feature_imp = pd.DataFrame({'feature':features, 'importance':rf_model.feature_importances_}).sort_values(by = "importance", ascending=False)

feature_imp

feature_imp.head()

g = sns.barplot(x = feature_imp.iloc[  :5,  1] , y = feature_imp.iloc[ :5, 0])

g.set_xticklabels(g.get_xticklabels(),rotation=90)
steps_xg = [('pca', PCA()),

            ('xg',  XGBClassifier(silent = False,

                                  n_jobs=2)        

            )

            ]



# Instantiate Pipeline object

pipe_xg = Pipeline(steps_xg)


parameters = {'xg__learning_rate':  [0.03, 0.05],

              'xg__n_estimators':   [100,  400],

              'xg__max_depth':      [4,6],

              'pca__n_components' : [0.8,0.95]

              }                             



# 7  Grid Search (16 * 2) iterations

#    Create Grid Search object 



clf = GridSearchCV(pipe_xg,            # pipeline object

                   parameters,         # possible parameters

                   n_jobs = 2,         # USe parallel cpu threads

                   cv =5 ,             # No of folds

                   verbose =2,         

                   scoring = ['accuracy'],  # Metrics for performance 

                   refit = 'accuracy'  # Refitting final model which maximise auc

                   )

                              

# 7.2.  fitting data to pipeline

start = time.time()

clf.fit(X_mtrain, y_mtrain)

end = time.time()

(end - start)/60               # 25 minutes



print(f"Best score: {clf.best_score_} ")

print(f"Best parameter set {clf.best_params_}")





# 7.4. Make predictions

y_m1pred = clf.predict(X_mtest)





# 7.5 Accuracy

accuracy = accuracy_score(y_mtest, y_m1pred)

print(f"Accuracy: {accuracy * 100.0}")



######### Accuracy with Grid Search is 68.01%   which is pretty low 
parameters = {'xg__learning_rate':  uniform(0, 1),

              'xg__n_estimators':   range(500,1000),

              'xg__max_depth':      range(3,25),

              'pca__n_components' : range(20, 30)}
rs = RandomizedSearchCV(pipe_xg,

                        param_distributions=parameters,

                        scoring= [ 'accuracy'],

                        n_iter=10,          # Max combination of

                                            # parameter to try. Default = 10

                        verbose = 3,

                        refit = 'accuracy',

                        n_jobs = 2,          # Use parallel cpu threads

                        cv = 5               # No of folds.

                                             # So n_iter * cv combinations

                        )
# random search - Using 10 iterations

start = time.time()

rs.fit(X_mtrain, y_mtrain)

end = time.time()

(end - start)/60

# Evaluate

print(f"Best score: {rs.best_score_} ")

print(f"Best parameter set: {rs.best_params_} ")





# Make predictions

y_m2pred = rs.predict(X_mtest)





# Accuracy

accuracy = accuracy_score(y_mtest, y_m2pred)

print(f"Accuracy: {accuracy * 100.0}")

## Accuracy is 66.39% 
############### FF. Fitting parameters in our model ##############

###############    Model Importance   #################



#  Model with parameters of grid search

model_gs = XGBClassifier(

                    learning_rate = clf.best_params_['xg__learning_rate'],

                    max_depth = clf.best_params_['xg__max_depth'],

                    n_estimators=clf.best_params_['xg__max_depth']

                    )



#  Model with parameters of random search

model_rs = XGBClassifier(

                    learning_rate = rs.best_params_['xg__learning_rate'],

                    max_depth = rs.best_params_['xg__max_depth'],

                    n_estimators=rs.best_params_['xg__max_depth']

                    )

#  Modeling with both parameters

start = time.time()

model_gs.fit(X_mtrain, y_mtrain)

model_rs.fit(X_mtrain, y_mtrain)

end = time.time()

(end - start)/60
#  Predictions with both models

y_pred_gs = model_gs.predict(X_mtest)

y_pred_rs = model_rs.predict(X_mtest)



#  Accuracy from both models

accuracy_gs = accuracy_score(y_mtest, y_pred_gs)

accuracy_rs = accuracy_score(y_mtest, y_pred_rs)

print("Accuracy using Grid search optimization is " ,accuracy_gs)

print("Accuracy using Random search optimization is " , accuracy_rs)



print(classification_report(y_mtest, y_pred_gs, target_names=['1','2','3','4']))

print(classification_report(y_mtest, y_pred_rs, target_names=['1','2','3','4']))

#  Get feature importances from both models



model_gs.feature_importances_

model_rs.feature_importances_

plot_importance(model_gs)

plot_importance(model_rs)

plt.show()



para_set_rf = {

           'max_features' :   (10, 20),

           'n_estimators':   (500,1000),               # any number between 500 to 1000

           'max_depth':      (5,30),                 # any depth between 5 to 30

           'n_components' :  (50,100)                 # any number between 50 to 100

            }



def xg_eval_et(max_features,n_estimators, max_depth,n_components):

    # 12.1 Make pipeline. Pass parameters directly here

    pipe_xg1 = make_pipeline (ss(),                        

                              PCA(n_components=int(round(n_components))),

                              ExtraTreesClassifier (

                                           criterion='gini',

                                           n_jobs=2,

                                           max_features=int(round(max_features)),

                                           max_depth=int(round(max_depth)),

                                           n_estimators=int(round(n_estimators))

                                           )

                             )



    # 12.2 Now fit the pipeline and evaluate

    cv_result = cross_val_score(estimator = pipe_xg1,

                                X= X_mtrain,

                                y = y_mtrain,

                                cv = 5,

                                n_jobs = 2,

                                scoring = 'f1_macro'

                                ).mean()             # take the average of all results





    # 12.3 Finally return maximum/average value of result

    return cv_result
xgBO = BayesianOptimization(

                             xg_eval_et,     # Function to evaluate performance.

                             para_set_rf     # Parameter set from where parameters will be selected

                             )
gp_params = {"alpha": 1e-5}      # Initialization parameter for gaussian
start = time.time()

xgBO.maximize(init_points=10,    # Number of randomly chosen points to

                                 # sample the target function before

                                 #  fitting the gaussian Process (gp)

                                 #  or gaussian graph

               n_iter=10,        # Total number of times the

               #acq="ucb",       # ucb: upper confidence bound

                                 #   process is to be repeated

                                 # ei: Expected improvement

               # kappa = 1.0     # kappa=1 : prefer exploitation; kappa=10, prefer exploration

              **gp_params

               )

end = time.time()

(end-start)/60
print(xgBO.max)

### Model with the best parameters

model_bet = ExtraTreesClassifier (

                    criterion='gini',

                    max_depth = 30,

                    max_features = 20,

                    n_estimators=852

                     )



start = time.time()

model_bet.fit(X_mtrain, y_mtrain)

end = time.time()

(end - start)/60



y_pred_mbet = model_bet.predict(X_mtest)

accuracy_xgB_et = accuracy_score(y_mtest, y_pred_mbet)

print("Accuracy using Bayesian Optimization is " , accuracy_xgB_et)

print("F1 macro score is: {}".format(f1_score(y_mtest, y_pred_mbet,average='macro')))

print(classification_report(y_mtest, y_pred_mbet, target_names=['1','2','3','4']))
feature_imp = pd.DataFrame({'feature':features, 'importance':model_bet.feature_importances_}).sort_values(by = "importance", ascending=False)

feature_imp

feature_imp.head()

g = sns.barplot(x = feature_imp.iloc[  :5,  1] , y = feature_imp.iloc[ :5, 0])

g.set_xticklabels(g.get_xticklabels(),rotation=90)
para_set_rf = {

           'max_features' :   (10, 20),

           'n_estimators':   (500,1000),               # any number between 500 to 1000

           'max_depth':      (5,30),                 # any depth between 3 to 10

           'n_components' :  (50,100)                 # any number between 50 to 100

            }
#  Create a function that when passed some parameters evaluates results using cross-validation

#  This function is used by BayesianOptimization() object



def xg_eval_rf(max_features,n_estimators, max_depth,n_components):

    # 12.1 Make pipeline. Pass parameters directly here

    pipe_xg1 = make_pipeline (ss(),                        

                              PCA(n_components=int(round(n_components))),

                              RandomForestClassifier (

                                           criterion='gini',

                                           n_jobs=2,

                                           max_features=int(round(max_features)),

                                           max_depth=int(round(max_depth)),

                                           n_estimators=int(round(n_estimators))

                                           )

                             )



    # fit the pipeline and evaluate

    cv_result = cross_val_score(estimator = pipe_xg1,

                                X= X_mtrain,

                                y = y_mtrain,

                                cv = 2,

                                n_jobs = 2,

                                scoring = 'f1_macro'

                                ).mean()             # take the average of all results





    #  return maximum/average value of result

    return cv_result
xgBO = BayesianOptimization(

                             xg_eval_rf,     # Function to evaluate performance.

                             para_set_rf     # Parameter set from where parameters will be selected

                             )
# 13. Gaussian process parameters

#     Modulate intelligence of Bayesian Optimization process

gp_params = {"alpha": 1e-5}      # Initialization parameter for gaussian

                                 # process.
# 14. Fit/train (so-to-say) the BayesianOptimization() object

#     Start optimization. 25minutes

#     Our objective is to maximize performance (results)

start = time.time()

xgBO.maximize(init_points=5,    # Number of randomly chosen points to

                                 # sample the target function before

                                 #  fitting the gaussian Process (gp)

                                 #  or gaussian graph

               n_iter=10,        # Total number of times the

               #acq="ucb",       # ucb: upper confidence bound

                                 #   process is to be repeated

                                 # ei: Expected improvement

               # kappa = 1.0     # kappa=1 : prefer exploitation; kappa=10, prefer exploration

              **gp_params

               )

end = time.time()

(end-start)/60
xgBO.res

print(xgBO.max)

### Model with the best parameters

model_brf = RandomForestClassifier (

                    criterion='gini',

                    max_depth = 30,

                    max_features = 19,

                    n_estimators=500

                     )



start = time.time()

model_brf.fit(X_mtrain, y_mtrain)

end = time.time()

(end - start)/60
y_pred_mbrf = model_brf.predict(X_mtest)
accuracy_xgB_rf = accuracy_score(y_mtest, y_pred_mbrf)

print("Accuracy using Bayesian Optimization is " , accuracy_xgB_rf)

print("F1 macro score is: {}".format(f1_score(y_mtest, y_pred_mbrf,average='macro')))

print(classification_report(y_mtest, y_pred_mbrf, target_names=['1','2','3','4']))
feature_imp = pd.DataFrame({'feature':features, 'importance':model_brf.feature_importances_}).sort_values(by = "importance", ascending=False)

feature_imp

feature_imp.head()

g = sns.barplot(x = feature_imp.iloc[  :5,  1] , y = feature_imp.iloc[ :5, 0])

g.set_xticklabels(g.get_xticklabels(),rotation=90)
para_set = {

           'learning_rate':  (0, 1),                 # any value between 0 and 1

           'n_estimators':   (500,1000),               # any number between 500 to 1000

           'max_depth':      (5,30),                 # any depth between 3 to 10

           'n_components' :  (50,100)                 # any number between 20 to 30

            }

# 12 Create a function that when passed some parameters

#    evaluates results using cross-validation

#    This function is used by BayesianOptimization() object



def xg_eval(learning_rate,n_estimators, max_depth,n_components):

    # 12.1 Make pipeline. Pass parameters directly here

    pipe_xg1 = make_pipeline (ss(),                        # Why repeat this here for each evaluation?

                              PCA(n_components=int(round(n_components))),

                              XGBClassifier(

                                           silent = True,

                                           objective='multi:softmax',

                                           booster='gbtree',

                                           n_jobs=2,

                                           learning_rate=learning_rate,

                                           max_depth=int(round(max_depth)),

                                           n_estimators=int(round(n_estimators))

                                           )

                             )



    # 12.2 Now fit the pipeline and evaluate

    cv_result = cross_val_score(estimator = pipe_xg1,

                                X= X_mtrain,

                                y = y_mtrain,

                                cv = 2,

                                n_jobs = 2,

                                scoring = 'f1_macro'

                                ).mean()             # take the average of all results





    # 12.3 Finally return maximum/average value of result

    return cv_result
#      Instantiate BayesianOptimization() object

#      This object  can be considered as performing an internal-loop

#      i)  Given parameters, xg_eval() evaluates performance

#      ii) Based on the performance, set of parameters are selected

#          from para_set and fed back to xg_eval()

#      (i) and (ii) are repeated for given number of iterations

#

xgBO = BayesianOptimization(

                             xg_eval,     # Function to evaluate performance.

                             para_set     # Parameter set from where parameters will be selected

                             )

gp_params = {"alpha": 1e-5}      # Initialization parameter for gaussian

                                 # process.
# 14. Fit/train  the BayesianOptimization() object

#     Start optimization. 25minutes

#     Our objective is to maximize performance (results)

start = time.time()

xgBO.maximize(init_points=5,    # Number of randomly chosen points to

                                 # sample the target function before

                                 #  fitting the gaussian Process (gp)

                                 #  or gaussian graph

               n_iter=20,        # Total number of times the

               #acq="ucb",       # ucb: upper confidence bound

                                 #   process is to be repeated

                                 # ei: Expected improvement

               # kappa = 1.0     # kappa=1 : prefer exploitation; kappa=10, prefer exploration

              **gp_params

               )

end = time.time()

(end-start)/60
xgBO.res

print(xgBO.max)

### Model with the best parameters

model_xgB =XGBClassifier(

                    learning_rate = 0.7518,

                    max_depth = 30,

                    n_estimators=503,

                    n_components = 99

                    )

start = time.time()

model_xgB.fit(X_mtrain, y_mtrain)

end = time.time()

(end - start)/60



y_pred_mxgB = model_xgB.predict(X_mtest)

accuracy_xgB = accuracy_score(y_mtest, y_pred_mxgB)

print("Accuracy using Bayesian Optimization is " , accuracy_xgB)

print("F1 macro score is: {}".format(f1_score(y_mtest, y_pred_mxgB,average='macro')))

print(classification_report(y_mtest, y_pred_mxgB, target_names=['1','2','3','4']))
feature_imp = pd.DataFrame({'feature':features, 'importance':model_xgB.feature_importances_}).sort_values(by = "importance", ascending=False)

feature_imp

feature_imp.head()

g = sns.barplot(x = feature_imp.iloc[  :5,  1] , y = feature_imp.iloc[ :5, 0])

g.set_xticklabels(g.get_xticklabels(),rotation=90)


model = lgb.LGBMClassifier(                # Using Classifier

                          objective='multiclass',

                          class_weight= 'balanced',

                          metric='None',   # This output must match with what

                                          #  we specify as input to Bayesian model

                          n_jobs=2,

                          verbose=0

                          )
# 22.1 Parameter search space for selected modeler

params = {

        'num_leaves': (5,45),              # Maximum tree leaves for base learners.

        'feature_fraction': (0.1, 0.9),	   # Randomly select part of features on each iteration

        'bagging_fraction': (0.8, 1),		  # Randomly select part of data without resampling

        'max_depth': (1, 50),              # Maximum tree depth for base learners, -1 means no limit.

        'learning_rate': (0.01, 1.0, 'log-uniform'), # Prob of interval 1 to 10 is same as 10 to 100

                                                     # Equal prob of selection from 0.01 to 0.1, 0.1

                                                     # to 1

                                                     #  Boosting learning rate.

        'min_child_samples': (1, 50),         # Minimum number of data needed in a child (leaf)

        'max_bin': (100, 1000),               # max number of bins that feature

                                              #  values will be bucketed in

                                              # small number of bins may reduce

                                              # training accuracy but may increase

                                              # general power (deal with over-fitting)

        'subsample': (0.01, 1.0, 'uniform'),  # Subsample ratio of the training instance (default: 1)

        'subsample_freq': (0, 10),            #   Frequence of subsample, <=0 means no enable (default = 0).

        'colsample_bytree': (0.01, 1.0, 'uniform'), #  Subsample ratio of columns when constructing each tree (default:1).

        'min_child_weight': (0, 10),         # Minimum sum of instance weight (hessian) needed in a child (leaf).

        'subsample_for_bin': (100000, 500000), #  Number of samples for constructing bins(default: 200000)

        'reg_lambda': (1e-9, 1000, 'log-uniform'),  # L2 regularization term on weights.

        'reg_alpha': (1e-9, 1.0, 'log-uniform'),

        'scale_pos_weight': (1e-6, 500, 'log-uniform'), #used only in binary application

                                                        # weight of labels with positive class

        'n_estimators': (500, 2000)  # Number of boosted trees to fit (default: 100).

        }

cvStrategy = StratifiedKFold(

                             n_splits=3,

                             shuffle=True,

                             random_state=42

                            )
bayes_cv_tuner = BayesSearchCV(

                              estimator = model,    # rf, lgb, xgb, nn etc--Black box

                              search_spaces = params,  # Specify params as required by the estimator

                              scoring = 'accuracy',  # Input to Bayes function

                                                    # modeler should return this

                                                    # peformence metric

                              cv = cvStrategy,      # Optional. Determines the cross-validation splitting strategy.

                                                    #           Can be cross-validation generator or an iterable,

                                                    #           Possible inputs for cv are: - None, to use the default 3-fold cv,

                                                    #           - integer, to specify the number of folds in a (Stratified)KFold,

                                                    #           - An object to be used as a cross-validation generator.

                              n_jobs = 2,           # Start two parallel threads for processing

                              n_iter = 10,        # Reduce to save time

                              verbose = 0,

                              refit = True,       #  Refit the best estimator with the entire dataset

                              random_state = 42

                               )
start = time.time()

result = bayes_cv_tuner.fit(

                           X_mtrain,       # Note that we use normal train data

                           y_mtrain       #  rather than lgb train-data matrix

                           )



end = time.time()

(end - start)/60
bayes_cv_tuner.best_estimator_



bst_bayes = bayes_cv_tuner.best_estimator_

bst_bayes

# 23.1 Train the best estimator

bst_bayes.fit(X_mtrain, y_mtrain)

#  Make predictions

y_mpred_lgbm = bst_bayes.predict(X_mtest)

accuracy_gen = accuracy_score(y_mtest, y_mpred_lgbm)

print(accuracy_gen)   # 68.68% %

print("F1 macro score is: {}".format(f1_score(y_mtest, y_mpred_lgbm,average='macro')))
print(classification_report(y_mtest, y_mpred_lgbm, target_names=['1','2','3','4']))
print('Plot feature importances...')

ax = lgb.plot_importance(bst_bayes, max_num_features=10)

ax.tick_params(labelsize=20)

plt.show()