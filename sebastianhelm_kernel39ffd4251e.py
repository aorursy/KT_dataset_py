import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
filename_1 = r'/kaggle/input/rki-covid19csv/408.csv' 
filename_2 = r'/kaggle/input/rki-covid19csv/428.csv' 
filename_3 = r'/kaggle/input/rki-covid19csv/502.csv'
ds_1=pd.read_csv(filename_1, index_col='ObjectId')
ds_1
ds_1.info()
import time
ds_1['Ref📅']=ds_1['Refdatum'].apply(lambda x : time.strptime(x[:10],'%Y-%m-%d')[7])
ds_1['Mel📅']=ds_1['Meldedatum'].apply(lambda x : time.strptime(x[:10],'%Y-%m-%d')[7])
ds_1.iloc[0]
ds = ds_1.iloc[0]['Datenstand']
ds_1[ds_1['Datenstand'].apply(lambda x : x!=ds)]
ds_1=ds_1.drop(['IdBundesland','Bundesland','Landkreis','Refdatum','Meldedatum','Datenstand'],axis=1)
ds_1.info()
ds_1
Altersgruppen_as_set = set(ds_1['Altersgruppe'].tolist()) # As set so we get the unique values.
Altersgruppen = list(Altersgruppen_as_set)
Altersgruppen.sort()     # Needed to make it reproduceable. Also, it's more intuitive this way.
Altersgruppen
Geschlechter_as_set = set(ds_1['Geschlecht'].tolist())
Geschlechter = list(Geschlechter_as_set)
Geschlechter.sort()     # Needed to make it reproduceable. Also, it's more intuitive this way.
Geschlechter
ds_2=pd.read_csv(filename_2, index_col='ObjectId')
ds_2
ds_2.info()
ds_2['Ref📅']=ds_2['Refdatum'].apply(lambda x : time.strptime(x[:10],'%Y-%m-%d')[7])
ds_2['Mel📅']=ds_2['Meldedatum'].apply(lambda x : time.strptime(x[:10],'%Y-%m-%d')[7])
ds = ds_2.iloc[0]['Datenstand']
ds_2[ds_2['Datenstand'].apply(lambda x : x!=ds)]
ds_2=ds_2.drop(['IdBundesland','Bundesland','Landkreis','Refdatum','Meldedatum','Datenstand'],axis=1)
ds_2.info()
ds_2
Altersgruppen_coarse_as_set = set(ds_2['Altersgruppe'].tolist())
Altersgruppen_coarse = list(Altersgruppen_coarse_as_set)
Altersgruppen_coarse.sort()
Altersgruppen_coarse
def Age_coarse(s):
   return Altersgruppen_coarse.index(s) 
Altersgruppen_fine_as_set = set(ds_2['Altersgruppe2'].tolist())
Altersgruppen_fine = list(Altersgruppen_fine_as_set)
Altersgruppen_fine.sort()  
Altersgruppen_fine
Geschlechter_as_set = set(ds_a28['Geschlecht'].tolist())
Geschlechter = list(Geschlechter_as_set)
Geschlechter.sort()     # Needed to make it reproduceable. Also, it's more intuitive this way.
Geschlechter
def Gender(s):
    return s[:1]
ds_3=pd.read_csv(filename_3, index_col='FID')
ds_3
ds_3.info()
ds_3['Ref📅']=ds_3['Refdatum'].apply(lambda x : time.strptime(x[:10],'%Y/%m/%d')[7])
ds_3['Mel📅']=ds_3['Meldedatum'].apply(lambda x : time.strptime(x[:10],'%Y/%m/%d')[7])
ds_3.iloc[0]
ds = ds_3.iloc[0]['Datenstand']
ds_3[ds_3['Datenstand'].apply(lambda x : x!=ds)]
ds_3=ds_3.drop(['IdBundesland','Bundesland','Landkreis','Refdatum','Meldedatum','Datenstand'],axis=1)
ds_3.info()
ds_3
Altersgruppen_ds3_as_set = set(ds_3['Altersgruppe'].tolist())
Altersgruppen_ds3 = list(Altersgruppen_ds3_as_set)
Altersgruppen_ds3.sort()
Altersgruppen_ds3
Altersgruppen2_ds3_as_set = set(ds_3['Altersgruppe2'].tolist())
Altersgruppen2_ds3 = list(Altersgruppen2_ds3_as_set)
Altersgruppen2_ds3.sort()  
Altersgruppen2_ds3
Altersgruppen_fine
Altersgruppen_fine.append('Nicht übermittelt')
Altersgruppen_fine
Altersgruppen_fine
Geschlechter_as_set = set(ds_3['Geschlecht'].tolist())
Geschlechter = list(Geschlechter_as_set)
Geschlechter.sort()     # Needed to make it reproduceable. Also, it's more intuitive this way.
Geschlechter
ds_2[ds_2['Ref📅']==ds_2['Mel📅']]
ds_1['agim'] = ds_1.apply(lambda row : 
    "{}{}{}.{}".format(
        Age_coarse(row['Altersgruppe']), 
        Gender(row['Geschlecht']), 
        str(row['IdLandkreis']).zfill(5), 
        str(row['Mel📅']).zfill(3))
                              ,axis=1)
ds_1
ds_2['agim'] = ds_2.apply(lambda row : 
    "{}{}{}.{}".format(
        Age_coarse(row['Altersgruppe']), 
        Gender(row['Geschlecht']), 
        str(row['IdLandkreis']).zfill(5), 
        str(row['Ref📅']).zfill(3))
                              ,axis=1)
ds_2
ds_3['agim'] = ds_3.apply(lambda row : 
    "{}{}{}.{}".format(
        Age_coarse(row['Altersgruppe']), 
        Gender(row['Geschlecht']), 
        str(row['IdLandkreis']).zfill(5), 
        str(row['Ref📅']).zfill(3))
                              ,axis=1)
ds_3
ds_1['agim'][ds_1['agim'].duplicated(False)==True]
ds_1[:][ds_1['agim'].duplicated(False)==True]
dup_agim = ds_2[:][ds_2['agim'].duplicated(False)==True]
dup_agim
def Age_fine(s):
   return Altersgruppen_fine.index(s) 
ds_2['fgim'] = ds_2.apply(lambda row : 
    "{}{}{}.{}".format(
        Age_fine(row['Altersgruppe2']), 
        Gender(row['Geschlecht']), 
        str(row['IdLandkreis']).zfill(5), 
        str(row['Ref📅']).zfill(3))
                              ,axis=1)
ds_2
dup_fine = ds_2[:][ds_2['fgim'].duplicated(False)==True]
dup_fine
ds_2['Altersgruppe'].value_counts()
ds_2['Altersgruppe2'].value_counts()
dup_babies = dup_agim[dup_agim['Altersgruppe']=='A00-A04']
dup_babies['agim'].value_counts()
dup_agim['agim'].value_counts()