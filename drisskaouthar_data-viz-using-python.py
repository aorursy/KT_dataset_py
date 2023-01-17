import pandas as pd

my_data = pd.read_csv('../input/output_sas_comp.csv')
import pandas as pd 
data_sas1 = pd.read_csv('../input/output_sas_comp.csv', sep=',',nrows=50)
print (data_sas1)
type(data_sas1)
##loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Dimension of the dataset
data_sas1.shape
data_sas1.dtypes


cols = data_sas1.columns[data_sas1.dtypes.eq(object)]
cols



#check wheter any of the columns contain null values
data_sas1.isnull().sum()

sns.lmplot(x = 'Population', y= 'Incidence', hue = 'diff',data = data_sas1)