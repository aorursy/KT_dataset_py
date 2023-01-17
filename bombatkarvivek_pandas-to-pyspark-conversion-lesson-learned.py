! pip install pyspark


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

from pyspark.sql import SparkSession

from pyspark.sql import functions as F
spark = SparkSession.builder.master("local[*]").config('spark.driver.memory','15g').getOrCreate()

spark
print(os.listdir("../input/vehicle/"))

PATH = '../input/vehicle/vehicle.csv'
df = pd.read_csv(PATH)

# type(df) #pandas.core.frame.DataFrame



print(df.dtypes)

print(df.columns)

print(df.index.values)

df.T
sdf = spark.read.csv(PATH, inferSchema=True, header=True)

print(sdf.printSchema()) # Transformation (well depend on the data source!!!)

sdf.show() # Action
df.to_csv('vehicle_pdf.csv',index= False)

# print( os.listdir('.'))
sdf.write.csv("vehicle_sdf",mode='overwrite', header=True) # .coalesce(1)

print( os.listdir('.'))

print( os.listdir('vehicle_sdf/'))

df['new_col'] = df['max_torque'] * df['max_horsepower']

df.T
df.drop(['drivetrain','max_torque','max_horsepower'],inplace= True,axis=1)

df.T
#ImMutable 

sdf_new_col = sdf.withColumn('new_col', (sdf['max_torque'] * sdf['max_horsepower']) )

sdf_new_col.printSchema()
sdf_drop_cols = sdf.drop('drivetrain','max_torque','max_horsepower')

sdf_drop_cols.printSchema()
df.T
df['forced_induction'] = df['forced_induction'].fillna(df['forced_induction'].mean())

df.T
#: java.lang.IllegalArgumentException: requirement failed: Column forced_induction must be of type equal to one of the following types: [double, float] but was actually of type string.

# mean (the default imputation strategy)

from pyspark.ml.feature import Imputer



# cast forced_induction to double

sdf_modified = sdf.withColumn('forced_induction',sdf['forced_induction'].cast("double"))



imputer = Imputer(

    inputCols=['forced_induction'], 

    outputCols=["forced_induction_mean"]

#     Strategy("mean")

)



sdf_modified_mean = imputer.fit(sdf_modified).transform(sdf_modified)





# sdf_mean_value = sdf.withColumn('forced_induction', F. mean(sdf['forced_induction']) )

sdf_modified_mean.toPandas().T
df['my_car'] = np.where(df['make'] == 'BMW','Y','N')

df.T
sdf_new_col = sdf.withColumn('my_car', F.when(sdf['make'] == 'BMW', 'Y').otherwise('N'))

sdf_new_col.toPandas().T
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

df_sorted = pd.read_csv(PATH)



df_sorted.sort_values(by=['max_torque']).reset_index(drop=True)

assert_frame_equal(df,df_sorted)
# In pyspark create a function with accept SDF and return SDF, then it only abt comparing dataframes.

sdf_compare = sdf.subtract(sdf_modified)

sdf_compare.show()