from pyspark.sql import SparkSession

sc = SparkSession\
    .builder\
    .master("local[*]")\
    .appName('example_spark')\
    .getOrCreate()
# creating a dataframe
data = [
    (1, "Gilbert Gathara", "24", "Nairobi"), 
    (2, "Someone Else", "32", "Nowhere")
]
headers = ("id", "Name", "Age", "Location")
df = sc.createDataFrame(data, headers)
df.show()
# 1. loading and caching data
# df = sc.read\
#     .format("com.databricks.spark.csv")\
#     .options(header=True)\
#     .load(filename)\
#     .cache()

# 2. loading and caching data
# df = sc.read\
#     .format("csv")\
#     .options("header", true)\
#     .options("inferSchema", true)\
#     .load(filename)\
#     .cache()
from pyspark.sql.functions import col
# left-outer join

# df1        - dataframe being topped up.
# column_id  - similar df column to compare the join with.
# column2_id - column of interest to be added to df1.
 
# new_df = df1.alias("a")\
#     .join(df2, df1.column_id == df2.column_id, "left_outer")\
#     .select(*[col("a."+c) for c in df1.columns] + [df2.column2_id])
# count null values
# def count_null(df, col):
#     return df.where(df[col].isNull()).count()
# print("Null-count on column '{}': {}".format(col, count_null(df, col)))
# from pyspark.sql.functions import col

# double_cols - columns to be double casted.
# other_cols  - columns to be attached to resulting dataframe.

# cast multiple columns to double
# df = df.select(*[col(c).cast("double").alias(c) for c in double_cols] + other_cols)
# from pyspark.sql.functions import col, udf
# from pyspark.sql.types import IntegerType

# int_cols   - columns to be integer casted.
# other_cols - columns to be attached to resulting dataframe.

# cast multiple columns to integers using a udf
# int_udf = udf(
#     lambda r: int(r),
#     IntegerType()
# )
# df = df.select(*[int_udf(col(col_name)).name(col_name) for col_name in int_cols] + other_cols)
# from pyspark.ml.feature import StringIndexer

# StringIndexer converts string cols to numerical.
# df = StringIndexer(
#     inputCol="col_1",
#     outputCol="col_1_indx")\
#     .fit(df)\
#     .transform(df)\
#     .drop("col_1")\
#     .withColumnRenamed("col_1_indx", "col_1")
# from pyspark.ml.linalg import Vectors
# from pyspark.sql import Row

# # vectorize features + labels
# df_indexed = df_indexed[feature_cols + [label_col]]
# row = Row("label", "features")
# df_vec = df_indexed.rdd.map(
#     lambda r: (row(r[-1], Vectors.dense(r[:-1])))
# ).toDF()
# from pyspark.ml.feature import StandardScaler

# normalize features to have a mean of 0 and 
# standard deviation of 1.

# df = StandardScaler(
#     inputCol="features",
#     outputCol="features_norm",
#     withStd=True,
#     withMean=True)\
#     .fit(df)\
#     .transform(df)\
#     .drop("features")\
#     .withColumnRenamed("features_norm", "features")

