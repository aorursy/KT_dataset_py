# imports
from subprocess import check_output
from pyspark.sql import SparkSession
from pyspark.sql.functions import (count, col)
from pyspark.ml import Pipeline
from pyspark.ml.feature import (OneHotEncoderEstimator, 
                                StringIndexer, 
                                VectorAssembler)
from pyspark.ml.classification import (LogisticRegression, 
                                       DecisionTreeClassifier, 
                                       RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# load spark session
sc = SparkSession\
    .builder\
    .master("local[*]")\
    .appName('adult_census')\
    .getOrCreate()

# check for csv file(s)
print(check_output(["ls", "../input"]).decode("utf8"))
df = sc.read.load('../input/adult.csv', 
                  format='com.databricks.spark.csv', 
                  header='true', 
                  inferSchema='true').cache()
df.printSchema()
df.toPandas().head(10).transpose()
# dot-name columns
education_num = col("`education.num`")
capital_gain = col("`capital.gain`")
capital_loss = col("`capital.loss`")
hours_per_week = col("`hours.per.week`")
marital_status = col("`marital.status`")
native_country = col("`native.country`")

# rename dot-name columns
df = df.withColumn("education_num", education_num).drop(education_num)\
    .withColumn("capital_gain", capital_gain).drop(capital_gain)\
    .withColumn("capital_loss", capital_loss).drop(capital_loss)\
    .withColumn("hours_per_week", hours_per_week).drop(hours_per_week)\
    .withColumn("marital_status", marital_status).drop(marital_status)\
    .withColumn("native_country", native_country).drop(native_country)


# numerical variabels
num_vars = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
# categorical variables
cat_vars = ["workclass", "education", "marital_status", "occupation", 
            "relationship", "race", "sex", "native_country"]
# count null values in columns
def countNull(df, var):
    return df.where(df[var].isNull()).count()

all_cols = num_vars + cat_vars
{var: countNull(df, var) for var in all_cols}
# featurize categorical columns
stages = []
for cat_var in cat_vars:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(
        inputCol=cat_var, 
        outputCol=cat_var+"_indx")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    encoder = OneHotEncoderEstimator(
        inputCols=[stringIndexer.getOutputCol()], 
        outputCols=[cat_var + "_vec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]
# Convert label into label indices using the StringIndexer
label_indx = StringIndexer(
    inputCol="income", 
    outputCol="label")
stages += [label_indx]
assembler_inputs = [c+"_vec" for c in cat_vars] + num_vars
assembler = VectorAssembler(
    inputCols=assembler_inputs, 
    outputCol="features")
stages += [assembler]
# Create a Pipeline.
pipeline = Pipeline(stages=stages)

# Run the feature transformations.
#  - fit() computes feature statistics as needed.
#  - transform() actually transforms the features.
df = pipeline.fit(df).transform(df)
# Keep relevant columns for later evaluation
selectedcols = ["label", "features"] + all_cols
df = df.select(selectedcols)
df.toPandas().head(15)
(train, test) = df.randomSplit([0.8, 0.2], seed=777)
print(train.count())
print(test.count())
# 1. Logistic regression model
logr = LogisticRegression(
    maxIter = 10,
    regParam = 0.05,
    labelCol="label",
    featuresCol="features"
)

# 2. decision tree model
d_tree = DecisionTreeClassifier(
    maxDepth = 10,
    labelCol = "label",
    featuresCol="features"
)

# 3. random forest model
r_forest = RandomForestClassifier(
    numTrees = 10,
    labelCol = "label",
    featuresCol="features"
)

# fit models
lr_model = logr.fit(train)
dt_model = d_tree.fit(train)
rf_model = r_forest.fit(train)
# model evaluator
def testModel(model, df):
    pred = model.transform(df)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    return evaluator.evaluate(pred)

# accuracy output
models = {
    "Logistic regression": lr_model,
    "Decision tree": dt_model,
    "Random forest": rf_model
}

# model performance comparisson
{model_name: testModel(model, test) for model_name,model in models.items()}
# check values predicted by logistic regression
predictions = lr_model.transform(test)
predictions.printSchema()
inspect_cols = ["label", "probability", "prediction", "age", "occupation", "sex", "hours_per_week"]
predictions.select(inspect_cols).show(10)
# to-do:
#     - remove question marks from dataset
#     - normalize/standardize dataset
#     - look into uniqueness/variance/correlation/inconsistencies

