!pip install pyspark


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pyspark.sql import SparkSession



spark = SparkSession.builder.appName('ml-bank').getOrCreate()

sdf = spark.read.csv('../input/bankbalanced/bank.csv', header = True, inferSchema = True)

sdf.printSchema()
import pandas as pd



pd.DataFrame(sdf.take(5), columns=sdf.columns).transpose()
sdf.toPandas().groupby(['deposit']).size()
numeric_features = [t[0] for t in sdf.dtypes if t[1] == 'int']

sdf.select(numeric_features).describe().toPandas().transpose()


numeric_data = sdf.select(numeric_features).toPandas()

axs = pd.plotting.scatter_matrix(numeric_data, figsize=(8, 8));

n = len(numeric_data.columns)



for i in range(n):

    v = axs[i, 0]

    v.yaxis.label.set_rotation(0)

    v.yaxis.label.set_ha('right')

    v.set_yticks(())

    h = axs[n-1, i]

    h.xaxis.label.set_rotation(90)

    h.set_xticks(())
sdf = sdf.select(

    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 

    'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit'

)

cols = sdf.columns

sdf.printSchema()
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler



stages = []

categoricalColumns = [

    'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome'

]



for categoricalCol in categoricalColumns:

    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')

    encoder = OneHotEncoderEstimator(

        inputCols=[stringIndexer.getOutputCol()], 

        outputCols=[categoricalCol + "classVec"]

    )

    stages += [stringIndexer, encoder]

    

label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')

stages += [label_stringIdx]

numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

stages += [assembler]
from pyspark.ml import Pipeline



pipeline = Pipeline(stages = stages)

pipelineModel = pipeline.fit(sdf)

sdf = pipelineModel.transform(sdf)

selectedCols = ['label', 'features'] + cols

sdf = sdf.select(selectedCols)

sdf.printSchema()
pdf = pd.DataFrame(sdf.take(5), columns=sdf.columns)

pdf.iloc[:,0:2] 

len(pdf.features[0])
train, test = sdf.randomSplit([0.7, 0.3], seed = 2018)

print("Training Dataset Count: " + str(train.count()))

print("Test Dataset Count: " + str(test.count()))
from pyspark.ml.classification import LogisticRegression



lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)

lrModel = lr.fit(train)
import matplotlib.pyplot as plt

import numpy as np



beta = np.sort(lrModel.coefficients)

plt.plot(beta)

plt.ylabel('Beta Coefficients')

plt.show()
trainingSummary = lrModel.summary

lrROC = trainingSummary.roc.toPandas()



plt.plot(lrROC['FPR'],lrROC['TPR'])

plt.ylabel('False Positive Rate')

plt.xlabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()



print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
pr = trainingSummary.pr.toPandas()

plt.plot(pr['recall'],pr['precision'])

plt.ylabel('Precision')

plt.xlabel('Recall')

plt.show()
lrPreds = lrModel.transform(test)

lrPreds.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
lrPreds.show()
from pyspark.ml.evaluation import BinaryClassificationEvaluator



lrEval = BinaryClassificationEvaluator()

print('Test Area Under ROC', lrEval.evaluate(lrPreds))
from pyspark.ml.classification import DecisionTreeClassifier



dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)

dtModel = dt.fit(train)

dtPreds = dtModel.transform(test)

dtPreds.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
dtEval = BinaryClassificationEvaluator()

dtROC = dtEval.evaluate(dtPreds, {dtEval.metricName: "areaUnderROC"})

print("Test Area Under ROC: " + str(dtROC))
from pyspark.ml.classification import RandomForestClassifier



rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')

rfModel = rf.fit(train)

rfPreds = rfModel.transform(test)

rfPreds.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
rfEval = BinaryClassificationEvaluator()

rfROC = rfEval.evaluate(rfPreds, {rfEval.metricName: "areaUnderROC"})

print("Test Area Under ROC: " + str(rfROC))
from pyspark.ml.classification import GBTClassifier



gbt = GBTClassifier(maxIter=10)

gbtModel = gbt.fit(train)

gbtPreds = gbtModel.transform(test)

gbtPreds.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
gbtEval = BinaryClassificationEvaluator()

gbtROC = gbtEval.evaluate(gbtPreds, {gbtEval.metricName: "areaUnderROC"})

print("Test Area Under ROC: " + str(gbtROC))
print(gbt.explainParams())
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator



paramGrid = (ParamGridBuilder()

             .addGrid(gbt.maxDepth, [2, 4, 6])

             .addGrid(gbt.maxBins, [20, 60])

             .addGrid(gbt.maxIter, [10, 20])

             .build())



cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=gbtEval, numFolds=5)



# Run cross validations.  

# This can take some minutes since it is training over 20 trees!

cvModel = cv.fit(train)

cvPreds = cvModel.transform(test)

gbtEval.evaluate(cvPreds)