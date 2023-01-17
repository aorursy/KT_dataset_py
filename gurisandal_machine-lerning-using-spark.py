import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/heart.csv")
df.head()
df.target.value_counts()
sns.countplot(x="target", data=df)

plt.show()
NoDisease = len(df[df.target==0])

HaveDisease = len(df[df.target==1])

print("%age of Patient have not heart disease: {:.2f}%".format((NoDisease/(len(df.target)))*100))

print("%age of Patient have heart disease: {:.2f}%".format((HaveDisease/(len(df.target)))*100))
sns.countplot(x='sex', data=df)

plt.xlabel("Sex (0 = female, 1 = male)")

plt.show()
Female = len(df[df.sex==0])

Male = len(df[df.sex==1])

print("%age of female: {:.2f}%".format((Female/(len(df.sex)))*100))

print("%age of male: {:.2f}%".format((Male/(len(df.sex)))*100))
df.groupby('target').mean()
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")

plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
f,ax = plt.subplots(figsize=(12,12))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
!pip install pyspark
import pyspark

sc = pyspark.SparkContext(appName="Heart")
from pyspark.sql import SQLContext



sqlContext = SQLContext(sc)

sdf = sqlContext.createDataFrame(df)
sdf.show()
sdf.dtypes
sdf.printSchema()
sdf.groupBy("age").count().sort("age",ascending=False).show()
from pyspark.ml.classification import LogisticRegression

sdf.printSchema()
from pyspark.ml.linalg import Vectors

from pyspark.ml.feature import VectorAssembler

 

assembler=VectorAssembler(inputCols=['age','sex','cp','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal'],outputCol='features')

 

output_data=assembler.transform(sdf)
output_data.printSchema()
final_data=output_data.select('features','target')         

train,test=final_data.randomSplit([0.7,0.3])          

model=LogisticRegression(labelCol='target')           

model=model.fit(train)        

summary=model.summary

summary.predictions.describe().show()   
from pyspark.ml.evaluation import BinaryClassificationEvaluator

 

predictions=model.evaluate(test)

evaluator=BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='target')

acc = evaluator.evaluate(predictions.predictions)

print("Accuracy = ",acc*100)
sc.stop()