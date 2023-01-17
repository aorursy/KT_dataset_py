!pip install pyspark
# pandas, numpy

import pandas as pd

import numpy as np
# pyspark.sql

from pyspark.sql import SparkSession, DataFrame

import pyspark.sql.types as types

import pyspark.sql.functions as f
# pyspark.ml

import pyspark.ml as ml



import pyspark.ml.feature as feature

import pyspark.ml.regression as regression

import pyspark.ml.classification as classification

import pyspark.ml.tuning as tuning

import pyspark.ml.evaluation as evaluation
# regex

import re
# visualization

import matplotlib.pyplot as plt

import seaborn as sns
# For reduce

import functools as ft
# Global random seed

SEED = 10
# Print Spark DataFrame schema, column per line

def print_schema(df):

    for item in df.schema:

        print(item)
# How many DataFrame has null values and in which columns

def null_values_count(df):

    df = df.select([f.count(f.when(f.isnan(c) | f.col(c).isNull(), c)).alias(c) for c in df.columns])

    return df.select([x for x in [None if df.collect()[0][c] == 0 else c for c in df.columns] if x is not None])
# Get median by DF column

def median(values_list):

    med = np.median(values_list)

    return float(med)
spark = SparkSession.builder.appName('Titanic').getOrCreate()
train_df_eda = spark.read.csv('../input/titanic/train.csv', header = 'True', inferSchema='True')

test_df_eda = spark.read.csv('../input/titanic/test.csv', header = 'True', inferSchema='True')
train_df_eda.show(10)
print_schema(train_df_eda)
null_values_count(train_df_eda).show()
null_values_count(test_df_eda).show()
train_df_eda.count()
train_df_eda.columns
train_df_eda.select('Age').distinct().show()
sns.distplot(train_df_eda.select('Survived').toPandas(), kde=False)
train_df_eda.groupBy('Sex', 'Survived').count().show()
train_df_eda.groupBy('Embarked', 'Survived').count().show()
train_df_eda.groupBy('Pclass', 'Survived').count().show()
sns.swarmplot(x='Survived', y='Age', data=train_df_eda.select('Age', 'Survived').toPandas())
sns.swarmplot(x='Survived', y='Fare', data=train_df_eda.select('Fare', 'Survived').toPandas())
family_size_df = train_df_eda.withColumn('Family_Size', f.col('Parch') + f.col('SibSp') + 1).select('Family_Size', 'Survived')

sns.swarmplot(x='Survived', y='Family_Size', data=family_size_df.toPandas())
train_df_eda_pd = train_df_eda.toPandas()
train_df_eda_pd.loc[train_df_eda_pd['Sex'] == 'male', 'Sex_i'] = 0

train_df_eda_pd.loc[train_df_eda_pd['Sex'] == 'female', 'Sex_i'] = 1
train_df_eda_pd.loc[train_df_eda_pd['Embarked'] == 'S', 'Embarked_i'] = 0

train_df_eda_pd.loc[train_df_eda_pd['Embarked'] == 'C', 'Embarked_i'] = 1

train_df_eda_pd.loc[train_df_eda_pd['Embarked'] == 'Q', 'Embarked_i'] = 2
sns.heatmap(np.abs(train_df_eda_pd.corr()), annot=True)
nulls = null_values_count(train_df_eda).collect()[0]
nulls['Age']/train_df_eda.count()
nulls['Cabin']/train_df_eda.count()
nulls['Embarked']/train_df_eda.count()
nulls = null_values_count(test_df_eda).collect()[0]
nulls['Fare']/test_df_eda.count()
full_df_eda = train_df_eda.unionByName(test_df_eda.withColumn('Survived', f.lit(None)))
title_regex = '([\w]+)\.'
full_df_eda = full_df_eda.withColumn('Title', f.regexp_extract(full_df_eda['Name'], title_regex, 1))
full_df_eda.select('Title').distinct().show()
class TitleTransformer(ml.Transformer):

    def _transform(self, df):

        df = df.withColumn('Title', f.regexp_extract(df['Name'], title_regex, 1))

        df = df.replace(['Mlle','Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr', 'Mrs',  'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mrs'])

        return df
title_unique = TitleTransformer().transform(full_df_eda).select('Title').distinct()
title_unique.show()
null_values_count(full_df_eda).show()
class FareAndEmbarkedInputtingTransformer(ml.Transformer):

    def _transform(self, df):

        embarked_mode = df.groupby('Embarked').count().orderBy('count', ascending=False).first()[0]

        fare_mean = df.agg(f.avg(f.col('Fare'))).collect()[0][0]



        df = df.na.fill({'Fare' : fare_mean, 'Embarked' : embarked_mode})    



        return df
null_values_count(FareAndEmbarkedInputtingTransformer().transform(full_df_eda)).show()
class FamilySizeTransformer(ml.Transformer):

    def _transform(self, df):

        df = df.withColumn('Family_size', f.col('SibSp') + f.col('Parch') + 1)

        df = df.withColumn('Is_alone', f.when(df['Family_size'] == 1, 1).otherwise(f.lit(0)))

        return df
FamilySizeTransformer().transform(full_df_eda).select('SibSp', 'Parch', 'Family_size', 'Is_alone').show(10)
full_df_eda = FamilySizeTransformer().transform(full_df_eda)
full_df_eda = full_df_eda.withColumn('Deck', f.when(full_df_eda["Cabin"].isNotNull(), full_df_eda["Cabin"].substr(1,1)).otherwise('Unknown'))
sns.set(rc={'figure.figsize':(12, 12)})

def show_count_of_pclass_by_deck(df):

    tmp = full_df_eda.groupby('Pclass', 'Deck').count()

    

    cols = tmp.select('Deck').distinct().sort(f.col('Deck').asc())  

    

    for i in range(cols.count()):

        plt.subplot(np.ceil(cols.count()/2), 2, i+1)

        sns.barplot(data=tmp.filter(f.col('Deck') == cols.collect()[i][0]).toPandas(), x='Pclass', y='count').set_title(cols.collect()[i][0])

        plt.tight_layout()
show_count_of_pclass_by_deck(full_df_eda)
class DeckTransformer(ml.Transformer):

    def _transform(self, df):

        df = df.withColumn('Deck', f.when(df["Cabin"].isNotNull(), df["Cabin"].substr(1,1)).otherwise('Unknown'))

        df = df.withColumn('Deck', f.when(df["Deck"].isin(['A', 'B', 'C']), 'ABC').when(df["Deck"].isin(['D', 'E']), 'DE').when(df["Deck"].isin(['F', 'G']), 'FG').otherwise(df["Deck"]))



        return df
DeckTransformer().transform(full_df_eda).select('Cabin', 'Deck').show(10)
class DiscretizerEstimator(ml.Estimator):

    def _fit(self, df):

        fare_disc = feature.QuantileDiscretizer(inputCol='Fare', outputCol='Fare_quant', numBuckets=10).fit(df)

        

        pipeline = ml.Pipeline(stages=[fare_disc]).fit(df)        

        return pipeline
DiscretizerEstimator().fit(full_df_eda).transform(full_df_eda).select('Fare', 'Fare_quant').show()
full_df_eda.groupby('Ticket').count().sort(f.col('count').desc()).show(10)
class TicketGroupTransform(ml.Transformer):

    def _transform(self, df):

        ticket_groups = df.groupBy('Ticket').count()

        ticket_groups = ticket_groups.withColumnRenamed('count', 'Ticket_group_count')

        return df.join(ticket_groups, 'Ticket')
TicketGroupTransform().transform(full_df_eda).select('Ticket', 'Ticket_group_count').filter(f.col('Ticket') == 'CA. 2343').show()
full_df_eda = TicketGroupTransform().transform(full_df_eda)
class StringIndexEstimator(ml.Estimator):

    def _fit(self, df):

        sex_ind = feature.StringIndexer(inputCol='Sex', outputCol="Sex_index")

        embarked_ind = feature.StringIndexer(inputCol='Embarked', outputCol="Embarked_index")

        title_ind = feature.StringIndexer(inputCol='Title', outputCol="Title_index")

        deck_ind = feature.StringIndexer(inputCol='Deck', outputCol="Deck_index")

        

        pipeline = ml.Pipeline(

            stages=

            [sex_ind, embarked_ind, title_ind, deck_ind]  

        ).fit(df)        

        return pipeline
StringIndexEstimator().fit(full_df_eda).transform(full_df_eda).select('Sex', 'Sex_index', 'Title', 'Title_index').show(10)
def median(values_list):

    med = np.median(values_list)

    return float(med)
family_regex = '.+?(?=,)'

punctuation_regex = '[.,\/#"!$%\^&\*;:{}=\-_`~()]'



class SurvivalRateTransformer(ml.Transformer):

    def __init__(self, first_test_passenger_id):

        self.first_test_passenger_id = first_test_passenger_id

        

    def _transform(self, full_df):

        train_df_n = full_df.filter(f.col('PassengerId') < self.first_test_passenger_id)

        test_df_n = full_df.filter(f.col('PassengerId') >= self.first_test_passenger_id)

        

        train_df_n = train_df_n.withColumn('Family', f.regexp_extract(train_df_n['Name'], family_regex, 0))

        train_df_n = train_df_n.withColumn('Family', f.regexp_replace('Family', punctuation_regex, ''))

        

        test_df_n = test_df_n.withColumn('Family', f.regexp_extract(test_df_n['Name'], family_regex, 0))

        test_df_n = test_df_n.withColumn('Family', f.regexp_replace('Family', punctuation_regex, ''))

        

        non_unique_families_s = train_df_n.select('Family').distinct().join(test_df_n.select('Family').distinct(), 'Family', 'inner')

        non_unique_tickets_s = train_df_n.select('Ticket').distinct().join(test_df_n.select('Ticket').distinct(), 'Ticket', 'inner')

        

        udf_median = f.udf(median, types.FloatType())

        

        df_family_survival_rate_s = (

        train_df_n.groupby(['Family']).agg(udf_median(f.collect_list(f.col('Survived'))).alias('Survived'), 

                                       udf_median(f.collect_list(f.col('Family_size'))).alias('Family_size')

                                      )

        )

        df_family_survival_rate_s = df_family_survival_rate_s.sort(df_family_survival_rate_s['Family'])

        

        

        df_ticket_survival_rate_s = (

        train_df_n.groupby(['Ticket']).agg(udf_median(f.collect_list(f.col('Survived'))).alias('Survived'), 

                                           udf_median(f.collect_list(f.col('Ticket_group_count'))).alias('Ticket_group_count')

                                          )

        )

        df_ticket_survival_rate_s = df_ticket_survival_rate_s.sort(df_ticket_survival_rate_s['Ticket'])

        

        family_rates_s = df_family_survival_rate_s.join(non_unique_families_s, 'Family', 'inner').withColumn('Rate', f.when(f.col('Family_size') > 1, f.col('Survived'))).select('Family', 'Rate').filter(f.col('Rate').isNotNull())

        ticket_rates_s = df_ticket_survival_rate_s.join(non_unique_tickets_s, 'Ticket', 'inner').withColumn('Rate', f.when(f.col('Ticket_group_count') > 1, f.col('Survived'))).select('Ticket', 'Rate').filter(f.col('Rate').isNotNull())

        

        mean_survival_rate_s = train_df_n.agg({'Survived': 'avg'}).collect()[0][0]

        

        train_family_survival_rate_s = train_df_n.join(family_rates_s, 'Family', 'left').withColumnRenamed('Rate', 'Family_rate').select('PassengerId', 'Family_rate')

        train_family_survival_rate_s = train_family_survival_rate_s.withColumn('Family_NA', f.when(f.col('Family_rate').isNotNull(), 1).otherwise(0))

        train_family_survival_rate_s = train_family_survival_rate_s.withColumn('Family_rate', f.when(f.col('Family_rate').isNotNull(), f.col('Family_rate')).otherwise(mean_survival_rate_s))



        test_family_survival_rate_s = test_df_n.join(family_rates_s, 'Family', 'left').withColumnRenamed('Rate', 'Family_rate').select('PassengerId', 'Family_rate')

        test_family_survival_rate_s = test_family_survival_rate_s.withColumn('Family_NA', f.when(f.col('Family_rate').isNotNull(), 1).otherwise(0))

        test_family_survival_rate_s = test_family_survival_rate_s.withColumn('Family_rate', f.when(f.col('Family_rate').isNotNull(), f.col('Family_rate')).otherwise(mean_survival_rate_s))





        train_ticket_survival_rate_s = train_df_n.join(ticket_rates_s, 'Ticket', 'left').withColumnRenamed('Rate', 'Ticket_rate').select('PassengerId', 'Ticket_rate')

        train_ticket_survival_rate_s = train_ticket_survival_rate_s.withColumn('Ticket_NA', f.when(f.col('Ticket_rate').isNotNull(), 1).otherwise(0))

        train_ticket_survival_rate_s = train_ticket_survival_rate_s.withColumn('Ticket_rate', f.when(f.col('Ticket_rate').isNotNull(), f.col('Ticket_rate')).otherwise(mean_survival_rate_s))



        test_ticket_survival_rate_s = test_df_n.join(ticket_rates_s, 'Ticket', 'left').withColumnRenamed('Rate', 'Ticket_rate').select('PassengerId', 'Ticket_rate')

        test_ticket_survival_rate_s = test_ticket_survival_rate_s.withColumn('Ticket_NA', f.when(f.col('Ticket_rate').isNotNull(), 1).otherwise(0))

        test_ticket_survival_rate_s = test_ticket_survival_rate_s.withColumn('Ticket_rate', f.when(f.col('Ticket_rate').isNotNull(), f.col('Ticket_rate')).otherwise(mean_survival_rate_s))





        avg_cols = f.udf(lambda array: sum(array)/len(array), types.DoubleType())



        surv_rate_train = (

            train_family_survival_rate_s

            .join(train_ticket_survival_rate_s, 'PassengerId', 'left')

            .withColumn('Survival_Rate', avg_cols(f.array('Family_rate', 'Ticket_rate')))

            .withColumn('Survival_Rate_NA', avg_cols(f.array('Family_NA', 'Ticket_NA')))

            .select('PassengerId', 'Survival_Rate', 'Survival_Rate_NA')

        )



        surv_rate_test = (

            test_family_survival_rate_s

            .join(test_ticket_survival_rate_s, 'PassengerId', 'left')

            .withColumn('Survival_Rate', avg_cols(f.array('Family_rate', 'Ticket_rate')))

            .withColumn('Survival_Rate_NA', avg_cols(f.array('Family_NA', 'Ticket_NA')))

            .select('PassengerId', 'Survival_Rate', 'Survival_Rate_NA')

        )



        train_df_n = train_df_n.join(surv_rate_train, 'PassengerId', 'left')

        test_df_n = test_df_n.join(surv_rate_test, 'PassengerId', 'left')



        return train_df_n.union(test_df_n)
SurvivalRateTransformer(test_df_eda.select(f.min(f.col('PassengerId'))).limit(1).collect()[0][0]).transform(full_df_eda).select('Survived', 'Survival_Rate', 'Survival_Rate_NA').show(10)
mutual_columns = ['Pclass', 'SibSp', 'Parch', 'Family_size', 'Is_alone', 'Ticket_group_count', 'Fare_quant', 'Deck_index', 'Sex_index', 'Embarked_index', 'Title_index', 'Survival_Rate', 'Survival_Rate_NA']

columns_for_age_pred = [*mutual_columns]

columns_for_pred = ['Age', *mutual_columns]
class SelectFeaturesTransformer(ml.Transformer):

    def _transform(self, df):

        df = df.select(['PassengerId', 'Survived'] + columns_for_pred)

        return df
full_df = (

    spark.read.csv('../input/titanic/train.csv', header = 'True', inferSchema='True').unionByName(

        spark.read.csv('../input/titanic/test.csv', header = 'True', inferSchema='True').withColumn('Survived', f.lit(None)))

)
first_test_passenger_id = test_df_eda.select(f.min(f.col('PassengerId'))).limit(1).collect()[0][0]
pipeline = ml.Pipeline(stages=[

    TitleTransformer(), 

    FareAndEmbarkedInputtingTransformer(),

    FamilySizeTransformer(), 

    DeckTransformer(), 

    TicketGroupTransform(),

    StringIndexEstimator(),

    DiscretizerEstimator(),

    SurvivalRateTransformer(first_test_passenger_id),

    SelectFeaturesTransformer()

])

pipeline = pipeline.fit(full_df)
full_df = pipeline.transform(full_df)
train_df = full_df.filter(f.col('PassengerId') < first_test_passenger_id)

test_df = full_df.filter(f.col('PassengerId') >= first_test_passenger_id).drop('Survived')
first_test_passenger_id
age_feature_transformer = feature.VectorAssembler(inputCols=columns_for_age_pred, outputCol='Age_features')

train_df_with_age_features = age_feature_transformer.transform(train_df.filter(train_df['Age'].isNotNull()))
age_scaler = feature.StandardScaler(inputCol='Age_features', outputCol='Age_features_sc').fit(train_df_with_age_features)

train_df_with_age_features = age_scaler.transform(train_df_with_age_features)

train_df_with_age_features = train_df_with_age_features.drop('Age_features').withColumnRenamed('Age_features_sc', 'Age_features')
(train_df_with_age_features_train, train_df_with_age_features_val) = train_df_with_age_features.randomSplit([0.8, 0.2], seed=SEED)
age_pred_evalutor = evaluation.RegressionEvaluator(labelCol='Age', predictionCol='Age_pred', metricName='r2')
titles = train_df_with_age_features.select('Title_index').distinct().sort(f.col('Title_index').desc()).collect()
titles = [x.Title_index for x in titles]

titles
cv_dfs = [None] * 5

(cv_dfs[0], cv_dfs[1], cv_dfs[2], cv_dfs[3], cv_dfs[4]) = train_df_with_age_features.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed=SEED) 
ind_set = set(range(0, 5))
r2_res = []



for val_ind in range(0, 5):

    train_ind = ind_set - {val_ind}    

    tr_df = ft.reduce(DataFrame.unionAll, [cv_dfs[i] for i in train_ind])    

    

    val_df = cv_dfs[val_ind].withColumn('Age_pred', f.lit(None).cast(types.FloatType()))



    mean_ages = val_df.groupby('Title_index').agg(f.avg(f.col('Age')))

    mean_ages = mean_ages.toPandas().set_index('Title_index').T.to_dict('list')

        

    for title in titles:

        

        if title in mean_ages:

            mean_age = mean_ages[title][0]

        else:

            mean_age = tr_df.agg({'Age': 'avg'}).collect()[0][0]

        

        val_df = val_df.withColumn('Age_pred', f.when(

            (val_df['Title_index'] == title), 

            mean_age

        ).otherwise(val_df['Age_pred']))

        

    r2 = age_pred_evalutor.evaluate(val_df)

    print(r2)

    r2_res.append(r2)

    

print(f'Cross-validated R^2 score: {np.mean(r2_res)}')
r2_res = []

for val_ind in range(0, 5):

    train_ind = ind_set - {val_ind}    

    tr_df = ft.reduce(DataFrame.unionAll, [cv_dfs[i] for i in train_ind])    

    val_df = cv_dfs[val_ind]

    

    gbtr = regression.RandomForestRegressor(labelCol='Age', featuresCol='Age_features', predictionCol='Age_pred', seed=SEED)

    val_df = gbtr.fit(tr_df).transform(val_df)

    

    r2 = age_pred_evalutor.evaluate(val_df)

    print(r2)

    r2_res.append(r2)

    

print(f'Cross-validated R^2 score: {np.mean(r2_res)}')
rfc_age = regression.RandomForestRegressor(labelCol='Age', featuresCol='Age_features', predictionCol='Age_pred', seed=SEED)

rfc_age = rfc_age.fit(train_df_with_age_features)
train_df = age_feature_transformer.transform(train_df)
train_df = rfc_age.transform(train_df)
train_df = train_df.withColumn('Age', f.when(

    (train_df['Age'].isNull()), 

    train_df['Age_pred']

).otherwise(train_df["Age"]))
train_df = train_df.drop('Age_pred', 'Age_features')
feature_transformer = feature.VectorAssembler(inputCols=columns_for_pred, outputCol="features")
feature_vector = feature_transformer.transform(train_df)
(train_data, test_data) = feature_vector.randomSplit([0.8, 0.2], seed = SEED) 
scaler = feature.StandardScaler(inputCol='features', outputCol='features_sc').fit(train_data)

train_data = scaler.transform(train_data)

train_data = train_data.drop('features').withColumnRenamed('features_sc', 'features')



test_data = scaler.transform(test_data)

test_data = test_data.drop('features').withColumnRenamed('features_sc', 'features')
evaluator = evaluation.MulticlassClassificationEvaluator(labelCol='Survived', predictionCol='prediction', metricName='accuracy')
# gbtc = classification.GBTClassifier(labelCol='Survived', featuresCol='features', seed=SEED)
# paramGrid = (tuning.ParamGridBuilder()

#              .addGrid(gbtc.maxIter, [10, 15, 20, 25, 30, 40])

#              .addGrid(gbtc.maxBins, [16, 23, 32, 48, 64, 96])

#              .addGrid(gbtc.maxDepth, [2, 3, 5, 7, 9, 12])

#              .build())



# crossval = tuning.CmachinerossValidator(estimator=gbtc,

#                           estimatorParamMaps=paramGrid,

#                           evaluator=evaluator,

#                           seed=SEED,

#                           numFolds=10)



# cvModel = crossval.fit(train_data)
# cvModel.bestModel.extractParamMap()
# gbtc = classification.GBTClassifier(labelCol='Survived', featuresCol='features', seed=SEED, maxIter=20, maxBins=32, maxDepth=5)

# val_df = gbtc.fit(trainingData).transform(test_data)



# evaluator.evaluate(val_df)
# rfc = classification.RandomForestClassifier(labelCol='Survived', featuresCol='features', seed=SEED)
# paramGrid = (tuning.ParamGridBuilder()

#              .addGrid(rfc.maxBins, [8, 16, 32, 48, 64, 96])

#              .addGrid(rfc.numTrees, [15, 20, 25, 30, 35, 40])

#              .addGrid(rfc.maxDepth, [2, 5, 7, 12, 16, 20])

#              .addGrid(rfc.minInstancesPerNode, [2, 3, 4, 5, 6])

#              .build())



# crossval = tuning.CrossValidator(estimator=rfc,

#                           estimatorParamMaps=paramGrid,

#                           evaluator=evaluator,

#                           seed=SEED,

#                           numFolds=10)



# cvModel = crossval.fit(train_data)
# cvModel.bestModel.extractParamMap()
# rfc = classification.RandomForestClassifier(labelCol='Survived', featuresCol='features', seed=SEED, maxBins=32, numTrees=20, maxDepth=5, minInstancesPerNode=1)

# val_df = rfc.fit(trainingData).transform(test_data)



# evaluator.evaluate(val_df)
rfc_model = classification.RandomForestClassifier(labelCol='Survived', featuresCol='features', seed=SEED, numTrees=20, maxDepth=5, minInstancesPerNode=2)

rfc_model = rfc_model.fit(train_data)

rfc_prediction = rfc_model.transform(test_data)

evaluator.evaluate(rfc_prediction)
pred_df = test_df
pred_df = age_feature_transformer.transform(pred_df)
pred_df = age_scaler.transform(pred_df)

pred_df = pred_df.drop('Age_features').withColumnRenamed('Age_features_sc', 'Age_features')
pred_df = rfc_age.transform(pred_df)
pred_df = pred_df.withColumn('Age', f.when(

    (f.col('Age').isNull()), 

    f.col('Age_pred')

).otherwise(f.col('Age')))
pred_df = pred_df.drop('Age_pred', 'Age_features')
pred_df = feature_transformer.transform(pred_df)
pred_df = scaler.transform(pred_df)

pred_df = pred_df.drop('features').withColumnRenamed('features_sc', 'features')
prediction = rfc_model.transform(pred_df) 

prediction = prediction.withColumnRenamed('prediction', 'Survived').select(['PassengerId', 'Survived'])

prediction = prediction.withColumn('Survived', prediction['Survived'].cast(types.IntegerType()))
prediction.toPandas().to_csv('submission.csv', index=False)