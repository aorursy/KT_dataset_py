! pip install pyspark
from pyspark.sql import SparkSession, DataFrame, functions as F

from pyspark.ml.feature import Imputer, StringIndexer, VectorIndexer, VectorAssembler, OneHotEncoderEstimator, PCA, Bucketizer

from pyspark.ml.classification import RandomForestClassifier

from pyspark.ml import Pipeline



import pandas as pd

import pandas_profiling

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# For EDA

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
spark = SparkSession.builder.appName("Titanic-Dataset").config('spark.driver.memory','15g').getOrCreate()

spark

# spark.stop()
sdf_train = spark.read.csv('/kaggle/input/titanic/train.csv', inferSchema = True, header = True)

sdf_test = spark.read.csv('/kaggle/input/titanic/test.csv', inferSchema = True, header = True)
def _clean_dataset(sdf: DataFrame, col_to_convert: list, col_to_impute: list) -> DataFrame:

    for col in col_to_convert:

        sdf = sdf.withColumn(col,sdf[col].cast('double'))

    col_to_impute += col_to_convert



    imputer = Imputer(inputCols = col_to_impute, outputCols = col_to_impute)

    sdf = imputer.fit(sdf).transform(sdf)

    

    return sdf
def _handle_missing_age(sdf: DataFrame) -> DataFrame:

    _sdf = sdf

    _sdf = _sdf.withColumn('Age', 

           F.when((F.isnull(_sdf['Age'])) & (_sdf['Initial'] == 'Mr') , 33 )\

            .otherwise(F.when((F.isnull(_sdf['Age'])) 

                              & (_sdf['Initial'] == 'Mrs') , 36)\

            .otherwise(F.when((F.isnull(_sdf['Age'])) 

                              & (_sdf['Initial'] == 'Master') , 5)\

            .otherwise(F.when((F.isnull(_sdf['Age'])) 

                              & (_sdf['Initial'] == 'Miss') , 22)\

            .otherwise(F.when((F.isnull(_sdf['Age'])) 

                              & (_sdf['Initial'] == 'Other') , 46)\

            .otherwise(_sdf['Age']) )))))

    return _sdf
def _evaluate_initials(sdf: DataFrame) -> DataFrame:

    dizip_initials = {k:v for k,v in (zip(['Mlle','Mme','Ms','Dr',

                                           'Major','Lady','Countess',

                                           'Jonkheer','Col','Rev',

                                           'Capt','Sir','Don'],

                                         ['Miss','Miss','Miss',

                                          'Mr','Mr','Mrs','Mrs',

                                          'Other','Other','Other',

                                          'Mr','Mr','Mr']))}

    _sdf = sdf.withColumn('Initial',  F.regexp_extract( sdf['Name'], ('([A-Za-z]+)\.'),1 ) )

    _sdf = _sdf.replace(dizip_initials,1,'Initial')

    return _sdf

def _create_family_size(sdf: DataFrame) -> DataFrame :

    _sdf = sdf.withColumn('FamilySize', sdf['Parch'] + sdf['SibSp'] + 1 )

    

    return _sdf

sdf_train_cleaned = _clean_dataset ( 

    _handle_missing_age(

    _evaluate_initials(

    _create_family_size(sdf_train)

    )) 

    ,['Ticket','SibSp','Parch'],['Fare'] 

)



sdf_test_cleaned = _clean_dataset ( 

    _handle_missing_age(

    _evaluate_initials(

    _create_family_size(sdf_test)

    )) 

    ,['Ticket','SibSp','Parch'],['Fare'] 

)



sdf_train_cleaned.limit(5).toPandas().T
pdf_sdf_train = sdf_train_cleaned.toPandas()

pdf_sdf_train.T
fig, ax = plt.subplots(1,2, figsize=(16,8))

ax[0].set_title('barplot: Age ratio with Initial')

sns.barplot(x= pdf_sdf_train['Initial'], y=pdf_sdf_train['Age'],ax=ax[0])



ax[1].set_title('violinplot: Age ratio with Initial')

sns.violinplot(pdf_sdf_train['Initial'],pdf_sdf_train['Age'], ax=ax[1])
# derive AgeGroup from age and sex

pdf_sdf_train['AgeGroup'] = None

pdf_sdf_train.loc[((pdf_sdf_train['Sex'] == 'male') & (pdf_sdf_train['Age'] <= 15)), 'AgeGroup'] = 'boy'

pdf_sdf_train.loc[((pdf_sdf_train['Sex'] == 'female') & (pdf_sdf_train['Age'] <= 15)), 'AgeGroup'] = 'girl'

pdf_sdf_train.loc[((pdf_sdf_train['Sex'] == 'male') & (pdf_sdf_train['Age'] > 15)), 'AgeGroup'] = 'adult male'

pdf_sdf_train.loc[((pdf_sdf_train['Sex'] == 'female') & (pdf_sdf_train['Age'] > 15)), 'AgeGroup'] = 'adult female'

pdf_sdf_train['AgeGroup'].value_counts()
fig, ax = plt.subplots(1,2,figsize=(16,7))

ax[0].set_title('pointplot: Survived ratio for Age group')

sns.pointplot(pdf_sdf_train['AgeGroup'],pdf_sdf_train['Survived'],ax=ax[0])

ax[1].set_title('countplot: Survived ratio for Age group')

sns.countplot(pdf_sdf_train['AgeGroup'],hue= pdf_sdf_train['Survived'],ax=ax[1])
fig, ax = plt.subplots(1,2,figsize=(16,8))

ax[0].set_title('pointplot: Survived ratio with Pclass')

sns.pointplot(pdf_sdf_train['Pclass'],pdf_sdf_train['Survived'],ax=ax[0])

ax[1].set_title('countplot: Survived ratio with Pclass')

sns.countplot(pdf_sdf_train['Pclass'],hue= pdf_sdf_train['Survived'],ax=ax[1])
fig, ax = plt.subplots(1,2, figsize=(16,8))

ax[0].set_title('barplot: Survived ratio with Pclass')

sns.barplot( pdf_sdf_train['Pclass'], pdf_sdf_train['Survived'], ax=ax[0])



ax[1].set_title('countplot: Survived ratio with Pclass')

sns.countplot(x=pdf_sdf_train['Pclass'], hue=pdf_sdf_train['Survived'], ax=ax[1])
fig, ax = plt.subplots(1,2, figsize=(16,8))

ax[0].set_title('countplot: Survived ratio with Sex')

sns.countplot(x=pdf_sdf_train['Sex'],hue= pdf_sdf_train['Survived'],ax=ax[0])



ax[1].set_title('barplot: Survived ratio with Sex')

sns.barplot(pdf_sdf_train['Sex'], pdf_sdf_train['Survived'],ax=ax[1])

plt.figure(figsize=(12,24))

# plt.xticks(rotation=90)

sns.countplot(y=pdf_sdf_train['Age'],hue= pdf_sdf_train['Survived'],orient='h')
pd.Categorical(pdf_sdf_train['Sex'])
import numpy as np

plt.figure(figsize=(25,10))

plt.xticks(rotation=90)

plt.axvspan(np.size(pdf_sdf_train[pdf_sdf_train['Age'] < 12]['Age'].unique())

            ,np.size(pdf_sdf_train[pdf_sdf_train['Age'] < 50]['Age'].unique())

            ,alpha = 0.25

            , color = 'green') # without alpha = 0.25, it will be dark green!

print(np.size(pdf_sdf_train[pdf_sdf_train['Age'] < 12]['Age'].unique()))

print(np.size(pdf_sdf_train[pdf_sdf_train['Age'] < 50]['Age'].unique()))



sns.barplot(pdf_sdf_train['Age'],pdf_sdf_train['Survived'], ci=None) # ci box plot details
plt.figure(figsize=(25,10))

plt.title('Age distribution among all Pasengers')

sns.distplot(pdf_sdf_train['Age'])
plt.subplot(1,2,1)

# plt.figure(figsize=(25,10))

plt.title('Age distribution for Survived')

plt.axis([0,100,0,100])

sns.distplot(pdf_sdf_train[pdf_sdf_train.Survived == 1]['Age'],kde=False)



plt.subplot(1,2,2)

plt.title('Age distribution for Non Survived')

sns.distplot(pdf_sdf_train[pdf_sdf_train.Survived == 0]['Age'],kde=False)



plt.subplots_adjust(right=1.7)

# plt.show()
g = sns.FacetGrid(pdf_sdf_train,col='Survived')

g = g.map(sns.distplot,'Age')
g = sns.kdeplot(pdf_sdf_train['Age']

                [(pdf_sdf_train['Survived']==0) 

                                     & (pdf_sdf_train['Age'].notnull())],

                color='Red',shade=True)

g = sns.kdeplot(pdf_sdf_train['Age']

                [(pdf_sdf_train['Survived']==1)  

                                     & (pdf_sdf_train['Age'].notnull())],

                color='Green',shade=True)

g.set_xlabel('Age')

g.set_ylabel('Frequency')

g = g.legend(['Not Survived','Survived'])
# https://homepage.divms.uiowa.edu/~luke/classes/STAT4580/stripplot.html

sns.stripplot(x="Survived", y="Age",data=pdf_sdf_train,jitter=True)
sns.pairplot(pdf_sdf_train)
numeric_cols = ['PassengerId','Survived', 'Pclass',

                'Age', 'SibSp','Parch','Ticket','Fare'] 

numeric_features = ['PassengerId','Pclass','Age', 'SibSp','Parch','Fare'] 

string_features = [ 'Embarked', 'Sex'] 
_stages = []

string_indexer =  [StringIndexer(inputCol = column , \

                                 outputCol = column + '_StringIndexer', 

                                 handleInvalid = "skip") for column in string_features]



one_hot_encoder = [OneHotEncoderEstimator(

    inputCols = [column + '_StringIndexer' for column in string_features ], \

    outputCols =  [column + '_OneHotEncoderEstimator' for column in string_features ])]



vect_indexer = [VectorIndexer(

    inputCol = column + '_OneHotEncoderEstimator',

    outputCol = column + '_VectorIndexer', 

    maxCategories=10) for column in string_features]



familt_size_splits = [1, 2, 5, 7, 100] 

bucketizer = Bucketizer(splits = familt_size_splits, 

                        inputCol = 'FamilySize',

                        outputCol = 'bucketized_FamilySize')



numeric_features += ['bucketized_FamilySize']



assemblerInput =  [f  for f in numeric_features]  

assemblerInput += [f + "_VectorIndexer" for f in string_features]

vector_assembler = VectorAssembler(inputCols = assemblerInput, \

                                   outputCol = 'VectorAssembler_features')



_stages += string_indexer

_stages += one_hot_encoder

_stages += vect_indexer

_stages += [bucketizer]

_stages += [vector_assembler]

_stages
pipeline = Pipeline(stages = _stages)
model = pipeline.fit(sdf_train_cleaned)
sdf_transformed_train = model.transform(sdf_train_cleaned)

sdf_transformed_train.limit(5).toPandas().T


rf = RandomForestClassifier(labelCol = 'Survived', 

                            featuresCol = 'VectorAssembler_features', 

                            numTrees = 100, 

                            maxDepth = 4, 

                            maxBins = 1000)

_stages += [rf]
_stages
pipeline = Pipeline(stages = _stages)

model = pipeline.fit(sdf_train_cleaned)



sdf_predict = model.transform(sdf_test_cleaned)
sdf_predict.toPandas().profile_report()