!pip install pyspark
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import pyspark
from pyspark.sql import SparkSession
import numpy as np

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext


from pyspark.ml import feature, regression,classification, Pipeline, evaluation 
from pyspark.sql import functions as fn, Row
from pyspark import sql

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
df = pd.read_csv('../input/diabetes/diabetic_data.csv')

df
import pandas_profiling
df.profile_report()
df.columns
df["readmitted"].value_counts()
df = pd.get_dummies(df,columns = [df.columns.values[i] for i in range(24,47) ], prefix=[df.columns.values[i] for i in range(24,47)], prefix_sep='_',drop_first=True) 
##Dummy reference Medication Down
df.shape
df.columns
df['readmitted'] = df['readmitted'].map({'NO': 0, '<30': 1, ">30":2})
df['readmittedbinary'] = df['readmitted'].map({0: 0, 1: 1, 2:1})
df = pd.get_dummies(df, columns=["change",'max_glu_serum','A1Cresult','diabetesMed'], prefix = ["change",'max_glu_serum','A1Cresult','diabetesMed'],prefix_sep='_',drop_first=True)
## Dummy Reference A1Cresult_>7,max_glu_serum>200,diabetesMed=No, change=ch
df['age'] = df['age'].map({'[0-10)':5,'[10-20)':15, '[20-30)':25,'[30-40)':35,'[40-50)':45,'[50-60)':55,'[60-70)':65,'[70-80)':75,'[80-90)':85,'[90-100)':95})
df.drop(['encounter_id','patient_nbr','weight','admission_type_id','discharge_disposition_id','admission_source_id','medical_specialty','payer_code'],axis=1,inplace=True)


df=df.loc[df['gender'].isin(['Male','Female'])]#df.loc[df['B'].isin(['one','three'])]
df.replace('?', np.nan, inplace = True)
df= df.dropna()##Clean pandas df without dummy variables 
df.columns
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
a = sns.countplot(x = df['age'], hue= df['readmitted'])
d = sns.countplot(x = df['readmitted'])
b = sns.countplot(x = df['gender'], hue= df['readmitted'])
c = sns.countplot(x = df['race'], hue= df['readmitted'])
count_of_y = df["age"].groupby(df["readmitted"]).value_counts().rename("counts").reset_index()
count_of_y
fig = sns.lineplot(x="age", y="counts", hue="readmitted", data=count_of_y)
sns.heatmap(df.corr())
plt.figure(figsize=(25, 8))
a = df.corr()
b = a['readmitted']
c= b.to_frame()
type(c)
c.sort_values(by = ['readmitted'], ascending = False , inplace = True)
pos = c.head(8)
c.sort_values(by = ['readmitted'], ascending = True , inplace = True)
neg = c.head(8)
neg

pos.index.name = 'feature'
pos.reset_index(inplace=True)
pos

neg.index.name = 'feature'
neg.reset_index(inplace=True)
neg
pos=pos.drop(pos.index[0:2])
pos
posplot = sns.barplot(x='feature', y="readmitted", data=pos)
posplot.set_xticklabels(posplot.get_xticklabels(),rotation=30)
negplot = sns.barplot(x='feature', y="readmitted", data=neg)
negplot
negplot.set_xticklabels(negplot.get_xticklabels(),rotation=40)
df.to_csv("clean.csv")
sorted(df.columns)
df
spark_df = spark.read.csv('clean.csv', header=True, inferSchema=True)
spark_df.show()
#spark_df=spark_df.withColumnRenamed("glimepiride-poglitazone","glimepiridepoglitazone").withColumnRenamed("glyburide-metformin","glyburidemetformin").withColumnRenamed("glipizide-metformin","glipizidemetformin").withColumnRenamed("glimepiride-pioglitazone","glimepiridepioglitazone").withColumnRenamed("metformin-rosiglitazone","metforminrosiglitazone").withColumnRenamed("metformin-pioglitazone","metforminpioglitazone")
def amit(row):    
    ma=0
    mb=0
    md=0
    me=0
    sa=0
    dr=dd=ddt=di=dm=dg=dn=dr2=dd2=ddt2=di2=dm2=dg2=dn2=dr3=dd3=ddt3=di3=dm3=dg3=dn3=0
    
    if "V" in row.diag_1 or "E" in row.diag_1:
        dr=dd=ddt=di=dm=dg=dn=0
    #elif 390 <= float(row.diag_1) <= 459 or float(row.diag_1) == 785: #DUMMY DROPPED REFERENCE
    #    dc =1 ##Circulatory
    elif 460 <= float(row.diag_1) <= 519 or float(row.diag_1) == 786:
        dr =1 #Respiratory
    elif 520 <= float(row.diag_1) <= 579 or float(row.diag_1) == 787:
        dd =1 #Digestive
    elif 250 <= float(row.diag_1) <= 250.999:
        ddt =1 #Diabetes
    elif 800 <= float(row.diag_1) <= 999:
        di =1 #Injury
    elif 710 <= float(row.diag_1) <= 739:
        dm =1 #musculoskeletal
    elif 580 <= float(row.diag_1) <= 629 or float(row.diag_1) == 788:
        dg =1 #Genitourinary
    elif 140 <= float(row.diag_1) <= 239:
        dn =1 #Neoplasms
    else:
        dr=dd=ddt=di=dm=dg=dn=0
        #do=1#others
        
    if "V" in row.diag_2 or "E" in row.diag_2:
        #do2=1
        dr2=dd2=ddt2=di2=dm2=dg2=dn2=0
    #elif 390 <= float(row.diag_2) <= 459 or float(row.diag_2) == 785: #DUMMY DROPPED REFERENCE
    #    dc2 =1 ##Circulatory
    elif 460 <= float(row.diag_2) <= 519 or float(row.diag_2) == 786:
        dr2 =1 #Respiratory
    elif 520 <= float(row.diag_2) <= 579 or float(row.diag_2) == 787:
        dd2 =1 #Digestive
    elif 250 <= float(row.diag_2) <= 250.999:
        ddt2 =1 #Diabetes
    elif 800 <= float(row.diag_2) <= 999:
        di2 =1 #Injury
    elif 710 <= float(row.diag_2) <= 739:
        dm2 =2 #musculoskeletal
    elif 580 <= float(row.diag_2) <= 629 or float(row.diag_2) == 788:
        dg2 =1 #Genitourinary
    #elif 140 <= float(row.diag_2) <= 239:
    #   dn2 =1 #Neoplasms
    else:
        #do2=1#others
        dr2=dd2=ddt2=di2=dm2=dg2=dn2=0
        
    if "V" in row.diag_3 or "E" in row.diag_3:
        #do3=1
        dr3=dd3=ddt3=di3=dm3=dg3=dn3=0
    #elif 390 <= float(row.diag_3) <= 459 or float(row.diag_3) == 785:#DUMMY DROPPED REFERENCE
    #    dc3 =1 ##Circulatory
    elif 460 <= float(row.diag_3) <= 519 or float(row.diag_3) == 786:
        dr3 =1 #Respiratory
    elif 520 <= float(row.diag_3) <= 579 or float(row.diag_3) == 787:
        dd3 =1 #Digestive
    elif 250 <= float(row.diag_3) <= 250.999:
        ddt3 =1 #Diabetes
    elif 800 <= float(row.diag_3) <= 999:
        di3 =1 #Injury
    elif 710 <= float(row.diag_3) <= 739:
        dm3 =1 #musculoskeletal
    elif 580 <= float(row.diag_3) <= 629 or float(row.diag_3) == 788:
        dg3 =1 #Genitourinary
    elif 140 <= float(row.diag_3) <= 239:
        dn3 =1 #Neoplasms
    else:
        dr3=dd3=ddt3=di3=dm3=dg3=dn3=0
        #do3=1#others  

    
    if row.race == "Caucasian":
        ma = 1
        #prfloat("A")
    elif row.race == "Asian":
        mb = 1
    #elif row.race == "AfricanAmerican": #DUMMY DROPPED REFERENCE
    #    mc = 1
    elif row.race =="Hispanic":
        me = 1
    else:# :
        ma=0
        mb=0
        me = 0


    if row.gender == "Male":
        sa = 1
    #elif row.gender == "Female": #DROPPED DUMMY REFERENCE
    #    sb = 1
    
    
    r = Row(Caucasian=int(ma) ,Asian=int(mb) ,Hispanic=int(me),male=float(sa),
            Respiratory=dr,
            Digestive= dd,
            Diabetes = ddt,
            Injury= di,
            Muscuskeletal= dm,
            Neoplasms=dn,
            Genitourinary = dg,
            
            Respiratory2=dr2,
            Digestive2= dd2,
            Diabetes2 = ddt2,
            Injury2= di2,
            Muscuskeletal2= dm2,
            Neoplasms2=dn2,
            Genitourinary2 = dg2,
            
            Respiratory3=dr3,
            Digestive3= dd3,
            Diabetes3 = ddt3,
            Injury3= di3,
            Muscuskeletal3= dm3,
            Neoplasms3=dn3,
            Genitourinary3 = dg3,
            
      )
    return(r)
dummy_df = spark.createDataFrame(spark_df.rdd.map(amit))
spark_df.show()
dummy_df.show()
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql.window import Window
# since there is no common column between these two dataframes add row_index so that it can be joined
spark_df=spark_df.withColumn('row_index', row_number().over(Window.orderBy(monotonically_increasing_id())))
dummy_df=dummy_df.withColumn('row_index', row_number().over(Window.orderBy(monotonically_increasing_id())))

dummy_df = dummy_df.join(spark_df, on=["row_index"]).drop("row_index")
dummy_df.show()

dummy_df = dummy_df.drop("diag_1","diag_2","diag_3","gender","race")
dummy_df = dummy_df.drop("_c0")
#Checking dummies
dummy_df.select('male',"Caucasian","Hispanic","Asian").show()#,'Male','Female', 'Circulatory','Circulatory2','Circulatory3').show()
dummy_df.printSchema()
training_df, validation_df, testing_df = dummy_df.randomSplit([0.6, 0.3, 0.1], seed=0)
dummy_df.columns
featlist = ['Asian',
 'Caucasian',
 'Diabetes',
 'Diabetes2',
 'Diabetes3',
 'Digestive',
 'Digestive2',
 'Digestive3',
 'Genitourinary',
 'Genitourinary2',
 'Genitourinary3',
 'Hispanic',
 'Injury',
 'Injury2',
 'Injury3',
 'Muscuskeletal',
 'Muscuskeletal2',
 'Muscuskeletal3',
 'Neoplasms',
 'Neoplasms2',
 'Neoplasms3',
 'Respiratory',
 'Respiratory2',
 'Respiratory3',
 'male',
 'age',
 'time_in_hospital',
 'num_lab_procedures',
 'num_procedures',
 'num_medications',
 'number_outpatient',
 'number_emergency',
 'number_inpatient',
 'number_diagnoses',
 'metformin_No',
 'metformin_Steady',
 'metformin_Up',
 'repaglinide_No',
 'repaglinide_Steady',
 'repaglinide_Up',
 'nateglinide_No',
 'nateglinide_Steady',
 'nateglinide_Up',
 'chlorpropamide_No',
 'chlorpropamide_Steady',
 'chlorpropamide_Up',
 'glimepiride_No',
 'glimepiride_Steady',
 'glimepiride_Up',
 'acetohexamide_Steady',
 'glipizide_No',
 'glipizide_Steady',
 'glipizide_Up',
 'glyburide_No',
 'glyburide_Steady',
 'glyburide_Up',
 'tolbutamide_Steady',
 'pioglitazone_No',
 'pioglitazone_Steady',
 'pioglitazone_Up',
 'rosiglitazone_No',
 'rosiglitazone_Steady',
 'rosiglitazone_Up',
 'acarbose_No',
 'acarbose_Steady',
 'acarbose_Up',
 'miglitol_No',
 'miglitol_Steady',
 'miglitol_Up',
 'troglitazone_Steady',
 'tolazamide_Steady',
 'tolazamide_Up',
 'insulin_No',
 'insulin_Steady',
 'insulin_Up',
 'glyburide-metformin_No',
 'glyburide-metformin_Steady',
 'glyburide-metformin_Up',
 'glipizide-metformin_Steady',
 'glimepiride-pioglitazone_Steady',
 'metformin-rosiglitazone_Steady',
 'metformin-pioglitazone_Steady',
 'change_No',
 'max_glu_serum_>300',
 'max_glu_serum_None',
 'max_glu_serum_Norm',
 'A1Cresult_>8',
 'A1Cresult_None',
 'A1Cresult_Norm',
 'diabetesMed_Yes']
model1 = Pipeline(stages=[feature.VectorAssembler(inputCols=featlist,
                                        outputCol='features'),feature.StandardScaler(inputCol='features',outputCol = 'sdfeatures'),
                 classification.LogisticRegression(labelCol='readmittedbinary', featuresCol='sdfeatures')])

pipe_model = model1.fit(training_df)
pipe_modeldf = pipe_model.transform(validation_df).select("readmittedbinary","prediction")
pipe_modeldf.show()
tp = pipe_modeldf[(pipe_modeldf.readmittedbinary == 1) & (pipe_modeldf.prediction == 1)].count()
tn = pipe_modeldf[(pipe_modeldf.readmittedbinary == 0) & (pipe_modeldf.prediction == 0)].count()
fp = pipe_modeldf[(pipe_modeldf.readmittedbinary == 0) & (pipe_modeldf.prediction == 1)].count()
fn = pipe_modeldf[(pipe_modeldf.readmittedbinary == 1) & (pipe_modeldf.prediction == 0)].count()
print ("True Positives:", tp)
print ("True Negatives:", tn)
print ("False Positives:", fp)
print ("False Negatives:", fn)
print ("Total", dummy_df.count())

r = (tp)/(tp + fn)
print ("recall", r)

p = float(tp) / (tp + fp)
print ("precision", p)
specificity = tn/(tn+fp)
print("specificity",specificity)
evaluator = evaluation.BinaryClassificationEvaluator(labelCol='readmittedbinary')
AUC1 = evaluator.evaluate(pipe_model.transform(validation_df))
AUC1
pd.DataFrame(list(zip(featlist, pipe_model.stages[-1].coefficients.toArray())),
            columns = ['column', 'Coefficients']).sort_values('Coefficients',ascending = False).head(10)
print("Intercept: " + str(pipe_model.stages[-1].intercept))
prob = 1/(1+np.exp(-pipe_model.stages[-1].intercept))
prob
np.exp(0.4582)
pd.DataFrame(list(zip(featlist, pipe_model.stages[-1].coefficients.toArray())),
            columns = ['column', 'Coefficients']).sort_values('Coefficients').head(10)
np.exp(-0.068755)
beta = pipe_model.stages[-1].coefficients
plt.plot(beta)
plt.ylabel('Coefficients')
plt.show()
trainingSummary = pipe_model.stages[-1].summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
roc
fMeasure = trainingSummary.fMeasureByThreshold
maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
    .select('threshold').head()['threshold']
maxFMeasure
pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()
#pr['recall']
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
va =feature.VectorAssembler(inputCols=featlist ,  outputCol='features')
sd = feature.StandardScaler(inputCol='features',outputCol = 'sdfeatures')
lr = classification.LogisticRegression(labelCol='readmittedbinary', featuresCol='sdfeatures')
pipe_model2 = Pipeline(stages=[va,sd, lr])
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.3, 0.5]).addGrid(lr.elasticNetParam, [0.2, 0.8, 0.5]).addGrid(lr.maxIter, [15, 30, 50]).build())
cv = CrossValidator(estimator=pipe_model2, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)
cvModel = cv.fit(training_df)
AUC2 = evaluator.evaluate(cvModel.transform(validation_df))
AUC2
param_dict = cvModel.bestModel.stages[-1].extractParamMap()

sane_dict = {}
for k, v in param_dict.items():
    #print(k)
    sane_dict[k.name] = v

best_reg = sane_dict["regParam"]
best_elastic_net = sane_dict["elasticNetParam"]
best_max_iter = sane_dict["maxIter"]
print(best_reg)
print(best_elastic_net)
print(best_max_iter)
modelrf = Pipeline(stages= [va, classification.RandomForestClassifier(labelCol='readmittedbinary', featuresCol="features")])
modelrffit= modelrf.fit(training_df)
modelrfdf = modelrffit.transform(validation_df)
modelrfdf
AUCrf = evaluator.evaluate(modelrffit.transform(validation_df))
AUCrf
pd.DataFrame(list(zip(featlist, modelrffit.stages[1].featureImportances.toArray())),
            columns = ['column', 'weight']).sort_values('weight', ascending = False).head(10)
tp = modelrfdf[(modelrfdf.readmittedbinary == 1) & (modelrfdf.prediction == 1)].count()
tn = modelrfdf[(modelrfdf.readmittedbinary == 1) & (modelrfdf.prediction == 0)].count()
fp = modelrfdf[(modelrfdf.readmittedbinary == 0) & (modelrfdf.prediction == 1)].count()
fn = modelrfdf[(modelrfdf.readmittedbinary == 0) & (modelrfdf.prediction == 0)].count()
print ("True Positives:", tp)
print ("True Negatives:", tn)
print ("False Positives:", fp)
print ("False Negatives:", fn)
#print ("Total", df.count())

r = (tp)/(tp + fn)
print ("recall", r)

p = float(tp) / (tp + fp)
print ("precision", p)
sensitivity = tn/(tn+fp)
print("Sensitivity",sensitivity)
beta = modelrffit.stages[-1].featureImportances
plt.plot(beta)
plt.ylabel('Importance')
plt.show()
rf=classification.RandomForestClassifier(labelCol='readmittedbinary', featuresCol="features")
mrf = Pipeline(stages=[va,rf])
paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [40]).addGrid(rf.maxDepth,[5,10,15,30]).build())


cvrf = CrossValidator(estimator=mrf, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=3)
cvrf1 = cvrf.fit(training_df)
AUCn = evaluator.evaluate(cvrf1.transform(validation_df))
AUCn
param_dict1 = cvrf1.bestModel.stages[-1].extractParamMap()

sane_dict1 = {}
for k, v in param_dict1.items():
    #print(k)
    sane_dict1[k.name] = v


best_max_depth = sane_dict1["maxDepth"]
print(best_max_depth)
modelrfmc = Pipeline(stages= [va, classification.RandomForestClassifier(labelCol='readmitted', featuresCol="features",maxDepth=15, numTrees=30)])
modelrffitmc= modelrfmc.fit(training_df)
evaluatormc = evaluation.MulticlassClassificationEvaluator(labelCol='readmitted',predictionCol="prediction",metricName="accuracy")
AUCrfmc = evaluatormc.evaluate(modelrffitmc.transform(validation_df))
AUCrfmc
pd.DataFrame(list(zip(featlist, modelrffitmc.stages[1].featureImportances.toArray())),
            columns = ['column', 'weight']).sort_values('weight', ascending = False).head(10)
model_new_lr2 = Pipeline(stages=[feature.VectorAssembler(inputCols=['number_inpatient', 'number_emergency', 'number_diagnoses', 'Diabetes', 'number_outpatient','time_in_hospital','diabetesMed_Yes','age','rosiglitazone_Steady','Caucasian'],
                                        outputCol='features'),sd,
                 classification.LogisticRegression(labelCol='readmittedbinary', featuresCol='sdfeatures')])

pipe_model3 = model_new_lr2.fit(training_df)
AUC5 = evaluator.evaluate(pipe_model3.transform(validation_df))
AUC5
modelrfselected = Pipeline(stages= [feature.VectorAssembler(inputCols=['number_inpatient', 'number_emergency', 'number_diagnoses', 'Diabetes', 'number_outpatient','time_in_hospital','diabetesMed_Yes','age','rosiglitazone_Steady','Caucasian'],
                                        outputCol='features'), classification.RandomForestClassifier(labelCol='readmittedbinary', featuresCol="features",maxDepth=15, numTrees=30)])
modelrffitselected= modelrfselected.fit(training_df)
AUCrfselected = evaluator.evaluate(modelrffitselected.transform(validation_df))
AUCrfselected
model_new_lr3 = Pipeline(stages=[feature.VectorAssembler(inputCols=['number_inpatient','num_medications','num_lab_procedures','number_diagnoses','time_in_hospital','age','number_emergency', 'number_outpatient','num_procedures','male'],
                                        outputCol='features'),sd,
                 classification.LogisticRegression(labelCol='readmittedbinary', featuresCol='sdfeatures')])

pipe_model4 = model_new_lr3.fit(training_df)
AUC6 = evaluator.evaluate(pipe_model4.transform(validation_df))
AUC6
best_model = Pipeline(stages= [va, classification.RandomForestClassifier(labelCol='readmittedbinary', featuresCol="features",maxDepth=10, numTrees=40)])
bestmodel_fit= best_model.fit(training_df)
AUCfinal = evaluator.evaluate(bestmodel_fit.transform(testing_df))
AUCfinal
pd.DataFrame(list(zip(featlist, bestmodel_fit.stages[1].featureImportances.toArray())),
            columns = ['column', 'weight']).sort_values('weight', ascending = False).head(10)
bestmodel_final_fit= best_model.fit(dummy_df)

