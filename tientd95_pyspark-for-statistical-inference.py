!pip install pyspark
import os
import sys
import pyspark
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions
from pyspark.sql.functions import lit, desc, col, size
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from IPython.core.interactiveshell import InteractiveShell
import matplotlib
from pylab import *
import scipy.stats as stats

# This helps auto print out the items without explixitly using 'print'
InteractiveShell.ast_node_interactivity = "all" 

# Initialize a spark session.

conf = pyspark.SparkConf().setMaster("local[*]")
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Statistical Inferences with Pyspark") \
        .config(conf=conf) \
        .getOrCreate()
    return spark

spark = init_spark()
filename_data = '../input/fitrec-data-pyspark/endomondoHR_proper.json'
df = spark.read.json(filename_data, mode="DROPMALFORMED")

# Load meta data file into pyspark data frame as well
print('Data frame type: {}'.format(type(df)))
print('Columns & datatypes:')
DataFrame(df.dtypes, columns =['Column Name','Data type'])
print('Data frame describe (string and numeric columns only):')
df.describe().toPandas()

print('\nFisrt 2 data rows:')
df.limit(2).toPandas()
# Summary function
def user_activity_workout_summarize(df):
    user_count = format(df.select('userId').distinct().count(), ',d')
    workout_count = format(df.select('id').distinct().count(), ',d')
    activity_count = str(df.select('sport').distinct().count())
    seqOp = (lambda x,y: x+y)
    sum_temp = df.rdd.map(lambda x: len(x.timestamp)).aggregate(0, seqOp, seqOp)
    total_records_count = format(sum_temp, ',d')
    columns=['Users count', 'Activity types count','Workouts count', 'Total records count']
    data = [[user_count], [activity_count], [workout_count], [total_records_count]]
    sum_dict = {column: data[i] for i, column in enumerate(columns)}
    sum_df = pd.DataFrame.from_dict(sum_dict)[columns]
    gender_user_count = df.select('gender','userId').distinct().groupBy('gender').count().toPandas()
    gender_activities_count = df.groupBy('gender').count().toPandas()
    gender_user_activity_count = gender_user_count.join(
        gender_activities_count.set_index('gender'), on='gender'
        , how='inner', lsuffix='_gu'
    )
    gender_user_activity_count.columns = ['Gender', '# of users', 'Activities (workouts) count']
    
    return sum_df, gender_user_activity_count

sum_dfs = user_activity_workout_summarize(df)
print('\nOverall data set summary on users, activities(workouts) and number of fitness records:')
sum_dfs[0]
print('\nSummarize on genders:')
sum_dfs[1]
rdd = df.rdd
def avgHeartRate(row):
    if row['heart_rate']!='':
        ht = np.mean(row['heart_rate'])
        return Row(id=int(row['id']), gender=str(row['gender'])
                   , sport=str(row['sport']), userId=row['userId'], avg_heart_rate=float(ht))
print('Top 3 rows of average heart rate per workout:')
rdd_avgHR = rdd.map(avgHeartRate)
rdd_avgHR.toDF().limit(3).toPandas()
print('\nDescribe:')
df1 = spark.createDataFrame(rdd_avgHR)
DataFrame(df1.dtypes, columns =['Column Name','Data type'])
df2 = df1.groupBy(['gender','sport']).avg('avg_heart_rate')
df2.createOrReplaceTempView("table1")
df2_Male = spark.sql("SELECT * FROM table1 WHERE gender='male'")
df2_Female = spark.sql("SELECT * FROM table1 WHERE gender='female'")
df2A=df2_Male.join(df2_Female, 'sport','outer')
df2_Male = df2_Male.withColumnRenamed('avg(avg_heart_rate)','maleAvgHR')
df2_Female = df2_Female.withColumnRenamed('avg(avg_heart_rate)','femaleAvgHR')
df2AInner = df2_Male.join(df2_Female, 'sport','inner')
df2AInner = df2AInner.withColumn('diffAvg',df2AInner.maleAvgHR-df2AInner.femaleAvgHR)
df2AInner.limit(5).toPandas()
rddAInner = df2AInner.rdd
sportType = rddAInner.map(lambda row: row['sport']).collect()
diffAvg = rddAInner.map(lambda row: row['diffAvg']).collect()
xticks = plt.xticks(rotation=90)
xlabel = plt.xlabel('Sport')
ylabel = plt.ylabel('Average heart rate(male) - Average heart rate(female) (bpm)')
plot = plt.bar(sportType, diffAvg, facecolor='#558866', edgecolor='white')
title = plt.title('The difference in average heart rate between male and female')
InteractiveShell.ast_node_interactivity = "all"
rdd2 = df.rdd
def covHeartRate(row):
    if row['heart_rate'] != '' and row['speed'] != '' and row['altitude'] != '':   
        if size(row['heart_rate']) == size(row['speed']) and size(row['heart_rate']) == size(row['altitude']):
            pearson_hr_al = stats.pearsonr(row['heart_rate'],row['altitude'])[0]
            pearson_hr_speed = stats.pearsonr(row['heart_rate'],row['speed'])[0] 
            abs_pearson_hr_al = abs(stats.pearsonr(row['heart_rate'],row['altitude'])[0])
            abs_pearson_hr_speed = abs(stats.pearsonr(row['heart_rate'],row['speed'])[0]) 
            return Row(id=int(row['id']), pearson_hr_al=float(pearson_hr_al)
                       , pearson_hr_speed=float(pearson_hr_speed), abs_pearson_hr_al=float(abs_pearson_hr_al)
                       , abs_pearson_hr_speed=float(abs_pearson_hr_speed)
                       , gender=str(row['gender']), sport=str(row['sport']),userId=row['userId'])
rddaHR2 = rdd2.filter(lambda row: row['speed'] is not None).map(covHeartRate)
df5 = spark.createDataFrame(rddaHR2).dropna()
print('\nSummary of coefficients table:')
df5.describe().toPandas()
# Aggregate by gender & sport
df6 = df5.groupBy(['gender', 'sport']).agg({'abs_pearson_hr_al':'mean', 'abs_pearson_hr_speed':'mean'})
df6.createOrReplaceTempView("table3")
df6M = spark.sql("SELECT * FROM table3 WHERE gender = 'male'")
df6M = df6M.withColumnRenamed('avg(abs_pearson_hr_al)','mAvgPerHtAt') \
    .withColumnRenamed('avg(abs_pearson_hr_speed)','mAvgPerHtSp')

df6FM = spark.sql("SELECT * FROM table3 WHERE gender = 'female'")
df6FM = df6FM.withColumnRenamed('avg(abs_pearson_hr_al)','fmAvgPerHtAt') \
    .withColumnRenamed('avg(abs_pearson_hr_speed)','fmAvgPerHtSp')
rdd6MInner = df6M.rdd
rdd6FMInner = df6FM.rdd
sportTypeM = rdd6MInner.map(lambda row: row['sport']).collect()
sportTypeFM = rdd6FMInner.map(lambda row: row['sport']).collect()
perMhtat = rdd6MInner.map(lambda row: row['mAvgPerHtAt']).collect()
perMhtsp = rdd6MInner.map(lambda row: row['mAvgPerHtSp']).collect()
perFMhtat = rdd6FMInner.map(lambda row: row['fmAvgPerHtAt']).collect()
perFMhtsp = rdd6FMInner.map(lambda row: row['fmAvgPerHtSp']).collect()

meanMhtat, varMhtat = mean(perMhtat), var(perMhtat)
meanMhtsp, varMhtsp = mean(perMhtsp), var(perMhtsp)
meanFMhtat, varFMhtat = mean(perFMhtat), var(perFMhtat)
meanFMhtsp, varFMhtsp = mean(perFMhtsp), var(perFMhtsp)
column_list = [{'gender': 'male', 'correlation': '(heart rate, altitude)'
      , 'mean': float(meanMhtat), 'variance': float(varMhtat)}
     , {'gender': 'male','correlation': '(heart rate, speed)'
        ,'mean': float(meanMhtsp), 'variance': float(varMhtsp) }
     , {'gender': 'female', 'correlation':'(heart rate, altitude)'
      , 'mean': float(meanFMhtat), 'variance': float(varFMhtat)}
        , {'gender':'female', 'correlation':'(heart rate, speed)'
           , 'mean': float(meanFMhtsp), 'variance': float(varFMhtsp)}]
showdf6 = pd.DataFrame(column_list)
print("Average correlation coefficient of different sports for male and female:")
showdf6

# Take the differences
diffMatsp = np.array(perMhtat) - np.array(perMhtsp)
diffFMatsp = np.array(perFMhtat) - np.array(perFMhtsp)

fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=plt.figaspect(0.35))
xstick_labels0 = axs[0].set_xticklabels(sportTypeM, rotation=90)
xlabel0 = axs[0].set_xlabel('Sport')
ylabel0 = axs[0].set_ylabel(
    'coe(heart rate, altitude) - coe(heart rate, altitude))'
)
title0 = axs[0].set_title('Male')
plot0 = axs[0].bar(sportTypeM, diffMatsp, facecolor='#9999ff', edgecolor='white')
xstick1 = plt.xticks(rotation=90)
xlabel1 = axs[1].set_xlabel('Sport')
ylabel1 = axs[1].set_ylabel('coe(heart rate, altitude) - coe(heart rate, altitude))')
title1 = axs[1].set_title('Female')
plot1 = axs[1].bar(sportTypeFM, diffFMatsp, facecolor='#ff9999', edgecolor='white')
title = fig.text(
    0.5, 1.02, 'Pearson coefficient difference between (heart rate, altitude) vs. (heart rate, speed)'
    , ha='center', va='top', transform=fig.transFigure
)
from datetime import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf

# Apply filter to select workouts that have all records occuring in the sameday (cross out the ones that lasted
# more than one day).
rdd3=df.rdd
def same_day(row):
    if row['heart_rate'] != '' and row['speed'] != '' and row['altitude'] != '':   
        if size(row['heart_rate']) == size(row['speed']) and size(row['heart_rate']) == size(row['altitude']):
            dateValue = [datetime.fromtimestamp(t) - timedelta(hours=7) for t in row['timestamp']]
            return Row(
                id=int(row['id']), gender=row['gender'], timestamp=row['timestamp']
                , heartrate=row['heart_rate'], speed=row['speed'], altitude=row['altitude']
                , year=dateValue[0].year, month=dateValue[0].month, day=dateValue[0].day
                , yearl=dateValue[-1].year, monthl=dateValue[-1].month, dayl=dateValue[-1].day
                , sport=str(row['sport']), userId=row['userId']
            )
#  return Row(ratiohtsp=float(vecsp))
rddaHR3 = rdd3.filter(lambda row: row['speed'] is not None) \
    .map(same_day) \
    .filter(lambda row: row['year'] == row['yearl'] and row['month'] == row['monthl'] and row['day'] == row['dayl'])

df7 = spark.createDataFrame(rddaHR3).drop('year').drop('yearl').drop('month') \
    .drop('day').drop('monthl').drop('dayl')

# Group and workouts by 4 time ranges of the day based on workout start hour 
# (0: start hour from 0 - 5)
# (1: start hour from 6 - 11)
# (2: start hour from 12 - 17)
# (3: start hour from 18 - 24)
def markWorkout(row):
    hours = [(datetime.fromtimestamp(t) - timedelta(hours=7)).hour for t in row['timestamp']]
    mark = -1
    upIndex = -1
    if hours[0] >= 6 and hours[0] < 12:
        upIndex = [i for i in range(len(hours)) if hours[i] < 12][-1]
        mark = 1
    elif hours[0] >= 12 and hours[0] < 18:
        upIndex=[i for i in range(len(hours)) if hours[i] < 19][-1]
        mark = 2
    elif hours[0] >= 18 and hours[0] < 24:
        upIndex = [i for i in range(len(hours)) if hours[i] < 24][-1]
        mark = 3
    elif hours[0] >= 0 and hours[0] < 6:
        upIndex = [i for i in range(len(hours)) if hours[i] < 8][-1]
        mark = 0
    if mark !=- 1 and upIndex !=- 1:
        reTimestamp = row['timestamp'][:upIndex]
        dateValue = [datetime.fromtimestamp(t) for t in reTimestamp]
        reHeartRate = row['heartrate'][:upIndex]
        reAltitude = row['altitude'][:upIndex]
        reSpeed = row['speed'][:upIndex]
        count = upIndex + 1
        return Row(
            id=row['id'], gender=row['gender'], mark=mark, countTimestamp=count
            , reTimestamp=reTimestamp, dateValue=dateValue, reHeartRate=reHeartRate
            , reAltitude=reAltitude, reSpeed=reSpeed, sport=str(row['sport']), userId=row['userId']
        )
    
rdd31 = df7.rdd.map(markWorkout).filter(lambda row: row['countTimestamp']>10)
df8 = spark.createDataFrame(rdd31)
DataFrame(df8.dtypes, columns=['Column Name','Data type'])
df8.describe().toPandas()
df8.limit(3).toPandas()
df8Count = df8.groupBy(['mark','sport','gender']).count().orderBy(['mark','sport','gender'])
rdd8Count = df8Count.rdd
df8Count.createOrReplaceTempView("table4")
df8CountM = spark.sql("SELECT * from table4 where gender='male'")
#df8CountM.toPandas()
df8CountFM = spark.sql("SELECT * from table4 where gender='female'")
#df8CountFM.toPandas()

sportTypeM0 = rdd8Count.filter(
    lambda row: row['mark'] == 0 and row['gender'] == 'male'
).map(lambda row: row['sport']).collect()
sportTypeFM0 = rdd8Count.filter(
    lambda row: row['mark'] == 0 and row['gender'] == 'female'
).map(lambda row: row['sport']).collect()
countM0 = rdd8Count.filter(
    lambda row: row['mark'] == 0 and row['gender'] == 'male'
).map(lambda row: row['count']).collect()
countFM0 = rdd8Count.filter(
    lambda row: row['mark'] == 0 and row['gender'] == 'female'
).map(lambda row: row['count']).collect()

sportTypeM1 = rdd8Count.filter(
    lambda row: row['mark'] == 1 and row['gender'] == 'male'
).map(lambda row: row['sport']).collect()
sportTypeFM1 = rdd8Count.filter(
    lambda row: row['mark'] == 1 and row['gender'] == 'female'
).map(lambda row: row['sport']).collect()
countM1 = rdd8Count.filter(
    lambda row: row['mark'] == 1 and row['gender'] == 'male'
).map(lambda row: row['count']).collect()
countFM1 = rdd8Count.filter(
    lambda row: row['mark'] == 1 and row['gender'] == 'female'
).map(lambda row: row['count']).collect()

sportTypeM2 = rdd8Count.filter(
    lambda row: row['mark'] == 2 and row['gender'] == 'male'
).map(lambda row: row['sport']).collect()
sportTypeFM2 = rdd8Count.filter(
    lambda row: row['mark'] == 2 and row['gender'] == 'female'
).map(lambda row: row['sport']).collect()

countM2 = rdd8Count.filter(
    lambda row: row['mark'] == 2 and row['gender'] == 'male'

).map(lambda row: row['count']).collect()
countFM2 = rdd8Count.filter(
    lambda row: row['mark'] == 2 and row['gender'] == 'female'
).map(lambda row: row['count']).collect()

sportTypeM3 = rdd8Count.filter(
    lambda row: row['mark'] == 3 and row['gender'] == 'male'
).map(lambda row: row['sport']).collect()
sportTypeFM3 = rdd8Count.filter(
    lambda row: row['mark'] == 3 and row['gender'] == 'female'
).map(lambda row: row['sport']).collect()
countM3 = rdd8Count.filter(
    lambda row: row['mark'] == 3 and row['gender'] == 'male'
).map(lambda row: row['count']).collect()
countFM3 = rdd8Count.filter(
    lambda row: row['mark'] == 3 and row['gender'] == 'female'
).map(lambda row: row['count']).collect()
fig,axes = plt.subplots(4,2,figsize=(10,25))
subplot_adj = plt.subplots_adjust(wspace=0.3, hspace=0.7)

for i in range(8):
    ax = plt.subplot(4, 2, i+1)
    sca_x = plt.sca(ax)
    xticks = plt.xticks(rotation=90)
    xlabel = plt.xlabel('Sport')
    ylabel = plt.ylabel('Workouts count')
    if i%2 == 0:
        title = plt.title('Time:' + str(int(i / 2 * 6)) + ' ~ ' + str(int(i / 2 * 6) + 6) + 'h___Male')
        sportType = locals()['sportTypeM' + str(int(i / 2))]
        count = locals()['countM' + str(int(i/2))]
        facecolor = '#9999ff'
        plot = bar(sportType, count, facecolor=facecolor, edgecolor='white')
    else:
        title = plt.title(
            'Time:' + str(int((i - 1) / 2 * 6)) + ' ~ ' + str(int((i - 1) / 2 * 6)+ 6) + 'h___Female'
        )
        sportType = locals()['sportTypeFM'+str(int(i/2))]
        count = locals()['countFM'+str(int(i/2))]
        facecolor = '#ff9999'
        plot = bar(sportType, count, facecolor=facecolor, edgecolor='white')
a = fig.tight_layout()
chart_title =fig.text(0.5, 1, 'Workout count per time range by sport & gender', 
            ha='center', va='top', fontsize='medium', transform=fig.transFigure)
fig,axes = plt.subplots(4,2, figsize=(15, 35))


for i in range(8):
    ax = plt.subplot(4, 2 , i + 1)
    plt.sca(ax)
    if i%2 == 0:
        subplot_title = plt.title('Time:' + str(int(i/2*6)) + ' ~ ' + str(int(i/2 * 6) + 6) + 'h___Male ')
        sportType = locals()['sportTypeM' + str(int(i/2))]
        count = locals()['countM' + str(int(i/2))]
        explode = tuple([0.1 for i in range(len(sportType))])
        
        facecolor = '#9999ff'
        plot = plt.pie(x=count, autopct='%.2f%%'#, shadow=True
                       , labels=sportType, explode=explode)
    else:
        subplot_title = plt.title('Time:' + str(int((i-1)/2*6)) + ' ~ ' + str(int((i-1)/2*6) + 6) +'h___Female ')
        sportType = locals()['sportTypeFM' + str(int(i/2))]
        count = locals()['countFM'+str(int(i/2))]
        explode = tuple([0.1 for i in range(len(sportType))])
        
        facecolor = '#ff9999'
        plot = plt.pie(x=count, labels=sportType#, shadow=True
                       , autopct='%.2f%%', explode=explode)

InteractiveShell.ast_node_interactivity = "all"
rdd8 = df8.rdd
def avgReHeartRate(row):
    if row['reHeartRate'] != '':
        reAvgHeartRate = np.mean(row['reHeartRate'])
        reVarHeartRate = np.var(row['reHeartRate'])
        reAvgAltitude = np.mean(row['reAltitude'])
        reVarAltitude = np.var(row['reAltitude'])
        reAvgSpeed = np.mean(row['reSpeed'])
        reVarSpeed = np.var(row['reSpeed'])
        return Row(
            id = int(row['id'])
            , mark = row['mark']
            , gender = str(row['gender'])
            , sport = str(row['sport'])
            , userId = int(row['userId'])
            , reAvgHeartRate = float(reAvgHeartRate)
            , reVarHeartRate = float(reVarHeartRate)
            , reAvgAltitude = float(reAvgAltitude)
            , reVarAltitude = float(reVarAltitude)
            , reAvgSpeed = float(reAvgSpeed)
            , reVarSpeed = float(reVarSpeed)
        )
   
rdda8HR = rdd8.map(avgReHeartRate)

df9 = spark.createDataFrame(rdda8HR)
DataFrame(df9.dtypes, columns = ['Column Name','Data type'])


df9AvgHearRate=df9.groupBy(['mark','sport','gender']) \
    .avg(
        'reAvgHeartRate', 'reVarHeartRate'
        , 'reAvgAltitude', 'reVarAltitude'
        , 'reAvgSpeed', 'reVarSpeed'
    ).orderBy(['mark','sport','gender'])

df9AvgHearRate.limit(5).toPandas()
rdd9AvgHearRate = df9AvgHearRate.rdd
rddtimeM0 = rdd9AvgHearRate.filter(lambda row: row['mark'] == 0 and row['gender'] == 'male')
rddtimeM1 = rdd9AvgHearRate.filter(lambda row: row['mark'] == 1 and row['gender'] == 'male')
rddtimeM2 = rdd9AvgHearRate.filter(lambda row: row['mark'] == 2 and row['gender'] == 'male')
rddtimeM3 = rdd9AvgHearRate.filter(lambda row: row['mark'] == 3 and row['gender'] == 'male')
rddtimeFM0 = rdd9AvgHearRate.filter(lambda row: row['mark'] == 0 and row['gender'] == 'female')
rddtimeFM1 = rdd9AvgHearRate.filter(lambda row: row['mark'] == 1 and row['gender'] == 'female')
rddtimeFM2 = rdd9AvgHearRate.filter(lambda row: row['mark'] == 2 and row['gender'] == 'female')
rddtimeFM3 = rdd9AvgHearRate.filter(lambda row: row['mark'] == 3 and row['gender'] == 'female')

dftimeM0 = spark.createDataFrame(rddtimeM0) \
    .withColumnRenamed('avg(reAvgHeartRate)', 'MreAvgSptAvgHeartRate0') \
    .withColumnRenamed('avg(reVarHeartRate)', 'MreAvgSptVarHeartRate0') \
    .withColumnRenamed('avg(reAvgAltitude)', 'MreAvgSptAvgAltitude0') \
    .withColumnRenamed('avg(reVarAltitude)', 'MreAvgSptVarAltitude0') \
    .withColumnRenamed('avg(reAvgSpeed)', 'MreAvgSptAvgSpeed0') \
    .withColumnRenamed('avg(reVarSpeed)', 'MreAvgSptVarSpeed0')
    
dftimeM1 = spark.createDataFrame(rddtimeM1) \
    .withColumnRenamed('avg(reAvgHeartRate)', 'MreAvgSptAvgHeartRate1') \
    .withColumnRenamed('avg(reVarHeartRate)', 'MreAvgSptVarHeartRate1') \
    .withColumnRenamed('avg(reAvgAltitude)', 'MreAvgSptAvgAltitude1') \
    .withColumnRenamed('avg(reVarAltitude)', 'MreAvgSptVarAltitude1') \
    .withColumnRenamed('avg(reAvgSpeed)', 'MreAvgSptAvgSpeed1') \
    .withColumnRenamed('avg(reVarSpeed)', 'MreAvgSptVarSpeed1')

dftimeM2 = spark.createDataFrame(rddtimeM2) \
    .withColumnRenamed('avg(reAvgHeartRate)', 'MreAvgSptAvgHeartRate2') \
    .withColumnRenamed('avg(reVarHeartRate)', 'MreAvgSptVarHeartRate2') \
    .withColumnRenamed('avg(reAvgAltitude)', 'MreAvgSptAvgAltitude2') \
    .withColumnRenamed('avg(reVarAltitude)', 'MreAvgSptVarAltitude2') \
    .withColumnRenamed('avg(reAvgSpeed)', 'MreAvgSptAvgSpeed2') \
    .withColumnRenamed('avg(reVarSpeed)', 'MreAvgSptVarSpeed2')

dftimeM3 = spark.createDataFrame(rddtimeM3) \
    .withColumnRenamed('avg(reAvgHeartRate)', 'MreAvgSptAvgHeartRate3') \
    .withColumnRenamed('avg(reVarHeartRate)', 'MreAvgSptVarHeartRate3') \
    .withColumnRenamed('avg(reAvgAltitude)', 'MreAvgSptAvgAltitude3') \
    .withColumnRenamed('avg(reVarAltitude)', 'MreAvgSptVarAltitude3') \
    .withColumnRenamed('avg(reAvgSpeed)', 'MreAvgSptAvgSpeed3') \
    .withColumnRenamed('avg(reVarSpeed)', 'MreAvgSptVarSpeed3')

dftimeFM0 = spark.createDataFrame(rddtimeFM0) \
    .withColumnRenamed('avg(reAvgHeartRate)', 'FMreAvgSptAvgHeartRate0') \
    .withColumnRenamed('avg(reVarHeartRate)', 'FMreAvgSptVarHeartRate0') \
    .withColumnRenamed('avg(reAvgAltitude)', 'FMreAvgSptAvgAltitude0') \
    .withColumnRenamed('avg(reVarAltitude)', 'FMreAvgSptVarAltitude0') \
    .withColumnRenamed('avg(reAvgSpeed)', 'FMreAvgSptAvgSpeed0') \
    .withColumnRenamed('avg(reVarSpeed)','FMreAvgSptVarSpeed0')

dftimeFM1 = spark.createDataFrame(rddtimeFM1) \
    .withColumnRenamed('avg(reAvgHeartRate)', 'FMreAvgSptAvgHeartRate1') \
    .withColumnRenamed('avg(reVarHeartRate)', 'FMreAvgSptVarHeartRate1') \
    .withColumnRenamed('avg(reAvgAltitude)', 'FMreAvgSptAvgAltitude1') \
    .withColumnRenamed('avg(reVarAltitude)', 'FMreAvgSptVarAltitude1') \
    .withColumnRenamed('avg(reAvgSpeed)', 'FMreAvgSptAvgSpeed1') \
    .withColumnRenamed('avg(reVarSpeed)', 'FMreAvgSptVarSpeed1')

dftimeFM2 = spark.createDataFrame(rddtimeFM2) \
    .withColumnRenamed('avg(reAvgHeartRate)', 'FMreAvgSptAvgHeartRate2') \
    .withColumnRenamed('avg(reVarHeartRate)', 'FMreAvgSptVarHeartRate2') \
    .withColumnRenamed('avg(reAvgAltitude)', 'FMreAvgSptAvgAltitude2') \
    .withColumnRenamed('avg(reVarAltitude)', 'FMreAvgSptVarAltitude2') \
    .withColumnRenamed('avg(reAvgSpeed)', 'FMreAvgSptAvgSpeed2') \
    .withColumnRenamed('avg(reVarSpeed)', 'FMreAvgSptVarSpeed2')

dftimeFM3 = spark.createDataFrame(rddtimeFM3) \
    .withColumnRenamed('avg(reAvgHeartRate)', 'FMreAvgSptAvgHeartRate3') \
    .withColumnRenamed('avg(reVarHeartRate)', 'FMreAvgSptVarHeartRate3') \
    .withColumnRenamed('avg(reAvgAltitude)', 'FMreAvgSptAvgAltitude3') \
    .withColumnRenamed('avg(reVarAltitude)', 'FMreAvgSptVarAltitude3') \
    .withColumnRenamed('avg(reAvgSpeed)', 'FMreAvgSptAvgSpeed3') \
    .withColumnRenamed('avg(reVarSpeed)', 'FMreAvgSptVarSpeed3')

dftimeMAll = dftimeM0.join(dftimeM1, 'sport', 'outer') \
    .join(dftimeM2, 'sport', 'outer') \
    .join(dftimeM3, 'sport', 'outer')
dftimeMAll.limit(3).toPandas()
dftimeFMAll = dftimeFM0.join(dftimeFM1, 'sport', 'outer') \
    .join(dftimeFM2, 'sport', 'outer').join(dftimeFM3, 'sport', 'outer')
dftimeFMAll.limit(3).toPandas()
fig, axes = plt.subplots(3, 2, figsize=(17, 27))
plot = plt.subplots_adjust(wspace=0.3, hspace=0.5)

bar_width = 0.2
colors = ['blue', 'lightgreen', 'yellow', 'grey']
        
for j in range(6):
    ax = plt.subplot(3, 2, j+1)
    plt.sca(ax)
    if int(j/2) == 1:
        nameStr1 = 'reAvgSptAvgAltitude'
        nameStr3 = 'Altitude'
        nameStr5 = 'altitude (m)'

    elif int(j/2) == 2:
        nameStr1 = 'reAvgSptAvgSpeed'
        nameStr3 = 'Speed'
        nameStr5 = 'speed (MPH)'

    elif int(j/2) == 0:
        nameStr1 = 'reAvgSptAvgHeartRate'
        nameStr3 = 'HeartRate'
        nameStr5 = 'heart rate (bpm)'        
    if j % 2 == 0:
        nameStr2 = 'M'
        nameStr4 = 'Male'
    else:
        nameStr2 = 'FM'
        nameStr4 = 'Female'
    dftimeAll = locals()['dftime' + nameStr2 + 'All']
    dftimeAllSportType = dftimeAll.rdd.map(lambda row: row['sport']).collect() 
    for k in range(4):
        rowName = nameStr2 + nameStr1 + str(k)
        dfTimeLi = dftimeAll.rdd.map(lambda row: row[rowName]).collect()
        dfTimeLi = [0 if i == None else i for i in dfTimeLi]
        plot = plt.bar(x=np.arange(len(dftimeAllSportType)) + k * bar_width
                , height=dfTimeLi, label='Time:' + str(int(k*6)) + ' ~ ' + str(int(k*6)+6) + 'h'
                , color=colors[k], alpha=0.8, width=bar_width)
    xsticks = plt.xticks(np.arange(len(dftimeAllSportType)) + 1.5 * bar_width
                         , dftimeAllSportType, rotation=90)
    title = plt.title("Comparison of average " + nameStr3 + " between different periods for " + nameStr4)
    legend = plt.legend()
    xlabel = plt.xlabel("Sport")
    ylabel = plt.ylabel("Average " + nameStr5)

InteractiveShell.ast_node_interactivity = "all"
dftimeMIN = dftimeM0.join(dftimeM1, 'sport', 'inner') \
    .join(dftimeM2, 'sport', 'inner') \
    .join(dftimeM3, 'sport', 'inner')
dftimeMIN.toPandas()
dftimeFMIN = dftimeFM0.join(dftimeFM1, 'sport', 'inner') \
    .join(dftimeFM2, 'sport', 'inner') \
    .join(dftimeFM3, 'sport', 'inner')
dftimeFMIN.limit(5).toPandas()
MaverageHeartRate = []
MaverageAltitude = []
MaverageSpeed = []
for i in range(4):
    MaverageHeartRate.append(
        mean(dftimeMIN.rdd.map(
            lambda row: row['MreAvgSptAvgHeartRate' + str(i)]).collect()
            )
    )
    
    MaverageAltitude.append(
        mean(dftimeMIN.rdd.map(
            lambda row: row['MreAvgSptAvgAltitude' + str(i)]).collect()
            )
    )
    MaverageSpeed.append(
        mean(dftimeMIN.rdd.map(
            lambda row: row['MreAvgSptAvgSpeed' + str(i)]).collect()))
    
FMaverageHeartRate = []
FMaverageAltitude = []
FMaverageSpeed = []
for i in range(4):
    FMaverageHeartRate.append(
        mean(dftimeFMIN.rdd.map(
            lambda row: row['FMreAvgSptAvgHeartRate' + str(i)]).collect()
            )
    )
    
    FMaverageAltitude.append(
        mean(dftimeFMIN.rdd.map(
            lambda row: row['FMreAvgSptAvgAltitude' + str(i)]).collect()
            )
    )
    FMaverageSpeed.append(mean(dftimeFMIN.rdd.map(lambda row: row['FMreAvgSptAvgSpeed' + str(i)]).collect()))
fig,axes = plt.subplots(2, 1,figsize=plt.figaspect(0.35))
subplot_adj = plt.subplots_adjust(wspace =0.3, hspace =0.5)

bar_width = 0.2
colors = ['blue', 'lightgreen', 'yellow']
timesP = ['0~6', '6~12', '12~18', '18~24']

for j in range(2):
    ax = plt.subplot(1, 2, j + 1)
    plotsca = plt.sca(ax)       
    if j % 2 == 0:
        nameStr2 = 'M'
        nameStr4 = 'Male'
    else:
        nameStr2 = 'FM'
        nameStr4 = 'Female'    
    for k in range(3):
        if k == 0:
            nameStr6 = 'averageHeartRate'
            nameStr7 = 'average heartrate (bpm)'
        elif k == 1:
            nameStr6 = 'averageAltitude'
            nameStr7 = 'average altitude (m)'
        elif k == 2:
            nameStr6 = 'averageSpeed'
            nameStr7 = 'average speed (MPH)'
        showY = locals()[nameStr2 + nameStr6]
        plot = plt.bar(
            x=np.arange(len(timesP)) + k * bar_width
            , height=showY, label=nameStr7, color=colors[k]
            , alpha=0.8, width=bar_width
        )
    xsticks = plt.xticks(np.arange(len(timesP))+ 0.7 * bar_width, timesP)
    title = plt.title("Average values for different period for " + nameStr4)
    legend = plt.legend()
    xlabel = plt.xlabel("Period (hour)")
    ylabel = plt.ylabel("Average measurement")
InteractiveShell.ast_node_interactivity = "all"
rdd10 = df.rdd

# same_day function already defined in the previous section.
rddSd10=rdd10.filter(lambda row: row['speed'] is not None).map(same_day) \
    .filter(
    lambda row: row['year'] == row['yearl'] and row['month'] == row['monthl'] and row['day'] == row['dayl'])
dfSd10=spark.createDataFrame(rddSd10) \
    .drop('year').drop('yearl').drop('month').drop('day').drop('monthl').drop('dayl')

def markWorkout1(row):
    hours = [(datetime.fromtimestamp(t) - timedelta(hours=7)).hour for t in row['timestamp']]
    mark = -1
    upIndex = -1
    if hours[0] >= 6 and hours[0] < 12:
        mark = 1
    elif hours[0] >= 12 and hours[0] < 18:
        mark=2
    elif hours[0] >= 18 and hours[0] < 24:
        mark = 3
    elif hours[0] >= 0 and hours[0] < 6:
        mark = 0
    if mark != -1:
        perAvgHeatRate = mean(np.array(row['heartrate']))
        perAverageAltitude = mean(np.array(row['altitude']))
        perAverageSpeed = mean(np.array(row['speed']))       
        return Row(
            id = row['id'] \
            , sport = str(row['sport']) \
            , userId = row['userId'] \
            , gender = row['gender'] \
            , mark = mark \
            , perAvgHeatRate = float(perAvgHeatRate) \
            , perAverageAltitude = float(perAverageAltitude) \
            , perAverageSpeed = float(perAverageSpeed)
        )

    
rddMAvg10 = dfSd10.rdd.map(markWorkout1)
dfMAvg10 = spark.createDataFrame(rddMAvg10)
print('Data set preparation:')
DataFrame(dfMAvg10.dtypes, columns = ['Column Name','Data type'])
dfMAvg10.describe().toPandas()
dfMAvg10.limit(3).toPandas()
print('Aggregate values:')
dftmp=dfMAvg10.groupBy(['userId', 'sport']) \
    .agg({'sport':'count', 'perAverageAltitude':'mean'
            , 'perAverageSpeed':'mean', 'perAvgHeatRate':'mean'})
dftmp.describe().toPandas()
dftmp.limit(3).toPandas()
df10MC=dfMAvg10.groupBy(['userId', 'mark']).count()
df10MC.describe().toPandas()
df10MC.limit(3).toPandas()
InteractiveShell.ast_node_interactivity = "all"
all_marks = [0, 1, 2, 3]
def mark_count(valueN): 
    '''
    valueN: tuple of (mark, count)
    '''
    markL = []
    dic = {}
    for item in valueN:
        markL.append(item[0])
    # Find the list of marks not belong to the current mark list in valueN 
    li = list(set(all_marks).difference(set(markL)))
    
    # Update count for the existed marks, and assign non-existed marks with 0
    for item in valueN:
        dic[item[0]] = item[1]
    for m in li:
        dic[m] = 0    
    dicSt = sorted(dic.items(), key=lambda d:d[0])
    markL = [value for key,value in dicSt]   
    return markL

# Generate the list of counts per hour mark for each userId
rdd10MCTF = df10MC.rdd.map(
    lambda row: (row['userId'], (row['mark'], row['count']))
).groupByKey().mapValues(mark_count) \
    .map(lambda row: Row(userId=row[0], markCt=row[1]))

df10MCTF = spark.createDataFrame(rdd10MCTF)

all_sports = sorted(
    dfMAvg10[['sport']].distinct().rdd.map(lambda row: row['sport']).collect()
)

# Generate workout count,  average speed, heart rate & altitude vectors for each user
def f1(valueN):
    sportDic = {}
    averageSpeed = {}
    averageHeartRate = {}
    averageAltitude = {}
    
    sportType = []
    for item in valueN:
        sportType.append(item[0])
    li = list(set(all_sports).difference(set(sportType)))
        
    for item in valueN:
        sportDic[item[0]] = item[4]
        averageSpeed[item[0]] = round(item[1])
        averageHeartRate[item[0]] = round(item[2])
        averageAltitude[item[0]] = round(item[3])
    
    for sp in li:
        sportDic[sp] = 0
        averageSpeed[sp] = 0
        averageHeartRate[sp] = 0
        averageAltitude[sp] = 0
    
    sportDicSt=sorted(sportDic.items(), key=lambda d:d[0])
    sportL=[value for key,value in sportDicSt]
    averageSpeedSt=sorted(averageSpeed.items(), key=lambda d:d[0])
    averageSpeedL=[value for key,value in averageSpeedSt]
    averageHeartRateSt=sorted(averageHeartRate.items(), key=lambda d:d[0])
    averageHeartRateL=[value for key,value in averageHeartRateSt]
    averageAltitudeSt=sorted(averageAltitude.items(), key=lambda d:d[0])
    averageAltitudeL=[value for key,value in averageAltitudeSt]
    
    return sportL, averageSpeedL, averageHeartRateL, averageAltitudeL  
    
rdd10AvgTF = dftmp.rdd.map(
    lambda row: (
        row['userId'], (
                        row['sport'], row['avg(perAverageSpeed)']
                        , row['avg(perAvgHeatRate)'], row['avg(perAverageAltitude)']
                        , row['count(sport)']
                        )
                )
).groupByKey().mapValues(f1).map(
                    lambda row: Row(userId=row[0], sportCt=row[1][0], averageSpeed=row[1][1]
                    , averageHeartRate=row[1][2], averageAltitude=row[1][3])
)

df10AvgTF = spark.createDataFrame(rdd10AvgTF)


rddgender = dfMAvg10[['userId','gender']].distinct().rdd \
    .filter(lambda row: row['gender'] == 'male' or row['gender'] == 'female') \
    .map(
        lambda row: Row(userId = row['userId'], gender = 0) if row['gender'] == 'male' \
        else Row(userId=row['userId'], gender = 1)
    )
dfgender = spark.createDataFrame(rddgender)
df10GMCAvgTF = df10MCTF.join(df10AvgTF, 'userId', 'inner').join(dfgender,'userId', 'inner')
df10GMCAvgTF.describe().toPandas()
print('Final co-ordinate vectors for users, take 3:')
df10GMCAvgTF.limit(3).toPandas()
def marginM(row):
    maxMarkC = max(row['markCt'])
    maxAverageAltitude = max(row['averageAltitude'])
    maxAverageHeartRate = max(row['averageHeartRate'])
    maxAverageSpeed = max(row['averageSpeed'])
    maxSportC = max(row['sportCt'])
    return Row(
        maxMarkC=maxMarkC, maxAverageAltitude=maxAverageAltitude
        , maxAverageHeartRate=maxAverageHeartRate, maxAverageSpeed=maxAverageSpeed
        , maxSportC=maxSportC
    )  

maxRecord = spark.createDataFrame(
    df10GMCAvgTF.rdd.map(marginM)
).groupBy().max(
    'maxMarkC', 'maxAverageAltitude', 'maxAverageHeartRate', 'maxAverageSpeed', 'maxSportC'
    ).rdd.map(lambda row: (row[0], row[1], row[2], row[3], row[4])).collect()

maxMarkC = maxRecord[0][0]
maxAverageAltitude = maxRecord[0][1]
maxAverageHeartRate = maxRecord[0][2]
maxAverageSpeed = maxRecord[0][3]
maxSportC = maxRecord[0][4]

# Define scaling factors to calculate distance for each feature later
markCM = round(maxAverageAltitude / maxMarkC)
AverageHeartRateM = round(maxAverageAltitude / maxAverageHeartRate)
AverageSpeedM = round(maxAverageAltitude / maxAverageSpeed)
SportCM = round(maxAverageAltitude / maxSportC)
genderM = maxAverageAltitude
import random
random.seed(50)
rdd10GMCAvgTF = df10GMCAvgTF.rdd
userId = rdd10GMCAvgTF.map(lambda row: row['userId']).collect()
initUsers = random.sample(userId, 5)
centroids = rdd10GMCAvgTF.filter(lambda row: row['userId'] in initUsers).collect()
# Apply different weights for different features
weightMark = 1
weightAltitude = 1
weightHeartRate = 5
weightSpeed = 1.2
weightSport = 1
weightGender = 3

# Function to assign user to the closest centroid the first time
def assigCent(row):
    distDic = {}
    for centroid in centroids:
        # calculate distances:
        disMark = np.sum(
            np.square(np.array(row['markCt']) - np.array(centroid['markCt']))
        )
        disAltitude = np.sum(
            np.square(np.array(row['averageAltitude']) - np.array(centroid['averageAltitude']))
        )
        disHeartRate = np.sum(
            np.square(np.array(row['averageHeartRate']) - np.array(centroid['averageHeartRate']))
        )
        disSpeed = np.sum(
            np.square(np.array(row['averageSpeed']) - np.array(centroid['averageSpeed']))
        )
        disSport = np.sum(
            np.square(np.array(row['sportCt']) - np.array(centroid['sportCt']))
        )
        disGender = np.sum(np.square(np.array(row['gender']) - np.array(centroid['gender'])))
        # Calculate final distances based on weighted value per feature defined above:
        distDic[centroid['userId']] = weightMark * markCM * disMark \
            + weightAltitude * disAltitude \
            + weightHeartRate * AverageHeartRateM * disHeartRate \
            + weightSpeed * AverageSpeedM * disSpeed \
            + weightSport * SportCM * disSport \
            + weightGender * genderM * disGender
        
    selectedCentroid = min(distDic, key = distDic.get)
    return selectedCentroid

assignedGroup = rdd10GMCAvgTF.map(
    lambda row: (assigCent(row), row['userId'])
).groupByKey().mapValues(list).map(lambda row: row[1]).collect()

firstAssignGroup = assignedGroup

def cluster_summarize_df(assignedGroup):
    display_list = []
    for index, centroid in enumerate(assignedGroup):
          display_list.append({'Group': index, 'Users count': len(centroid)})
    display_df = pd.DataFrame(display_list)
    return display_df

print('After initializing centroid:')
cluster_summarize_df(firstAssignGroup)
# Function to re-calculate co-ordinates for new centroids
def new_coordinates(bList):
    countSum = []
    for i in range(len(bList)):
        if i == 0:
            countSum = np.array(bList[i])
        else:
            countSum += np.array(bList[i])
    return countSum / len(bList)

# Function to assign users to clusters from second time
def assigLCent(row):
    distDic = {}
    for key, value in centroids.items():
        disMark = np.sum(
            np.square(np.array(row['markCt']) - np.array(value['markCt']))
        )
        disAltitude = np.sum(
            np.square(np.array(row['averageAltitude']) - np.array(value['averageAltitude']))
        )
        disHeartRate = np.sum(
            np.square(np.array(row['averageHeartRate']) - np.array(value['averageHeartRate']))
        )
        disSpeed = np.sum(
            np.square(np.array(row['averageSpeed']) - np.array(value['averageSpeed']))
        )
        disSport = np.sum(
            np.square(np.array(row['sportCt']) - np.array(value['sportCt']))
        )
        disGender = np.sum(
            np.square(np.array(row['gender']) - np.array(value['gender']))
        )
        distDic[key] = weightMark * markCM * disMark \
            + weightAltitude * disAltitude \
            + weightHeartRate * AverageHeartRateM * disHeartRate \
            + weightSpeed * AverageSpeedM * disSpeed \
            + weightSport * SportCM * disSport \
            + weightGender * genderM * disGender
    selectedCentroid = min(distDic, key=distDic.get)
    return selectedCentroid

# Function to compare 2 clusters
def compare(r1, r2):
    if len(r1) != len(r2):
        return False
    for i in range(len(r1)):
        list_found = False
        for j in range(len(r2)):
            res_list = r1[i]
            a_list = r2[j]            
            if res_list == a_list:
                list_found = True                  
                break
        if not list_found:
            return False
    return True   


def sortedgroup(groups):
    sortedG = []
    for group in groups:
        sortedG.append(sorted(group))
    return sortedG
        
times = 0
maxIter = 20
compResult = False

# Converge / loop break condition
while compResult == False and times <= maxIter:
    
    newCentroid={}
    i = 0
    # Re-calculate co-ordinates for new centroids
    for group in assignedGroup:
        markList = new_coordinates(
            rdd10GMCAvgTF.filter(
                lambda row: row['userId'] in group
            ).map(lambda row: row['markCt']).collect()
        )
        
        altitudeList = new_coordinates(
            rdd10GMCAvgTF.filter(
                lambda row: row['userId'] in group
            ).map(lambda row: row['averageAltitude']).collect()
        )
        
        heartRateList = new_coordinates(
            rdd10GMCAvgTF.filter(
                lambda row: row['userId'] in group
            ).map(lambda row: row['averageHeartRate']).collect()
        )
        
        speedList = new_coordinates(
            rdd10GMCAvgTF.filter(
                lambda row: row['userId'] in group
            ).map(lambda row: row['averageSpeed']).collect()
        )
        sportList = new_coordinates(
            rdd10GMCAvgTF.filter(
                lambda row: row['userId'] in group
            ).map(lambda row: row['sportCt']).collect()
        )
        
        genderList = new_coordinates(
            rdd10GMCAvgTF.filter(
                lambda row: row['userId'] in group
            ).map(lambda row: row['gender']).collect()
        )
        
        newCentroid[i] = {
            'markCt':markList, 'averageAltitude':altitudeList
            , 'averageHeartRate':heartRateList, 'averageSpeed':speedList
            , 'sportCt':sportList, 'gender':genderList
        }
        
        i += 1

    centroids = newCentroid 
    
    # Assign users to new groups
    newAssignedGroup = rdd10GMCAvgTF.map(
        lambda row: (assigLCent(row), row['userId'])
    ).groupByKey().mapValues(list).map(lambda row: row[1]).collect()
    
    print('\nIteration #{} - Group summary:'.format(times))
    cluster_summarize_df(newAssignedGroup)
    
    # Check if new clusters are the same with the preivous iteration
    compResult = compare(sortedgroup(assignedGroup), sortedgroup(newAssignedGroup))
    
    if compResult == True or times == maxIter:
        print("---------")
        print('Clusters converged after {} iterations'.format(times + 1))
        
    assignedGroup=newAssignedGroup
        
    times+=1
    
InteractiveShell.ast_node_interactivity = "all"

def reRow(li):
    count = 0
    sumV = 0
    for i in li:
        if i != 0:
            sumV += i
            count += 1
    if count != 0:
        value = sumV / count
        return value
    else:
        return 0        

def countAverage(row):
    avgAltitudeSP = reRow(row['averageAltitude'])
    avgHeartRateSP = reRow(row['averageHeartRate'])
    avgSpeedSP = reRow(row['averageSpeed'])
    return Row(
        useId = row['userId'], markCt = row['markCt']
        , avgAltitudeSP = float(avgAltitudeSP), avgHeartRateSP = float(avgHeartRateSP)
        , avgSpeedSP = float(avgSpeedSP), sportCt = row['sportCt'], gender=row['gender']
    )
    
temprdds = []
for group in firstAssignGroup:
    temprdds.append(
        spark.createDataFrame(
            rdd10GMCAvgTF.filter(lambda row: row['userId'] in group).map(countAverage)
        )
    )
print('Summary for first assigned clusters:')
for i, temprdd in enumerate(temprdds):
    print('\nGroup #{}:'.format(i))
    temprdd.toPandas().describe()

for group in assignedGroup:
    temprdds.append(
        spark.createDataFrame(
            rdd10GMCAvgTF.filter(lambda row: row['userId'] in group).map(countAverage)
        )
    )
print('Summary for final clusters:')    
for i, temprdd in enumerate(temprdds[5:]):
    print('\nGroup #{}:'.format(i))
    temprdd.toPandas().describe()


fig, axes = plt.subplots(1, 2, figsize=plt.figaspect(0.3))
subplot_adj = plt.subplots_adjust(wspace =0.3, hspace =0.5)

bar_width = 0.2
color = ['blue', 'lightgreen', 'yellow']
xShow = ['Group {}'.format(i+1) for i in range(5)]

avgAltitude = []
avgHeartrate = []
avgSpeed = []
avgAltitude1 = []
avgHeartrate1 = []
avgSpeed1 = []
for i in range(len(temprdds)):
    if i < 5:
        avgAltitude.append(
            np.mean(
                np.array(temprdds[i].rdd.map(lambda row: row['avgAltitudeSP']).collect())
            )
        )
        
        avgHeartrate.append(
            np.mean(np.array(temprdds[i].rdd.map(lambda row: row['avgHeartRateSP']).collect())
                   )
        )
        
        avgSpeed.append(
            np.mean(np.array(temprdds[i].rdd.map(lambda row: row['avgSpeedSP']).collect())
                   )
        )
    else:
        avgAltitude1.append(
            np.mean(np.array(temprdds[i].rdd.map(lambda row: row['avgAltitudeSP']).collect())
                   )
        )
        avgHeartrate1.append(
            np.mean(np.array(temprdds[i].rdd.map(lambda row: row['avgHeartRateSP']).collect())
                   )
        )
        avgSpeed1.append(
            np.mean(np.array(temprdds[i].rdd.map(lambda row: row['avgSpeedSP']).collect())
                   )
        )
ax = plt.subplot(1,2,1)
sca = plt.sca(ax)    
plot = plt.bar(
    x=np.arange(len(xShow)), height=avgHeartrate
    , label='Heart rate (bpm)', color=color[0], alpha=0.8, width=bar_width
)

plot = plt.bar(
    x=np.arange(len(xShow)) + 1 * bar_width, height=avgAltitude
    , label='Altitude (m)', color=color[1], alpha=0.8, width=bar_width
)
plot = plt.bar(
    x=np.arange(len(xShow)) + 2 * bar_width, height=avgSpeed
    , label='Speed (mph)', color=color[2], alpha=0.8, width=bar_width
)
xsticks = plt.xticks(np.arange(len(xShow)) + 0.7 * bar_width, xShow, rotation=90)
title = plt.title("First users assignment / classification")
legend = plt.legend()
xlabel = plt.xlabel("User Group")
ylabel = plt.ylabel("Average measurement")

ax=plt.subplot(1,2,2)
sca = plt.sca(ax) 
plot = plt.bar(
    x=np.arange(len(xShow)), height=avgHeartrate1
    , label='Heart rate (bpm)', color=color[0], alpha=0.8, width=bar_width
)
plot = plt.bar(
    x=np.arange(len(xShow)) + 1 * bar_width, height=avgAltitude1
    , label='Altitude (m)', color=color[1], alpha=0.8, width=bar_width
)
plot = plt.bar(
    x=np.arange(len(xShow)) + 2 * bar_width, height=avgSpeed1
    , label='Speed (mph)', color=color[2], alpha=0.8, width=bar_width
)
xsticks = plt.xticks(np.arange(len(xShow)) + 0.7 * bar_width, xShow, rotation=90)
title = plt.title("Classification result after {} times".format(times))
legend = plt.legend()
xlabel = plt.xlabel("User Group")
ylabel = plt.ylabel("Average measurement")
fig_title = fig.text(
    0.5, 1.01, 'Comparison between first and final iteration of k-means clustering (k = 5)'
    , ha='center', va='top', transform=fig.transFigure, fontsize='medium'
)
