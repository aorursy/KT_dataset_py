# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
database = ("../input/database.sqlite")
conn = sqlite3.connect(database)


tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)
tables
year=pd.read_sql("""SELECT DISTINCT BusinessYear FROM Rate;""", conn)
year
table_servicearea = pd.read_sql("""SELECT *
                        FROM servicearea
                        LIMIT 100;""", conn)
table_servicearea
state_in_area=pd.read_sql("""SELECT DISTINCT statecode from servicearea order by statecode;""", conn)
state_in_area
servicearea_issuerid=pd.read_sql("""select statecode, count(distinct(issuerid)) as num_issuer
                                    , count(distinct(serviceareaid)) as num_service
                                    , count(distinct (serviceareaname)) as num_servicename
                                    from servicearea
                                    group by statecode
                                    order by statecode;""", conn)
servicearea_issuerid
source_type=pd.read_sql("""SELECT distinct(SourceName)
                        from ServiceArea 
                        ; """, conn)
source_type
# see how the popuar of each source in states
source_popularity=pd.read_sql("""select sourcename, count(distinct(statecode)) 
                            from servicearea group by sourcename; """, conn)
source_popularity
# see how many services of each source in states
source_count=pd.read_sql("""select sourcename, count(statecode) as service_number
                            from servicearea group by sourcename order by sourcename; """, conn)
source_count
source_state=pd.read_sql("""SELECT StateCode, SourceName, COUNT(SourceName) as num_source 
                        from ServiceArea 
                        group by StateCode, SourceName 
                        order by num_source desc; """, conn)
source_state
table_rate = pd.read_sql("""SELECT *
                        FROM Rate
                        LIMIT 30;""", conn)
table_rate
rate_state=pd.read_sql("""select businessyear, statecode, avg(individualrate) as rate_ave
                        from rate 
                        group by businessyear, statecode
                        order by rate_ave desc;""", conn)
rate_state
rate_wy_TOP=pd.read_sql("""select individualrate 
                        from rate 
                        where statecode='WY' and businessyear=2014
                        ORDER BY INDIVIDUALRATE DESC
                        limit 1000;""", conn)
rate_wy_TOP
rate_AK_TOP=pd.read_sql("""select individualrate 
                        from rate 
                        where statecode='AK' and businessyear=2014
                        ORDER BY INDIVIDUALRATE DESC
                        LIMIT 550;""", conn)
rate_AK_TOP
rate_wy_TOP_2016=pd.read_sql("""select individualrate 
                        from rate 
                        where statecode='WY' and businessyear=2016
                        ORDER BY INDIVIDUALRATE DESC
                        LIMIT 1000;""", conn)
rate_wy_TOP_2016
rate_PLAN_WY=pd.read_sql("""select DISTINCT(PLANID)
                                from rate 
                                WHERE STATECODE='WY' AND INDIVIDUALRATE=999999
                                ;""", conn)
rate_PLAN_WY
rate_PLANS_2014_WY=pd.read_sql("""select DISTINCT(PLANID)
                                from rate 
                                WHERE BUSINESSYEAR=2014 AND STATECODE='WY'
                                ;""", conn)
rate_PLANS_2014_WY
rate_PLANS_2014_EXP=pd.read_sql("""select  STATECODE, PLANID
                                    from rate 
                                    WHERE BUSINESSYEAR=2014 AND INDIVIDUALRATE=999999
                                    GROUP BY PLANID
                                    ORDER BY STATECODE;""", conn)
rate_PLANS_2014_EXP
rate_PLANS_2016_2014EXP=pd.read_sql("""SELECT STATECODE, PLANID
                                    FROM RATE
                                    WHERE BUSINESSYEAR=2016 AND 
                                    PLANID IN
                                    (select PLANID
                                    from rate 
                                    WHERE BUSINESSYEAR=2014 AND INDIVIDUALRATE=999999
                                    )
                                     ;""", conn)
rate_PLANS_2016_2014EXP
rate_PLANS_2016=pd.read_sql("""SELECT STATECODE, PLANID
                                    FROM RATE
                                    WHERE BUSINESSYEAR=2016 AND 
                                    PLANID = '74819AK0010001'
                                     ;""", conn)
rate_PLANS_2016
rate_PLANS_2016WY=pd.read_sql("""SELECT DISTINCT(PLANID)
                                    FROM RATE
                                    WHERE BUSINESSYEAR=2016 AND 
                                    STATECODE = 'WY' 
                                    AND PLANID NOT IN ( '47731WY0030002', '47731WY0030001','47731WY0020002', '47731WY0020001' ) 
                                     ;""", conn)
rate_PLANS_2016WY
PLANTYPE=pd.read_sql("""SELECT PLANID, PLANTYPE, BenefitPackageId
                        FROM PLANATTRIBUTES
                        WHERE PLANID IN
                                    (select PLANID
                                    from rate 
                                    WHERE BUSINESSYEAR=2014 AND INDIVIDUALRATE=999999)
                        ;""", conn)
PLANTYPE
PLANTYPE1=pd.read_sql("""SELECT PLANID, PLANTYPE, BenefitPackageId
                        FROM PLANATTRIBUTES
                        WHERE PLANID IN
                                    (SELECT DISTINCT(PLANID)
                                    FROM RATE
                                    WHERE BUSINESSYEAR=2014 AND 
                                    STATECODE = 'WY' 
                                    AND PLANID NOT IN ( '47731WY0030002', '47731WY0030001','47731WY0020002', '47731WY0020001'))
                        ;""", conn)
PLANTYPE1
PLANID_IN_ATTRI=pd.read_sql("""SELECT DISTINCT (PLANID)
                                FROM planattributes
                                where statecode='WY' AND BUSINESSYEAR=2014;""", conn)
PLANID_IN_ATTRI
PLANTYPE_MODIFY=pd.read_sql("""SELECT planid, PLANTYPE, BenefitPackageId, PlanMarketingName, ISSUERID
                                FROM PLANATTRIBUTES
                                WHERE SUBSTR(PLANATTRIBUTES.PLANID,1, 14) IN
                                (select PLANID
                                from rate 
                                WHERE BUSINESSYEAR=2014 AND INDIVIDUALRATE=999999)
                                ;""", conn)
PLANTYPE_MODIFY
PLANTYPE_MODIFY=pd.read_sql("""SELECT PLANTYPE, PlanMarketingName
                                FROM PLANATTRIBUTES
                                WHERE SUBSTR(PLANATTRIBUTES.PLANID,1, 14) IN
                                (select PLANID
                                from rate 
                                WHERE BUSINESSYEAR=2014 AND INDIVIDUALRATE=999999)
                                GROUP BY PLANMARKETINGNAME
                                ;""", conn)
PLANTYPE_MODIFY
rate_state_reg=pd.read_sql("""select businessyear, statecode, avg(individualrate) as rate_ave
                        from rate 
                        WHERE INDIVIDUALRATE != 999999
                        group by businessyear, statecode
                        order by STATECODE;""", conn)
rate_state_reg
rate_state_pivot1=pd.read_sql("""select  statecode, businessyear,avg(individualrate) as rate_ave
                                        from rate 
                                        WHERE businessyear in (2014, 2015, 2016) and INDIVIDUALRATE != 999999
                                        group by businessyear, statecode
                                        ;""", conn)
rate_state_pivot1
rate_state_pivot=pd.read_sql("""select statecode,
                                        SUM(CASE WHEN BusinessYear = 2014 THEN rate_ave END) AS '2014',
                                         SUM(CASE WHEN BusinessYear = 2015 THEN rate_ave  END) AS '2015',
                                         SUM(CASE WHEN BusinessYear = 2016 THEN rate_ave  END) AS '2016'
                                from (select  statecode, businessyear,avg(individualrate) as rate_ave
                                        from rate 
                                        WHERE INDIVIDUALRATE != 999999
                                        group by businessyear, statecode
                                        )
                                group by statecode;""", conn)
rate_state_pivot
dental_plan=pd.read_sql("""select statecode, businessyear, count(distinct(planid)) as num_dental
                                from planattributes
                                where dentalonlyplan = 'Yes'
                                group by statecode, businessyear
                                order by statecode;""", conn)
dental_plan
total_plan=pd.read_sql("""select statecode, businessyear, count (distinct (planid)) as total_plan
                                from planattributes
                                group by statecode, businessyear
                                order by statecode;""", conn)
total_plan
dental_total_plan=dental_plan.merge(total_plan)
dental_total_plan
medical_rate=pd.read_sql("""select rate.statecode, rate.businessyear, avg(rate.individualrate) as medical_rate
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.dentalonlyplan='No' 
                            group by rate.statecode, rate.businessyear
                            order by rate.statecode;""", conn)
medical_rate
dental_rate=pd.read_sql("""select rate.statecode, rate.businessyear, avg(rate.individualrate) as medicine_rate
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.dentalonlyplan='Yes' 
                            group by rate.statecode, rate.businessyear
                            order by rate.statecode;""", conn)
dental_rate
dental_realrate=pd.read_sql("""select rate.statecode, rate.businessyear, avg(rate.individualrate) as dental_rate
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.dentalonlyplan='Yes' and rate.individualrate !=999999 
                            group by rate.statecode, rate.businessyear
                            order by rate.statecode;""", conn)
dental_realrate
medical_dental_rate=medical_rate.merge(dental_realrate)
medical_dental_rate
age_rate=pd.read_sql("""select distinct (age) from rate;""", conn)
age_rate
rate_age=pd.read_sql("""select avg(individualrate) as rate, age
                        from rate
                        where individualrate !=999999
                        group by age
                        ;""", conn)
rate_age
fig, ax=plt.subplots(figsize=[20, 5])
sns.barplot(x='Age', y='rate', data=rate_age)
medical_rate_age=pd.read_sql("""select rate.statecode, avg(rate.individualrate) as medical_rate, rate.age
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.dentalonlyplan='No' 
                            group by rate.statecode, rate.age
                            order by rate.statecode;""", conn)
medical_rate_age
medical_rate_age=medical_rate_age.pivot(index= 'StateCode', columns= 'Age', values='medical_rate')
medical_rate_age.head()
fig, ax=plt.subplots(figsize=[20,10])
sns.heatmap(medical_rate_age)
dental_realrate_age=pd.read_sql("""select rate.statecode,  avg(rate.individualrate) as dental_rate, rate.age
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.dentalonlyplan='Yes' and rate.individualrate !=999999 
                            group by rate.statecode, rate.age
                            order by rate.statecode;""", conn)
dental_realrate_age
dental_realrate_age=dental_realrate_age.pivot(index= 'StateCode', columns= 'Age', values='dental_rate')
dental_realrate_age.head()
fig, (axes1, axes2)=plt.subplots(2,1,figsize=[20,20])
sns.heatmap(medical_rate_age, ax=axes1)
sns.heatmap(dental_realrate_age, ax=axes2)
dental_UT=pd.read_sql("""select rate.planid, rate.individualrate as dental_rate
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.statecode= 'UT' and planattributes.dentalonlyplan='Yes'
                            group by rate.planid
                            order by dental_rate desc;""", conn)
dental_UT
dental_wy=pd.read_sql("""select rate.planid, rate.individualrate as dental_rate
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.statecode= 'WY' and planattributes.dentalonlyplan='Yes'
                            group by rate.planid
                            ORDER BY dental_rate desc;""", conn)
dental_wy