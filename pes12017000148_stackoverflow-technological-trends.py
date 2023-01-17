%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # graphs and charts

import pandas_profiling # generating Profile Report



import bq_helper # accessing bigQuery database



import sklearn

from sklearn.model_selection import train_test_split # data splitting

import statsmodels.api as sm

from sklearn import metrics

from sklearn.linear_model import LinearRegression # Linear model



import wordcloud
stackoverflow = bq_helper.BigQueryHelper("bigquery-public-data","stackoverflow")
stackoverflow.list_tables()
stackoverflow.head("posts_questions")
stackoverflow.table_schema("posts_questions")
queryx = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >= 2009 and extract(year from creation_date) < 2019

        group by year

        order by year

        """



PostsCount = stackoverflow.query_to_pandas(queryx)

print(PostsCount)
PostsCount.describe()
# data.isnull.sum()

# data['favorite_count'].fillna(0,inplace=True)

# data.head()
data = pd.DataFrame(PostsCount)

pandas_profiling.ProfileReport(data)
PostsCount.head()
query4 = """SELECT tags

         FROM `bigquery-public-data.stackoverflow.posts_questions`

         LIMIT 200000;

         """



alltags = stackoverflow.query_to_pandas_safe(query4)

tags = ' '.join(alltags.tags).lower()
cloud = wordcloud.WordCloud(background_color='black',

                            max_font_size=200,

                            width=1600,

                            height=800,

                            max_words=300,

                            relative_scaling=.5).generate(tags)

plt.figure(figsize=(20,10))

plt.axis('off')

plt.savefig('stackOverflow.png')

plt.imshow(cloud);
pd.to_numeric(PostsCount['year'])
year=PostsCount['year'].values.reshape(-1,1)

#print (year)

posts=PostsCount['posts'].values.reshape(-1,1)

#print (posts)
reg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(year,posts,test_size=0.2,shuffle=False)

# print(X_train)

# print(X_test)

# print(y_train)

# print(y_test)
reg.fit(X_train,y_train)

predictions = reg.predict(X_test)
print('Predicted values\n',predictions)
plt.scatter(X_train,y_train, color = "black")

plt.scatter(X_test, y_test, color = "green")

plt.plot(X_test, predictions, color = "red")

plt.gca().legend(('Y-Predicted','Y-Train', 'Y-Test'))

plt.title('Y-train and Y-test and Y-predicted')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
plt.scatter(X_test, y_test, color = "green")

plt.plot(X_test, predictions, color = "red")

plt.gca().legend(('Y-Train','Y-Test'))

plt.title('Y-test and Y-predicted')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
reg.score(X_test,y_test)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#angularjs,bootstrap,php,html,javascript,css

query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and (tags like '%bootstrap%' or 

        tags like '%angularjs%' or tags like '%php%' or tags like '%html%' or tags like '%javascript%' or tags like '%css%')

        group by year

        order by year

        """



WebDev_Posts = stackoverflow.query_to_pandas(query)

WebDev_Posts['posts']= WebDev_Posts['posts']*100/PostsCount.posts

WebDev_Posts
WebDev_Posts.describe()
pd.to_numeric(WebDev_Posts['year'])
WebDevYear=WebDev_Posts['year'].values.reshape(-1,1)

#print (WebDevYear)

WebDevPosts=WebDev_Posts['posts'].values.reshape(-1,1)

#print (WebDevPosts)
XWebDev_train, XWebDev_test, yWebDev_train, yWebDev_test = train_test_split(WebDevYear,WebDevPosts,test_size=0.2,shuffle=False)

# print(XWebDev_train)

# print(XWebDev_test)

# print(yWebDev_train)

# print(yWebDev_test)
WebDevReg=LinearRegression()

WebDevReg.fit(XWebDev_train,yWebDev_train)

WebDevPredictions = WebDevReg.predict(XWebDev_test)

print('Predicted Values:\n',WebDevPredictions)
plt.scatter(XWebDev_train,yWebDev_train, color = "black")

plt.scatter(XWebDev_test, yWebDev_test, color = "green")

plt.plot(XWebDev_test, WebDevPredictions, color = "red")

plt.gca().legend(('Y-Predicted','Y-Train', 'Y-Test'))

plt.title('WEB DEVELOPMENT')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
plt.scatter(XWebDev_test, yWebDev_test, color = "green")

plt.plot(XWebDev_test, WebDevPredictions, color = "red")

plt.gca().legend(('Y-Train','Y-Test'))

plt.title('Web Development')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
WebDevReg.score(XWebDev_test,yWebDev_test)
print('Mean Squared Error:',metrics.mean_squared_error(yWebDev_test, WebDevPredictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yWebDev_test, WebDevPredictions)))
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%angularjs%'

        group by year

        order by year

        """



AngularJSPosts = stackoverflow.query_to_pandas(query)

AngularJSPosts['posts']= AngularJSPosts['posts']*100/PostsCount.posts

AngularJSPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%bootstrap%'

        group by year

        order by year

        """



BootstrapPosts = stackoverflow.query_to_pandas(query)

BootstrapPosts['posts']= BootstrapPosts['posts']*100/PostsCount.posts

pd.to_numeric(BootstrapPosts['year'])

BootstrapPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%php%'

        group by year

        order by year

        """



PHPPosts = stackoverflow.query_to_pandas(query)

PHPPosts['posts']= PHPPosts['posts']*100/PostsCount.posts

pd.to_numeric(PHPPosts['year'])

PHPPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%html%'

        group by year

        order by year

        """



htmlPosts = stackoverflow.query_to_pandas(query)

htmlPosts['posts']= htmlPosts['posts']*100/PostsCount.posts

pd.to_numeric(htmlPosts['year'])

htmlPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%javascript%'

        group by year

        order by year

        """



JavaScriptPosts = stackoverflow.query_to_pandas(query)

JavaScriptPosts['posts']= JavaScriptPosts['posts']*100/PostsCount.posts

pd.to_numeric(JavaScriptPosts['year'])

JavaScriptPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%css%'

        group by year

        order by year

        """



CSSPosts = stackoverflow.query_to_pandas(query)

CSSPosts['posts']= CSSPosts['posts']*100/PostsCount.posts

pd.to_numeric(CSSPosts['year'])

CSSPosts
WebDev= pd.merge(PHPPosts, htmlPosts, how='inner', on = 'year')

WebDev=WebDev.set_index('year')

WebDev= pd.merge(WebDev, JavaScriptPosts, how='inner', on = 'year')

WebDev =WebDev.set_index('year')

WebDev=pd.merge(WebDev,AngularJSPosts,how='inner',on='year')

WebDev = WebDev.set_index('year')

WebDev=pd.merge(WebDev,BootstrapPosts,how='inner',on='year')

WebDev = WebDev.set_index('year')

WebDev=pd.merge(WebDev,CSSPosts,how='inner',on='year')

WebDev = WebDev.set_index('year')



WebDev.plot(kind='line')

plt.xlabel('Year', fontsize=15)

plt.ylabel('Posts %', fontsize=15)

y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]



plt.xticks(y_pos,fontsize=10)

plt.yticks(fontsize=10)

plt.title('Web Development')

plt.legend(['PHP','HTML','JavaScript','AngularJS','BootStrap','CSS'],loc=[1.0,0.5])

plt.show()
#mysql,mongodb,nosql,postgresql,cassandra

query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date)>= 2009 and extract(year from creation_date) < 2019 

        and (tags like '%mysql%' or tags like '%nosql%' or tags like '%mongodb%' 

        or tags like '%postgresql%' or tags like '%cassandra%')

        group by year

        order by year

        """



DataBase_Posts = stackoverflow.query_to_pandas(query)

DataBase_Posts['posts']= DataBase_Posts['posts']*100/PostsCount.posts

DataBase_Posts
DataBase_Posts.describe()
pd.to_numeric(DataBase_Posts['year'])
DataBaseYear=DataBase_Posts['year'].values.reshape(-1,1)

# print (DataBaseYear)

DataBasePosts=DataBase_Posts['posts'].values.reshape(-1,1)

# print (DataBasePosts)
XDataBase_train, XDataBase_test, yDataBase_train, yDataBase_test = train_test_split(DataBaseYear,DataBasePosts,test_size=0.2,shuffle=False)

# print(XDataBase_train)

# print(XDataBase_test)

# print(yDataBase_train)

# print(yDataBase_test)
DataBaseReg=LinearRegression()

DataBaseReg.fit(XDataBase_train,yDataBase_train)

DataBasePredictions = DataBaseReg.predict(XDataBase_test)

print('Predicted Values:\n',DataBasePredictions)
plt.scatter(XDataBase_train,yDataBase_train, color = "black")

plt.scatter(XDataBase_test, yDataBase_test, color = "green")

plt.plot(XDataBase_test, DataBasePredictions, color = "red")

plt.gca().legend(('Y-Predicted', 'Y-Train','Y-Test'))

plt.title('Database Technologies')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
plt.scatter(XDataBase_test, yDataBase_test, color = "green")

plt.plot(XDataBase_test, DataBasePredictions, color = "red")

plt.gca().legend(('Y-Train','Y-Test'))

plt.title('Database Technologies')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
DataBaseReg.score(XDataBase_test, yDataBase_test)
print('Mean Squared Error:', metrics.mean_squared_error(yDataBase_test, DataBasePredictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yDataBase_test, DataBasePredictions)))
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%mysql%'

        group by year

        order by year

        """



MySQLPosts = stackoverflow.query_to_pandas(query)

MySQLPosts['posts']= MySQLPosts['posts']*100/PostsCount.posts

pd.to_numeric(MySQLPosts['year'])

MySQLPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%mongodb%'

        group by year

        order by year

        """



MongoDBPosts = stackoverflow.query_to_pandas(query)

MongoDBPosts['posts']= MongoDBPosts['posts']*100/PostsCount.posts

pd.to_numeric(MongoDBPosts['year'])

MongoDBPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%nosql%'

        group by year

        order by year

        """



NoSQLPosts = stackoverflow.query_to_pandas(query)

NoSQLPosts['posts']= NoSQLPosts['posts']*100/PostsCount.posts

pd.to_numeric(NoSQLPosts['year'])

NoSQLPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%postgresql%'

        group by year

        order by year

        """



PostgreSQLPosts = stackoverflow.query_to_pandas(query)

PostgreSQLPosts['posts']= PostgreSQLPosts['posts']*100/PostsCount.posts

pd.to_numeric(PostgreSQLPosts['year'])

PostgreSQLPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 

        and tags like '%cassandra%'

        group by year

        order by year

        """



CassandraPosts = stackoverflow.query_to_pandas(query)

CassandraPosts['posts']= CassandraPosts['posts']*100/PostsCount.posts

pd.to_numeric(CassandraPosts['year'])

CassandraPosts
DataBase= pd.merge(MySQLPosts, NoSQLPosts, how='inner', on = 'year')

DataBase=DataBase.set_index('year')

DataBase= pd.merge(DataBase, MongoDBPosts, how='inner', on = 'year')

DataBase=DataBase.set_index('year')

DataBase= pd.merge(DataBase, PostgreSQLPosts, how='inner', on = 'year')

DataBase=DataBase.set_index('year')

DataBase= pd.merge(DataBase, CassandraPosts, how='inner', on = 'year')

DataBase=DataBase.set_index('year')





DataBase.plot(kind='line')

plt.xlabel('Year', fontsize=15)

plt.ylabel('Posts %', fontsize=15)

y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]



plt.xticks(y_pos,fontsize=10)

plt.yticks(fontsize=10)

plt.title('Database Technologies')

plt.legend(['MySQL','NoSQL','MongoDB','PostgreSQL','Cassandra'],loc=[1.0,0.5])

plt.show()
#hadoop,hive,spark,hbase,kafka

query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date)>= 2009 and extract(year from creation_date) < 2019 and (tags like '%hadoop%' or 

        tags like '%spark%' or tags like '%hive%' or tags like '%hbase%' or tags like '%kafka%')

        group by year

        order by year

        """



BigData_Posts = stackoverflow.query_to_pandas(query)

BigData_Posts['posts']= BigData_Posts['posts']*100/PostsCount.posts

BigData_Posts
BigData_Posts.describe()
pd.to_numeric(BigData_Posts['year'])
BigDataYear=BigData_Posts['year'].values.reshape(-1,1)

# print (BigDataYear)

BigDataPosts=BigData_Posts['posts'].values.reshape(-1,1)

# print (BigDataPosts)
XBigData_train, XBigData_test, yBigData_train, yBigData_test = train_test_split(BigDataYear,BigDataPosts,test_size=0.2,shuffle=False)

# print(XBigData_train)

# print(XBigData_test)

# print(yBigData_train)

# print(yBigData_test)
BigDataReg=LinearRegression()

BigDataReg.fit(XBigData_train,yBigData_train)

BigDataPredictions = BigDataReg.predict(XBigData_test)

print('Predicted Values:\n',BigDataPredictions)
plt.scatter(XBigData_train,yBigData_train, color = "black")

plt.scatter(XBigData_test, yBigData_test, color = "green")

plt.plot(XBigData_test, BigDataPredictions, color = "red")

plt.gca().legend(('Y-Predicted', 'Y-Train','Y-Test'))

plt.title('Big Data')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
plt.scatter(XBigData_test, yBigData_test, color = "green")

plt.plot(XBigData_test, BigDataPredictions, color = "red")

plt.gca().legend(('Y-Train','Y-Test'))

plt.title('Big Data')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
BigDataReg.score(XBigData_test, yBigData_test)
print('Mean Squared Error:', metrics.mean_squared_error(yBigData_test, BigDataPredictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yBigData_test, BigDataPredictions)))
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%hadoop%'

        group by year

        order by year

        """



HadoopPosts = stackoverflow.query_to_pandas(query)

HadoopPosts['posts']= HadoopPosts['posts']*100/PostsCount.posts

pd.to_numeric(HadoopPosts['year'])

HadoopPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%hive%'

        group by year

        order by year

        """



HivePosts = stackoverflow.query_to_pandas(query)

HivePosts['posts']= HivePosts['posts']*100/PostsCount.posts

pd.to_numeric(HivePosts['year'])

HivePosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%spark%'

        group by year

        order by year

        """



SparkPosts = stackoverflow.query_to_pandas(query)

SparkPosts['posts']= SparkPosts['posts']*100/PostsCount.posts

pd.to_numeric(SparkPosts['year'])

SparkPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%hbase%'

        group by year

        order by year

        """



HBasePosts = stackoverflow.query_to_pandas(query)

HBasePosts['posts']= HBasePosts['posts']*100/PostsCount.posts

pd.to_numeric(HBasePosts['year'])

HBasePosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%kafka%'

        group by year

        order by year

        """



KafkaPosts = stackoverflow.query_to_pandas(query)

KafkaPosts['posts']= KafkaPosts['posts']*100/PostsCount.posts

pd.to_numeric(KafkaPosts['year'])

KafkaPosts
df = pd.DataFrame({"year":[2009,2010],"posts":[0,0]})

KafkaPosts = KafkaPosts.append(df, ignore_index = True)

KafkaPosts.sort_values("year", axis = 0, ascending = True, inplace = True)

KafkaPosts = KafkaPosts.reset_index(drop=True)

KafkaPosts
BigData= pd.merge(HadoopPosts, SparkPosts, how='inner', on = 'year')

BigData=BigData.set_index('year')

BigData= pd.merge(BigData, HivePosts, how='inner', on = 'year')

BigData=BigData.set_index('year')

BigData= pd.merge(BigData, HBasePosts, how='inner', on = 'year')

BigData=BigData.set_index('year')

BigData= pd.merge(BigData, KafkaPosts, how='inner', on = 'year')

BigData=BigData.set_index('year')



BigData.plot(kind='line')

plt.xlabel('Year', fontsize=15)

plt.ylabel('Posts %', fontsize=15)

y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]



plt.xticks(y_pos,fontsize=10)

plt.yticks(fontsize=10)

plt.title('Big Data')

plt.legend(['Hadoop','Spark','Hive','HBase','Kafka'],loc=[1.0,0.5])

plt.show()
#pandas,matplotlib,regression,svm,kaggle

query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date)>= 2009 and extract(year from creation_date) < 2019 

        and (tags like '%pandas%' or tags like '%matplotlib%'

        or tags like '%regression%' or tags like '%svm%' or tags like '%kaggle%')

        group by year

        order by year

        """



DataScience_Posts = stackoverflow.query_to_pandas(query)

DataScience_Posts['posts']= DataScience_Posts['posts']*100/PostsCount.posts

DataScience_Posts
DataScience_Posts.describe()
pd.to_numeric(DataScience_Posts['year'])
DataScienceYear=DataScience_Posts['year'].values.reshape(-1,1)

# print (DataScienceYear)

DataSciencePosts=DataScience_Posts['posts'].values.reshape(-1,1)

# print (DataSciencePosts)
XDataScience_train, XDataScience_test, yDataScience_train, yDataScience_test = train_test_split(DataScienceYear,DataSciencePosts,test_size=0.2,shuffle=False)

# print(XDataScience_train)

# print(XDataScience_test)

# print(yDataScience_train)

# print(yDataScience_test)
DataScienceReg=LinearRegression()

DataScienceReg.fit(XDataScience_train,yDataScience_train)

DataSciencePredictions = DataScienceReg.predict(XDataScience_test)

print('Predicted Values:\n',DataSciencePredictions)
plt.scatter(XDataScience_train,yDataScience_train, color = "black")

plt.scatter(XDataScience_test, yDataScience_test, color = "green")

plt.plot(XDataScience_test, DataSciencePredictions, color = "red")

plt.gca().legend(('Y-Predicted', 'Y-Train','Y-Test'))

plt.title('Data Science')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
plt.scatter(XDataScience_test, yDataScience_test, color = "green")

plt.plot(XDataScience_test, DataSciencePredictions, color = "red")

plt.gca().legend(('Y-Train','Y-Test'))

plt.title('Data Science')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
DataScienceReg.score(XDataScience_test,yDataScience_test)
print('Mean Squared Error:', metrics.mean_squared_error(yDataScience_test, DataSciencePredictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yDataScience_test, DataSciencePredictions)))
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%pandas%'

        group by year

        order by year

        """



PandasPosts = stackoverflow.query_to_pandas(query)

PandasPosts['posts']= PandasPosts['posts']*100/PostsCount.posts

pd.to_numeric(PandasPosts['year'])

PandasPosts
df = pd.DataFrame({"year":[2009],"posts":[0]})

PandasPosts = PandasPosts.append(df, ignore_index = True)

PandasPosts.sort_values("year", axis = 0, ascending = True, inplace = True)

PandasPosts = PandasPosts.reset_index(drop=True)

PandasPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%matplotlib%'

        group by year

        order by year

        """



MatplotlibPosts = stackoverflow.query_to_pandas(query)

MatplotlibPosts['posts']= MatplotlibPosts['posts']*100/PostsCount.posts

pd.to_numeric(MatplotlibPosts['year'])

MatplotlibPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 

        and tags like '%regression%'

        group by year

        order by year

        """



RegressionPosts = stackoverflow.query_to_pandas(query)

RegressionPosts['posts']= RegressionPosts['posts']*100/PostsCount.posts

pd.to_numeric(RegressionPosts['year'])

RegressionPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 

        and tags like '%svm%'

        group by year

        order by year

        """



SVMPosts = stackoverflow.query_to_pandas(query)

SVMPosts['posts']= SVMPosts['posts']*100/PostsCount.posts

pd.to_numeric(SVMPosts['year'])

SVMPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 

        and tags like '%kaggle%'

        group by year

        order by year

        """



KagglePosts = stackoverflow.query_to_pandas(query)

KagglePosts['posts']= KagglePosts['posts']*100/PostsCount.posts

pd.to_numeric(KagglePosts['year'])

KagglePosts
df = pd.DataFrame({"year":[2009,2010],"posts":[0,0]})

KagglePosts = KagglePosts.append(df, ignore_index = True)

KagglePosts.sort_values("year", axis = 0, ascending = True, inplace = True)

KagglePosts = KagglePosts.reset_index(drop=True)

KagglePosts
DataScience= pd.merge(PandasPosts, MatplotlibPosts, how='inner', on = 'year')

DataScience=DataScience.set_index('year')

DataScience= pd.merge(DataScience, RegressionPosts, how='inner', on = 'year')

DataScience=DataScience.set_index('year')

DataScience= pd.merge(DataScience, SVMPosts, how='inner', on = 'year')

DataScience=DataScience.set_index('year')

DataScience= pd.merge(DataScience, KagglePosts, how='inner', on = 'year')

DataScience=DataScience.set_index('year')



DataScience.plot(kind='line')

plt.xlabel('Year', fontsize=15)

plt.ylabel('Posts %', fontsize=15)

y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]



plt.xticks(y_pos,fontsize=10)

plt.yticks(fontsize=10)

plt.title('Data Science')

plt.legend(['Pandas','Matplotlib','Regression','SVM','Kaggle'],loc=[1.0,0.5])

plt.show()
#C++,ruby,java,c#,python

query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >= 2009 and extract(year from creation_date) < 2019 

        and (tags like '%c++%' or tags like '%python%' or tags like '%ruby%' 

        or tags like '%c#%' or (tags like '%java%' and tags not like '%javascript%'))

        group by year

        order by year

        """



ProgLang_Posts = stackoverflow.query_to_pandas(query)

ProgLang_Posts['posts']=ProgLang_Posts['posts']*100/PostsCount.posts

ProgLang_Posts
ProgLang_Posts.describe()
pd.to_numeric(ProgLang_Posts['year'])
ProgLangYear=ProgLang_Posts['year'].values.reshape(-1,1)

# print (ProgLangYear)

ProgLangPosts=ProgLang_Posts['posts'].values.reshape(-1,1)

# print (ProgLangPosts)
XProgLang_train, XProgLang_test, yProgLang_train, yProgLang_test = train_test_split(ProgLangYear,ProgLangPosts,test_size=0.2,shuffle=False)

# print(XProgLang_train)

# print(XProgLang_test)

# print(yProgLang_train)

# print(yProgLang_test)
ProgLangReg=LinearRegression()

ProgLangReg.fit(XProgLang_train,yProgLang_train)

ProgLangPredictions = ProgLangReg.predict(XProgLang_test)

print('Predicted Values:\n',ProgLangPredictions)
plt.scatter(XProgLang_train,yProgLang_train, color = "black")

plt.scatter(XProgLang_test, yProgLang_test, color = "green")

plt.plot(XProgLang_test, ProgLangPredictions, color = "red")

plt.gca().legend(('Y-Predicted', 'Y-Train','Y-Test'))

plt.title('Programming Languages')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
plt.scatter(XProgLang_test, yProgLang_test, color = "green")

plt.plot(XProgLang_test, ProgLangPredictions, color = "red")

plt.gca().legend(('Y-Train','Y-Test'))

plt.title('Programming Languages')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
ProgLangReg.score(XProgLang_test, yProgLang_test)
print('Mean Squared Error:', metrics.mean_squared_error(yProgLang_test, ProgLangPredictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yProgLang_test, ProgLangPredictions)))
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%c++%'

        group by year

        order by year

        """



CplusPosts = stackoverflow.query_to_pandas(query)

CplusPosts['posts']= CplusPosts['posts']*100/PostsCount.posts

pd.to_numeric(CplusPosts['year'])

CplusPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%ruby%'

        group by year

        order by year

        """



RubyPosts = stackoverflow.query_to_pandas(query)

RubyPosts['posts']= RubyPosts['posts']*100/PostsCount.posts

pd.to_numeric(RubyPosts['year'])

RubyPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%java%' and tags not like '%javascript%'

        group by year

        order by year

        """



JavaPosts = stackoverflow.query_to_pandas(query)

JavaPosts['posts']= JavaPosts['posts']*100/PostsCount.posts

pd.to_numeric(JavaPosts['year'])

JavaPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%c#%'

        group by year

        order by year

        """



CHashPosts = stackoverflow.query_to_pandas(query)

CHashPosts['posts']= CHashPosts['posts']*100/PostsCount.posts

pd.to_numeric(CHashPosts['year'])

CHashPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%python%'

        group by year

        order by year

        """



PythonPosts = stackoverflow.query_to_pandas(query)

PythonPosts['posts']= PythonPosts['posts']*100/PostsCount.posts

pd.to_numeric(PythonPosts['year'])

PythonPosts
ProgLang= pd.merge(RubyPosts, CplusPosts, how='inner', on = 'year')

ProgLang =ProgLang.set_index('year')

ProgLang= pd.merge(ProgLang, PythonPosts, how='inner', on = 'year')

ProgLang =ProgLang.set_index('year')

ProgLang=pd.merge(ProgLang,CHashPosts,how='inner',on='year')

ProgLang = ProgLang.set_index('year')

ProgLang=pd.merge(ProgLang,JavaPosts,how='inner',on='year')

ProgLang = ProgLang.set_index('year')



ProgLang.plot(kind='line')

plt.xlabel('Year', fontsize=15)

plt.ylabel('Posts %', fontsize=15)

y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]



plt.xticks(y_pos,fontsize=10)

plt.yticks(fontsize=10)

plt.title('Programming Languages')

plt.legend(['Ruby','C++','Python','C#','Java'],loc=[1.0,0.5])

plt.show()
PastTrends= pd.merge(WebDev_Posts, DataBase_Posts, how='inner', on = 'year')

PastTrends =PastTrends.set_index('year')

PastTrends= pd.merge(PastTrends, BigData_Posts, how='inner', on = 'year')

PastTrends =PastTrends.set_index('year')

PastTrends=pd.merge(PastTrends,DataScience_Posts,how='inner',on='year')

PastTrends = PastTrends.set_index('year')

PastTrends=pd.merge(PastTrends,ProgLang_Posts,how='inner',on='year')

PastTrends = PastTrends.set_index('year')



PastTrends.plot(kind='line')

plt.xlabel('Year', fontsize=15)

plt.ylabel('Posts %', fontsize=15)

y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]



plt.xticks(y_pos,fontsize=10)

plt.yticks(fontsize=10)

plt.title('Past Trends')

plt.legend(['Web Development','DataBase Technologies','Big Data','Data Science','Programming Languages'],

           loc=[1.0,0.5])

plt.show()
def trends(dfall, labels=None, Year = 2019, title="Trends in Technologies in ", **kwargs):



    plt.figure(figsize=(20,10))

   

    predict = []

    for df in dfall :

        year=df['year'].values.reshape(-1,1)

        posts=df['posts'].values.reshape(-1,1)

        reg=LinearRegression()

        X_train = year

        Y_train = posts

        X_test = [[Year]]

        reg.fit(X_train,Y_train)

        predictions = reg.predict(X_test)

        predict.append(predictions)



    trend = pd.DataFrame(columns = ['Technology','Posts %'])

    trend['Technology'] = labels

    trend['Posts %'] = predict

    

    x_pos = np.arange(len(trend['Technology']))

    plt.bar(x_pos,trend['Posts %'])

    plt.xticks(x_pos, trend['Technology'],fontsize=15)

    plt.yticks(fontsize=15)

    plt.xlabel('Technologies',fontsize=20)

    plt.ylabel('Posts Percentage',fontsize=20)

    plt.title(title+str(Year),fontsize=30)

    plt.show()
trends([WebDev_Posts, DataBase_Posts, BigData_Posts, DataScience_Posts, ProgLang_Posts],

       ["Web Development",'DataBase Technologies','Big Data','Data Science','Programming Languages'])
def PastTrends(dfall, labels = None, title="Past Trends", **kwargs):



    query1 = "select EXTRACT(year FROM creation_date) AS year, sum(id) as posts from `bigquery-public-data.stackoverflow.posts_questions` where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%"

    query3 ="%' group by year order by year"

    df = []

    

    if labels==None:

        labels = dfall

        

    l = len(dfall)

    for i in range(l):

        query2 = dfall[i]

        query = query1+query2+query3

        Posts = stackoverflow.query_to_pandas(query)

        Posts['posts']= Posts['posts']*100/PostsCount.posts

        pd.to_numeric(Posts['year'])

        df.append(Posts)

    

    trend = pd.merge(df[0], df[1], how='inner', on = 'year')

    trend = trend.set_index('year')

    if(l>2):

        for i in range(2,l):

            trend = pd.merge(trend, df[i], how='inner', on = 'year')

            trend = trend.set_index('year')

            

    trend.plot(kind='line')

    plt.xlabel('Year', fontsize=15)

    plt.ylabel('Posts %', fontsize=15)

    y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

    plt.xticks(y_pos,fontsize=10)

    plt.yticks(fontsize=10)

    plt.title(title)

    plt.legend(labels, loc=[1.0,0.5])

    plt.show()
PastTrends(["android","javascript","cassandra"])
def FutureTrends(dfall, Year = 2019, labels = None, title="Trends in Technologies in ", **kwargs):



    plt.figure(figsize=(20,10))

    

    query1 = "select EXTRACT(year FROM creation_date) AS year, sum(id) as posts from `bigquery-public-data.stackoverflow.posts_questions` where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%"

    query3 ="%' group by year order by year"

    df = []

    l = len(dfall)

    

    if (labels==None):

        labels = dfall

        

    for i in range(l):

        query2 = dfall[i]

        query = query1+query2+query3

        Posts = stackoverflow.query_to_pandas(query)

        Posts['posts']= Posts['posts']*100/PostsCount.posts

        pd.to_numeric(Posts['year'])

        df.append(Posts)

        

    predict = []

    for d in df:

        year=d['year'].values.reshape(-1,1)

        posts=d['posts'].values.reshape(-1,1)

        reg=LinearRegression()

        X_train = year

        Y_train = posts

        X_test = [[Year]]

        reg.fit(X_train,Y_train)

        predictions = reg.predict(X_test)

        predict.append(predictions)

    #print(predict)

    

    trend = pd.DataFrame(columns = ['Technology','Posts %'])

    trend['Technology'] = labels

    trend['Posts %'] = predict

    

    x_pos = np.arange(len(trend['Technology']))

    plt.bar(x_pos,trend['Posts %'])

    plt.xticks(x_pos, trend['Technology'],fontsize=15)

    plt.yticks(fontsize=15)

    plt.xlabel('Technologies',fontsize=20)

    plt.ylabel('Posts Percentage',fontsize=20)

    plt.title(title+str(Year),fontsize=30)

    plt.show()
FutureTrends(["spark","hive","python"])
FutureTrends(["jquery","javascript","html"],2020, ['JQuery','JavaScript','HTML'])