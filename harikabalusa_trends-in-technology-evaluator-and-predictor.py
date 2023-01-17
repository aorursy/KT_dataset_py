%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import sklearn

import pandas_profiling



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

import scipy

import seaborn as sns

sns.set()



import bq_helper

stackoverflow = bq_helper.BigQueryHelper("bigquery-public-data","stackoverflow")
stackoverflow.list_tables()
stackoverflow.head("posts_questions",num_rows=20)
#query="""SELECT * FROM `bigquery-public-data.stackoverflow.posts_questions` WHERE EXTRACT (YEAR FROM creation_date)=2019"""

#data = stackoverflow.query_to_pandas(query)



#data.isnull.sum()

#data['favorite_count'].fillna(0,inplace=True)

#data['accepted_answer_id'].fillna(0,inplace=True)

#data['last_editor_user_id'].fillna(0,inplace=True)

#data.head()
stackoverflow.table_schema("posts_questions")
#data=pd.DataFrame(data)

#pandas_profiling.ProfileReport(data)
queryx = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >= 2009 and extract(year from creation_date) < 2019

        group by year

        order by year

        """



PostsCount= stackoverflow.query_to_pandas(queryx)

print(PostsCount)
PostsCount.head()
pd.to_numeric(PostsCount['year'])
type(PostsCount['year'][0])
PostsCount
year=PostsCount['year'].values.reshape(-1,1)

print (year)

posts=PostsCount['posts'].values.reshape(-1,1)

print (posts)
reg=LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(year,posts,

                                                    test_size=0.2,shuffle=False)

print(X_test)

reg.fit(X_train,y_train)

predictions = reg.predict(X_test)
print(predictions)
plt.scatter(X_train,y_train)

plt.scatter(X_test, y_test, color = "green")

plt.plot(X_test, predictions, color = "red")

plt.gca().legend(('Y-Train','Y-Test', 'Y-predicted'))

plt.title('Y-test and Y-predicted')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
plt.scatter(X_test, y_test, color = "green")

plt.plot(X_test, predictions, color = "red")

plt.gca().legend(('Y-Train','Y-Test', 'Y-predicted'))

plt.title('Y-test and Y-predicted')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
reg.score(X_test,y_test)
#angularjs,bootstrap,php,html,javascript,css

query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and (tags like '%bootstrap%' or 

        tags like '%angularjs%' or tags like '%php%' or tags like '%html%' or tags like '%javascript%' or tags like '%css%')

        group by year

        order by year

        """



WebDevPosts = stackoverflow.query_to_pandas(query)

WebDevPosts['posts']= WebDevPosts['posts']*100/PostsCount.posts

WebDevPosts
pd.to_numeric(WebDevPosts['year'])
WebDevYear=WebDevPosts['year'].values.reshape(-1,1)

print (WebDevYear)

WebDevPosts=WebDevPosts['posts'].values.reshape(-1,1)

print (WebDevPosts)
XWebDev_train, XWebDev_test, yWebDev_train, yWebDev_test = train_test_split(WebDevYear,WebDevPosts,

                                                    test_size=0.2,shuffle=False)

print(XWebDev_train)

print(XWebDev_test)

print(yWebDev_train)

print(yWebDev_test)
WebDevReg=LinearRegression()

WebDevReg.fit(XWebDev_train,yWebDev_train)

WebDevPredictions = WebDevReg.predict(XWebDev_test)

print(WebDevPredictions)
plt.scatter(XWebDev_train,yWebDev_train)

plt.scatter(XWebDev_test, yWebDev_test, color = "green")

plt.plot(XWebDev_test, WebDevPredictions, color = "red")

plt.gca().legend(('Y-Train','Y-Test', 'Y-predicted'))

plt.title('Web Development')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
plt.scatter(XWebDev_test, yWebDev_test, color = "green")

plt.plot(XWebDev_test, WebDevPredictions, color = "red")

plt.gca().legend(('Y-Train','Y-Test', 'Y-predicted'))

plt.title('Web Development')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
WebDevReg.score(XWebDev_test,yWebDev_test)
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%angularjs%'

        group by year

        order by year

        """



AngularJSPosts = stackoverflow.query_to_pandas(query)

AngularJSPosts['posts']= AngularJSPosts['posts']*100/PostsCount.posts

AngularJSPosts
# pd.to_numeric(AngularJSPosts['year'])

# AngularJSYear=AngularJSPosts['year'].values.reshape(-1,1)

# print (AngularJSYear)

# AngularJSPosts=AngularJSPosts['posts'].values.reshape(-1,1)

# print (AngularJSPosts)



# XAngularJS_train, XAngularJS_test, yAngularJS_train, yAngularJS_test = train_test_split(AngularJSYear,AngularJSPosts,

#                                                     test_size=0.2,shuffle=False)

# AngularJSReg=LinearRegression()

# AngularJSReg.fit(XAngularJS_train,yAngularJS_train)

# AngularJSPredictions = AngularJSReg.predict(XAngularJS_test)

# print("Predicted Values")

# print(AngularJSPredictions)



# plt.scatter(XAngularJS_train,yAngularJS_train)

# plt.xlabel('Year')

# plt.ylabel('Posts')

# plt.show()



# AngularJSReg.score(XAngularJS_test,yAngularJS_test)
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%bootstrap%'

        group by year

        order by year

        """



BootstrapPosts = stackoverflow.query_to_pandas(query)

BootstrapPosts['posts']= BootstrapPosts['posts']*100/PostsCount.posts

BootstrapPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%php%'

        group by year

        order by year

        """



PHPPosts = stackoverflow.query_to_pandas(query)

PHPPosts['posts']= PHPPosts['posts']*100/PostsCount.posts

PHPPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%html%'

        group by year

        order by year

        """



htmlPosts = stackoverflow.query_to_pandas(query)

htmlPosts['posts']= htmlPosts['posts']*100/PostsCount.posts

htmlPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%javascript%'

        group by year

        order by year

        """



JavaScriptPosts = stackoverflow.query_to_pandas(query)

JavaScriptPosts['posts']= JavaScriptPosts['posts']*100/PostsCount.posts

JavaScriptPosts
x_pos = np.arange(len(JavaScriptPosts['year']))

plt.bar(x_pos,JavaScriptPosts['posts'])

plt.xticks(x_pos, JavaScriptPosts['year'],fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Year',fontsize=10)

plt.ylabel('% Posts Count',fontsize=10)

plt.title('JavaScript',fontsize=20)

plt.show()
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%css%'

        group by year

        order by year

        """



CSSPosts = stackoverflow.query_to_pandas(query)

CSSPosts['posts']= CSSPosts['posts']*100/PostsCount.posts

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

plt.title('WebDev')

plt.legend(['PHP','HTML','JavaScript','AngularJS','BootStrap','CSS'],loc=[1.0,0.5])

plt.show()
#mysql,nosql,maybe mongodb

query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date)>= 2009 and extract(year from creation_date) < 2019 and (tags like '%mysql%' or 

        tags like '%nosql%')

        group by year

        order by year

        """



DataBasePosts = stackoverflow.query_to_pandas(query)

DataBasePosts['posts']= DataBasePosts['posts']*100/PostsCount.posts

DataBasePosts
pd.to_numeric(DataBasePosts['year'])

DataBaseYear=DataBasePosts['year'].values.reshape(-1,1)

print (DataBaseYear)

DataBasePosts=DataBasePosts['posts'].values.reshape(-1,1)

print (DataBasePosts)





XDataBase_train, XDataBase_test, yDataBase_train, yDataBase_test = train_test_split(DataBaseYear,DataBasePosts,

                                                    test_size=0.2,shuffle=False)



print(XDataBase_train)

print(XDataBase_test)

print(yDataBase_train)

print(yDataBase_test)
DataBaseReg=LinearRegression()

DataBaseReg.fit(XDataBase_train,yDataBase_train)

DataBasePredictions = DataBaseReg.predict(XDataBase_test)

print("Predicted Values")

print(DataBasePredictions)







plt.scatter(XDataBase_train,yDataBase_train)

plt.scatter(XDataBase_test, yDataBase_test, color = "green")

plt.plot(XDataBase_test, DataBasePredictions, color = "red")

plt.gca().legend(('Y-Train','Y-Test', 'Y-predicted'))

plt.title('DataBase')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
plt.scatter(XDataBase_test, yDataBase_test, color = "green")

plt.plot(XDataBase_test, DataBasePredictions, color = "red")

plt.gca().legend(('Y-Train','Y-Test', 'Y-predicted'))

plt.title('DataBase')

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%mysql%'

        group by year

        order by year

        """



MySQLPosts = stackoverflow.query_to_pandas(query)



MySQLPosts['posts']= MySQLPosts['posts']*100/PostsCount.posts

MySQLPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%mongodb%'

        group by year

        order by year

        """



MongoDBPosts = stackoverflow.query_to_pandas(query)



MongoDBPosts['posts']= MongoDBPosts['posts']*100/PostsCount.posts

MongoDBPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%nosql%'

        group by year

        order by year

        """



NoSQLPosts = stackoverflow.query_to_pandas(query)



NoSQLPosts['posts']= NoSQLPosts['posts']*100/PostsCount.posts

NoSQLPosts
DataBase= pd.merge(MySQLPosts, NoSQLPosts, how='inner', on = 'year')

DataBase=DataBase.set_index('year')

DataBase= pd.merge(DataBase, MongoDBPosts, how='inner', on = 'year')

DataBase=DataBase.set_index('year')



DataBase.plot(kind='line')

plt.xlabel('Year', fontsize=15)

plt.ylabel('Posts %', fontsize=15)

y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]



plt.xticks(y_pos,fontsize=10)

plt.yticks(fontsize=10)

plt.title('DataBase')

plt.legend(['MySQL','NoSQL','MongoDB'],loc=[1.0,0.5])

plt.show()
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date)>= 2009 and extract(year from creation_date) < 2019 and (tags like '%hadoop%' or 

        tags like '%spark%' or tags like '%hive%')

        group by year

        order by year

        """



BigDataPosts = stackoverflow.query_to_pandas(query)

BigDataPosts['posts']= BigDataPosts['posts']*100/PostsCount.posts

BigDataPosts
pd.to_numeric(BigDataPosts['year'])
BigDataPosts


BigDataYear=BigDataPosts['year'].values.reshape(-1,1)

print (BigDataYear)

BigDataPosts=BigDataPosts['posts'].values.reshape(-1,1)

print (BigDataPosts)



XBigData_train, XBigData_test, yBigData_train, yBigData_test = train_test_split(BigDataYear,BigDataPosts,

                                                    test_size=0.2,shuffle=False)

BigDataReg=LinearRegression()

BigDataReg.fit(XBigData_train,yBigData_train)

BigDataPredictions = BigDataReg.predict(XBigData_test)

print("Predicted Values")

print(BigDataPredictions)



plt.scatter(XBigData_train,yBigData_train)

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()



print("Score")

BigDataReg.score(XBigData_test,yBigData_test)
print(BigDataPosts)
print(BigDataPosts.index)
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%hadoop%'

        group by year

        order by year

        """



HadoopPosts = stackoverflow.query_to_pandas(query)



HadoopPosts['posts']= HadoopPosts['posts']*100/PostsCount.posts

HadoopPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%hive%'

        group by year

        order by year

        """



HivePosts = stackoverflow.query_to_pandas(query)



HivePosts['posts']= HivePosts['posts']*100/PostsCount.posts

HivePosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%spark%'

        group by year

        order by year

        """



SparkPosts = stackoverflow.query_to_pandas(query)



SparkPosts['posts']= SparkPosts['posts']*100/PostsCount.posts

SparkPosts
BigData= pd.merge(HadoopPosts, SparkPosts, how='inner', on = 'year')

BigData=BigData.set_index('year')

BigData= pd.merge(BigData, HivePosts, how='inner', on = 'year')

BigData=BigData.set_index('year')

BigData.plot(kind='line')

plt.xlabel('Year', fontsize=15)

plt.ylabel('Posts %', fontsize=15)

y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

#y_pos = np.arange(5)

plt.xticks(y_pos,fontsize=10)

plt.yticks(fontsize=10)

plt.title('BigData')

plt.legend(['Hadoop','Spark','Hive'],loc=[1.0,0.5])

plt.show()
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

       where extract(year from creation_date)>= 2009 and extract(year from creation_date) < 2019 and (tags like '%python%' or 

        tags like '%pandas%' or tags like '%matplotlib%')

        group by year

        order by year

        """



DataSciencePosts = stackoverflow.query_to_pandas(query)

DataSciencePosts['posts']= DataSciencePosts['posts']*100/PostsCount.posts

DataSciencePosts

pd.to_numeric(DataSciencePosts['year'])

DataScienceYear=DataSciencePosts['year'].values.reshape(-1,1)

print (DataScienceYear)

DataSciencePosts=DataSciencePosts['posts'].values.reshape(-1,1)

print (DataSciencePosts)



XDataScience_train, XDataScience_test, yDataScience_train, yDataScience_test = train_test_split(DataScienceYear,DataSciencePosts,

                                                    test_size=0.2,shuffle=False)

DataScienceReg=LinearRegression()

DataScienceReg.fit(XDataScience_train,yDataScience_train)

DataSciencePredictions = DataScienceReg.predict(XDataScience_test)

print("Predicted Values")

print(DataSciencePredictions)



plt.scatter(XDataScience_train,yDataScience_train)

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()



print("Score")

DataScienceReg.score(XDataScience_test,yDataScience_test)
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%python%'

        group by year

        order by year

        """



PythonPosts = stackoverflow.query_to_pandas(query)

PythonPosts['posts']= PythonPosts['posts']*100/PostsCount.posts

PythonPosts
x_pos = np.arange(len(PythonPosts['year']))

plt.bar(x_pos,PythonPosts['posts'])

plt.xticks(x_pos, PythonPosts['year'],fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Year',fontsize=10)

plt.ylabel('Posts Count',fontsize=10)

plt.title('Python 2019',fontsize=20)

plt.show()
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%pandas%'

        group by year

        order by year

        """



PandasPosts = stackoverflow.query_to_pandas(query)

PandasPosts['posts']= PandasPosts['posts']*100/PostsCount.posts

PandasPosts
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%matplotlib%'

        group by year

        order by year

        """



MatplotlibPosts = stackoverflow.query_to_pandas(query)

MatplotlibPosts['posts']= MatplotlibPosts['posts']*100/PostsCount.posts

MatplotlibPosts
DataScience= pd.merge(PandasPosts, MatplotlibPosts, how='inner', on = 'year')

DataScience=DataScience.set_index('year')

DataScience= pd.merge(DataScience, PythonPosts, how='inner', on = 'year')

DataScience=DataScience.set_index('year')

DataScience.plot(kind='line')

plt.xlabel('Year', fontsize=15)

plt.ylabel('Posts %', fontsize=15)

y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

#y_pos = np.arange(5)

plt.xticks(y_pos,fontsize=10)

plt.yticks(fontsize=10)

plt.title('DataScience')

plt.legend(['Pandas','Matplotlib','Python'],loc=[1.0,0.5])

plt.show()
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >= 2009 and extract(year from creation_date) < 2019 and (tags like '%c++%' or 

        tags like '%python%' or tags like '%ruby%' or tags like '%c#%' or (tags like '%java%' and tags not like '%javascript%'))

        group by year

        order by year

        """



ProgLangPosts = stackoverflow.query_to_pandas(query)

ProgLangPosts['posts']=ProgLangPosts['posts']*100/PostsCount.posts

ProgLangPosts
pd.to_numeric(ProgLangPosts['year'])

ProgLangYear=ProgLangPosts['year'].values.reshape(-1,1)

print (ProgLangYear)

ProgLangPosts=ProgLangPosts['posts'].values.reshape(-1,1)

print (ProgLangPosts)



XProgLang_train, XProgLang_test, yProgLang_train, yProgLang_test = train_test_split(ProgLangYear,ProgLangPosts,

                                                    test_size=0.2,shuffle=False)

ProgLangReg=LinearRegression()

ProgLangReg.fit(XProgLang_train,yProgLang_train)

ProgLangPredictions = ProgLangReg.predict(XProgLang_test)

print("Predicted Values")

print(ProgLangPredictions)



plt.scatter(XProgLang_train,yProgLang_train)

plt.xlabel('Year')

plt.ylabel('Posts')

plt.show()



print("Score")

ProgLangReg.score(XProgLang_test,yProgLang_test)
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%c++%'

        group by year

        order by year

        """



CplusPosts = stackoverflow.query_to_pandas(query)

CplusPosts['posts']= CplusPosts['posts']*100/PostsCount.posts

CplusPosts

#CplusPosts = CplusPosts.set_index('month')



y_pos = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

plt.plot(y_pos, CplusPosts['posts'])

plt.xticks(y_pos, CplusPosts['year'],fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Year', fontsize=15)

plt.ylabel('Posts', fontsize=15)

plt.title('Trends in C+')





plt.show()
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%ruby%'

        group by year

        order by year

        """



RubyPosts = stackoverflow.query_to_pandas(query)



RubyPosts['posts']= RubyPosts['posts']*100/PostsCount.posts

RubyPosts



query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%java%' and tags not like '%javascript%'

        group by year

        order by year

        """



JavaPosts = stackoverflow.query_to_pandas(query)



JavaPosts['posts']= JavaPosts['posts']*100/PostsCount.posts

JavaPosts

query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%c#%'

        group by year

        order by year

        """



CHashPosts = stackoverflow.query_to_pandas(query)

CHashPosts['posts']= CHashPosts['posts']*100/PostsCount.posts

CHashPosts
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

plt.ylabel('Posts', fontsize=15)

y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

#y_pos = np.arange(5)

plt.xticks(y_pos,fontsize=10)

plt.yticks(fontsize=10)

plt.title('Trends in Technology')

plt.legend(['Ruby','C++','Python','C#','Java'],loc=[1.0,0.5])

plt.show()

query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date)>= 2009 and extract(year from creation_date) < 2019 and (tags like '%hadoop%' or 

        tags like '%spark%' or tags like '%hive%')

        group by year

        order by year

        """



BigDataPosts = stackoverflow.query_to_pandas(query)

BigDataPosts['posts']= BigDataPosts['posts']*100/PostsCount.posts

BigDataPosts
pd.to_numeric(BigDataPosts['year'])
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%hive%'

        group by year

        order by year

        """



HivePosts = stackoverflow.query_to_pandas(query)



HivePosts['posts']= HivePosts['posts']*100/PostsCount.posts

HivePosts
pd.to_numeric(HivePosts['year'])
def trends(dfall, labels=None, title="Trends in Technologies in 2019",  H="x", **kwargs):



    plt.figure(figsize=(20,10))

   

    predict = []

    for df in dfall :

        #print('DataFrame')

        year=df['year'].values.reshape(-1,1)

        #print('Done')

        #print(df['year'],'\n')

        #print(year)

        posts=df['posts'].values.reshape(-1,1)

#         print('Done')

#         print(df,'\n',posts)

        reg=LinearRegression()

        X_train = year[:9]

        #print(X_train)

        Y_train = posts[:9]

        X_test = [[2018]]

        reg.fit(X_train,Y_train)

        predictions = reg.predict(X_test)

        predict.append(predictions)

    print(predict)

    trend = pd.DataFrame(columns = ['Technology','Posts %'])

    trend['Technology'] = labels

    trend['Posts %'] = predict

    

    x_pos = np.arange(len(trend['Technology']))

    plt.bar(x_pos,trend['Posts %'])

    plt.xticks(x_pos, trend['Technology'],fontsize=15)

    plt.yticks(fontsize=15)

    plt.xlabel('Technologies',fontsize=20)

    plt.ylabel('Posts Percentage',fontsize=20)

    plt.title(title,fontsize=30)

    plt.show()


# Then, just call :

trends([BigDataPosts, HivePosts],["Big Data","Hive"])
