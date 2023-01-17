#pip install mplcursors

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pandas_profiling



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import bq_helper

stackoverflow = bq_helper.BigQueryHelper("bigquery-public-data","stackoverflow")



# Any results you write to the current directory are saved as output.
stackoverflow.list_tables()
stackoverflow.head("posts_questions",num_rows=20)
#query="""SELECT * FROM `bigquery-public-data.stackoverflow.posts_questions` WHERE EXTRACT (YEAR FROM creation_date)=2019"""

#data = stackoverflow.query_to_pandas(query)



#data['favorite_count'].fillna(0,inplace=True)

#data['accepted_answer_id'].fillna(0,inplace=True)

#data['last_editor_user_id'].fillna(0,inplace=True)

#data.head()
#data=pd.DataFrame(data)

#pandas_profiling.ProfileReport(data)
stackoverflow.table_schema("posts_questions")
query1 = """select favorite_count,creation_date,tags

    from `bigquery-public-data.stackoverflow.posts_questions`

    where extract(year from creation_date) = 2019 

            

    """

answered_questions = stackoverflow.query_to_pandas(query1)

answered_questions.head(15)
answered_questions['favorite_count'].fillna(0,inplace=True)
query2 = """select favorite_count,creation_date

    from `bigquery-public-data.stackoverflow.posts_questions`

    where extract(year from creation_date) = 2019 and extract(month from creation_date)=01 and tags like '%javascript%' and tags like '%python%'

            

    """

answered_questions2 = stackoverflow.query_to_pandas(query2)

answered_questions2['favorite_count'].fillna(0,inplace=True)

answered_questions2.head(15)



query1 = """select EXTRACT(month FROM creation_date) AS month, sum(favorite_count) as favourite_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%javascript%'

        group by month

        order by month

        """



JavaScriptFav = stackoverflow.query_to_pandas(query1)

JavaScriptFav.head(10)
x_pos = np.arange(len(JavaScriptFav['month']))

plt.bar(x_pos,JavaScriptFav['favourite_count'])

plt.xticks(x_pos, JavaScriptFav['month'],fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Month',fontsize=10)

plt.ylabel('Favourite Count',fontsize=10)

plt.title('JavaScript 2019',fontsize=20)

plt.show()
query2 = """select EXTRACT(month FROM creation_date) AS month, sum(view_count) as view_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%javascript%'

        group by month

        order by month

        """



JavaScriptView = stackoverflow.query_to_pandas(query2)

JavaScriptView.head(10)



x_pos = np.arange(len(JavaScriptView['month']))

plt.bar(x_pos,JavaScriptView['view_count'])

plt.xticks(x_pos, JavaScriptView['month'],fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Month',fontsize=10)

plt.ylabel('View Count',fontsize=10)

plt.title('JavaScript 2019',fontsize=20)

plt.show()
query2 = """select EXTRACT(month FROM creation_date) AS month, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%javascript%'

        group by month

        order by month

        """



JavaScriptPosts = stackoverflow.query_to_pandas(query2)

JavaScriptPosts.head(10)
x_pos = np.arange(len(JavaScriptPosts['month']))

plt.bar(x_pos,JavaScriptPosts['posts'])

plt.xticks(x_pos, JavaScriptPosts['month'],fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Month',fontsize=10)

plt.ylabel('Posts Count',fontsize=10)

plt.title('JavaScript 2019',fontsize=20)

plt.show()
query2 = """select EXTRACT(month FROM creation_date) AS month, sum(answer_count) as answer_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%javascript%'

        group by month

        order by month

        """



JavaScriptAnswers = stackoverflow.query_to_pandas(query2)

JavaScriptAnswers.head(10)
x_pos = np.arange(len(JavaScriptAnswers['month']))

plt.bar(x_pos,JavaScriptAnswers['answer_count'])

plt.xticks(x_pos, JavaScriptAnswers['month'],fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Month',fontsize=10)

plt.ylabel('Answer Count',fontsize=10)

plt.title('JavaScript 2019',fontsize=20)

plt.show()
query2 = """select EXTRACT(month FROM creation_date) AS month, sum(comment_count) as comment_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%javascript%'

        group by month

        order by month

        """



JavaScriptComments = stackoverflow.query_to_pandas(query2)

JavaScriptComments.head(10)
x_pos = np.arange(len(JavaScriptComments['month']))

plt.bar(x_pos,JavaScriptComments['comment_count'])

plt.xticks(x_pos, JavaScriptComments['month'],fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Month',fontsize=10)

plt.ylabel('Comments Count',fontsize=10)

plt.title('JavaScript 2019',fontsize=20)

plt.show()
query = """select EXTRACT(month FROM creation_date) AS month, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%python%'

        group by month

        order by month

        """



PythonPosts = stackoverflow.query_to_pandas(query)





query2 = """select EXTRACT(month FROM creation_date) AS month, sum(view_count) as view_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%python%'

        group by month

        order by month

        """



PythonView = stackoverflow.query_to_pandas(query2)



query1 = """select EXTRACT(month FROM creation_date) AS month, sum(favorite_count) as favourite_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%python%'

        group by month

        order by month

        """



PythonFav = stackoverflow.query_to_pandas(query1)
query2 = """select EXTRACT(month FROM creation_date) AS month, sum(answer_count) as answer_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%python%'

        group by month

        order by month

        """



PythonAnswers = stackoverflow.query_to_pandas(query2)



query2 = """select EXTRACT(month FROM creation_date) AS month, sum(comment_count) as comment_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%python%'

        group by month

        order by month

        """



PythonComments = stackoverflow.query_to_pandas(query2)
Python= pd.merge(PythonComments, PythonAnswers, how='inner', on = 'month')

Python = Python.set_index('month')

Python=pd.merge(Python,PythonFav,how='inner',on='month')

Python = Python.set_index('month')

#z=Python.groupby('month')



Python.plot(kind="bar", stacked=True)

#plt.xticks(y_pos, Python['month'],fontsize=10)

plt.title("Python Stats")

plt.xlabel('Month',fontsize=10)
JavaScript= pd.merge(JavaScriptComments, JavaScriptAnswers, how='inner', on = 'month')

JavaScript = JavaScript.set_index('month')

JavaScript= pd.merge(JavaScript, JavaScriptFav, how='inner', on = 'month')

JavaScript = JavaScript.set_index('month')



JavaScript.plot(kind="bar", stacked=True)

plt.title("JavaScript Stats")

plt.xlabel('Month',fontsize=10)
#axe.subplot(0.5,0.5,0.5)
def plot_clustered_stacked(dfall, labels=None, title="Trends in Programming Languages in 2019",  H="x", **kwargs):



    n_df = len(dfall)

    n_col = len(dfall[0].columns) 

    n_ind = len(dfall[0].index)

    plt.figure(figsize=(20,10))

    axe = plt.subplot(111)

    

    #fig_size = plt.rcParams["figure.figsize"]

    #print(fig_size)

    #fig_size[0] = 20

    #fig_size[1] = 8

    #plt.rcParams["figure.figsize"] = fig_size

    #print(fig_size)

    

    for df in dfall : # for each data frame

        

        axe = df.plot(kind="bar",

                      linewidth=1,

                      stacked=True,

                      ax=axe,

                      legend=False,

                      grid=False,

                      **kwargs)  # make bar plots

    #df = {'index': [1,2,3,4,5],'comment_count':[0,0,0,0,0],'answer_count':[0,0,0,0,0],'favourite_count':[0,0,0,0,0]}

    #df = pd.DataFrame(df)

    #axe = df.plot(kind="bar",linewidth=1,stacked=True, ax=axe,legend=False,grid=False,**kwargs)  # make bar plots





    h,l = axe.get_legend_handles_labels() # get the handles we want to modify

    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df

        for j, pa in enumerate(h[i:i+n_col]):

            for rect in pa.patches: # for each index

                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))

                rect.set_hatch(H * int(i / n_col)) #edited part     

                rect.set_width(1 / float(n_df + 1))



    #axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 1.5)

    #axe.set_xticklabels(df.index, rotation = 0, fontsize = 20)

    plt.title(title,fontsize=30)

    plt.xlabel('Month', rotation = 0, fontsize=20)

    #plt.ylabel(fontsize=20)

    plt.xticks(rotation = 0,fontsize=20)

    plt.yticks(fontsize=20)

    #axe.set_index('month')



    # Add invisible data to add another legend

    n=[]        

    for i in range(n_df):

        n.append(axe.bar(0, 0, color="gray", hatch=H * i))



    l1 = axe.legend(h[:n_col], l[:n_col], loc=[0.01, -0.2])

    if labels is not None:

        l2 = plt.legend(n, labels, loc=[0.3, -0.2]) 

    axe.add_artist(l1)

    return axe



# Then, just call :

plot_clustered_stacked([Python, JavaScript],["Python","JavaScript"])
plot_clustered_stacked([PythonView, JavaScriptView],["Python","JavaScript"])
plot_clustered_stacked([PythonPosts, JavaScriptPosts],["Python","JavaScript"])
query = """select EXTRACT(month FROM creation_date) AS month, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%c++%'

        group by month

        order by month

        """



CplusPosts = stackoverflow.query_to_pandas(query)





query2 = """select EXTRACT(month FROM creation_date) AS month, sum(view_count) as view_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%c++%'

        group by month

        order by month

        """



CplusView = stackoverflow.query_to_pandas(query2)



query1 = """select EXTRACT(month FROM creation_date) AS month, sum(favorite_count) as favourite_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%c++%'

        group by month

        order by month

        """



CplusFav = stackoverflow.query_to_pandas(query1)



query3 = """select EXTRACT(month FROM creation_date) AS month, sum(answer_count) as answer_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%c++%'

        group by month

        order by month

        """



CplusAnswers = stackoverflow.query_to_pandas(query3)



query4 = """select EXTRACT(month FROM creation_date) AS month, sum(comment_count) as comment_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%c++%'

        group by month

        order by month

        """



CplusComments = stackoverflow.query_to_pandas(query4)

print(CplusPosts)

print(CplusPosts)

list(CplusPosts)
queryx = """select EXTRACT(month FROM creation_date) AS month, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 

        group by month

        order by month

        """



PostsCount= stackoverflow.query_to_pandas(queryx)

print(PostsCount)
PostsCount.posts
244539922967/9636060052421
CplusPosts['posts']= CplusPosts['posts']*100/PostsCount.posts
CplusPosts['posts']
#CplusPosts = CplusPosts.set_index('month')



y_pos = np.arange(5)

plt.plot(y_pos, CplusPosts['posts'])

plt.xticks(y_pos, CplusPosts['month'],fontsize=10)

plt.yticks(fontsize=10)

plt.xlabel('Month', fontsize=15)

plt.ylabel('Posts', fontsize=15)

plt.title('Trends in C+')





#fig = plt.figure()



#def on_plot_hover(event):

 #   for curve in plot.get_lines():

  #      if curve.contains(event)[0]:

   #         print("over %s", curve.get_gid())



#fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)



plt.show()
Cplus= pd.merge(CplusComments, CplusAnswers, how='inner', on = 'month')

Cplus = Cplus.set_index('month')

Cplus=pd.merge(Cplus,CplusFav,how='inner',on='month')

Cplus = Cplus.set_index('month')

#z=Python.groupby('month')



Cplus.plot(kind="bar", stacked=True)

#plt.xticks(y_pos, Python['month'],fontsize=10)

plt.title("C++ Stats")

plt.xlabel('Month',fontsize=10)



plot_clustered_stacked([Python, JavaScript,Cplus],["Python","JavaScript","Cplus"])
query = """select EXTRACT(month FROM creation_date) AS month, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%html%'

        group by month

        order by month

        """



htmlPosts = stackoverflow.query_to_pandas(query)





query2 = """select EXTRACT(month FROM creation_date) AS month, sum(view_count) as view_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%html%'

        group by month

        order by month

        """



htmlView = stackoverflow.query_to_pandas(query2)



query1 = """select EXTRACT(month FROM creation_date) AS month, sum(favorite_count) as favourite_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%html%'

        group by month

        order by month

        """



htmlFav = stackoverflow.query_to_pandas(query1)



query3 = """select EXTRACT(month FROM creation_date) AS month, sum(answer_count) as answer_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%html%'

        group by month

        order by month

        """



htmlAnswers = stackoverflow.query_to_pandas(query3)



query4 = """select EXTRACT(month FROM creation_date) AS month, sum(comment_count) as comment_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%html%'

        group by month

        order by month

        """



htmlComments = stackoverflow.query_to_pandas(query4)

htmlPosts.head()
HTML= pd.merge(htmlComments, htmlAnswers, how='inner', on = 'month')

HTML = HTML.set_index('month')

HTML=pd.merge(HTML,htmlFav,how='inner',on='month')

HTML = HTML.set_index('month')



HTML.plot(kind="bar", stacked=True)



plt.title("HTML Stats")

plt.xlabel('Month',fontsize=10)
plot_clustered_stacked([Python, JavaScript,Cplus,HTML],["Python","JavaScript","C++","HTML"])
plot_clustered_stacked([PythonPosts, JavaScriptPosts,CplusPosts,htmlPosts],["Python","JavaScript","C++","html"])
query = """select EXTRACT(month FROM creation_date) AS month, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%ruby%'

        group by month

        order by month

        """



RubyPosts = stackoverflow.query_to_pandas(query)





query2 = """select EXTRACT(month FROM creation_date) AS month, sum(view_count) as view_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%ruby%'

        group by month

        order by month

        """



RubyView = stackoverflow.query_to_pandas(query2)



query1 = """select EXTRACT(month FROM creation_date) AS month, sum(favorite_count) as favourite_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%ruby%'

        group by month

        order by month

        """



RubyFav = stackoverflow.query_to_pandas(query1)



query3 = """select EXTRACT(month FROM creation_date) AS month, sum(answer_count) as answer_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%ruby%'

        group by month

        order by month

        """



RubyAnswers = stackoverflow.query_to_pandas(query3)



query4 = """select EXTRACT(month FROM creation_date) AS month, sum(comment_count) as comment_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%ruby%'

        group by month

        order by month

        """



RubyComments = stackoverflow.query_to_pandas(query4)

RubyFav.head()



Ruby= pd.merge(RubyComments, RubyAnswers, how='inner', on = 'month')

Ruby =Ruby.set_index('month')

Ruby=pd.merge(Ruby,RubyFav,how='inner',on='month')

Ruby = Ruby.set_index('month')



Ruby.plot(kind="bar", stacked=True)



plt.title("Ruby Stats")

plt.xlabel('Month',fontsize=10)
plot_clustered_stacked([Python, JavaScript,Cplus,HTML,Ruby],["Python","JavaScript","C++","HTML","Ruby"])
plot_clustered_stacked([PythonPosts, JavaScriptPosts,CplusPosts,htmlPosts,RubyPosts],["Python","JavaScript","C++","HTML","Ruby"])
query = """select EXTRACT(month FROM creation_date) AS month, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%mysql%'

        group by month

        order by month

        """



MySQLPosts = stackoverflow.query_to_pandas(query)





query2 = """select EXTRACT(month FROM creation_date) AS month, sum(view_count) as view_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%mysql%'

        group by month

        order by month

        """



MySQLView = stackoverflow.query_to_pandas(query2)



query1 = """select EXTRACT(month FROM creation_date) AS month, sum(favorite_count) as favourite_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%mysql%'

        group by month

        order by month

        """



MySQLFav = stackoverflow.query_to_pandas(query1)



query3 = """select EXTRACT(month FROM creation_date) AS month, sum(answer_count) as answer_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%mysql%'

        group by month

        order by month

        """



MySQLAnswers = stackoverflow.query_to_pandas(query3)



query4 = """select EXTRACT(month FROM creation_date) AS month, sum(comment_count) as comment_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%mysql%'

        group by month

        order by month

        """



MySQLComments = stackoverflow.query_to_pandas(query4)





MySQLFav.head()
MySQL= pd.merge(MySQLComments, MySQLAnswers, how='inner', on = 'month')

MySQL =MySQL.set_index('month')

MySQL=pd.merge(MySQL,MySQLFav,how='inner',on='month')

MySQL = MySQL.set_index('month')



MySQL.plot(kind="bar", stacked=True)



plt.title("MySQL Stats")

plt.xlabel('Month',fontsize=10)
plot_clustered_stacked([Python, JavaScript,Cplus,HTML,Ruby,MySQL],["Python","JavaScript","C++","HTML","Ruby","MySQL"])
query = """select EXTRACT(month FROM creation_date) AS month, sum(id) as posts

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%c#%'

        group by month

        order by month

        """



CHashPosts = stackoverflow.query_to_pandas(query)





query2 = """select EXTRACT(month FROM creation_date) AS month, sum(view_count) as view_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%c#%'

        group by month

        order by month

        """



CHashView = stackoverflow.query_to_pandas(query2)



query1 = """select EXTRACT(month FROM creation_date) AS month, sum(favorite_count) as favourite_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%c#%'

        group by month

        order by month

        """



CHashFav = stackoverflow.query_to_pandas(query1)



query3 = """select EXTRACT(month FROM creation_date) AS month, sum(answer_count) as answer_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%c#%'

        group by month

        order by month

        """



CHashAnswers = stackoverflow.query_to_pandas(query3)



query4 = """select EXTRACT(month FROM creation_date) AS month, sum(comment_count) as comment_count

        from `bigquery-public-data.stackoverflow.posts_questions`

        where extract(year from creation_date) = 2019 and extract(month from creation_date) < 6 and tags like '%c#%'

        group by month

        order by month

        """



CHashComments = stackoverflow.query_to_pandas(query4)





CHashFav.head()
CHash=pd.merge(CHashComments, CHashAnswers, how='inner', on = 'month')

CHash =CHash.set_index('month')

CHash=pd.merge(CHash,CHashFav,how='inner',on='month')

CHash = CHash.set_index('month')



CHash.plot(kind="bar", stacked=True)



plt.title("CHash Stats")

plt.xlabel('Month',fontsize=10)
plot_clustered_stacked([Python, JavaScript,Cplus,HTML,Ruby,MySQL],["Python","JavaScript","C++","HTML","Ruby","MySQL"])
Ruby= pd.merge(RubyPosts, htmlPosts, how='inner', on = 'month')

Ruby =Ruby.set_index('month')

Ruby= pd.merge(Ruby, JavaScriptPosts, how='inner', on = 'month')

Ruby =Ruby.set_index('month')

Ruby=pd.merge(Ruby,PythonPosts,how='inner',on='month')

Ruby = Ruby.set_index('month')

Ruby=pd.merge(Ruby,MySQLPosts,how='inner',on='month')

Ruby = Ruby.set_index('month')

Ruby=pd.merge(Ruby,CHashPosts,how='inner',on='month')

Ruby = Ruby.set_index('month')

#Ruby['posts']= Ruby['posts']*100/PostsCount.posts
Ruby.plot(kind='line')

plt.xlabel('Month', fontsize=15)

plt.ylabel('Posts', fontsize=15)

y_pos=[1,2,3,4,5]

plt.xticks(y_pos,fontsize=10)

plt.yticks(fontsize=10)

plt.title('Trends in Technology')

plt.legend(['Ruby','HTML','JavaScript','Python','MySQL','C#'],loc=[1.0,0.7])

plt.show()