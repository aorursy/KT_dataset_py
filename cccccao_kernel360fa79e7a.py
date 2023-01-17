import time

import pandas as pd

import plotly.graph_objs as go

import plotly.express as px



from Bio import Entrez, Medline

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected = False)



print("导入成功")
# 如果要搜索其他文章，可以更改此处的搜索关键词

KEYWORDS = "(circadian) AND (breast cancer)"



# 设置搜索的数据库，NCBI有许多数据库，默认从PubMed搜索

DBNAME = "pubmed"



# 默认即可，是用于告诉NCBI的Entrez是什么脚本在运行

Entrez.email = "sicheng.gu@foxmail.com"

Entrez.tool = "PreMetaAnalysis"



print("设定完毕")
def get_db_counts(terms):

    """

    使用EGQuery来搜索NCBI旗下各个数据库收录信息，并且绘制成Bar chart

    参数：

        terms: 搜索的关键词。字符串类项str

    """

    handle = Entrez.egquery(term = terms)

    records = Entrez.read(handle)

    db_count = {record['DbName']: record['Count'] for record in records["eGQueryResult"]}

    

    # 将返回结果Error修正为0，并修正所有类项到int

    for k, v in db_count.items():

        if v == "Error": 

            db_count[k] = 0
handle = Entrez.egquery(term = KEYWORDS)

records = Entrez.read(handle)

db_count = {record['DbName']: record['Count'] for record in records["eGQueryResult"]}

handle.close()

# 将返回结果Error修正为0，并修正所有类项到int

for k, v in db_count.items():

    if v == "Error": 

        db_count[k] = 0

    else:

        db_count[k] = int(db_count[k])



# 将db_count按照值从大到小排序

df = pd.DataFrame.from_dict(db_count, 

                            orient = 'index', 

                            columns = ['count'])

df.sort_values(by = ['count'], 

               inplace = True, 

               ascending = False)



# 绘制条形图

dbnames = [str(i) for i in df.index]

counts = [i for i in df['count']]



trace = [go.Bar(x = dbnames, 

                y = counts, 

                text = counts, 

                textfont_size = 14, 

                textposition = 'inside')]

layout = go.Layout(title = 'Bar Chart of Entrez Database Counts', 

                   titlefont = {'family': "Times New Roman", 

                                'size': 22}, 

                   xaxis = {'title': "Database", 

                            'titlefont': {

                                'family': "Times New Roman", 

                                'size': 14

                            }

                           }, 

                   yaxis = {'title': 'Counts', 

                            'titlefont': {

                                'family': 'Times New Roman', 

                                'size': 14

                            }

                           }

                  )

fig = go.Figure(data = trace, 

                layout = layout)

fig.update_traces(marker_color = px.colors.sequential.Viridis)

fig.update_yaxes(type = 'log')

py.offline.iplot(fig)





#fig = px.bar()