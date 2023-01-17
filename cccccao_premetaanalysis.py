# 运行本cell来安装相关依赖

!pip install biopython plotly
# 运行本cell来导入相关模块

import time

import pandas as pd

import plotly as py

import plotly.graph_objs as go



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) # 这行可以让图表正常渲染 不要遗漏



from Bio import Entrez, Medline

print("模块导入完成！")
# 如果要搜索其他文章，可以更改此处的搜索关键词

KEYWORDS = "((((night shift work) OR (shift work)) OR (light at night)) OR (circadian disruption)) AND (breast cancer)"



# 设置搜索的数据库，NCBI有许多数据库，默认从PubMed搜索

DBNAME = "pubmed"



# 默认即可，是用于告诉NCBI的Entrez是什么脚本在运行

Entrez.email = "sicheng.gu@foxmail.com"

Entrez.tool = "PreMetaAnalysis"
def get_db_counts(db, terms):

    """

    使用EGQuery来获取给定关键词搜索的情况，并且绘制条形图

    """

    # 获取给定关键词在各个数据库中收录的条目

    handle = Entrez.egquery(term=terms)

    record = Entrez.read(handle)

    dict = {row['DbName']: row['Count'] for row in record["eGQueryResult"]}

    # 因为部分数据库返回Error 要纠正成0

    for k, v in dict.items():

        if v == 'Error':

            dict[k] = 0

    dbnames = [str(i) for i in dict.keys()]

    counts = [int(i) for i in dict.values()]

    print("在[{}]数据库中，搜索条目有[{}]个".format(db, dict[db]))

    handle.close()

    

    # 绘图

    trace = [go.Bar(y = DbName, x = Count, orientation='h')]

    fig = go.Figure(data = trace)

    py.offline.iplot(fig)



    return counts



counts = get_db_counts(db=DBNAME, terms=KEYWORDS)
def get_pmid(dbname, terms, maxcount):

    """

    使用ESearch获取UID

    """

    handle = Entrez.esearch(db=dbname, term=terms, retmax=maxcount)

    record = Entrez.read(handle)

    id_list = record["IdList"]

    print("PMID搜索...done!")

    handle.close()

    return id_list



id_list = get_pmid(dbname=DBNAME, terms=KEYWORDS, maxcount=counts)
def get_all_info(db, id):

    handle = Entrez.efetch(db=db,

                       id=id,

                       rettype="medline",

                       retmode="text")

    records = Medline.parse(handle)

    # 迭代records来获取其中部分数据

    # 取文献状态STAT 发表日期DP 标题TI DOI号码LID 摘要AB 期刊名JT 作为需要数据

    key_filter = ['STAT', 'DP', 'TI', 'LID', 'AB', 'JT']

    dfs = pd.DataFrame(columns=key_filter)

    i = 0

    for record in records:

        info_dict = {k: v for k, v in record.items() if k in key_filter}

        df = pd.DataFrame([info_dict])

        # 一行行加入dfs

        dfs = dfs.append(df)

        title = info_dict['TI']

        date = info_dict['DP']

        print("第{}篇  发表日期: [{}]  标题为: [{}]".format(i + 1, date, title))

        i += 1

        # time.sleep(0.05)

    dfs.reset_index(drop=True)

    return dfs

    



df = get_all_info(db=DBNAME, id=id_list)
# 保存到CSV文件

FILENAME = 'ORI_' + counts + '_' + KEYWORDS + '.csv'

df.to_csv(FILENAME, header=True, index=False)

print("写入完成！")