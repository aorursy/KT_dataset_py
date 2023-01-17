# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings("ignore")
!pip install pyecharts
!pip install echarts-china-provinces-pypkg
survey = pd.read_excel('../input/survey/survey.xlsx')
survey.head()
mapdf = pd.read_excel('../input/mapcount/map.xlsx')
mapdf.columns = ['provices' ,'count']
mapdf.describe()
from pyecharts.charts import Map
from pyecharts.charts import Geo
from pyecharts import options as opts
from pyecharts import options as opts
from pyecharts.globals import ChartType
from pyecharts.faker import Faker
def map_base() -> Map:
    c = (
        Map()
        .add("map", [list(z) for z in zip(mapdf['provices'], mapdf['count'])], "china")
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
                            visualmap_opts=opts.VisualMapOpts(),
                            title_opts=opts.TitleOpts(title="调查大学所在省人数分布"),
                        )
        )
    return c


map_base().render_notebook()
source = survey['来源']
source = pd.DataFrame(source.value_counts())
from pyecharts.charts import Pie
def pie_base() -> Pie:
    c = (
        Pie()
        .add("source", [list(z) for z in zip(['wechat', 'others'], source['来源'])])
        .set_colors([  "orange", "purple"])
    .set_global_opts(title_opts=opts.TitleOpts(title="Pie-source"))
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
        )
    return c


pie_base().render_notebook()
sex = survey['3、您的性别：']
sex = pd.DataFrame(sex.value_counts())

def pie_base() -> Pie:
    c = (
        Pie()
        .add("sex", [list(z) for z in zip(['male', 'female'], sex['3、您的性别：'])])
        .set_colors(["blue", "green", "yellow", "red", "pink", "orange", "purple"])
    .set_global_opts(title_opts=opts.TitleOpts(title="Pie-source"))
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
        )
    return c


pie_base().render_notebook()
major_df = pd.DataFrame(survey['4、您所学的专业'].value_counts())
major_df.head()
from scipy import stats
print(stats.shapiro(survey['4、您所学的专业']))
def major_pie() -> Pie:
    c = (
        Pie()
    .add(
        "",
        [list(z) for z in zip(["理工科类","文史类","经管类","医学类","艺术类"], major_df['4、您所学的专业'].tolist())],
        radius=["40%", "55%"],
        label_opts=opts.LabelOpts(
            position="outside",
            formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",
            background_color="#eee",
            border_color="#aaa",
            border_width=1,
            border_radius=4,
            rich={
                "a": {"color": "#999", "lineHeight": 22, "align": "center"},
                "abg": {
                    "backgroundColor": "#e3e3e3",
                    "width": "100%",
                    "align": "right",
                    "height": 22,
                    "borderRadius": [4, 4, 0, 0],
                },
                "hr": {
                    "borderColor": "#aaa",
                    "width": "100%",
                    "borderWidth": 0.5,
                    "height": 0,
                },
                "b": {"fontSize": 16, "lineHeight": 33},
                "per": {
                    "color": "#eee",
                    "backgroundColor": "#334455",
                    "padding": [2, 4],
                    "borderRadius": 2,
                },
            },
        ),
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="专业分析"))
    )
    return c
major_pie().render_notebook()
x_data = ["大三", "大一", "大二", "研究生", "大四"]
grade = pd.DataFrame(survey['4、您所学的专业'].value_counts())
data_pair = [list(z) for z in zip(x_data, grade['4、您所学的专业'])]
def pie_base() -> Pie:
    c = (
        Pie(init_opts=opts.InitOpts(width="800px", height="800px"))
        .add(
            series_name="grade",
            data_pair=data_pair,
            rosetype="radius",
            radius="55%",
            center=["50%", "50%"],
            label_opts=opts.LabelOpts(is_show=False, position="center"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Customized Pie - grades",
                pos_left="center",
                pos_top="20",
                title_textstyle_opts=opts.TextStyleOpts(color="#2c343c"),
                ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
        .set_series_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
            ),
            label_opts=opts.LabelOpts(color="rgba(0, 0, 0, 3)"),
        )
    )
    return c

pie_base().render_notebook()
activity = []
for i in range(6, 11):
    activity.append(survey.iloc[:,i].value_counts()[0])
activity = pd.DataFrame(activity)
activity.index = ["企业实习","社会志愿活动","学术报告会议","科研立项","学科竞赛"]
activity.columns = ['人数']
activity.head(10)
from pyecharts.charts import Bar
def bar_base() -> Bar:
    c=(
        Bar()
        .add_xaxis(["企业实习","社会志愿活动","学术报告会议","科研立项","学科竞赛"])
        .add_yaxis("", activity['人数'].tolist(), category_gap="80%")
        .set_global_opts(title_opts=opts.TitleOpts(title="Bar-单系列柱间距离"))
    )
    return c
bar_base().render_notebook() 
from pyecharts.charts import Timeline, Bar
from pyecharts.globals import ThemeType
activitylist =  ["企业实习","社会志愿活动","学术报告会议","科研立项","学科竞赛"]
data_major = {"文史类": [11,11,11,13,13],
              "经管类": [9,10,8,12,11],
              "理工科类":[17,31,23,27,26],
              "艺术类":[7,4,6,4,7],
              "医学类":[7,10,9,9,4]
             }
total_data = pd.DataFrame(data_major)
total_data.index = ["企业实习","社会志愿活动","学术报告会议","科研立项","学科竞赛"]
colorlist = ["#90EE90","#749f83","#87CEFA","#d48265","#749f83"]
def get_cross_major_chart(major: str, color:str) -> Bar:
    bar = (
        Bar()
        .add_xaxis(xaxis_data=activitylist)
        .add_yaxis(
          "", total_data[major].tolist(),
            itemstyle_opts=opts.ItemStyleOpts(color=color)
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title= "参加活动交叉分析", subtitle=major
            ),
            tooltip_opts=opts.TooltipOpts(
                is_show=True, trigger="axis", axis_pointer_type="shadow"
            ),
        )
    )
    return bar
majors = ["文史类","经管类","理工科类","艺术类","医学类"]
timeline = Timeline(init_opts=opts.InitOpts(width="1000px", height="500px"))

for major,color in zip(majors, colorlist):
    #print(total_data["企业实习"][major])
    timeline.add(get_cross_major_chart(major, color), time_point=str(major))
    
timeline.add_schema(is_auto_play=True, play_interval=2000)
timeline.render_notebook()
from pyecharts.charts import Gauge

c = (
    Gauge()
    .add("", [("", 51.77)])
    .set_global_opts(title_opts=opts.TitleOpts(title="目前科研完成进度均值"))
    
)
c.render_notebook()
from pyecharts.charts import Scatter
sche_df = pd.read_excel('../input/schedule/schedule.xlsx')
(
    Scatter(init_opts=opts.InitOpts(width="800px", height="800px"))
    .add_xaxis(xaxis_data=sche_df['阶段'])
    .add_yaxis(
        series_name="",
        y_axis=sche_df['进度'],
        symbol_size=20,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_series_opts()
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(is_show=False),
    )
).render_notebook()


from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
anova_reA= anova_lm(ols('阶段~C(进度)',data=sche_df[['阶段','进度']]).fit())
print(anova_reA)
from sklearn.linear_model import LinearRegression
lrModel = LinearRegression()
def status(x) : 
    return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),
                      x.quantile(.75),x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),
                      x.std(),x.skew(),x.kurt()],index=['总数','最小值','最小值位置','25%分位数',
                    '中位数','75%分位数','均值','最大值','最大值位数','平均绝对偏差','方差','标准差','偏度','峰度'])
sche_df.corr()
r,p = stats.pearsonr(sche_df['阶段'],sche_df['进度'])  # 相关系数和P值
print('相关系数r为 = %6.3f，p值为 = %6.3f'%(r,p))
sche_df.apply(status)
sche_df = pd.read_excel('../input/schedule/schedule.xlsx')
grouped = sche_df.groupby('阶段')
begin_state = ["大一上","大一下","大二上","大二下","大三上","大三下"]
mean_list = []
std_list = []
max_list = []
min_list = []
median_list = []
count_list = []
for name,group in grouped:
    mean_list.append(group["进度"].mean())
    max_list.append(group["进度"].max())
    min_list.append(min(group["进度"]))
    std_list.append(group["进度"].std())
    median_list.append(group["进度"].median())
sche_df = pd.DataFrame([mean_list, std_list, max_list, min_list, median_list])
sche_df.columns = begin_state
sche_df.index = ["mean","std","max","min","median"]
sche_df["max"] = 100
begin_state.append("max")
sche_df
from pyecharts.charts import Sankey
tl = Timeline()
item_list = ["均值","方差","最大值","最小值","中位数"]
nodes = [{"name": be} for be in begin_state]
i = 0
for item, row in sche_df.iterrows():
    row = row.sort_values(axis=0)
    state = pd.DataFrame(row).index.values
    links = [
        {"source": state[0], "target": state[1], "value": row[0]},
        {"source": state[1], "target": state[2], "value": row[1]},
        {"source": state[2], "target": state[3], "value": row[2]},
        {"source": state[3], "target": state[4], "value": row[3]},
        {"source": state[4], "target": state[5], "value": row[4]},
        {"source": state[5], "target": state[6], "value": row[5]},
    ]
    sankey = (
        Sankey()
        .add(
            "data",
            nodes,
            links,
            linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
            label_opts=opts.LabelOpts(position="right"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="描述性统计比较")
        )
    )
    tl.add(sankey, item_list[i])
    i += 1
tl.render_notebook()
from pyecharts.charts import Boxplot
ability_df = pd.read_excel("../input/ability/ability.xlsx")
y_data = []
for index, row in ability_df.iteritems():
    y_data.append(row.to_list())
box_plot = Boxplot()

c = Boxplot()
c.add_xaxis(ability_df.columns.values)
c.add_yaxis("", c.prepare_data(y_data[1:]))
c.set_global_opts(title_opts=opts.TitleOpts(title="BoxPlot-基本示例"))

box_plot = (
    box_plot.add_xaxis(xaxis_data=["innovation","reading","experiment","teamwork"])
    .add_yaxis(series_name="", y_axis=box_plot.prepare_data(y_data[1:]))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            pos_left="center", title="箱线图"
        ),
        tooltip_opts=opts.TooltipOpts(trigger="item", axis_pointer_type="shadow"),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=True,
            splitarea_opts=opts.SplitAreaOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="",
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
        ),
    )
    .set_series_opts(tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}"))
)
box_plot.render_notebook()
ability_df.apply(status)
ability_df.head()
ax = ability_df.plot.scatter(x='grade', y='innovation', color='b', label='innovation')
ability_df.plot.scatter(x='grade', y='reading', color='g', label='reading',ax=ax)
ability_df.plot.scatter(x='grade', y='experiment', color='r', label='experiment',ax=ax)
ability_df.plot.scatter(x='grade', y='teamwork', color='gray', label='teamwork',ax=ax)
import seaborn as sns
sns.pairplot(ability_df.drop(labels='grade', axis=1),kind="reg" )
ability_df.drop(labels='grade', axis=1).corr()
print(stats.shapiro(ability_df['teamwork']))
stats.levene(ability_df['innovation'], ability_df['reading'])
r,p = stats.pearsonr(ability_df['teamwork'],ability_df['teamwork'])  # 相关系数和P值
print('相关系数r为 = %6.3f，p值为 = %6.6f'%(r,p))
from pyecharts.commons.utils import JsCode
grouped = ability_df.groupby('grade')
ability = ["innovation","reading","experiment","teamwork"]
data = []
for item, group in grouped:
    temp = {"ability": ability,
           "boxData":[group["innovation"].tolist(),
                     group["reading"].tolist(),
                     group["experiment"].tolist(),
                     group["teamwork"].tolist()
                     ]
           }
    data.append(temp)

(
    Boxplot(init_opts=opts.InitOpts(width="800px", height="600px"))
    .add_xaxis(xaxis_data=ability)
    .add_yaxis(
        series_name="大一",
        y_axis=data[0]["boxData"],
        tooltip_opts=opts.TooltipOpts(
            formatter=JsCode(
                """function(param) { return [
                            'Experiment ' + param.name + ': ',
                            'upper: ' + param.data[0],
                            'Q1: ' + param.data[1],
                            'median: ' + param.data[2],
                            'Q3: ' + param.data[3],
                            'lower: ' + param.data[4]
                        ].join('<br/>') }"""
            )
        ),
    )
    .add_yaxis(
        series_name="大二",
        y_axis=data[1]["boxData"],
        tooltip_opts=opts.TooltipOpts(
            formatter=JsCode(
                """function(param) { return [
                            'Experiment ' + param.name + ': ',
                            'upper: ' + param.data[0],
                            'Q1: ' + param.data[1],
                            'median: ' + param.data[2],
                            'Q3: ' + param.data[3],
                            'lower: ' + param.data[4]
                        ].join('<br/>') }"""
            )
        ),
    )
    .add_yaxis(
        series_name="大三",
        y_axis=data[2]["boxData"],
        tooltip_opts=opts.TooltipOpts(
            formatter=JsCode(
                """function(param) { return [
                            'Experiment ' + param.name + ': ',
                            'upper: ' + param.data[0],
                            'Q1: ' + param.data[1],
                            'median: ' + param.data[2],
                            'Q3: ' + param.data[3],
                            'lower: ' + param.data[4]
                        ].join('<br/>') }"""
            )
        ),
    )
    .add_yaxis(
        series_name="大四",
        y_axis=data[3]["boxData"],
        tooltip_opts=opts.TooltipOpts(
            formatter=JsCode(
                """function(param) { return [
                            'Experiment ' + param.name + ': ',
                            'upper: ' + param.data[0],
                            'Q1: ' + param.data[1],
                            'median: ' + param.data[2],
                            'Q3: ' + param.data[3],
                            'lower: ' + param.data[4]
                        ].join('<br/>') }"""
            )
        ),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Grades and ability", pos_left="center"),
        legend_opts=opts.LegendOpts(pos_top="3%"),
        tooltip_opts=opts.TooltipOpts(trigger="item", axis_pointer_type="shadow"),
        xaxis_opts=opts.AxisOpts(
            name_gap=20,
            boundary_gap=True,
            splitarea_opts=opts.SplitAreaOpts(
                areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
            splitline_opts=opts.SplitLineOpts(is_show=False),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            min_=0,
            max_=100,
            splitarea_opts=opts.SplitAreaOpts(is_show=False),
        ),
        datazoom_opts=[
            opts.DataZoomOpts(type_="inside", range_start=0, range_end=20),
            opts.DataZoomOpts(type_="slider", xaxis_index=0, is_show=True),
        ],
    )
).render_notebook()

ab_major_df = pd.read_excel("../input/major-ab/major_ability.xlsx")
grouped = ab_major_df.groupby('major')
ability = ["innovation","reading","experiment","teamwork"]
ab_major_data = []
for item, group in grouped:
    ab_major_data.append([[group["innovation"].mean(),group["reading"].mean(), group["experiment"].mean(),group["teamwork"].mean()]])

print(ab_major_data)
from pyecharts.charts import Radar
majors = ["文史类", "经管类", "理工科", "艺术类", "医学类"]
(
    Radar(init_opts=opts.InitOpts(width="800px", height="720px", bg_color="#CCCCCC"))
    .add_schema(
        schema=[
            opts.RadarIndicatorItem(name="创新-innovation", max_=100),
            opts.RadarIndicatorItem(name="文献阅读-reading", max_=100),
            opts.RadarIndicatorItem(name="实验-experiment", max_=100),
            opts.RadarIndicatorItem(name="团队-teamwork", max_=100),
        ],
        splitarea_opt=opts.SplitAreaOpts(
            is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
        ),
        textstyle_opts=opts.TextStyleOpts(color="#fff"),
    )
    .add(
        series_name="文史类",
        data=ab_major_data[0],
        linestyle_opts=opts.LineStyleOpts(color="#CD0000",width=5),
    )
    .add(
        series_name="经管类",
        data=ab_major_data[1],
        linestyle_opts=opts.LineStyleOpts(color="#5CACEE", width=5),
    )
    .add(
        series_name="理工科类",
        data=ab_major_data[2],
        linestyle_opts=opts.LineStyleOpts(color="#87CEEB",width= 5),

    )
    .add(
        series_name="艺术类",
        data=ab_major_data[3],
        linestyle_opts=opts.LineStyleOpts(color="#E0FFFF",width = 5),
    )
    .add(
        series_name="医学类",
        data=ab_major_data[4],
        linestyle_opts=opts.LineStyleOpts(color="#B0C4DE",width=5),
    )
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    
    .set_global_opts(
        title_opts=opts.TitleOpts(title="不同专业雷达图"), legend_opts=opts.LegendOpts()
    )
    
).render_notebook()

majors = ["文史类", "经管类", "理工科", "艺术类", "医学类"]
(
    Radar(init_opts=opts.InitOpts(width="800px", height="720px", bg_color="#CCCCCC"))
    .add_schema(
        schema=[
            opts.RadarIndicatorItem(name="创新-innovation", max_=100),
            opts.RadarIndicatorItem(name="文献阅读-reading", max_=100),
            opts.RadarIndicatorItem(name="实验-experiment", max_=100),
            opts.RadarIndicatorItem(name="团队-teamwork", max_=100),
        ],
        splitarea_opt=opts.SplitAreaOpts(
            is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
        ),
        textstyle_opts=opts.TextStyleOpts(color="#00008B"),
    )
    .add(
        series_name="文史类",
        data=[[52, 67, 33, 45]],
        linestyle_opts=opts.LineStyleOpts(color="#CD0000",width=5),
    )
    .add(
        series_name="经管类",
        data=ab_major_data[1],
        linestyle_opts=opts.LineStyleOpts(color="#5CACEE", width=5),
    )
    .add(
        series_name="理工科类",
        data=[[55, 43, 67, 54]],
        linestyle_opts=opts.LineStyleOpts(color="#87CEEB",width= 5),

    )
    .add(
        series_name="艺术类",
        data=[[65, 46, 32, 57]],
        linestyle_opts=opts.LineStyleOpts(color="#E0FFFF",width = 5),
    )
    .add(
        series_name="医学类",
        data=[[43, 72, 68, 73]],
        linestyle_opts=opts.LineStyleOpts(color="#B0C4DE",width=5),
    )
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    
    .set_global_opts(
        title_opts=opts.TitleOpts(title="不同专业雷达图"), legend_opts=opts.LegendOpts()
    )
    
).render_notebook()
infor_df = pd.read_excel('../input/getinformation/getinformation.xlsx')
grades = infor_df['X\Y'].tolist()
ways = infor_df.columns.values[1:].tolist()
from pyecharts.charts import Funnel
colorlist = ["#90EE90","#749f83","#87CEFA","#d48265","#749f83"]
def get_cross_ways_chart(grade: str, color:str, row) -> Funnel:
    c = (
        Funnel()
        .add("来源", [list(z) for z in zip(ways, row)])
        .set_global_opts(title_opts=opts.TitleOpts(title="年级-来源"))
    )
    return c
timeline = Timeline(init_opts=opts.InitOpts(width="800px", height="600px"))
for grade, color,(item, row) in zip(grades, colorlist, infor_df.iterrows()):
    timeline.add(get_cross_ways_chart(grade, color, row[1:]), time_point=str(grade))
    
timeline.add_schema(is_auto_play=True, play_interval=2000)
timeline.render_notebook()
trouble_df = pd.read_excel("../input/trouble/trouble.xlsx")
grouped = trouble_df.groupby('sex')
trouble_sex_data = []
for item, group in grouped:
    temp = []
    temp.append(group["压榨了我的学习和生活时间"].sum())
    temp.append(group["与他人工作让我不自在"].sum())
    temp.append(group["参加科研有风险，害怕花了时间和精力却没有得到预期的回报"].sum())
    temp.append(group["没有熟悉的学长学姐带，担心自己能力不足无法开展"].sum())
    temp.append(group["不了解学校科研动态，不知道在哪里关注"].sum())
    temp.append(group["学业压力过大"].sum())
    temp.append(group["没有合适的导师人选，不敢冒然入圈"].sum())
    temp.append(group["做科研让自己没有时间维系与同学们的感情"].sum())
    trouble_sex_data.append(temp)
xaxis = trouble_df.columns.values[1:].tolist()
trouble_sex_data[1]
c = (
    Bar()
    .add_xaxis(xaxis)
    .add_yaxis("男性",[11, 12, 20, 23, 17, 22, 20, 3])
    .add_yaxis("女性", [16, 14, 20, 30, 29, 26, 25, 11])
    .reversal_axis()
    .set_series_opts(label_opts=opts.LabelOpts(position="right"))
    .set_global_opts(title_opts=opts.TitleOpts(title="性别与困难"))
)
c.render_notebook()
