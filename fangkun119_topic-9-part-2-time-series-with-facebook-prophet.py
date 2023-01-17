import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

from scipy import stats

import statsmodels.api as sm

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/medium_posts.csv')

df.head()
# 只用published，url两列，去掉空值，去掉重复数据

df = df[['published', 'url']].dropna().drop_duplicates()
# 将时间列转成pandas._libs.tslibs.timestamps.Timestamp类型

df['published'] = pd.to_datetime(df['published'])

print(type(df['published'][1]))

df.head()
# 按发布时间排序

df.sort_values(by=['published']).head(n=3)
# 发布时间中值是2012-08-15

# 只实用2012-08-15到2017-06-26之间的数据，过早的数据太久远不需要，另外也很可能是脏数据（例如1970-01-01 00:00:00)

df = df[(df['published'] > '2012-08-15') & (df['published'] < '2017-06-26')].sort_values(by=['published'])

df.head(n=3)
df.tail(n=3)
# 我们关心的是，url被发布的数量，因此对其计数

aggr_df = df.groupby('published')[['url']].count()

aggr_df.columns = ['posts']

aggr_df.head(n=2)
aggr_df.head(n=3)
# 这段代码有问题，没有把时区去掉

# 输出应该是 2012-08-15 

# 而不是 2012-08-15 00:00:00+00:00	

daily_df = aggr_df.resample('D').apply(sum)

# print(type(daily_df))

# print(type(daily_df.index))

# print(type(daily_df.index.date))

# print(type(daily_df.index.strftime("%Y-%m-%d")))

# print(daily_df.index.strftime("%Y-%m-%d")[:3])

# daily_df.index = daily_df.index.strftime("%Y-%m-%d")

daily_df.head(n=3)
# 初始化Plotly Library用于交互式可视化

from plotly.offline import init_notebook_mode, iplot

from plotly import graph_objs as go



# Initialize plotly

init_notebook_mode(connected=True)
# 用于可视化时间序列的函数

def plotly_df(df, title=''):

    """Visualize all the dataframe columns as line plots."""

    common_kw = dict(x=df.index, mode='lines')                                 #横坐标是日期(df.index)

    data      = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns] #添加数据：每列一个散点图，散点图名称是列名

    layout    = dict(title=title)                                              #添加标题

    fig       = dict(data=data, layout=layout)                                 #将标题和数据组装在一起

    iplot(fig, show_link=False)                                                #绘图
# 绘制时间序列图（可以zoom in）

plotly_df(daily_df, title='Posts on Medium (daily)')
# 数据粒度由天改为周

weekly_df = daily_df.resample('W').apply(sum)

weekly_df.head(n=3)
# 可视化weekly bin粒度的数据

plotly_df(weekly_df, title='Posts on Medium (weekly)')
daily_df = daily_df.loc[daily_df.index >= '2015-01-01']

daily_df.head(n=3)
from fbprophet import Prophet



import logging

logging.getLogger().setLevel(logging.ERROR)
# 上面的数据还不能用于训练，ds中包含了时区信息，需要将其抹去

# 例如：将2012-08-15 00:00:00+00:00改成2012-08-15

print("daily_df:{}\ndaily_df.index:{}\ndaily_df.index.date:{}\ndaily_df.daily_df.index.strftime:{}".format(

        type(daily_df), type(daily_df.index), type(daily_df.index.date), type(daily_df.index.strftime("%Y-%m-%d"))))

daily_df.index = daily_df.index.tz_localize(None)

# daily_df.index = daily_df.index.strftime("%Y-%m-%d")

# print("daily_df.index:{}".format(type(daily_df.index[1])))

# daily_df.head(n=3)
df = daily_df.reset_index()

df.columns = ['ds', 'y']

df.tail(n=3)
# 训练集（不包含最后30天的数据）

prediction_size = 30

train_df = df[:-prediction_size]

train_df.tail(n=3)
# 训练模型

m = Prophet()

m.fit(train_df);

train_df.head(n=2)
# 生成的扩展的DataFrame用于存放预测结果，该DataFrame能容纳的数据天数比训练集多prediction_size天

future = m.make_future_dataframe(periods=prediction_size)

print(prediction_size)

print(len(train_df))

print(len(future))

future.tail(n=2)
# 生成预测序列：时间范围比训练集多prediction_size=30天，除了预测值yhat以外，还有趋势、周期、上下界等分量

forecast = m.predict(future)

forecast.tail(n=3)
#forecast['ds'] = pd.to_datetime(df['ds'])

#forecast.index = forecast.ds

print(type(forecast.index))

print(type(forecast['ds'][0]))

print(type(m))

forecast.head(n=2)
forecast_to_plot = forecast

#forecast_to_plot['ds'] = list(map(lambda x: x.date(), forecast_to_plot['ds'])) #print(type(forecast_to_plot['ds'][0]))

#m.plot(forecast_to_plot);
# m.plot_components(forecast);
print(', '.join(forecast.columns))
# 定义一个函数，用来把预测结果和原始样本join起来

def make_comparison_dataframe(historical, forecast):

    """Join the history with the forecast.

    

       The resulting dataset will contain columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.

    """

    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))
print(len(df))

print(len(forecast))



# 调用上面的函数，把预测结果和原始样本join起来

cmp_df = make_comparison_dataframe(df, forecast)

cmp_df.tail(n=3)
# 用来计算MAP，MAPE的函数

def calculate_forecast_errors(df, prediction_size):

    """Calculate MAPE and MAE of the forecast.

    

       Args:

           df: joined dataset with 'y' and 'yhat' columns.

           prediction_size: number of days at the end to predict.

    """

    

    # Make a copy

    df = df.copy()

    

    # Now we calculate the values of e_i and p_i according to the formulas given in the article above.

    df['e'] = df['y'] - df['yhat']

    df['p'] = 100 * df['e'] / df['y']

    

    # Recall that we held out the values of the last `prediction_size` days

    # in order to predict them and measure the quality of the model. 

    

    # Now cut out the part of the data which we made our prediction for.

    predicted_part = df[-prediction_size:]

    

    # Define the function that averages absolute error values over the predicted part.

    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))

    

    # Now we can calculate MAPE and MAE and return the resulting dictionary of errors.

    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}
# 计算MAE，MAPE

for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():

    print(err_name, err_value)
def show_forecast(cmp_df, num_predictions, num_values, title):

    """Visualize the forecast."""

    

    def create_go(name, column, num, **kwargs):

        points = cmp_df.tail(num)

        args = dict(name=name, x=points.index, y=points[column], mode='lines')

        args.update(kwargs)

        return go.Scatter(**args)

    

    lower_bound = create_go('Lower Bound', 'yhat_lower', num_predictions,

                            line=dict(width=0),

                            marker=dict(color="gray"))

    upper_bound = create_go('Upper Bound', 'yhat_upper', num_predictions,

                            line=dict(width=0),

                            marker=dict(color="gray"),

                            fillcolor='rgba(68, 68, 68, 0.3)', 

                            fill='tonexty')

    forecast = create_go('Forecast', 'yhat', num_predictions,

                         line=dict(color='rgb(31, 119, 180)'))

    actual = create_go('Actual', 'y', num_values,

                       marker=dict(color="red"))

    

    # In this case the order of the series is important because of the filling

    data = [lower_bound, upper_bound, forecast, actual]



    layout = go.Layout(yaxis=dict(title='Posts'), title=title, showlegend = False)

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, show_link=False)



show_forecast(cmp_df, prediction_size, 100, 'New posts on Medium')
# 定义逆转换函数

def inverse_boxcox(y, lambda_):

    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)
# 构造一份新的数据集

train_df2 = train_df.copy().set_index('ds')
# 做boxcox转换

train_df2['y'], lambda_prophet = stats.boxcox(train_df2['y'])

train_df2.reset_index(inplace=True)
# 从新训练模型，并做预测

m2 = Prophet()

m2.fit(train_df2)

future2   = m2.make_future_dataframe(periods=prediction_size) #构造用于存放预测结果的DataFrame

forecast2 = m2.predict(future2)
# 将预测结果逆转换，得到对应的不经转换的预测值

for column in ['yhat', 'yhat_lower', 'yhat_upper']:

    forecast2[column] = inverse_boxcox(forecast2[column], lambda_prophet)
# 评估MAPE，MAE

cmp_df2 = make_comparison_dataframe(df, forecast2)

for err_name, err_value in calculate_forecast_errors(cmp_df2, prediction_size).items():

    print(err_name, err_value)
# 分别可视化不使用box-cox转换，以及使用box-cox转换的预测结果

show_forecast(cmp_df, prediction_size, 100, 'No transformations')

show_forecast(cmp_df2, prediction_size, 100, 'Box–Cox transformation')