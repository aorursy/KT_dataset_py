# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools

import plotly.figure_factory as ff
df = pd.read_csv('/kaggle/input/home-credit-default-risk-pk/application_train.csv')
df.shape
df.dtypes.value_counts()
columns_count_dict = (df.count()*100/df.shape[0]).to_dict()   # This dict has column names as keys and % of non-null values in column as values
columns_null_rem = {k: v for k, v in columns_count_dict.items() if v < 55}  
list(columns_null_rem.keys())[:5]
columns_null_rem.pop('EXT_SOURCE_1')
df = df.drop(columns_null_rem.keys(),axis=1)     # Removing columns with more than 45% null values
df.shape
df.dtypes.value_counts()
corr_dict={}

# This dict has column name as keys and it's correlation with TARGET column as values



for i in df.select_dtypes(exclude=['object']).columns:

    corr_dict[i] = df[i].corr(df['TARGET'])
corr_dict.pop('TARGET')
from collections import Counter



k = Counter(corr_dict)

high = k.most_common(3)

print("top high correlated columns and thier correlation") 

for i in high: 

    print(i[0]," :",i[1]," ")

    

print('\n')



low = k.most_common()[:-4:-1]

print("top low correlated columns and thier correlation") 

for i in low: 

    print(i[0]," :",i[1]," ")

df[df.select_dtypes(include='object').columns] = df[df.select_dtypes(include='object').columns].fillna('null')
outlier_meta_df = pd.DataFrame(columns=['column','perc_outlier','mean_outlier','max','min','mean'])

# outlier_meta_df has information about the % of outliers in each column



for i in df.select_dtypes(exclude=['object']).columns:

    mean_col = np.mean(df[i])

    sd_col = np.std(df[i])

    if sd_col !=0:

        true_temp = df[i].apply(lambda x: abs((x-mean_col)/sd_col)>2)

    else:

        true_temp = df[i].apply(lambda x: abs((x-mean_col))>2)

    

    outlier_meta_df = outlier_meta_df.append({'column':i,'perc_outlier':"{0:.2f}".format(np.sum(true_temp)*100/len(df[i]))\

                                              ,'mean_outlier':"{0:.2f}".format(np.mean(df[i][true_temp])),\

                                              'max':"{0:.2f}".format(np.max(df[i])),\

                                              'min':"{0:.2f}".format(np.min(df[i]))\

                                              ,'mean':"{0:.2f}".format(np.mean(df[i]))},ignore_index=True)

outlier_meta_df.head()
df.boxplot(column=['DAYS_EMPLOYED'])
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: np.min(df['DAYS_EMPLOYED']) if x>10000 else x) 

# outliers are replaced with the min value in the column
df.boxplot(column=['DAYS_EMPLOYED'])
df.boxplot(column=['AMT_INCOME_TOTAL'])
df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].apply(lambda x: 2000000 if x>2000000 else x)
df.boxplot(column=['AMT_INCOME_TOTAL'])
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df_pos = df[df['TARGET']==1]

df_neg = df[df['TARGET']==0]



print('There are %0.2f%% positive records in complete data' % (len(df_pos)*100/len(df)))
temp = pd.cut(df['DAYS_EMPLOYED']*-1/365,bins=np.arange(0,40,4))



pos_dict = (temp[df['TARGET']==1].value_counts()).to_dict()

com_dict = (temp.value_counts()).to_dict()



final_dict = {}



for i in pos_dict.keys():

    if com_dict.get(i)!=0:

        final_dict[i] = pos_dict.get(i)*100/com_dict.get(i)

    else:

        final_dict[i] = 0

    

trace = go.Bar(

    x=[(i.left+i.right)/2 for i in list(final_dict.keys()) ],

    y=list(final_dict.values()),

    marker=dict(

        color=list(final_dict.values()),

        colorscale = 'Reds'

    )

)



data = [trace]



layout = go.Layout(title='Employment Experience Distribution',

                   xaxis = go.layout.XAxis(

                        title = 'Employment experience in years', 

                        tickmode = 'linear',

                        tick0 = 0,

                        dtick = 5

                    ),

                   yaxis = dict(title = '% of defaulters for each bin'),

                   width=1000,

                   height=500

                  )



fig = go.Figure(data=data,layout=layout)



fig.show()
temp = pd.cut(df['DAYS_BIRTH']*-1/365,bins=np.arange(20,70,5))



pos_dict = (temp[df['TARGET']==1].value_counts()).to_dict()

com_dict = (temp.value_counts()).to_dict()



final_dict = {}



for i in pos_dict.keys():

    if com_dict.get(i)!=0:

        final_dict[i] = pos_dict.get(i)*100/com_dict.get(i)

    else:

        final_dict[i] = 0

    

trace = go.Bar(

    x=[(i.left+i.right)/2 for i in list(final_dict.keys()) ],

    y=list(final_dict.values()),

    marker=dict(

        color=list(final_dict.values()),

        colorscale = 'Reds'

    )

)



data = [trace]



layout = go.Layout(title='Age Distribution',

                   xaxis = go.layout.XAxis(

                        title = 'Age in years', 

                        tickmode = 'linear',

                        tick0 = 0,

                        dtick = 5

                    ),

                   yaxis = dict(title = '% of defaulters for each bin'),

                   width=1000,

                   height=500

                  )



fig = go.Figure(data=data,layout=layout)



fig.show()



temp = pd.cut(df['AMT_INCOME_TOTAL'],bins=np.arange(25000,325000,25000))



pos_dict = (temp[df['TARGET']==1].value_counts()).to_dict()

com_dict = (temp.value_counts()).to_dict()



final_dict = {}



for i in pos_dict.keys():

    final_dict[i] = pos_dict.get(i)*100/com_dict.get(i)

    

trace = go.Bar(

    x=[(i.left+i.right)/2 for i in list(final_dict.keys()) ],

    y=list(final_dict.values()),

    marker=dict(

        color=list(final_dict.values()),

        colorscale = 'Reds'

    )

)



data = [trace]



# layout = go.layout(title='Income Distribution', 

#                    yaxis = dict(title = '% of defaulters for each bin'),

#                    xaxis = dict(title = 'Income'),

#                    width=1500,

#                    height=500

#                   )



layout = go.Layout(title='Income Distribution',

                   xaxis = dict(title = 'Income'),

                   yaxis = dict(title = '% of defaulters for each bin'),

                   width=1000,

                   height=500

                  )



fig = go.Figure(data=data,layout=layout)



fig.show()
temp = pd.cut(df['AMT_CREDIT'],bins=np.arange(0,1500000,100000))



pos_dict = (temp[df['TARGET']==1].value_counts()).to_dict()

com_dict = (temp.value_counts()).to_dict()



final_dict = {}



for i in pos_dict.keys():

    final_dict[i] = pos_dict.get(i)*100/com_dict.get(i)

    

trace = go.Bar(

    x=[(i.left+i.right)/2 for i in list(final_dict.keys()) ],

    y=list(final_dict.values()),

    marker=dict(

        color=list(final_dict.values()),

        colorscale = 'Reds'

    )

)



data = [trace]



layout = go.Layout(title='Credit Distribution',

                   xaxis = go.layout.XAxis(

                        title = 'Credit Amount', 

                        tickmode = 'linear',

                        tick0 = 0,

                        dtick = 100000

                    ),

                   yaxis = dict(title = '% of defaulters for each bin'),

                   width=1000,

                   height=500

                  )



fig = go.Figure(data=data,layout=layout)



fig.show()


temp = pd.cut(df['AMT_GOODS_PRICE'],bins=np.arange(0,1500000,100000))



pos_dict = (temp[df['TARGET']==1].value_counts()).to_dict()

com_dict = (temp.value_counts()).to_dict()



final_dict = {}



for i in pos_dict.keys():

    final_dict[i] = pos_dict.get(i)*100/com_dict.get(i)

    

trace = go.Bar(

    x=[(i.left+i.right)/2 for i in list(final_dict.keys()) ],

    y=list(final_dict.values()),

    marker=dict(

        color=list(final_dict.values()),

        colorscale = 'Reds'

    )

)



data = [trace]



layout = go.Layout(title='Asset Price Distribution',

                   xaxis = go.layout.XAxis(

                        title = 'Asset Price', 

                        tickmode = 'linear',

                        tick0 = 0,

                        dtick = 100000

                    ),

                   yaxis = dict(title = '% of defaulters for each bin'),

                   width=1000,

                   height=500

                  )



fig = go.Figure(data=data,layout=layout)



fig.show()

col = 'FLAG_OWN_CAR'



x = list(df[col].unique())



temp_dict = {}



for i in x:

#     print(i)

    temp_df = df[df[col]==i]

    temp_dict[i] = len(temp_df[temp_df['TARGET']==1])*100/(len(temp_df))



# x = x.append('total')



temp_dict_2 = {'N':'doesn\'t own car', 'Y':'owns car' }



temp_dict = {temp_dict_2.get(k):v for k,v in sorted(temp_dict.items())}



trace = go.Bar(

    x=list(temp_dict.keys()),

    y=list(temp_dict.values()),

    marker=dict(

        color = ['lightgreen', 'grey']

    )

)



data = [trace]

layout = go.Layout(title='Car Ownership Default Percentage', 

                   yaxis = dict(title = '% of defaulters'),

                   width=500,

                   height=500

                  )



fig = go.Figure(data=data,layout=layout)

fig['layout']['xaxis'].update(dict(title = 'Car ownership status'))



py.iplot(fig)
col = 'FLAG_OWN_REALTY'



x = list(df[col].unique())



temp_dict = {}



for i in x:

#     print(i)

    temp_df = df[df[col]==i]

    temp_dict[i] = len(temp_df[temp_df['TARGET']==1])*100/(len(temp_df))



# x = x.append('total')

temp_dict_2 = {'N':'doesn\'t own house', 'Y':'owns house' }



temp_dict = {temp_dict_2.get(k):v for k,v in sorted(temp_dict.items())}



trace = go.Bar(

    x=list(temp_dict.keys()),

    y=list(temp_dict.values()),

    marker=dict(

        color = ['lightgreen', 'grey']

    )

)



data = [trace]

layout = go.Layout(title='House Ownership Default Percentage', 

                   yaxis = dict(title = '% of defaulters'),

                   width=500,

                   height=500

                  )



fig = go.Figure(data=data,layout=layout)

fig['layout']['xaxis'].update(dict(title = 'House ownership status'))



py.iplot(fig)
# # plt.figure(figsize = (10, 8))



# # sns.kdeplot(df[df['AMT_INCOME_TOTAL']<600000].loc[df['TARGET'] == 0, 'AMT_INCOME_TOTAL'], label = 'target == 0')

# # sns.kdeplot(df[df['AMT_INCOME_TOTAL']<600000].loc[df['TARGET'] == 1, 'AMT_INCOME_TOTAL'], label = 'target == 1')





# # plt.xlabel('Income Amount'); plt.ylabel('Density'); plt.title('Income Distribution');





# x0 =df[df['AMT_INCOME_TOTAL']<400000].loc[df['TARGET'] == 0, 'AMT_INCOME_TOTAL']

# x1 =df[df['AMT_INCOME_TOTAL']<400000].loc[df['TARGET'] == 1, 'AMT_INCOME_TOTAL']



# fig = go.Figure()

# fig.add_trace(go.Histogram(

#     x=x0,

#     histnorm='percent',

#     name='Target == 0', # name used in legend and hover labels

#     xbins=dict( # bins used for histogram

#         start=0,

#         end=400000,

#         size=25000

#     ),

#     marker_color='#EB89B5',

#     opacity=0.75

# ))

# fig.add_trace(go.Histogram(

#     x=x1,

#     histnorm='percent',

#     name='Target == 1',

#     xbins=dict(

#         start=0,

#         end=400000,

#         size=25000

#     ),

#     marker_color='#330C73',

#     opacity=0.75

# ))



# fig.update_layout(

#     title_text='Income Distribution', # title of plot

#     xaxis_title_text='Income amount', # xaxis label

#     yaxis_title_text='Percentage of counts', # yaxis label

#     bargap=0.2, # gap between bars of adjacent location coordinates

#     bargroupgap=0.1 # gap between bars of the same location coordinates

# )



# fig.show()
col = 'NAME_INCOME_TYPE'



x = list(df[col].unique())



temp_dict = {}



for i in x:

#     print(i)

    temp_df = df[df[col]==i]

    temp_dict[i] = len(temp_df[temp_df['TARGET']==1])*100/(len(temp_df))



# x = x.append('total')



temp_dict.pop('Student')

temp_dict.pop('Businessman')



trace = go.Bar(

    x=list(temp_dict.keys()),

    y=list(temp_dict.values()),

    marker=dict(

        color = ['lightgreen', 'grey',  '#D0F9B1', 'khaki', 'aqua','lightgrey']

    )

)



data = [trace]

layout = go.Layout(title='Income type Default Percentage', 

                   yaxis = dict(title = '% of defaulters'),

                   width=500,

                   height=500

                  )



fig = go.Figure(data=data,layout=layout)

fig['layout']['xaxis'].update(dict(title = 'Income type status'))



py.iplot(fig)
   

col = 'NAME_EDUCATION_TYPE'



x = list(df[col].unique())



temp_dict = {}



for i in x:

#     print(i)

    temp_df = df[df[col]==i]

    temp_dict[i] = len(temp_df[temp_df['TARGET']==1])*100/(len(temp_df))



# x = x.append('total')









trace = go.Bar(

    x=list(temp_dict.keys()),

    y=list(temp_dict.values()),

    marker=dict(

        color = ['lightgreen', 'grey',  '#D0F9B1', 'khaki', 'aqua','lightgrey']

    )

)



data = [trace]

layout = go.Layout(title='Eductaion type Default Percentage', 

                   yaxis = dict(title = '% of defaulters'),

                   width=500,

                   height=500

                  )



fig = go.Figure(data=data,layout=layout)

fig['layout']['xaxis'].update(dict(title = 'Education type'))



py.iplot(fig)
col = 'NAME_FAMILY_STATUS'



x = list(df[col].unique())



temp_dict = {}



for i in x:

#     print(i)

    temp_df = df[df[col]==i]

    temp_dict[i] = len(temp_df[temp_df['TARGET']==1])*100/(len(temp_df))



# x = x.append('total')



temp_dict.pop('Unknown')





trace = go.Bar(

    x=list(temp_dict.keys()),

    y=list(temp_dict.values()),

    marker=dict(

        color = ['lightgreen', 'grey',  '#D0F9B1', 'khaki', 'aqua','lightgrey']

    )

)



data = [trace]

layout = go.Layout(title='Family Status Default Percentage', 

                   yaxis = dict(title = '% of defaulters'),

                   width=500,

                   height=500

                  )



fig = go.Figure(data=data,layout=layout)

fig['layout']['xaxis'].update(dict(title = 'Family status'))



py.iplot(fig)
col = 'NAME_HOUSING_TYPE'



x = list(df[col].unique())



temp_dict = {}



for i in x:

#     print(i)

    temp_df = df[df[col]==i]

    temp_dict[i] = len(temp_df[temp_df['TARGET']==1])*100/(len(temp_df))



# x = x.append('total')



trace = go.Bar(

    x=list(temp_dict.keys()),

    y=list(temp_dict.values()),

    marker=dict(

        color = ['lightgreen', 'grey',  '#D0F9B1', 'khaki', 'aqua','lightgrey']

    )

)



data = [trace]

layout = go.Layout(title='Housing Type Default Percentage', 

                   yaxis = dict(title = '% of defaulters'),

                   width=800,

                   height=500

                  )



fig = go.Figure(data=data,layout=layout)

fig['layout']['xaxis'].update(dict(title = 'Housing type'))



py.iplot(fig)
col = 'OCCUPATION_TYPE'



x = list(df[col].unique())



temp_dict = {}



for i in x:

#     print(i)

    temp_df = df[df[col]==i]

    temp_dict[i] = len(temp_df[temp_df['TARGET']==1])*100/(len(temp_df))



# x = x.append('total')



trace = go.Bar(

    x=list(temp_dict.keys()),

    y=list(temp_dict.values()),

    marker=dict(

        color=list(temp_dict.values()),

        colorscale = 'Reds'

    )

)



data = [trace]

layout = go.Layout(title='Occupation Type Default Percentage', 

                   yaxis = dict(title = '% of defaulters'),

                   width=1000,

                   height=500

                  )



fig = go.Figure(data=data,layout=layout)

fig['layout']['xaxis'].update(dict(title = 'Occupation type'))



py.iplot(fig)

    