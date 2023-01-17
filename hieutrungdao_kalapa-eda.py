import re
import traceback
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import cufflinks as cf
cf.go_offline()

import warnings
warnings.filterwarnings("ignore")
from multiprocessing import cpu_count
%matplotlib inline
def get_len(x):
    if x is np.nan:
        return 0
    else:
        return len(x)
def get_unknown_category(train, test, cat=cat_field):
    result = pd.DataFrame(columns = ['Feature name', 'Train unique', 'Test unique', 'Unknown value'])
    train_unique = train[cat].stack().reset_index(level=0, drop=True)
    train_unique = train_unique.groupby(level=0).unique().reindex(index=train[cat].columns)
    test_unique = test[cat].stack().reset_index(level=0, drop=True)
    test_unique = test_unique.groupby(level=0).unique().reindex(index=test[cat].columns)
    for index, feat in enumerate(cat):
        a = test_unique.loc[feat]
        u1 = train_unique.loc[feat]
        u2 = test_unique.loc[feat]
        unknown = np.where(~np.in1d(u2, u1))[0]
        result.loc[index] = [feat, get_len(u1), get_len(u2), get_len(unknown)]
    return result
train = pd.read_csv('../input/kalapas/train.csv')
train = train.drop(1311)
test = pd.read_csv('../input/kalapas/test.csv')
test_id = test['id']
df = pd.concat([train, test])
print('Data shape:',df.shape)
time_field = [
    'Field_1', 'Field_2', 'Field_3', 'Field_5', 'Field_6', 'Field_7','Field_8',
    'Field_9', 'Field_11', 'Field_15', 'Field_25', 'Field_32', 'Field_33',
    'Field_35', 'Field_40', 'Field_43', 'Field_44', 'ngaySinh',
    'F_startDate', 'F_endDate', 'E_startDate', 'E_endDate', 'C_startDate',
    'C_endDate', 'G_startDate', 'G_endDate', 'A_startDate', 'A_endDate'
    ]

cat_field = [
    'Field_4',
    'Field_18', 'Field_12', 'Field_34', 'Field_36', 'Field_38', 'Field_45', 'Field_46',
    'Field_47', 'Field_48', 'Field_49', 'Field_54', 'Field_55', 'Field_56', 'Field_61', 'Field_62',
    'Field_65', 'Field_66', 'Field_68', 'gioiTinh', 'diaChi', 'maCv', 
    'info_social_sex', 'data.basic_info.locale', 'currentLocationCity',
    'currentLocationCountry', 'currentLocationName', 'currentLocationState',
    'homeTownCity', 'homeTownCountry', 'homeTownName', 'homeTownState', 'brief'
    ]

num_field = [col for col in df.columns if col not in time_field+cat_field]
print(num_field)
result = get_unknown_category(train.astype(str), test.astype(str))
for field in time_field:
    df[field] = pd.to_datetime(df[field])
df.describe(include='all')
df.head()
# checking missing data
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
temp = df["label"].value_counts()
label_df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
label_df.iplot(kind='pie',labels='labels',values='values', title='Label')
result
train_df = train
test_df = test
for col in cat_field:
    lb = LabelEncoder()
    lb.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train_df[col] = lb.transform(list(train[col].values.astype('str')))
    test_df[col] = lb.transform(list(test[col].values.astype('str')))
for col in time_field:
    train_df[col] = pd.to_datetime(train_df[col]).map(datetime.datetime.toordinal)
    test_df[col] = pd.to_datetime(test_df[col]).map(datetime.datetime.toordinal)
train_df.fillna(-999, inplace = True)
data = [
    go.Heatmap(
        z= train_df.corr().values,
        x= train_df.columns.values,
        y= train_df.columns.values,
        colorscale='Viridis',
        reversescale = False,
        opacity = 1.0 )
]

layout = go.Layout(
    title='Correlation of features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 700,
margin=dict(
    l=240,
),)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')
rf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=42)
rf.fit(train_df.drop(['id', 'label'],axis=1), train_df.label)
features = train_df.drop(['id', 'label'],axis=1).columns.values
x, y = (list(x) for x in zip(*sorted(zip(rf.feature_importances_, features), 
                                                            reverse = False)))
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Viridis',
        reversescale = True
    ),
    name='Random Forest Feature importance',
    orientation='h',
)

layout = dict(
    title='Barplot of Feature importances',
     width = 900, height = 2000,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
#         domain=[0, 0.85],
    ),
    margin=dict(
    l=300,
),
)

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')
for field in num_field:
    try:
        plt.figure(figsize=(12,5))
        plt.title("Distribution of " + field)
        ax = sns.distplot(train[field])
    except:
        plt.figure(figsize=(12,5))
        plt.title("Distribution of "+ field)
        ax = sns.distplot(train[field], kde=False)
def visual_percentage(field):
    temp = train[field].value_counts()
    fig = {
      "data": [
        {
          "values": temp.values,
          "labels": temp.index,
          "domain": {"x": [0, .48]},
          #"name": "Types of Loans",
          #"hoverinfo":"label+percent+name",
          "hole": .7,
          "type": "pie"
        },

        ],
      "layout": {
            "annotations": [
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": field,
                    "x": 0.17,
                    "y": 0.5
                }

            ]
        }
    }
    iplot(fig, filename='donut')
visual_percentage('Field_4')
visual_percentage('Field_12')
visual_percentage('Field_47')
visual_percentage('Field_61')
visual_percentage('Field_62')
visual_percentage('Field_65')
visual_percentage('Field_66')
visual_percentage('info_social_sex')
visual_percentage('brief')
