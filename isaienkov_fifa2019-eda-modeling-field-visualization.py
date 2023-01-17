import numpy as np

import pandas as pd

import math

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

import plotly.express as px

from sklearn.metrics import mean_squared_error as mse

from iso3166 import countries

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
SCATTER_SIZE = 800
df = pd.read_csv('/kaggle/input/fifa19/data.csv')

df.head()
missed = pd.DataFrame()

missed['column'] = df.columns



missed['percent'] = [round(100* df[col].isnull().sum() / len(df), 2) for col in df.columns]

missed = missed[missed['percent']>0].sort_values('percent')



fig = px.bar(

    missed, 

    x='percent',

    y="column", 

    orientation='h', 

    title='Missed values percent for every column (percent > 0)', 

    height=1300, 

    width=800

)



fig.show()
data = df['Club'].value_counts().reset_index()

data.columns = ['club', 'count']

data = data.sort_values('count')



fig = px.bar(

    data.tail(50), 

    x='count',

    y="club", 

    orientation='h', 

    title='Top 50 teams by number of players', 

    height=900, 

    width=800

)



fig.show()
data = df['Club'].value_counts().reset_index()

data.columns = ['club', 'count']

data = data.sort_values('count')



fig = px.bar(

    data.head(50), 

    x='count',

    y="club", 

    orientation='h', 

    title='Top 50 teams with less number of players', 

    height=900, 

    width=800

)



fig.show()
df.describe()
def plot_bar_plot(data, categorical_feature, target_feature, orientation, title, top_records=None, sort=False):

    data = data.groupby(categorical_feature)[target_feature].count().reset_index()

    fig = px.bar(

        data, 

        x=categorical_feature, 

        y=target_feature, 

        orientation=orientation, 

        title=title,

        height=600,

        width=800

    )

    fig.show()

    

def plot_pie_count(data, field="Nationality", percent_limit=0.5, title="Number of players by "):

    

    title += field

    data[field] = data[field].fillna('NA')

    data = data[field].value_counts().to_frame()



    total = data[field].sum()

    data['percentage'] = 100 * data[field]/total    



    percent_limit = percent_limit

    otherdata = data[data['percentage'] < percent_limit] 

    others = otherdata['percentage'].sum()  

    maindata = data[data['percentage'] >= percent_limit]



    data = maindata

    other_label = "Others(<" + str(percent_limit) + "% each)"

    data.loc[other_label] = pd.Series({field:otherdata[field].sum()}) 

    

    labels = data.index.tolist()   

    datavals = data[field].tolist()

    

    trace=go.Pie(

        labels=labels,

        values=datavals

    )



    layout = go.Layout(

        title = title,

        height=500,

        width=800

        )

    

    fig = go.Figure(data=[trace], layout=layout)

    iplot(fig)
plot_bar_plot(

    df, 

    'Position', 

    'Value', 

    'v', 

    'Number of players by position'

)
df[df['Position']=='ST'].head(10)
categorical = ['Nationality', 'Club', 'Preferred Foot', 'Work Rate', 'Body Type', 'Position']
plot_pie_count(df, 'Nationality')

plot_pie_count(df, 'Preferred Foot')

plot_pie_count(df, 'Work Rate', 0.1)

plot_pie_count(df, 'Body Type', 0.1)
df['Value'] = df['Value'].str.replace('€','').str.replace('M',' 1000000').str.replace('K',' 1000')

df['Value'] = df['Value'].str.split(' ', expand=True)[0].astype(float) * df['Value'].str.split(' ', expand=True)[1].astype(float)

df['Value'] = df['Value'].fillna(0).astype(np.float32)
body_dict = {

    'PLAYER_BODY_TYPE_25' : np.nan,

    'Messi' : np.nan,

    'Shaqiri': np.nan,

    'Neymar': np.nan,

    'Akinfenwa': np.nan,

    'C. Ronaldo': np.nan,

    'Courtois': np.nan

}



df['Body Type'] = df['Body Type'].replace(body_dict)

df['Height'] = df['Height'].str.replace("'",".").astype(float)

df['Weight'] = df['Weight'].str.replace("lbs","").astype(float)
fig = px.scatter(

    df, 

    x='Overall', 

    y='Value', 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for Value and Overall' 

)



fig.show()
df.sort_values("Value", ascending=False)[['Name', "Age", "Value", "Overall"]].head(20)
fig = px.scatter(

    df, 

    x='Age', 

    y='Overall', 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for Age and Overall' 

)



fig.show()
fig = px.scatter(

    df, 

    x='Age', 

    y='Potential', 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for Age and Potential' 

)



fig.show()
fig = px.scatter(

    df, 

    x='Potential', 

    y='Overall', 

    height=SCATTER_SIZE,

    width=SCATTER_SIZE,

    title='Scatter plot for Potential and Overall' 

)



fig.show()
df.sort_values("Potential", ascending=False)[['Name', "Age", "Value", "Overall", 'Potential']].head(20)
fig = px.histogram(

    df, 

    "Value", 

    nbins=100, 

    title='Value distribution',

    width=800,

    height=600

)



fig.show()
age = df.groupby('Age')['Value'].mean().reset_index()

fig = px.bar(

    age, 

    x="Age", 

    y="Value", 

    orientation='v', 

    title='Mean Value by Age'

)



fig.show()
df.sort_values("Age", ascending=False)[['Name', "Age", "Value", "Overall"]].head(20)
fig = px.histogram(df, "Age", nbins=50, title='Age distribution')

fig.show()
club = df.groupby('Club')['Value'].mean().reset_index().sort_values('Value', ascending=True).tail(50)



fig = px.bar(

    club, 

    x="Value", 

    y="Club", 

    orientation='h', 

    width=800, 

    height=900

)



fig.show()
club = df.groupby('Club')['Overall'].mean().reset_index().sort_values('Overall', ascending=True).tail(50)



fig = px.bar(

    club, 

    x="Overall", 

    y="Club", 

    orientation='h',

    title="Top 50 teams with highest player's average Overall rating",

    width=800,

    height=900

)



fig.show()
club = df.groupby('Nationality')['Overall'].max().reset_index().sort_values('Overall', ascending=True).tail(40)



fig = px.bar(

    club, 

    x="Overall", 

    y="Nationality", 

    orientation='h',

    width=800,

    height=800

)



fig.show()
club = df.groupby('Nationality')['Overall'].mean().reset_index().sort_values('Overall', ascending=True).tail(40)



fig = px.bar(

    club, 

    x="Overall", 

    y="Nationality", 

    orientation='h', 

    title="Top 40 countries with highest player's average Overall rating",

    width=800,

    height=800

)



fig.show()
club = df.groupby('Nationality')['Value'].mean().reset_index().sort_values('Value', ascending=True).tail(40)



fig = px.bar(

    club, 

    x="Value", 

    y="Nationality", 

    orientation='h', 

    title="Top 40 countries with highest player's average Value",

    width=800,

    height=800

)



fig.show()
country_dict = {}

for c in countries:

    country_dict[c.name] = c.alpha3

    

df['alpha3'] = df['Nationality']

df = df.replace({"alpha3": country_dict})



gbr = ['England', 'Wales', 'Scotland', 'Northern Ireland']



df.loc[df['Nationality'].isin(gbr), 'alpha3'] = 'GBR'

df.loc[df['Nationality'] == 'Bosnia Herzegovina', 'alpha3'] = 'BIH'

df.loc[df['Nationality'] == 'Korea Republic', 'alpha3'] = 'KOR'

df.loc[df['Nationality'] == 'Czech Republic', 'alpha3'] = 'CZE'

df.loc[df['Nationality'] == 'St Lucia', 'alpha3'] = 'LCA'

df.loc[df['Nationality'] == 'Palestine', 'alpha3'] = 'PSE'

df.loc[df['Nationality'] == 'Antigua & Barbuda', 'alpha3'] = 'ATG'

df.loc[df['Nationality'] == 'St Kitts Nevis', 'alpha3'] = 'KNA'

df.loc[df['Nationality'] == 'Korea DPR', 'alpha3'] = 'PRK'

df.loc[df['Nationality'] == 'São Tomé & Príncipe', 'alpha3'] = 'STP'

df.loc[df['Nationality'] == 'Trinidad & Tobago', 'alpha3'] = 'TTO'

df.loc[df['Nationality'] == 'Bolivia', 'alpha3'] = 'BOL'

df.loc[df['Nationality'] == 'Moldova', 'alpha3'] = 'MDA'

df.loc[df['Nationality'] == 'Curacao', 'alpha3'] = 'CUW'

df.loc[df['Nationality'] == 'Tanzania', 'alpha3'] = 'TZA'

df.loc[df['Nationality'] == 'Guinea Bissau', 'alpha3'] = 'GNB'

df.loc[df['Nationality'] == 'China PR', 'alpha3'] = 'CHN'

df.loc[df['Nationality'] == 'FYR Macedonia', 'alpha3'] = 'MKD'

df.loc[df['Nationality'] == 'Iran', 'alpha3'] = 'IRN'

df.loc[df['Nationality'] == 'Syria', 'alpha3'] = 'SYR'

df.loc[df['Nationality'] == 'Cape Verde', 'alpha3'] = 'CPV'

df.loc[df['Nationality'] == 'United States', 'alpha3'] = 'USA'

df.loc[df['Nationality'] == 'Republic of Ireland', 'alpha3'] = 'IRL'

df.loc[df['Nationality'] == 'Venezuela', 'alpha3'] = 'VEN'

df.loc[df['Nationality'] == 'Russia', 'alpha3'] = 'RUS'

df.loc[df['Nationality'] == 'Ivory Coast', 'alpha3'] = 'CIV'

df.loc[df['Nationality'] == 'DR Congo', 'alpha3'] = 'COD'

df.loc[df['Nationality'] == 'Central African Rep.', 'alpha3'] = 'CAF'
data = df.groupby(['alpha3', 'Nationality'])['Name'].count().reset_index()

data.columns = ['alpha3', 'nationality', 'count']



fig = px.choropleth(

    data, 

    locations="alpha3",

    hover_name='nationality',

    color='count',

    projection="natural earth",

    color_continuous_scale=px.colors.sequential.Plasma,

    title='Number of players from every country',

    width=800, 

    height=700

)



fig.show()
data = df.groupby(['alpha3', 'Nationality'])['Overall'].max().reset_index()

data.columns = ['alpha3', 'nationality', 'max_rating']



fig = px.choropleth(

    data, 

    locations="alpha3",

    hover_name='nationality',

    color="max_rating",

    projection="natural earth",

    color_continuous_scale=px.colors.sequential.Plasma,

    title='Max rating for every country',

    width=800, 

    height=700

)



fig.show()
data = df.groupby(['alpha3', 'Nationality'])['Age'].max().reset_index()

data.columns = ['alpha3', 'nationality', 'max_age']



fig = px.choropleth(

    data, 

    locations="alpha3",

    hover_name='nationality',

    color="max_age",

    projection="natural earth",

    color_continuous_scale=px.colors.sequential.Plasma,

    title='Max age of sportsman for every country',

    width=800, 

    height=700

)



fig.show()
data = df.groupby(['alpha3', 'Nationality'])['Value'].max().reset_index()

data.columns = ['alpha3', 'nationality', 'max_value']



fig = px.choropleth(

    data, 

    locations="alpha3",

    hover_name='nationality',

    color="max_value",

    projection="natural earth",

    color_continuous_scale=px.colors.sequential.Plasma,

    title='Max Value of sportsman for every country',

    width=800, 

    height=700

)



fig.show()
data = df['alpha3'].value_counts().reset_index()

data.columns=['alpha3', 'national_count']

df = pd.merge(df, data, on='alpha3')

data = df[df['national_count']>=50]

df = df.drop(['national_count'], axis=1)

data = data.groupby(['alpha3', 'Nationality'])['Overall'].mean().reset_index()

data.columns = ['alpha3', 'nationality', 'mean_rating']



fig = px.choropleth(

    data, 

    locations="alpha3",

    hover_name='nationality',

    color="mean_rating",

    projection="natural earth",

    color_continuous_scale=px.colors.sequential.Plasma,

    title='Mean rating for sportsmen for every country (minimum 50 players)',

    width=800, 

    height=700

)



fig.show()
def draw_pitch(pitch, line, orientation,view):

    

    orientation = orientation

    view = view

    line = line

    pitch = pitch

    

    if view.lower().startswith("h"):

        fig,ax = plt.subplots(figsize=(20.8, 13.6))

        plt.ylim(98, 210)

        plt.xlim(-2, 138)

    else:

        fig,ax = plt.subplots(figsize=(13.6, 20.8))

        plt.ylim(-2, 210)

        plt.xlim(-2, 138)

    ax.axis('off')



    # side and goal lines #

    lx1 = [0, 0, 136, 136, 0]

    ly1 = [0, 208, 208, 0, 0]



    plt.plot(lx1,ly1,color=line,zorder=5)



    # boxes, 6 yard box and goals

        #outer boxes#

    lx2 = [27.68, 27.68, 108.32, 108.32] 

    ly2 = [208, 175, 175, 208]

    plt.plot(lx2,ly2,color=line,zorder=5)



    lx3 = [27.68, 27.68, 108.32, 108.32] 

    ly3 = [0, 33, 33, 0]

    plt.plot(lx3,ly3,color=line,zorder=5)



        #goals#

    lx4 = [60.68, 60.68, 75.32, 75.32]

    ly4 = [208, 208.4, 208.4, 208]

    plt.plot(lx4,ly4,color=line,zorder=5)



    lx5 = [60.68, 60.68, 75.32, 75.32]

    ly5 = [0, -0.4, -0.4, 0]

    plt.plot(lx5,ly5,color=line,zorder=5)



       #6 yard boxes#

    lx6 = [49.68, 49.68, 86.32, 86.32]

    ly6 = [208, 199, 199, 208]

    plt.plot(lx6,ly6,color=line,zorder=5)



    lx7 = [49.68, 49.68, 86.32, 86.32]

    ly7 = [0, 9, 9, 0]

    plt.plot(lx7,ly7,color=line,zorder=5)



    #Halfway line, penalty spots, and kickoff spot

    lx8 = [0, 136] 

    ly8 = [104, 104]

    plt.plot(lx8,ly8,color=line,zorder=5)



    plt.scatter(68, 186, color=line, zorder=5)

    plt.scatter(68, 22, color=line, zorder=5)

    plt.scatter(68, 104, color=line, zorder=5)



    circle1 = plt.Circle((68, 187), 18.30, ls='solid', lw=3, color=line, fill=False, zorder=1, alpha=1)

    circle2 = plt.Circle((68, 21), 18.30, ls='solid', lw=3, color=line, fill=False, zorder=1, alpha=1)

    circle3 = plt.Circle((68, 104), 18.30, ls='solid', lw=3, color=line, fill=False, zorder=2, alpha=1)



    rec1 = plt.Rectangle((40, 175), 60, 33, ls='-', color=pitch, zorder=1, alpha=1)

    rec2 = plt.Rectangle((40, 0), 60, 33, ls='-', color=pitch, zorder=1, alpha=1)

    rec3 = plt.Rectangle((-1, -1), 140, 212, ls='-', color=pitch, zorder=1, alpha=1)



    ax.add_artist(rec3)

    ax.add_artist(circle1)

    ax.add_artist(circle2)

    ax.add_artist(rec1)

    ax.add_artist(rec2)

    ax.add_artist(circle3)   
draw_pitch("#195905", "#faf0e6", "v", "full")

x = [68, 68, 68, 32, 104, 68, 32, 104, 68, 44, 88, 20, 116, 12, 124, 68, 68, 16, 120, 16, 120, 40, 96, 32, 104, 32, 104]

y = [186, 150, 1, 150, 150, 112, 114, 114, 14, 16, 16, 24, 24, 50, 50, 50, 74, 74, 74, 130, 130, 74, 74, 186, 186, 50, 50]

n = ['ST', 'CF', 'GK', 'LF', 'RF', 'CAM', 'LAM', 'RAM', 'CB', 'LCB', 'RCB', 'LB', 'RB', 'LWB', 'RWB', 'CDM', 'CM', 'LM', 'RM', 'LW', 'RW', 'LCM', 'RCM', 'LS', 'RS', 'LDM', 'RDM']



for i, pos in enumerate(n):

    x_c = x[i]

    y_c = y[i]

    plt.scatter(x_c, y_c, marker='o', color='red', edgecolors="black", zorder=10)

    plt.text(x_c-2.5, y_c+1, pos, fontsize=16)
res = []

for item in n:

    test_df = df[df['Position']==item]

    test_df = test_df.sort_values(['Overall'], ascending=False)

    res.append(test_df.iloc[0]['Name'] + ' (' + str(test_df.iloc[0]['Overall']) + ')')



draw_pitch("#195905", "#faf0e6", "v", "full")



for i, pos in enumerate(res):

    x_c = x[i]

    y_c = y[i]

    plt.scatter(x_c, y_c, marker='o', color='red', edgecolors="black", zorder=10)

    plt.text(x_c-2.5, y_c+1, pos, fontsize=16)
res = []

for item in n:

    test_df = df[df['Position']==item]

    test_df = test_df.sort_values(['Value'], ascending=False)

    res.append(test_df.iloc[0]['Name'])



draw_pitch("#195905","#faf0e6","v","full")



for i, pos in enumerate(res):

    x_c = x[i]

    y_c = y[i]

    plt.scatter(x_c, y_c, marker='o', color='red', edgecolors="black", zorder=10)

    plt.text(x_c-2.5, y_c+1, pos, fontsize=16)
res = []

for item in n:

    test_df = df[(df['Position']==item) & (df['Nationality']=='Ukraine')]

    test_df = test_df.sort_values(['Overall'], ascending=False)

    if len(test_df) > 0:

        res.append(test_df.iloc[0]['Name'] + ' (' + str(test_df.iloc[0]['Overall']) + ')')

    else:

         res.append('NO PLAYER')   



draw_pitch("#195905","#faf0e6","v","full")



for i, pos in enumerate(res):

    x_c = x[i]

    y_c = y[i]

    plt.scatter(x_c, y_c, marker='o', color='red', edgecolors="black", zorder=10)

    plt.text(x_c-2.5, y_c+1, pos, fontsize=16)
res = []

for item in n:

    test_df = df[df['Position']==item]

    test_df = test_df.sort_values(['Potential'], ascending=False)

    res.append(test_df.iloc[0]['Name'] + ' (' + str(test_df.iloc[0]['Overall']) + ')')



draw_pitch("#195905","#faf0e6","v","full")



for i, pos in enumerate(res):

    x_c = x[i]

    y_c = y[i]

    plt.scatter(x_c, y_c, marker='o', color='red', edgecolors="black", zorder=10)

    plt.text(x_c-2.5, y_c+1, pos, fontsize=16)
drop = [

    'Unnamed: 0', 'ID', 'Name', 'Photo', 'Flag', 'Potential', 'Club Logo', 'Special', 

    'Real Face', 'Jersey Number',  'Contract Valid Until',  'Release Clause',

    'Wage', 'Joined', 'Loaned From', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 

    'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 

    'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'

]



df = df.drop(drop, axis=1)
for item in categorical:

    df[item] = df[item].fillna('0') 

    le = LabelEncoder()

    df[item] = le.fit_transform(df[item])
f = plt.figure(figsize=(19, 15))

plt.matshow(df.corr(), fignum=f.number)

plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)

plt.yticks(range(df.shape[1]), df.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
df
df = df.drop(['alpha3'], axis=1)

for col in df.columns:

    if abs(df[col].corr(df['Value'])) < 0.15:

        df = df.drop([col], axis=1)



df.columns
target = np.log1p(df["Value"])

original_target = df['Value']

df = df.drop(['Value'], axis=1)
new_categorical = []

for item in categorical:

    if item in df.columns:

        new_categorical.append(item)

        

categorical = new_categorical
df = df.fillna(-1)
X_embedded = TSNE(n_components=2, random_state=666).fit_transform(df)

X_embedded = pd.DataFrame(X_embedded)
analysis = pd.DataFrame()

analysis['color'] = df['Overall']

analysis['x'] = X_embedded[0]

analysis['y'] = X_embedded[1]



fig = px.scatter(

    analysis, 

    x='x', 

    y='y', 

    color='color',

    height=800,

    width=800,

    title='TSNE for dataset'

)



fig.show()
params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'rmse'},

    'subsample': 0.25,

    'subsample_freq': 1,

    'learning_rate': 0.05,

    'num_leaves': 20,

    'feature_fraction': 0.9

}



folds = 5

seed = 666



kf = KFold(n_splits=folds, shuffle=True, random_state=seed)



models = []

for train_index, val_index in kf.split(df):

    train_X = df[df.columns].iloc[train_index]

    val_X = df[df.columns].iloc[val_index]

    train_y = target.iloc[train_index]

    val_y = target.iloc[val_index]

    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categorical)

    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categorical)

    gbm = lgb.train(

        params,

        lgb_train,

        num_boost_round=10000,

        valid_sets=(lgb_train, lgb_eval),

        early_stopping_rounds=100,

        verbose_eval = 100

    )

    models.append(gbm)
res=sum(np.expm1([model.predict(df) for model in models])/folds)
def root_mean_squared_error(y_true, y_pred):

    return math.sqrt(mse(y_true, y_pred))
print('RMSE: ', root_mean_squared_error(original_target, res))
df['prediction'] = res

df['Value'] = original_target

df[['Value', 'prediction']]
df.to_csv('sub.csv', index=False)