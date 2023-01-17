import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns
df = pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv')
df.head()
df.describe()
df.info()
df.shape
df.columns
df.columns = ['country','year','status','life_expectancy','adult_morality','infant_deaths','alcohol','percentage_expenditure',

             'hepatitis_b','measles','bmi','under_five_deaths','polio', 'total_expenditure','diphtheria','hiv/aids','gdp',

              'population','thinness 1-19 years','thinness 5-9 years','income_composition_of_resources','schooling']
df.describe()
# FIRST COPYING OUR DATASET



df_copy = df.copy()
# CHECKING OUR DATASET FOR ANY OUTLIIERS BY THE USE OF BOXPLOTS



plt.figure(figsize=(15,10))



for i, column in enumerate(['adult_morality','infant_deaths','bmi','gdp','population','under_five_deaths'],start=1):

    plt.subplot(2,3,i)

    df_copy.boxplot(column)
# ADULT MORALITY RATES LOWER THAN 5T PERCENTILE



morality_less_5_percentile = np.percentile(df_copy['adult_morality'].dropna(),5)

df_copy['adult_morality']=df_copy.apply(lambda x: np.nan if x['adult_morality']<morality_less_5_percentile else x['adult_morality'],axis=1)
# REMOVE INFANT DEATHS OF 0



df_copy['infant_deaths']=df_copy['infant_deaths'].replace(0,np.nan)
# REMOVE THE INVALID BMI



df_copy['bmi']=df_copy.apply(lambda x: np.nan if (x['bmi']<10 or x['bmi']>50) else x['bmi'],axis=1)
# REMO0VE UNDER 5 DEATHS



df_copy['under_five_deaths']=df_copy['under_five_deaths'].replace(0,np.nan)
df.isna().sum()
# Filling the missing values of life_expectancy, adult_morality, bmi, polio, diphtheria, thinness 1-19 years, thinness 5-9 years 

# by their mean as they have very less number of missing values
df['life_expectancy']=df['life_expectancy'].fillna(df['life_expectancy'].mean())

df['adult_morality']=df['adult_morality'].fillna(df['adult_morality'].mean())

df['bmi']=df['bmi'].fillna(df['bmi'].mean())

df['polio']=df['polio'].fillna(df['polio'].mean())

df['diphtheria']=df['diphtheria'].fillna(df['diphtheria'].mean())

df['thinness 1-19 years']=df['thinness 1-19 years'].fillna(df['thinness 1-19 years'].mean())

df['thinness 5-9 years']=df['thinness 5-9 years'].fillna(df['thinness 5-9 years'].mean())
df.isna().sum()
correlation = df.corr()

correlation
plt.figure(figsize=(14,12))

sns.heatmap(correlation , annot = True)
df['group']=pd.cut(df['schooling'],bins=[0,5,10,15,21],labels=['g1','g2','g3','g4'])

df['group'].value_counts()

grouped = df.groupby(df.group)['alcohol'].mean()

grouped
def impute_alcohol(col):

    a=col[0]

    b=col[1]

    if pd.isnull(a):

        if(a<5):

            return 1.57

        elif(5<=a<10):

            return 2.07

        elif(10<=a<15):

            return 4.2

        elif(a>=15):

            return 9.0

    else:

        return a
df['alcohol']=df[['alcohol','schooling']].apply(impute_alcohol,axis=1)

df['alcohol']=df['alcohol'].fillna(df['alcohol'].mean())

print(df['alcohol'].isna().sum())

df = df.drop(['group'],axis=1)
df['group']=pd.cut(df['schooling'],bins=[0,5,10,15,21],labels=['g1','g2','g3','g4'])

df['group'].value_counts()

grouped = df.groupby(df.group)['bmi'].mean()

grouped
def impute_bmi(col):

    m=col[0]

    n=col[1]

    if pd.isnull(m):

        if (n<5):

            return 17.68

        elif(5<=n<10):

            return 21.93

        elif(10<=n<15):

            return 41.18

        elif(n>=15):

            return 52.45

    else:

        return m
df['bmi']=df[['bmi','schooling']].apply(impute_bmi,axis=1)

df['bmi']=df['bmi'].fillna(df['bmi'].mean())

print(df['bmi'].isna().sum())

df = df.drop(['group'],axis=1)
df['group']=pd.cut(df['diphtheria'],bins=[0,10,20,30,40,50,60,70,80,90,100],labels=['g1','g2','g3','g4','g5','g6','g7','g8','g9','g10'])

df['group'].value_counts()

grouped = df.groupby(df.group)['hepatitis_b'].mean()

grouped
def impute_hepatitis(col):

    d=col[0]

    f=col[1]

    if pd.isnull(d):

        if(f<10):

            return 36.56

        elif(10<=f<20):

            return 16.5

        elif(20<=f<30):

            return 27.5

        elif(30<=f<40):

            return 39.89

        elif(40<=f<50):

            return 44.83

        elif(50<=f<60):

            return 53

        elif(60<=f<70):

            return 57.93

        elif(70<=f<80):

            return 66.35

        elif(80<=f<90):

            return 78.84

        elif(f>=90):

            return 91.05

    else:

        return d
df['hepatitis_b']=df[['hepatitis_b','diphtheria']].apply(impute_hepatitis,axis=1)

df['hepatitis_b']=df['hepatitis_b'].fillna(df['hepatitis_b'].mean())

print(df['hepatitis_b'].isna().sum())

df = df.drop(['group'],axis=1)
df['group']=pd.cut(df['life_expectancy'],bins=[35,45,55,65,75,85,95],labels=['g1','g2','g3','g4','g5','g6'])

df['group'].value_counts()

grouped = df.groupby(df.group)['schooling'].mean()

grouped
def impute_schooling(col):

    c=col[0]

    d=col[1]

    if pd.isnull(c):

        if (d<45):

            return 9.13

        elif(45<=d<55):

            return 7.8

        elif(55<=d<65):

            return 8.94

        elif(65<=d<75):

            return 12.3

        elif(75<=d<85):

            return 15.04

        elif(d>=85):

            return 16.74

    else:

        return c
df['schooling']=df[['schooling','life_expectancy']].apply(impute_schooling,axis=1)

df['schooling']=df['schooling'].fillna(df['schooling'].mean())

print(df['schooling'].isna().sum())

df = df.drop(['group'],axis=1)
df['group']=pd.cut(df['percentage_expenditure'],bins=[0,4000,8000,12000,16000,20000],labels=['g1','g2','g3','g4','g5'])

df['group'].value_counts()

grouped = df.groupby(df.group)['gdp'].mean()

grouped
def impute_gdp(col):

    s=col[0]

    t=col[1]

    if pd.isnull(s):

        if (t<4000):

            return 4405.87

        elif(4000<=t<8000):

            return 41249.13

        elif(8000<=t<12000):

            return 54962.398

        elif(12000<=t<16000):

            return 83014.85

        elif(t>=16000):

            return 98694.92

    else:

        return s
df['gdp']=df[['gdp','percentage_expenditure']].apply(impute_gdp,axis=1)

df['gdp']=df['gdp'].fillna(df['gdp'].mean())

print(df['gdp'].isna().sum())

df = df.drop(['group'],axis=1)
df['group']=pd.cut(df['infant_deaths'],bins=[0,400,800,1200,1600,2000],labels=['g1','g2','g3','g4','g5'])

df['group'].value_counts()

grouped = df.groupby(df.group)['population'].mean()

grouped
def impute_population(col):

    i=col[0]

    j=col[1]

    if pd.isnull(i):

        if (j<400):

            return 1.228551e+07

        elif(400<=j<800):

            return 5.975911e+07

        elif(800<=j<1200):

            return 2.810998e+08

        elif(1200<=j<1600):

            return 8.088425e+08

        elif(j>=1600):

            return 5.095718e+07

    else:

        return i
df['population']=df[['population','infant_deaths']].apply(impute_population,axis=1)

df['population']=df['population'].fillna(df['population'].mean())

print(df['population'].isna().sum())

df = df.drop(['group'],axis=1)
df['group']=pd.cut(df['alcohol'],bins=[0,4,8,12,16,20],labels=['g1','g2','g3','g4','g5'])

df['group'].value_counts()

grouped = df.groupby(df.group)['total_expenditure'].mean()

grouped
def impute_total(col):

    y=col[0]

    z=col[1]

    if pd.isnull(y):

        if (z<4):

            return 5.28

        elif(4<=z<8):

            return 6.29

        elif(8<=z<12):

            return 7.17

        elif(12<=z<16):

            return 6.45

        elif(z>=16):

            return 5.38

    else:

        return y
df['total_expenditure']=df[['total_expenditure','alcohol']].apply(impute_total,axis=1)

df['total_expenditure']=df['total_expenditure'].fillna(df['total_expenditure'].mean())

print(df['total_expenditure'].isna().sum())

df = df.drop(['group'],axis=1)
df['group']=pd.cut(df['schooling'],bins=[0,5,10,15,21],labels=['g1','g2','g3','g4'])

df['group'].value_counts()

grouped = df.groupby(df.group)['income_composition_of_resources'].mean()

grouped
def impute_income(col):

    u=col[0]

    v=col[1]

    if pd.isnull(u):

        if (v<5):

            return 0.264

        elif(5<=v<10):

            return 0.43

        elif(10<=v<15):

            return 0.66

        elif(v>=15):

            return 0.845

    else:

        return u
df['income_composition_of_resources']=df[['income_composition_of_resources','schooling']].apply(impute_income,axis=1)

df['income_composition_of_resources']=df['income_composition_of_resources'].fillna(df['income_composition_of_resources'].mean())

print(df['income_composition_of_resources'].isna().sum())

df = df.drop(['group'],axis=1)
df.isna().sum()
status=df['status'].unique()

status=list(status)



fig = px.scatter(data_frame=df,

                x='infant_deaths',

                y='life_expectancy',

                size='adult_morality',

                size_max=10,

                color='status',

                opacity=0.7,

                template='seaborn',

                hover_name=df['country'],

                hover_data=[df['schooling'],df['population'],df['total_expenditure']],

                marginal_x='rug',

                marginal_y='histogram',

                range_color=(0,10),

                color_discrete_map={'Developed':'rgb(225,0,0)',

                                   'Developing':'rgb(0,0,250)'},

                category_orders={'status':['Developed','Developing']},

                height=550,

                width=880

                )



fig.update_layout(title='Infant Deaths vs Life Expectancy',

                 xaxis=dict(

                 title='Infant Deaths',

                 gridcolor='white',

                 gridwidth=2,

                 type='log'

                 ),

                 yaxis=dict(

                 title='Life Expectancy',

                 gridcolor='white',

                 gridwidth=2,

                 type='log'

                 ),

                 paper_bgcolor='rgb(235,235,235)',

                 plot_bgcolor='rgb(243,243,243)'

                 )



fig.show()
status=df['status'].unique()

status=list(status)



fig = px.scatter(data_frame=df,

                x='adult_morality',

                y='life_expectancy',

                size='alcohol',

                size_max=10,

                color='status',

                opacity=0.7,

                template='seaborn',

                hover_name=df['country'],

                hover_data=[df['schooling'],df['population'],df['total_expenditure']],

                marginal_x='rug',

                marginal_y='histogram',

                range_color=(0,10),

                color_discrete_map={'Developed':'rgb(225,0,0)',

                                   'Developing':'rgb(0,0,250)'},

                category_orders={'status':['Developed','Developing']},

                height=550,

                width=880

                )



fig.update_layout(title='Adult Morality vs Life Expectancy',

                 xaxis=dict(

                 title='Adult Morality',

                 gridcolor='white',

                 gridwidth=2,

                 type='log'

                 ),

                 yaxis=dict(

                 title='Life Expectancy',

                 gridcolor='white',

                 gridwidth=2,

                 type='log'

                 ),

                 paper_bgcolor='rgb(235,235,235)',

                 plot_bgcolor='rgb(243,243,243)'

                 )



fig.show()
fig = px.histogram(data_frame=df,

                  x='schooling',

                  color='status',

                  barmode='overlay',

                  marginal='rug',

                  opacity=0.9,

                  hover_name='status',

                  template='seaborn',

                  color_discrete_map=dict(Developed='#26828e',Developing='#cf4446')

                  )



fig.update_layout(title='Schooling In Developed And Developing Countries',

                 xaxis=dict(

                 title='Schooling',

                 gridcolor='white',

                 gridwidth=2

                 ),

                 yaxis=dict(

                 title='Count',

                 gridcolor='white',

                 gridwidth=2

                 ),

                 paper_bgcolor='rgb(230,230,230)',

                 plot_bgcolor='rgb(243,243,243)'

                 )



fig.show()
bins=[]

for i in range(35,90,5):

    bins.append(i)

    

fig = px.histogram(data_frame=df,

                  x='life_expectancy',

                  color='status',

                  marginal='rug',

                  hover_name='status',

                  barmode='group',

                  template='seaborn',

                  color_discrete_map=dict(Developed='#bd3786',Developing='#cf4446'),

                  nbins=11,

                  opacity=0.9,

                  range_x=(35,90)

                  )





fig.update_layout(title='Distribution Of Life Expectancy In Developed And Developing Countries',

                 xaxis=dict(

                 title='Life Expectancy',

                 gridcolor='white',

                 gridwidth=2

                 ),

                 yaxis=dict(

                 title='Count',

                 gridcolor='white',

                 gridwidth=2

                 ),

                 paper_bgcolor='rgb(230,230,230)',

                 plot_bgcolor='rgb(243,243,243)',

                 bargap=0.1,

                 bargroupgap=0.1

                 )



fig.show()
grouped = df.groupby(df['country'])['population'].mean()

grouped = pd.DataFrame(index = df['country'].unique(),data=grouped)

grouped = grouped.sort_values(by='population',ascending=False)

grouped=grouped.head(10)



fig=px.pie(data_frame=grouped,

          names=grouped.index,

          values='population',

          opacity=0.9,

          template='seaborn',

          color_discrete_sequence=px.colors.sequential.Cividis,

          hole=0.5,

          )



fig.update_traces(pull=0.05,textinfo='percent+label',rotation=90)



fig.update_layout(title='Pie Chart',

                 paper_bgcolor='rgb(230,230,230)',

                 plot_bgcolor='rgb(243,243,243)',

                 annotations=[dict(text='Mean Population',showarrow=False,font_size=20)]

                 )



fig.show()
bins=[]

for i in range(35,90,5):

    bins.append(i)

    

    

fig = px.histogram(data_frame=df,

                  x=['thinness 1-19 years','thinness 5-9 years'],

                  opacity=0.8,

                  color_discrete_map={'thinness 1-19 years':'#440f76',

                                     'thinness 5-9 years':'#26828e'},

                  marginal='rug',

                  nbins=9,

                  range_x=(0,30)

                  )



fig.update_layout(title='Relation Between Thinness Columns w.r.t Life Expectancy',

                 xaxis=dict(

                 title='Thinness',

                 gridcolor='white',

                 gridwidth=2

                 ),

                 yaxis=dict(

                 title='Count',

                 gridcolor='white',

                 gridwidth=2

                 ),

                 paper_bgcolor='rgb(230,230,230)',

                 plot_bgcolor='rgb(243,243,243)',

                 bargap=0.1,

                 bargroupgap=0.1

                 )



fig.show()
df['life_type']=pd.cut(df['life_expectancy'],bins=[0,50,65,75,85,100],labels=['Bad','Average','Good','Very Good','Excellent'])



fig=px.scatter(data_frame=df,

              x='hepatitis_b',

              y='life_expectancy',

              color='life_type',

              template='seaborn',

#              color_discrete_sequence=px.colors.sequential.Plasma,

              color_discrete_map={'Bad':'#fc67fd',

                                 'Average':'#35b779',

                                 'Good':px.colors.sequential.Inferno[4],

                                 'Very Good':'#f1605d',

                                 'Excelent':'#bd3786'

                                 },

              log_x=True,

              size_max=15,

              size='alcohol',

              marginal_x='rug',

              marginal_y='histogram',

              hover_name='country',

              animation_frame='year'

              )



fig.update_layout(title='Animated Scatter Plot',

                 xaxis=dict(

                 title='HEPATITIS_b',

                 gridcolor='white',

                 gridwidth=2

                 ),

                 yaxis=dict(

                 title='LIFE EXPECTANCY',

                 gridcolor='white',

                 gridwidth=2

                 ),

                 paper_bgcolor='rgb(230,230,230)',

                 plot_bgcolor='rgb(243,243,243)'

                 )



fig.show()
df['schooling_type']=pd.cut(df['schooling'],bins=[-1,5,10,15,22],labels=['Bad','Good','Very Good','Excellent'])



fig = px.box(data_frame=df,

            x='schooling_type',

            y='income_composition_of_resources',

            color='status',

            points='suspectedoutliers',

            category_orders=dict(schooling_type=['Bad','Good','Very Good','Excellent']),

            template='seaborn',

            hover_name='status',

            animation_frame='year',

            boxmode='group'

            )



fig.update_layout(title='Animated Box Plot',

                 xaxis=dict(

                 title='Schooling Type',

                 gridcolor='white',

                 gridwidth=2

                 ),

                 yaxis=dict(

                 title='Income Composition Of Resources',

                 gridcolor='white',

                 gridwidth=2

                 ),

                 paper_bgcolor='rgb(230,230,230)',

                 plot_bgcolor='rgb(243,243,243)'

                 )



fig.show()
gdf=df.groupby(df['year'])['hiv/aids'].max()

ha=pd.DataFrame(columns=df.columns)

for i in gdf:

    a=df[df['hiv/aids']==i]

    ha=ha.append(a)

    

ha=ha.drop_duplicates(subset=['hiv/aids']) 

ha=ha.sort_values(by='year',ascending=False)



fig=px.sunburst(data_frame=ha,

               path=['year','country'],

               values='hiv/aids',

               color='measles',

               template='seaborn',

               color_discrete_sequence=px.colors.sequential.Viridis,

               color_continuous_scale=px.colors.sequential.Viridis)



fig.update_layout(title='Sunburst Plot',

                 paper_bgcolor='rgb(230,230,230)',

                 plot_bgcolor='rgb(243,243,243)',

                 )



fig.update_traces(branchvalues='total')



fig.show()
