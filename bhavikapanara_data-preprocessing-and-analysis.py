import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
df = pd.read_csv('../input/restaurant-and-market-health-violations.csv')

print ('Size of the health-violance data : ' , df.shape)

df.head()
df.info()
cat_col = df.select_dtypes(include = 'object').columns.tolist()
num_col = df.select_dtypes(exclude='object').columns.tolist()

print ('categorical feature :', cat_col)
print ('\nnumeric feature :' ,num_col)
print ('\nnumber of categorical feature : ' , len(cat_col))
print ('\nnumber of numeric feature : ' , len(num_col))
df.describe(include=["O"])
def check_missing(df):
    
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
    

missing_data_df = check_missing(df)
missing_data_df.head()

def find_uni(df):
    col_list = df.columns
    redundant_col =[]
    for col in col_list:
        if df[col].nunique() == 1:
            redundant_col.append(col)
    return redundant_col


redundant_col = find_uni(df)
print ('Number of redundant features in data :',len(redundant_col))
print ('Redundant Feature :', redundant_col)
df.drop(redundant_col,axis=1,inplace =True)
df['service_description'].value_counts()
df['service_code'].value_counts()
df.drop('service_code' , axis =1 , inplace=True)

df['program_element_pe'].value_counts()
df['pe_description'].value_counts()
df.drop('program_element_pe' , axis =1, inplace =True)
df[['score','points',]].describe()
df['score'].hist().plot()
df['points'].value_counts()
temp = df['grade'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='grade')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
le = LabelEncoder()
df['grade'] = le.fit_transform(df['grade'])
df['grade'].corr(df['score'])
top_violated_place = df["facility_name"].value_counts().head(15)
pd.DataFrame({'Count':top_violated_place.values},index = top_violated_place.index)

temp = df["facility_name"].value_counts().head(25)

trace = go.Bar(
    x = temp.index,
    y = temp.values,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of facility_name",
    xaxis=dict(
        title='facility_name',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='how many times health-violations occur',
        titlefont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='facility_name')
temp1 = df[['facility_name','score']].sort_values(['score'],ascending = False).drop_duplicates()
temp1.head(10)
temp1 = df[['facility_name','score']].sort_values(['score']).drop_duplicates()
temp1.head(10)

