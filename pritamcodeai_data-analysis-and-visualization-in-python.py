#Import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

import seaborn as sns

from sklearn import preprocessing

import matplotlib.font_manager as font_manager

from matplotlib import rcParams

import json

%matplotlib inline

import time 

!pip install python-nvd3



import nvd3

from IPython.display import display, HTML



nvd3.ipynb.initialize_javascript(use_remote=True)
data = pd.read_csv("../input/loan-dataset/loan_payments_data.csv")
data.head()
data.isnull().sum()
print('Percent of missing "paid_off_time" records is %.2f%%' %((data['paid_off_time'].isnull().sum()/data.shape[0])*100))
print('Percent of missing "past_due_days" records is %.2f%%' %((data['past_due_days'].isnull().sum()/data.shape[0])*100))
data['Principal'].unique()
data['education'].unique()
loan_in_list = list(data['loan_status'].unique())

loan_in_list.sort() 

print(loan_in_list)
data.info()
df = data
df.head(n = 10)
df.tail(n=5)
# pandas DataFrame has an immutable ndarray implementing an ordered, sliceable set for pandas objects and this set is known as as  index for the table.



df.index
df.columns
# it shows a quick statistic summary of data

df.describe()
# it shows correlations 

import time

start_time = time.time()

df1 = df.corr() 

print("corr computed in {} seconds".format(time.time()-start_time))

df1
# it shows numerical data ranks 

df.rank() 
df["age"].head()
df[0:6]
df.loc[1:6, ["age", "Principal"]]
# access to a single value

df.at[1, "Gender"]
df[df["age"] == 45]
df[df["age"].isin([33,45])]
df.dropna().head()
df.shape
pd.Series(df["age"])
df.dtypes
type(df)
# By default, axis is equal to 0, and will compute the mean of each column. 

# it can be set to 1 to compute the mean of each row of the numerical values in each row.

df.mean(axis=1)
# df.dropna(how=’all’)



# the second, how=’all’, will drop any row or column where all values are missing. 



# in this topic the argument thresh, drop any rows or columns with a n number of missing values.



df.dropna(axis = 1, how = "all", thresh = 300).head()
# df.dropna(how=’any’)



# The argument how=’any’ is the default and will drop any row(or column) with any missing data.



df.dropna(axis = 1, how = "any", thresh = 300).head()
# statistical analysis of 'principal' column a of df



mean_find = df['Principal'].mean()

print("mean of values found on Principal column = {}".format(mean_find))



sum_find = df['Principal'].sum()

print("sum of values found on Principal column = {}".format(sum_find))



max_find = df['Principal'].max()

print("max of values found on Principal column = {}".format(max_find))



min_find = df['Principal'].min()

print("min of values found on Principal column = {}".format(min_find))



count_find = df['Principal'].count()

print("count of values found on Principal column = {}".format(count_find))



median_find = df['Principal'].median() 

print("median of values found on Principal column  = {}".format(median_find))



std_find = df['Principal'].std() 

print("std of values found on Principal column = {}".format(std_find))



var_find = df['Principal'].var()  

print("variance of values found on Principal column= {}".format(var_find))
# sum of values, grouped by the 'Principal'

print("grouped by sum found for column principal.")

      

df.groupby(['Principal']).sum()
# count of values, grouped by the 'Principal'

print("grouped by count found for column principal.")

      

df.groupby(['Principal']).count()
print("print of first 5 rows and every column.")

df.iloc[0:5,:]



print("print of entire rows and columns.")

df.iloc[:,:]



print("print from 5th rows and first 5 columns.")

df.iloc[5:,:5]



print("Print of first 5 rows of age column.")

df.loc[:5,["age"]]
data_preview = data.copy()

label_unique = data['loan_status'].unique()

label_occurance_count = data_preview.groupby('loan_status').size()

plt.pie(label_occurance_count, labels = label_occurance_count,counterclock=False, shadow=True, radius = 2, autopct='%1.1f%%', labeldistance = 1.1)

plt.title('loan status types as percentage in a graph view', y=1.5, bbox={'facecolor':'#EBF1DE', 'pad':18})

plt.legend(label_unique,loc="top right", bbox_to_anchor=(1.36,1.26))

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75)

plt.show()

time.sleep(26)
new_data = data.copy()
le=preprocessing.LabelEncoder()

data['loan_status']=le.fit_transform(data['loan_status'])

data['Gender']=le.fit_transform(data['Gender'])

data['education']=le.fit_transform(data['education'])

data['past_due_days']=le.fit_transform(data['past_due_days'])
data.head()
data.info()
new_data.groupby('loan_status')['Gender'].agg(['count'])
sns.barplot(x="Gender", y="loan_status", hue="education", data=data);
data['past_due_days'].unique()
x=data.groupby('Gender')['past_due_days'].agg(['sum'])

x=pd.DataFrame(x)

x
data['loan_status']
from nvd3 import scatterChart



loan_status1 = list(data['loan_status'])
extra_serie = {"tooltip": {"y_start": "", "y_end": " min"}}
chart = scatterChart(name='scatterChart1', height=400, width=1000)





Gender1 = list(data['Gender'])



kwargs = {'shape': 'circle', 'size': '30'}



chart.add_serie(name="loan_status to gender", y=Gender1, x=loan_status1, extra=extra_serie, **kwargs)



chart.buildhtml()

chart_html = chart.htmlcontent



display(HTML(chart_html))



time.sleep(26) 
chart = scatterChart(name='scatterChart2', height=400, width=1000)



education1 = list(data['education'])



kwargs = {'shape': 'cross', 'size': '10'}



chart.add_serie(name="loan_status to education", y=education1, x=loan_status1, extra=extra_serie, **kwargs)



chart.buildhtml()

chart_html = chart.htmlcontent



display(HTML(chart_html))

time.sleep(26) 
from nvd3 import stackedAreaChart

chart = stackedAreaChart(name='stackedAreaChart', height=400, width=1000)



principal_gh = list(data['Principal'])

terms_gh = list(data['terms'])

age_gh = list(data['age'])



extra_serie = {"tooltip": {"y_start": " min", "y_end": " max"}}

chart.add_serie(name="principal to terms", y=terms_gh, x=principal_gh, extra=extra_serie)

chart.add_serie(name="principle to age", y=age_gh, x=principal_gh, extra=extra_serie)

chart.buildhtml()

chart_html = chart.htmlcontent



display(HTML(chart_html))

time.sleep(26)
from nvd3 import multiBarHorizontalChart

chart = multiBarHorizontalChart(name='multiBarHorizontalChart', height=400, width=1000)



chart.add_serie(name="principal to terms", y=terms_gh, x=principal_gh, extra=extra_serie)

chart.add_serie(name="principle to age", y=age_gh, x=principal_gh, extra=extra_serie)



chart.buildcontent()



chart_html = chart.htmlcontent



display(HTML(chart_html))

time.sleep(26)
df10 = new_data.groupby('loan_status')['education'].agg(['count'])

df10
from nvd3 import discreteBarChart

chart = discreteBarChart(name='discreteBarChart1', height=400, width=1000)



xdata = loan_in_list

ydata = list(df10["count"])



chart.add_serie(y=ydata, x=xdata)

chart.buildhtml()



chart_html = chart.htmlcontent

display(HTML(chart_html))

time.sleep(26)
df6 = new_data.groupby('education')['Gender'].agg(['count'])

df6
education_in_list = list(data['education'].unique())

education_in_list.sort() 

print(education_in_list)
print(list(df6["count"]))
from nvd3 import pieChart

chart = pieChart(name='pieChart1', color_category='category20c', height=460, width=600)

xdata = new_data['education']

ydata = list(df6["count"])



chart.add_serie(y=ydata, x=xdata)

chart.buildhtml()



chart_html = chart.htmlcontent

display(HTML(chart_html))

time.sleep(26)
plt.style.use('seaborn-white')

fam_perishable = data.groupby(['education', 'Gender']).size()

font = font_manager.FontProperties(family='Lucida Fax', size=26)

rcParams['font.family'] = 'Britannic Bold'

fam_perishable.unstack().plot(kind='bar',fontsize = 26, stacked=True, colormap= 'coolwarm', figsize=(10,8),  grid=False)

title_font = {'fontname':'Monotype Corsiva'}

plt.title(label = "education to gender graph view", color = "green", fontsize = 82, loc = "center", fontweight = "bold", **title_font)

plt.ylabel('Count of gender in education', fontsize = 26)

plt.xlabel('education', fontsize = 46)

legend = plt.legend(loc = "upper left", labelspacing=1, borderpad=0.6, prop=font)

frame = legend.get_frame()

frame.set_facecolor("#EBF1DE")

frame.set_edgecolor('chartreuse')

frame.set_linewidth(10)

plt.margins(0.36)

plt.grid()

plt.show()

time.sleep(26)
import plotly.graph_objs as go

import pandas as pd

import plotly.offline as offline
df_plotly = new_data.pivot_table(index="loan_status",columns="due_date",values="Principal",aggfunc="sum").fillna(0)
import plotly.io as pio

png_renderer = pio.renderers["iframe"]

png_renderer.width = 1200

png_renderer.height = 600

pio.renderers.default = "iframe"



data10 = []

for index,place in df_plotly.iterrows():    

    trace = go.Bar(x = df_plotly.columns, y = place, name=index)    

    data10.append(trace)

layout = go.Layout(title="due date by loan_id and Principal", showlegend=True, barmode="stack")

figure = go.Figure(data=data10, layout=layout)

#offline.plot(figure)

figure.show()

time.sleep(26)
import plotly.io as pio

pio.renderers
sns.barplot(x='Gender',y='loan_status',data=data)
sns.barplot(x='age',y='loan_status',data=data)
sns.factorplot(x='age',y='loan_status',data=data)
sns.barplot(x='education',y='loan_status',data=data)
sns.countplot(x='Gender',data=data)
processed_analysis_data=data
label = processed_analysis_data.pop('loan_status')
processed_analysis_data.drop('Loan_ID', axis=1, inplace=True)

processed_analysis_data.drop('effective_date', axis=1, inplace=True)

processed_analysis_data.drop('due_date', axis=1, inplace=True)

processed_analysis_data.drop('paid_off_time', axis=1, inplace=True)
processed_analysis_data.head(5)