import pandas as pd

import numpy as np

import datetime as dt

import plotly.offline as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

from plotly import tools

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn import linear_model

from sklearn.cluster import KMeans

import statsmodels.api as sm

import utils

from itertools import cycle, islice

from pandas.tools.plotting import parallel_coordinates

import collections



from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from math import sqrt



from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import AgglomerativeClustering

from matplotlib import pyplot as plt
py.init_notebook_mode(connected=True)

#Read in MA town size data 

town_size = pd.read_csv('../input/massachusetts-town-size-2010/MA_town_size.csv', sep=',')

#remove NA columns, set column headers

town_size.drop(town_size.columns[0],axis=1,inplace=True)

town_size.drop(town_size.index[:5],axis=0,inplace=True)

town_size.columns = town_size.iloc[0]

town_size.drop(town_size.index[0],axis=0,inplace=True)

town_size.columns.values[1]='Pop_per_sqm'

#convert object to integer

town_size['Pop_per_sqm']=town_size['Pop_per_sqm'].astype(int)

town_size.head()
# A histogram showed that 1,000 per sqm was a good cutoff for urban/suburban

town_size['rural?']=(town_size['Pop_per_sqm']<1000).astype(int)

percentage = town_size['rural?'].sum()/town_size['rural?'].count()

print(f'Percentage of towns classified as suburban or rural (i.e. not urban) {percentage:.0%}')
# Read school data

schoolDf = pd.read_csv('../input/massachusetts-public-schools-data/MA_Public_Schools_2017.csv', sep=',',header=0)

# merge in town size flag

schoolDf=pd.merge(schoolDf,town_size[['Community','rural?']], left_on=['Town'],right_on=['Community'],how='left')

#select schools covering grades 9-12

TypeOfSchool=('09,10,11,12')

schoolDf['HigherEd?']=schoolDf['Grade'].str.contains(TypeOfSchool)

no_high_schools = schoolDf['HigherEd?'].sum()

print(f'Number of schools covering grades 9-12 =  {no_high_schools}')
Descriptors=[

'% First Language Not English',

'% Students With Disabilities',

'% High Needs',

'% Economically Disadvantaged',

'% African American',

'% Asian',

'% Hispanic',

'% White',

'% Multi-Race, Non-Hispanic',

'% Females',

'rural?']
Intervention=[ 'Total # of Classes', 

               'Average Class Size',

               'Number of Students', 

               'Average Salary', 

               'Average Expenditures per Pupil']
Outcomes=['Average SAT_Reading',

'Average SAT_Writing',

'Average SAT_Math',

'% Dropped Out',

'% Graduated',

'% Attending College',

'% MA Community College']
FullList=list(set(Outcomes)|set(Descriptors)|set(Intervention)|set(['HigherEd?']))

FullDf=schoolDf[FullList]

FullDf=FullDf[FullDf['HigherEd?']==True]

FullDf=FullDf.dropna()

print(f'Number of schools to analyse ={FullDf.shape[0]}')
FullDf.drop(['HigherEd?'],axis=1,inplace=True)
FullDf.describe().transpose()
def create_heatmap(corr,title):

    layout = go.Layout(

        title=title,

        font=dict(family='Courier New, monospace', size=16, color='#7f7f7f'),

        xaxis=dict(

            #title='x Axis',

            autorange=True,

            showgrid=True,

            zeroline=True,

            showline=True,

            ticks='',

            showticklabels=True

        ),

        yaxis=dict(

            #title='y Axis',

            autorange=True,

            showgrid=True,

            zeroline=True,

            showline=True,

            ticks='',

            showticklabels=True,

            automargin= True

        )

    )

    trace = go.Heatmap(z=corr.values, x=corr.index,y=corr.columns)

    data=[trace]

    fig = go.Figure(data=data, layout=layout)

    return(fig)
#Examine descriptive and intervention variables

DesIn=list(set(Descriptors)|set(Intervention))

DesInDf=FullDf[DesIn]
corr=DesInDf.corr()

layout=create_heatmap(corr,'Correlation of descriptive and intervention factors')

iplot(layout)
#Let's examine the eigenvalues and vectors

corr = np.corrcoef(DesInDf, rowvar=False)

w, v = np.linalg.eig(corr) 

float_formatter = lambda x: "%.1f" % x

np.around(w,decimals =1)
# Let's examine the eigenvectors to locate the colinearity

eigenvectors =v

df = pd.DataFrame(data=eigenvectors,columns=DesInDf.columns, index=DesInDf.columns)

fig=create_heatmap(df,'Eigenvectors of correlation matric <br> (large values indicates colinearity)')

iplot(fig)
trace1 = go.Scatter(

    y = DesInDf['Average Expenditures per Pupil'],

    x = DesInDf['% Multi-Race, Non-Hispanic'],

    mode = 'markers'

)



layout = go.Layout(

    title='Average Expenditures per Pupil vs % multi-Race ',

   font=dict(family='Courier New, monospace', size=16, color='#7f7f7f'),

    yaxis=dict( title='Average Expenditures per Pupil',showline=True,),

    xaxis=dict( title='% Multi-Race, Non-Hispanic',showline=True)

)

data = [trace1]

fig = go.Figure(data=data, layout=layout)

# Plot and embed in ipython notebook!

iplot(fig)  
DescriptorsDf=FullDf[Descriptors]

def plot_dendrogram(Amodel, **kwargs):



    # Children of hierarchical clustering

    children = Amodel.children_



    # Distances between each pair of children

    # Since we don't have this information, we can use a uniform one for plotting

    distance = np.arange(children.shape[0])



    # The number of observations contained in each cluster level

    no_of_observations = np.arange(2, children.shape[0]+2)



    # Create linkage matrix and then plot the dendrogram

    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)



    # Plot the corresponding dendrogram

    dendrogram(linkage_matrix, **kwargs)
Amodel = AgglomerativeClustering(n_clusters=3)



Amodel = Amodel.fit(DescriptorsDf)
plt.title('Hierarchical Clustering Dendrogram')

plot_dendrogram(Amodel=Amodel, labels=Amodel.labels_)

plt.show()
kmeans = KMeans(n_clusters=5,random_state=95)

model = kmeans.fit(DescriptorsDf)

# check size of clusters

Size=collections.Counter(model.labels_)

Size=np.array(list(Size.items()))

Size[:,1]
#add cluser tags to dataframe

FullDf['cluster'] = model.labels_
centers = model.cluster_centers_

# Function that creates a DataFrame with a column for Cluster Number



def pd_centers(featuresUsed, centers):

	colNames = list(featuresUsed)

	colNames.append('prediction')



	# Zip with a column called 'prediction' (index)

	Z = [np.append(A, index) for index, A in enumerate(centers)]



	# Convert to pandas data frame for plotting

	P = pd.DataFrame(Z, columns=colNames)

	P['prediction'] = P['prediction'].astype(int)

	return P



P= pd_centers(DescriptorsDf, centers)

P["CLUSTER SIZE"]=Size[:,1]

pd.options.display.float_format = '{:.0f}'.format

P.transpose()
def RegressionAndPlot(X,y,title,outcome_type):

    Xdata=StandardScaler().fit_transform(X)

    X=pd.DataFrame(data=Xdata,    # values

                  index=X.index,    # 1st column as index

                  columns=X.columns)  # 1st row as the column names

    #Split the Dataset into Training and Test Datasets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=200)

    #Linear Regression: Fit a model to the training set 

    regressor = LinearRegression()

    regressor.fit(X_train, y_train)

    # Calculate R2 score

    y_prediction = regressor.predict(X_test)

    R2= r2_score(y_test,y_prediction)



    RegressionDf=pd.DataFrame(data=regressor.coef_,    # values

                  index=X.columns,    # 1st column as index

                  columns=['Correlation'])  # 1st row as the column names

    RegressionDf.sort_values('Correlation',ascending=True,inplace=True)

    data = [go.Bar(

                x=RegressionDf['Correlation'],

                y=RegressionDf.index,

                orientation = 'h'

    )]

    layout = go.Layout(

        title=f'{title} <br> correlation with {outcome_type} (R2 = {R2:.0%})',

        xaxis=dict(

            title='Correlation',

            autorange=True,

            showgrid=True,

            zeroline=True,

            showline=True,

            ticks='',

            showticklabels=True

        ),

        yaxis=dict(

            #title='y Axis',

            autorange=True,

            showgrid=True,

            zeroline=True,

            showline=True,

            ticks='',

            showticklabels=True,

            automargin= True

        )

    )

    fig = go.Figure(data=data, layout=layout)

    return fig
DesIn=list(set(Descriptors)|set(Intervention))

X=FullDf[DesIn]

y = FullDf['Average SAT_Math']

title="Descriptors and interventions, all schools"

outcome_type = 'Maths SAT score'

fig=RegressionAndPlot(X,y,title,outcome_type)

iplot(fig)

X=FullDf[DesIn]

y = FullDf['% Graduated']

title="Descriptors and interventions, all schools"

outcome_type = 'Graduation rate'

fig=RegressionAndPlot(X,y,title,outcome_type)

iplot(fig)
X=FullDf[Intervention]

y = FullDf['Average SAT_Math']

title="Interventions only, all schools"

outcome_type = 'Maths SAT score'

fig=RegressionAndPlot(X,y,title,outcome_type)

iplot(fig)
X=FullDf[Intervention]

y = FullDf['% Graduated']

title="Interventions only, all schools"

outcome_type = 'Graduation rate'

fig=RegressionAndPlot(X,y,title,outcome_type)

iplot(fig)
X=FullDf[FullDf['cluster'] ==2][Intervention]

y = FullDf[FullDf['cluster'] ==2]['% Graduated']

title="Interventions only, Urban Disadvantaged schools (n=54)"

outcome_type = 'Graduation rate'

fig=RegressionAndPlot(X,y,title,outcome_type)

iplot(fig)
X=FullDf[FullDf['cluster'] ==4][Intervention]

y = FullDf[FullDf['cluster'] ==4]['Average SAT_Math']

title="Interventions only, WASP Privileged schools (n=30)"

outcome_type = 'Average maths SAT'

fig=RegressionAndPlot(X,y,title,outcome_type)

iplot(fig)