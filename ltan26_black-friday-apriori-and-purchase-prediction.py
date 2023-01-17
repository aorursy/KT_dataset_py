import numpy as np 
import pandas as pd 
import seaborn as sns

from pylab import rcParams
rcParams['figure.figsize'] = 10, 5

import os
print(os.listdir("../input/"))
df=pd.read_csv("../input/BlackFriday.csv")
df.head()
print("Number of unique users:",len(df.User_ID.unique()))
print("Number of unique products:",len(df.Product_ID.unique()))
print("Number of unique occupations recorded:",len(df.Occupation.unique()))
df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
df['Product_Category_3'] = df['Product_Category_3'].fillna(0)
user_profiles=df[['User_ID','Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status']].drop_duplicates()
user_profiles.head()
import matplotlib.pyplot as plt

columns=['Gender','Age','City_Category','Stay_In_Current_City_Years','Marital_Status']
fig = plt.figure(figsize=(30, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, (len(columns)**2)+1):
    ax=fig.add_subplot(len(columns), len(columns), i)
    ax=sns.countplot(user_profiles[columns[int((i-1)/len(columns))]],hue=columns[((i-1)%len(columns))],data=user_profiles)
    ax=plt.legend(loc='best')

plt.show()
columns=['Age','Gender','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3']
fig = plt.figure(figsize=(30, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, (len(columns))+1):
    temp=df[['User_ID',columns[i-1],'Purchase']].groupby(['User_ID',columns[i-1]]).agg('sum').reset_index().sort_values('Purchase',ascending=False)
    temp=temp.sort_values(columns[i-1]) 
    ax=fig.add_subplot(3, 3, i)
    ax=sns.boxplot(x=columns[i-1],y=np.log(temp.Purchase),data=temp)
columns=['Age','Gender','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status']
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, (len(columns))+1):
    label=[]
    temp=df[['User_ID',columns[i-1],'Purchase']].groupby(['User_ID',columns[i-1]]).agg('sum').reset_index().sort_values('Purchase',ascending=False)
    ax=fig.add_subplot(2, 3, i)
    for j in list(df[columns[i-1]].unique()):
        hist_data=temp[temp[columns[i-1]]==j]
        ax=sns.distplot(np.log(hist_data.Purchase),hist=False,label=j)
        label.append(j)
def product_popularity_barplot(Category):    
    by_cat=df[['Product_Category_1',Category,'User_ID']].groupby(['Product_Category_1',Category]).agg('count').reset_index()
    by_cat=by_cat.pivot(index='Product_Category_1',columns=Category)['User_ID'].reset_index()

    for i in by_cat.columns[1:]:
        by_cat[i] = by_cat[i]/df[df[Category]==i].count()[0]

    by_cat=by_cat.melt('Product_Category_1', var_name=Category, value_name='Val')

    columns=by_cat[Category].unique().tolist()
    fig,ax=plt.subplots(1,len(columns),sharey=True)
    fig.set_figheight(5)
    fig.set_figwidth(8*len(columns))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for i in range(0, (len(columns))):
        temp=by_cat[by_cat[Category]==columns[i]]
        title=str(Category)+':'+str(columns[i])  
        sns.barplot(x='Product_Category_1',y='Val',data=temp,ax=ax[i])
        ax[i].set_title(title,fontsize=20)
product_popularity_barplot('Gender')
product_popularity_barplot('Age')
product_popularity_barplot('City_Category')
product_popularity_barplot('Stay_In_Current_City_Years')
product_popularity_barplot('Marital_Status')
purchase_by_prod_id=df[['Product_ID','Purchase']].groupby('Product_ID').agg('sum').reset_index().sort_values('Purchase',ascending=False).head(10)
print(purchase_by_prod_id.head(10))
purchase_by_user=df[['User_ID','Purchase']].groupby('User_ID').agg('sum').reset_index().sort_values('Purchase',ascending=False).head(10)
temp=df[df['User_ID'].isin(list(purchase_by_user['User_ID']))][['User_ID','Gender','Age','Occupation','City_Category',
                                                          'Stay_In_Current_City_Years','Marital_Status']].drop_duplicates()
temp.merge(purchase_by_user,how='left').sort_values('Purchase',ascending=False)
purchase_by_user=df[['User_ID','Purchase']].groupby('User_ID').agg('sum').reset_index().sort_values('Purchase',ascending=False)
sns.distplot(purchase_by_user['Purchase'])
plt.show()
lst=[]
for item in df['User_ID'].unique():
    lst2=list(set(df[df['User_ID']==item]['Product_ID']))
    if len(lst2)>0:
        lst.append(lst2)
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

te=TransactionEncoder()
te_data=te.fit(lst).transform(lst)
df_x=pd.DataFrame(te_data,columns=te.columns_)
print(df_x.head())
#get the frequent items (support >= 0.03)
frequent_items=apriori(df_x,use_colnames=True,min_support=0.03)
frequent_items.head()
rules=association_rules(frequent_items,metric='lift',min_threshold=1)
rules.antecedents=rules.antecedents.apply(lambda x: next(iter(x)))
rules.consequents=rules.consequents.apply(lambda x: next(iter(x)))
rules=rules.sort_values('lift',ascending=False)
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import networkx as nx

nx_data=rules[rules.lift>=3]
GA=nx.from_pandas_edgelist(nx_data,source='antecedents',target='consequents',edge_attr='lift')
pos=nx.kamada_kawai_layout(GA,weight='lift')
# pos = nx.nx_agraph.graphviz_layout(GA)
# pos = nx.nx_agraph.graphviz_layout(GA, prog='dot')

edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5,color='#888'),
    hoverinfo='none',
    mode='lines')

for edge in GA.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=2)))

for node in GA.nodes():
    x, y = pos[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

for node,adjacencies in enumerate(GA.adjacency()):
    node_trace['marker']['color']+=tuple([len(adjacencies[1])])
    node_info = str(adjacencies[0])+' - # of connections: '+str(len(adjacencies[1]))
    node_trace['text']+=tuple([node_info])

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

iplot(fig, filename='networkx')
sns.heatmap(df.corr(),annot=True)
#Convert Product Category 2 and 3 into integers
df['Product_Category_2']=df['Product_Category_2'].astype(int)
df['Product_Category_3']=df['Product_Category_3'].astype(int)

#remove Product ID and User ID
data=df.drop(['Product_ID','User_ID'],axis=1)

#label categorical variables
data['Gender']=data['Gender'].map( {'M': 0, 'F': 1} ).astype(int)
data['City_Category']=data['City_Category'].map( {'A': 0, 'B': 1, 'C':2} ).astype(int)
data['Age']=data['Age'].map( {'0-17': 0, '18-25': 1, '26-35': 2,'36-45':3,'46-50':4,
                         '51-55':5,'55+':6} ).astype(int)
data['Stay_In_Current_City_Years']=data['Stay_In_Current_City_Years'].map( {'0': 0, '1': 1, '2': 2,'3':3,'4+':4}).astype(int)

#Get an array of feature variables X and target variable y
X=data.drop(['Purchase'],axis=1).values
y=data['Purchase'].values

#Select features to keep based on percentile of the highest scores
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
Selector_f = SelectPercentile(f_regression, percentile=25)
Selector_f.fit(X,y)

#get the scores of all the features
name_score=list(zip(data.drop(['Purchase'],axis=1).columns.tolist(),Selector_f.scores_))
name_score_df=pd.DataFrame(data=name_score,columns=['Feat_names','F_scores'])
name_score_df.sort_values('F_scores',ascending=False)
data=df.copy()
data=data[['City_Category','Product_Category_1', 
       'Product_Category_3','Purchase']]

#One-Hot Encoding
data=pd.get_dummies(data=data,columns=['City_Category','Product_Category_1','Product_Category_3'])

#Avoid dummy variable trap by removing one category of each categorical feature after encoding but before training
data.drop(['City_Category_A','Product_Category_1_1','Product_Category_3_0'],axis=1,inplace=True)

X=data.drop(['Purchase'],axis=1).values
y=data['Purchase'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
print("Prediction\n",y_pred)
print("Actual\n",y_test)

print("R_squared Score:",regressor.score(X_test,y_test))

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,y_pred)
print("MAE:",mae)

from sklearn.metrics import mean_squared_error
print("RMSE:",mean_squared_error(y_test,y_pred)**0.5)
data=df.copy()
data.drop(['User_ID','Product_ID'],axis=1,inplace=True)

data['Gender']=data['Gender'].map( {'M': 0, 'F': 1} ).astype(int)

data['Age']=data['Age'].map( {'0-17': 0, '18-25': 1, '26-35': 2,'36-45':3,'46-50':4,
                         '51-55':5,'55+':6} ).astype(int)

data['City_Category']=data['City_Category'].map( {'A': 0, 'B': 1, 'C':2} ).astype(int)

data['Stay_In_Current_City_Years']=data['Stay_In_Current_City_Years'].map( {'0': 0, '1': 1, '2': 2,'3':3,'4+':4}).astype(int)

X=data.drop(['Gender'],axis=1).values
y=data['Gender'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
print("Prediction:",y_pred)
print("Actual:",y_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix\n", cm)

from sklearn.metrics import precision_score
print("Precision Score\n",precision_score(y_test,y_pred,average=None))

from sklearn.metrics import accuracy_score
print("Accuracy Score: ",accuracy_score(y_test,y_pred))