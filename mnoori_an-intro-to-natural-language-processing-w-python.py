import pandas as pd
import numpy as np
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
    
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
submissions = pd.read_csv('../input/HN_posts_year_to_Sep_26_2016.csv')
print('Shape of dataset is:',submissions.shape)
submissions.head()
#percent of missing values for each column
pd.DataFrame(submissions.isnull().sum()/submissions.shape[0]*100,columns=['% Missing Values']).round(2)
#who has posted the most? and how many?
print('Highest number of posts is {1} made by {0}'.format(submissions['author'].value_counts().index.tolist()[0],submissions['author'].value_counts().tolist()[0]))
jonbaer=submissions[submissions['author']=='jonbaer']
print('jonbaer recieved {0:.2f} average points, while average points for all posts is {1:.2f}'.format(jonbaer['num_points'].mean(),submissions['num_points'].mean()))
ave_votes_byauthor=submissions.groupby('author').mean()
ave_votes_byauthor['num_points'].sort_values(ascending=False).head(5)
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

def histly(df,target):
    title_text='Histogram of log of average {0} by user'.format(target)
    
    data = [go.Histogram(x=np.log1p(df[target]))]
    
    shapes_list=[{
        'type': 'line',
        'xref': 'x',
        'yref': 'paper',
        'x0': np.log1p(df[target].mean()),
        'y0':0,
        'x1': np.log1p(df[target].mean()),
        'y1':1,
        'line': {
            'color': 'b',
            'width': 5,
            'dash': 'dashdot'
        }}]
        
    annotations_list=[{
            'x':np.log1p(df[target].mean()),
            'y': 50,
            'xref':'x',
            'yref':'y',
            'text':'Average across all data',
            'showarrow':True,
            'arrowhead':7,
            'ax':100,
            'ay':-100
            }]
        
    layout = go.Layout(
        title=title_text,
        font=dict(size=14, color='b'),
        xaxis={
        'title':'Log of average',
        'titlefont':{
            'size':18,
            'color':'b'
        }
        },
        yaxis={
        'title':'Count',
        'titlefont':{
            'size':18,
            'color':'b'
        }
        },
        autosize=True,
        shapes=shapes_list,
        annotations=annotations_list
        )
    
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
histly(ave_votes_byauthor,'num_points')
train=submissions.loc[:,['title','num_points']]

#sampling 5% of the daset for the representation purposes of the next two steps.
train=train.sample(frac=0.05,axis=0).reset_index()

train=train.dropna()
train.shape
#removing the punctuations.
import string
train['title_nopuncs']=train['title'].apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))
#lower casing titles
train['title_nopuncs']=train['title_nopuncs'].apply(lambda x: x.lower())
# tokenizing the headlines
train['tokenz'] = train['title_nopuncs'].apply(lambda x: x.split())
train['tokenz'].head()
import itertools

#this will create a list of all words
words=list(itertools.chain.from_iterable(train['tokenz']))

#this will create a list of unique words
unique_words=list(set(words))

print('Number of unique words:',len(set(unique_words)))
#forming a dataframe of 0 values
counts = pd.DataFrame(0,index=np.arange(train.shape[0]), columns=unique_words)
#counts.shape
#now counting the number of words in each headline and adding it to our dataframe
for index, row in train.iterrows():
    for token in row['tokenz']:
        counts.iloc[index][token]+=1
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X=vectorizer.fit_transform(list(train['title']))
counts=pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
#print( vectorizer.vocabulary_)
count_sum=counts.sum()
counts=counts.drop(count_sum[(count_sum>100) | (count_sum<5)].index,axis=1)
# spliting data into train and validation sets
from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(counts,train['num_points'],train_size=0.8,random_state=1)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
rmse=(mean_squared_error(pred,y_test))**0.5
print('RMSE is: {0:.2f}'.format(rmse))