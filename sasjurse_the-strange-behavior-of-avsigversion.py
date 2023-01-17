import os
import pandas as pd
train_columns = ['AvSigVersion', 'HasDetections']
test_columns = ['AvSigVersion']

train_path = '../input/train.csv'  # path to training file
test_path = '../input/test.csv'  # path to testing file

# Note: We are only keeping the columns we actually need.
train = pd.read_csv(train_path, usecols=train_columns)
test = pd.read_csv(test_path, usecols=test_columns)
train['TopVersion']=train['AvSigVersion'].apply(lambda x:  x.replace('.', '')[0:4] )

avsig_train = train[['TopVersion', 'HasDetections']].groupby('TopVersion').agg(['mean', 'count'])
avsig_train.columns = avsig_train.columns.droplevel()
avsig_train.rename({'count':'train count', 'mean':'HasDetections mean' }, axis='columns', inplace=True)

test['TopVersion']=test['AvSigVersion'].apply(lambda x:  x.replace('.', '')[0:4]  )

avsig_test = test['TopVersion'].value_counts()
avsig_test.name='Submission file count'

combined = pd.merge(avsig_train, pd.DataFrame(avsig_test), how='outer', left_index=True, right_index=True )
combined = combined.reset_index().rename({'index':'major_version'}, axis='columns')

print(combined.head(5))

combined.query("not major_version== '12&#' " , inplace=True)
combined['major_version'] = pd.to_numeric( combined['major_version'] )
combined = combined.query("major_version>1221")
combined[['train count', 'Submission file count']] = combined[['train count', 'Submission file count']].fillna(0)

import plotly.plotly 
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
scatter1 = go.Scatter(y=combined['train count'], x=combined['major_version'],  
                      name='Train count', 
#                      mode='markers',
                      yaxis='y1')
scatter2 = go.Scatter(y=combined['Submission file count'], x=combined['major_version'], 
                      name='Test count')
scatter3 = go.Scatter(y=combined['HasDetections mean'], x=combined['major_version'], 
                      mode='markers',
                      yaxis='y2',
                      name='Mean HasDetections')

plotly_data = [scatter1 , scatter2, scatter3]

layout= go.Layout(
        title="AvSigVersion counts and mean 'HasDetections'",
        yaxis2=dict(
                title='HasDetections mean',
                side='right',
                overlaying='y',
                ),
        xaxis=dict(title='Major Version number (first four digits)'),
        yaxis=dict(title='Number of observations (train and test)')
            )

figure = go.Figure(plotly_data, layout)

iplot(figure)