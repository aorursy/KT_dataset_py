#deal with data
import numpy as np
import pandas as pd

#plotting
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# To create interactive plots
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)
# read the files and create a pandas dataframe
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# check the dimensions of the data
print(train.shape)
print(test.shape)
# first 5 rows of training data
train.head()
# summary of the training data
train.describe()
# See unique values in the column
unique_train = train.nunique().reset_index()
unique_train.columns = ["col_name", "unique_count"]
colms = unique_train[unique_train["unique_count"]==1]
colms.shape
# Delete the columns with unique values in the column
train = train.drop(colms.col_name.tolist(), axis=1)
#no of unique customers
len(np.unique(train['ID']))
#missing value counts in each of these columns
miss = train.isnull().sum()/len(train)
miss = miss[miss > 0]
miss.sort_values(inplace=True)
# knowing about variables type
dtype = train.dtypes.reset_index()
dtype.columns = ["Count", "Column Type"]
dtype.groupby("Column Type").aggregate('count').reset_index()
#knowing about response variable
#distribution of target variable
target = train['target']
sns.distplot(target)
#this line shuold be run only one time 
#log transforming the target variable
s = np.log(train['target']+1)
print ('Skewness is', s.skew())
sns.distplot(s)
# Create target and id
target = train['target']
id_train =train['ID']
id_test = test['ID']


title = 'Histogram: Target, Log(Target) And Log10(Target) Santander Dataset'

fig = tools.make_subplots(rows=3, cols=1)

data_1 = go.Histogram(x = target, # y for rotated graph
                    histnorm = 'count', #'probability'
                    name = 'Target',
                    marker = dict(color = '#1b9e77'),
                    opacity = 1.0,
                    cumulative = dict(enabled = False))

data_2 = go.Histogram(x = np.log(target), # y for rotated graph
                    histnorm = 'count', #'probability'
                    name = 'Log(Target)',
                    marker = dict(color = '#d95f02'),
                    opacity = 1.0,
                    cumulative = dict(enabled = False))
data_3 = go.Histogram(x = np.log10(target), # y for rotated graph
                    histnorm = 'count', #'probability'
                    name = 'Log10(Target)',
                    marker = dict(color = '#7570b3'),
                    opacity = 1.0,
                    cumulative = dict(enabled = False))

fig.append_trace(data_1, 1, 1)
fig.append_trace(data_2, 2, 1)
fig.append_trace(data_3, 3, 1)

layout = go.Layout(title = title,
                   bargap = 0.2,
                   bargroupgap = 0.1)
fig['layout'].update(title=title, bargap=0.2)
fig['layout']['xaxis1'].update(title='Target')
fig['layout']['xaxis2'].update(title='Log(Target)')
fig['layout']['xaxis3'].update(title='Log10(Target)')
fig['layout']['yaxis1'].update(title='Count')
fig['layout']['yaxis2'].update(title='Count')
fig['layout']['yaxis3'].update(title='Count')

iplot(fig)
#take log of target variable for the further analysis
train.target = np.log10(train.target)
#data partition for analysis
X_train = train.drop(["ID", "target"], axis=1)
y_train = np.log1p(train["target"].values)

X_test = test.drop(["ID"], axis=1)
# fit Random Forest model to the cross-validation data
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
forest.fit(X_train, y_train)

importances = forest.feature_importances_

# make importance relative to the max importance
feature_importance = 100.0 * (importances / importances.max())
sorted_idx = np.argsort(feature_importance)
feature_names = list(X_train.columns.values)
feature_names_sort = [feature_names[indice] for indice in sorted_idx]
pos = np.arange(sorted_idx.shape[0]) + .5
print('Top 10 features are: ')
for feature in feature_names_sort[::-1][:10]:
    print(feature)

# plot the result
plt.figure(figsize=(12, 10))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_names_sort)
plt.title('Relative Feature Importance', fontsize=20)
plt.show()