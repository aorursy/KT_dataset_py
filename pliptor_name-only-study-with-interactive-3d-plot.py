import pandas as pd
import numpy  as np

np.random.seed(2018)

# load data sets 
train = pd.read_csv('../input/train.csv', usecols =['Survived','PassengerId','Name'])
test  = pd.read_csv('../input/test.csv', usecols =['PassengerId','Name'])

# combine train and test for joint processing 
test['Survived'] = np.nan
comb = pd.concat([ train, test ])
comb.head()
# define a filter for the name field to remove punctuations and single letters
def clean_name(x):
    x = x.replace(',',' ').replace('.',' ').replace('(',' ').replace(')',' ').replace('"',' ').replace('-',' ')
    return ' '.join([ w for w in x.split() if len(w)> 1])
from sklearn.feature_extraction.text import CountVectorizer
# setting punctuation filter and a minimum of terms to 2. 
count_vect = CountVectorizer(preprocessor=clean_name, min_df=2)
# the following assigns a unique number for each word in the Name feature
count_vect.fit(comb['Name'])
for i,k in enumerate(count_vect.vocabulary_):
    if i>5:
        break
    else:
        print(k, count_vect.vocabulary_[k])
# the following transforms the Name feature to a vector indicating which word they contain
v = count_vect.transform(comb['Name'] )
va = np.array(v.toarray())
va.shape
from sklearn.decomposition import PCA

#reducing va to 6 dimensions using PCA
pca = PCA(n_components = 6, random_state = 2018, whiten = True)
va = pca.fit_transform(va)

# we will plot only those that have a Survived label
va_survived = va[comb.index[comb.loc[:,'Survived']==1],...] 
va_perished = va[comb.index[comb.loc[:,'Survived']==0], ...]
va_na       = va[comb.index[comb.loc[:,'Survived']== np.nan],...]
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

trace1 = go.Scatter3d(
    x=va_survived[:,1],
    y=va_survived[:,2],
    z=va_survived[:,3],
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    ),
    name = 'Survived'
)
trace2 = go.Scatter3d(
    x=va_perished[:,1],
    y=va_perished[:,2],
    z=va_perished[:,3],
    mode='markers',
    marker=dict(
        size=6,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    ),
    name = 'Perished'
)
trace3 = go.Scatter3d(
    x=va_na[:,1],
    y=va_na[:,2],
    z=va_na[:,3],
    mode='markers',
    marker=dict(
        size=6,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    ),
    name = 'Perished'
)
data = [trace1, trace2, trace3]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# build a DataFrame from the array
feature_names = ['Nv' + str(i) for i in range(va.shape[1])]
NameVect = pd.DataFrame(data = va, index = comb.index, columns = feature_names)
NameVect.head()
# comb2 now becomes the combined data in numeric form
comb2 = pd.concat([comb[['Survived']], NameVect],axis =1)
comb2.head()
comb2.to_csv('name_only_df.csv',index=False)
df_train = comb2.loc[comb2['Survived'].isin([np.nan]) == False]
df_test  = comb2.loc[comb2['Survived'].isin([np.nan]) == True]

print(df_train.shape)
df_train.head()
print(df_test.shape)
df_test.head()
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knclass = KNeighborsClassifier(n_neighbors=11, metric = 'manhattan')
param_grid = ({'n_neighbors':[3,4,5,6,7,8,9,11,12,13],'metric':['manhattan','minkowski'],'p':[1,2]}) 
grs = GridSearchCV(knclass, param_grid, cv = 28, n_jobs=1, return_train_score = True, iid = False)
grs.fit(np.array(df_train[feature_names]), np.array(df_train['Survived']))
print("Best parameters " + str(grs.best_params_))
gpd = pd.DataFrame(grs.cv_results_)
print("Estimated accuracy of this model for unseen data:{0:1.4f}".format(gpd['mean_test_score'][grs.best_index_]))
pred_knn = grs.predict(np.array(df_test[feature_names]))

sub = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred_knn})
sub.to_csv('name_only_knn.csv', index = False, float_format='%1d')
sub.head()