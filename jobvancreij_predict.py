df2['vote_classes'] = pd.cut(df2['vote_average'],4, labels=["low",'medium-low','medium-high','high'])

num_list = ['budget','popularity','revenue','runtime','vote_count','vote_classes']

movie_num = df2[num_list]

movie_num.head()



from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np



df = movie_num

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



train, test = df[df['is_train']==True], df[df['is_train']==False]



features = df.columns[:5]

clf = RandomForestClassifier(n_jobs=10)

y, _ = pd.factorize(train['vote_classes'])

clf.fit(train[features],y)



preds = df.vote_classes[clf.predict(test[features])]

pd.crosstab(test['vote_classes'], preds.values, rownames=['actual'], colnames=['preds'])
df2['vote_classes'] = pd.cut(df2['vote_average'],10, labels=range(10))
df3['vote_classes'] = pd.cut(df2['vote_average'],4, labels=['low','medium-low','medium-high','high'])

num_list = ['log_budget','log_popularity','log_revenue','log_runtime','log_vote_count','vote_classes']

movie_num = df3[num_list]

movie_num.head()



from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np



df = movie_num

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



train, test = df[df['is_train']==True], df[df['is_train']==False]



features = df.columns[:5]

clf = RandomForestClassifier(n_jobs=1000)

y, _ = pd.factorize(train['vote_classes'])

clf.fit(train[features],y)



preds = df.vote_classes[clf.predict(test[features])]

pd.crosstab(test['vote_classes'], preds.values, rownames=['actual'], colnames=['preds'])
df2['vote_classes'] = pd.cut(df2['vote_average'],4, labels=['low','medium-low','medium-high','high'])

num_list = ['budget','popularity','revenue','runtime','vote_count','vote_classes']

movie_num = df2[num_list]

movie_num.head()



from sklearn.preprocessing import StandardScaler



movie_num['normBudget'] = StandardScaler().fit_transform(movie_num['budget'].reshape(-1, 1))

movie_num['normPopularity'] = StandardScaler().fit_transform(movie_num['popularity'].reshape(-1, 1))

movie_num['normRevenue'] = StandardScaler().fit_transform(movie_num['revenue'].reshape(-1, 1))

movie_num['normVoteCount'] = StandardScaler().fit_transform(movie_num['vote_count'].reshape(-1, 1))

movie_num['normRuntime'] = StandardScaler().fit_transform(movie_num['runtime'].reshape(-1, 1))

#movie_num['vote_classes'] = pd.cut(movie_num['vote_average'],2, labels=[0,1])



movie_test = movie_num.drop(['budget','popularity','vote_count','revenue','runtime'],axis=1)

cols=['normBudget','normPopularity','normRevenue','normVoteCount','normRuntime','vote_classes']

movie_num = movie_test[cols]

#movie_test = movie_test[:-1] + movie_test[-1:]

movie_num.head()
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np



df = movie_num

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



train, test = df[df['is_train']==True], df[df['is_train']==False]



features = df.columns[:5]

clf = RandomForestClassifier(n_jobs=1000)

y, _ = pd.factorize(train['vote_classes'])

clf.fit(train[features],y)



preds = df.vote_classes[clf.predict(test[features])]

pd.crosstab(test['vote_classes'], preds.values, rownames=['actual'], colnames=['preds'])
df2['revenue_classes'] = pd.cut(df2['revenue'],3, labels=['low','medium','high'])

num_list = ['budget','popularity','vote_average','runtime','vote_count','revenue_classes']

movie_num = df2[num_list]

movie_num.head()



from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np



df = movie_num

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



train, test = df[df['is_train']==True], df[df['is_train']==False]



features = df.columns[:5]

clf = RandomForestClassifier(n_jobs=100)

y, _ = pd.factorize(train['revenue_classes'])

clf.fit(train[features],y)



preds = df.revenue_classes[clf.predict(test[features])]

pd.crosstab(test['revenue_classes'], preds.values, rownames=['actual'], colnames=['preds'])
df3['revenue_classes'] = pd.cut(df2['revenue'],3, labels=['low','medium','high'])

num_list = ['log_budget','log_popularity','log_vote_average','log_runtime','log_vote_count','revenue_classes']

movie_num = df3[num_list]

movie_num.head()



from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np



df = movie_num

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



train, test = df[df['is_train']==True], df[df['is_train']==False]



features = df.columns[:5]

clf = RandomForestClassifier(n_jobs=1000)

y, _ = pd.factorize(train['revenue_classes'])

clf.fit(train[features],y)



preds = df.revenue_classes[clf.predict(test[features])]

pd.crosstab(test['revenue_classes'], preds.values, rownames=['actual'], colnames=['preds'])
df2['revenue_classes'] = pd.cut(df2['revenue'],3, labels=range(3))

num_list = ['budget','popularity','vote_average','runtime','vote_count','revenue_classes']

movie_num = df2[num_list]

movie_num.head()



from sklearn.preprocessing import StandardScaler



movie_num['normBudget'] = StandardScaler().fit_transform(movie_num['budget'].reshape(-1, 1))

movie_num['normPopularity'] = StandardScaler().fit_transform(movie_num['popularity'].reshape(-1, 1))

#movie_num['normVoteAverage'] = StandardScaler().fit_transform(movie_num['vote_average'].reshape(-1, 1))

movie_num['normRuntime'] = StandardScaler().fit_transform(movie_num['runtime'].reshape(-1, 1))

movie_num['normVoteCount'] = StandardScaler().fit_transform(movie_num['vote_count'].reshape(-1, 1))





#movie_num['revenue_classes'] = pd.cut(movie_num['vote_average'],2, labels=[0,1])



movie_test = movie_num.drop(['budget','popularity','runtime','vote_count',],axis=1)

cols=['normBudget','normPopularity','vote_average','normVoteCount','normRuntime','revenue_classes']

movie_num = movie_test[cols]

#movie_test = movie_test[:-1] + movie_test[-1:]

movie_num.head()
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np



df = movie_num

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



train, test = df[df['is_train']==True], df[df['is_train']==False]



features = df.columns[:5]

clf = RandomForestClassifier(n_jobs=1000)

y, _ = pd.factorize(train['revenue_classes'])

clf.fit(train[features],y)



preds = df.revenue_classes[clf.predict(test[features])]

pd.crosstab(test['revenue_classes'], preds.values, rownames=['actual'], colnames=['preds'])