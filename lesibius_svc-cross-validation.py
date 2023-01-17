import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.utils import resample

from sklearn.model_selection import GridSearchCV



from sklearn.cluster import MiniBatchKMeans



from sklearn.pipeline import Pipeline



from sklearn.kernel_approximation import RBFSampler

from sklearn.linear_model import SGDClassifier



from datetime import datetime
print("Importing Data")



df_train = pd.read_csv("../input/train.csv",index_col=None)



crime_cat = df_train.Category.value_counts().index

main_cat = crime_cat[:10]



df_train = df_train.loc[df_train.Category.isin(main_cat),:]



print("Features")



print("Making X strictly positive")

df_train["X"] = df_train.X.map(lambda x: -x)



def is_weekend(day):

    if day in ["Friday","Saturday","Sunday"]:

        return(1)

    else:

        return(0)

    



def get_hour_norm(d):

    _ = datetime.strptime(d,'%Y-%m-%d %H:%M:%S').hour

    return(_/24.0)



def get_month_norm(d):

    _ = datetime.strptime(d,'%Y-%m-%d %H:%M:%S').month

    return(_/12.0)





print("Hour feature")

df_train.loc[:,"Hour"] = df_train.Dates.map(get_hour_norm)



print("Month feature")

df_train.loc[:,"Month"] = df_train.Dates.map(get_month_norm)



print("Year feature")

df_train.loc[:,"Year"] = df_train.Dates.map(lambda d: datetime.strptime(d,'%Y-%m-%d %H:%M:%S').year)



print("Keeping used features")

print("Categories (y variable)")

df_train_pred = df_train.loc[:,["Year","Category"]]



df_train = df_train.loc[:,["Year","Hour","Month","DayOfWeek","X","Y"]]

print("Creating dummies for train data")

df_train = pd.get_dummies(df_train)



X = df_train.ix[:,1:].values.tolist()

y = df_train_pred.ix[:,"Category"].values.tolist()

cls = MiniBatchKMeans(n_clusters=100)

a = cls.fit_predict(X)

unique, counts = np.unique(a, return_counts=True)

dict(zip(unique, counts))
ppl = Pipeline([('rbf',RBFSampler(random_state=0)),('sgd',SGDClassifier(loss='log'))])





X, y = resample(X,y,n_samples=10000,random_state=0)



gamma_list = [0.01,0.1,1,10] 

alpha_list = [0.001,0.01,0.1]

#L1 ratio:

# 0 => l1

# 1 => l2

l1_ratio_list = [0.01,0.25,0.5,0.75,1]



tuned_parameters = [{'rbf__gamma':gamma_list,"sgd__penalty":['none']},

                   {'rbf__gamma':gamma_list,

                    'sgd__penalty':['elasticnet'],

                    'sgd__l1_ratio':l1_ratio_list,

                   'sgd__alpha':alpha_list}]



clf = GridSearchCV(ppl, tuned_parameters, cv=5,)

clf.fit(X, y)



print(clf.best_params_)
clf.cv_results_