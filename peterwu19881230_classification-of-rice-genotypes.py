#source for original data: https://www.nature.com/articles/ncomms1467

#the re-analysis paper:  https://peerj.com/articles/7259/

#Github of the re-analysis paper: https://github.com/bongsongkim/logit.regression.rice



#prep data

import numpy as np

import pandas as pd



dat=pd.read_csv("/kaggle/input/rice-phenotypic-profile/ricediversity.44k.germplasm3.csv")

dat=dat.rename(columns = {'NSFTV.ID':'NSFTVID'})

phe=pd.read_csv("/kaggle/input/rice-phenotypic-profile/RiceDiversity_44K_Phenotypes_34traits_PLINK.txt")



dat_phe=dat.join(phe.set_index('NSFTVID'),on='NSFTVID',how='inner')

dat_phe=dat_phe.rename(columns = {'Sub-population':'Sub_population'})

dat_phe=dat_phe.query('Sub_population in ["JAP","IND"]')



#These are the author's subset features: 

X=dat_phe.loc[:,['Panicle number per plant','Seed number per panicle','Florets per panicle','Panicle fertility','Straighthead suseptability ','Blast resistance','Protein content']]



#These are all features

#X=dat_phe.iloc[:,13:] #features should be all columns after 'Culm habit'



#print(X.shape)

#print(X.columns)

#print(X.head())





y=np.array(dat_phe.Sub_population)



#To ensure y labels are interpreted correctly

map_={'JAP':0,'IND':1}

y=np.array([map_[class_] for class_ in y])



#print(X.isna().sum()) #seems ok if I do imputation



import numpy as np

from sklearn.impute import SimpleImputer



def impute_by_mean(data):

  imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean') #It seems that the imputed values are based on the column means, not the mean of the whole dataset (Ref: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

  imp_mean.fit(data)

  return(pd.DataFrame(imp_mean.transform(data)))



X=impute_by_mean(X)

X=pd.get_dummies(X)



#normalize the data (definition of normalization: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(0, 1))

normalized_df=pd.DataFrame((min_max_scaler.fit_transform(X.T)).T) #ref: https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame

X=normalized_df
dat_phe.head()
# logtistic regression 

#ref for one-vs-rest logtistic regression: https://chrisalbon.com/machine_learning/logistic_regression/one-vs-rest_logistic_regression/

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=31,stratify=y)



cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10)

model=LogisticRegression(multi_class='ovr',penalty='l2',solver='liblinear',max_iter=1000)

C=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

grid_search = GridSearchCV(model, param_grid={'C':C},cv=cv,scoring='accuracy') 

grid_search.fit(X_train, y_train) 





print('Accuracy= ',accuracy_score(y_test,grid_search.best_estimator_.predict(X_test)))

print(grid_search.best_params_)

print(grid_search.best_estimator_.coef_)
# logtistic regression with L1 penalty (Lasso)

#ref for one-vs-rest logtistic regression: https://chrisalbon.com/machine_learning/logistic_regression/one-vs-rest_logistic_regression/

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=31,stratify=y)



cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10)

model=LogisticRegression(multi_class='ovr',penalty='l1',solver='liblinear',max_iter=1000)

C=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

grid_search = GridSearchCV(model, param_grid={'C':C},cv=cv,scoring='accuracy') 

grid_search.fit(X_train, y_train) 





print('Accuracy= ',accuracy_score(y_test,grid_search.best_estimator_.predict(X_test)))

print(grid_search.best_params_)

print(grid_search.best_estimator_.coef_)
# t-SNE

perplexity=5 #default: 30





##ref: https://medium.com/@sourajit16.02.93/tsne-t-distributed-stochastic-neighborhood-embedding-state-of-the-art-c2b4b875b7da

from sklearn.manifold import TSNE 

import matplotlib.pyplot as plt





# Module for standardization

from sklearn.preprocessing import StandardScaler

#Get the standardized data

standardized_data = StandardScaler().fit_transform(X)





model = TSNE(n_components=2) #n_components means the lower dimension



low_dim_data = pd.DataFrame(model.fit_transform(standardized_data))





finalDf = pd.concat([low_dim_data, pd.DataFrame(y)], axis = 1)

finalDf.columns = ['Dim 1', 'Dim 2','target']





fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Dim 1', fontsize = 15)

ax.set_ylabel('Dim 2', fontsize = 15)

ax.set_title('2 component tSNE', fontsize = 20)

targets = np.unique(y)



from itertools import cycle

cycol = cycle('bgrcmk')



for target in targets:

    indicesToKeep = (finalDf['target'] == target)

    ax.scatter(finalDf.loc[indicesToKeep, 'Dim 1'], finalDf.loc[indicesToKeep, 'Dim 2'], s = 50, color=next(cycol), alpha=0.3)

ax.legend(['JAP','IND'])

ax.grid()

plt.show()