#Data preparation



#Data source:

#https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-6094/?query=%22Alzheimer%27s%20disease%22%20&fbclid=IwAR34ZJeiWwT1eXNnO9pydDcMP3lEElZPU8Z1Sl6spZeWTokuTuJYYvgERnY



import pandas as pd

import numpy as np

import re



df=pd.read_csv('/kaggle/input/genomewide-mrna-microarray-alzheimer-disease/Norm_NR.csv',header=0)

gene_name=pd.read_csv('/kaggle/input/genomewide-mrna-microarray-alzheimer-disease/Norm_NR.csv').iloc[:,1].values







#I used this script to get differentially expressed genes: https://github.com/peterwu19881230/test_repo/blob/master/RNA_seq.R

## Load the gene index for the differentially expressed genes (The authors claimed to have found 593 using pfp=0.05 but I only found 217)

diff_gene=pd.read_csv("https://raw.githubusercontent.com/peterwu19881230/test_repo/master/diff_gene_index.csv")

diff_gene=(np.ravel(diff_gene.values)-1).tolist()



df=df.iloc[:,3:]

df=df.T

df.columns=gene_name



X=df.iloc[:,diff_gene]

y=np.array([int(bool(re.search("CDA", label))) for label in X.index.values])



# Show the distribution of NA

no_of_na=np.array(X.isna().sum())

import matplotlib.pyplot as plt

plt.hist(no_of_na, color = 'blue', edgecolor = 'black',

         bins = int(180/5))



#impute NA by mean of each feature 

from sklearn.impute import SimpleImputer



def impute_by_mean(data):

  imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean') #It seems that the imputed values are based on the column means, not the mean of the whole dataset (Ref: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

  imp_mean.fit(data)

  return(pd.DataFrame(imp_mean.transform(data)))



X=impute_by_mean(X) 

print(np.sum(X.isna().sum()))  #this verifies that all nan values have been imputed
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier





def my_inner_cv(X,y,model,cv,param_grid,test_size,random_state):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state,stratify=y)



  grid_search = GridSearchCV(model, param_grid=param_grid,cv=cv,iid=False) 

  grid_search.fit(X_train, y_train)



  accuracy=accuracy_score(y_test,grid_search.best_estimator_.predict(X_test))

  

  return([grid_search.best_estimator_,grid_search.best_params_,accuracy])





def my_logistic_regression(X,y,cv=5,param_grid={},test_size=0.2,random_state=101):

  model=LogisticRegression(solver='newton-cg',multi_class='ovr',penalty='l2')

  result=my_inner_cv(X,y,model,cv,param_grid,test_size,random_state)

  return(result)



def my_lasso(X,y,cv=5,param_grid={},test_size=0.2,random_state=101):

  model=LogisticRegression(multi_class='ovr',penalty='l1',solver='liblinear',max_iter=1000)

  result=my_inner_cv(X,y,model,cv,param_grid,test_size,random_state)

  return(result)
#set some parameters

cv=3

test_size=0.2

random_state=101
#logistic regression

result=my_logistic_regression(X,y,cv=cv,test_size=test_size,random_state=random_state)

print('accuracy= ',result[2])
# logtistic regression with L1 penalty (Lasso)

C=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

result=my_lasso(X,y,cv=cv,test_size=test_size,param_grid={'C':C},random_state=random_state)

print('best hyper parameter: ',result[1])

print('accuracy= ', result[2])
#histogram for distribution of the coefficients

coef_list=result[0].coef_.tolist()[0]

_ = plt.hist(coef_list)

plt.show()
#get the top 10 most important features (genes that have largest absolute value of coeffieients)

sig_gene_name = [df.columns.values.tolist()[i] for i in diff_gene]



gene_coef=[[gene,coef] for gene,coef in zip(sig_gene_name,coef_list)]

sorted_gene_coef=sorted(gene_coef,key=lambda x: np.abs(x[1]),reverse=True)



top10_gene=[x[0] for x in sorted_gene_coef][0:10]

top10_coef=[x[1] for x in sorted_gene_coef][0:10]



plt.figure(figsize=(20,10))

plt.bar(top10_gene, top10_coef, align='center', alpha=0.5)

plt.show()