import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



## line 1066 has misterious values -> ignored those values

expression=pd.read_table("/kaggle/input/gene-expression-of-schizophrenia-and-bipolar/Adjusted_expression_values.txt",index_col=0,usecols=list(range(0,547))).T

expression.index=expression.index.astype('int64')

expression.shape



individuals=pd.read_table("/kaggle/input/gene-expression-of-schizophrenia-and-bipolar/E-MTAB-8018.sdrf.txt",index_col=0)



#examine what's inside individuals

'''

print(individuals.shape)



#unique values for each column

for col in individuals:

    no_of_unique_values=len(np.unique(individuals[col]))

    

    if no_of_unique_values>=10: #if there are too many unique values I will just print out the summary

        print(individuals[col].name,': ','no. of unique values= ',no_of_unique_values)



print('===========================================================================')

        

for col in individuals:

    no_of_unique_values=len(np.unique(individuals[col]))

    

    if no_of_unique_values<10: #if there are too many unique values I will just print out the summary        

        print(individuals[col].value_counts())

'''



df=individuals.join(expression,how='inner')



#Columns in addition to gene expression I think should remain in the model: sex (column 2, 0 indexed), age (column 3, 0 indexed)

col_in_model=[2,3]

col_in_model.extend(list(range(28,47311)))

X=df.iloc[:,col_in_model]

X=pd.get_dummies(X) #get dummy variables for (male,female) => after this gender becomes (0,1) at the last column



#response variable (target)

y=np.array(df['Characteristics[disease]'])

y=np.where(y!='normal',1,0)



# Show how many NAs are in the data (X)

#no_of_na=np.array(X.isna().sum())

#print('No. of NAs in the data: ',sum(no_of_na[no_of_na!=0])) #there's only 1 -> impute by means of that feature column



#impute NA by mean of each feature 

from sklearn.impute import SimpleImputer



def impute_by_mean(data):

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean') 

    imp_mean.fit(data)

    return(pd.DataFrame(imp_mean.transform(data)))



X=impute_by_mean(X) 

# Show how many NAs are in the data (X) after imputation

#no_of_na=np.array(X.isna().sum())

#print('No. of NAs after imputation: ',sum(no_of_na[no_of_na!=0]))
#feature selection

top_features=100 #no.of features I want to use



#I used these 2 scripts to get the p-values using t test

#https://github.com/peterwu19881230/test_repo/blob/master/generate_X_and_binaryLabel.py

#https://github.com/peterwu19881230/test_repo/blob/master/find_diff_expressed_genes.R

    



#load p-values, rank for feature selection

pvals=np.array(pd.read_csv('https://raw.githubusercontent.com/peterwu19881230/test_repo/master/pvals.csv')).ravel()



def rank_array(array):

    temp = array.argsort()

    ranks = np.empty_like(temp)

    ranks[temp] = np.arange(len(array))

    return(ranks)



pval_ranks=rank_array(pvals)



selected=np.where(pval_ranks<top_features,True,False) #note: ranking starts from 0



age_sex=X.iloc[:,[0,-1]]

gene_expression=X.iloc[:,1:-1]

selected_gene_expression=gene_expression.iloc[:,selected]



X=pd.concat([age_sex,selected_gene_expression],axis=1)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier





def my_inner_cv(X,y,model,cv,param_grid,test_size,random_state,train_test_boostrap=1):

  

  results=[]

  for b in range(train_test_boostrap):        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state,stratify=y)

        grid_search = GridSearchCV(model, param_grid=param_grid,cv=cv,iid=False) 

        grid_search.fit(X_train, y_train)



        accuracy=accuracy_score(y_test,grid_search.best_estimator_.predict(X_test))

        result=[grid_search.best_estimator_,grid_search.best_params_,accuracy]

        results.append(result)

        

        random_state=random_state+1

        

  return(results)





def my_logistic_regression(X,y,cv=5,param_grid={},test_size=0.2,random_state=101,train_test_boostrap=1):

  model=LogisticRegression(solver='newton-cg',multi_class='ovr',penalty='l2')

  result=my_inner_cv(X,y,model,cv,param_grid,test_size,random_state,train_test_boostrap)

  return(result)



def my_lasso(X,y,cv=5,param_grid={},test_size=0.2,random_state=101,train_test_boostrap=1):

  model=LogisticRegression(multi_class='ovr',penalty='l1',solver='liblinear',max_iter=1000)

  result=my_inner_cv(X,y,model,cv,param_grid,test_size,random_state,train_test_boostrap)

  return(result)
#set some parameters

cv=3

test_size=0.2

random_state=101

train_test_boostrap=10
#logistic regression

result=my_logistic_regression(X.iloc[:,0:2],y,cv=cv,test_size=test_size,random_state=random_state,train_test_boostrap=10)



accuracies=[result[2] for result in result] 

##->I have verified that each bootstrap train/test sample are different. However, they seem to get the same accuracy



print('Bootstrapped accuracies= ',accuracies) 

print('Average accuracy=',np.mean(accuracies))
# logtistic regression with L1 penalty (Lasso)

C=[0.0001,0.001,0.01,0.1,1,10,100,1000,10000]

result=my_lasso(X,y,cv=cv,test_size=test_size,param_grid={'C':C},random_state=random_state,train_test_boostrap=10)



hyper_params=[result[1] for result in result]

accuracies=[result[2] for result in result]

##->I have verified that each bootstrap train/test sample are different. However, they seem to get the same accuracy



print('Bootstrapped best hyper parameters: ',hyper_params)

print('Bootstrapped accuracies= ',accuracies)

print('Average accuracy=',np.mean(accuracies))