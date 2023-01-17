import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train =pd.read_csv('/kaggle/input/sec2-eda-feature-engineering/engineered_train.csv')

test =pd.read_csv('/kaggle/input/sec2-eda-feature-engineering/engineered_test.csv')

traintest = pd.concat([train,test], axis = 0,ignore_index = False)

train.head()
catcols = train.select_dtypes(include=np.object).columns

print('Number of categorical columns:',len(catcols))
def generate_plots(r,c,columns):

    """

    Generate pair of boxplot and countplot for each column in columns each row contains two such pairs'

    

    """

    fig ,axs = plt.subplots(r,c,figsize=(20,40))



    axs = axs.flatten()

    i = 0

    for col in columns:

        

        sns.boxplot(x=train[col],y=train['LogPrice'],ax=axs[i])

        sns.countplot(train[col], ax=axs[i+1])

        

        if train[col].nunique()>6:

            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)

            axs[i+1].set_xticklabels(axs[i+1].get_xticklabels(), rotation=45) 

            

        i=i+2

        plt.tight_layout()

    
generate_plots(11,4,catcols[:22])
generate_plots(11,4,catcols[22:])
low_var_cols = []

for col in catcols:

    freq_db = (traintest[col].value_counts(normalize = True))      # We will analyse for whole dataset (include train and test)

    if freq_db[freq_db>0.95].sum() != 0:

        low_var_cols.append(col)

low_var_cols

    
import statsmodels.api as sm

from statsmodels.formula.api import ols



const_mean_across_grp = []



print('Columns | P-value\n','-'*30)



for col in catcols:

    mod = ols('LogPrice ~ '+col ,data=train).fit()

    anova_table = sm.stats.anova_lm(mod,typ=2)

    

    pr  = anova_table.loc[col,'PR(>F)']

   

    if pr > 0.05:

        print(col,'|',pr)

        const_mean_across_grp.append(col)

s1 = set(const_mean_across_grp)

s2 = set(low_var_cols)

dropcols = list(s1.union(s2))



train.drop(dropcols,axis = 1,inplace=True)

test.drop(dropcols,axis = 1,inplace = True)

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression



# one hot encoding for categoricals

cats = list(train.select_dtypes(object).columns)

all_X = pd.get_dummies(data = train,columns = cats,sparse = True).copy()

all_X.drop(['LogPrice','SalePrice','Id'],axis =1, inplace=True)

all_y = train['LogPrice']



# Modelling and validating simple regressor

scores =cross_val_score(LinearRegression(),all_X,all_y, cv=3,scoring = 'neg_mean_squared_error')



# RMSE score

np.sqrt(-scores.mean())
train.to_csv('eng_filt_train.csv',index = False)

test.to_csv('eng_filt_test.csv',index = False)