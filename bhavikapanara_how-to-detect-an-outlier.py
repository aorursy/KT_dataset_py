import numpy as np

import pandas as pd

import matplotlib.pylab as plt

import os
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.shape,test.shape
basement_columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

                    'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
basement_missing_df = pd.DataFrame(columns=['train','test'], index=basement_columns)
for e in basement_columns:

    basement_missing_df.loc[e]['train'] = train[e].isna().sum()

    basement_missing_df.loc[e]['test'] = test[e].isna().sum()
basement_missing_df
def draw_graph(df, variables, n_rows, n_cols):

    fig=plt.figure(figsize=(15,50))

    for i, var_name in enumerate(variables):

        ax=fig.add_subplot(n_rows,n_cols,i+1)

                

        ingredients = df[var_name].value_counts().index

        data = df[var_name].value_counts().values

        

        def func(pct, allvals):

            total = sum(allvals)

            val = int(round(pct*total/100.0))

            return "{:.1f}%({:d})".format(pct, val)





        wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),

                                          textprops=dict(color="w"))

        ax.legend(wedges, ingredients,

          title=var_name,

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))

        plt.setp(autotexts, size=15, weight="bold")



        ax.set_title(var_name, fontsize = 20)

        ax.tick_params(axis = 'both', which = 'major', labelsize = 35)

        ax.tick_params(axis = 'both', which = 'minor', labelsize = 35)

        ax.set_xlabel('')

    fig.tight_layout(rect = [0, 0.03, 1, 0.95])  # Improves appearance a bit.

    plt.show()

    

draw_graph(train, ['BsmtCond','BsmtQual'], 1, 2)
print("NA (no basement) values in train set of BsmtQual:",train['BsmtQual'].isna().sum())

print("NA (no basement) values in test set of BsmtQual :",test['BsmtQual'].isna().sum())

print("******************************************************")

print("NA (no basement) values in train set of BsmtCond:",train['BsmtCond'].isna().sum())

print("NA (no basement) values in test set of BsmtCond :",test['BsmtCond'].isna().sum())
no_basement_train_cnt = len(train.loc[(train['BsmtCond'].isna()) & (train['BsmtQual'].isna())])

no_basement_test_cnt = len(test.loc[(test['BsmtCond'].isna()) & (test['BsmtQual'].isna())])
print("The house which have no basement in train : ",no_basement_train_cnt)

print("The house which have no basement in test  : ",no_basement_test_cnt)
test[['Id','BsmtQual','BsmtCond']].loc[(test['BsmtQual'].isna()) & (~test['BsmtCond'].isna())]
test[['Id','BsmtQual','BsmtCond']].loc[(test['BsmtCond'].isna()) & (~test['BsmtQual'].isna())]
print("NA (no basement) values in train set of BsmtExposure:",train['BsmtExposure'].isna().sum())

print("NA (no basement) values in test set of BsmtExposure :",test['BsmtExposure'].isna().sum())
train[['Id','BsmtQual','BsmtCond','BsmtExposure']].loc[(train['BsmtExposure'].isna()) & (~train['BsmtQual'].isna())]
train['BsmtExposure'].iloc[948] = 'No'
test[['Id','BsmtQual','BsmtCond','BsmtExposure']].loc[(test['BsmtExposure'].isna()) & (~test['BsmtCond'].isna()) & (~test['BsmtQual'].isna())]
test['BsmtExposure'].iloc[27] = 'No'

test['BsmtExposure'].iloc[888] = 'No'
print("Missing value in train set of TotalBsmtSF:",train['TotalBsmtSF'].isna().sum())

print("Missing value in test set of TotalBsmtSF :",test['TotalBsmtSF'].isna().sum())
test[basement_columns].loc[test['TotalBsmtSF'].isna()]
test['BsmtQual'].iloc[660] = 'NA'

test['BsmtCond'].iloc[660] = 'NA'

test['BsmtExposure'].iloc[660] = 'NA'

test['BsmtFinType1'].iloc[660] = 'NA'

test['BsmtFinSF1'].iloc[660] = 0

test['BsmtFinType2'].iloc[660] = 'NA'

test['BsmtFinSF2'].iloc[660] = 0

test['BsmtUnfSF'].iloc[660] = 0

test['TotalBsmtSF'].iloc[660] =  0
test[basement_columns].iloc[660]
print("No basement in train set of TotalBsmtSF:",(train['TotalBsmtSF']==0).sum())

print("No basement in test set of TotalBsmtSF :",(test['TotalBsmtSF']==0).sum())
print("NA (no basement-type 1) houses in train set of BsmtFinType1:",train['BsmtFinType1'].isna().sum())

print("NA (no basement- type1) houses in test set of BsmtFinType1 :",test['BsmtFinType1'].isna().sum())

print("***************************************************")

print("Null values in train set of BsmtFinSF1:",train['BsmtFinSF1'].isna().sum())

print("Null values in test set of BsmtFinSF1 :",test['BsmtFinSF1'].isna().sum())

print("****************************************************")

print("no basement-type1 houses in train set of BsmtFinSF1:", (train['BsmtFinSF1'] == 0).sum())

print("no basement-type1 houses in test set of BsmtFinSF1 :",(test['BsmtFinSF1'] == 0).sum())
train[basement_columns].loc[(train['BsmtFinType1'].isna()) & (train['BsmtFinSF1']!=0)]
print("NA (no basement - type2) houses in train set of BsmtFinType2:",train['BsmtFinType2'].isna().sum())

print("NA (no basement -type2) houses in test set of BsmtFinType2 :",test['BsmtFinType2'].isna().sum())

print("***************************************************")

print("Null values in train set of BsmtFinSF2:",train['BsmtFinSF2'].isna().sum())

print("Null values in test set of BsmtFinSF2 :",test['BsmtFinSF2'].isna().sum())

print("****************************************************")

print("no basement-type2 houses in train set of BsmtFinSF2:", (train['BsmtFinSF2'] == 0).sum())

print("no basement-type2 houses in test set of BsmtFinSF2 :",(test['BsmtFinSF2'] == 0).sum())
train[basement_columns].loc[(train['BsmtFinType2'].isna()) & (train['BsmtFinSF2']!=0)]
print("Null values in train set of BsmtUnfSF:",train['BsmtUnfSF'].isna().sum())

print("Null values in test set of BsmtUnfSF :",test['BsmtUnfSF'].isna().sum())

print("****************************************************")

print("No unfinished basement houses in train set of BsmtUnfSF:", (train['BsmtUnfSF'] == 0).sum())

print("No unfinished basement houses in test set of BsmtUnfSF :",(test['BsmtUnfSF'] == 0).sum())