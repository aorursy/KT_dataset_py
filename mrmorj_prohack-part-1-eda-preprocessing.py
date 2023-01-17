import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



pd.set_option('chained_assignment',None)



seed=47



train = pd.read_csv('/kaggle/input/prohack-hackathon/train.csv')

test = pd.read_csv('/kaggle/input/prohack-hackathon/test.csv')

sample_submit = pd.read_csv('/kaggle/input/prohack-hackathon/sample_submit.csv')



total_df_fillna = pd.concat([train, test], ignore_index=True)



print(train.shape, test.shape)
total_df_fillna
total_df_fillna.describe()
print('Unique galaxies:', total_df_fillna['galaxy'].nunique())

print('Number of years:', total_df_fillna['galactic year'].nunique())



total_df_fillna.drop(['galactic year','galaxy','y'],axis=1, inplace=True)
import missingno as msno 

msno.matrix(total_df_fillna[list(total_df_fillna.columns)[:]])

print('Train missing values:', train.isna().sum().sum())

print('Test missing values:', test.isna().sum().sum())
import pickle

from sklearn.linear_model import ElasticNet



test_na = test.dropna()

train_na = train.dropna()

total_df = pd.concat([train_na, test_na], ignore_index=True)

total_df.drop(['galactic year','galaxy','y'],axis=1, inplace=True)



corrMatrix = total_df.corr()

plt.figure(figsize=(20,20))

sns.heatmap(corrMatrix, annot=False)

plt.show()



koef_corr = 0.65

koef_na = 0.8

top_features = 5

num_model = 1

list_target_features = []

list_corr_features = []



for big_col in list(total_df.columns):

    list_correlated=[]

    correlations=[]

    na_rows=[]

    na_big_col= total_df_fillna[big_col].isna().sum()



    for col in list(total_df.columns):

        cor = total_df[big_col].corr(total_df[col])



        if np.absolute(cor)>koef_corr and np.absolute(cor)<0.99:

            list_correlated.append(col)

            correlations.append(np.absolute(cor))

            na_rows.append(total_df_fillna[col].isna().sum())

            df = pd.DataFrame(list(zip(list_correlated, correlations,na_rows)),columns =['features', 'correlation', 'na']) 



    df = df[df.na<koef_na*na_big_col].sort_values(by='correlation', ascending=False)

    list_correlated = list(df[:top_features].features)



    if len(list_correlated)>0:

        list_target_features.append(big_col)

        list_corr_features.append(list_correlated)

        model = ElasticNet(random_state = seed)     

        model.fit(total_df[list_correlated], total_df[big_col])

        pkl_filename = "pickle_model_%i.pkl"%len(list_target_features)

        with open(pkl_filename, 'wb') as file:

            pickle.dump(model, file)

        num_model = num_model + 1
for i in range(len(list_target_features)):

    target = list_target_features[i]

    corr_feat = list_corr_features[i]



    with open("pickle_model_%i.pkl"%(i+1), 'rb') as file:

        pickle_model = pickle.load(file)



    for i in range(len(total_df_fillna.index)):

        time_corr_features = []

        

        for j in range(len(corr_feat)):

            if len(total_df_fillna[corr_feat[j]][i].shape) == 1:

                feat = total_df_fillna[corr_feat[j]][i].values[0]

            else:

                feat = total_df_fillna[corr_feat[j]][i]

            time_corr_features.append(feat)



        if np.isnan(time_corr_features).sum() == 0:

            if np.isnan(total_df_fillna[target][i]) == True:

                pred = pickle_model.predict(np.asarray(time_corr_features).reshape(1,-1))

                total_df_fillna[target][i] = pred



print(total_df_fillna.isna().sum().sum())
new_train = total_df_fillna[:len(train.index)].reset_index()

new_test = total_df_fillna[len(train.index):].reset_index()



new_train = new_train.join(train[['galactic year','galaxy','y']])

new_test = new_test.join(test[['galactic year','galaxy']])



df = pd.concat([new_train, new_test], ignore_index=True)



df_sort = df.sort_values(by=['galaxy','galactic year'], ascending=True, ignore_index=True)

title_galaxy = list(df['galaxy'].unique())



num_col = df.iloc[:,:-3].columns



total_df = pd.DataFrame()

for name in title_galaxy:

    temp = df[df['galaxy']==name]

    for column in num_col:

        if temp[column].isna().sum()+4<=len(temp):

            temp[column] = temp[column].interpolate(method='linear', limit_direction ='backward')

        if temp[column].isna().sum()<len(temp):

            m = temp[column].mean()

            temp[column]=temp[column].fillna(m)

    total_df = pd.concat([total_df, temp], ignore_index=True)

    

print(total_df.isna().sum().sum())
for col in total_df.iloc[:,:-3].columns.values:

    mean = total_df[col].mean()

    total_df[col] = total_df[col].fillna(mean)
train = total_df[:len(train.index)].reset_index()

test = total_df[len(train.index):].reset_index()



train = train.drop(['level_0', 'index'], axis=1)

test = test.drop(['y','index', 'level_0'], axis=1)
total_df['galaxy'].unique()
import category_encoders as ce

cat_cols=['galaxy']

target_enc = ce.CatBoostEncoder(cols=cat_cols)

target_enc.fit(train[cat_cols], train['y'])

train = train.join(target_enc.transform(train[cat_cols]).add_suffix('_cb'))



test = test.join(target_enc.transform(test[cat_cols]).add_suffix('_cb'))

train[['galaxy', 'galaxy_cb']]
sns.distplot(train['Gross income per capita'], label='train')

sns.distplot(test['Gross income per capita'], label='test')

plt.title('Train & Test Distribution')

plt.legend()

plt.show()
sns.distplot(train['y'], label='train')

plt.title('Target Distribution')