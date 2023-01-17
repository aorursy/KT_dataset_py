import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
print('Shape of train data: ',train_df.shape)

print('Shape of test data: ',test_df.shape)
target = train_df['target']

train_df.drop('target',axis = 1, inplace =True)

df = pd.concat([train_df,test_df],ignore_index=True)

id_ =df.id

df.drop('id',axis=1, inplace=True)
df.shape
df.dtypes
del train_df, test_df
sns.set_style('darkgrid')
plt.figure(figsize=(5,4))

graph = sns.countplot(target,palette = 'viridis')



for i, p  in enumerate(graph.patches):

    graph.annotate('{}'.format(target.value_counts().values[i]),(p.get_x()+0.25,p.get_height()),fontsize=12)
binary_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

nominal_features = ['nom_0', 'nom_1', 'nom_2','nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

ordinal_features = ['ord_0','ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

other_features = ['day','month']
def plot_binary_dist(features):

    fig, ax = plt.subplots(1,5,figsize=(20,5))

    

    for i in range(len(features)):

        sns.countplot(df[features[i]],ax = ax[i], palette = 'Oranges')

        for n,p in enumerate(ax[i].patches):

            ax[i].annotate('{}'.format(df[features[i]].value_counts().values[n]),(p.get_x()+0.2,p.get_height()+0.1),fontsize=12)

            ax[i].set_title(features[i])

            plt.tight_layout()
plot_binary_dist(binary_features)
def percent_of_nulls(features,palette_name,fig_size=(8,4)):

    nulls = df[features].isna().sum()

    plt.figure(figsize=fig_size)

    graph = sns.barplot(x = nulls.index, y = nulls.values,palette=palette_name)

    

    for i, p  in enumerate(graph.patches):

        percent_of_null = round((nulls.values[i]/len(df))*100,2)

        graph.annotate('{}'.format(str(percent_of_null)+'%'),((p.get_width()/3)+p.get_x(),p.get_height()),fontsize=12)

    
percent_of_nulls(binary_features,'Purples')
percent_of_nulls(ordinal_features,'Blues')
percent_of_nulls(nominal_features,'RdBu',(15,4))
percent_of_nulls(other_features,'viridis',(4,4))
def cardinality(features):

    count = []

    for col in features:

        count.append(df[col].nunique(dropna=False))

    plt.figure(figsize=(12,5))

    graph = sns.barplot(x=features,y=count,palette='Blues')

        

    for i, p in enumerate(graph.patches):

        graph.annotate('{}'.format(count[i]),(p.get_x()+0.3,p.get_height()),fontsize=12)

        
cardinality(nominal_features)
cardinality(ordinal_features)
numeric_features = df.select_dtypes(include=['float','integer'])

categorical_features = df.select_dtypes(include=['object'])
numeric_features.columns
categorical_features.columns
assert len(df.columns) == len(numeric_features.columns) + len(categorical_features.columns)
mean_df=numeric_features.copy()

for col in numeric_features.columns:

    mean_df[col].fillna(mean_df[col].mean(),inplace = True)

median_df=numeric_features.copy()

for col in numeric_features.columns:

    median_df[col].fillna(median_df[col].median(),inplace = True)

mode_df=numeric_features.copy()

for col in numeric_features.columns:

    mode_df[col].fillna(mode_df[col].mode()[0],inplace = True)

from sklearn.impute import KNNImputer



knn_df = numeric_features[:10000].copy()

knn = KNNImputer(n_neighbors=5)

imputed = knn.fit_transform(knn_df)



knn_imp_df = pd.DataFrame(imputed,columns=numeric_features.columns)

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



mice_df = numeric_features.copy()

mice_imp = IterativeImputer(max_iter=20,random_state=1)

mice_imputed = mice_imp.fit_transform(mice_df)

mice_imp_df = pd.DataFrame(mice_imputed, columns=numeric_features.columns)
del numeric_features, mean_df, mode_df, median_df, knn_imp_df
for column in categorical_features.columns:

    categorical_features[column] = categorical_features[column].fillna('missing')

categorical_features.isna().sum()
df = pd.concat([mice_imp_df,categorical_features],axis=1)

df.shape
del mice_imp_df, categorical_features
print('Memory usage before conversion: '+str(round(df.memory_usage().sum()/(1024*1024),1))+'MB')
for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category')
print('Memory usage after conversion: '+str(round(df.memory_usage().sum()/(1024*1024),1))+'MB')
df_cat = df.select_dtypes(include='category')

print(list(df_cat.columns))
import category_encoders as ce



ohe_enc = df_cat.copy()

ohe = ce.one_hot.OneHotEncoder(return_df=True)

ohe_enc = ohe.fit_transform(ohe_enc.iloc[:,:5])

ohe_enc.shape
le_enc = df_cat.copy()

for column in df_cat.columns:

    le_enc[column] = le_enc[column].cat.codes

le_enc.head()
ord_enc = df_cat.copy()

ore = ce.ordinal.OrdinalEncoder(return_df=True)

ord_enc = ore.fit_transform(ord_enc)

ord_enc.head()
helmert_enc = df_cat.copy()

helmert = ce.HelmertEncoder(return_df=True)

helmert_enc = helmert.fit_transform(helmert_enc.iloc[:,:5])

helmert_enc.shape
binary_df = df_cat.copy()

binary_encoder = ce.BinaryEncoder(return_df=True)

binary_df  = binary_encoder.fit_transform(binary_df)

binary_df.head()
freq_df = df_cat.copy()



for column in freq_df.columns:

    freq = round(freq_df.groupby(column).size()/len(freq_df),3)

    freq_df[column] = freq_df[column].map(freq)

freq_df.head()
mean_df = df_cat.copy()

train_cat_df = mean_df.iloc[:600000,:]

train_cat_df['target'] = target



for column in mean_df.columns:

    mean = train_cat_df.groupby(column)['target'].mean().round(2)

    mean_df[column] = mean_df[column].map(mean)

mean_df.head()


train_tar_df = train_cat_df.copy()

train_target = train_tar_df['target']

train_features = train_tar_df.drop('target',axis=1)



target_encoder = ce.TargetEncoder(return_df=True)

target_encoder.fit(train_features,train_target)

encoded_features = target_encoder.transform(df_cat)

encoded_features.head()


train_tar_df = train_cat_df.copy()

train_target = train_tar_df['target']

train_features = train_tar_df.drop('target',axis=1)



leaveone_encoder = ce.LeaveOneOutEncoder(return_df=True)

leaveone_encoder.fit(train_features,train_target)

encoded_features = leaveone_encoder.transform(df_cat)

encoded_features.head()