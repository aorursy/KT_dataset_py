import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns 

sns.set_style('white')
#Load data 

df=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.head()


df_shape=df.shape #shape of data frame 

df_dtypes=df.dtypes #data types associated with data 

df_info=df.info()

df_info
#NULL CHECKING 

df_null=df.isnull().sum()



#plot using heatmap

sns.heatmap(df.isnull(),cbar=False)
target=df['DEATH_EVENT']

data=df.drop('DEATH_EVENT',axis=1)

cols=data.columns

cols
data[cols[0]].describe()
sns.distplot(data[cols[0]])
data['age_cut']=pd.cut(data['age'],bins=[40,60,80,100],labels=[0,1,2])
#Plot age feature 

sns.set_style(style="darkgrid")

sns.countplot(data['age_cut'],hue=target,palette='Blues_r')
sns.countplot(data['diabetes'],hue=target, palette='Spectral')
sns.countplot(data['sex'],hue=target)
sns.distplot(data['platelets'],color='r')
sns.countplot(data['high_blood_pressure'],hue=target)
sns.countplot(data['smoking'],hue=target)
sns.distplot(data['serum_sodium'])
cat_feats=['age_cut','sex','smoking','high_blood_pressure','diabetes']

data[cat_feats].head()
dp=data[cat_feats[2]].value_counts().reset_index()

dp.columns=[cat_feats[2],'Counts']

fig,ax=plt.subplots()

ax.pie(dp['Counts'],labels=[0,1],explode=(0, 0.1),autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.set_title('Smoking')
dp=data[cat_feats[1]].value_counts().reset_index()

dp.columns=[cat_feats[1],'Counts']

fig,ax=plt.subplots()

ax.pie(dp['Counts'],labels=[0,1],explode=(0, 0.1),autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.set_title(cat_feats[1])

dp=data[cat_feats[4]].value_counts().reset_index()

dp.columns=[cat_feats[4],'Counts']

fig,ax=plt.subplots()

ax.pie(dp['Counts'],labels=[0,1],explode=(0, 0.1),autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.set_title(cat_feats[4])
sns.countplot(df['DEATH_EVENT'],palette='Blues_r')
all_corr=df.corr()

target_corr=df.corr()['DEATH_EVENT'].sort_values(ascending=False)
sns.heatmap(all_corr, cmap="YlGnBu")
from sklearn.feature_selection import SelectKBest,f_classif,chi2



X=df.drop('DEATH_EVENT',axis=1)

Y=df['DEATH_EVENT']





feat=SelectKBest(k=5,score_func=f_classif)

selector=feat.fit(X,Y)



cols=X.columns

df_features = pd.DataFrame(cols)

df_scores = pd.DataFrame(selector.scores_)



df_new = pd.concat([df_features, df_scores], axis=1)

df_new.columns = ['Features', 'Score']



df_new = df_new.sort_values(by='Score', ascending=False)



imp_feature=df_new['Features'].to_list()

imp_feature=imp_feature[:5]



imp_feature
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score





X=df[imp_feature]

Y=target

logreg=LogisticRegression()

scores=cross_val_score(logreg,X,Y,cv=4)

index=np.argmax(scores)

scores[index]*100