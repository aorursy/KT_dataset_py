#Import des librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.utils import shuffle



#Drawing modules

import seaborn as sns

import matplotlib.pyplot as plt



#Learning modules

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score

from catboost import CatBoostClassifier

df=pd.read_csv('../input/heart.csv')
df.head()
#Check Null values

df.isnull().sum()
df.groupby('target').count().iloc[:,0].plot.bar()
# Define variables types : 

var_label=['target',] #simply our target

var_bool=['sex','fbs','exang'] # 0,1 

var_cont=['age','trestbps','chol','thalach','oldpeak'] #continuous variables

var_cat=['cp','restecg','slope','thal','ca'] #categorial variables
#Check correlations 

def CorrMtx(d):

    df=np.around(abs(d.corr()),2)

    mask = np.zeros_like(df, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    sns.set_style(style = 'white')

    # Set up  matplotlib figure

    f, ax = plt.subplots(figsize=(11, 9))

    # Add diverging colormap from red to blue

    cmap = sns.diverging_palette(250, 10, as_cmap=True)



    # Draw correlation plot with or without duplicates

    sns.heatmap(df, mask=mask, cmap=cmap, annot = True,

               square=True,

               linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)



CorrMtx(df)

#No evident correlation. Some might not even be relevant !
#Continuous variables : Violin Plot

data = df[var_label+var_cont].copy() #Create a dataframe containing Target+continuous variables

data[var_cont]=(data[var_cont] - data[var_cont].mean()) / (data[var_cont].std()) #normalize the dataset



data = pd.melt(data,id_vars='target',var_name='features',value_name='value')



plt.figure(figsize=(10,10))

sns.violinplot(x='features', y="value", hue="target", data=data,split=True, inner="quart")

plt.xticks(rotation=90)



del data #Free memory
#count plots for binary variables

for i in var_bool:

    plt.figure(figsize=(5,5))

    sns.countplot(x=i,data=df,hue='target')
#count plots for categorical variables

for i in var_cat:

    plt.figure(figsize=(10,5))

    sns.countplot(x=i,data=df,hue='target')
#Let's make sure data are not ordered

df = shuffle(df) 
#Create Test/Train subsets



y=df['target'].copy()

X=df.drop('target', axis=1).copy()



X[var_cont]=(X[var_cont] - X[var_cont].mean()) / (X[var_cont].std()) #Normalise les données continues

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123, stratify=y)

print(X_train.shape,"-",y_train.shape)

print(X_test.shape,"-",y_test.shape)
#From here we will start training

# Get Cat columns indexes

cat_feat=[X_train.columns.get_loc(c) for c in X_train.columns if c in var_cat]



cbmodel = CatBoostClassifier(

                       random_seed=42,

                       eval_metric='Logloss',

                        verbose=0

                      )

cbmodel.fit(X_train,

          y_train,

          cat_features=cat_feat,

          eval_set=(X_test, y_test),

          use_best_model=True

         )



cb_score=cbmodel.score(X_test, y_test)

print('Score on Train Set:',cbmodel.score(X_train, y_train)) # train (learn) score

print('Score on Test Set:',cb_score )# val (test) score



y_pred=cbmodel.predict(X_test)

confusion_matrix(y_pred,y_test)
#What are our best features ?

best_feat=pd.DataFrame(cbmodel.get_feature_importance(),index=X_train.columns,columns=['Features'])

best_feat.sort_values('Features').plot.barh()
#cp : OK

#restecg : Not enough 2  ... let's requalify them as 0

X['restecg']=np.where(X['restecg']==2,1,X['restecg']) #It becomes boolean

var_cat=['cp','slope','thal','ca'] #categorial variables : remove restecg

var_bool=['sex','fbs','exang','restecg'] # 0,1 : add restecg

#Slope : OK

#Thal : does not correspond to the description. I'll arbitrary put 0 as 1

X['thal']=np.where(X['thal']==0,1,X['thal'])



#ca : we'll change it to 0-1-2&more 

X['ca']=np.where(X['ca']>2,2,X['ca'])
# Get Cat columns indexes

cat_feat=[X_train.columns.get_loc(c) for c in X_train.columns if c in var_cat]



cbmodel2 = CatBoostClassifier(

                       random_seed=42,

                       eval_metric='Logloss',

                        verbose=0

                      )

cbmodel2.fit(X_train,

          y_train,

          cat_features=cat_feat,

          eval_set=(X_test, y_test),

          use_best_model=True

         )



cb2_score=cbmodel2.score(X_test, y_test)

print('Score on Train Set:',cbmodel2.score(X_train, y_train)) # train (learn) score

print('Score on Test Set:',cb2_score )# val (test) score
y_pred=cbmodel.predict(X_test)

confusion_matrix(y_pred,y_test)
#What are our best features ?

best_feat=pd.DataFrame(cbmodel2.get_feature_importance(),index=X_train.columns,columns=['Features'])

best_feat.sort_values('Features').plot.barh()
del X['fbs']

del X_train['fbs']

del X_test['fbs']

cat_feat=[X_train.columns.get_loc(c) for c in X_train.columns if c in var_cat]



cbmodel3 = CatBoostClassifier(

                       random_seed=42,

                       eval_metric='Logloss',

                        verbose=0

                      )

cbmodel3.fit(X_train,

          y_train,

          cat_features=cat_feat,

          eval_set=(X_test, y_test),

          use_best_model=True

         )

cb3_score=cbmodel3.score(X_test, y_test)

print('Score on Train Set:',cbmodel3.score(X_train, y_train)) # train (learn) score

print('Score on Test Set:',cbmodel3.score(X_test, y_test) )# val (test) score
#Print the test results of each test :

print('Score on Original Set:',cb_score)

print('Score on Balanced Set:',cb2_score)

print('Score without fbs :',cb3_score)