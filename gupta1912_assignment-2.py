import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

import sklearn.metrics as score

from sklearn.preprocessing import StandardScaler



df=pd.read_csv('../input/train.csv')

df_test=pd.read_csv("../input/test.csv")





df.head()



corr = df.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='Greys', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df.columns)

ax.set_yticklabels(df.columns)

plt.show()



df.artistID.value_counts().head()

len(df['artistID'].unique().tolist())
df.songID.value_counts().head()



len(df['songID'].unique().tolist())



df.artistname.value_counts().head()

len(df['artistname'].unique().tolist())



df.songtitle.value_counts().head()
len(df['songtitle'].unique().tolist())

del df["artistname"],df["songID"],df["songtitle"]

del df_test["artistname"],df_test["songID"],df_test["songtitle"]
df.timesignature.value_counts().head()
artistID=pd.get_dummies(pd.concat((df.drop(["Top10"],axis=1).artistID,df_test.artistID),axis=0))

len(df['year'].unique().tolist())
df.year.value_counts().head()

year=pd.get_dummies(pd.concat((df.drop(["Top10"],axis=1).year,df_test.year),axis=0))

del df["artistID"],df_test["artistID"]

y_train=df.iloc[:,34].values

del df["Top10"]

columns=df.columns.tolist()

x_train=df.iloc[:,:].values





from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier() 

rf.fit(x_train, y_train)
feature_importances = pd.DataFrame(rf.feature_importances_,

                                   index = columns,

                                    columns=['importance']).sort_values('importance',ascending=False)




feature_importances



data=pd.concat((df,artistID.iloc[0:4999]),axis=1)

data_test=pd.concat((df_test,artistID.iloc[4999:]),axis=1)

data=pd.concat((data,year.iloc[0:4999]),axis=1)

data_test=pd.concat((data_test,year.iloc[4999:]),axis=1)

del data["AR03BDP1187FB5B324"],data_test["AR03BDP1187FB5B324"]

del data["year"],data_test["year"]

del data[2010],data_test[2010]
x_train=data.iloc[:,:].values

x_test=data_test.iloc[:,:].values
#sc_x=StandardScaler()

#x_train[:,0:31]=sc_x.fit_transform(x_train[:,0:31])

#x_test[:,0:31]=sc_x.transform(x_test[:,0:31])

lr=LogisticRegression(max_iter=400)

grid={"C":[1,10,100,190]}

gs=GridSearchCV(estimator=lr,param_grid=grid,scoring="roc_auc",cv=10)

gs.fit(x_train,y_train)

y_pred_proba=gs.best_estimator_.predict_proba(x_train)

print("AUC area on training data:",score.roc_auc_score(y_train,y_pred_proba.T[1]))

print("AUC area on testing data with cross validation:",gs.best_score_)
y_pred_proba=gs.predict_proba(data_test)

y_pred_df=pd.DataFrame({})

y_pred_df["songID"]=pd.read_csv("../input/test.csv").songID

y_pred_df["Top10"]=y_pred_proba.T[1]

y_pred_df.to_csv("submission.csv",index=False)
