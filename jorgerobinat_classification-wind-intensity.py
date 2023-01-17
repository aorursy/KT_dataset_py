import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import classification_report 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
master=pd.read_csv("../input/meteorological-model-versus-real-data/vigo_model_vs_real.csv",index_col="datetime",parse_dates=True)
master.columns
var_o="mod_o"
var_f="mod"
labels_number=3
master[var_o+"_l"]=pd.qcut(master[var_o],labels_number,retbins=True,labels=False ,duplicates="drop")[0]
bins=pd.qcut(master[var_o],labels_number,retbins=True,labels=False,duplicates="drop")[1]
master[var_f+"_4K_l"]=pd.cut(master[var_f+"_4K"],bins=bins,labels=False )
master[var_f+"_36K_l"]=pd.cut(master[var_f+"_36K"],bins=bins,labels=False )

table_4K=pd.crosstab(master[var_o+"_l"], master[var_f+"_4K_l"], margins=True,)
table_36K=pd.crosstab(master[var_o+"_l"], master[var_f+"_36K_l"], margins=True,)
table_two_model=pd.crosstab(master[var_f+"_4K_l"],master[var_f+"_36K_l"] , margins=True,)


fig, axs = plt.subplots(3,figsize = (15,15))
sns.heatmap(table_4K,annot=True,cmap="YlGnBu",ax=axs[0],fmt='.0f',linewidths=3)
sns.heatmap(table_36K,annot=True,cmap="YlGnBu",ax=axs[1],fmt='.0f',linewidths=3)
sns.heatmap(table_two_model,annot=True,cmap="YlGnBu",ax=axs[2],fmt='.0f',linewidths=3)

fig, bxs = plt.subplots(3,figsize = (20,20))
master[var_o+"_l"].value_counts(normalize=True).plot.pie(autopct='%1.0f%%',ax=bxs[0])
master[var_f+"_4K_l"].value_counts(normalize=True).plot.pie(autopct='%1.0f%%',ax=bxs[1])
master[var_f+"_36K_l"].value_counts(normalize=True).plot.pie(autopct='%1.0f%%',ax=bxs[2])
df4K=pd.concat([master[var_o+"_l"],master[var_f+"_4K_l"]],axis=1).dropna()
df36K=pd.concat([master[var_o+"_l"],master[var_f+"_36K_l"]],axis=1).dropna()
print("Bins:",bins)
print("Variable model 4 Km ")
print(classification_report(df4K[var_o+"_l"], df4K[var_f+"_4K_l"], )) 
print("Variable model 36 Km ")
print(classification_report(df36K[var_o+"_l"], df36K[var_f+"_36K_l"], )) 
#remove outliers
master["mod_dif"]=master['mod_o']-master["mod_4K"]
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] >= fence_low) & (df_in[col_name] <= fence_high)]
    return df_out
master_clean=remove_outlier(master,"mod_dif").copy()
master_clean[var_o+"_l"]=pd.qcut(master_clean[var_o],labels_number,retbins=True,labels=False ,duplicates="drop")[0]
bins=pd.qcut(master_clean[var_o],labels_number,retbins=True,labels=False,duplicates="drop")[1]
master_clean[var_f+"_4K_l"]=pd.cut(master_clean[var_f+"_4K"],bins=bins,labels=False )
master_clean[var_f+"_36K_l"]=pd.cut(master_clean[var_f+"_36K"],bins=bins,labels=False )

table_4K=pd.crosstab(master_clean[var_o+"_l"], master_clean[var_f+"_4K_l"], margins=True,)
table_36K=pd.crosstab(master_clean[var_o+"_l"], master_clean[var_f+"_36K_l"], margins=True,)
table_two_model=pd.crosstab(master_clean[var_f+"_4K_l"],master_clean[var_f+"_36K_l"] , margins=True,)


fig, axs = plt.subplots(3,figsize = (15,15))
sns.heatmap(table_4K,annot=True,cmap="YlGnBu",ax=axs[0],fmt='.0f',linewidths=3)
sns.heatmap(table_36K,annot=True,cmap="YlGnBu",ax=axs[1],fmt='.0f',linewidths=3)
sns.heatmap(table_two_model,annot=True,cmap="YlGnBu",ax=axs[2],fmt='.0f',linewidths=3)

fig, bxs = plt.subplots(3,figsize = (20,20))
master[var_o+"_l"].value_counts(normalize=True).plot.pie(autopct='%1.0f%%',ax=bxs[0])
master[var_f+"_4K_l"].value_counts(normalize=True).plot.pie(autopct='%1.0f%%',ax=bxs[1])
master[var_f+"_36K_l"].value_counts(normalize=True).plot.pie(autopct='%1.0f%%',ax=bxs[2])
df4K=pd.concat([master_clean[var_o+"_l"],master_clean[var_f+"_4K_l"]],axis=1).dropna()
df36K=pd.concat([master_clean[var_o+"_l"],master_clean[var_f+"_36K_l"]],axis=1).dropna()
print("Bins:",bins*1.95)
print("Variable model 4 Km ")
print(classification_report(df4K[var_o+"_l"], df4K[var_f+"_4K_l"], )) 
print("Variable model 36 Km ")
print(classification_report(df36K[var_o+"_l"], df36K[var_f+"_36K_l"], )) 
#creating dynamic variables
#with or without master_clean
master_clean["mslp_36K_V3H"]=master_clean["mod_36K"].pct_change(freq="3H")
X=master_clean[[ 'mod_36K', 'wind_gust_36K','mod_4K', 'wind_gust_4K',"mslp_36K_V3H"]]

df=pd.concat([X,master_clean[var_o+"_l"]],axis=1).dropna()
X=StandardScaler().fit_transform(df[df.columns[:-1]])
lb = preprocessing.LabelBinarizer()
lb.fit(df[var_o+"_l"])
Y=lb.transform(df[var_o+"_l"])
#neural network
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix ,classification_report 
from sklearn.model_selection import cross_val_score,cross_validate
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, AlphaDropout


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,)
mlp = Sequential()
mlp.add(Input(shape=(x_train.shape[1], )))
mlp.add(Dense(48, activation='relu'))
mlp.add(Dropout(0.5))
mlp.add(Dense(48, activation='relu'))
mlp.add(Dropout(0.5))
mlp.add(Dense(len(bins)-1, activation='sigmoid'))
mlp.summary()
mlp.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy',]
           )

history = mlp.fit(x=x_train,
                  y=y_train,
                  batch_size=48,
                  epochs=20,
                  validation_data=(x_test, y_test),
                  verbose=0).history
pd.DataFrame(history).plot(grid=True,figsize=(12,12),yticks=np.linspace(0.0, 1.0, num=11))
y_pred=mlp.predict(x_test)
y_pred_l=lb.inverse_transform(y_pred)
y_test_l=lb.inverse_transform(y_test)
print("Bins",bins)
print(classification_report(y_test_l, y_pred_l, )) 
result=pd.crosstab(y_test_l, y_pred_l, margins=True,)
fig = plt.figure(figsize = (18,18)) # width x height
ax1 = fig.add_subplot(3, 3, 1) 
g=sns.heatmap(result,annot=True,cmap="YlGnBu",ax=ax1,fmt='.0f',linewidths=2)