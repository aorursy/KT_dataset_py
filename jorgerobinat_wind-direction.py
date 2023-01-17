import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import classification_report 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
master=pd.read_csv("../input/meteorological-model-versus-real-data/vigo_model_vs_real.csv",index_col="datetime",parse_dates=True)
master.columns
import plotly.express as px
df = pd.DataFrame()
df["direction"]=master.dir_o[master.visibility_o<1000]
df["strength"]=master.mod_o[master.visibility_o<1000]
df['freq'] = df.groupby('direction')['direction'].transform('count')

fig = px.scatter_polar(df,r= "freq",theta="direction")
fig
import plotly.express as px
df = px.data.wind()
fig = px.scatter_polar(df, r="frequency", theta="direction",
                       color="strength", symbol="strength", size="frequency",)
                       
fig
df = px.data.wind()
fig = px.line_polar(df, r="frequency", theta="direction", color="strength", line_close=True,
                                  template="plotly_dark",)
fig.show()
labels=["NE","SE","SW","NW"]
master["dir_o_l"]=pd.cut(master.dir_o[master.dir_o!=-1], len(labels),labels=labels)
master["dir_4K_l"]=pd.cut(master.dir_4K[master.dir_o!=-1], len(labels),labels=labels)
master["dir_36K_l"]=pd.cut(master.dir_36K[master.dir_o!=-1], len(labels),labels=labels)

table_4K=pd.crosstab(master.dir_o_l, master.dir_4K_l, margins=True,)
table_36K=pd.crosstab(master.dir_o_l, master.dir_36K_l, margins=True,)
table_two_model=pd.crosstab(master.dir_4K_l, master.dir_36K_l, margins=True,)


fig, axs = plt.subplots(3,figsize = (15,15))
sns.heatmap(table_4K,annot=True,cmap="YlGnBu",ax=axs[0],fmt='.0f',linewidths=5)
sns.heatmap(table_36K,annot=True,cmap="YlGnBu",ax=axs[1],fmt='.0f',linewidths=5)
sns.heatmap(table_two_model,annot=True,cmap="YlGnBu",ax=axs[2],fmt='.0f',linewidths=5)


fig, bxs = plt.subplots(3,figsize = (15,15))
master["dir_o_l"].value_counts(normalize=True).plot.pie(autopct='%1.0f%%',ax=bxs[0])
master["dir_4K_l"].value_counts(normalize=True).plot.pie(autopct='%1.0f%%',ax=bxs[1])
master["dir_36K_l"].value_counts(normalize=True).plot.pie(autopct='%1.0f%%',ax=bxs[2])
df4K=pd.concat([master["dir_o_l"],master["dir_4K_l"]],axis=1).dropna()
df36K=pd.concat([master["dir_o_l"],master["dir_36K_l"]],axis=1).dropna()
print("Wind direction model 4 Km ")
print(classification_report(df4K.dir_o_l, df4K.dir_4K_l, labels=labels,)) 
print("Wind direction model 36 Km ")
print(classification_report(df36K.dir_o_l, df36K.dir_36K_l, labels=labels,)) 
#creating dynamic variables
master["mslp_36K_V3H"]=master["mod_36K"].pct_change(freq="3H")
master['HGT500_4K-V3H']=master["HGT500_4K"].pct_change(freq="3H")
#X and Y definition
X=master[[ "dir_4K", "dir_36K", "mod_4K", "mod_36K", "wind_gust_4K","wind_gust_36K","mslp_36K_V3H",'HGT500_4K-V3H',]]
df=pd.concat([X,master["dir_o_l"]],axis=1).dropna()
X=StandardScaler().fit_transform(df[df.columns[:-1]])
lb = preprocessing.LabelBinarizer()
lb.fit(df.dir_o_l)
Y=lb.transform(df.dir_o_l)

#neural network
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix ,classification_report 
from sklearn.model_selection import cross_val_score,cross_validate
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, AlphaDropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,)
mlp = Sequential()
mlp.add(Input(shape=(x_train.shape[1], )))
mlp.add(Dense(24, activation='relu'))
#mlp.add(Dropout(0.5))
mlp.add(Dense(24, activation='relu'))
#mlp.add(Dropout(0.5))
mlp.add(Dense(len(labels), activation='softmax'))
mlp.summary()
mlp.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy',]
           )

history = mlp.fit(x=x_train,
                  y=y_train,
                  batch_size=48,
                  epochs=100,
                  validation_data=(x_test, y_test),
                  verbose=0).history
pd.DataFrame(history).plot(grid=True,figsize=(12,12),yticks=np.linspace(0.0, 1.0, num=11))
y_pred=mlp.predict(x_test)

y_pred_l=lb.inverse_transform(y_pred)
y_test_l=lb.inverse_transform(y_test)
print(classification_report(y_test_l, y_pred_l, )) 
result=pd.crosstab(y_test_l, y_pred_l, margins=True,)
fig = plt.figure(figsize = (18,18)) # width x height
ax1 = fig.add_subplot(3, 3, 1) 
g=sns.heatmap(result,annot=True,cmap="YlGnBu",ax=ax1,fmt='.0f',linewidths=4)
#variable winds detector
var=master[["dir_o", "dir_4K", "dir_36K", "mod_4K", "mod_36K", "wind_gust_4K","wind_gust_36K",]].dropna()
X=var[["dir_4K", "dir_36K", "mod_4K", "mod_36K", "wind_gust_4K","wind_gust_36K",]]
Y=pd.DataFrame({"datetime":var.index,
                     "var_o":["var" if c<=-1 else "no_var" for c in var.dir_o]}).set_index("datetime")
#decission tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3,criterion="entropy") #max_depth is maximum number of levels in the tree
clf.fit(X, Y)
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=var.columns[1:],  
                     class_names=["no_var","var"],  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix ,classification_report 
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

var_wind=master[["dir_o_l", "dir_4K", "dir_36K", "mod_4K", "mod_36K", "wind_gust_4K","wind_gust_36K","mslp_36K_V3H",'HGT500_4K-V3H',]].dropna()
Y=var_wind["dir_o_l"]
X=var_wind[["dir_4K", "dir_36K", "mod_4K", "mod_36K", "wind_gust_4K","wind_gust_36K","mslp_36K_V3H",'HGT500_4K-V3H',]]

#we do not scale!!
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,)
clf = DecisionTreeClassifier(max_depth=10,criterion="gini").fit(x_train,y_train) 
y_pred=clf.predict(x_test)
#plot results
print(classification_report(y_test.values,y_pred))
print("**** Confusion matrix ****")
print(confusion_matrix(y_test,y_pred))

#plot tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=var_wind.columns[1:],  
                     class_names=["NE","NW","SE","SW"],  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

from sklearn.model_selection import cross_validate
results=cross_validate(clf, X, Y, cv=10,scoring="accuracy")
print("Mean:","{0:.3f}".format(results['test_score'].mean()),",Standard deviation:","{0:.3f}".format(results['test_score'].std()))
from sklearn.ensemble import RandomForestClassifier
var_wind=master[["dir_o_l", "dir_4K", "dir_36K", "mod_4K", "mod_36K", "wind_gust_4K","wind_gust_36K","mslp_36K_V3H",'HGT500_4K-V3H',]].dropna()
Y=var_wind["dir_o_l"]
X=var_wind[["dir_4K", "dir_36K", "mod_4K", "mod_36K", "wind_gust_4K","wind_gust_36K","mslp_36K_V3H",'HGT500_4K-V3H',]]

#we do not scale!!
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,)
clf =RandomForestClassifier (n_estimators=1500).fit(x_train,y_train) 
y_pred=clf.predict(x_test)
#plot results
print(classification_report(y_test.values,y_pred))
print("**** Confusion matrix ****")
print(confusion_matrix(y_test,y_pred))

from sklearn.model_selection import cross_validate
results=cross_validate(clf, X, Y, cv=10,scoring="accuracy")
print("Mean:","{0:.3f}".format(results['test_score'].mean()),",Standard deviation:","{0:.3f}".format(results['test_score'].std()))
