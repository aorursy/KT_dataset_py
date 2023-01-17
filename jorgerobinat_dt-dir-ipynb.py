import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix ,classification_report 

from sklearn.model_selection import cross_val_score,cross_validate

import seaborn as sns

from sklearn import preprocessing







#@title select 

labels = ["NE","SE","SW","NW","VRB"]

threshold = 2 #@param {type:"integer"}





join=pd.read_csv("../input/wind-coron/cortegadaD1res4K.csv")

table_columns=[]

table=[]

table_index=[]

X_var=["dir_NE","dir_SE","dir_SW","dir_NW"]

for var_pred0 in X_var:

  var_obs="dir_o"

  join[var_obs+"_l"]=pd.cut(join[var_obs], bins = len(labels), labels = labels).astype(str)

  join[var_pred0+"_l"]=pd.cut(join[var_pred0],bins = len(labels),labels=labels).astype(str)

  join.loc[join['spd_o'] < threshold, [var_obs+"_l"]] = "VRB"      

  join.loc[join["mod_"+var_pred0[-2:]]< threshold,[var_pred0+"_l"]]="VRB"



  #results tables

  res_df=pd.DataFrame({"pred_var":join[var_pred0+"_l"],"obs_var":join[var_obs+"_l"]})

  table.append(pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,))

  table_columns.append(pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="columns"))

  table_index.append(pd.crosstab(res_df.obs_var,res_df.pred_var, margins=True,normalize="index")  )



#plt results tables and classification report

for i in range(0,len(X_var)):

  print("Independent variable:",X_var[i]," Observed result:",var_obs)

  print(classification_report(join[var_obs+"_l"],join[X_var[i]+"_l"]))

  fig, axs = plt.subplots(3,figsize = (8,10))

  sns.heatmap(table[i],annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)

  sns.heatmap(table_columns[i],annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')

  sns.heatmap(table_index[i],annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")

  plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

from sklearn import tree

import graphviz

import pickle



#@title Load or train model. If load, same quantile and var_pred1



X_var = ['dir_NE', 'dir_SE','dir_NW', 'dir_SW',"mod_NE","mod_SE","mod_SW","mod_NW"]

var_obs = "dir_o" 



max_depth=9 #@param ["2", "5", "10", "15"] {type:"raw", allow-input: true}

criterion = "gini" #@param ["gini", "entropy"]

RF = False #@param {type:"boolean"}

n_estimators =  200#@param {type:"integer"}

cross_validation= True #@param {type:"boolean"}



#cut in bins

Y=join[var_obs+"_l"]





#independent variables. 

X=join[X_var]





#we do not scale!!

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,)



#select classification

#train and save as tree2 or load tree

if RF:

  clf1=RandomForestClassifier(n_estimators=n_estimators).fit(x_train,y_train) 

else:

  clf1 = DecisionTreeClassifier(max_depth=max_depth,criterion=criterion).fit(x_train,y_train) 





y_pred=clf1.predict(x_test)



#plot results

print(classification_report(y_test,y_pred))

y_pred_df=pd.DataFrame({"var_pred":y_pred},index=y_test.index)



#plot results

table=pd.crosstab(y_test,y_pred_df["var_pred"], margins=True,)

table_columns1=pd.crosstab(y_test,y_pred_df["var_pred"], margins=True,normalize="columns")

table_index=pd.crosstab(y_test,y_pred_df["var_pred"], margins=True,normalize="index")





fig, axs = plt.subplots(3,figsize = (8,10))

sns.heatmap(table,annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)

sns.heatmap(table_columns1,annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')

sns.heatmap(table_index,annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")

plt.show()





print("Features importances:")

fi=["{:.0%}".format(x) for x in clf1.feature_importances_]

print(dict(zip(X.columns,fi )))



if RF==False:

  #tree save file name: tree



  dot_data = tree.export_graphviz(clf1, out_file=None, 

                                    feature_names=X.columns,  

                                    class_names=labels,  

                                    filled=True, 

                                    rounded=True,  

                                    special_characters=True)  

  graph = graphviz.Source(dot_data) 

  graph.render("tree")

   



#cross validation

if cross_validation:

  print ("***Accuracy score***")

  print(cross_val_score(clf1, X, Y, cv=5,scoring="accuracy"))

  print ("***F1_macro score***")

  print(cross_val_score(clf1, X, Y, cv=5,scoring='f1_macro'))



#Save trained model



pickle.dump(clf1, open("tree_dir.h5", 'wb'))