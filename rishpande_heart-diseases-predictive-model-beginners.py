import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/heart.csv")
df.head(5)
df.info()
df.describe()
f, ax = plt.subplots(1, 2, figsize = (15, 7))

f.suptitle("Heart disease?", fontsize = 18.)

_ = df.target.value_counts().plot.bar(ax = ax[0], rot = 0, color = (sns.color_palette()[0], sns.color_palette()[2])).set(xticklabels = ["No", "Yes"])

_ = df.target.value_counts().plot.pie(labels = ("No", "Yes"), autopct = "%.2f%%", label = "", fontsize = 13., ax = ax[1],\

colors = (sns.color_palette()[0], sns.color_palette()[2]), wedgeprops = {"linewidth": 1.5, "edgecolor": "#F7F7F7"}), ax[1].texts[1].set_color("#F7F7F7"), ax[1].texts[3].set_color("#F7F7F7")
fig, ax = plt.subplots(4,2, figsize=(16,16))

sns.distplot(df.age, bins = 20, ax=ax[0,0]) 

sns.distplot(df.oldpeak, bins = 20, ax=ax[0,1]) 

sns.distplot(df.trestbps, bins = 20, ax=ax[1,0]) 

sns.distplot(df.chol, bins = 20, ax=ax[1,1]) 

sns.distplot(df.ca, bins = 20, ax=ax[2,0])

sns.distplot(df.thal, bins = 20, ax=ax[2,1])

sns.distplot(df.thalach, bins = 20, ax=ax[3,0]) 

sns.distplot(df.slope, bins = 20, ax=ax[3,1]) 

plt.show()
plt.figure(figsize=(16,12))

sns.heatmap(df.corr(),annot=True,cmap='YlGnBu',fmt='.2f',linewidths=2)
df.info()
fig,ax = plt.subplots(nrows=4, ncols=2, figsize=(18,18))

plt.suptitle('Violin Plots',fontsize=24)

sns.violinplot(x="cp", data=df,ax=ax[0,0],palette='Set3')

sns.violinplot(x="trestbps", data=df,ax=ax[0,1],palette='Set3')

sns.violinplot (x ='chol', data=df, ax=ax[1,0], palette='Set3')

sns.violinplot(x='fbs', data=df, ax=ax[1,1],palette='Set3')

sns.violinplot(x='restecg', data=df, ax=ax[2,0], palette='Set3')

sns.violinplot(x='thalach', data=df, ax=ax[2,1],palette='Set3')

sns.violinplot(x='exang', data=df, ax=ax[3,0],palette='Set3')

sns.violinplot(x='age', data=df, ax=ax[3,1],palette='Set3')

plt.show()
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier



X = df.iloc[:, :-1]

y = df.iloc[:, -1]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#Model

LR = LogisticRegression()



#fiting the model

LR.fit(X_train, y_train)



#prediction

y_pred = LR.predict(X_test)



#Accuracy

print("Accuracy ", LR.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()
#Model

DT = DecisionTreeClassifier()



#fiting the model

DT.fit(X_train, y_train)



#prediction

y_pred = DT.predict(X_test)



#Accuracy

print("Accuracy ", DT.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()
from sklearn import tree

import graphviz
#Plotting the graph

tree_graph = tree.export_graphviz(DT, out_file=None)

graphviz.Source(tree_graph)
feature_names = [i for i in df.columns if df[i].dtype in [np.int64]]
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=DT, dataset=df, model_features=feature_names, feature='age')



# plot it

pdp.pdp_plot(pdp_goals, 'age')

plt.show()
pdp_dist = pdp.pdp_isolate(model=DT, dataset=df, model_features=feature_names, feature='trestbps')



pdp.pdp_plot(pdp_dist, 'trestbps')

plt.show()
features_to_plot = ['age', 'trestbps']

inter1  =  pdp.pdp_interact(model=DT, dataset=df, model_features=feature_names, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')

plt.show()
#Model

model = GradientBoostingClassifier()



#fiting the model

model.fit(X_train, y_train)



#prediction

y_pred = model.predict(X_test)



#Accuracy

print("Accuracy ", model.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()