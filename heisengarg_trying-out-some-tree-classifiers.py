import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv("voice.csv")

#Source: https://www.kaggle.com/primaryobjects/voicegender
df.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier



X=df.ix[:,df.columns!='label']

Y=df.label



clf_dt = DecisionTreeClassifier()

clf_dt = clf_dt.fit(X,Y)



clf_rft = RandomForestClassifier()

clf_rft = clf_rft.fit(X,Y)



clf_et=tree.ExtraTreeClassifier()

clf_et.fit(X,Y)



score_dt=[]

score_rft=[]

score_et=[]

start=3

end=10



for i in range(start,end+1):

    

    score_dt.append(cross_val_score(clf_dt, X, Y,cv=i).mean())

    score_rft.append(cross_val_score(clf_rft,X,Y,cv=i).mean())

    score_et.append(cross_val_score(clf_et,X,Y,cv=i).mean())

p1=plt.plot(range(start,end+1),score_dt,'r',label='Decision Tree Classifier')

p2=plt.plot(range(start,end+1),score_rft,'b',label='Random Forest Classifier')

p3=plt.plot(range(start,end+1),score_et,'g',label='Extra Trees Classifier')

plt.legend(loc=4)

plt.ylabel('Mean Accuracy (%)')

plt.xlabel('Number of folds for Cross Validation')

plt.ylim((0.89,0.97))
from sklearn import tree

tree.export_graphviz(clf_dt, out_file='tree.dot') #clf_dt has 10 folds
import pydotplus

from sklearn.externals.six import StringIO

from IPython.display import Image



import gi

#gi.require_version('Gtk', '3.0')

#from gi.repository import Gtk, Pango

#import pango



dotfile = StringIO()

tree.export_graphviz(clf_dt, out_file=dotfile)

graph = pydotplus.graph_from_dot_data(dotfile.getvalue())

Image(graph.create_png())
from sklearn.feature_selection import SelectFromModel



#clf=RandomForestClassifier()

#clf.fit(X,Y)



#Random Forest Classifier



fi_rft=clf_rft.feature_importances_

fi_dt=clf_dt.feature_importances_

fi_et=clf_et.feature_importances_



fig=plt.figure(figsize=(18,5))

ax=plt.subplot(111)



wi=0.3

w=0.3



ax.bar(np.arange(1,21),fi_dt,width=wi,align='center',color='r',alpha=0.5)

ax.bar(np.arange(1,21)+w,height=fi_rft,width=wi,align='center',color='b',alpha=0.2)

ax.bar(np.arange(1,21)+2*w,height=fi_rft,width=wi,align='center',color='g',alpha=0.3)



ax.legend(loc=4)



ax.set_xticks(np.arange(1,21)+wi) #set position of xlabels to match with on the plot

ax.set_xticklabels(np.delete(df.columns,20).values) #now rename those labels



ax.set_ylabel('Feature Importance')

ax.set_xlabel('Features')



ax.legend(('Decision Tree','Random Forest','Extra Tree'))

fig.suptitle('Comparison of feature importance based on classifier used',fontsize=20)

#ax.show()
# We only choose the IQR,sd, Q25 and meanfun as the features for our classifications



#Decision Tree (IQR and meanfun)



X_dt=df[['IQR','meanfun']]



clf_dt=clf_dt.fit(X_dt,Y)

#score_dt=cross_val_score(clf_dt,X_dt,Y)



#Random Forest and Extra Tree (meanfun,sd,Q25)



X_new=df[['meanfun','sd','Q25','IQR','sp.ent']]



clf_et=clf_et.fit(X_new,Y)

#score_et=cross_val_score(clf_et,X_new,Y)



clf_rft=clf_dt.fit(X_new,Y)

#score_rft=cross_val_score(clf_rft,X_new,Y)



score_dt=[]

score_rft=[]

score_et=[]

start=3

end=10





for i in range(3,11):

    



    score_dt.append(cross_val_score(clf_dt, X_dt, Y,cv=i).mean())

    score_rft.append(cross_val_score(clf_rft,X_new,Y,cv=i).mean())

    score_et.append(cross_val_score(clf_et,X_new,Y,cv=i).mean())

    

p1=plt.plot(range(start,end+1),score_dt,'r',label='Decision Tree Classifier')

p2=plt.plot(range(start,end+1),score_rft,'b',label='Random Forest Classifier')

p3=plt.plot(range(start,end+1),score_et,'g',label='Extra Trees Classifier')

plt.legend(loc=4)

plt.ylabel('Mean Accuracy (%)')

plt.ylim((0.89,0.97))

plt.xlabel('Number of folds for Cross Validation')