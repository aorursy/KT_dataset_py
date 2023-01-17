import pandas as pd

import numpy as np

#---For jupyter-notebook---

%matplotlib inline

#%matplotlib notebook

#---For jupyter-Lab---

#%matplotlib widget

from matplotlib import pyplot as plot, axes

import seaborn as sns

import numpy as np

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn.metrics import (silhouette_samples,adjusted_rand_score,v_measure_score,

    classification_report,confusion_matrix,roc_curve,roc_auc_score,accuracy_score,

    precision_recall_curve)

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,Normalizer

from sklearn.tree import DecisionTreeClassifier,export_graphviz,plot_tree,export_text

import statsmodels.api as sm

from IPython.display import display

import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
#get .csv file as dataFrame object

data=pd.read_csv('../input/hrdataset-v13/HRDataset_v13.csv')

display(data.head(5))

display(data.describe())
display(data.shape)

display(data.isna().sum())
statData=pd.DataFrame(data[['GenderID','DeptID','PositionID','MaritalStatusID','PerfScoreID','PayRate','EmpStatusID'

          ,'RaceDesc', 'EngagementSurvey','EmpSatisfaction','SpecialProjectsCount']],copy=True)



statData.isna().sum()
statData['Age']=((pd.to_datetime('today')-pd.to_datetime(data['DOB']))/np.timedelta64(1,'Y')).astype(int)

statData['EmployedDays']=((pd.to_datetime('today')-pd.to_datetime(data['DateofHire'])

                           )/np.timedelta64(1,'Y')).astype(int)



# Label encoding Encoding

statData['RaceDesc']=LabelEncoder().fit_transform(statData['RaceDesc'])



display(statData.head(10))

display(statData.shape)
plot.figure(1 , figsize = (18 , 14))

n = 0 

for x in ['PayRate','DeptID','MaritalStatusID','EngagementSurvey','EmpSatisfaction','SpecialProjectsCount','Age',

          'EmployedDays','EmpStatusID','PositionID','RaceDesc','PerfScoreID']:

    n += 1

    plot.subplot(4 , 3 , n)

    plot.subplots_adjust(hspace =0.2 , wspace = 0.2)

    sns.distplot(statData[x] , bins = 20,color="green")

#     plot.title('Distplot of {}'.format(x))

plot.show()
plot.figure(1 , figsize = (10 , 6))

c=0

for y in ['GenderID']:

    c+=1

    plot.subplot(2,1,c)

    sns.countplot(y = y , data = statData,palette='viridis')

    

    if(c==1):

        plot.title('MALE: 0 | FEMALE: 1')

plot.show()
plot.figure(figsize = (11 , 11))

sns.heatmap(statData.corr(),square=True,cmap='YlGnBu',cbar_kws={"shrink": 0.80},annot=True)

plot.title('Correlation',fontsize=20)

plot.show()
X_train,X_test, Y_train,Y_test= train_test_split(statData.drop(['PerfScoreID','SpecialProjectsCount','EmpStatusID'],

                        axis='columns'),statData[['PerfScoreID']],train_size=0.80, random_state=70)



#Normalizing training data.

scaled_data=Normalizer().fit_transform(X_train)

X_train_norm= pd.DataFrame(scaled_data,columns=X_train.columns.values.tolist(),copy=True)



#Standardizing training data. 

scaled_data=StandardScaler().fit_transform(X_train)

X_train_std= pd.DataFrame(scaled_data,columns=X_train.columns.values.tolist(),copy=True)



#Normalizing testing data.

scaled_data=Normalizer().fit_transform(X_test)

X_test_norm= pd.DataFrame(scaled_data,columns=X_train.columns.values.tolist(),copy=True)



#Standardizing testing data.

scaled_data=StandardScaler().fit_transform(X_test)

X_test_std= pd.DataFrame(scaled_data,columns=X_train.columns.values.tolist(),copy=True)

print("X_train :"+str(X_train.shape))

print("X_test :"+str(X_test.shape))

print("Y_train :"+str(Y_train.shape))

print("Y_test :"+str(Y_test.shape))
serpl3d,xyzse=plot.subplots(figsize=(10,6))

distortions = []

for i in range(1, 11):

    km = KMeans(

        n_clusters=i, init='k-means++',

        n_init=10, max_iter=600,

        tol=1e-04

)

    km.fit(X_train_norm,Y_train)

    distortions.append(km.inertia_)



# plot

xyzse.plot(range(1, 11), distortions, marker='o',linestyle='-',color='slateblue',markerSize=5,

markerfacecolor='purple',mec='lime')

xyzse.set_title('Mean Square Error',fontsize=16)

xyzse.set_xlabel('Number of clusters',fontsize=12)

xyzse.set_ylabel('Distortion',fontsize=12)

xyzse.set_facecolor(".95")

xyzse.grid(True, color='w', linestyle='-', linewidth=1)

plot.show()
n_clusters=2

kMeans = KMeans(

    n_clusters=n_clusters, init='k-means++',

    n_init=10, max_iter=600,

    tol=1e-04,random_state=40

)

kMeans.fit(X_train_norm,Y_train)

train_label=kMeans.labels_

test_label=kMeans.predict(X_test_norm)
plot.figure(1 , figsize = (10 , 4))

cplot=sns.countplot(y = 'clusters' , data = pd.DataFrame({"clusters":test_label},copy=True),palette='magma')

cplot.set_xlabel("count",fontsize=12)

cplot.set_ylabel("clusters",fontsize=12)

plot.show()
x_new = PCA(n_components=2).fit_transform(X_train_norm)

bplot_data= pd.DataFrame({

    'pca_1':x_new[:,0].ravel(),

    'pca_2':x_new[:,1].ravel(),

    'labels': train_label

})



with sns.axes_style('darkgrid'):

    # plot the N clusters for 2D-plot

    fig, bPlot =plot.subplots(figsize=(12,8))

    

    sns.scatterplot(x='pca_1',y='pca_2',data=bplot_data,hue='labels',linewidth=1,

       ax=bPlot,palette=['seagreen','orangered'],size="labels",sizes=(100,100),alpha=.8)



bPlot.set_xlabel('pca_1',{'fontsize':13})

bPlot.set_ylabel('pca_2',{'fontsize':13})

bPlot.set_title("Biplot of k-means clusters",{'fontsize':16})



plot.show()

#Parallal cordinate plot

df=pd.DataFrame(X_train_std,copy=True)

df['km_clust']=train_label

plot.figure(figsize=(20,10))

pd.plotting.parallel_coordinates(df,'km_clust',cols=['GenderID','DeptID','PositionID',

    'MaritalStatusID','PayRate','RaceDesc'],colormap='Dark2')

plot.title("Parallal cordinate plot k-means [1 of 2]",{'fontsize':26})

plot.yticks(fontsize=16)

plot.xticks(fontsize=16)

plot.show()



plot.figure(figsize=(20,10))

pd.plotting.parallel_coordinates(df,'km_clust',cols=['RaceDesc','EngagementSurvey','EmpSatisfaction',

    'Age','EmployedDays'],colormap='Dark2')

plot.title("Parallal cordinate plot k-means [2 of 2]",{'fontsize':26})

plot.yticks(fontsize=16)

plot.xticks(fontsize=16)

plot.show()
fig, ax1 = plot.subplots(1, 1)

fig.set_size_inches(16, 6)



ax1.set_xlim([-0.2, 1])

ax1.set_ylim([0, len(Y_test) + (n_clusters + 1) * 4])

sh_samples=silhouette_samples(X_test_norm,test_label)

color=['lightseagreen','blueviolet','springgreen','darkorange']

y_lower, y_upper = n_clusters, 0

for i in range(n_clusters):

    ith_cluster_silhouette_values =sh_samples[test_label == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]



    y_upper =y_lower+ size_cluster_i

    ax1.barh(range(y_lower, y_upper), ith_cluster_silhouette_values, edgecolor=None, 

             height=1, align='edge',color=color[i])  

    ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))

    

    # Compute the new y_lower for next plot

    y_lower =y_upper + n_clusters

       

ax1.set_title("The silhouette plot of k-means clusters",fontsize=18)

ax1.set_xlabel("The silhouette coefficient values",fontsize=12)

ax1.set_ylabel("Cluster label",fontsize=12)

# The vertical line for average silhouette score of all the values

ax1.axvline(x=sh_samples.mean(), color="crimson", linestyle="-.")

ax1.set_yticks([])  # Clear the yaxis labels / ticks



plot.show()
#----Applying LOGISTIC REGRESSION ON Labeled Data for Analysis----

model=LogisticRegression(random_state=20,solver='liblinear',multi_class='auto',penalty='l1')

model.fit(X_train_std,train_label)

y_lr_pred = model.predict(X_test_std)

x_lr_pred=model.predict(X_train_std)

x_lr_pred_proba=model.predict_proba(X_train_std)[:,1]
x_new = PCA(n_components=2).fit_transform(X_train_norm)

frame=pd.DataFrame({

    'pca1':x_new[:,0].ravel(),

    'pca2':x_new[:,1].ravel(),

    'labels': x_lr_pred

})

with sns.axes_style('darkgrid'):

    sns.lmplot(x="pca1", y="pca2", hue="labels", data=frame,

               markers=["o", "D"], palette=['orangered','slateblue'],size=6.5,aspect=1.5)

    ax = plot.gca()

    ax.set_xlabel('pca1',{'fontsize':13})

    ax.set_ylabel('pca2',{'fontsize':13})

    ax.set_title("Logistic Regression Decision Boundary",{'fontsize':17})

x_new = PCA(n_components=2).fit_transform(X_train_norm)

with sns.axes_style('darkgrid'):

    fig, db_plot=plot.subplots(figsize=(10,6),dpi=80)



    sns.regplot(x=x_new[:,0], y=x_lr_pred_proba,color='crimson', line_kws={"color":"indigo"},fit_reg=True,

                logistic=True,scatter_kws={'edgecolor':'white','s':70},n_boot=200)

    

    

db_plot.set_xlabel('pca1',{'fontsize':13})

db_plot.set_ylabel(' probabilities',{'fontsize':13})

db_plot.set_title("Logistic Regression Probability Curve",{'fontsize':16})

plot.show()
dt = DecisionTreeClassifier(criterion='entropy',max_depth=None).fit(X_train,train_label)

dt.fit(X_train,train_label);

dt_pred=dt.predict(X_test)

dt_pred_proba=dt.predict_proba(X_test)[:,1]



file=open("dt_plot.txt", "w")

file.write(export_text(dt, feature_names=X_train.columns.values.tolist()))

file.close()

# import graphviz

# from IPython.display import Image



# export_graphviz(dt, out_file=tree_nonlimited.dot, 

#       feature_names=X_train.columns.values.tolist(),class_names=True,  

#       filled=True, rounded=True, special_characters=True,proportion=False)



# graphviz.Source(dot_data)

# !dot -Tpng tree_nonlimited.dot -o tree_nonlimited.png -Gdpi=90

# Image(filename = 'tree_nonlimited.png')



plot.figure(figsize=(22,14))

dot_data=plot_tree(dt,feature_names=X_train.columns.values.tolist(),

                        class_names=True,filled=True,proportion=False,rounded=True)

plot.show()
from itertools import product

# Loading some example data

x_new = PCA(n_components=2).fit_transform(X_train_std)

X = x_new

y = train_label



# Training classifiers

clf1 = DecisionTreeClassifier(max_depth=None)

clf1.fit(X, y)



# Plotting decision regions

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.15),

                     np.arange(y_min, y_max, 0.2))



f, axarr = plot.subplots(figsize=(10, 8))



Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)



axarr.contourf(xx, yy, Z, alpha=0.9,cmap='YlGnBu')

axarr.scatter(X[:, 0], X[:, 1],c=y, s=70, edgecolor='navy',linewidths=1.5,cmap="autumn")

axarr.set_xlabel("pca1",fontsize=14)

axarr.set_ylabel("pca2",fontsize=14)

axarr.set_title('DT-CART Decision Surface',fontsize=18)





plot.show()
print('K-Means Clustering scores:\n')

print('  SILHOUETTE Score:    {}'.format(round(silhouette_samples(X_test,test_label).mean(),6)))

print("  RANDOM_INDEX Score:  {}".format(round(adjusted_rand_score(Y_test.to_numpy().ravel(),test_label),6)))

print("  V-MEASURE Score:     {}".format(round(v_measure_score(Y_test.to_numpy().ravel(),test_label),6)))
print('Logistic Regression Classification Report: \n')

print(classification_report(test_label,y_lr_pred))
logit_model=sm.GLM(train_label,sm.add_constant(X_train_std))

result=logit_model.fit()

print(result.summary())
print('CART-DT Classification Report: \n')

print(classification_report(test_label,dt_pred))
fhm,hm=plot.subplots(1,2, figsize = (12 , 5))

fhm.subplots_adjust(hspace =0.2 , wspace = 0.6)

sns.heatmap(confusion_matrix(test_label,y_lr_pred),ax=hm[0],annot=True,annot_kws={"size": 15},

            square=True,cmap='viridis',cbar_kws={"shrink": 0.75})



sns.heatmap(confusion_matrix(test_label,dt_pred),ax=hm[1],annot=True,annot_kws={"size": 15},

            square=True,cmap='plasma',cbar_kws={"shrink": 0.75})



hm[0].set_title("LOGISTIC-REGRESSION\n [score: {}]".format(round(

    model.score(X_test_std,test_label),2)),fontsize=16)

hm[1].set_title("DECISION TREE (CART)\n [score: {}]".format(round(

    dt.score(X_test,test_label),2)),fontsize=16)

fhm.suptitle("CONFUSION MATRIX",fontsize=20)

plot.show()
fig,pr_plot=plot.subplots(figsize=(10,6),dpi=80)



no_skill = len(y[y==1]) / len(y)

pr_plot.plot([0, 1], [no_skill, no_skill], linestyle=':', label='No Skills',c="crimson")



precision, recall, _ = precision_recall_curve(test_label, model.predict_proba(X_test_std)[:,1])

pr_plot.plot(recall, precision, linestyle='-', label='Logistic-R', c='deepskyblue', alpha=0.9,linewidth=2.5)



precision, recall, _ = precision_recall_curve(test_label, dt_pred_proba)

pr_plot.plot(recall, precision, linestyle='--', label='DT-CART', c='indigo', linewidth=1.5)





pr_plot.set_xlabel('Recall',fontsize=12)

pr_plot.set_ylabel('Precision',fontsize=12)

pr_plot.set_title('Precision-Recall curve',fontsize=14)

pr_plot.legend()

pr_plot.set_facecolor(".95")

pr_plot.grid(True, color='w', linestyle='-', linewidth=1)

# show the legend

pr_plot.legend()

# show the plot

plot.show()
# generate a no skill prediction (majority class)

ns_probs = [0 for _ in range(len(test_label))]

# predict probabilities

lr_probs = model.predict_proba(X_test_std)[:,1]



# calculate scores

dt_auc = roc_auc_score(test_label, dt_pred_proba)

lr_auc = roc_auc_score(test_label, lr_probs)



# summarize scores

print('DT-CART: ROC AUC=%.3f' % (dt_auc))

print('Logistic Regression: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves

ns_fpr, ns_tpr, _ = roc_curve(test_label, ns_probs)

dt_fpr, dt_tpr, _ = roc_curve(test_label, dt_pred_proba)

lr_fpr, lr_tpr, _ = roc_curve(test_label, lr_probs)



# plot the roc curve for the model

fig,roc_plot=plot.subplots(figsize=(10,6),dpi=80)

roc_plot.plot(ns_fpr, ns_tpr, linestyle=':', label='No Skills',c='crimson')

roc_plot.plot(lr_fpr, lr_tpr, linestyle='-', label='Logistic-R',c='springgreen',alpha=0.9,linewidth=2.5)

roc_plot.plot(dt_fpr, dt_tpr, linestyle='--', label='DT-CART',c='darkslateblue',linewidth=1.5)



# axis labels

roc_plot.set_xlabel('False Positive Rate',fontsize=12)

roc_plot.set_ylabel('True Positive Rate',fontsize=12)

roc_plot.set_title('ROC curve',fontsize=14)

roc_plot.legend()

roc_plot.set_facecolor(".95")

roc_plot.grid(True, color='w', linestyle='-', linewidth=1)

plot.show()
pd.DataFrame({

    'Actual':Y_test.values.ravel(),

    'Clustered':test_label,

    'Regression': y_lr_pred,

    'CART-DT':dt_pred

}).head(10)
features=pd.DataFrame(statData.drop(['PerfScoreID','SpecialProjectsCount','EmpStatusID'],axis='columns'),copy=True)

features.describe()


pie=pd.DataFrame({

    'Actual':Y_train.values.ravel(),

    'Clustered':train_label,

},copy=True)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:

colors1 = ['#ffab00','#e91e63','#ae52d4','#64dd17']

colors2 = ['#c158dc','#00b248']



fig1, ax1 = plot.subplots(1,2,figsize = (12 , 6),subplot_kw=dict(aspect="equal"))



unique_elements, counts_elements = np.unique(pie['Actual'].to_numpy(), return_counts=True)

ax1[0].pie(counts_elements,labels=unique_elements, autopct='%1.1f%%',startangle=90,

           textprops=dict(fontsize=13),pctdistance=0.75,explode = (0.03,0.03,0.03,0.03),colors=colors1)

ax1[0].set_title('Actual',fontsize=16)



unique_elements, counts_elements =np.unique(pie['Clustered'].to_numpy(), return_counts=True)

ax1[1].pie(counts_elements,labels=unique_elements, autopct='%1.1f%%',

        startangle=90,textprops=dict(fontsize=14),pctdistance=0.48,explode = (0.02,0.02),colors=colors2)

ax1[1].set_title('Clustered',fontsize=16)



plot.tight_layout()

plot.show()