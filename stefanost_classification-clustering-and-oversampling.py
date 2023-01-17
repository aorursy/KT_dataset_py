import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/birds-bones-and-living-habits/bird.csv')
df.head()
df.drop(columns='id', inplace=True)
df.info()
#visualize where NaN values are located in the dataset
plt.figure(figsize=(6,5))
sns.heatmap(df.isna());
df[(df.isna().cumsum(axis=1).iloc[:,-1])>=2]
df.dropna(axis=0, thresh=10, inplace=True)
#drop all NaNs
df.dropna(how='any', inplace=True)
#how many samples for each class
df['type'].value_counts()
sns.pairplot(df.iloc[:,:-1]);
#correlation heatmap
plt.figure(figsize=(10,7))
sns.heatmap(df.iloc[:,:-1].corr(), annot=True, fmt='.2f');
df=pd.read_csv('../input/birds-bones-and-living-habits/bird.csv')  \
.drop(columns=['id','ulnal','ulnaw','femw','tibw']).dropna(how='any')
sns.pairplot(df, vars= df.columns[:-1], hue='type');
plt.figure(figsize=(7,5))
sns.heatmap(df.iloc[:,:-1].corr(), annot=True, fmt='.2f');
#distributions per class
def boxes(features):
    plt.figure(figsize=(12,5));
    plt.subplot(1,2,1);
    sns.boxplot(x='type', y=features[0], data=df);
    plt.title(features[0]);
    plt.subplot(1,2,2);
    sns.boxplot(x='type', y=features[1], data=df);
    plt.title(features[1]);

boxes(df.columns[:2])
boxes(df.columns[2:4])
boxes(df.columns[4:6])
#PCA
sc=StandardScaler()
x_sc=sc.fit_transform(df.iloc[:,:-1].values)

pca=PCA(n_components=2)
x_pc=pca.fit_transform(x_sc)

plt.figure(figsize=(10,7));
sns.scatterplot(x_pc[:,0],x_pc[:,1],hue=df['type']);
pca=PCA()
pca.fit(x_sc)

sns.lineplot(x=np.arange(pca.components_.shape[0])+1, 
             y=pca.explained_variance_ratio_.cumsum());
plt.ylim(0.0,1.0);
plt.title('Cumulative explained variance of PCA');
plt.xlabel('number of principal component');
plt.ylabel('cumulative sum of explained variance');
sns.barplot(x=np.arange(pca.components_.shape[0])+1, y=pca.explained_variance_ratio_, color='grey')
plt.title('Explained variance of PCA');
plt.xlabel('Number of principal component');
plt.ylabel('Explained variance');

print('Number of Principal Components needed to explain total variance:',
      pca.components_.shape[0] )
print('Cumulative explained variance of first two components:',
      pca.explained_variance_ratio_[:2].cumsum()[1])
print('Individual explained variance of first three components:',
      pca.explained_variance_ratio_[:3])
#PCA on raw data
pc=PCA(n_components=2)
x_dem=pc.fit_transform(df.iloc[:,:-1].values)
plt.figure(figsize=(10,7));
sns.scatterplot(x_dem[:,0], x_dem[:,1], hue=df['type']);
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
#use 'elbow' method to find optimal cluster number
inertia=[]
for i in range(1,11):
    km=KMeans(n_clusters=i, random_state=33)
    km.fit_predict(x_sc)
    inertia.append(km.inertia_)
    
#inertia plot
sns.lineplot(range(1,11),inertia);
plt.title('Inertia');
#five clusters
km=KMeans(n_clusters=5, random_state=33)
clusters=km.fit_predict(x_sc)

#Silhouette Graph
#code from Sebastian Raschka's book 'Python Machine Learning'
labels=np.unique(clusters)
n_clusters=labels.shape[0]
sils=silhouette_samples(x_sc,clusters,metric='euclidean')
y_ax_lower, y_ax_upper=0, 0
yticks=[]
plt.figure(figsize=(6,5))
for i,c in enumerate(labels):
    cluster_sil=sils[clusters==c]
    cluster_sil.sort()
    y_ax_upper +=len(cluster_sil)
    color=cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
            cluster_sil, height=1.0,
            edgecolor='none', color=color)
    yticks.append((y_ax_lower+y_ax_upper)/2.)
    y_ax_lower+=len(cluster_sil)
silhouette_avg=np.mean(sils)
plt.axvline(silhouette_avg,color='red', linestyle='--')
plt.yticks(yticks,labels+1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show();

#visualize clusters on PCA data
pca=PCA(n_components=2)
dec=pca.fit_transform(x_sc)
plt.figure(figsize=(9,6));
sns.scatterplot(dec[:,0],dec[:,1], hue=clusters);
km=KMeans(n_clusters=6, random_state=33)
clusters=km.fit_predict(x_sc)

#Silhouette Graph
labels=np.unique(clusters)
n_clusters=labels.shape[0]
sils=silhouette_samples(x_sc,clusters,metric='euclidean')
y_ax_lower, y_ax_upper=0, 0
yticks=[]
plt.figure(figsize=(6,5))
for i,c in enumerate(labels):
    cluster_sil=sils[clusters==c]
    cluster_sil.sort()
    y_ax_upper +=len(cluster_sil)
    color=cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
            cluster_sil, height=1.0,
            edgecolor='none', color=color)
    yticks.append((y_ax_lower+y_ax_upper)/2.)
    y_ax_lower+=len(cluster_sil)
silhouette_avg=np.mean(sils)
plt.axvline(silhouette_avg,color='red', linestyle='--')
plt.yticks(yticks,labels+1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show();

#visualize clusters on PCA data
pca=PCA(n_components=2)
dec=pca.fit_transform(x_sc)
plt.figure(figsize=(9,6));
sns.scatterplot(dec[:,0],dec[:,1], hue=clusters);
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# create train and test sets, data and labels
x=x_sc.copy()

enc=LabelEncoder()
y=enc.fit_transform(df['type'])
print ('original classes:',enc.classes_)

x_tr, x_ts, y_tr, y_ts=train_test_split(x,y,random_state=42)
#graph class proportions in the dataset
plt.figure(figsize=(12,5));
plt.subplot(1,2,1);
plt.title('class_count');
sns.countplot(x='type', data=df, color='grey');
plt.subplot(1,2,2);
df['type'].value_counts().plot(kind='pie',autopct='%1.1f%%');
plt.title('class_proportions');
#dummy classifications
np.random.RandomState(seed=7)
print('random selection accuracy:' ,
 accuracy_score(y,np.random.choice([0,1,2,3,4,5],(x.shape[0]))))

dum=DummyClassifier(strategy='uniform', random_state=7)
dum.fit(x,y)
print ('random classifier accuracy:',
      accuracy_score (y, dum.predict(x)))

print("")
print('majority class selection accuracy:' ,
 accuracy_score(y,np.random.choice([2,3],(x.shape[0]))))
#training on the imbalanced data
lr=LogisticRegression()
lr.fit(x_tr,y_tr)
print('LogistReG accuracy:', accuracy_score(y_ts, lr.predict(x_ts) ))
print('LogistReG f1_score:', f1_score(y_ts, lr.predict(x_ts),average='macro' ))

rf=RandomForestClassifier(1000, max_depth=8)
rf.fit(x_tr,y_tr)
print('RandForest accuracy:', accuracy_score(y_ts,rf.predict(x_ts)))
print('RandForest f1-score:', f1_score(y_ts,rf.predict(x_ts), average='macro'))
#first upsampling implementation
#mere duplicating samples from minority classes

print('x_tr.shape=', x_tr.shape)
print('y_tr.shape=', y_tr.shape)
print('')


#initializations
majority_class=np.argmax(np.bincount(y_tr))
length_majority_class = np.max(np.bincount(y_tr))
additional_samples=np.array([])
additional_labels=np.array([])


for i in np.unique(y_tr):

    if i != majority_class:
        length_this_class=np.bincount(y_tr)[i]
        wanted=length_majority_class-length_this_class
        print('class:',i)
        print('%d additional samples' % wanted)
        indexes=np.argwhere(y_tr==i)
        indexes=np.reshape(indexes, (len(indexes),))
        choices=np.random.choice(indexes,size=wanted)
        temp_x=x_tr[choices]
        additional_samples=np.append(additional_samples,temp_x, axis=None)
        additional_labels=np.append(additional_labels,np.full(wanted,i), axis=None)
        

#new samples
print('')
additional_samples=np.reshape(additional_samples,(int(len(additional_samples)/6),6))
print('new_samples.shape=', additional_samples.shape)
print('new_labels.shape=', additional_labels.shape)

#new x_tr --> x_tr1
flat_x = np.append(x_tr,additional_samples)
x_columns_number = x_tr.shape[1]
new_row_number = int(len(flat_x)/x_columns_number)
x_tr1 = np.reshape(flat_x,(new_row_number,x_columns_number))

#new y_tr --> y_tr1
flat_y = np.append (y_tr, additional_labels)
new_shape = (len(y_tr)+len(additional_labels),)
y_tr1 = np.reshape(flat_y, new_shape)
y_tr1 = y_tr1.astype('int64')

print('')
print ('x_tr1.shape=',x_tr1.shape)
print ('y_tr1.shape=',y_tr1.shape)
print('')
print('number of samples for each class, before resampling:', np.bincount(y_tr))
print('number of samples for each class, after resampling: ', np.bincount(y_tr1))
#second upsampling implementation
#duplicating samples plus generating synthetic ones
#for every five samples, the fifth will be the average of the previous four
#the previous four are duplicates of samples present in the original dataset

print('x_tr.shape=', x_tr.shape)
print('y_tr.shape=', y_tr.shape)
print('')


#initializations
majority_class=np.argmax(np.bincount(y_tr))
length_majority_class = np.max(np.bincount(y_tr))
additional_samples=[]
additional_labels=[]
counter=0


for i in np.unique(y_tr):

    if i != majority_class:
        length_this_class=np.bincount(y_tr)[i]
        wanted=length_majority_class-length_this_class
        n_new_samples=0
        indexes=np.argwhere(y_tr==i)
        indexes=np.reshape(indexes, (len(indexes),))
        print('class:',i)
        print('%d additional samples' % wanted)
        while n_new_samples < wanted:
            n_new_samples +=1
            choice_index=np.random.choice(indexes,size=1)
            choice_sample= x_tr [choice_index]
            additional_samples=np.append(additional_samples,choice_sample)
            additional_labels=np.append(additional_labels,i)
            counter +=1
            if counter == 4:
                additional_samples=np.reshape(additional_samples,
                                              (int(len(additional_samples)/6),6))
                new_sample=np.mean(additional_samples[-4:,:],axis=0)
                additional_samples=np.append(additional_samples,[new_sample])
                additional_labels=np.append(additional_labels,i)
                counter=0
                n_new_samples += 1
        if n_new_samples > wanted:
            additional_samples=np.reshape(additional_samples,(int(len(additional_samples)/6),6))
            redundant = n_new_samples - wanted
            additional_samples = additional_samples[:-redundant,:].copy()
            additional_labels=additional_labels[:-redundant]
      
        

#new samples
print('')
print('new_samples.shape=', additional_samples.shape)
print('new_labels.shape=', additional_labels.shape)

#new x_tr --> x_tr2
flat_x = np.append(x_tr,additional_samples)
x_columns_number = x_tr.shape[1]
new_row_number = int(len(flat_x)/x_columns_number)
x_tr2 = np.reshape(flat_x,(new_row_number,x_columns_number))

#new y_tr --> y_tr2
flat_y = np.append (y_tr, additional_labels)
new_shape = (len(y_tr)+len(additional_labels),)
y_tr2 = np.reshape(flat_y, new_shape)
y_tr2 = y_tr2.astype('int64')

print('')
print ('x_tr2.shape=',x_tr2.shape)
print ('y_tr2.shape=',y_tr2.shape)
print('')
print('number of samples for each class, before resampling:', np.bincount(y_tr))
print('number of samples for each class, after resampling: ', np.bincount(y_tr2))
#training on the new, balanced data
lr=LogisticRegression()
lr.fit(x_tr1,y_tr1)
print('LogistReG accuracy:', accuracy_score(y_ts, lr.predict(x_ts) ))
print('LogistReg f1-score:', f1_score(y_ts,lr.predict(x_ts), average='macro'))

rf=RandomForestClassifier(1000, max_depth=8)
rf.fit(x_tr1,y_tr1)
print('RandForest accuracy:', accuracy_score(y_ts,rf.predict(x_ts)))
print('RandForest f1-score:', f1_score(y_ts,rf.predict(x_ts), average='macro'))
#training on the new, balanced and synthetic data
lr=LogisticRegression()
lr.fit(x_tr2,y_tr2)
print('LogistReG accuracy:', accuracy_score(y_ts, lr.predict(x_ts) ))
print('LogistReg f1-score:', f1_score(y_ts,lr.predict(x_ts), average='macro'))

rf=RandomForestClassifier(1000, max_depth=8)
rf.fit(x_tr2,y_tr2)
print('RandForest accuracy:', accuracy_score(y_ts,rf.predict(x_ts)))
print('RandForest f1-score:', f1_score(y_ts,rf.predict(x_ts), average='macro'))
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier


#dataframes to store experimental results, for later comparison
results=pd.DataFrame([], 
    columns=['model', 'parameters','accuracy','precision','recall','F1-score'])
results1=pd.DataFrame([], 
    columns=['model', 'parameters','accuracy','precision','recall','F1-score'])
results2=pd.DataFrame([], 
    columns=['model', 'parameters','accuracy','precision','recall','F1-score'])
majority_vote_results=pd.DataFrame([], 
    columns=['training dataset','accuracy','precision','recall','F1-score' ])
#Naive Bayes
gnb=GaussianNB()
gnb.fit(x_tr,y_tr)
y_pred=gnb.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results=results.append(pd.DataFrame([['Gaussian NB', 'default', accuracy,precision,recall,f1]],columns=list(results.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results.iloc[-1:]

#KNN
knn=KNeighborsClassifier()
knn.fit(x_tr,y_tr)
y_pred=knn.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results=results.append(pd.DataFrame([['KNN', 'default', accuracy,precision,recall,f1]],
                                    columns=list(results.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results.iloc[-1:]
#SVM
svm=SVC()
svm.fit(x_tr,y_tr)
y_pred=svm.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results=results.append(pd.DataFrame([['SVM', 'default', accuracy,precision,recall,f1]],
                                    columns=list(results.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results.iloc[-1:]
# SGD
sgd=SGDClassifier()
sgd.fit(x_tr,y_tr)
y_pred=sgd.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results=results.append(pd.DataFrame([['SGD', 'default', accuracy,precision,recall,f1]],
                                    columns=list(results.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results.iloc[-1:]
#SGD with different parameters
sgd_m=SGDClassifier(alpha=0.01, loss='modified_huber', penalty='l1')
sgd_m.fit(x_tr,y_tr)
y_pred=sgd_m.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results=results.append(pd.DataFrame([['SGD 2', 'alpha=0.01, loss=modified_huber, penalty=l1', accuracy,precision,recall,f1]],
                                    columns=list(results.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results.iloc[-1:]
# Logistic Regression
lr=LogisticRegression()
lr.fit(x_tr,y_tr)
y_pred=lr.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results=results.append(pd.DataFrame([['Logistic Regression', 'default', accuracy,precision,recall,f1]],
                                    columns=list(results.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results.iloc[-1:]
#Decision Tree
tree=DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(x_tr,y_tr)
y_pred=tree.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results=results.append(pd.DataFrame([['Decision Tree', 'max_depth=5', accuracy,precision,recall,f1]],
                                    columns=list(results.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results.iloc[-1:]
#Random Forest
rf=RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=42)
rf.fit(x_tr,y_tr)
y_pred=rf.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results=results.append(pd.DataFrame([['Random Forest', 'n_estimators=1000, max_depth=7', accuracy,precision,recall,f1]],
                                    columns=list(results.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results.iloc[-1:]
#Ada Boost
tree_ada=DecisionTreeClassifier(criterion='entropy',random_state=1, max_depth=5)
ada=AdaBoostClassifier(base_estimator=tree_ada,n_estimators=1000,learning_rate=0.1, random_state=5)
ada.fit(x_tr,y_tr)
y_pred=ada.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results=results.append(pd.DataFrame([['Ada Boost', 'criterion=entropy, max_depth=5', accuracy,precision,recall,f1]],
                                    columns=list(results.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results.iloc[-1:]
#Gradient Boosted Tree
gb=GradientBoostingClassifier(random_state=5, n_estimators=1000, max_depth=4)
gb.fit(x_tr,y_tr)
y_pred=gb.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results=results.append(pd.DataFrame([['Gradient Boosted Classifier', 'n_estimators=1000, max_depth=4',
                                      accuracy,precision,recall,f1]],
                                    columns=list(results.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results.iloc[-1:]
results= results.reset_index().drop(columns='index')
results
# majority vote implementation
pred_knn = knn.predict(x_ts) 
pred_svm = svm.predict(x_ts)
pred_sgd_m= sgd_m.predict(x_ts)
pred_lr= lr.predict(x_ts)
pred_tree= tree.predict(x_ts)
pred_rf= rf.predict(x_ts)
pred_ada=ada.predict(x_ts)
pred_gb=gb.predict(x_ts)

mode=[]
for i in np.arange(y_ts.shape[0]):
    votes=np.array([pred_knn[i], pred_svm[i], pred_sgd_m[i],pred_lr[i],
                    pred_tree[i], pred_rf[i], pred_ada[i], pred_gb[i]])
    bincount=np.bincount(votes)
    mode=np.append(mode,np.argmax(bincount)) #mode is the most common vote, the majority vote


majority_vote_results=majority_vote_results.append(pd.DataFrame([['original',
            accuracy_score(y_ts, mode),precision_score(y_ts, mode,average='macro'),
            recall_score(y_ts, mode,average='macro'),f1_score(y_ts, mode,average='macro')]],
            columns=list(majority_vote_results.columns)))
majority_vote_results.iloc[-1:]
#Naive Bayes
gnb1=GaussianNB()
gnb1.fit(x_tr1,y_tr1)
y_pred=gnb1.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results1=results1.append(pd.DataFrame([['Gaussian NB', 'default', accuracy,precision,recall,f1]],columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results1.iloc[-1:]

#KNN
knn1=KNeighborsClassifier()
knn1.fit(x_tr1,y_tr1)
y_pred=knn1.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results1=results1.append(pd.DataFrame([['KNN', 'default', accuracy,precision,recall,f1]],
                                    columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results1.iloc[-1:]
#SVM
svm1=SVC()
svm1.fit(x_tr1,y_tr1)
y_pred=svm1.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results1=results1.append(pd.DataFrame([['SVM', 'default', accuracy,precision,recall,f1]],
                                    columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results1.iloc[-1:]
#SGD
sgd1=SGDClassifier()
sgd1.fit(x_tr1,y_tr1)
y_pred=sgd1.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results1=results1.append(pd.DataFrame([['SGD', 'default', accuracy,precision,recall,f1]],
                                    columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results1.iloc[-1:]
#SGD with diferrent parameters
sgd_m1=SGDClassifier(alpha=0.01, loss='modified_huber', penalty='l1')
sgd_m1.fit(x_tr1,y_tr1)
y_pred=sgd_m1.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results1=results1.append(pd.DataFrame([['SGD 2', 'alpha=0.01, loss=modified_huber, penalty=l1', accuracy,precision,recall,f1]],
                                    columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results1.iloc[-1:]
#Logistic Regression
lr1=LogisticRegression()
lr1.fit(x_tr1,y_tr1)
y_pred=lr1.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results1=results1.append(pd.DataFrame([['Logistic Regression', 'default', accuracy,precision,recall,f1]],
                                    columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results1.iloc[-1:]
#Decision Tree
tree1=DecisionTreeClassifier(max_depth=5, random_state=42)
tree1.fit(x_tr1,y_tr1)
y_pred=tree1.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results1=results1.append(pd.DataFrame([['Decision Tree', 'max_depth=5', accuracy,precision,recall,f1]],
                                    columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results1.iloc[-1:]
#Random Forest
rf1=RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=42)
rf1.fit(x_tr1,y_tr1)
y_pred=rf1.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results1=results1.append(pd.DataFrame([['Random Forest', 'n_estimators=1000, max_depth=7', accuracy,precision,recall,f1]],
                                    columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results1.iloc[-1:]
#Ada Boost
tree_ada1=DecisionTreeClassifier(criterion='entropy',random_state=1, max_depth=5)
ada1=AdaBoostClassifier(base_estimator=tree_ada1,n_estimators=1000,learning_rate=0.1, random_state=5)
ada1.fit(x_tr1,y_tr1)
y_pred=ada1.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results1=results1.append(pd.DataFrame([['Ada Boost', 'criterion=entropy, max_depth=5', accuracy,precision,recall,f1]],
                                    columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results1.iloc[-1:]
#Gradient Boosted Tree
gb1=GradientBoostingClassifier(random_state=5, n_estimators=1000, max_depth=4)
gb1.fit(x_tr1,y_tr1)
y_pred=gb1.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results1=results1.append(pd.DataFrame([['Gradient Boosted Classifier', 'n_estimators=1000, max_depth=4',
                                      accuracy,precision,recall,f1]],
                                    columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results1.iloc[-1:]
results1=results1.reset_index().drop(columns='index')
results1
# majority vote implementation
pred_knn1 = knn1.predict(x_ts) 
pred_svm1 = svm1.predict(x_ts)
pred_sgd_m1= sgd_m1.predict(x_ts)
pred_lr1= lr1.predict(x_ts)
pred_tree1= tree1.predict(x_ts)
pred_rf1= rf1.predict(x_ts)
pred_ada1=ada1.predict(x_ts)
pred_gb1=gb1.predict(x_ts)

mode1=[]
for i in np.arange(y_ts.shape[0]):
    votes=np.array([pred_knn1[i], pred_svm1[i], pred_sgd_m1[i],pred_lr1[i],
                    pred_tree1[i], pred_rf1[i], pred_ada1[i], pred_gb1[i]])
    bincount=np.bincount(votes)
    mode1=np.append(mode1,np.argmax(bincount)) #mode is the most common vote, the majority vote


majority_vote_results=majority_vote_results.append(pd.DataFrame([['1st resampled',
            accuracy_score(y_ts, mode1),precision_score(y_ts, mode1,average='macro'),
            recall_score(y_ts, mode1,average='macro'),f1_score(y_ts, mode1,average='macro')]],
            columns=list(majority_vote_results.columns)))
majority_vote_results
#Naive Bayes
gnb2=GaussianNB()
gnb2.fit(x_tr2,y_tr2)
y_pred=gnb2.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results2=results2.append(pd.DataFrame([['Gaussian NB', 'default', accuracy,precision,recall,f1]],columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results2.iloc[-1:]

#KNN
knn2=KNeighborsClassifier()
knn2.fit(x_tr2,y_tr2)
y_pred=knn2.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results2=results2.append(pd.DataFrame([['KNN', 'default', accuracy,precision,recall,f1]],
                                    columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results2.iloc[-1:]
#SVM
svm2=SVC()
svm2.fit(x_tr2,y_tr2)
y_pred=svm2.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results2=results2.append(pd.DataFrame([['SVM', 'default', accuracy,precision,recall,f1]],
                                    columns=list(results2.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results2.iloc[-1:]
#SGD
sgd2=SGDClassifier()
sgd2.fit(x_tr2,y_tr2)
y_pred=sgd2.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results2=results2.append(pd.DataFrame([['SGD', 'default', accuracy,precision,recall,f1]],
                                    columns=list(results2.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results2.iloc[-1:]
#SGD with different parameters
sgd_m2=SGDClassifier(alpha=0.01, loss='modified_huber', penalty='l1')
sgd_m2.fit(x_tr2,y_tr2)
y_pred=sgd_m2.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results2=results2.append(pd.DataFrame([['SGD 2', 'alpha=0.01, loss=modified_huber, penalty=l1', accuracy,precision,recall,f1]],
                                    columns=list(results2.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results2.iloc[-1:]
#Logistic Regression
lr2=LogisticRegression()
lr2.fit(x_tr2,y_tr2)
y_pred=lr2.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results2=results2.append(pd.DataFrame([['Logistic Regression', 'default', accuracy,precision,recall,f1]],
                                    columns=list(results2.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results2.iloc[-1:]
#Decision Tree
tree2=DecisionTreeClassifier(max_depth=5, random_state=42)
tree2.fit(x_tr2,y_tr2)
y_pred=tree2.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results2=results2.append(pd.DataFrame([['Decision Tree', 'max_depth=5', accuracy,precision,recall,f1]],
                                    columns=list(results2.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results2.iloc[-1:]
#Random Forest
rf2=RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=42)
rf2.fit(x_tr2,y_tr2)
y_pred=rf2.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results2=results2.append(pd.DataFrame([['Random Forest', 'n_estimators=1000, max_depth=7', accuracy,precision,recall,f1]],
                                    columns=list(results2.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results2.iloc[-1:]
#Ada Boost
tree_ada2=DecisionTreeClassifier(criterion='entropy',random_state=1, max_depth=5)
ada2=AdaBoostClassifier(base_estimator=tree_ada2,n_estimators=1000,learning_rate=0.1, random_state=5)
ada2.fit(x_tr2,y_tr2)
y_pred=ada2.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results2=results2.append(pd.DataFrame([['Ada Boost', 'criterion=entropy, max_depth=5', accuracy,precision,recall,f1]],
                                    columns=list(results1.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results2.iloc[-1:]
#Gradient Boosting Tree
gb2=GradientBoostingClassifier(random_state=5, n_estimators=1000, max_depth=4)
gb2.fit(x_tr2,y_tr2)
y_pred=gb2.predict(x_ts)
accuracy=accuracy_score(y_ts,y_pred)
precision=precision_score(y_ts,y_pred, average='macro')
recall=recall_score(y_ts,y_pred, average='macro')
f1=f1_score(y_ts,y_pred, average='macro')
results2=results2.append(pd.DataFrame([['Gradient Boosted Classifier', 'n_estimators=1000, max_depth=4',
                                      accuracy,precision,recall,f1]],
                                    columns=list(results2.columns)))
sns.heatmap(confusion_matrix(y_ts,y_pred),xticklabels= enc.classes_,
            yticklabels=enc.classes_, annot=True,fmt='d');
plt.ylabel('true')
plt.xlabel('predicted');
results2.iloc[-1:]
results2=results2.reset_index().drop(columns='index')
results2
# majority vote implementation
pred_knn2 = knn2.predict(x_ts) 
pred_svm2 = svm2.predict(x_ts)
pred_sgd_m2= sgd_m2.predict(x_ts)
pred_lr2= lr2.predict(x_ts)
pred_tree2= tree2.predict(x_ts)
pred_rf2= rf2.predict(x_ts)
pred_ada2=ada2.predict(x_ts)
pred_gb2=gb2.predict(x_ts)

mode2=[]
for i in np.arange(y_ts.shape[0]):
    votes=np.array([pred_knn2[i], pred_svm2[i], pred_sgd_m2[i],pred_lr2[i],
                    pred_tree2[i], pred_rf2[i], pred_ada2[i], pred_gb2[i]])
    bincount=np.bincount(votes)
    mode2=np.append(mode2,np.argmax(bincount)) #mode is the most common vote, the majority vote


majority_vote_results=majority_vote_results.append(pd.DataFrame([['2nd resampled plus synthetic',
            accuracy_score(y_ts, mode2),precision_score(y_ts, mode2,average='macro'),
            recall_score(y_ts, mode2,average='macro'),f1_score(y_ts, mode2,average='macro')]],
            columns=list(majority_vote_results.columns)))
majority_vote_results
display(results)
display(results1)
display(results2)