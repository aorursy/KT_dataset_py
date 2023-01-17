import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import scipy.stats as stats

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn import metrics
import os

for dirname, _, filenames in os.walk('/kaggle/input/ramen-ratings/ramen-ratings.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data= pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv') # Read Csv File
data.head(4)
from sklearn import preprocessing
data.columns
df = data.drop('Top Ten',axis=1) # Dropping Top Ten as the columns has many missing values
df.head()
df.describe(include='all')
df['Variety'].value_counts()
df.isnull().sum() # To find if there are any null values
#df['Style']=df['Style'].fillna('Pack')
df.isnull().sum() 
df['Variety'].value_counts().head(10).plot(kind='bar') #Top 10 variety
df['Country'].value_counts().plot(kind='bar') #Distribution of Countries
df['Country'].value_counts().head(5).plot(kind='bar') #Top Five countries selling Ramen
df['Country'].value_counts().tail(5).plot(kind='bar') # Least Popular countries
df[df['Style'].isnull()] #Missing values in the column Style
df_kamfen=df[df['Brand']=='Kamfen']  # Filling the Style value according to the brand Kamfen on how it sells in China

Df_kam_ch=df_kamfen[df_kamfen['Country']=='China']

Df_kam_ch['Style'].value_counts()
df_unif=df[df['Brand']=='Unif'] # Filling the Style value according to the brand Unif on how it sells in Taiwan

df_unif_ta=df_unif[df_unif['Country']=='Taiwan']

df_unif_ta['Style'].value_counts()
df.loc[2152,'Style'] = 'Pack'  # Replacing the values
df.loc[2442,'Style'] = 'Bowl'  # Replacing the values
df.isnull().sum()
df['Stars'].value_counts()
df[df['Stars']=='Unrated'] # 3 values of stars were unrated
df.loc[32,'Stars'] = 0 #Assuming the ramen was not rated because it wasnt liked, we assign rating as 0

df.loc[122,'Stars'] = 0

df.loc[993,'Stars'] = 0
df['Stars']=pd.to_numeric(df['Stars']) #Converting Value of stars to numeric
sns.distplot(df['Stars'], bins=15) #Distribution plot for Stars
df.head()
country_rate = df.groupby('Country', as_index=False)['Stars'].median() # grouping countries by their median star rating
country_rate.sort_values(['Stars'], ascending=False).head(10) # Top 10 countries according to star rating
country_rate.sort_values(['Stars'], ascending=False).tail(10)  # Last 10 countries according to star rating
df['Stars'].describe()
sns.boxplot(df['Style'],df['Stars']) # boxplot to show the spread
df_high=df[df['Stars']>4.5]
pd.crosstab(df_high['Stars'],df_high['Style']).plot(kind='bar') # distribution of packages having more than 4.75 
pd.crosstab(df_high['Country'],df_high['Stars']).plot(kind='bar') #distribution of countries having more than 4.75 stars
features=['Stars','Review #']    # subplots for Stars and Review

fig=plt.subplots(figsize=(15,15))

for i, j in enumerate(features):

    plt.subplot(8, 2, i+1)

    plt.subplots_adjust(hspace = 1.0)

    sns.boxplot(x=j,data = df)

    plt.xticks(rotation=90)

    #plt.title("Telecom")

    

plt.show()
df.columns
features=['Style', 'Country'] # Subplot for count plot

fig=plt.subplots(figsize=(25,20))

for i, j in enumerate(features):

    plt.subplot(4, 2, i+1)

    plt.subplots_adjust(hspace = 1.0)

    sns.countplot(x=j,data = df)

    plt.xticks(rotation=90)

    plt.title("Ramen")

    

plt.show()
sns.boxplot(x = df.Style, y = df.Stars)
df_cup=df[(df['Style']=='Cup') & (df['Stars']<1.5)]

df_cup

df_pack=df[(df['Style']=='Pack') & (df['Stars']<1.75)]

df_pack['Brand'].value_counts().head()

df_knorr=df[(df['Brand']=='Knorr')]

df_knorr
df_box=df[(df['Style']=='Box') & (df['Stars']>4)]

df_box['Brand'].value_counts().head()

df_bowl=df[(df['Style']=='Bowl') & (df['Stars']<3.25)]

df_bowl['Brand'].value_counts().head()
df_tray=df[(df['Style']=='Tray') & (df['Stars']<1.5)]

df_tray['Brand'].value_counts().head()

df_canbar=df[(df['Style']=='Can') | (df['Style']=='Bar')]

df_canbar

df['Style'].value_counts()
from sklearn import preprocessing
#data=df.drop('Variety',axis=1)
df.head()
data = df.apply(preprocessing.LabelEncoder().fit_transform)
data.head()
data['Style'].value_counts()
data['Category'] = [0 if x == 5 else 1 for x in data['Style']] 

  

# Print the DataFrame 

print(data.head()) 
df=data.drop('Style',axis= 1)


from sklearn.model_selection import train_test_split



X=df.drop('Category',axis=1) # spliting dataframe as X and y for test train model

y=df['Category']
dfx=X

dfy=y
from sklearn.model_selection import train_test_split # spliting as 70 / 30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=425)
 
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test= sc.fit_transform(X_test)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'gini',random_state = 0)

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)
print(accuracy_score(y_test,y_pred))

print(metrics.precision_score(y_test,y_pred,average='macro'))

print(metrics.recall_score(y_test,y_pred,average='macro'))
y_pred_proba = classifier.predict_proba(X_test)[::,1]

fpr,tpr,_ = metrics.roc_curve(y_test,y_pred_proba)

auc = metrics.roc_auc_score(y_test,y_pred_proba)

plt.plot(fpr,tpr,label='data 1,auc='+str(auc))

plt.xlabel("FPR")

plt.ylabel("TPR")

plt.title("DT- ROC")

plt.legend(loc= 4)

plt.show()

feature_cols=['Review #', 'Brand', 'Country','Stars']
clf = DecisionTreeClassifier(class_weight=None,criterion = 'gini',max_depth=10,max_features=None, max_leaf_nodes= 5, min_samples_leaf=3,

                             min_samples_split=2,min_weight_fraction_leaf=0.0,presort=False , random_state = 0)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)
print(accuracy_score(y_test,y_pred))

print(metrics.precision_score(y_test,y_pred,average='macro'))

print(metrics.recall_score(y_test,y_pred,average='macro'))
from sklearn.ensemble import RandomForestClassifier



rclf = RandomForestClassifier(n_estimators= 100)
rclf.fit(X_train,y_train)
y_pred = rclf.predict(X_test)
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)

print(accuracy_score(y_test,y_pred))

print(metrics.precision_score(y_test,y_pred,average='macro'))

print(metrics.recall_score(y_test,y_pred,average='macro'))
y_pred_proba = rclf.predict_proba(X_test)[::,1]

fpr,tpr,_ = metrics.roc_curve(y_test,y_pred_proba)

auc = metrics.roc_auc_score(y_test,y_pred_proba)

plt.plot(fpr,tpr,label='data 1,auc='+str(auc))

plt.xlabel("FPR")

plt.ylabel("TPR")

plt.title("DT- ROC")

plt.legend(loc= 4)

plt.show()
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors= 5)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)
print(accuracy_score(y_test,y_pred))

print(metrics.precision_score(y_test,y_pred,average='macro'))

print(metrics.recall_score(y_test,y_pred,average='macro'))
from sklearn.model_selection import cross_val_score



k_range = range(1,31)

k_scores =[]

#loop through reasonable values of k

for k in k_range:

    #run knn

    knn = KNeighborsClassifier(n_neighbors= k)

    # obtin cross_val_Score

    scores = cross_val_score(knn,X,y,cv= 10,scoring='accuracy')

    #append mean scores

    k_scores.append(scores.mean())

print(k_scores)



k=pd.DataFrame(k_scores,columns= ['dist'])

#print(k)

k[k['dist']== k['dist'].max()]
plt.plot(k_range,k_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross validated Accuracy')
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors= 19)

classifier.fit(X_train,y_train)

# from sklearn.neighbors import KNeighborsClassifier

# classifier = KNeighborsClassifier(n_neighbors= 19)

# classifier.fit(X_train,y_train)



y_pred = classifier.predict(X_test)

# cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

# print('Confusion Matrix \n',cm)

# print(accuracy_score(y_test,y_pred))

# print(metrics.precision_score(y_test,y_pred,average='macro'))

# print(metrics.recall_score(y_test,y_pred,average='macro'))
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)
print(accuracy_score(y_test,y_pred))

print(metrics.precision_score(y_test,y_pred,average='macro'))

print(metrics.recall_score(y_test,y_pred,average='macro'))
k_range= list(range(1,31))

weight_options=["uniform","distance"]
param_grid = dict(n_neighbors = k_range, weights = weight_options)

knn= KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(knn,param_grid=param_grid,cv=10,scoring='accuracy')

grid.fit(X,y)
#grid = GridSearchCV(knn,param_grid=param_grid,cv=10,scoring='accuracy')

#grid.fit(X,y)

print(grid.best_score_)

print(grid.best_estimator_)

print(grid.best_params_)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors= 12)

classifier.fit(X_train,y_train)



y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)

print(accuracy_score(y_test,y_pred))

print(metrics.precision_score(y_test,y_pred,average='macro'))

print(metrics.recall_score(y_test,y_pred,average='macro'))
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train,y_train)
model.fit(X_train,y_train)
predicted = model.predict(X_test)

print('Predicted Value',predicted)
cm = confusion_matrix(y_true=y_test,y_pred=predicted)

print('Confusion Matrix \n',cm)
print(accuracy_score(y_test,predicted))

print(metrics.precision_score(y_test,predicted,average='macro'))

print(metrics.recall_score(y_test,predicted,average='macro'))
# Bagged Decision Trees for Classification - necessary dependencies



from sklearn import model_selection

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier
seed=7

kfold = model_selection.KFold(n_splits=10, random_state=21)

cart = DecisionTreeClassifier()

num_trees = 100

model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)

results = model_selection.cross_val_score(model, X, y, cv=kfold)

print(results.mean())
 # fit a ensemble.BaggingClassifier() model to the data

model = BaggingClassifier()

model.fit(X_train, y_train)

print(); print(model)
# make predictions

expected_y  = y_test

predicted_y = model.predict(X_test)
from sklearn import metrics

print(); print('ensemble.BaggingClassifier(): ')

print();print("Accuracy:",metrics.accuracy_score(expected_y, predicted_y))

print(); print(metrics.classification_report(expected_y, predicted_y))
# fit a ensemble.ExtraTreesClassifier() model to the data

from sklearn.ensemble import ExtraTreesClassifier 

model = ExtraTreesClassifier()

model.fit(X_train, y_train)

print(); print(model)

    

# make predictions

expected_y  = y_test

predicted_y = model.predict(X_test)
# summarize the fit of the model

print(); print('ensemble.ExtraTreesClassifier(): ')

print();print("Accuracy:",metrics.accuracy_score(expected_y, predicted_y))

print(); print(metrics.classification_report(expected_y, predicted_y))

print(); print(metrics.confusion_matrix(expected_y, predicted_y))
# Boosting Methods
#import the libraries

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder



from sklearn import metrics

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report
classifier = AdaBoostClassifier(

    DecisionTreeClassifier(max_depth=1),

    n_estimators=200

)

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

confusion_matrix(y_test, predictions)

# Model Accuracy, how well the model performs

print("Accuracy:",metrics.accuracy_score(y_test, predictions))
confusion_matrix(y_test, predictions)
from sklearn.ensemble import GradientBoostingClassifier
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]



for learning_rate in lr_list:

    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)

    gb_clf.fit(X_train, y_train)

    

    print("Learning rate: ", learning_rate)

    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))

    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.75, max_features=2, max_depth=2, random_state=0)

gb_clf2.fit(X_train, y_train)

predictions = gb_clf2.predict(X_test)



print("Confusion Matrix:")

print(confusion_matrix(y_test, predictions))



print();print("Accuracy:",metrics.accuracy_score(y_test, predictions))



print();print("Classification Report")

print();print(classification_report(y_test, predictions))
#!pip install xgboost

from xgboost import XGBClassifier



classifier = XGBClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

y_pred
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_pred, y_test)

cm
# Model Accuracy, how well the model performs

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
import pandas as pd

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from vecstack import stacking



from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB 

from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import StackingClassifier

from xgboost.sklearn import XGBClassifier

import numpy as np

import warnings



warnings.simplefilter('ignore')
models = [

    KNeighborsClassifier(n_neighbors=5,

                        n_jobs=-1),



    RandomForestClassifier(random_state=0, n_jobs=-1,

                           n_estimators=100, max_depth=3),



    XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,

                  n_estimators=100, max_depth=3)

]
S_train, S_test = stacking(models,

                           X_train, y_train, X_test,

                           regression=False,



                           mode='oof_pred_bag',



                           needs_proba=False,



                           save_dir=None,



                           metric=accuracy_score,



                           n_folds=4,



                           stratified=True,



                           shuffle=True,



                           random_state=0,



                           verbose=2)
X.head()
X.columns
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df = sc.fit_transform(X)
X=pd.DataFrame(df,columns=['Review #', 'Brand', 'Variety','Country', 'Stars'])
cluster_range = range(1,15)

cluster_errors=[]
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
for num_clusters in cluster_range:

    clusterrs = KMeans(num_clusters)

    clusterrs.fit(X)

    cluster_errors.append(clusterrs.inertia_)
culters_df = pd.DataFrame({"num_clusters":cluster_range,"cluster_errors":cluster_errors})

culters_df[0:10]
plt.figure(figsize=(12,6))

plt.plot(culters_df.num_clusters,culters_df.cluster_errors,marker = 'o')
kmeans = KMeans(n_clusters=4, n_init = 10, random_state=251)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
centroid_df = pd.DataFrame(centroids, columns = list(X) )
centroid_df = pd.DataFrame(centroids, columns = list(X) )

df_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))



df_labels['labels'] = df_labels['labels'].astype('category')
# df_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))



# df_labels['labels'] = df_labels['labels'].astype('category')

snail_df_labeled = X.join(df_labels)

# df_analysis = (snail_df_labeled.groupby(['labels'] , axis=0)).head(4177) 

# df_analysis.head(3)
df_analysis = (snail_df_labeled.groupby(['labels'] , axis=0)).head(4177) 

df_analysis.head(3)
y_test = y

y_pred= df_analysis['labels']

cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)

print(accuracy_score(y_test,y_pred))

print(metrics.precision_score(y_test,y_pred,average='macro'))

print(metrics.recall_score(y_test,y_pred,average='macro'))
from sklearn.model_selection import train_test_split  



X= df_analysis.drop('labels',axis =1)

y= df_analysis['labels']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train,y_train)
# predict Model

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)

accuracy_score(y_test,y_pred)
# DTfrom sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'gini',random_state = 0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)

print(accuracy_score(y_test,y_pred))

print(metrics.precision_score(y_test,y_pred,average='macro'))

print(metrics.recall_score(y_test,y_pred,average='macro'))
X.columns
feature_cols=['Review #', 'Brand', 'Country', 'Stars']
clf = DecisionTreeClassifier(class_weight=None,criterion = 'gini',max_depth=10,max_features=None, max_leaf_nodes= 5, min_samples_leaf=3,

                             min_samples_split=2,min_weight_fraction_leaf=0.0,presort=False , random_state = 0)

clf.fit(X_train,y_train)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'gini',random_state = 0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)

print(accuracy_score(y_test,y_pred))

print(metrics.precision_score(y_test,y_pred,average='macro'))

print(metrics.recall_score(y_test,y_pred,average='macro'))
rclf = RandomForestClassifier(n_estimators= 100)

rclf.fit(X_train,y_train)

y_pred = rclf.predict(X_test)

cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)

print(accuracy_score(y_test,y_pred))

print(metrics.precision_score(y_test,y_pred,average='macro'))

print(metrics.recall_score(y_test,y_pred,average='macro'))
classifier = KNeighborsClassifier(n_neighbors= 5)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_true=y_test,y_pred=y_pred)

print('Confusion Matrix \n',cm)

print(accuracy_score(y_test,y_pred))

print(metrics.precision_score(y_test,y_pred,average='macro'))

print(metrics.recall_score(y_test,y_pred,average='macro'))
model = GaussianNB()

model.fit(X_train,y_train)

predicted = model.predict(X_test)

print('Predicted Value',predicted)
cm = confusion_matrix(y_true=y_test,y_pred=predicted)

print('Confusion Matrix \n',cm)

print(accuracy_score(y_test,predicted))

print(metrics.precision_score(y_test,predicted,average='macro'))

print(metrics.recall_score(y_test,predicted,average='macro'))
X=dfx

y=dfy


from scipy.cluster.hierarchy import dendrogram , linkage

linked = linkage(X,'ward')



plt.figure(figsize=(10,7))

dendrogram(linked,orientation='top',distance_sort='descending',show_leaf_counts=True)

plt.show()

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')

cluster.fit_predict(X)

y=data['Category']

cm = confusion_matrix(y_true=y,y_pred=cluster.fit_predict(X))

print('Confusion Matrix \n',cm)

print(accuracy_score(y,cluster.fit_predict(X)))
cluster = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')

cluster.fit_predict(X)
y=data['Category']
cm = confusion_matrix(y_true=y,y_pred=cluster.fit_predict(X))

print('Confusion Matrix \n',cm)

print(accuracy_score(y,cluster.fit_predict(X)))
df_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))



df_labels['labels'] = df_labels['labels'].astype('category')

snail_df_labeled = X.join(df_labels)

df_analysis = (snail_df_labeled.groupby(['labels'] , axis=0)).head(4177) 

df_analysis.head(3)
X=df_analysis.drop('labels',axis=1)

y=df_analysis['labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

names = X.columns

models = []

models.append(('LR', LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))



seed = 7

from sklearn import model_selection

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
X = dfx

y =dfy

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.decomposition import PCA


pca = PCA(n_components= 2)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

plt.plot(pca.explained_variance_ratio_)

plt.xlabel('no. of omponents')

plt.ylabel('cumulative explained variance')

plt.show()
names = X.columns

models = []

models.append(('LR', LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))



seed = 7

from sklearn import model_selection

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# PCA Without scaling

X = dfx

y = dfy



from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=425)
data.head()
pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

names = data.columns

models = []

models.append(('LR', LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))





seed = 7

from sklearn import model_selection

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)