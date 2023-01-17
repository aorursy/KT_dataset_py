import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

sns.set()

heart_data = pd.read_csv("../input/heart-disease-uci/heart.csv")
heart_data.head()
heart_data.info()
# Class percentage in the dataset



round(heart_data.target.value_counts()/len(heart_data)*100,2)
sns.countplot(heart_data.target)

plt.show()
# Investigating how many continuous and categorical variable there are in the dataset

# Count of unique values in every feature of the dataset



unique_counts=[len(heart_data[column].unique()) for column in heart_data.columns.values]

features_unique_counts=pd.Series(unique_counts,index=heart_data.columns.values)
features_unique_counts.sort_values()
x=heart_data.iloc[:,:-1]

y=heart_data.iloc[:,-1]

x_vis=(x - x.mean()) / (x.std())  

vis_data=pd.concat([x_vis,y], axis=1)
# investigate the continuous variables using swarmplot



melted_data = pd.melt(vis_data.loc[:,['age','trestbps','thalach','chol','target','oldpeak']],id_vars="target",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,7))

sns.swarmplot(x="features", y="value", hue="target", data=melted_data)

plt.show()
# investigate the continuous variables using violinplot

plt.figure(figsize=(10,7))

sns.violinplot(x="features", y="value", hue="target", data=melted_data,split=True, inner="quart")

plt.xticks(rotation=90)
# investigate the continuous variables using boxplot

plt.figure(figsize=(10,7))

sns.boxplot(x="features", y="value", hue="target", data=melted_data)

plt.xticks(rotation=90)
# Define our matrix of features and target variable

X=heart_data.iloc[:,:-1]

y=heart_data.iloc[:,-1]



# Splitting data into two sets, one for training and the other for evaluation

from sklearn.model_selection import train_test_split

x_train,x_holdout,y_train,y_holdout=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)

x_holdout=scaler.transform(x_holdout)
# Quick KNN default k=5 neighbors

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



knn=KNeighborsClassifier()

knn.fit(x_train,y_train)

y_pred_holdout=knn.predict(x_holdout)

y_pred_train=knn.predict(x_train)

print('Train acc: {:.3f}\nTest acc: {:.3f}'.format(accuracy_score(y_train,y_pred_train),accuracy_score(y_holdout,y_pred_holdout)))

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



# Using distance metric as eucledian and cross validation of 10 folds

k_values=np.arange(1,60,2)



accuracies_euc=[]

for k in k_values:

    knn=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)

    avg_acc_score=cross_val_score(knn,x_train,y_train,cv=10).mean()

    accuracies_euc.append(avg_acc_score)



# Using distance metric as manhatten and cross validation of 10 folds

accuracies_man=[]

for k in k_values:

    knn=KNeighborsClassifier(n_neighbors=k,metric='manhattan',p=1)

    avg_acc_score=cross_val_score(knn,x_train,y_train,cv=10).mean()

    accuracies_man.append(avg_acc_score)
# Visualizing accuracy for euclidean KNN

plt.plot(k_values,accuracies_euc)

plt.title('Euclidean KNN')

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')

plt.show()
# Visualizing accuracy for manhattan KNN

plt.plot(k_values,accuracies_man)

plt.title('Manhattan KNN')

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')

plt.show()
# testing on holdout set with manhattan distance and k=37

knn=KNeighborsClassifier(n_neighbors=11,metric='manhattan',p=1)

knn.fit(x_train,y_train)

y_pred_holdout=knn.predict(x_holdout)

y_pred_train=knn.predict(x_train)

print('Train acc: {:.3f}\nTest acc: {:.3f}'.format(accuracy_score(y_train,y_pred_train),accuracy_score(y_holdout,y_pred_holdout)))
corr_mat=heart_data.corr()

plt.figure(figsize=(15,12))

sns.heatmap(corr_mat,annot=True, cmap='RdYlGn')

plt.show()
cat_features=X.drop(['age','trestbps','thalach','chol','oldpeak'],axis=1)

cont_features=X.loc[:,['age','trestbps','thalach','chol','oldpeak']]
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



# Feature extraction

chi= SelectKBest(score_func=chi2,k=4)

chi_features = chi.fit(cat_features,y)



# Summarize features and scores in pandas series

chi_scores=pd.Series(chi_features.scores_, index=cat_features.columns.values)

chi_scores=chi_scores.sort_values(ascending=False)

chi_scores
from sklearn.feature_selection import f_classif



# Feature extraction

anova= SelectKBest(score_func=f_classif,k=4)

anova_features = anova.fit(cont_features,y)



# Summarize features and scores in pandas series

anova_scores=pd.Series(anova_features.scores_, index=cont_features.columns.values)

anova_scores=anova_scores.sort_values(ascending=False)

anova_scores
# building final model with chosen features and testing it

x_chosen=X.loc[:,['thalach','oldpeak','ca','cp']]

x_train,x_holdout,y_train,y_holdout=train_test_split(x_chosen,y,test_size=0.2,stratify=y,random_state=42)

x_train=scaler.fit_transform(x_train)

x_holdout=scaler.transform(x_holdout)

knn=KNeighborsClassifier(n_neighbors=37,metric='manhattan',p=1)

knn.fit(x_train,y_train)

y_pred_holdout=knn.predict(x_holdout)

print('accuracy: ',accuracy_score(y_holdout,y_pred_holdout))