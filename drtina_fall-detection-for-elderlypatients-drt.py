# General imports

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import random



import warnings 

warnings.filterwarnings('ignore')



# Modules used for assessing the performance of the model

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import f1_score

from sklearn.metrics import  confusion_matrix





# Dimensionality reduction modules

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



# Model Training

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier



# Scaling and Sampling the data

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE



# Graph Visualization

from sklearn.tree import export_graphviz



# Unsupervised learning model

from sklearn.cluster import KMeans

from scipy.stats import mode
# # Mounting the drive to enable the file import

# from google.colab import drive

# drive.mount('/content/drive')
Data = pd.read_csv('../input/falldeteciton.csv')

Data.head()
print(sorted(Data['ACTIVITY'].unique()))

print('Here 0 indicates Standing, 1- Walking, 2- Sitting, 3 - Falling, 4 - Cramps, 5 - Running')
Data.describe()
features = Data.iloc[:,1:]

sns.pairplot(features) 

plt.show()
single_dimension_pca = PCA(n_components=1)

single_dimention_data = single_dimension_pca.fit_transform(features.T)
print(single_dimension_pca.explained_variance_ratio_)

print(single_dimension_pca.mean_)
cols = list(features.columns)
plt.figure(figsize=(10,5))

y_axis_zeros = np.zeros(len(single_dimention_data))



#plt.scatter(single_dimention_data,y_axis_zeros)

for i in range(len(single_dimention_data)):

  plt.scatter(single_dimention_data[i],y_axis_zeros[i])

  plt.annotate(cols[i],(single_dimention_data[i],y_axis_zeros[i]),rotation=20)



plt.title('PCA Analysis single dimension')

plt.show()
b = cols.index('BP')

h = cols.index('HR')
single_dimention_data[b],single_dimention_data[h]
sns.pairplot(Data,x_vars=cols,y_vars='ACTIVITY')

plt.show()
Data[Data['ACTIVITY']==1].head()
Data[Data['ACTIVITY']==3].head()
for i in cols:

  sns.boxplot(features[i])

  plt.show()
Data.isna().sum()
original_data = Data.copy()
initial = len(Data)

for i in cols:

  IQR = Data[i].quantile(0.75) - Data[i].quantile(0.25)

  Data = Data[(Data[i]< (Data[i].quantile(0.75)+IQR)) & (Data[i] > (Data[i].quantile(0.25)-IQR))]

  

final = len(Data)

print('Number of outliers removed',initial-final)
# After removing the outliers checking the distribution of each feature in the Dataset



for i in cols:

  sns.boxplot(Data[i])

  plt.show()
Feature_clean_data = Data.iloc[:,1:]

Target_label_activity = Data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(Feature_clean_data, Target_label_activity, test_size=0.25)
log_regress = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=10500)

log_regress.fit(X_train,y_train)

y_pred = log_regress.predict(X_test)





accuracy_score(y_test,y_pred)

plt.scatter(X_test.iloc[:,0],X_test.iloc[:,1],c=y_pred)

plt.show()
# We also need to check if there is a correlation between the features 

Data.corr()
log_regress = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=10500)

log_regress.fit(X_train[['CIRCLUATION','EEG']],y_train)

y_pred = log_regress.predict(X_test[['CIRCLUATION','EEG']])



from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score



accuracy_score(y_test,y_pred)
sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)

X_test_scaled = sc.transform(X_test)



log_regress = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=10500)

log_regress.fit(X_train_scaled,y_train)

y_pred = log_regress.predict(X_test_scaled)





log_accuracy = round(accuracy_score(y_test,y_pred),3)

print(log_accuracy)
comparing_models = {}

comparing_models['LogisticRegression'] = log_accuracy
model = GaussianNB()





model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_test_scaled)





print('Gaussian Naive Bayes')

print('Accuracy-score'.ljust(15),'Recall-score'.ljust(15),'F1 score'.ljust(15))

print(str(round(accuracy_score(y_test,y_pred),3)).ljust(15),str(round(recall_score(y_test,y_pred,average='weighted'),3)).ljust(15),

      str(round(f1_score(y_test,y_pred,average='weighted'),3)).ljust(15))



nb_accuracy = round(accuracy_score(y_test,y_pred),3)



comparing_models['Gaussian Naive Bayes'] = nb_accuracy
k = range(1,51)

best_k = None

best_score = -np.inf

best_recall = -np.inf
for n in k:

  knn = KNeighborsClassifier(n)

  knn.fit(X_train_scaled,y_train)

  y_pred = knn.predict(X_test_scaled)

  

  score = accuracy_score(y_test,y_pred)

  recall = recall_score(y_test,y_pred,average='weighted')

  

  if (score>best_score) | (recall > best_recall):

    best_score=score

    best_recall = recall

    best_k=n

    

print('Best k'.ljust(10),'Accuracy'.ljust(10),'Recall'.ljust(10))

print(str(best_k).ljust(10),str(round(best_score,3)).ljust(10),str(round(best_recall,3)).ljust(10))



knn_accuracy = round(best_score,3)



knn=KNeighborsClassifier(best_k)

knn.fit(X_train_scaled,y_train)

y_pred = knn.predict(X_test_scaled)

  

knn_mat = confusion_matrix(y_test,y_pred)
comparing_models['KNN'] = knn_accuracy
rfc = RandomForestClassifier(random_state=0)

param_grid = {'n_estimators':[50,100,150,200],'max_depth':[50,100,150,None]}

grid = GridSearchCV(rfc,param_grid=param_grid)
grid.fit(X_train_scaled,y_train)
random_forest = grid.best_estimator_

random_forest
random_forest.fit(X_train_scaled,y_train)

y_pred = random_forest.predict(X_test_scaled)



print('Random Forest Classifier')

print('Accuracy-score'.ljust(15),'Recall-score'.ljust(15),'F1 score'.ljust(15))

print(str(round(accuracy_score(y_test,y_pred),3)).ljust(15),str(round(recall_score(y_test,y_pred,average='weighted'),3)).ljust(15),

      str(round(f1_score(y_test,y_pred,average='weighted'),3)).ljust(15))



rf_accuracy = round(accuracy_score(y_test,y_pred),3)
comparing_models['Random Forest Classifier'] = rf_accuracy
from sklearn.metrics import  confusion_matrix

rf_mat = confusion_matrix(y_test,y_pred)

rf_mat = np.round(rf_mat,1)

sns.heatmap(rf_mat,annot=True)

plt.title('Confusion Matrix')

plt.show()
rows  = list(Feature_clean_data.columns)

plt.bar(rows,random_forest.feature_importances_)

plt.xlabel('Features')#,color='white')

plt.ylabel('Importance')#,color='white')

# plt.xticks(color='white')

# plt.yticks(color='white')

plt.title('Feature Importance plot')

plt.show()
mlp_classifier = MLPClassifier()
param_grid = {'activation':['relu','logistic','tanh'],'hidden_layer_sizes':[(50,),(100,),(150,)],'learning_rate_init':[0.01,0.1]}
grid = GridSearchCV(mlp_classifier,param_grid)

grid.fit(X_train_scaled,y_train)
best_mlp_classifier = grid.best_estimator_
best_mlp_classifier
best_mlp_classifier.fit(X_train_scaled,y_train)

y_pred = best_mlp_classifier.predict(X_test_scaled)



print('Multi Layer Perceptron Classifier')

print('Accuracy-score'.ljust(15),'Recall-score'.ljust(15),'F1 score'.ljust(15))

print(str(round(accuracy_score(y_test,y_pred),3)).ljust(15),str(round(recall_score(y_test,y_pred,average='weighted'),3)).ljust(15),

      str(round(f1_score(y_test,y_pred,average='weighted'),3)).ljust(15))



mlp_accuracy = round(accuracy_score(y_test,y_pred),3)

mlp_mat = confusion_matrix(y_test,y_pred)
comparing_models['MLP classifier'] = mlp_accuracy
dtc = DecisionTreeClassifier()

dtc.fit(X_train_scaled,y_train)

y_pred = dtc.predict(X_test_scaled)

dt_accuracy = accuracy_score(y_test, y_pred)



print('Decision Tree Classifier')

print('Accuracy-score'.ljust(15),'Recall-score'.ljust(15),'F1 score'.ljust(15))

print(str(round(accuracy_score(y_test,y_pred),3)).ljust(15),str(round(recall_score(y_test,y_pred,average='weighted'),3)).ljust(15),

      str(round(f1_score(y_test,y_pred,average='weighted'),3)).ljust(15))



dt_accuracy = round(accuracy_score(y_test,y_pred),3)

decision_tree_mat = confusion_matrix(y_test,y_pred)
comparing_models['Decision Tree Classifier'] = dt_accuracy
svc = SVC(gamma='auto')

svc.fit(X_train_scaled,y_train)



y_pred = svc.predict(X_test_scaled)





print('SVM Classifier')

print('Accuracy-score'.ljust(15),'Recall-score'.ljust(15),'F1 score'.ljust(15))

print(str(round(accuracy_score(y_test,y_pred),3)).ljust(15),str(round(recall_score(y_test,y_pred,average='weighted'),3)).ljust(15),

      str(round(f1_score(y_test,y_pred,average='weighted'),3)).ljust(15))



svm_accuracy = round(accuracy_score(y_test,y_pred),3)
comparing_models['SVM classifier'] = svm_accuracy
plt.bar(comparing_models.keys(),comparing_models.values())

plt.xticks(rotation=75)

plt.ylabel('Accuracy')

plt.xlabel('Model')

plt.title('Comparision across models')

plt.show()
y_train.value_counts() # To check how the data is balanced with the training dataset
sm = SMOTE()

X_train_fin, y_train_fin = sm.fit_sample(X_train_scaled,y_train.ravel())
param_grid = {'n_estimators':[50,100,150,200],'max_depth':[50,100,150,None]}

grid = GridSearchCV(rfc,param_grid=param_grid)

grid.fit(X_train_fin,y_train_fin)

new_random_forest = grid.best_estimator_
new_random_forest.fit(X_train_fin,y_train_fin)

Y_pred = new_random_forest.predict(X_test_scaled)
print('Random Forest after SMOTE')

print('Accuracy'.ljust(10),'Recall'.ljust(10))

print(str(round(accuracy_score(y_test,Y_pred),3)).ljust(10),str(round(recall_score(y_test,Y_pred,average='micro'),3)).ljust(10))
knn = KNeighborsClassifier(best_k)

knn.fit(X_train_fin,y_train_fin)

ypred = knn.predict(X_test_scaled)
print('KNN after SMOTE')

print('Accuracy'.ljust(10),'Recall'.ljust(10))

print(str(round(accuracy_score(y_test,ypred),3)).ljust(10),str(round(recall_score(y_test,ypred,average='micro'),3)).ljust(10))
print(Data['ACTIVITY'].value_counts())

x_row = Data['ACTIVITY'].value_counts().index

y_row = Data['ACTIVITY'].value_counts().values

plt.bar(x_row,y_row)

plt.xlabel('Activity')

plt.ylabel('Amount of Data')

plt.show()
sm = SMOTE()

sampled_X, sampled_Y = sm.fit_sample(Feature_clean_data,Target_label_activity)

sampled_X_train, sampled_X_test, sampled_Y_train, sampled_Y_test = train_test_split(sampled_X, sampled_Y, test_size=0.3, random_state=0)
random_forest
rf = RandomForestClassifier(max_depth=50,n_estimators=200) # using the same max_depth and n_estimators as the random_forest classifier before SMOTE analysis
rf.fit(sampled_X_train,sampled_Y_train)
Y_pred = rf.predict(sampled_X_test)
print('Random forest after over-sampling the entire dataset')

print('Accuracy'.ljust(10),'Recall'.ljust(10),'F1-score'.ljust(10))

print(str(round(accuracy_score(sampled_Y_test,Y_pred),3)).ljust(10),str(round(recall_score(sampled_Y_test,Y_pred,average='weighted'),3)).ljust(10),

     str(round(f1_score(sampled_Y_test,Y_pred,average='weighted'),3)).ljust(10))
from sklearn.metrics import  confusion_matrix

mat = confusion_matrix(sampled_Y_test,Y_pred)

mat = np.round(mat,1)

result_mat = mat.T

sns.heatmap(result_mat,annot=True)

plt.xlabel('True label')

plt.ylabel('Predicted label')

plt.show()
rows = list(Feature_clean_data.columns)

plt.bar(rows,random_forest.feature_importances_)

plt.xlabel('Features')#,color='white')

plt.ylabel('Importance')#,color='white')

# plt.xticks(color='white')

# plt.yticks(color='white')

plt.title('Feature Importance plot')

plt.show()
rf_visualization = RandomForestClassifier(max_depth=5,n_estimators=200)

rf_visualization.fit(sampled_X_train,sampled_Y_train)

Y_pred = rf_visualization.predict(sampled_X_test)

print('Random forest for visualization')

print('Accuracy'.ljust(10),'Recall'.ljust(10))

print(str(round(accuracy_score(sampled_Y_test,Y_pred),3)).ljust(10),str(round(recall_score(sampled_Y_test,Y_pred,average='weighted'),3)).ljust(10))
r = random.randint(0,len(rf_visualization.estimators_))
estim = rf_visualization.estimators_[r]

export_graphviz(estim,out_file='tree.dot',feature_names=list(Feature_clean_data.columns),class_names=['Standing','Walking','Sitting','Falling','Cramps','Running'],

               rounded=True,precision=True,filled=True,proportion=False)
from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=50'])

from IPython.display import Image

Image(filename = 'tree.png')
tsne = TSNE(n_components=2,random_state=0,perplexity=200,learning_rate=400,angle=0.99)

projected = tsne.fit_transform(Feature_clean_data)



num_clusters = len(Target_label_activity.unique())



kmeans = KMeans(n_clusters=num_clusters)

clusters = kmeans.fit_predict(projected)
plt.scatter(projected[:,0],projected[:,1],c=kmeans.labels_,cmap=plt.cm.get_cmap('Spectral', 10),alpha=0.5)

plt.xlabel('component 1')

plt.ylabel('component 2')

plt.colorbar();



labels = np.zeros_like(clusters)

for i in range(10):

    mask = (clusters == i)

    labels[mask] = mode(Target_label_activity[mask])[0]

    

accuracy_score(Target_label_activity, labels)
print(result_mat)
no_false_positives = sum(result_mat[3,:]) - result_mat[3,3]

no_false_negative = sum(result_mat[:,3]) - result_mat[3,3]



print('Number of False positives with random forest classifer is',no_false_positives)

print('Number of False negatives with random forest classifer is',no_false_negative)



print('Ratio of false negatives',no_false_negative/np.sum(result_mat))

print('Ratio of false positives',no_false_positives/np.sum(result_mat))
print('Ratio of False positive to False negatives')
new_labels = [1 if i==3 else 0 for i in Target_label_activity]
sm = SMOTE()

sampled_X, sampled_Y = sm.fit_sample(Feature_clean_data,new_labels)

sampled_X_train, sampled_X_test, sampled_Y_train, sampled_Y_test = train_test_split(sampled_X, sampled_Y, test_size=0.3, random_state=0)
threshold = 0.4 



rfc = RandomForestClassifier(random_state=0)

param_grid = {'n_estimators':[50,100,150,200],'max_depth':[50,100,150,None]}

grid = GridSearchCV(rfc,param_grid=param_grid,scoring='f1')



grid.fit(sampled_X_train,sampled_Y_train)



random_forest = grid.best_estimator_





random_forest.fit(sampled_X_train,sampled_Y_train)

predicted_proba = random_forest.predict_proba(sampled_X_test)



predicted = (predicted_proba [:,1] >= threshold).astype('int')





print('Random Forest Classifier')

print('Accuracy-score'.ljust(15),'Recall-score'.ljust(15),'F1 score'.ljust(15))

print(str(round(accuracy_score(sampled_Y_test,predicted),3)).ljust(15),str(round(recall_score(sampled_Y_test,predicted,average='weighted'),3)).ljust(15),

      str(round(f1_score(sampled_Y_test,predicted,average='weighted'),3)).ljust(15))



rf_accuracy = round(accuracy_score(sampled_Y_test,predicted),3)
random_forest
new_matrix = confusion_matrix(sampled_Y_test,predicted)
sns.heatmap(new_matrix,annot=True)

plt.xlabel('True labels')

plt.ylabel('Predicted labels')

plt.show()

result_new_matrix = new_matrix.T

result_new_matrix
neg_false = result_new_matrix[0,1]

pos_false = result_new_matrix[1,0]
print('Ratio of false negatives',neg_false/np.sum(new_matrix))

print('Ratio of false positives',pos_false/np.sum(new_matrix))