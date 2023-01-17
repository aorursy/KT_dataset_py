import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
data= pd.read_csv('../input/sonaralldata/sonar.all-data.csv', header=None) 
data.head()
#missing value checking
data.isnull().sum().sum()
data.describe()
#target value
data.iloc[:,-1].unique()
#Turning ROCK and MINE into 0 and 1 (binary)
data= data.replace(to_replace="R", value=0, regex=True)
data= data.replace(to_replace="M", value=1, regex=True)
data.head()
#train-test data
X= data.iloc[:,:-1]
y= data.iloc[:,-1]
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size = 0.2, random_state=0)
#visualizing for 2 Principal Components representing features

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(pca, y, random_state=0)
plt.figure(dpi=120)
plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=1, label='Rock', s=4)
plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=1, label='Mine', s=4)
plt.legend()
plt.title('Sonar Data Set\nFirst Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.gca().set_aspect('equal')
knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print('train score: {:.4f}'.format(knn.score(X_train, y_train)))
print('test score: {:.4f}'.format(knn.score(X_test, y_test)))
#scaling the data for better accuracy
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn.fit(X_train_scaled, y_train)
print('train score: {:.4f}'.format(knn.score(X_train_scaled, y_train)))
print('test score: {:.4f}'.format(knn.score(X_test_scaled, y_test)))
knn_prediction = knn.predict(X_test_scaled)
cm= confusion_matrix(y_test, knn_prediction)
print(cm)
#trying k values for best score
scores=[]
k_range = range(1,20)
for k in k_range:
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])
scores
print("max score for k=", scores.index(max(scores)) +1,"\n","max score=", max(scores))
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in t:

    scores = []
    for i in range(1,50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy')
log= LogisticRegression().fit(X_train_scaled, y_train)
print('logreg train score: {:.3f}'.format(log.score(X_train_scaled, y_train)))
print('logreg test score: {:.3f}'.format(log.score(X_test_scaled, y_test)))
lr_prediction = log.predict(X_test_scaled)
cm= confusion_matrix(y_test, lr_prediction)
print(cm)
#an alternative to visualize confusion matrix
cm = confusion_matrix(y_test, log.predict(X_test_scaled))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Rock', 'Predicted Mine'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Rock', 'Actual Mine'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=14)
plt.show()
nb= GaussianNB().fit(X_train_scaled, y_train)
print('clf train score: {:.2f}'.format(nb.score(X_train_scaled, y_train)))
print('clf test score: {:.2f}'.format(nb.score(X_test_scaled, y_test)))
nb_prediction= nb.predict(X_test_scaled)
cm= confusion_matrix(y_test, nb_prediction)
print(cm)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)
vifs=calc_vif(data)
vifs["VIF"].sort_values(ascending=False)
svm= SVC().fit(X_train_scaled, y_train) 
print('svm train score: {:.2f}'.format(svm.score(X_train_scaled, y_train)))
print('svm test score: {}'.format(svm.score(X_test_scaled, y_test)))
svm_prediction = svm.predict(X_test_scaled)
cm = confusion_matrix(y_test, svm_prediction)
print(cm)
svm_2= LinearSVC().fit(X_train_scaled, y_train)
print('svm train score: {}'.format(svm_2.score(X_train_scaled, y_train)))
print('svm test score: {}'.format(svm_2.score(X_test_scaled, y_test)))
dt= DecisionTreeClassifier().fit(X_train_scaled, y_train) #maxdepth kac dallanma oldugu
print('clf train score: {:.2f}'.format(dt.score(X_train_scaled, y_train)))
print('clf test score: {:.2f}'.format(dt.score(X_test_scaled, y_test)))
dt_prediction = dt.predict(X_test_scaled)
cm = confusion_matrix(y_test, dt_prediction)
print(cm)
names= X_train.columns
sorted(zip(map(lambda x: round(x, 4), dt.feature_importances_), names), reverse=True)[:10]
Importance = pd.DataFrame({'Importance':dt.feature_importances_*100},
                         index = X_train.columns)

Importance_nonzero = Importance[(Importance.T != 0).any()]
Importance_nonzero.sort_values(by ='Importance',
                      axis = 0,
                      ascending = True).plot(kind = 'barh',
                                            color = 'b')

plt.xlabel('Variable Importance %')
plt.ylabel("Column Name")
plt.gca().legend_ = None
rf= RandomForestClassifier().fit(X_train_scaled, y_train)
print('clf train score: {:.2f}'.format(rf.score(X_train_scaled, y_train)))
print('clf test score: {:.2f}'.format(rf.score(X_test_scaled, y_test)))
rf_prediction = rf.predict(X_test_scaled)
cm = confusion_matrix(y_test,rf_prediction)
print(cm)
sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)[:10]
Importance = pd.DataFrame({'Importance':rf.feature_importances_*100},
                         index = X_train.columns)

#filtering less than %2 importance to highlight the most important ones
Importance_mostly = Importance[(Importance.T >= 2).any()]
Importance_mostly.sort_values(by ='Importance',
                      axis = 0,
                      ascending = True).plot(kind = 'barh',
                                            color = 'g')

plt.xlabel('Variable Importance %')
plt.ylabel("Column Name")
plt.gca().legend_ = None
mlp = MLPClassifier(random_state=1, max_iter=300).fit(X_train_scaled, y_train)
print('clf train score: {:.2f}'.format(mlp.score(X_train_scaled, y_train)))
print('clf test score: {:.2f}'.format(mlp.score(X_test_scaled, y_test)))
mlp_prediction = mlp.predict(X_test_scaled)
cm = confusion_matrix(y_test,mlp_prediction)
print(cm)
pr_dict = {'KNN' : knn_prediction,'Logistic Regression' : lr_prediction,'SVM' : svm_prediction,
           'Decision Tree' : dt_prediction, 'Random Forest' : rf_prediction,'Naive Bayes' : nb_prediction, 'MLP' : mlp_prediction}

all_predictions = pd.DataFrame(pr_dict)

final_prediction = [] #final prediction list

for i in range(all_predictions.shape[0]):    
    if all_predictions.mean(axis=1)[i] <= 0.5:
        final_prediction.append(0)  #means rock
    else:
        final_prediction.append(1) #means mine
all_predictions["final pred"] = final_prediction
all_predictions["real"] = y_test.values
all_predictions.head()
cm = confusion_matrix(final_prediction, y_test)
print(cm)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, final_prediction))
models= [knn,log,svm,dt,rf,nb,mlp]
all_scores = pd.DataFrame(np.zeros((8,2)))
all_scores.columns = ["train score", "test score"]
all_scores.index = ['KNN', 'Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest',
       'Naive Bayes', 'MLP', "Final Pred"]
train_scores=[]
test_scores=[]
for model in models:
    model.fit(X_train_scaled,y_train)
    train_scores.append(model.score(X_train_scaled, y_train))
    test_scores.append(model.score(X_test_scaled, y_test))
#for final pred row
train_scores.append("-")
test_scores.append(accuracy_score(y_test, final_prediction))
all_scores["train score"]= train_scores
all_scores["test score"]= test_scores
all_scores
all_scores["test score"].sort_values(ascending=False)
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

# Define dictionary with performance metrics
scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score)}

# Import required libraries for machine learning classifiers

# Instantiate the machine learning classifiers
knn_model = KNeighborsClassifier()
log_model = LogisticRegression()
svc_model = SVC()
dtr_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()
gnb_model = GaussianNB()
mlp_model = MLPClassifier()

# Define the models evaluation function
def models_evaluation(X, y, folds):
    
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    
    '''
    
    # Perform cross-validation to each machine learning classifier
    knn = cross_validate(knn_model, X, y, cv=folds, scoring=scoring)
    log = cross_validate(log_model, X, y, cv=folds, scoring=scoring)
    svc = cross_validate(svc_model, X, y, cv=folds, scoring=scoring)
    dtr = cross_validate(dtr_model, X, y, cv=folds, scoring=scoring)
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
    gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scoring)
    mlp = cross_validate(mlp_model, X, y, cv=folds, scoring=scoring)

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({ 'KNN': [knn['test_accuracy'].mean(),
                                                knn['test_precision'].mean(),
                                                knn['test_recall'].mean(),
                                                knn['test_f1_score'].mean()],
                                        
                                        'Logistic Regression':[log['test_accuracy'].mean(),
                                                               log['test_precision'].mean(),
                                                               log['test_recall'].mean(),
                                                               log['test_f1_score'].mean()],
                                       
                                      'Support Vector Classifier':[svc['test_accuracy'].mean(),
                                                                   svc['test_precision'].mean(),
                                                                   svc['test_recall'].mean(),
                                                                   svc['test_f1_score'].mean()],
                                       
                                      'Decision Tree':[dtr['test_accuracy'].mean(),
                                                       dtr['test_precision'].mean(),
                                                       dtr['test_recall'].mean(),
                                                       dtr['test_f1_score'].mean()],
                                       
                                      'Random Forest':[rfc['test_accuracy'].mean(),
                                                       rfc['test_precision'].mean(),
                                                       rfc['test_recall'].mean(),
                                                       rfc['test_f1_score'].mean()],
                                       
                                      'Gaussian Naive Bayes':[gnb['test_accuracy'].mean(),
                                                              gnb['test_precision'].mean(),
                                                              gnb['test_recall'].mean(),
                                                              gnb['test_f1_score'].mean()],
                                       "MLP": [mlp['test_accuracy'].mean(),
                                                mlp['test_precision'].mean(),
                                                mlp['test_recall'].mean(),
                                                mlp['test_f1_score'].mean()]},
                                       index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
                                      

    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    
    # Return models performance metrics scores data frame
    return(models_scores_table)
  
# Run models_evaluation function
models_evaluation(X, y, 5)
#algorithm comparison with boxplots
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('SVM', SVC()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('MLP', MLPClassifier()))

results = []
names = []
from sklearn.model_selection import cross_val_score
for name, model in models:
    kfold = 5
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
for i in range(len(results)):
    print(names[i],results[i].mean())
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
fig.set_size_inches(8,6)
plt.show()