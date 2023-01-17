import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats
import plotly
import itertools
import warnings
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_graphviz
import graphviz
warnings.filterwarnings('ignore')
%matplotlib inline
data = pd.read_csv('../input/data.csv')
data.info()
list = ['id', 'Unnamed: 32']
data.drop(list, axis = 1, inplace = True)
data.head()
data.describe()
plt.figure(figsize=(10,10))
sns.countplot(data['diagnosis'],  palette = "husl")
fig = plt.figure(figsize = (20,15))
plt.subplot(221)
stats.probplot(data['radius_mean'], dist = 'norm', plot = plt)
plt.title('QQPlot for radius mean')
plt.subplot(222)
stats.probplot(data['texture_mean'], dist = 'norm', plot = plt)
plt.title('QQPlot for texture mean')
plt.subplot(223)
stats.probplot(data['perimeter_mean'], dist = 'norm', plot = plt)
plt.title('QQPlot for perimeter mean')
plt.subplot(224)
stats.probplot(data['area_mean'], dist = 'norm', plot = plt)
plt.title('QQPlot for area mean')
fig.suptitle('Features distribution', fontsize = 20)
# To see qqplots for rest features delete #!

#plt.figure(figsize=(15,8))
#stats.probplot(data['smoothness_mean'], dist = 'norm', plot = plt)
#plt.title('QQPlot for smoothness mean')
#plt.show()
#stats.probplot(data['compactness_mean'], dist = 'norm', plot = plt)
#plt.title('QQPlot for compactness mean')
#plt.show()
#stats.probplot(data['concavity_mean'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concavity mean')
#plt.show()
#stats.probplot(data['concave points_mean'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave points mean')
#plt.show()
#stats.probplot(data['fractal_dimension_mean'], dist = 'norm', plot = plt)
#plt.title('QQPlot for fractal dimension mean')
#plt.show()
#stats.probplot(data['radius_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for radius se')
#plt.show()
#stats.probplot(data['texture_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for texture se')
#plt.show()
#stats.probplot(data['perimeter_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for perimeter se')
#plt.show()
#stats.probplot(data['area_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave area se')
#plt.show()
#stats.probplot(data['smoothness_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for smoothness se')
#plt.show()
#stats.probplot(data['compactness_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for compactness se')
#plt.show()
#stats.probplot(data['concavity_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concavity se')
#plt.show()
#stats.probplot(data['concave points_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave points se')
#plt.show()
#stats.probplot(data['symmetry_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave symmetry se')
#plt.show()
#stats.probplot(data['fractal_dimension_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for fractal dimension se')
#plt.show()
#stats.probplot(data['radius_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for radius worst')
#plt.show()
#stats.probplot(data['texture_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for texture worst')
#plt.show()
#stats.probplot(data['perimeter_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for perimeter worst')
#plt.show()
#stats.probplot(data['area_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave area worst')
#plt.show()
#stats.probplot(data['smoothness_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for smoothness worst')
#plt.show()
#stats.probplot(data['compactness_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for compactness worst')
#plt.show()
#stats.probplot(data['concavity_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concavity worst')
#plt.show()
#stats.probplot(data['concave points_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave points worst')
#plt.show()
#stats.probplot(data['symmetry_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave symmetry worst')
#plt.show()
#stats.probplot(data['fractal_dimension_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for fractal dimension worst')
#plt.show()
fig = plt.figure(figsize = (20, 20))
plt.subplot(321)
sns.boxplot(x = data['diagnosis'], y = data['radius_mean'], palette = "husl")
plt.title('Radius mean')
plt.subplot(322)
sns.boxplot(x = data['diagnosis'], y = data['texture_mean'], palette = "husl")
plt.title('Texture mean')
plt.subplot(323)
sns.boxplot(x = data['diagnosis'], y = data['perimeter_mean'], palette = "husl")
plt.title('Perimeter mean')
plt.subplot(324)
sns.boxplot(x = data['diagnosis'], y = data['area_mean'], palette = "husl")
plt.title('Area mean')
plt.subplot(325)
sns.boxplot(x = data['diagnosis'], y = data['smoothness_mean'], palette = "husl")
plt.title('Smoothness mean')
plt.subplot(326)
sns.boxplot(x = data['diagnosis'], y = data['compactness_mean'], palette = "husl")
plt.title('Compactness mean')
fig.suptitle('Features boxplots to compare malignant and benign', fontsize = 20)
fig = plt.figure(figsize = (20, 20))
plt.subplot(321)
sns.boxplot(x = data['diagnosis'], y = data['fractal_dimension_mean'], palette = "husl")
plt.title('Fractal dimension mean')
plt.subplot(322)
sns.boxplot(x = data['diagnosis'], y = data['texture_se'], palette = "husl")
plt.title('Texture se')
plt.subplot(323)
sns.boxplot(x = data['diagnosis'], y = data['smoothness_se'], palette = "husl")
plt.title('Smoothness se')
plt.subplot(324)
sns.boxplot(x = data['diagnosis'], y = data['symmetry_se'], palette = "husl")
plt.title('Symmetry se')
plt.subplot(325)
sns.boxplot(x = data['diagnosis'], y = data['fractal_dimension_se'], palette = "husl")
plt.title('Fractal dimension se')
fig.suptitle('Features boxplots to compare malignant and benign', fontsize = 20)

df1 = data[data['diagnosis'] == 'M']
df2 = data[data['diagnosis'] == 'B']
df1.drop('diagnosis', axis = 1, inplace = True)
df2.drop('diagnosis', axis = 1, inplace = True)
feature = []
t_value = []
p_value = []
for column in df1.columns:
    ttest = stats.ttest_ind(df1[column], df2[column])
    feature.append(column)
    t_value.append(ttest[0])
    p_value.append(ttest[1])
ttest_data = {'feature' : feature, 't_value' : t_value, 'p_value' : p_value}
ttest_df = pd.DataFrame(ttest_data)
ttest_df.loc[ttest_df['p_value'] > 0.05]
fig = plt.figure(figsize = (20,20))
plt.subplot(321)
sns.pointplot(y = data['diagnosis'], x = data['fractal_dimension_mean'], join= False, capsize= 0.1, palette= 'husl')
plt.title('Confidence interval for fractal dimencion mean')
plt.subplot(322)
sns.pointplot(y = data['diagnosis'], x = data['texture_se'], join= False, capsize= 0.1, palette= 'husl')
plt.title('Confidence interval for texture se')
plt.subplot(323)
sns.pointplot(y = data['diagnosis'], x = data['smoothness_se'], join= False, capsize= 0.1, palette= 'husl')
plt.title('Confidence interval for smoothness se')
plt.subplot(324)
sns.pointplot(y = data['diagnosis'], x = data['symmetry_se'], join= False, capsize= 0.1, palette= 'husl')
plt.title('Confidence interval for symmetry se')
plt.subplot(325)
sns.pointplot(y = data['diagnosis'], x = data['fractal_dimension_se'], join= False, capsize= 0.1, palette= 'husl')
plt.title('Confidence interval for fractal dimension se')
fig.suptitle('Confidence intervals', fontsize = 20)
list = ['fractal_dimension_mean', 'texture_se', 'smoothness_se', 'symmetry_se', 'fractal_dimension_se']
data.drop(list, axis = 1, inplace = True)
plt.figure(figsize=(25,20))
plt.title('Correlation matrix')
sns.heatmap(data.corr(), cmap = "Blues_r", annot = True)
fig = plt.figure(figsize = (20,15))
plt.subplot(231)
sns.scatterplot(x = data['perimeter_mean'], y = data['radius_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Perimeter mean vs radius mean')
plt.subplot(232)
sns.scatterplot(x = data['area_mean'], y = data['radius_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs radius mean')
plt.subplot(233)
sns.scatterplot(x = data['radius_mean'], y = data['radius_worst'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Radius mean vs radius worst')
plt.subplot(234)
sns.scatterplot(x = data['area_mean'], y = data['perimeter_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs perimeter mean')
plt.subplot(235)
sns.scatterplot(x = data['area_se'], y = data['perimeter_se'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area se vs perimeter se')
plt.subplot(236)
sns.scatterplot(x = data['perimeter_mean'], y = data['radius_worst'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Perimeter_mean vs radious worst')
fig.suptitle('Correlation > 0.9', fontsize = 20)
fig = plt.figure(figsize = (20,15))
plt.subplot(231)
sns.scatterplot(x = data['concavity_mean'], y = data['concavity_worst'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Concavity mean vs concavity worst (correlation = 0.88)')
plt.subplot(232)
sns.scatterplot(x = data['concavity_mean'], y = data['concave points_worst'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Concavity mean vs concave points worst (correlation = 0.86)')
plt.subplot(233)
sns.scatterplot(x = data['area_mean'], y = data['concave points_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs concave points mean (correlation = 0.82)')
plt.subplot(234)
sns.scatterplot(x = data['area_mean'], y = data['radius_se'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs radius se (correaltion = 0.73)')
plt.subplot(235)
sns.scatterplot(x = data['compactness_mean'], y = data['symmetry_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Compactness mean vs symmetry mean (correlation = 0.6)')
plt.subplot(236)
sns.scatterplot(x = data['area_mean'], y = data['compactness_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs compactness mean (correlation = 0.5)')
fig.suptitle('0.5 < correlation < 0.9', fontsize = 20)
fig = plt.figure(figsize = (20,15))
plt.subplot(221)
sns.scatterplot(x = data['area_mean'], y = data['texture_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs texture mean (correlation = 0.32)')
plt.subplot(222)
sns.scatterplot(x = data['area_mean'], y = data['compactness_se'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs compactness se (correlation = 0.21)')
plt.subplot(223)
sns.scatterplot(x = data['concavity_se'], y = data['texture_worst'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Concavity se vs texture worst (correlation = 0.1)')
plt.subplot(224)
sns.scatterplot(x = data['radius_mean'], y = data['fractal_dimension_worst'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Radius mean vs fractal dimension worst (correaltion = 0.0071)')
fig.suptitle('correlation < 0.5', fontsize = 20)
y = data['diagnosis'].map({'M' : 1, 'B' : 0})
drop_list = ['diagnosis', 'radius_mean', 'perimeter_mean', 'concavity_mean', 'radius_se', 'perimeter_se', 'radius_worst', 'perimeter_worst', 
             'compactness_mean', 'concave points_mean', 'area_se', 'area_worst', 'smoothness_worst', 'compactness_worst', 'compactness_se', 
             'concavity_worst', 'concavity_se', 'fractal_dimension_worst', 'smoothness_mean']
X = data.drop(drop_list, axis = 1)
y.shape, X.shape
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 17)
acc_score = []
f_tree = DecisionTreeClassifier(random_state = 17)
tree_params = {'max_depth' : np.arange(1, 11), 'max_features' : np.arange(1, 8)}
tree_grid = GridSearchCV(f_tree, tree_params, cv = 20, n_jobs = -1)
%%time
tree_grid.fit(X_train, Y_train)
tree_grid.grid_scores_
tree_grid.best_estimator_
dot_data = tree.export_graphviz(tree_grid.best_estimator_, out_file = None, feature_names = X_test.columns, class_names= ['B', 'M'], filled = True, leaves_parallel = True)
graph = graphviz.Source(dot_data)
graph
Y_predict = tree_grid.best_estimator_.predict(X_test)
acc_score.append(accuracy_score(Y_test, Y_predict))
accuracy_score(Y_test, Y_predict)
cm1 = confusion_matrix(Y_test, Y_predict)
classes_name = ['B', 'M']
plt.figure(figsize = (10, 10))
plot_confusion_matrix(cm1, classes = classes_name,  normalize= False, title = 'Confusion matrix for Decision Tree Classifier' )
log_reg_cv = LogisticRegressionCV(n_jobs= -1, random_state= 17, cv = 20, solver= 'lbfgs' )
%%time
log_reg_cv.fit(X_train, Y_train)
Y_predict = log_reg_cv.predict(X_test)
acc_score.append(accuracy_score(Y_test, Y_predict))
accuracy_score(Y_test, Y_predict)
cm2 = confusion_matrix(Y_test, Y_predict)
classes_name = ['B', 'M']
plt.figure(figsize = (10, 10))
plot_confusion_matrix(cm2, classes = classes_name,  normalize= False, title = 'Confusion matrix for Logistic Regression CV Classifier')
log_reg = LogisticRegression()
%%time
log_reg.fit(X_train, Y_train)
Y_predict = log_reg.predict(X_test)
acc_score.append(accuracy_score(Y_test, Y_predict))
accuracy_score(Y_test, Y_predict)
cm3 = confusion_matrix(Y_test, Y_predict)
classes_name = ['B', 'M']
plt.figure(figsize = (10, 10))
plot_confusion_matrix(cm3, classes = classes_name,  normalize= False, title = 'Confusion matrix for Logidtic Regression Classifier')
knn = KNeighborsClassifier(algorithm='ball_tree', weights = 'distance')
knn_params = {'n_neighbors' : np.arange(1, 20)}
grid = GridSearchCV(knn, knn_params, cv = 20, n_jobs = -1)
grid.fit(X_train, Y_train)
grid.grid_scores_
grid.best_estimator_
Y_predict_knn = grid.best_estimator_.predict(X_test)
acc_score.append(accuracy_score(Y_test,Y_predict_knn))
accuracy_score(Y_test, Y_predict_knn)
cm4 = confusion_matrix(Y_test, Y_predict_knn)
classes_name = ['B', 'M']
plt.figure(figsize = (10, 10))
plot_confusion_matrix(cm4, classes = classes_name,  normalize= False, title = 'Confusion matrix for KNN (weight points by the inverse of their distance)')
knn = KNeighborsClassifier(algorithm='ball_tree')
knn_params = {'n_neighbors' : np.arange(1, 20)}
grid = GridSearchCV(knn, knn_params, cv = 20, n_jobs = -1)
grid.fit(X_train, Y_train)
grid.best_estimator_
grid.grid_scores_
Y_predict_knn = grid.best_estimator_.predict(X_test)
acc_score.append(accuracy_score(Y_test, Y_predict_knn))
accuracy_score(Y_test, Y_predict_knn)
cm5 = confusion_matrix(Y_test, Y_predict_knn)
classes_name = ['B', 'M']
plt.figure(figsize = (10, 10))
plot_confusion_matrix(cm5, classes = classes_name,  normalize= False, title = 'Confusion matrix for KNN')
fig = plt.figure(figsize = (30,20))
plt.subplot(321)
classes_name = ['B', 'M']
plot_confusion_matrix(cm1, classes = classes_name,  normalize= False, title = 'Confusion matrix for Decision Tree Classifier' )
plt.subplot(322)
classes_name = ['B', 'M']
plot_confusion_matrix(cm2, classes = classes_name,  normalize= False, title = 'Confusion matrix for Logistic Regression CV Classifier')
plt.subplot(323)
classes_name = ['B', 'M']
plot_confusion_matrix(cm3, classes = classes_name,  normalize= False, title = 'Confusion matrix for Logidtic Regression Classifier')
plt.subplot(324)
classes_name = ['B', 'M']
plot_confusion_matrix(cm4, classes = classes_name,  normalize= False, title = 'Confusion matrix for KNN (weight points by the inverse of their distance)')
plt.subplot(325)
classes_name = ['B', 'M']
plot_confusion_matrix(cm5, classes = classes_name,  normalize= False, title = 'Confusion matrix for KNN')
model_data = {'model' : ['Decision tree', 'Logic Regression CV', 'Logic Regression', 'KNN (distance)', 'KNN'], 'accuracy' : acc_score}
model_df = pd.DataFrame(model_data)
model_df