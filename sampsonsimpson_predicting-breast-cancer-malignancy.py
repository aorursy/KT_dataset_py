%matplotlib inline



import numpy as np

import pandas as pd

import seaborn as s

from sklearn import cross_validation

from sklearn.cross_validation import train_test_split

from sklearn.cross_validation import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn import svm

from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
#%% Get and Clean Data



#Read data as pandas dataframe

d = pd.read_csv('../input/data.csv')



df = d.drop('Unnamed: 32', axis=1)



#if using diagnosis as categorical

df.diagnosis = df.diagnosis.astype('category')



#Create references to subset predictor and outcome variables

x = list(df.drop('diagnosis',axis=1).drop('id',axis=1))

y ='diagnosis'



# -- Feature Normalization / Scaling -----------------------------------------

#  Normalize features for SVM and MLPClassifier

#-----------------------------------------------------------------------------

df2 = df[x]

df_norm = (df2 - df2.mean()) / (df2.max() - df2.min())

df_norm = pd.concat([df_norm, df[y]], axis=1)

#-----------------------------------------------------------------------------



#show first 10 rows

df.head(10)
#Explore correlations

plt.rcParams['figure.figsize']=(12,8)

s.set(font_scale=1.4)

s.heatmap(df.drop('diagnosis', axis=1).drop('id',axis=1).corr(), cmap='coolwarm')
plt.rcParams['figure.figsize']=(10,5)

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('diagnosis',y='radius_mean',data=df, ax=ax1)

s.boxplot('diagnosis',y='texture_mean',data=df, ax=ax2)

s.boxplot('diagnosis',y='perimeter_mean',data=df, ax=ax3)

s.boxplot('diagnosis',y='area_mean',data=df, ax=ax4)

s.boxplot('diagnosis',y='smoothness_mean',data=df, ax=ax5)

f.tight_layout()



f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('diagnosis',y='compactness_mean',data=df, ax=ax2)

s.boxplot('diagnosis',y='concavity_mean',data=df, ax=ax1)

s.boxplot('diagnosis',y='concave points_mean',data=df, ax=ax3)

s.boxplot('diagnosis',y='symmetry_mean',data=df, ax=ax4)

s.boxplot('diagnosis',y='fractal_dimension_mean',data=df, ax=ax5)    

f.tight_layout()
#%%

plt.rcParams['figure.figsize']=(10,5)

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('diagnosis',y='radius_se',data=df, ax=ax1, palette='cubehelix')

s.boxplot('diagnosis',y='texture_se',data=df, ax=ax2, palette='cubehelix')

s.boxplot('diagnosis',y='perimeter_se',data=df, ax=ax3, palette='cubehelix')

s.boxplot('diagnosis',y='area_se',data=df, ax=ax4, palette='cubehelix')

s.boxplot('diagnosis',y='smoothness_se',data=df, ax=ax5, palette='cubehelix')

f.tight_layout()



f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('diagnosis',y='compactness_se',data=df, ax=ax2, palette='cubehelix')

s.boxplot('diagnosis',y='concavity_se',data=df, ax=ax1, palette='cubehelix')

s.boxplot('diagnosis',y='concave points_se',data=df, ax=ax3, palette='cubehelix')

s.boxplot('diagnosis',y='symmetry_se',data=df, ax=ax4, palette='cubehelix')

s.boxplot('diagnosis',y='fractal_dimension_se',data=df, ax=ax5, palette='cubehelix')    

f.tight_layout()
plt.rcParams['figure.figsize']=(10,5)

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('diagnosis',y='radius_worst',data=df, ax=ax1, palette='coolwarm')

s.boxplot('diagnosis',y='texture_worst',data=df, ax=ax2, palette='coolwarm')

s.boxplot('diagnosis',y='perimeter_worst',data=df, ax=ax3, palette='coolwarm')

s.boxplot('diagnosis',y='area_worst',data=df, ax=ax4, palette='coolwarm')

s.boxplot('diagnosis',y='smoothness_worst',data=df, ax=ax5, palette='coolwarm')

f.tight_layout()



f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

s.boxplot('diagnosis',y='compactness_worst',data=df, ax=ax2, palette='coolwarm')

s.boxplot('diagnosis',y='concavity_worst',data=df, ax=ax1, palette='coolwarm')

s.boxplot('diagnosis',y='concave points_worst',data=df, ax=ax3, palette='coolwarm')

s.boxplot('diagnosis',y='symmetry_worst',data=df, ax=ax4, palette='coolwarm')

s.boxplot('diagnosis',y='fractal_dimension_worst',data=df, ax=ax5, palette='coolwarm')    

f.tight_layout()
#--------------------------------------------------------------------------------------#

# Train Random Forest

np.random.seed(10)



traindf, testdf = train_test_split(df, test_size = 0.3)



x_train = traindf[x]

y_train = traindf[y]



x_test = testdf[x]

y_test = testdf[y]



forest = RandomForestClassifier(n_estimators=1000)

fit = forest.fit(x_train, y_train)

accuracy = fit.score(x_test, y_test)

predict = fit.predict(x_test)

cmatrix = confusion_matrix(y_test, predict)



#--------------------------------------------------------------------------------------#

# Perform k fold cross-validation





print ('Accuracy of Random Forest: %s' % "{0:.2%}".format(accuracy))



# Cross_Validation

v = cross_val_score(fit, x_train, y_train, cv=10)

for i in range(10):

    print('Cross Validation Score: %s'%'{0:.2%}'.format(v[i,]))
plt.rcParams['figure.figsize']=(14,8)

ax = plt.axes()

s.heatmap(cmatrix, annot=True, fmt='d', ax=ax, cmap='BrBG', annot_kws={"size": 30})

ax.set_title('Random Forest Confusion Matrix')
#%%Feature importances

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]



print("Feature ranking:")

for f in range(traindf[x].shape[1]):

    print("feature %s (%f)" % (list(traindf[x])[f], importances[indices[f]]))
feat_imp = pd.DataFrame({'Feature':list(traindf[x]),

                        'Gini importance':importances[indices]})

plt.rcParams['figure.figsize']=(8,12)

s.set_style('whitegrid')

ax = s.barplot(x='Gini importance', y='Feature', data=feat_imp)

ax.set(xlabel='Gini Importance')

plt.show()
#---------------------------------------------------------------------------------------#

# Train Support Vector Machine ---------------------------------------------------------#

#---------------------------------------------------------------------------------------#



np.random.seed(10)



traindf, testdf = train_test_split(df_norm, test_size = 0.3)



x_train = traindf[x]

y_train = traindf[y]



x_test = testdf[x]

y_test = testdf[y]



svmf = svm.SVC()

svm_fit = svmf.fit(x_train, y_train)

accuracy = svm_fit.score(x_test, y_test)

predict = svm_fit.predict(x_test)

svm_cm = confusion_matrix(y_test, predict)



#--------------------------------------------------------------------------------------#

# Perform k fold cross-validation

print ('Accuracy of Support Vector Machine: %s' % "{0:.2%}".format(accuracy))



# Cross_Validation

v = cross_val_score(svm_fit, x_train, y_train, cv=10)

for i in range(10):

    print('Cross Validation Score: %s'%'{0:.2%}'.format(v[i,]))
#   Visualize SVM Confusion Matrix

plt.rcParams['figure.figsize']=(14,8)

ax = plt.axes()

s.heatmap(svm_cm, annot=True, fmt='d', ax=ax, cmap="YlGnBu", annot_kws={"size": 30})

ax.set_title('Support Vector Machine Confusion Matrix')


#---------------------------------------------------------------------------------------#

# Train MLPClassifier ------------------------------------------------------------------#

#---------------------------------------------------------------------------------------#

np.random.seed(10)



traindf, testdf = train_test_split(df_norm, test_size = 0.3)



x_train = traindf[x]

y_train = traindf[y]



x_test = testdf[x]

y_test = testdf[y]



clf = MLPClassifier(solver='lbfgs', alpha=5, hidden_layer_sizes=(500,), random_state=10)

mlp_fit = clf.fit(x_train, y_train)

accuracy = mlp_fit.score(x_test, y_test)

predict = mlp_fit.predict(x_test)

mlp_cm = confusion_matrix(y_test, predict)



#--------------------------------------------------------------------------------------#

# Perform k fold cross-validation

print ('Accuracy of Multilayer Perceptron: %s' % "{0:.2%}".format(accuracy))



# Cross_Validation

v = cross_val_score(mlp_fit, x_train, y_train, cv=10)

for i in range(10):

    print('Cross Validation Score: %s'%'{0:.2%}'.format(v[i,]))
#   Visualize MLP Confusion Matrix

plt.rcParams['figure.figsize']=(14,8)

ax = plt.axes()

s.heatmap(mlp_cm, annot=True, fmt='d', ax=ax, annot_kws={"size": 30})

ax.set_title('Multilayer Perceptron Confusion Matrix')
diagnosis = df['diagnosis']

mean_cols = [col for col in df.columns if 'mean' in col]

meandf = pd.concat([diagnosis,df[mean_cols]], axis=1)



plt.rcParams['figure.figsize']=(12,12)

g = s.PairGrid(meandf, hue="diagnosis")

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend();



plt.tight_layout()