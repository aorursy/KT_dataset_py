import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



np.random.seed(123)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

path_dataset = '../input/acamica_diabetes.csv'

df = pd.read_csv(path_dataset)

df_original = df
df.shape
df.head()
df.isna().sum()
df.groupby('Outcome').size()
#sns.boxplot( x=df["Outcome"], y=df["Pregnancies"])



sns.set(rc={'figure.figsize':(5,20)})

f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8)

sns.boxplot(x=df["Outcome"], y=df["Pregnancies"], ax=ax1)

sns.boxplot(x=df["Outcome"], y=df["Glucose"], ax=ax2)

sns.boxplot(x=df["Outcome"], y=df["BloodPressure"], ax=ax3)

sns.boxplot(x=df["Outcome"], y=df["SkinThickness"], ax=ax4)

sns.boxplot(x=df["Outcome"], y=df["Insulin"], ax=ax5)

sns.boxplot(x=df["Outcome"], y=df["BMI"], ax=ax6)

sns.boxplot(x=df["Outcome"], y=df["DiabetesPedigreeFunction"], ax=ax7)

sns.boxplot(x=df["Outcome"], y=df["Age"], ax=ax8);
sns.set(rc={'figure.figsize':(5,20)})

f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8)

sns.barplot(x=df["Outcome"], y=df["Pregnancies"], ax=ax1)

sns.barplot(x=df["Outcome"], y=df["Glucose"], ax=ax2)

sns.barplot(x=df["Outcome"], y=df["BloodPressure"], ax=ax3)

sns.barplot(x=df["Outcome"], y=df["SkinThickness"], ax=ax4)

sns.barplot(x=df["Outcome"], y=df["Insulin"], ax=ax5)

sns.barplot(x=df["Outcome"], y=df["BMI"], ax=ax6)

sns.barplot(x=df["Outcome"], y=df["DiabetesPedigreeFunction"], ax=ax7)

sns.barplot(x=df["Outcome"], y=df["Age"], ax=ax8)

sns.despine(bottom=True)

plt.setp(f.axes, yticks=[])

#plt.tight_layout(h_pad=2);
g = sns.FacetGrid(df, col="Outcome")

g.map(plt.hist, "Pregnancies");
g = sns.FacetGrid(df, col="Outcome")

g.map(plt.hist, "Glucose");
g = sns.FacetGrid(df, col="Outcome")

g.map(plt.hist, "BloodPressure");
g = sns.FacetGrid(df, col="Outcome")

g.map(plt.hist, "SkinThickness");
g = sns.FacetGrid(df, col="Outcome")

g.map(plt.hist, "Insulin");
g = sns.FacetGrid(df, col="Outcome")

g.map(plt.hist, "BMI");
g = sns.FacetGrid(df, col="Outcome")

g.map(plt.hist, "DiabetesPedigreeFunction");
g = sns.FacetGrid(df, col="Outcome")

g.map(plt.hist, "Age");
sns.set(style="white")
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(11, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)





sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})
print("Total : ", df[df.Glucose == 0].shape[0])

print(df[df.Glucose == 0].groupby('Outcome')['Glucose'].count())

print("Total : ", df[df.BloodPressure == 0].shape[0])

print(df[df.BloodPressure == 0].groupby('Outcome')['BloodPressure'].count())

print("Total : ", df[df.SkinThickness == 0].shape[0])

print(df[df.SkinThickness == 0].groupby('Outcome')['SkinThickness'].count())

print("Total : ", df[df.Insulin == 0].shape[0])

print(df[df.Insulin == 0].groupby('Outcome')['Insulin'].count())

print("Total : ", df[df.BMI == 0].shape[0])

print(df[df.BMI == 0].groupby('Outcome')['BMI'].count())

print("Total : ",df[df.DiabetesPedigreeFunction == 0].shape[0])

print(df[df.DiabetesPedigreeFunction == 0].groupby('Outcome')['DiabetesPedigreeFunction'].count())

print("Total : ", df[df.Age == 0].shape[0])

print(df[df.Age == 0].groupby('Outcome')['Age'].count())
#NO MEJORÓ

#df['Outcome'] = df['Outcome'].replace(0, 'Negativo')

#df['Outcome'] = df['Outcome'].replace(1, 'Positivo')

#df['Pregnancies'] = df['Pregnancies'].replace(1, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(2, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(3, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(4, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(5, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(6, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(7, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(8, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(9, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(10, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(11, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(12, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(13, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(14, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(15, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(16, 1)

#df['Pregnancies'] = df['Pregnancies'].replace(17, 1)
df
df.head()
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(df)

transformer
scaled_df = transformer.transform(df)

scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

scaled_df.head()


from sklearn.model_selection import train_test_split

X = scaled_df.drop(['Outcome'], axis=1)

y = scaled_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)



print(X_train.shape[0], X_test.shape[0])
X_train.shape
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.metrics import classification_report
models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('SVC', SVC(gamma='auto')))

models.append(('LR', LogisticRegression()))

models.append(('DT', DecisionTreeClassifier()))

models.append(('RF', RandomForestClassifier()))

models.append(('GB', GradientBoostingClassifier()))
names = []

scores = []

for name, model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    scores.append(accuracy_score(y_test, y_pred))

    names.append(name)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})

print(tr_split)
names = []

scores = []

for name, model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    scores.append(roc_auc_score(y_test, y_pred))

    names.append(name)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})

print(tr_split)
sns.set(rc={'figure.figsize':(10,5)})

axis = sns.barplot(x = 'Name', y = 'Score', data = tr_split)

axis.set(xlabel='Classifier', ylabel='Accuracy')

for p in axis.patches:

    height = p.get_height()

    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 

    

plt.show()
from sklearn.model_selection import GridSearchCV
c_values = list(np.arange(1, 10))

param_grid = [

    {'C': c_values, 'penalty': ['l1'], 'solver' : ['liblinear'], 'multi_class' : ['ovr']},

    {'C': c_values, 'penalty': ['l2'], 'solver' : ['liblinear', 'newton-cg', 'lbfgs'], 'multi_class' : ['ovr']}

]
grid = GridSearchCV(LogisticRegression(), param_grid, cv=10, scoring='accuracy')

grid.fit(X_train, y_train)
print(grid.best_params_)

print(grid.best_estimator_)
logreg_new = LogisticRegression(C=1, multi_class='ovr', penalty='l2', solver='liblinear')

logreg_new.fit(X_train, y_train)

initial_score = cross_val_score(logreg_new, X_test, y_test, cv=10, scoring='accuracy').mean()

print("Final accuracy : {} ".format(initial_score))
y_pred = logreg_new.predict(X_test)
import numpy as np

import itertools

import matplotlib.pylab as plt

plt.rcParams.update(plt.rcParamsDefault)

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



    print(cm)



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



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()



from sklearn.metrics import confusion_matrix



class_names=['0', '1']

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Matriz de confusión sin normalizar (Logistic Regression)')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Matriz de confusión normalizada (Logistic Regression)')



plt.show()
roc_auc_score (y_test, y_pred)
print(classification_report(y_test, y_pred))
from sklearn.feature_selection import RFECV

logreg_model = LogisticRegression(solver = 'lbfgs')

rfecv = RFECV(estimator=logreg_model, step=1, cv=20, scoring='accuracy')

rfecv.fit(X_train, y_train)

rfecv.ranking_
plt.figure()

plt.title('Logistic Regression CV score vs No of Features')

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
new_features = (X_train.columns[rfecv.get_support()])

print(new_features)
# Calculate accuracy scores 

X_new = X_test[new_features]

initial_score = cross_val_score(logreg_model, X_test, y_test, cv=10, scoring='accuracy').mean()

print("Initial accuracy : {} ".format(initial_score))

fe_score = cross_val_score(logreg_model, X_new, y_test, cv=10, scoring='accuracy').mean()

print("Accuracy after Feature Selection : {} ".format(fe_score))
from sklearn.model_selection import GridSearchCV

from sklearn.svm import LinearSVC
param_grid=[

    {'C':[0.001, 0.01, 0.1, 1, 10], 'loss': ['hinge', 'squared_hinge']},

]
linear_clf=LinearSVC(random_state=42)
grid_search = GridSearchCV (linear_clf, param_grid, cv=10, scoring = 'accuracy', refit= True, return_train_score=True,n_jobs=-1)

grid_search.fit(X_train, y_train)
grid_search.scorer_
grid_search.best_params_
grid_search.cv_results_['mean_train_score']
grid_search.best_score_
best_linearSCV = grid_search.best_estimator_

best_linearSCV
y_pred_linear=best_linearSCV.predict(X_test)
plt.rcParams.update(plt.rcParamsDefault)

class_names=['0', '1']

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred_linear)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Matriz de confusión sin normalizar (Linear SVC)')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Matriz de confusión normalizada (Linear SVC)')



plt.show()
roc_auc_score (y_test, y_pred_linear)
print(classification_report(y_test, y_pred_linear))
accuracy_score(y_test, y_pred_linear)
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(X_train, y_train)
y_pred_knn=neigh.predict(X_test)
neigh.score(X_test, y_test)
plt.rcParams.update(plt.rcParamsDefault)

class_names=['0', '1']

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred_knn)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Matriz de confusión sin normalizar(KNN)')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Matriz de confusión normalizada (KNN)')



plt.show()
roc_auc_score (y_test, y_pred_knn)
print(classification_report(y_test, y_pred_knn))
accuracy_score(y_test, y_pred_knn)
param_grid=[

    {'n_estimators':[100, 500, 1000], 'max_depth': [3, 7, None],'min_samples_split': [2,3,10],

     'bootstrap': [True, False],'criterion':['gini','entropy']}

]
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier()
grid_search = GridSearchCV (rnd_clf, param_grid, cv=5, scoring = 'accuracy', return_train_score=True,n_jobs=-1)

grid_search.fit(X_train, y_train)
grid_search.best_params_
best_rfc = grid_search.best_estimator_

y_pred_rfc=best_rfc.predict(X_test)
class_names=['0', '1']

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred_rfc)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Matriz de confusión sin normalizar (Random Forest)')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Matriz de confusión normalizada (Random Forest)')



plt.show()
roc_auc_score (y_test, y_pred_rfc)
print(classification_report(y_test, y_pred_rfc))
accuracy_score(y_test, y_pred_rfc)
from keras.models import Sequential

from keras.layers import Dense
# create model

model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, batch_size=50)


# evaluate the model

scores = model.evaluate(X_train, y_train)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
y_pred_model = model.predict(X_test)
y_pred_model
plt.rcParams.update(plt.rcParamsDefault)

class_names=['0', '1']

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred_model.round())

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Matriz de confusión sin normalizar (Keras Model)')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Matriz de confusión normalizada (Keras Model)')



plt.show()
roc_auc_score (y_test, y_pred_model)
print(classification_report(y_test, y_pred_model.round()))
accuracy_score(y_test, y_pred_model.round())
print('roc_auc_score LogisticRegression:',roc_auc_score (y_test, y_pred))

print('roc_auc_score Linear_SVC:',roc_auc_score (y_test, y_pred_linear))

print('roc_auc_score Random Forest:',roc_auc_score (y_test, y_pred_rfc))

print('roc_auc_score KNN:',roc_auc_score (y_test, y_pred_knn))

print('roc_auc_score KerasNN:',roc_auc_score (y_test, y_pred_model))



print('accuracy_score LogisticRegression:',accuracy_score (y_test, y_pred))

print('accuracy_score Linear_SVC:',accuracy_score (y_test, y_pred_linear))

print('accuracy_score Random Forest:',accuracy_score (y_test, y_pred_rfc))

print('accuracy_score KNN:',accuracy_score (y_test, y_pred_knn))

print('raccuracy_score KerasNN:',accuracy_score (y_test, y_pred_model.round()))


prediction=X_test

prediction['Outcome']=y_test

prediction['Pred']=y_pred_model.round()

prediction.head()

bad_prediction=prediction

bad_prediction.drop(bad_prediction[bad_prediction.Outcome == bad_prediction.Pred].index, inplace=True)
bad_prediction.head()
bad_prediction.shape
bad_prediction