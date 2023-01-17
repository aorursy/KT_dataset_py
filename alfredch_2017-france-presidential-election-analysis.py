import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns




df = pd.read_csv('../input/myData.csv')
df = df.drop(columns=['Médiane du revenu disponible par UC en 2012'])
df = df.drop(columns=[ 'Ets actifs agriculture en %'])
df = df.drop(columns=[ 'Ets actifs industrie en %'])
df = df.drop(columns=['Ets actifs construction en %'])
df = df.drop(columns=['Ets actifs commerce services en %'])
df = df.drop(columns=['Ets actifs commerce réparation auto en %'])
df = df.drop(columns=['Ets actifs adm publique en %'])
df = df.drop(columns=['Chômeurs 15-64 ans en %'])

print(df.describe())


plt.figure()
df.hist(figsize = (10,10))
plt.show()


sns.heatmap(df.corr())
plt.show()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('../input/myData.csv', delimiter = ',')

# features & target selection
features = ['ARTHAUD', 'ASSELINEAU', 'CHEMINADE', 'DUPONT-AIGNAN', 'FILLON', 'HAMON',
          'LASSALLE', 'LE PEN', 'MACRON', 'MÉLENCHON', 'POUTOU',
          'Médiane du revenu disponible par UC en 2012',
          'Ets actifs agriculture en %', 'Ets actifs industrie en %',
          'Ets actifs construction en %', 'Ets actifs commerce services en %',
          'Ets actifs commerce réparation auto en %', 'Ets actifs adm publique en %',
          'Chômeurs 15-64 ans en %']
x = df.loc[:, features].values
y = df.loc[:,['Nom']].values

# standardizing the features to allow PCA
x = StandardScaler().fit_transform(x)

# pca                        
pca = PCA(n_components=2)

# create new dataframe with the principal components and the target
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 
                                                                  'principal component 2'])
finalDf = pd.concat([principalDf, df[['Nom']]], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

# show the result graphically
targets = ['MACRON', 'LE PEN']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Nom'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 1)
ax.legend(targets)
ax.grid()
plt.show()
plt.clf()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# step 1 read the file
df = pd.read_csv('../input/myData.csv')
df = df.drop(columns=['Médiane du revenu disponible par UC en 2012',
                      'Ets actifs agriculture en %',
                      'Ets actifs industrie en %',
                      'Ets actifs construction en %',
                      'Ets actifs commerce services en %',
                      'Ets actifs commerce réparation auto en %',
                      'Ets actifs adm publique en %',
                      'Chômeurs 15-64 ans en %'])

# bar plot
sns.countplot(x = 'Nom', data = df)
plt.show()
print(df.loc[:, 'Nom'].value_counts())

x = df.loc[:, df.columns != 'Nom']
y = df.loc[:, 'Nom']

standard_scaler = StandardScaler()
Xtr_s = standard_scaler.fit_transform(x)
X = standard_scaler.transform(Xtr_s)

# choose train and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# model  complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []

# loop over different values of k
for i, k in enumerate(neig):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    train_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))
    
# plot
plt.figure(figsize=[13, 8])
plt.plot(neig, test_accuracy, label = 'Testing accuracy')
plt.plot(neig, train_accuracy, label = 'Training accuracy')
plt.legend()
plt.title('k value VS accuracy')
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.show()
print('Best accuracy is {} with k = {}'.format(np.max(test_accuracy), 1+test_accuracy.index(np.max(test_accuracy))))
plt.clf()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# load data
df = pd.read_csv('../input/myData.csv')


# features selection
feature_names = ['ARTHAUD', 'ASSELINEAU', 'CHEMINADE', 'DUPONT-AIGNAN', 'FILLON', 'HAMON',
                 'LASSALLE', 'LE PEN', 'MACRON', 'MÉLENCHON', 'POUTOU']

# features and target assignment
X = df[feature_names]
y = df.Nom

# models selection
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = df.Nom, random_state = 0)
names = []
scores = []

for name, model in models:
    model.fit(X_train, y_train) # train model
    y_pred = model.predict(X_test) # test model
    scores.append(accuracy_score(y_test, y_pred)) # get score of the model
    names.append(name)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 10) # split data
    score = cross_val_score(model, X, y, cv = kfold, scoring = 'accuracy').mean() # evaluate score 
    names.append(name)
    scores.append(score)

kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel = 'Classifier', ylabel = 'Accuracy')

for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha = "center")
plt.show()

# logistic regression
logreg_model = LogisticRegression()
rfecv = RFECV(estimator = logreg_model, step = 1, cv = StratifiedKFold(2), scoring = 'accuracy')
rfecv.fit(X, y)

# show graphical result
plt.figure()
plt.title('Logistic Regression CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
plt.show()
plt.clf()
# dertermine features importance
feature_importance = list(zip(feature_names, rfecv.support_))
print(feature_importance)
new_features = []
for key, value in enumerate(feature_importance):
    if(value[1]) == True:
        new_features.append(value[0])
   
# give new features  
print(new_features)

# calculate accuracy scores
X_new = df[new_features]
initial_score = cross_val_score(logreg_model, X, y, cv=StratifiedKFold(2), scoring='accuracy').mean()
print("Intitial accuracy : {}".format(initial_score))
fe_score = cross_val_score(logreg_model, X_new, y, cv=StratifiedKFold(2), scoring='accuracy').mean()
print("Accuracy after features selection : {}".format(fe_score))

# gradient boosting classifier
gb_model = GradientBoostingClassifier()
gb_rfecv = RFECV(estimator=gb_model, step=1, cv=StratifiedKFold(2), scoring='accuracy')
gb_rfecv.fit(X, y)

# show graphical result
plt.figure()
plt.title('Gradient Boost CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(gb_rfecv.grid_scores_)+1), gb_rfecv.grid_scores_)
plt.show()

# dertermine features importance
feature_importance = list(zip(feature_names, gb_rfecv.support_))
print(feature_importance)
new_features = []
for key, value in enumerate(feature_importance):
    if(value[1]) == True:
        new_features.append(value[0])

# give new features  
print(new_features)

# calculate accuracy scores
X_new = df[new_features]
initial_score = cross_val_score(logreg_model, X, y, cv=StratifiedKFold(2), scoring='accuracy').mean()
print("Intitial accuracy : {}".format(initial_score))
fe_score = cross_val_score(logreg_model, X_new, y, cv=StratifiedKFold(2), scoring='accuracy').mean()
print("Accuracy after features selection : {}".format(fe_score))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from pandas import Series, DataFrame

# load data
df = pd.read_csv('../input/myDataForLinReg.csv', delimiter=',')

# select features and target
X = df.iloc[:,0:11]
y = df.iloc[:,19]

# select train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# create the linear regression model
lr = linear_model.LinearRegression()
# train the model 
lr.fit(X_train,y_train)
# get the mean squared error (MSE)
baseline_error = mean_squared_error(y_test, lr.predict(X_test))
print('MSE', baseline_error)
# get the mean absolute error (MABE)
mean_error = mean_absolute_error(y_test, lr.predict(X_test))
print('MABE : ', mean_error)
# r^2 (coefficient of determination) regression score function
r2_coef = r2_score(y_test, lr.predict(X_test))
print('R^2 : ', r2_coef)

# calculating coefficients
coeff = DataFrame(X_train.columns)
coeff['Coefficient Estimate'] = Series(lr.coef_)
print(coeff)


# RIDGE REGRESSION
ridge = Ridge()
# we define the hyperparameter alpha
n_alphas = 300
alphas = np.logspace(2, 8, n_alphas)
coefs = []
errorsRi = []
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)
    errorsRi.append([mean_error, mean_absolute_error(y_test, ridge.predict(X_test))])

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
ax.legend(X.columns, loc='center left', bbox_to_anchor=(1, 0.5))
plt.axis('tight')
plt.show()

ax = plt.gca()
ax.plot(alphas, errorsRi)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Mean Absolute Error')
plt.axis('tight')
plt.legend(['Linear Regression', 'Ridge Regression'])
plt.show()

print('MABE Linear Reg vs Ridge Reg : ', min(errorsRi))

ridgecv = linear_model.RidgeCV()
ridgecv.fit(X, y)
ridgecv_score = ridgecv.score(X, y)
ridgecv_alpha = ridgecv.alpha_
coeffRi = DataFrame(ridgecv.coef_, X_train.columns)
print(coeffRi)


# LASSO REGRESSION
lasso = Lasso()
n_alphas = 300
alphas = np.logspace(-2, 3, n_alphas)
lasso = linear_model.Lasso(fit_intercept=False)

coefs = []
errorsLa = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    errorsLa.append([mean_error, mean_absolute_error(y_test, lasso.predict(X_test))])

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization')
ax.legend(X.columns, loc='center left', bbox_to_anchor=(1, 0.5))
plt.axis('tight')
plt.show()

ax = plt.gca()
ax.plot(alphas, errorsLa)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Mean Absolute Error')
plt.legend(['Linear Regression', 'Lasso Regression'])
plt.axis('tight')
plt.show()

print('MABE Linear Reg vs Lasso Reg : ', min(errorsLa))

lassocv = linear_model.LassoCV()
lassocv.fit(X, y)
lassocv_score = lassocv.score(X, y)
lassocv_alpha = lassocv.alpha_
coeffLa = DataFrame(lassocv.coef_, X_train.columns)
print(coeffLa)