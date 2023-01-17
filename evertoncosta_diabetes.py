# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy             as np # linear algebra

import pandas            as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn           as sns

import scikitplot        as skplt

import itertools



from sklearn                       import metrics

from sklearn                       import svm

from sklearn.model_selection       import train_test_split

from sklearn.model_selection       import KFold

from sklearn.model_selection       import cross_val_score

from sklearn.linear_model          import LogisticRegression

from sklearn.tree                  import DecisionTreeClassifier

from sklearn.neighbors             import KNeighborsClassifier

from sklearn.ensemble              import RandomForestClassifier

from sklearn.ensemble              import GradientBoostingClassifier

from sklearn.ensemble              import VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics               import accuracy_score

from sklearn.preprocessing         import Imputer

from sklearn.preprocessing         import StandardScaler #Standardisation

from xgboost                       import XGBClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv ('../input/diabetes.csv')
df.shape
df.info()
df.head(20).T
df.describe()
print((df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] == 0).sum())
# Identificando os zeros inválidos como NaN

dfn = df.copy()

dfn[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = dfn[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.NaN)



print(df.isnull().sum())
dfn.head(20).T
valores = df.values



X = valores[:,0:8]



y = valores[:,8]
modelo    = LinearDiscriminantAnalysis()



kfold     = KFold(n_splits=3, random_state=42)



resultado = cross_val_score(modelo, X, y, cv=kfold, scoring='accuracy')



print(resultado.mean())
# Exclui linhas com valor nulo

dft = dfn.copy()

dft.dropna(inplace=True)

# Mostra a nova quantidade de linhas e colunas

dft.shape
valores = dft.values



X = valores[:,0:8]



y = valores[:,8]



modelo    = LinearDiscriminantAnalysis()



kfold     = KFold(n_splits=3, random_state=42)



resultado = cross_val_score(modelo, X, y, cv=kfold, scoring='accuracy')



print(resultado.mean())
dff = dfn.copy()



dff.fillna(dff.mean(), inplace=True)



print(dff.isnull().sum())
valores = dff.values



X = valores[:,0:8]



y = valores[:,8]



modelo    = LinearDiscriminantAnalysis()



kfold     = KFold(n_splits=3, random_state=42)



resultado = cross_val_score(modelo, X, y, cv=kfold, scoring='accuracy')



print(resultado.mean())
# Preenchendo os valores nulos com a média



values = dfn.values



imputer = Imputer()



transformed_values = imputer.fit_transform(values)



# Verificando a quatidade de valores nulos por coluna



print(np.isnan(transformed_values).sum())
model = LinearDiscriminantAnalysis()



kfold = KFold(n_splits=3, random_state=7)



result = cross_val_score(model, transformed_values, y, cv=kfold, scoring='accuracy')



print(result.mean())
df['Pregnancies'].max()
def plotCorrelationMatrix(df, graphWidth):

    dfc = df.copy()

    dfc.dataframeName = 'diabetes.csv'

    filename = dfc.dataframeName

#    dfc = dfc.dropna('columns') # drop columns with NaN

#    dfc = dfc[[col for col in df if dfc[col].nunique() > 1]] # keep columns where there are more than 1 unique values

#    if dfc.shape[1] < 2:

#        print(f'Sem gráfico para mostrar: O número de non-NaN ou constantes ({df.shape[1]}) é menor que 2')

#        return

    corr = dfc.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Matriz de Correlação para {filename}', fontsize=15)

    plt.show()

plotCorrelationMatrix(df, 10)
df.corr()
# Gráficos de distribuição (histogram/bar graph)

def plotPerColumnDistribution(dfd, nGraphShown, nGraphPerRow):

    nunique = dfd.nunique()

    dfd = dfd[[col for col in dfd if nunique[col] > 1 and nunique[col] < 50]]

    nRow, nCol = dfd.shape

    columnNames = list(dfd)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = dfd.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

plotPerColumnDistribution(df, 10, 5)
def plotScatterMatrix(dfs, plotSize, textSize):

    dfs = dfs.select_dtypes(include =[np.number]) # somente colunas numéricas

    # Remove linhas and linha

    dfs = dfs.dropna('columns')

    dfs = dfs[[col for col in dfs if dfs[col].nunique() > 1]] # colunas com mais de um valor único

    columnNames = list(dfs)

    if len(columnNames) > 10: # reduz número de colunas, caso maior que 10 limita a 10

        columnNames = columnNames[:10]

    dfs = dfs[columnNames]

    ax = pd.plotting.scatter_matrix(dfs, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = dfs.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter e Densidade Plot')

    plt.show()

plotScatterMatrix(df, 20, 10)
dfm = df.copy()

max_skinthickness = dfm.SkinThickness.max()

dfm = dfm[dfm.SkinThickness!=max_skinthickness]
dfm.count()
dfz = df.copy()

def replace_zero(dfr, field, target):

    mean_by_target = dfr.loc[dfr[field] != 0, [field, target]].groupby(target).mean()

    dfr.loc[(dfr[field] == 0)&(dfr[target] == 0), field] = mean_by_target.iloc[0][0]

    dfr.loc[(dfr[field] == 0)&(dfr[target] == 1), field] = mean_by_target.iloc[1][0]



    # run the function

for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:   

    replace_zero(dfz, col, 'Outcome')
sns.countplot(x='Outcome',data=dfm)

plt.show()
columns = dfm.columns[:8]

plt.subplots(figsize=(18,15))

length = len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    df[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
dfa=df[df['Outcome']==1]

columns=dfa.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    dfa[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
train, test  = train_test_split(dfm, test_size=0.20, random_state=42,stratify=dfm['Outcome'])

train, valid = train_test_split(train, test_size=0.20, random_state=42)
train.shape, valid.shape, test.shape
feats = [c for c in dfm.columns if c not in ['Outcome']]
rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, max_depth=4, random_state=42)
rf.fit(train[feats], train['Outcome'])
preds = rf.predict(valid[feats])
accuracy_score(valid['Outcome'],preds)
accuracy_score(test['Outcome'],rf.predict(test[feats]))
accuracy_train = []

accuracy_test  = []

for x in range(10):

    if x != 0:

        rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, max_depth=x, random_state=42)

        rf.fit(train[feats], train['Outcome'])

        accuracy_train.append(accuracy_score(valid['Outcome'],rf.predict(valid[feats])))

accuracy_train
pd.Series(accuracy_train).plot.line()
accuracy_trainG = []

for x in range(10):

    if x != 0:

        gbm = GradientBoostingClassifier(n_estimators=290, learning_rate=1.0, max_depth=x, random_state=42)

        gbm.fit(train[feats], train['Outcome'])

        accuracy_trainG.append(accuracy_score(valid['Outcome'], gbm.predict(valid[feats])))

pd.Series(accuracy_trainG).plot.line()
accuracy_trainX = []

for x in range(10):

    if x != 0:

        xgb = XGBClassifier(n_estimators=200, learning_rate=x/100, random_state=42)

        xgb.fit(train[feats], train['Outcome'])

        accuracy_trainX.append(accuracy_score(valid['Outcome'], xgb.predict(valid[feats])))

pd.Series(accuracy_trainX).plot.line()
accuracy_train  = []

accuracy_trainG = []

accuracy_trainX = []

for y in range(4):

    if y != 0:

        for x in range (10):

            if x != 0:

                rf = RandomForestClassifier(n_estimators=y*100, min_samples_split=5, max_depth=x, random_state=42)

                rf.fit(train[feats], train['Outcome'])

                accuracy_train.append(accuracy_score(valid['Outcome'],rf.predict(valid[feats])))

                

                gbm = GradientBoostingClassifier(n_estimators=y*100, learning_rate=1.0, max_depth=x, random_state=42)

                gbm.fit(train[feats], train['Outcome'])

                accuracy_trainG.append(accuracy_score(valid['Outcome'], gbm.predict(valid[feats])))

                

                xgb = XGBClassifier(n_estimators=y*100, learning_rate=x/100, random_state=42)

                xgb.fit(train[feats], train['Outcome'])

                accuracy_trainX.append(accuracy_score(valid['Outcome'], xgb.predict(valid[feats])))
pd.Series(accuracy_train).plot.line()

plt.grid(True)

plt.show()
pd.Series(accuracy_trainG).plot.line()

plt.grid(True)

plt.show()
pd.Series(accuracy_trainX).plot.line()

plt.grid(True)

plt.show()
x=4

y=2

rf = RandomForestClassifier(n_estimators=y*100, min_samples_split=5, max_depth=x, random_state=42)

rf.fit(train[feats], train['Outcome'])

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
x=4

y=2

gbm = GradientBoostingClassifier(n_estimators=y*100, learning_rate=1.0, max_depth=x, random_state=42)

gbm.fit(train[feats], train['Outcome'])

pd.Series(gbm.feature_importances_, index=feats).sort_values().plot.barh()
x=3

y=3

xgb = XGBClassifier(n_estimators=y*100, learning_rate=x/100, random_state=42)

xgb.fit(train[feats], train['Outcome'])

pd.Series(xgb.feature_importances_, index=feats).sort_values().plot.barh()
skplt.metrics.plot_confusion_matrix(valid['Outcome'], preds)
model = RandomForestClassifier(n_estimators=100,random_state=42)

X = dfm[dfm.columns[:8]]

Y = dfm['Outcome']

model.fit(X,Y)

pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False).plot.barh()
dfy = df.copy()

dfy = dfy[['Glucose','BMI','Age','DiabetesPedigreeFunction','Outcome']]



features          = dfy[dfy.columns[:4]]

features_standard = StandardScaler().fit_transform(features)# Gaussian Standardisation

x                 = pd.DataFrame(features_standard,columns=[['Glucose','BMI','Age','DiabetesPedigreeFunction']])

x['Outcome']      = dfy['Outcome']

outcome           = x['Outcome']



train1, test1     = train_test_split(x,test_size=0.25,random_state=0,stratify=x['Outcome'])



train_X1 = train1[train1.columns[:4]]

test_X1  = test1[test1.columns[:4]]



train_Y1 = train1['Outcome']

test_Y1  = test1['Outcome']
abc         = []

classifiers = ['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']

models      = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]

for i in models:

    model       = i

    model.fit(train_X1,train_Y1)

    prediction = model.predict(test_X1)

    abc.append(metrics.accuracy_score(prediction,test_Y1))

new_models_dataframe         = pd.DataFrame(abc,index=classifiers)   

new_models_dataframe.columns = ['New Accuracy']    
outcome = dfm['Outcome']

dfz     = dfm[dfm.columns[:8]]



train,test = train_test_split(df,test_size=0.25,random_state=0,stratify=df['Outcome'])



train_X = train[train.columns[:8]]

test_X  = test[test.columns[:8]]

train_Y = train['Outcome']

test_Y  = test['Outcome']
types = ['rbf','linear']

for i in types:

    model = svm.SVC(kernel=i)

    model.fit(train_X,train_Y)

    prediction = model.predict(test_X)

    print('Acurácia para o SVM =',i,'é',metrics.accuracy_score(prediction,test_Y))
model = LogisticRegression()

model.fit(train_X,train_Y)

prediction = model.predict(test_X)

print('Acurácia para a Logistic Regression é',metrics.accuracy_score(prediction,test_Y))
model = DecisionTreeClassifier()

model.fit(train_X,train_Y)

prediction = model.predict(test_X)

print('Acurácia para a Decision Tree é',metrics.accuracy_score(prediction,test_Y))
linear_svc = svm.SVC(kernel='linear',C=0.1,gamma=10,probability=True)

radial_svm = svm.SVC(kernel='rbf',C=0.1,gamma=10,probability=True)

lr         = LogisticRegression(C=0.1)
ensemble_lin_rbf = VotingClassifier(estimators=[('Linear_svm', linear_svc), ('Radial_svm', radial_svm)], 

                       voting='soft', weights=[2,1]).fit(train_X1,train_Y1)

print('Acurácia para Linear e Radial SVM é:',ensemble_lin_rbf.score(test_X1,test_Y1))
ensemble_lin_lr=VotingClassifier(estimators=[('Linear_svm', linear_svc), ('Logistic Regression', lr)], 

                       voting='soft', weights=[2,1]).fit(train_X1,train_Y1)

print('Acurácia para Linear SVM e Logistic Regression é:',ensemble_lin_lr.score(test_X1,test_Y1))
ensemble_rad_lr=VotingClassifier(estimators=[('Radial_svm', radial_svm), ('Logistic Regression', lr)], 

                       voting='soft', weights=[1,2]).fit(train_X1,train_Y1)

print('Acurácia para Radial SVM e Logistic Regression é:',ensemble_rad_lr.score(test_X1,test_Y1))
ensemble_rad_lr_lin=VotingClassifier(estimators=[('Radial_svm', radial_svm), ('Logistic Regression', lr),('Linear_svm',linear_svc)], 

                       voting='soft', weights=[2,1,3]).fit(train_X1,train_Y1)

print('O modelo ensembled com os 3 classifiers é:',ensemble_rad_lr_lin.score(test_X1,test_Y1))