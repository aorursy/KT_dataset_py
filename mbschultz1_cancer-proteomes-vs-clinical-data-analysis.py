#load libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style("whitegrid")
#load data

proteomes_orig = pd.read_csv('../input/77_cancer_proteomes_CPTAC_itraq.csv')

clinical = pd.read_csv('../input/clinical_data_breast_cancer.csv', index_col=0)

PAM50 = pd.read_csv('../input/PAM50_proteins.csv')
#drop unused columns in proteomes

proteomes = proteomes_orig.drop(['gene_symbol','gene_name'], axis=1)
#Match patient IDs between datasets

clinical.index = clinical.index.to_series().apply(lambda title : title.split('CGA-')[1])

proteomes.rename(columns = proteomes.columns.to_series().apply(lambda title: title.split('.')[0]), inplace=True)
#Transpose and organize proteomes data

proteomes = proteomes.transpose()

proteomes.columns =  proteomes.iloc[0]

proteomes.drop('RefSeq_accession_number', axis=0, inplace=True)
#Convert gender to numbers

def num_gender(gender):

    if gender == 'MALE':

        return 0

    elif gender == 'FEMALE':

        return 1

    else:

        return float('NaN')

    

clinical['Gender'] = clinical['Gender'].apply(lambda gender: num_gender(gender))
#Convert status to numbers

def num_status(status):

    if status == 'Negative':

        return 0

    elif status == 'Positive':

        return 1

    else:

        return 



clinical['ER Status'] = clinical['ER Status'].apply(lambda status: num_status(status))

clinical['PR Status'] = clinical['PR Status'].apply(lambda status: num_status(status))

clinical['HER2 Final Status'] = clinical['HER2 Final Status'].apply(lambda status: num_status(status))
#Convert tumor, node, metastasis to numbers

clinical['Tumor'] = clinical['Tumor'].apply(lambda tumor: tumor.split('T')[1])

clinical['Node'] = clinical['Node'].apply(lambda tumor: tumor.split('N')[1])

clinical['Metastasis'] = clinical['Metastasis'].apply(lambda tumor: tumor.split('M')[1])
#Remove unused columns

clinical.drop('Tumor--T1 Coded', axis=1, inplace=True)

clinical.drop('Metastasis-Coded', axis=1, inplace=True)

clinical.drop('Node-Coded', axis=1, inplace=True)
#Merge clinical data with proteomes data

dataset = clinical.merge(proteomes, left_index=True,right_index=True)
clinical.columns
#Patient age

sns.distplot(dataset['Age at Initial Pathologic Diagnosis'], kde=False, bins=[30,40,50,60,70,80,90], hist_kws=dict(alpha=1))
#Cancer spread to lymph nodes

sns.countplot(dataset['Node'])
#Tumor size

sns.countplot(dataset['Tumor'])
#Tumor stage

fig, ax = plt.subplots(figsize=(12,4))

sns.countplot(sorted(clinical['AJCC Stage']), ax=ax)
#Before I go further, I will fill NaNs with the column average

means = dataset.mean()

dataset = dataset.fillna(means)
#Lets see if I can predict whether the tumor has spread to lymph nodes based on the proteome
#Drop all other columns besides node

nodedata = dataset.drop([x for x in list(clinical.columns.values) if x != 'Node'], axis=1)
#0=Clean lymph nodes, 1=spread to 1+ lymph nodes

nodedata['Node'] = nodedata['Node'].apply(lambda x: 1 if x != '0' else 0)
from sklearn.model_selection import train_test_split

X = nodedata.drop('Node', axis=1)

y = nodedata['Node']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Let's try a kNN model first
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
error_rate = []



# Will take some time

for i in range(1,20):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
#How about logistic regression
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

pred = logmodel.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
#How about random forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=1000)

rfc.fit(X_train,y_train)

pred = rfc.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
#And finally a support vector machine
from sklearn.svm import SVC

svc_model = SVC()

svc_model.fit(X_train,y_train)

pred = svc_model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001, 0.00001], 'kernel': ['rbf']} 

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_

pred = grid.predict(X_test)

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
#Now let me try to predict lymph node invasion as a continuous variable with linear regression
nodedata = dataset.drop([x for x in list(clinical.columns.values) if x != 'Node'], axis=1)
X = nodedata.drop('Node', axis=1)

y = nodedata['Node']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

pred = lm.predict( X_test)

plt.scatter(y_test,pred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
#None of these seem very accurate. Perhaps I can predict another variable better
#Tumor size

sizedata = dataset.drop([x for x in list(clinical.columns.values) if x != 'Tumor'], axis=1)

X = sizedata.drop('Tumor', axis=1)

y = sizedata['Tumor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

pred = lm.predict( X_test)

plt.scatter(y_test,pred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
#Not accurate at all
#Patient age

agedata = dataset.drop([x for x in list(clinical.columns.values) if x != 'Age at Initial Pathologic Diagnosis'], axis=1)

X = agedata.drop('Age at Initial Pathologic Diagnosis', axis=1)

y = agedata['Age at Initial Pathologic Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

pred = lm.predict( X_test)

results = pd.DataFrame({'true_values': y_test, 'predicted_values': pred})

sns.lmplot(data=results, x='true_values', y='predicted_values')
#Not terrible

#And let's look at the residuals

plt.hist(y_test-pred, bins=range(-25,30,5))
#Tumor stage

stagedata = dataset.drop([x for x in list(clinical.columns.values) if x != 'AJCC Stage'], axis=1)



def stagetonum(stage):

    if 'IV' in stage:

        return 4

    elif 'III' in stage:

        return 3

    elif 'II' in stage:

        return 2

    else:

        return 1

    

stagedata['AJCC Stage'] = stagedata['AJCC Stage'].apply(lambda x: stagetonum(x))



X = stagedata.drop('AJCC Stage', axis=1)

y = stagedata['AJCC Stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

pred = lm.predict( X_test)

results = pd.DataFrame({'true_values': y_test, 'predicted_values': pred})

sns.lmplot(data=results, x='true_values', y='predicted_values')
#No predictive value

#Maybe certain genes are more predictive of stage

coeffs = pd.DataFrame(data=lm.coef_, index=X.columns, columns=['Coefficient'])

top50 = abs(coeffs).sort_values('Coefficient', ascending=False)[:50].index

X = X[list(top50)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

pred = lm.predict( X_test)

results = pd.DataFrame({'true_values': y_test, 'predicted_values': pred})

sns.lmplot(data=results, x='true_values', y='predicted_values')
#Even worse
#How about with KNN?

stagedata = dataset.drop([x for x in list(clinical.columns.values) if x != 'AJCC Stage'], axis=1)

stagedata['AJCC Stage'] = stagedata['AJCC Stage'].apply(lambda x: stagetonum(x))

X = stagedata.drop('AJCC Stage', axis=1)

y = stagedata['AJCC Stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
#Or random forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=1000)

rfc.fit(X_train,y_train)

pred = rfc.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
#I should at least be able to predict stage from clinical data, since that is how its defined
#With a linear regression

X = dataset[['Tumor','Node']]

y = stagedata['AJCC Stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

pred = lm.predict( X_test)

results = pd.DataFrame({'true_values': y_test, 'predicted_values': pred})

sns.lmplot(data=results, x='true_values', y='predicted_values')
#That looks good
#To score it based on a classifier:

pred = pred.round()

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
#And with knn

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
#Also very good
#If I add proteomes data with clinical data, can I improve the predictions further?
droplist = [x for x in list(clinical.columns.values) if x != 'Tumor']

droplist = [x for x in list(droplist) if x != 'Node']

X = dataset.drop(droplist, axis=1)

y = stagedata['AJCC Stage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

pred = lm.predict( X_test)

results = pd.DataFrame({'true_values': y_test, 'predicted_values': pred})

sns.lmplot(data=results, x='true_values', y='predicted_values')
#And with knn

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
#Finally, let's cluster based on the PAM50 genes

PamGenes = PAM50['RefSeqProteinID']

PamGenes.apply(lambda x: 1 if x in list(dataset.columns) else 0).mean()
#We only have data on 43% of the PamGenes
PamGenes2 = []

def PaminList(PamGene, list):

    if PamGene in list:

        PamGenes2.append(PamGene)

    else:

        None



for x in PamGenes:

    PaminList(x, list(dataset.columns))
PamData = dataset[PamGenes2]
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)

kmeans.fit(PamData)
kmgroup = kmeans.labels_

from scipy import stats

#Convert data to real values

PamDataReal = 2**(-1*PamData)

#Bonferroni adjusted p-values

pvalues = pd.Series(PamGenes2).apply(lambda x: stats.ttest_ind(PamDataReal[x][kmgroup==0],PamDataReal[x][kmgroup==1]).pvalue)*len(PamGenes2)



#How many genes have significant differences between the groups?

len(pd.Series(PamGenes2)[pvalues > .05])
#Magnitude of the differences 

mags = PamDataReal.apply(lambda x: x[kmgroup==1].mean()/x[kmgroup==0].mean())

magslog2 = np.log2(mags)

sns.distplot(magslog2, kde=False, bins=20)
#Volcanoplot

pvalueswords = pvalues.apply(lambda x: 'Not Significant' if x  > .05 else 'Significant')



volcano = {'x':list(magslog2), 'y':list(-np.log2(pvalues)), 'p-Value':list(pvalueswords)}

volcano = pd.DataFrame(volcano)

sns.lmplot(data=volcano, x='x', y='y', fit_reg=False, hue='p-Value')
type(pvalues)