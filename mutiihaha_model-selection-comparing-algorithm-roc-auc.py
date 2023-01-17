import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')





from sklearn.impute import SimpleImputer

from scipy import stats 

from math import pi



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split 

from sklearn.svm import SVC

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer,confusion_matrix, precision_recall_curve, auc, roc_auc_score,roc_curve,recall_score



from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
data = pd.read_csv('../input/DataSet.csv', delimiter =';', header=None)

data.columns = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','y']

data.head(5)
for h in [1,2,7,13]:

    for i in np.arange(690):

        x = data.iloc[i,h]

        x = x.replace(',','.')



        if x =='?':

            data.iloc[i,h] = np.nan

        else:

            xnew = float(x.replace(',','.'))

            data.iloc[i,h] = xnew

        

data = data.replace('?', np.nan)
#banyak data null

len(data[data.isnull().any(axis=1)])
data.info()
g = sns.pairplot(data, vars=['B','C','H','K','N','O'],hue='y', palette="plasma")

plt.show()
#[1,2,7,10,13,14]

sns.set_style('whitegrid')

f, axes = plt.subplots(2,3, figsize=(17,8)) #1row 2 colums



vis1 = sns.violinplot(data=data, y=data.iloc[:,1], x=data.iloc[:,-1], ax=axes[0,0],palette="Set2")

vis2 = sns.violinplot(data=data, y=data.iloc[:,2], x=data.iloc[:,-1],ax=axes[0,1],palette="Set2")

vis3 = sns.violinplot(data=data, y=data.iloc[:,7], x=data.iloc[:,-1],ax=axes[0,2],palette="Set2")

vis4 = sns.violinplot(data=data, y=data.iloc[:,10], x=data.iloc[:,-1],ax=axes[1,0],palette="Set2")

vis5 = sns.violinplot(data=data, y=data.iloc[:,13], x=data.iloc[:,-1],ax=axes[1,1],palette="Set2")

vis6 = sns.violinplot(data=data, y=data.iloc[:,14], x=data.iloc[:,-1],ax=axes[1,2],palette="Set2")

plt.show()
#[1,2,7,10,13,14]

#['B','C','H','K','N','O']

#Korelasi tertinggiz

#HB,KH,HC,NC



f, axes = plt.subplots(2,2, figsize=(15,15)) #sharex, sharey = True

datadropNA = data.copy()

datadropNA = datadropNA.dropna()

vis1 = sns.kdeplot(datadropNA.H, datadropNA.B,ax=axes[0,0],cmap='Spectral_r',shade=True, shade_lowest=True)

vis2 = sns.kdeplot(datadropNA.K, datadropNA.H,ax=axes[0,1],cmap='Spectral_r',shade=True, shade_lowest=True)

vis3 = sns.kdeplot(datadropNA.H, datadropNA.C,ax=axes[1,0],cmap='Spectral_r',shade=True, shade_lowest=True)

vis4 = sns.kdeplot(datadropNA.N, datadropNA.C,ax=axes[1,1],cmap='Spectral_r',shade=True, shade_lowest=True)



vis1.set(xlim=(-1.5,6))

vis1.set(ylim=(10,60))



vis2.set(xlim=(-1.9,10))

vis2.set(ylim=(-1.5,5))



vis3.set(xlim=(-1.5,5))

vis3.set(ylim=(-4,15))



vis4.set(xlim=(-120,750))

vis4.set(ylim=(-4,20))



plt.show()
#MAXIMUM

maxdata = data.groupby('y').max().loc[:,['B','C','H','K','N','O']]

maxdata.loc[maxdata.index== '+' ,'group'] = 'Pmax'

maxdata.loc[maxdata.index== '-' ,'group'] = 'Nmax'



#RATARATA

meandata = data.groupby('y').mean()

meandata.loc[meandata.index== '+' ,'group'] = 'Pmean'

meandata.loc[meandata.index== '-' ,'group'] = 'Nmean'



#GROUPING

df = maxdata.append([meandata]).iloc[:,[-1,0,1,2,3,4,5]]

df =  df.reset_index()

df = df.iloc[:,1:]

df
# ------- Create background 

categories=list(df)[1:]

N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]

angles += angles[:1]

ax = plt.subplot(111, polar=True)

ax.set_theta_offset(pi / 2)

ax.set_theta_direction(-1)

# Draw one axe per variable + add labels labels yet

plt.xticks(angles[:-1], categories)

# Draw ylabels

ax.set_rlabel_position(0)

plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)

plt.ylim(0,2000)



# ------- Add plots

#Positive

values=df.loc[0].drop('group').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="+ MAX")

ax.fill(angles, values, 'b', alpha=0.1)

#Negative

values=df.loc[1].drop('group').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="- MAX")

ax.fill(angles, values, 'r', alpha=0.1)

# Add legend

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.title('Radar chart of Maximum Values')

plt.show()
# ------- Create background

categories=list(df)[1:]

N = len(categories)

 

angles = [n / float(N) * 2 * pi for n in range(N)]

angles += angles[:1]

 

# Initialise the spider plot

ax = plt.subplot(111, polar=True)

ax.set_theta_offset(pi / 2)

ax.set_theta_direction(-1)

plt.xticks(angles[:-1], categories)

 

# Draw ylabels

ax.set_rlabel_position(0)

plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)

plt.ylim(0,200)



# ------- Add plots

#Positif

values=df.loc[2].drop('group').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="+ MEAN")

ax.fill(angles, values, 'b', alpha=0.1)

#Negative

values=df.loc[3].drop('group').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="- MEAN")

ax.fill(angles, values, 'r', alpha=0.1)

 

# Add legend

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.title('Radar chart of Mean Values')

plt.show()
dataDum = data.copy()

dataDum = dataDum.dropna()

dataCont = dataDum.loc[:,['B','C','H','K','N','O']]

#A,D,E,F,G,I,J,L,M
dataDummies = pd.get_dummies(dataDum['J'])

dataJ = pd.concat([dataCont,dataDummies],axis=1)

Jcorrelations = dataJ.corr().iloc[:6,[-2,-1]]
sns.heatmap(Jcorrelations,annot=True)

plt.show()
#Missing Data #B,N #A,D,E,F,G 

plt.rcParams['figure.figsize']=7,5

sns.heatmap(data.corr(),annot=True)

plt.show()
#before input missing data

sns.set_style('whitegrid')

sns2 = sns.lmplot(data=data, x='B',y='H',fit_reg=False, size=4, scatter_kws={'s':50},aspect=2)
#NEW DATA

corrBnH = data.copy()

corrBnH.loc[corrBnH['B'].isnull(), 'BH'] = 1

corrBnH.loc[corrBnH['B'].notnull(), 'BH'] = 0



imp = SimpleImputer(missing_values=np.nan, strategy='median')

X = corrBnH['B'].values.reshape(-1,1)

imp.fit(X)

corrBnH['B'] = imp.transform(X)

data['B'] = imp.transform(X)



sns.set_style('whitegrid')

sns2 = sns.lmplot(data=corrBnH, x='B',y='H',  hue='BH',fit_reg=False, size=4, scatter_kws={'s':50},aspect=2)

#it doesn't looks like outlier. so go ahead
#before input missing data

sns.set_style('whitegrid')

sns2 = sns.lmplot(data=data, x='N',y='C', fit_reg=False, size=4, scatter_kws={'s':50},aspect=2)
#NEW DATA

corrNC = data.copy()

corrNC.loc[corrNC['N'].isnull(), 'NC'] = 1

corrNC.loc[corrNC['N'].notnull(), 'NC'] = 0



imp = SimpleImputer(missing_values=np.nan, strategy='median')

X = corrNC['N'].values.reshape(-1,1)

imp.fit(X)

corrNC['N'] = imp.transform(X)

data['N'] = imp.transform(X)



sns.set_style('whitegrid')

sns2 = sns.lmplot(data=corrNC, x='N',y='C',  hue='NC', fit_reg=False, size=4, scatter_kws={'s':50},aspect=2)

#it doesn't looks like outlier. so go ahead
#Missing Data #B,N

#A,D,E,F,G
# # Membuat Stack Column Chart dari two-way table

# temp = pd.crosstab(data.A, data.D)

# temp.plot.bar(stacked=True)

# plt.show()
def bivariateAn(data1,data2):

    

    crosstab = pd.crosstab(data1, data2, margins = True)

    crosstab.rename(columns={'All':'row_totals'}, inplace=True)

    crosstab.rename(index={'All':'col_totals'}, inplace=True)

    

    observed = crosstab.iloc[:-1,:-1]

    

    col_names = list(observed.columns)

    index_names = list(observed.index)

    

    expected =  np.outer(crosstab["row_totals"][0:-1],crosstab.loc["col_totals"][0:-1]) / crosstab.iloc[-1,-1]

    expected = pd.DataFrame(expected)

    expected.columns = col_names

    expected.index = index_names

    

    chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()

    crit = stats.chi2.ppf(q = 0.95,df = 1) #Critical value

    p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,df=1)



    

    if chi_squared_stat<crit:

        corr='There is NO correlation'

    else: 

        corr='There is correlation'

        

    return(corr,chi_squared_stat,crit)
AllChis = {}

AllCrit = {}

AllCorr = {}

list1 = ['A','D','E','F','G','I','J','L','M','y']     

for k in np.arange(10) :

    i = list1[k]

    listValChis = []

    listValCrit = []

    listValCorr = []

    for j in ['A','D','E','F','G','I','J','L','M','y']:

        corr,chis,crit=bivariateAn(data[i], data[j])

        listValChis.append(chis)

        listValCrit.append(crit)

        listValCorr.append(corr)

    

    AllChis['{}'.format(list1[k])] = listValChis

    AllCrit['Crirical{}'.format(list1[k])] = listValCrit

    AllCorr['Correlation{}'.format(list1[k])] = listValCorr

    

    # pd.DataFrame({'i':'i','j':j,'ChiSquared':AllChiS,'CriticalVal':AllCrit})
keyslist = list(AllChis.keys())

chisquaredStat = pd.DataFrame()

for a in np.arange(10):

    chisquaredStat[keyslist[a]] = AllChis[keyslist[a]]

chisquaredStat.index=keyslist



#change looks like matrix identity

for x in np.arange(0,10):

    chisquaredStat.iloc[x,x] = 0

chisquaredStat
chisquaredStat.max()
# dengan mempertimbangkan nilai chisquaredStat> critical value dan dengan nilai tertinggi, 

# serta perhitungan terhadap korelasi dari tiap feature (contoh: korelasi tertinggi D dg E,namun karena kedua feature 

# tersebut memiliki data null yg pada index yang persis sama, maka korelasi D diganti dengan M, yang merupakan korelasi tertinggi ke3,

# karena korelasi tertinggi ke2 ada pada G, namun nilai G dan D juga null pada index yang sama)

# berikut korelasi yang diterapkan untuk mendapatkan mode pada tiap kategori.



#A,D,E,F,G #THESeNULL

#A,D,E,F,G,I,J,L,M



# 2 nomor, berarti dilakukan replacenull 2x, karena pada prosess replacenull yg pertama masih menyisakan nilai null

# disebabkan feature dependentnya juga null. contoh: 1.7, replace datanull A dg data F. setelah F replacenull dari data I, 

#data A di replacenull lagi



#tofeature - fromfeature

# 1.7.#A-F >>>---->>(2.A-F)

# 2.  #D-E(D-M)>>>>>

# 3.  #E-D >>>>> 

# 4.6.#F-G(F-A)>>>---->>(1.F-I)

# 5.8.#G-F >>>---->>(3.G-F)
def replaceNullMode(tofeature, fromfeature):

    

    cekMode = data.groupby(fromfeature).apply(lambda x: x.mode().iloc[0])

    lenCekMode = len(cekMode)

    listNilaiTO = list(cekMode[tofeature])

    

    for i in np.arange(lenCekMode):

        nilai1 = cekMode.index[i]

        nilai2 = listNilaiTO[i]

        data.loc[(data[fromfeature]==nilai1)&(data[tofeature].isnull()),tofeature] = nilai2
#replaceNullMode(tofeature, fromfeature)

replaceNullMode('A','F') 

replaceNullMode('D','M')

replaceNullMode('E','D')

replaceNullMode('F','A')

replaceNullMode('G','F')

replaceNullMode('F','I')

replaceNullMode('A','F')

replaceNullMode('G','F')
data.info()

#there is no missing value
X = data.iloc[:,:-1].values

y = data.iloc[:,-1].values
#Karena Data tidak diketahui kategori dengan level (tdk puas(0), puas(1), sangat puas(2))

#atau tanpa adanya level (lk(0), pr(1))

#maka preprocessing preprocessing berikut ini menganggap data berkategori tanpa ada level

#sehingga HARUS DITERAPKAN DUMMIES



#categorical data dengan 2 category = A,I,J,L,y # ga perlu di dummy. cukup di encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



labelencoderA = LabelEncoder()

X[:,0] = labelencoderA.fit_transform(X[:,0])



labelencoderD = LabelEncoder()

X[:,3] = labelencoderD.fit_transform(X[:,3])



labelencoderE = LabelEncoder()

X[:,4] = labelencoderE.fit_transform(X[:,4])



labelencoderF = LabelEncoder()

X[:,5] = labelencoderF.fit_transform(X[:,5])



labelencoderG = LabelEncoder()

X[:,6] = labelencoderG.fit_transform(X[:,6])



labelencoderI = LabelEncoder()

X[:,8] = labelencoderI.fit_transform(X[:,8])



labelencoderJ = LabelEncoder()

X[:,9] = labelencoderJ.fit_transform(X[:,9])



labelencoderL = LabelEncoder()

X[:,11] = labelencoderL.fit_transform(X[:,11])



labelencoderM = LabelEncoder()

X[:,12] = labelencoderM.fit_transform(X[:,12])



# jangan gunakan y[:].error

labelencodery = LabelEncoder()

y= labelencodery.fit_transform(y)
#GAPERLU DI ONEHOTENCODER = A,I,J,L,y (categorynya cuma 2)

#D,E,F,G,M

#3,



#onehotencoder D 

onehotencoderD = OneHotEncoder(categorical_features = [3])

X = onehotencoderD.fit_transform(X).toarray()

#Avoid dummy variable trap

X = X[:,1:]



#onehotencoder E

onehotencoderE = OneHotEncoder(categorical_features = [5])

X = onehotencoderE.fit_transform(X).toarray()

#Avoid dummy variable trap

X = X[:,1:]



#onehotencoder F

onehotencoderF = OneHotEncoder(categorical_features = [7])

X = onehotencoderF.fit_transform(X).toarray()

#Avoid dummy variable trap

X = X[:,1:]



#onehotencoder G

onehotencoderG = OneHotEncoder(categorical_features = [20])

X = onehotencoderG.fit_transform(X).toarray()

#Avoid dummy variable trap

X = X[:,1:]



#onehotencoder M

onehotencoderM = OneHotEncoder(categorical_features = [33])

X = onehotencoderM.fit_transform(X).toarray()

#Avoid dummy variable trap

X = X[:,1:]





#untuk mempermudah update kolom, cek posisi kolom terbaru 

#setelah dilakukan dummy variabel untuk setiap categorical feature yg memiliki > 2 category

cekdata = pd.DataFrame(X)
from sklearn.model_selection import train_test_split 

#Test Data

Xtraindata, Xtest, ytraindata, ytest = train_test_split(X,y,test_size=0.2,random_state = 0,stratify=y)



# #Train & Val

# Xtrain, Xval, ytrain, yval = train_test_split(Xtraindata,ytraindata,test_size=0.2,random_state = 0,stratify=ytraindata)
#Train & Val

Xtrain, Xval, ytrain, yval = train_test_split(Xtraindata,ytraindata,test_size=0.1,random_state = 0,stratify=ytraindata)
# #BOXPLOT UNTUK DECISION U/ FEATURE SCALING

# #[1,2,7,10,13,14]

# sns.set_style('whitegrid')

# f, axes = plt.subplots(2,3, figsize=(17,8)) #1row 2 colums



# vis1 = sns.boxplot(data=data, y=data.iloc[:,1], ax=axes[0,0])

# vis2 = sns.boxplot(data=data, y=data.iloc[:,2], ax=axes[0,1])

# vis3 = sns.boxplot(data=data, y=data.iloc[:,7], ax=axes[0,2])

# vis4 = sns.boxplot(data=data, y=data.iloc[:,10], ax=axes[1,0])

# vis5 = sns.boxplot(data=data, y=data.iloc[:,13], ax=axes[1,1])

# vis6 = sns.boxplot(data=data, y=data.iloc[:,14], ax=axes[1,2])

# plt.show()



# # boxplot(data=dataset[dataset.IncomeGroup=='High income'],

# #                       x=dataset.CountryRegion, 
from sklearn.preprocessing import RobustScaler

robustscaler = RobustScaler()

Xtrain = robustscaler.fit_transform(Xtrain)

Xval = robustscaler.transform(Xval)

Xtest = robustscaler.transform(Xtest)
# LAHdata=pd.DataFrame(X)

# LAHdata['y'] = y

# LAHdata.to_csv('LAHtraindataCermati.csv', index=False)
def LearningCurve(XtrainLC,ytrainLC,XvalLC,yvalLC):

    classifierRF = RandomForestClassifier()



    classifierRF.fit(XtrainLC, ytrainLC) #ini untuk feature selection, jadi pake Xval&yval



    yvalpredLC = classifierRF.predict(XvalLC)

    cmvalLC = confusion_matrix(yvalLC, yvalpredLC)

    accvalLC = (cmvalLC[0,0] + cmvalLC[1,1])/np.sum(cmvalLC)

    errorvalLC = 1-accvalLC



    ytrainpredLC = classifierRF.predict(XtrainLC)

    cmtrainLC = confusion_matrix(ytrainLC, ytrainpredLC)

    acctrainLC = (cmtrainLC[0,0] + cmtrainLC[1,1])/np.sum(cmtrainLC)

    errortrainLC = 1-acctrainLC

    

    return(errortrainLC,errorvalLC)
def gridsearchCurve(Xval,yval):

    rf = RandomForestClassifier()

    rf_param = {'criterion':('gini', 'entropy'), 'min_samples_leaf':(1, 2, 5),

                'max_features':('auto', 'sqrt', 'log2', None), 

                'min_samples_split':(2, 3, 5, 10, 50, 100), 'bootstrap':('True', 'False')}



    rf_clf = GridSearchCV(rf, rf_param, scoring='accuracy', cv=10)

    rf_clf = rf_clf.fit(Xval, yval)

    best_parameterRF = rf_clf.best_params_

    best_accuracyRF = rf_clf.best_score_

    errorRF = 1-best_accuracyRF

    return(errorRF)
error_train, error_val = [], []

num_valdata_size = np.arange(0.1,0.9,0.1)

num_traindata_size = []

for i in num_valdata_size:

    #Train & Val

    XtrainLC, XvalLC, ytrainLC, yvalLC = train_test_split(Xtraindata,ytraindata,test_size=i,random_state = 0,stratify=ytraindata)    

    errortrainLC,errorvalLC = LearningCurve(XtrainLC, ytrainLC, XvalLC,yvalLC)

    errorValGS = gridsearchCurve(XvalLC,yvalLC)

    

    lendata = len(XtrainLC)

    error_train.append(errortrainLC)

    error_val.append(errorValGS)

    num_traindata_size.append(lendata)

    

#VISUALIZING

plt.figure(figsize=(8,5))

plt.plot(num_traindata_size,error_train,label='TrainData')

plt.plot(num_traindata_size,error_val,label='ValidationData')

plt.legend()

plt.title('Learning Curve')

plt.xlabel('Number of Train Data')

plt.ylabel('Error')

# plt.ylim(0.000,0.5)

plt.grid(True)    
num_valdata_size
num_traindata_size
len(Xtrain)
from sklearn.ensemble import RandomForestClassifier as RF

rf = RF()

rf_param = {'criterion':('gini', 'entropy'), 'min_samples_leaf':(1, 2, 5),

            'max_features':('auto', 'sqrt', 'log2', None), 

            'min_samples_split':(2, 3, 5, 10, 50, 100), 'bootstrap':('True', 'False')}



rf_clf = GridSearchCV(rf, rf_param, scoring='accuracy', cv=10)

rf_clf = rf_clf.fit(Xval, yval)

best_parameterRF = rf_clf.best_params_

best_accuracyRF = rf_clf.best_score_



print('best accuracy in data validation:', best_accuracyRF)

print('\nbest parameters:', best_parameterRF)
classifierRF = RandomForestClassifier(bootstrap = True,criterion='gini',max_features='sqrt', 

                                      min_samples_leaf=5, min_samples_split= 3, random_state = 0)

classifierRF.fit(Xtrain, ytrain)

# Predicting the Test set results

ypredRF = classifierRF.predict(Xtest)

cmRF = confusion_matrix(ytest, ypredRF)

accRF = (cmRF[0,0] + cmRF[1,1])/np.sum(cmRF)

print('accuracy Random Forest = ', accRF)
classifierDT = DecisionTreeClassifier(criterion = 'gini',min_samples_leaf=1, min_samples_split=50)

classifierDT.fit(Xtrain,ytrain)



ypredDT = classifierDT.predict(Xtest)

cmDT = confusion_matrix(ytest, ypredDT)



accDT = (cmDT[0,0] + cmDT[1,1])/np.sum(cmDT)

print('accuracy Decision Tree = ', accDT)
classifierKNN = KNeighborsClassifier(algorithm= 'auto',leaf_size=10,n_neighbors=5, p=1, weights='uniform')

classifierKNN.fit(Xtrain, ytrain)

# Predicting the Test set results

ypredKNN = classifierKNN.predict(Xtest)



cmKNN = confusion_matrix(ytest, ypredKNN)

accKNN = (cmKNN[0,0] + cmKNN[1,1])/np.sum(cmKNN)

print('accuracy KNN = ', accKNN)
classifierSVC = SVC(kernel = 'rbf',C=10,gamma=0.01)

classifierSVC.fit(Xtrain, ytrain)



# Predicting the Test set results

ypredSVC = classifierSVC.predict(Xtest)



cmSVC = confusion_matrix(ytest, ypredSVC)



accSVC = (cmSVC[0,0] + cmSVC[1,1])/np.sum(cmSVC)

print('accuracy SVC = ', accSVC)
#RANDOM FOREST

fpr, tpr, thresholds = roc_curve(ytest,ypredRF)

roc_auc = auc(fpr,tpr)

#DECISION TREE

fprDT, tprDT, thresholdsDT = roc_curve(ytest,ypredDT)

roc_aucDT = auc(fprDT,tprDT)

#KNN

fprKNN, tprKNN, thresholdsKNN = roc_curve(ytest,ypredKNN)

roc_aucKNN = auc(fprKNN,tprKNN)

#SVC

fprSVC, tprSVC, thresholdsSVC = roc_curve(ytest,ypredSVC)

roc_aucSVC = auc(fprSVC,tprSVC)



print(roc_auc)

# Plot ROC

plt.title('Receiver Operating Characteristic')



plt.plot(fpr, tpr, 'b',label='Random Forest = %0.2f'% roc_auc, color='red')

plt.plot(fprDT, tprDT, 'b',label='Decision Tree = %0.2f'% roc_aucDT,color='blue')

plt.plot(fprKNN, tprKNN, 'b',label='KNN = %0.2f'% roc_aucKNN,color='green')

plt.plot(fprSVC, tprSVC, 'b',label='SVC = %0.2f'% roc_aucSVC,color='orange')



plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    sns.set_style('white')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')



    #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cm = confusion_matrix(ytest,ypredRF)



#bukan make index untuk akses confusion matrix. 

print("Recall =", round(cm[1,1]/(cm[1,0]+cm[1,1]),2))

print("Precision =",round(cm[1,1]/(cm[0,1]+cm[1,1]),2))

print("Accuracy=",(cm[1,1]+cm[0,0])/(cm[1,1]+cm[0,0]+cm[1,0]+cm[0,1]) )

# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cm

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
from sklearn.metrics import classification_report

print(classification_report(ytest,ypredRF))