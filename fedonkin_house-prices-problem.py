

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_path = '../input/train.csv'
test_path = '../input/test.csv'
# On charge les données
training_data = pd.read_csv(train_path) 
test_data = pd.read_csv(test_path) 


# Nous allons chercher à modéliser le prix, on le stock dans une variable séparée
Y = training_data.SalePrice

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
#iplot([go.Scatter(x=training_data.head(1000)['SalePrice'], y=training_data.head(1000)['SalePrice'], mode='markers')])

'''iplot([go.Histogram2dContour(x=training_data.head(500)['SalePrice'], 
                             y=training_data.head(500)['SalePrice'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=training_data.head(1000)['SalePrice'], y=training_data.head(1000)['SalePrice'], mode='markers')])
'''
training_data['SalePrice'].value_counts().sort_index().plot.line()


# Y = training_data.SalePrice
#On code le data pour n'avoir que des chiffres. Codage matriciel
print(training_data.dtypes.sample(10))
training_data = pd.get_dummies(training_data)
print(training_data.dtypes.sample(10))
training_data = training_data.dropna(axis=1)
print(training_data.head())
#Y = training_data.SalePrice
#On normalise le data entre 0 et 1
print(training_data.describe()) 
# --> On a 287 parametres possibles

#On normalise
for k in range (1,287):
    #print(training_data.columns)
    #print("Nom de la colonne k",training_data.columns[1])
    #print(training_data.columns[1])
        colonne_k = training_data[training_data.columns[k]]
        max_k = max(colonne_k)
        min_k = min(colonne_k)
        training_data[training_data.columns[k]]=(colonne_k-min_k)/(max_k-min_k)
print(training_data.head())
 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats.stats import pearsonr   
#pearson_corr = np.array((217,217))
pearson_corr = [[0 for x in range(1,217)] for y in range(1,217)] 
i=0
a=[]
for j in range (1,217):
    a.append(j)
    for k in range(1,217):
        colonne_k = training_data[training_data.columns[k]]
        colonne_j = training_data[training_data.columns[j]]
        pearson_corr[j-1][k-1]=pearsonr(colonne_j,colonne_k)[0]
        # Coefficien de coréllation linéaire

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde


# Make the plot
plt.pcolormesh(a, a, pearson_corr,cmap=plt.cm.Greens_r)
plt.show()
 

'''
i=00
for k in range (1,217):
    colonne_k = training_data[training_data.columns[k]]
    pearson_corr_0=pearsonr(Y,(colonne_k))[0]
    pearson_corr_1=pearsonr(Y,(colonne_k))[1]
    if abs(pearson_corr_0) > 0.6 and abs(pearson_corr_1)<1:
        i+=1
        print(i,'Paramètre ',training_data.columns[k],'\t:', k, "\tPearsonCor 0:",pearson_corr_0,"\tPearsonCor 1:",pearson_corr_1 )
        #iplot([go.Scatter(x=colonne_k, y=Y, mode='markers')])
print("fin de boucle linéaire")

# Coefficien de coréllation exponentielle
for k in range (1,287):
    colonne_k = np.exp(training_data[training_data.columns[k]])
    pearson_corr_0=pearsonr(Y,(colonne_k))[0]
    pearson_corr_1=pearsonr(Y,(colonne_k))[1]
    if abs(pearson_corr_0) > 0.6 and abs(pearson_corr_1)<0.05:
        i+=1
        print(i,'Paramètre ',training_data.columns[k],'\t:', k, "\tPearsonCor 0:",pearson_corr_0,"\tPearsonCor 1:",pearson_corr_1 )
        iplot([go.Scatter(x=colonne_k, y=Y, mode='markers')])
print("fin de boucle exponentielle")

#Regression avec données statistiques sur un seul parametre
import statsmodels.api as sm
X = training_data['OverallQual']
y = Y
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model
print("Nouveau Graph"); iplot([go.Scatter(x=predictions, y=Y, mode='markers')])
# Print out the statistics
model.summary()'''
#https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9'''
#On a 287 paramètres en tout, dont le prix. On va choisir les prametres dont la déviation standard est importante
import numpy as np

Std =[]
for k in range (1,217):
    colonne_k = training_data[training_data.columns[k]]
    std_k = np.std(colonne_k)
    Std.append(std_k)
sorted_std = sorted(Std)

import matplotlib.pyplot as plt
plt.plot(sorted_std)

# On prend les 5 parametres qui donnent la plus grande déviation
max_std = max(Std)
i=0
for k in range (1,217):
    colonne_k = training_data[training_data.columns[k]]
    std_k = np.std(colonne_k)
    if std_k > 0.49:
        i+=1
        print('Paramètre ',training_data.columns[i],'\t:', k, "\tstd:",std_k )
iplot([go.Scatter(x=training_data['SalePrice'], y=training_data['MSSubClass'], mode='markers')])


#print(training_data.head())
#On a 290 paramètres en tout. On va choisir les 
#print(training_data.columns)
#print(training_data.describe().index)
#print ("Déviation de la colonne k :", training_data.describe().columns[1])

#print ("Déviation du prix :", training_data.describe().SalePrice[2])
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

data = pd.read_csv('../input/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

#Let's start importing the libraries that we will use
import csv as csv 
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from random import randint
from scipy import stats  

#Here are the sklearn libaries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.decomposition import PCA

#Here we define a function to calculate the Pearson's correlation coefficient
#which we will use in a later part of the notebook
def pearson(x,y):
	if len(x)!=len(y):
		print("I can't calculate Pearson's Coefficient, sets are not of the same length!")
		return
	else:
		sumxy = 0
		for i in range(len(x)):
			sumxy = sumxy + x[i]*y[i]
		return (sumxy - len(x)*np.mean(x)*np.mean(y))\
			/((len(x)-1)*np.std(x, ddof=1)*np.std(y, ddof=1))
        
        
#Import and prepare data
#Import the dataset and do some basic manipulation
traindf = pd.read_csv('../input/train.csv', header=0) 
#testdf = pd.read_csv('../input/test.csv', header=0) I won't use the test set in this notebook

#We can have a look at the data, shape and types, but I'll skip this step here
#traindf.dtypes
#traindf.info()
#traindf.describe
#The dataset is complete, so there's no need here to clean it from empty entries.
#traindf = traindf.dropna() 

#We separate the features from the classes, 
#we can either put them in ndarrays or leave them as pandas dataframes, since sklearn can handle both. 
#x_train = traindf.values[:, 2:] 
#y_train = traindf.values[:, 1]
x_train = traindf.drop(['Id', 'SalePrice'], axis=1)
y_train = traindf.pop('SalePrice')
#x_test = traindf.drop(['id'], axis=1)

#Sometimes it may be useful to encode labels with numeric values, but is unnecessary in this case 
#le = LabelEncoder().fit(traindf['species']) 
#y_train = le.transform(train['species'])
#classes = list(le.classes_)

#However, it's a good idea to standardize the data (namely to rescale it around zero 
#and with unit variance) to avoid that certain unscaled features 
#may weight more on the classifier decision 
scaler = StandardScaler().fit(x_train) #find mean and std for the standardization
x_train = scaler.transform(x_train) #standardize the training values
#x_test = scaler.transform(x_test)

#First classifiers
#Initialise the K-fold with k=5
kfold = KFold(n_splits=5, shuffle=True, random_state=4)

#Initialise Naive Bayes
nb = GaussianNB()
#We can now run the K-fold validation on the dataset with Naive Bayes
#this will output an array of scores, so we can check the mean and standard deviation
nb_validation=[nb.fit(x_train[train], y_train[train]).score(x_train[test], y_train[test]).mean() \
           for train, test in kfold.split(x_train)]

#Initialise Extra-Trees Random Forest
rf = ExtraTreesClassifier(n_estimators=500, random_state=0)
#Run K-fold validation with RF
#Again the classifier is trained on the k-1 sub-sets and then tested on the remaining k-th subset
#and scores are calcualted
rf_validation=[rf.fit(x_train[train], y_train[train]).score(x_train[test], y_train[test]).mean() \
               for train, test in kfold.split(x_train)]

#We extract the importances, their indices and standard deviations
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
imp_std = np.std([est.feature_importances_ for est in rf.estimators_], axis=0)

#And we plot the first and last 10 features out of curiosity
fig = plt.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(1, 2)#, height_ratios=[1, 1]) 
ax1, ax2 = fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1])
ax1.margins(0.05), ax2.margins(0.05) 
ax1.bar(range(10), importances[indices][:10], \
       color="#6480e5", yerr=imp_std[indices][:10], ecolor='#31427e', align="center")
ax2.bar(range(10), importances[indices][-10:], \
       color="#e56464", yerr=imp_std[indices][-10:], ecolor='#7e3131', align="center")
ax1.set_xticks(range(10)), ax2.set_xticks(range(10))
ax1.set_xticklabels(indices[:10]), ax2.set_xticklabels(indices[-10:])
ax1.set_xlim([-1, 10]), ax2.set_xlim([-1, 10])
ax1.set_ylim([0, 0.035]), ax2.set_ylim([0, 0.035])
ax1.set_xlabel('Feature #'), ax2.set_xlabel('Feature #')
ax1.set_ylabel('Random Forest Normalized Importance') 
ax2.set_ylabel('Random Forest Normalized Importance')
ax1.set_title('First 10 Important Features'), ax2.set_title('Last 10 Important Features')
gs1.tight_layout(fig)
#plt.show()


#We first define the ranges for each parameter we are interested in searching 
#(while the others are left as default):
#C is the inverse of the regularization strength
#tol is the tolerance for stopping the criteria
params = {'C':[100, 1000], 'tol': [0.001, 0.0001]}
#We initialise the Logistic Regression
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
#We initialise the Exhaustive Grid Search, we leave the scoring as the default function of 
#the classifier singe log loss gives an error when running with K-fold cross validation
#add n_jobs=-1 in a parallel computing calculation to use all CPUs available
#cv=3 increasing this parameter makes it too difficult for kaggle to run the script
gs = GridSearchCV(lr, params, scoring=None, refit='True', cv=3) 
gs_validation=[gs.fit(x_train[train], y_train[train]).score(x_train[test], y_train[test]).mean() \
               for train, test in kfold.split(x_train)]

print("Validation Results\n==========================================")
print("Naive Bayes: " + '{:1.3f}'.format(np.mean(nb_validation)) + u' \u00B1 ' \
       + '{:1.3f}'.format(np.std(nb_validation)))
print("Random Forest: " + '{:1.3f}'.format(np.mean(rf_validation)) + u' \u00B1 ' \
       + '{:1.3f}'.format(np.std(rf_validation)))
print("Logistic Regression: " + '{:1.3f}'.format(np.mean(gs_validation)) + u' \u00B1 ' \
       + '{:1.3f}'.format(np.std(gs_validation)))
