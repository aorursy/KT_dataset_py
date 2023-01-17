import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
%matplotlib inline
data = pd.read_csv(r'../input/winedataset/02_WineDataset.csv')
pd.options.display.max_columns = None #Display all columns
pd.options.display.max_rows = None #Display all rows
data.head()
data.describe()
data.isnull().any().any() #Check is there is any NULL value in the data set
data.rename(columns={'fixed acidity': 'fixed_acidity',
                     'citric acid':'citric_acid',
                     'volatile acidity':'volatile_acidity',
                     'residual sugar':'residual_sugar',
                     'free sulfur dioxide':'free_sulfur_dioxide',
                     'total sulfur dioxide':'total_sulfur_dioxide'},
            inplace=True)
data.head()
data['quality'].unique()
data.quality.value_counts().sort_index()
plt.figure(figsize=(10,15))

for pl,col in enumerate(list(data.columns.values)):
    plt.subplot(4,3,pl+1)
    sns.set()
    sns.boxplot(col,data=data)
    plt.tight_layout()

sns.catplot(x='quality', data=data, kind='count');
plt.title('Distribution of the Quality');
data.corr()['quality'].sort_values()
data_Cor = data.drop(['fixed_acidity', 'volatile_acidity', 'density', 'residual_sugar', 'chlorides','total_sulfur_dioxide'], axis=1)
sns.pairplot(data_Cor,hue = 'quality');
plt.figure(figsize=(14,8))
ax = sns.heatmap(data_Cor.corr(), annot = True, cmap='RdPu')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.1, top - 0.1);
data['pH'].describe()
data['sulphates'].describe()
data['free_sulfur_dioxide'].describe()
data['alcohol'].describe()
data.iloc[:,:11].head() #Removing the quality column
plt.figure(figsize=(10,15))

for pl,col in enumerate(list(data.iloc[:,:11].columns.values)):
    plt.subplot(4,3,pl+1)
    sns.violinplot(y= data[col],x='quality',data=data, scale='count')
    plt.title(f'quality/{col}')
    plt.tight_layout()
    
    

#This plots a 2d scatter plot with a regression line. Easily showing the correlation, distribution, and outliers!

for col in (data.iloc[:,:11].columns.values):
 
    sns.lmplot(x='quality',y=col,data=data, fit_reg=False)
  
    plt.title(f'quality/{col}');
    plt.ylabel(col);
    plt.show();
    plt.tight_layout();
    plt.close() 
    
    sns.lmplot(x='quality',y=col,data=data)
  
    plt.title(f'quality/{col}');
    plt.ylabel(col);
    plt.show();
    plt.tight_layout();
    plt.close() 
    
    print('   ')
 
condition = [(data['quality']>6),(data['quality']<=4)]#Setting the condition for good and bad ratings

rating = ['good','bad']

data['rating'] = np.select(condition,rating,default='average')
data.rating.value_counts()
data.head(25)
#This cell takes roughly about 15mins to an hour+ to run depending on the specifications of your workstation

for col in data.iloc[:,:11].columns.values:
 
    
    sns.set()
    sns.violinplot(y= col ,x='rating',data=data, scale='count')
    plt.title(f'rating/{col}');
    plt.ylabel(col);
    plt.show();
    plt.tight_layout();
    plt.close() 
    
    sns.set()
    sns.swarmplot(x='rating',y=col,data=data)
    plt.title(f'rating/{col}');
    plt.ylabel(col);
    plt.show();
    plt.tight_layout();
    plt.close() 
    
    print('   ')
    
data[[('rating'),('quality')]].head(25)
data.groupby('rating')['quality'].value_counts()
#This changes the quality from numbers to ratings between good and bad

bins = (2, 4, 9)
group_names = ['bad', 'good']
data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)


data.head(25)
data[[('rating'),('quality')]].head(25)
#This basically maps all good values to 1 and all bad values to 0 in the quality column

dfL = np.array(data['quality'])

dfL = pd.DataFrame(dfL)

data['quality'] = dfL.apply(lambda x: x.map({'good':1,'bad':0})) 



data.head(30)
data[[('rating'),('quality')]].head(25)
#Setting the values of X and Y

X =  data[['alcohol','density','sulphates','pH','free_sulfur_dioxide','citric_acid']]
y =  data['quality']

X_tr,X_t,y_tr,y_t = train_test_split(X,y)
X_tr.shape, X_t.shape
y_tr.shape, y_t.shape
stds= StandardScaler()

X_tr= stds.fit_transform(X_tr)
X_t = stds.fit_transform(X_t)

#The functions below will be used to measure the accuracy of the model

def generateClassificationReport_Tr(y_true,y_pred):
    '''Train data accuracy tester'''
    print(classification_report(y_true,y_pred));
    print(confusion_matrix(y_true,y_pred));
    print('\n\nTrain Accuracy is: ',
          round(100*accuracy_score(y_true,y_pred),3),'%\n');
    
def generateClassificationReport_T(y_true,y_pred):
    '''Test data accuracy tester'''
    print(classification_report(y_true,y_pred));
    print(confusion_matrix(y_true,y_pred));
    print('\n\nTest Accuracy is: ',
          round(100*accuracy_score(y_true,y_pred),3),'%\n');
#LOGISTIC REGRESSION

logr = LogisticRegression(max_iter=1000);
logr.fit(X_tr,y_tr);

#TRAIN DATA

ytr_pred = logr.predict(X_tr)
generateClassificationReport_Tr(y_tr,ytr_pred)

#TEST DATA

yt_pred = logr.predict(X_t)
generateClassificationReport_T(y_t,yt_pred)
#RANDOM FOREST

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_tr,y_tr);
#TRAIN DATA

ytr_pred = rfc.predict(X_tr)
generateClassificationReport_Tr(y_tr,ytr_pred)
#TEST DATA

yt_pred = rfc.predict(X_t);
generateClassificationReport_T(y_t,yt_pred);
#GAUSSIAN NORMAL DISTRIBUTION 

gnd = GaussianNB()
gnd.fit(X_tr,y_tr);
#TRAIN DATA

ytr_pred = gnd.predict(X_tr)
generateClassificationReport_Tr(y_tr,ytr_pred)
#TEST DATA

yt_pred = gnd.predict(X_t)
generateClassificationReport_T(y_t,yt_pred)
#SUPPORT VECTOR CLASSIFIER

svc = SVC()
svc.fit(X_tr,y_tr);
#TRAIN DATA

ytr_pred = svc.predict(X_tr)
generateClassificationReport_Tr(y_tr,ytr_pred)
#TEST DATA

yt_pred = svc.predict(X_t)
generateClassificationReport_T(y_t,yt_pred)
#DESCISION TREE

dtc = DecisionTreeClassifier()
dtc.fit(X_tr,y_tr);
#TRAIN DATA

ytr_pred = dtc.predict(X_tr)
generateClassificationReport_Tr(y_tr,ytr_pred)
#TEST DATA

yt_pred = dtc.predict(X_t)
generateClassificationReport_T(y_t,yt_pred)
#STOCHASTIC GRADIENT DESCENT

sgd = SGDClassifier()
sgd.fit(X_tr, y_tr);
#TRAIN DATA

ytr_pred = sgd.predict(X_tr)
generateClassificationReport_Tr(y_tr,ytr_pred)
#TEST DATA

yt_pred = sgd.predict(X_t)
generateClassificationReport_T(y_t,yt_pred)
