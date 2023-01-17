import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(style = "white",color_codes=True)
sns.set(font_scale=1.5)
from IPython.display import display
pd.options.display.max_columns = None

from sklearn.model_selection import GridSearchCV #to fine tune Hyperparamters using Grid search
from sklearn.model_selection import RandomizedSearchCV# to seelect the best combination(advance ver of Grid Search)

# importing some ML Algorithms 
from sklearn.linear_model import LinearRegression # y=mx+c
from sklearn.tree import DecisionTreeRegressor # Entropy(impurities),Gain. 
from sklearn.ensemble import RandomForestRegressor # Average of Many DT's

# Testing Libraries - Scipy Stats Models
from scipy.stats import shapiro # Normality Test 1
from scipy.stats import normaltest # Normality Test 2
from scipy.stats import anderson # Normality Test 3
from statsmodels.graphics.gofplots import qqplot # plotting the Distribution of Y with a Line of dot on a 45 degree Line.

# Model Varification/Validation Libraries
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit


# Matrices and Reporting Libraries
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import make_scorer
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import learning_curve
ds = pd.read_csv("../input/incometrain/income-train.csv")
dt = pd.read_csv("../input/incometest/income-test.csv")
ds.head(),dt.head()
ds.shape, dt.shape
ds.dtypes.unique()
ds.select_dtypes(include='float').head()
ds.select_dtypes(include='int64').head()
ds.select_dtypes(include='O').head()
for a in [ds,dt]:
    a['dependency'] = a['dependency'].replace({'yes':1,'no':2}).astype(np.float64)
    a['edjefe'] = a["edjefe"].replace({'yes':1,"no":2}).astype(np.float64)
    a["edjefa"]  = a["edjefa"].replace({'yes':1,"no":2}).astype(np.float64)
ds.select_dtypes(include='O').head()
f_null = ds.select_dtypes("float").isnull().sum()
f_null[f_null > 0]
i_null = ds.select_dtypes("int64").isnull().sum()
i_null[i_null>0]
o_null = ds.select_dtypes("O").isnull().sum()
o_null[o_null>0]
ds.columns , dt.columns
for a,b in zip(ds.columns,dt.columns):
    if a == b:
        print(a)
ds[ds['v2a1'].isnull()][["v2a1",'tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5']]
cleaned_Data = ds[ds['v2a1'].isnull()][['tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5']].sum()
cleaned_Data
# plotting the above analysis. 
plt.style.use("dark_background")

cleaned_Data.plot.bar(figsize =(13,5),color ='red',edgecolor = 'k',linewidth = 3)

plt.title('Monthly Rent Missing Data', size = 18);
plt.xticks([0, 1, 2, 3, 4],
           ['own and fully paid house', 'own-paying installments', 'Rented', 'Precarious', 'Other(assigned,borrowed)'],
           rotation =50)
for null in [ds,dt]:
    null["v2a1"].fillna(value=0,inplace=True)
ds["v2a1"].isnull().sum()
ds.head()
dt['parentesco1'].head()
# Heads of household
HOD = ds.loc[ds['parentesco1'] == 1].copy()
HOD.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())
# plotting the same
plt.figure(figsize = (8, 6))
ds['v18q1'].value_counts().sort_index().plot.bar(color = 'blue',edgecolor = 'k',linewidth = 2)                              
plt.xlabel('v18q1')
plt.ylabel('Count')
plt.title('v18q1 Counts')
for x in [ds, dt]:
    x['v18q1'].fillna(value=0, inplace=True)

ds[['v18q1']].isnull().sum()
# Lets look at the data with not null values first.
ds[ds['rez_esc'].notnull()]['age'].describe()
ds.loc[ds['rez_esc'].isnull()]['age'].describe()
ds.loc[(ds['rez_esc'].isnull() & 
                     ((ds['age'] > 7) & (ds['age'] < 17)))]['age'].describe()
#There is one value that has Null for the 'behind in school' column with age between 7 and 17
ds[(ds['age'] ==10) & ds['rez_esc'].isnull()].head()
ds[(ds['Id'] =='ID_f012e4242')].head()
#there is only one member in household for the member with age 10 and who is 'behind in school'. This explains why the member is 
#behind in school.
#from above we see that  the 'behind in school' column has null values 
# Lets use the above to fix the data
for x in [ds, dt]:
    x['rez_esc'].fillna(value=0, inplace=True)
ds[['rez_esc']].isnull().sum()
data = ds[ds['meaneduc'].isnull()].head()

columns=['edjefe','edjefa','instlevel1','instlevel2']
data[columns][data[columns]['instlevel1']>0].describe()
#from the above, we find that meaneduc is null when no level of education is 0
#Lets fix the data
for x in [ds, dt]:
    x['meaneduc'].fillna(value=0, inplace=True)
ds[['meaneduc']].isnull().sum()
data = ds[ds['SQBmeaned'].isnull()].head()

columns=['edjefe','edjefa','instlevel1','instlevel2']
data[columns][data[columns]['instlevel1']>0].describe()
#from the above, we find that SQBmeaned is null when no level of education is 0
#Lets fix the data
for x in [ds, dt]:
    x['SQBmeaned'].fillna(value=0, inplace=True)
ds[['SQBmeaned']].isnull().sum()
#Lets look at the overall data
null_counts = ds.isnull().sum()
null_counts[null_counts > 0].sort_values(ascending=False)
# Groupby the household and figure out the number of unique values
all_equal = ds.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
#Lets check one household
ds[ds['idhogar'] == not_equal.index[0]][['idhogar', 'parentesco1', 'Target']]
#Lets use Target value of the parent record (head of the household) and update rest. But before that lets check
# if all families has a head. 

households_head = ds.groupby('idhogar')['parentesco1'].sum()

# Find households without a head
households_no_head = ds.loc[ds['idhogar'].isin(households_head[households_head == 0].index), :]

print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))
# Find households without a head and where Target value are different
households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
print('{} Households with no head have different Target value.'.format(sum(households_no_head_equal == False)))
#Lets fix the data
#Set poverty level of the members and the head of the house within a family.
# Iterate through each household
for household in not_equal.index:
    # Find the correct label (for the head of household)
    true_target = int(ds[(ds['idhogar'] == household) & (ds['parentesco1'] == 1.0)]['Target'])
    
    # Set the correct label for all members in the household
    ds.loc[ds['idhogar'] == household, 'Target'] = true_target
    
    
# Groupby the household and figure out the number of unique values
all_equal = ds.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
# 1 = extreme poverty 2 = moderate poverty 3 = vulnerable households 4 = non vulnerable households 
target_counts = HOD['Target'].value_counts().sort_index()
target_counts
target_counts.plot.bar(figsize = (8, 6),linewidth = 2,edgecolor = 'k',title="Target vs Total_Count")
#Lets remove them
print(ds.shape)
cols=['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']


for df in [ds, dt]:
    df.drop(columns = cols,inplace=True)

print(ds.shape)
id_ = ['Id', 'idhogar', 'Target']

ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone']

ind_ordered = ['rez_esc', 'escolari', 'age']

hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
#Check for redundant household variables
heads = ds.loc[ds['parentesco1'] == 1, :]
heads = heads[id_ + hh_bool + hh_cont + hh_ordered]
heads.shape
# Create correlation matrix
corr_matrix = heads.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop
corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9]

sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.7, corr_matrix['tamhog'].abs() > 0.9],
            annot=True, cmap = plt.cm.Accent_r, fmt='.3f')

sns.set(style="whitegrid",font_scale=0.7)
cols=['tamhog', 'hogar_total', 'r4t3']
for x in [ds, dt]:
    x.drop(columns = cols,inplace=True)

ds.shape
#Check for redundant Individual variables
ind = ds[id_ + ind_bool + ind_ordered]
ind.shape
# Create correlation matrix
corr_matrix = ind.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop
# This is simply the opposite of male! We can remove the male flag.
for x in [ds, dt]:
    x.drop(columns = 'male',inplace=True)

ds.shape
#lets check area1 and area2 also
# area1, =1 zona urbana 
# area2, =2 zona rural 
#area2 redundant because we have a column indicating if the house is in a urban zone

for x in [ds, dt]:
    x.drop(columns = 'area2',inplace=True)

ds.shape
#Finally lets delete 'Id', 'idhogar'
cols=['Id','idhogar']
for x in [ds, dt]:
    x.drop(columns = cols,inplace=True)

ds.shape
x_features=ds.iloc[:,0:-1] # feature without target
y_features=ds.iloc[:,-1] # only target
print(x_features.shape)
print(y_features.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report

x_train,x_test,y_train,y_test=train_test_split(x_features,y_features,test_size=0.2,random_state=1)
rmclassifier = RandomForestClassifier()
rmclassifier.fit(x_train,y_train)
y_predict = rmclassifier.predict(x_test)
print(accuracy_score(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))
y_predict_testdata = rmclassifier.predict(dt)
y_predict_testdata
from sklearn.model_selection import KFold,cross_val_score
seed=7
kfold=KFold(n_splits=5,random_state=seed,shuffle=True)

rmclassifier=RandomForestClassifier(random_state=10,n_jobs = -1)
print(cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy'))
results=cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy')
print(results.mean()*100)
num_trees= 100

rmclassifier=RandomForestClassifier(n_estimators=100, random_state=10,n_jobs = -1)
print(cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy'))
results=cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy')
print(results.mean()*100)
rmclassifier.fit(x_features,y_features)
labels = list(x_features)
feature_importances = pd.DataFrame({'feature': labels, 'importance': rmclassifier.feature_importances_})
feature_importances=feature_importances[feature_importances.importance>0.015]
feature_importances.head()
y_predict_testdata = rmclassifier.predict(dt)
y_predict_testdata
feature_importances.sort_values(by=['importance'], ascending=True, inplace=True)
feature_importances['positive'] = feature_importances['importance'] > 0
feature_importances.set_index('feature',inplace=True)
feature_importances.head()

feature_importances.importance.plot(kind='barh', figsize=(11, 6),color = feature_importances.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')