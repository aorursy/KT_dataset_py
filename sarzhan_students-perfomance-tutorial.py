import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')



#Statistics

from scipy import stats

from scipy.stats import norm, skew

from scipy.special import boxcox1p



#Algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier



from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PolynomialFeatures, Normalizer

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.pipeline import make_pipeline



import warnings 

warnings.filterwarnings('ignore')



%matplotlib inline
#Look at 5 random dataframe samples

df=pd.read_csv('../input/xAPI-Edu-Data.csv')

df.sample(5)
df.info()
n_uniques=df.select_dtypes(include=object).nunique()

n_uniques[n_uniques==2]
df.Class.value_counts(normalize=True)
continuous_variables=df.columns[df.dtypes==int]

plt.figure(figsize=(10,7))

for i, column in enumerate(continuous_variables):

    plt.subplot(2,2, i+1)

    sns.distplot(df[column], label=column, bins=10, fit=norm)

    plt.ylabel('Density');
plt.figure(figsize=(10,7))

for i, column in enumerate(continuous_variables):

    plt.subplot(2,2, i+1)

    df[column]=boxcox1p(df[column], 0.3)

    sns.distplot(df[column], label=column, bins=10, fit=norm)

    plt.ylabel('Density')
df['raisedhands_bin']=np.where(df.raisedhands>df.raisedhands.mean(),1,0)

df['VisITedResources_bin']=np.where(df.VisITedResources>df.VisITedResources.mean(),1,0)
plt.figure(figsize=(10,7))

for i, column in enumerate(continuous_variables):

    plt.subplot(2,2,i+1)

    sns.boxplot(x=df.Class, y=df[column]);
plt.figure(figsize=(7,5))

sns.heatmap(df.corr(), annot=True, fmt='.1g', cmap='RdBu');
sns.pairplot(df);
categorical_variables=df.columns[df.dtypes==object]

print('Percent of students\' nationality - Kuwait or Jordan: {}'.format(

            round(100*df.NationalITy.isin(['KW','Jordan']).sum()/df.shape[0],2)))

print('Percent of students, who was born in Kuwait or Jordan: {}'.format(

            round(100*df.PlaceofBirth.isin(['KuwaIT','Jordan']).sum()/df.shape[0],2)))

print('Percent of studets, who has same nationality and place of birth: {}'.format(

            round(100*(df.NationalITy==df.PlaceofBirth).sum()/df.shape[0])))
df['NationalITy'][df['NationalITy']=='KW']='KuwaIT'

pb_count=pd.DataFrame(df.PlaceofBirth.value_counts(normalize=True)*100)

pb_count.reset_index(inplace=True)

nt_count=pd.DataFrame(df.NationalITy.value_counts(normalize=True)*100)

nt_count.reset_index(inplace=True)

pb_nt_count=pd.merge(nt_count, pb_count, on='index')

pb_nt_count.rename(columns={'index':'Country'}, inplace=True)

pb_nt_count
plt.figure(figsize=(14,5))

for i, column in enumerate(df[['NationalITy','PlaceofBirth']]):

    data=df[column].value_counts().sort_values(ascending=False)

    plt.subplot(1,2,i+1)

    sns.barplot(x=data, y=data.index);
#Rename all coutries with percentage less that 4% to 'Other'

small_countries=list(pb_nt_count['Country'][(pb_nt_count.PlaceofBirth<4)&(pb_nt_count.NationalITy<4)])



for column in ['PlaceofBirth', 'NationalITy']:

    df[column][df[column].isin(small_countries)]='Other'

    

print('After renaming unique values are {}'.format(df.PlaceofBirth.unique()))
plt.figure(figsize=(14,5))

for i, column in enumerate(df[['GradeID','Topic']]):

    data=df[column].value_counts().sort_values(ascending=False)

    plt.subplot(1,2,i+1)

    sns.barplot(x=data, y=data.index);
plt.figure(figsize=(15,8))

for i, column in enumerate(categorical_variables.drop(['NationalITy','PlaceofBirth','GradeID','Topic','Class'])):

    plt.subplot(2,4,i+1)

    sns.countplot(df[column]);
plt.figure(figsize=(15,12))

for i, column in enumerate(categorical_variables.drop(['GradeID','Topic','Class'])):

    plt.subplot(4,3,i+1)

    sns.countplot(x=df.Class, hue=df[column]);
#Cut of target variable from dataset

target=df['Class']

df=df.drop('Class', axis=1)
#Create new feature - type of topic (technical, language, other)

Topic_types={'Math':'technic', 'IT':'technic','Science':'technic','Biology':'technic',

 'Chemistry':'technic', 'Geology':'technic', 'Arabic':'language', 'English':'language',

 'Spanish':'language','French':'language', 'Quran':'other' ,'History':'other'}

df['Topic_type']=df.Topic.map(Topic_types)
for column in continuous_variables:

    SS=StandardScaler().fit(df[[column]])

    df[[column]]=SS.transform(df[[column]])
categorical_variables=df.select_dtypes(include='object').columns

for column in categorical_variables:

    #Binarize and LabelEncode

    #Кодируем переменные, у которых 2 уникальных значения и StageID, GradeID, так как в них важен порядок

    if (df[column].value_counts().shape[0]==2) | (column=='StageID') | (column=='GradeID'):

        le=LabelEncoder().fit(df[column])

        df[column]=le.transform(df[column])



#One-hot encoding

df=pd.get_dummies(df)
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.20, random_state=42)



def modelling(model):

    model.fit(X_train, y_train)

    preds=model.predict(X_test)

    print('Accuracy = {}'.format(100*round(accuracy_score(y_test,preds),2)))

    print(classification_report(y_test, preds))

    plt.figure(figsize=(7,5))

    sns.heatmap(confusion_matrix(y_test,preds), annot=True, vmax=50)

    plt.show()
modelling(make_pipeline(PolynomialFeatures(2),LogisticRegression(random_state=42, C=0.1)))
modelling(XGBClassifier(n_estimators=100, n_jobs=-1, learning_rate=0.03));
modelling(KNeighborsClassifier(n_neighbors=25, n_jobs=-1))
modelling(DecisionTreeClassifier(random_state=42, max_depth=5))
modelling(RandomForestClassifier(n_estimators=2000, n_jobs=-1, max_depth=6, random_state=42))
modelling(SVC(random_state=42, C=10, kernel='rbf', degree=3, gamma=0.1))
modelling(LGBMClassifier(learning_rate=0.02,random_state=42, n_estimators=2000))
criterions=['gini','entropy']

max_depth=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, None]

max_features=[1,2,3,4,5, None]

criterions=['gini','entropy']

min_samples_spilts=[2,3,4,5]

min_samples_leafs=[1,2,3,4]

class_weights=['balanced',None]



max_accuracy=0

best_params=None

best_model=None

for class_weight in class_weights:

    for min_sample_leaf in min_samples_leafs:

        for min_samples_spilt in min_samples_spilts:

            for crit in criterions:

                for splitter in splitters:

                    for depth in max_depth:

                        for feature in max_features:

                            DT=DecisionTreeClassifier(class_weight=class_weight,min_samples_leaf=min_sample_leaf,

                                max_depth=depth, min_samples_split=min_samples_spilt, criterion=crit, splitter=splitter,

                                max_features=feature, random_state=42)

                            DT.fit(X_train, y_train)

                            acc=accuracy_score(y_test,DT.predict(X_test))

                            if acc>max_accuracy:

                                max_accuracy=acc

                                best_model=DT

                                best_params=DT.get_params()

print("Best accuracy at validation set is: {}%".format(round(100*max_accuracy,2)))