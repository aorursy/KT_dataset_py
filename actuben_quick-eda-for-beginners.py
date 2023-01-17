import pandas as pd 
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
% matplotlib inline
df = pd.read_csv('../input/train.csv')
data = df.drop('Survived', axis=1)
y = df.Survived
# few function to get the basic information of our data
def view(data,y):
    print("Shape :",data.shape)  
    print("\nMissing Values\n")
    count_nan=len(data)-data.count()
    result_nan=(count_nan[count_nan!=0]/data.shape[0]).apply(lambda x: '{:.2%}'.format(x))
    print(result_nan)
    print('\nValue Counts\n')
    print(y.value_counts())
    return data.head()

def unique_values(data):
    print("Unique Values")
    unique_values={}
    for col in data.columns:
        unique_values[col]=len(data[col].unique())
    unique = pd.DataFrame(list(unique_values.items()))
    return unique

def types(data):
    print("Types")
    types={}
    for col in data.columns:
        types[col]=data[col].dtype
    types=pd.DataFrame(list(types.items()))  
    return types
# cramer coefficient allow to calculate the correlation between two categorical features 
# https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

from scipy.stats import chi2_contingency
def cramers(crosstab):
    chi2 = chi2_contingency(crosstab)[0]
    n = sum(crosstab.sum())
    return np.sqrt(chi2 / (n*(min(crosstab.shape)-1)))


def df_cramer(data, categorical_features):
    ''' return a dataframe with cramer coefficients'''
    from itertools import combinations
    # list of 2 categorical among all categoricals
    all_combinaisons = list(combinations(categorical_features,2))
    df = {}
    for combinaison in all_combinaisons : 
        crosstab = pd.crosstab(data[categorical_features][combinaison[0]], data[categorical_features][combinaison[1]])
        df[combinaison] = cramers(crosstab)
    return pd.DataFrame(list(df.items()), columns=['Categorical', 'Cramer'])
view(data,y)
unique_values(data)
# first drop all Passenger Id
data = data.drop('PassengerId' , axis=1)
# drop Cabin (too many missing values)
data = data.drop('Cabin', axis = 1)
types(data)
# change types
col_to_categorical = ['Pclass', 'SibSp', 'Parch']


def convert_to_categorical(columns):
    for col in columns:
        data[col]=data[col].astype('object')
        
convert_to_categorical(col_to_categorical)

# numeric and categorical features
numeric_features = data.dtypes[data.dtypes != "object"].index
categorical_features = [col for col in data.columns if col not in numeric_features]
data.describe()
# we now want to fill na for Age and Emarked 
# look at the different correlation
# correlation with numerical features
data.corr(method='spearman')
# let's have a closer look for the Age
fig, ax = plt.subplots(figsize=(12,4),ncols=2)
sns.regplot(data.Age, data.Fare,ax=ax[0], fit_reg=False)
# what's happen if we remove these oultiers
sns.regplot(data.Age[data.Fare<500], data.Fare[data.Fare<500], ax=ax[1], fit_reg=False)
# how Age behave with categorical features 
NROWS=3
NCOLS=2
ax_cord=[(i,j) for i in range(NROWS) for j in range(NCOLS)]
fig, ax =plt.subplots(nrows=NROWS, ncols=NCOLS,figsize=(20,15))
c=0
for col in [x for x in categorical_features if x not in ['Ticket', 'Name']]:
    sns.boxplot(y='Age', x=col, data= data,ax=ax[ax_cord[c]])
    c+=1
plt.tight_layout()

# fill nan of Age
index_NaN_age = list(data["Age"][data["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = data["Age"].median()
    age_pred = data["Age"][((data['SibSp'] == data.iloc[i]["SibSp"]) 
                            & (data['Parch'] == data.iloc[i]["Parch"]) 
                            & (data['Pclass'] == data.iloc[i]["Pclass"]))].median()
    if np.isnan(age_pred) :
        data['Age'].iloc[i] = age_med
    else :
        data['Age'].iloc[i] = age_pred
# fill Embarked by the most common categories (no many missing values)
most_common_embarked = data.Embarked.value_counts().idxmax()
data["Embarked"] = data["Embarked"].fillna(most_common_embarked)
# correlation with categorical features
df_cramer(data,categorical_features)
# don't look at Name and Ticket because they have a lot of unique values
g =sns.pairplot(data[numeric_features])
g.fig.set_size_inches(10,10)
def plot_numeric(data,y,numeric_feat):
    import matplotlib.gridspec as gridspec
    bins = 25
    plt.figure(figsize=(12,28*4))
    gs = gridspec.GridSpec(28, 1)
    for i, f in enumerate(data[numeric_feat]):
        ax = plt.subplot(gs[i])
        sns.distplot(data[f][y == 1], bins = bins, color = 'red',label='Yes')
        sns.distplot(data[f][y == 0], bins = bins, color = 'blue',label='No')
        ax.set_xlabel('')
        ax.set_title('Survived (No vs Yes) of feature: ' + str(f))
        ax.legend()
    plt.show() 
plot_numeric(data,y,numeric_features)
def plot_categorical(data,y,categorical_feat,NROWS, NCOLS):
    sns.set(font_scale=1.5)
    ax_cord=[(i,j) for i in range(NROWS) for j in range(NCOLS)]
    fig, ax =plt.subplots(nrows=NROWS, ncols=NCOLS,figsize=(20,20))
    c=0
    for col in categorical_feat:
        sns.countplot(x=y, hue=data[col], ax=ax[ax_cord[c]])
        sns.barplot(x=data[col],y=y,ax=ax[ax_cord[c+1]])
        ax[ax_cord[c]].legend(loc=1)
        ax[ax_cord[c]].set_title(col)
        c+=2
    plt.tight_layout()

to_plot = [x for x in categorical_features if x not in ['Ticket', 'Name']]
plot_categorical(data,y,to_plot,5,2)

