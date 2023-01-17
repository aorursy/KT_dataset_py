import pandas as pd
import numpy as np

# sklearn preprocessing for dealing with categorical variables and PCA
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# File system manangement
import os

# For string manipulation
import re

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt

# Missing value analysis
import missingno as msno

# modeling 
import lightgbm as lgb

# utilities
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# # Plotly Library
# from plotly.offline import init_notebook_mode, iplot
# import plotly.graph_objs as go
# import plotly.plotly as py
# from plotly import tools
# import plotly.figure_factory as ff
# init_notebook_mode(connected=True)
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_all = pd.concat([df_train, df_test], sort=False)
df = df_train.copy()
data = []
for col in df.columns:
    if col == 'Target':
        role = 'target'
    elif col == 'Id':
        role = 'id'
    else:
        role = 'input'
    
    col_dict = {
        'varname': col,
        'role': role,
        'dtype': df[col].dtype,
        'nunique': df[col].nunique(),
        'response_rate': 100 * df[col].notnull().sum() / df.shape[0]
    }
    data.append(col_dict)

meta = pd.DataFrame(data, columns=['varname', 'role', 'dtype', 'nunique', 'response_rate'])
foo = [
("v2a1"," Monthly rent payment"),
("hacdor"," =1 Overcrowding by bedrooms"),
("rooms","  number of all rooms in the house"),
("hacapo"," =1 Overcrowding by rooms"),
("v14a"," =1 has toilet in the household"),
("refrig"," =1 if the household has refrigerator"),
("v18q"," owns a tablet"),
("v18q1"," number of tablets household owns"),
("r4h1"," Males younger than 12 years of age"),
("r4h2"," Males 12 years of age and older"),
("r4h3"," Total males in the household"),
("r4m1"," Females younger than 12 years of age"),
("r4m2"," Females 12 years of age and older"),
("r4m3"," Total females in the household"),
("r4t1"," persons younger than 12 years of age"),
("r4t2"," persons 12 years of age and older"),
("r4t3"," Total persons in the household"),
("tamhog"," size of the household"),
("tamviv"," no. of persons living in the household"),
("escolari"," years of schooling"),
("rez_esc"," Years behind in school"),
("hhsize"," household size"),
("paredblolad"," =1 if predominant material on the outside wall is block or brick"),
("paredzocalo"," =1 if predominant material on the outside wall is socket (wood, zinc or absbesto"),
("paredpreb"," =1 if predominant material on the outside wall is prefabricated or cement"),
("pareddes"," =1 if predominant material on the outside wall is waste material"),
("paredmad"," =1 if predominant material on the outside wall is wood"),
("paredzinc"," =1 if predominant material on the outside wall is zink"),
("paredfibras"," =1 if predominant material on the outside wall is natural fibers"),
("paredother"," =1 if predominant material on the outside wall is other"),
("pisomoscer"," =1 if predominant material on the floor is mosaic ceramic   terrazo"),
("pisocemento"," =1 if predominant material on the floor is cement"),
("pisoother"," =1 if predominant material on the floor is other"),
("pisonatur"," =1 if predominant material on the floor is  natural material"),
("pisonotiene"," =1 if no floor at the household"),
("pisomadera"," =1 if predominant material on the floor is wood"),
("techozinc"," =1 if predominant material on the roof is metal foil or zink"),
("techoentrepiso"," =1 if predominant material on the roof is fiber cement,   mezzanine "),
("techocane"," =1 if predominant material on the roof is natural fibers"),
("techootro"," =1 if predominant material on the roof is other"),
("cielorazo"," =1 if the house has ceiling"),
("abastaguadentro"," =1 if water provision inside the dwelling"),
("abastaguafuera"," =1 if water provision outside the dwelling"),
("abastaguano"," =1 if no water provision"),
("public"," =1 electricity from CNFL,  ICE, ESPH/JASEC"),
("planpri"," =1 electricity from private plant"),
("noelec"," =1 no electricity in the dwelling"),
("coopele"," =1 electricity from cooperative"),
("sanitario1"," =1 no toilet in the dwelling"),
("sanitario2"," =1 toilet connected to sewer or cesspool"),
("sanitario3"," =1 toilet connected to  septic tank"),
("sanitario5"," =1 toilet connected to black hole or letrine"),
("sanitario6"," =1 toilet connected to other system"),
("energcocinar1"," =1 no main source of energy used for cooking (no kitchen)"),
("energcocinar2"," =1 main source of energy used for cooking electricity"),
("energcocinar3"," =1 main source of energy used for cooking gas"),
("energcocinar4"," =1 main source of energy used for cooking wood charcoal"),
("elimbasu1"," =1 if rubbish disposal mainly by tanker truck"),
("elimbasu2"," =1 if rubbish disposal mainly by botan hollow or buried"),
("elimbasu3"," =1 if rubbish disposal mainly by burning"),
("elimbasu4"," =1 if rubbish disposal mainly by throwing in an unoccupied space"),
("elimbasu5"," =1 if rubbish disposal mainly by throwing in river,   creek or sea"),
("elimbasu6"," =1 if rubbish disposal mainly other"),
("epared1"," =1 if walls are bad"),
("epared2"," =1 if walls are regular"),
("epared3"," =1 if walls are good"),
("etecho1"," =1 if roof are bad"),
("etecho2"," =1 if roof are regular"),
("etecho3"," =1 if roof are good"),
("eviv1"," =1 if floor are bad"),
("eviv2"," =1 if floor are regular"),
("eviv3"," =1 if floor are good"),
("dis"," =1 if disable person"),
("male"," =1 if male"),
("female"," =1 if female"),
("estadocivil1"," =1 if less than 10 years old"),
("estadocivil2"," =1 if free or coupled uunion"),
("estadocivil3"," =1 if married"),
("estadocivil4"," =1 if divorced"),
("estadocivil5"," =1 if separated"),
("estadocivil6"," =1 if widow/er"),
("estadocivil7"," =1 if single"),
("parentesco1"," =1 if household head"),
("parentesco2"," =1 if spouse/partner"),
("parentesco3"," =1 if son/doughter"),
("parentesco4"," =1 if stepson/doughter"),
("parentesco5"," =1 if son/doughter in law"),
("parentesco6"," =1 if grandson/doughter"),
("parentesco7"," =1 if mother/father"),
("parentesco8"," =1 if father/mother in law"),
("parentesco9"," =1 if brother/sister"),
("parentesco10"," =1 if brother/sister in law"),
("parentesco11"," =1 if other family member"),
("parentesco12"," =1 if other non family member"),
("idhogar"," Household level identifier"),
("hogar_nin"," Number of children 0 to 19 in household"),
("hogar_adul"," Number of adults in household"),
("hogar_mayor"," # of individuals 65+ in the household"),
("hogar_total"," # of total individuals in the household"),
("dependency"," Dependency rate"),
("edjefe"," years of education of male head of household"),
("edjefa"," years of education of female head of household"),
("meaneduc","average years of education for adults (18+)"),
("instlevel1"," =1 no level of education"),
("instlevel2"," =1 incomplete primary"),
("instlevel3"," =1 complete primary"),
("instlevel4"," =1 incomplete academic secondary level"),
("instlevel5"," =1 complete academic secondary level"),
("instlevel6"," =1 incomplete technical secondary level"),
("instlevel7"," =1 complete technical secondary level"),
("instlevel8"," =1 undergraduate and higher education"),
("instlevel9"," =1 postgraduate higher education"),
("bedrooms"," number of bedrooms"),
("overcrowding"," # persons per room"),
("tipovivi1"," =1 own and fully paid house"),
("tipovivi2"," =1 own,   paying in installments"),
("tipovivi3"," =1 rented"),
("tipovivi4"," =1 precarious"),
("tipovivi5"," =1 other(assigned"),
("computer"," =1 if the household has notebook or desktop computer,   borrowed)"),
("television"," =1 if the household has TV"),
("mobilephone"," =1 if mobile phone"),
("qmobilephone"," # of mobile phones"),
("lugar1"," =1 region Central"),
("lugar2"," =1 region Chorotega"),
("lugar3"," =1 region PacÃƒÂ­fico central"),
("lugar4"," =1 region Brunca"),
("lugar5"," =1 region Huetar AtlÃƒÂ¡ntica"),
("lugar6"," =1 region Huetar Norte"),
("area1"," =1 zona urbana"),
("area2"," =2 zona rural"),
("age"," Age in years"),
("SQBescolari"," escolari squared"),
("SQBage"," age squared"),
("SQBhogar_total"," hogar_total squared"),
("SQBedjefe"," edjefe squared"),
("SQBhogar_nin"," hogar_nin squared"),
("SQBovercrowding"," overcrowding squared"),
("SQBdependency"," dependency squared"),
("SQBmeaned"," meaned squared"),
("agesq"," Age squared"),]

description = pd.DataFrame(foo, columns=['varname', 'description'])
meta = meta.merge(description, on='varname')
meta.sort_values(by='response_rate').head(10)
df.info()
df.head()
meta.iloc[:20,:]
df.loc[df['v2a1'].isnull(),['Id','Target']].groupby(by='Target').count().plot()
print(df['v18q1'].unique())
df.loc[df['v18q1'].isnull(),['Id','Target']].groupby(by='Target').count().plot()
meta.iloc[20:80,:]
meta.iloc[80:141,:]
((df['Target'].value_counts())*100/len(df)).plot(kind='bar')
plt.ylabel("Percent")
#Correlation heatmap with target variable for 20 most correlated variables (Credit - https://www.kaggle.com/ishaan45)
corrmat = df.corr().abs()['Target'].sort_values(ascending=False).drop('Target')
corr_df = corrmat.to_frame(name='values')
plt.figure(figsize=(12,8))
sns.heatmap(corr_df[:20])
meta.loc[meta['varname'].isin(corr_df[:20].index)]
def missing_values(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()*100/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
# Scikit-learn Imputer for missing-value imputation
df_numerical = df.select_dtypes(exclude=['object'])
imputed_df = pd.DataFrame(Imputer(missing_values ='NaN', strategy='mean', axis=0).fit_transform(df_numerical.values), columns=df_numerical.columns)
# Missing value analysis using 'missingno' package by Aleksey Bilogur
print(msno.matrix(df_train.sample(100)))

# Zooming-in on first 25 features
print(msno.matrix(df_train.iloc[0:100, :25]))
missing_df = missing_values(df).head(10).reset_index()
missing_df
meta.loc[meta['varname'].isin(missing_df['index'].head(5).values)]
df['v2a1'].fillna(0.0, inplace = True)
df['v18q1'].fillna(0.0, inplace = True)
df['rez_esc'].fillna(0.0, inplace = True)

df['meaneduc'].fillna(df['meaneduc'].mean(), inplace = True)
df['SQBmeaned'].fillna(df['SQBmeaned'].mean(), inplace = True)
missing_values(df).head(5)
# No. of unique values in categorical columns
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
#Identifying categorical/binary values 
meta.sort_values(by='nunique')
#get binary variable dataframe
binary = list(meta.loc[meta['nunique'].values == 2, 'varname']) + ["Target"]

fig = plt.figure(figsize=(25,120))
fig.subplots_adjust(hspace=0.4)

for i,col in enumerate(binary[:len(binary)-1]):
    ax = fig.add_subplot(26,4,(i+1))
    sns.violinplot(x='Target', y=col, data=df, hue='Target', ax=ax)
    title = meta.loc[meta['varname'] == col, 'description'].iloc[0]
    title_clean = re.sub("[^a-zA-Z ]","", title)
    plt.title(title_clean)
    ax.legend_.remove()
cols_assets = ["v18q", "Refrig", "computer", "television", "mobilephone"]
titles_assets = ["Tablet", "Refrigirator", "Computer", "Television", "MobilePhone"]
dict_outside = {'paredblolad' : "Block / Brick", "paredpreb" : "Cement", "paredmad" : "Wood",
      "paredzocalo" : "Socket", "pareddes" : "Waste Material", "paredfibras" : "Fibres",
      "paredother" : "Other", "paredzinc": "Zink"}

dict_floor = {'pisomoscer' : "Mosaic / Ceramic", "pisocemento" : "Cement", "pisonatur" : "Natural Material",
      "pisonotiene" : "No Floor", "pisomadera" : "Wood", "pisoother" : "Other"}

dict_roof = {'techozinc' : "Zinc", "techoentrepiso" : "Fibre / Cement", "techocane" : "Natural Fibre", "techootro" : "Other"}

dict_sanitary = {'sanitario1' : "No Toilet", "sanitario2" : "Sewer / Cesspool", "sanitario3" : "Septic Tank",
       "sanitario5" : "Black Hole", "sanitario6" : "Other System"}

dict_energy = {'energcocinar1' : "No Kitchen", "energcocinar2" : "Electricity", "energcocinar3" : "Cooking Gas",
       "energcocinar4" : "Wood Charcoal"}

dict_disposal = {"elimbasu1":"Tanker truck", "elimbasu2": "Buried", "elimbasu3": "Burning", "elimbasu4": "Unoccupied space", 
       "elimbasu5": "River", "elimbasu6": "Other"}

titles_residence = ["Outside Wall Material", "Floor Material", "Roof Material", "Sanitary Conditions", "Cooking Energy Sources", "Disposal Methods"]
dict_edu = {"instlevel1": "No Education", "instlevel2": "Incomplete Primary", "instlevel3": "Complete Primary", 
       "instlevel4": "Incomplete Sc.", "instlevel5": "Complete Sc.", "instlevel6": "Incomplete Tech Sc.",
       "instlevel7": "Complete Tech Sc.", "instlevel8": "Undergraduation", "instlevel9": "Postgraduation"}

dict_marital = {"estadocivil1": "< 10 years", "estadocivil2": "Free / Coupled union", "estadocivil3": "Married", 
       "estadocivil4": "Divorced", "estadocivil5": "Separated", "estadocivil6": "Widow", "estadocivil7": "Single"}

dict_member = {"parentesco1": "Household Head", "parentesco2": "Spouse/Partner", "parentesco3": "Son/Daughter", 
       "parentesco4": "Stepson/Daughter", "parentesco5" : "Son/Daughter in Law" , "parentesco6": "Grandson/Daughter", 
       "parentesco7": "Mother/Father", "parentesco8": "Mother/Father in Law", "parentesco9" : "Brother/Sister" , 
       "parentesco10" : "Brother/Sister in law", "parentesco11" : "Other Family Member", "parentesco12" : "Other Non Family Member"}
# Note: We have to observe manually for the limit of unique values limit, here it is 22
meta[meta['nunique'].values > 2].sort_values(by='nunique')
categorical = list(meta.loc[(meta['nunique'].between(3, 22, inclusive=True)) & (meta['dtype']!='object'), 'varname'])

fig = plt.figure(figsize=(20,30))
fig.subplots_adjust(hspace=0.4)

for i,col in enumerate(categorical[:len(categorical)-1]):
    ax = fig.add_subplot(7,4,(i+1))
    sns.barplot(x='Target', y=col, data=df, ax=ax)
    title = meta.loc[meta['varname'] == col, 'description'].iloc[0]
    title_clean = re.sub("[^a-zA-Z ]","", title)
    plt.title(title_clean)
meta[meta['nunique'].values >20]
cont_collist = list(meta.loc[(meta['nunique'] > 20) & (meta['dtype']!='object'), 'varname'])

fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(hspace=0.4)

for i,col in enumerate(cont_collist[:len(cont_collist)]):
    ax = fig.add_subplot(4,3,(i+1))
    sns.kdeplot(df[col], legend=False, ax=ax)
    title = meta.loc[meta['varname'] == col, 'description'].iloc[0]
    title_clean = re.sub("[^a-zA-Z ]","", title)
    plt.title(title_clean)
# Threshold for removing correlated variables
threshold = 0.9

# Absolute value correlation matrix
corr_matrix = df.corr().abs()
corr_matrix.head()

# Upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print('There are %d columns to remove.' % (len(to_drop)))

# Drop correlated variables
train_df = df.drop(columns = to_drop)
print('Training shape: ', df.shape)
train_df = train_df.drop(df.select_dtypes('object').columns, axis=1).reset_index()

# Initialize an empty array to hold feature importances
feature_importances = np.zeros(train_df.shape[1])

# Create the model with several hyperparameters
model = lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 10000, 
                           class_weight = 'balanced')
labels = train_df['Target']
ids = train_df['index']

# Fit the model twice to avoid overfitting
for i in range(2):
    
    # Split into training and validation set
    train_features, valid_features, train_y, valid_y = train_test_split(train_df, labels, test_size = 0.25, 
                                                                        random_state = i)
    
    # Train using early stopping
    model.fit(train_features, train_y, early_stopping_rounds=100, eval_set = [(valid_features, valid_y)], 
              eval_metric = 'auc', verbose = 200)
    
    # Record the feature importances
    feature_importances += model.feature_importances_
# Make sure to average feature importances! 
feature_importances = feature_importances / 2
feature_importances = pd.DataFrame({'feature': list(train_df.columns), 'importance': 
                                    feature_importances}).sort_values('importance', ascending = False)

feature_importances.head()
# Make sure to drop the ids and target
df_pca = df.copy().drop(df.select_dtypes('object').columns, axis=1).reset_index()
labels = df_pca['Target']
ids = df_pca['index']
df_pca = df_pca.drop(columns = ['index', 'Target'])

# Make a pipeline with imputation and pca
pipeline = Pipeline(steps = [('imputer', Imputer(strategy = 'median')),
             ('pca', PCA())])

# Fit and transform on the training data
df_pca = pipeline.fit_transform(df_pca)
# Extract the pca object
pca = pipeline.named_steps['pca']

# Plot the cumulative variance explained
plt.figure(figsize = (5, 4))
plt.plot(list(range(df_pca.shape[1])), np.cumsum(pca.explained_variance_ratio_), 'r-')
plt.xlabel('Number of PC'); plt.ylabel('Cumulative Variance Explained');
plt.title('Cumulative Variance Explained with PCA');
# Dataframe of pca results
df_pca_final = pd.DataFrame({'pc_1': df_pca[:, 0], 'pc_2': df_pca[:, 1], 'target': labels})

# Plot pc2 vs pc1 colored by target
sns.lmplot('pc_1', 'pc_2', data = df_pca_final, hue = 'target', fit_reg=False, size = 5)
plt.title('PC2 vs PC1 by Target')

print('2 principal components account for {:.4f}% of the variance.'.format
      (100 * np.sum(pca.explained_variance_ratio_[:2])))
