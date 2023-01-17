import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
alcdata = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/alcoholism/student-mat.csv")
fifadata = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/fifa18/data.csv")
accidata1 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2005_to_2007.csv")
accidata2 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2009_to_2011.csv")
accidata3 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2012_to_2014.csv")
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
alcdata
#taking average of all the three grades of g1, g2, g3 as g_avg
alcdata['g_avg'] = alcdata['G1']/3 + alcdata['G2']/3 + alcdata['G3']/3
alcdata
del alcdata['G1']
del alcdata['G2']
del alcdata['G3']
x = alcdata_new.iloc[:, :-1] ## independent features
y = alcdata_new.iloc[:,-1] ## dependent features
alcdata.corr()
#only numerical data are taken into consider
corrmat = alcdata.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(alcdata[top_corr_features].corr(), annot = True, cmap = "RdYlGn")
from sklearn.ensemble import ExtraTreesRegressor #solves outlier problem
model  = ExtraTreesRegressor()
model.fit(x,y)
model.feature_importances_
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index = x.columns)
feat_importances.nlargest(7).plot(kind = 'barh') ## top 7 features
plt.show()
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
alcdata.info()
alcdata_new = alcdata.copy()
for feature in alcdata_new.dtypes[alcdata_new.dtypes == 'object'].index:
    alcdata_new[feature] = alcdata_new[feature].astype('category')
del alcdata_new['job_type']
alcdata_new.info()
from sklearn.preprocessing import LabelEncoder
for feature in alcdata_new.dtypes[alcdata_new.dtypes == 'category'].index:
    le = LabelEncoder()
    alcdata_new[feature] = le.fit_transform(alcdata_new[feature]) #encoding data for model training
alcdata_new.info()
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
alcdata_new[['famrel', 'Pstatus']]
x_new = alcdata_new[['famrel', 'Pstatus', 'g_avg']]
sns.pairplot(x_new)
x_new.corr()
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
sns.pairplot(alcdata)
alc2 = alcdata.copy()
for feature in alc2.dtypes[alc2.dtypes == 'object'].index:
    del alc2[feature]
for feature in alc2.dtypes[alc2.dtypes == 'int64'].index:
    alc2[feature] = alc2[feature].astype(float)
alc2.info()
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
alc = scale.fit_transform(alc2)
j = 0
for feature in alc2.dtypes[alc2.dtypes != 'object'].index:
    for i in range(0,395):
        alc2[feature][i] = alc[i][j]
    j = j+1
alc2
sns.pairplot(alc2)
alc2.hist(bins=50, figsize=(20,15))
plt.show()
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
fifadata
del fifadata['Unnamed: 0']
#top 10 most economical club
fifadata.sort_values(by = ['Release Clause']).head(10)
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
x = fifadata[['Age', 'Potential']]
x.corr()
sns.pairplot(x) # no strong correlation
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
fifadata.info()
fif = fifadata.copy()
## delete unrequired
del fif['ID']
del fif['Name']
del fif['Nationality']
del fif['Photo']
del fif['Flag']
del fif['Club Logo']
del fif['Loaned From']
fif = fif.dropna()
fif.info()
for feature in fif.dtypes[fif.dtypes == 'object'].index:
    fif[feature] = fif[feature].astype('category')
from sklearn.preprocessing import LabelEncoder
for feature in fif.dtypes[fif.dtypes == 'category'].index:
    le = LabelEncoder()
    fif[feature] = le.fit_transform(fif[feature]) #encoding data for model training
fif2 = fif.copy()
x = fif['Potential']
del fif['Potential']
from sklearn.ensemble import ExtraTreesRegressor #solves outlier problem
model  = ExtraTreesRegressor()
model.fit(fif,x)
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index = fif.columns)
feat_importances.nlargest(7).plot(kind = 'barh') ## top 7 features
plt.show()
#x = fif2['Wage']
#del fif2['Wage']
from sklearn.ensemble import ExtraTreesRegressor #solves outlier problem
model  = ExtraTreesRegressor()
model.fit(fif2,x)
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index = fif2.columns)
feat_importances.nlargest(7).plot(kind = 'barh') ## top 7 features
plt.show()
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
sns.countplot(y='Age', data= fif, order = fif['Age'].value_counts().index)
plt.show()
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
frames = [accidata1, accidata2, accidata3]
df = pd.concat(frames)
df
#enter code/answer in this cell. You can add more code/markdown cells below for your answer.
df.sort_values(by = ['Number_of_Casualties'], ascending = False)
#enter code/answer in this cell. You can add more code/markdown cells below for your answer.
df.info()
print(df['Speed_limit'].max())
print(df['Speed_limit'].min())
del df['Junction_Control']
del df['Junction_Detail']
del df['LSOA_of_Accident_Location']
df.dropna()
df = df.dropna()
df.info()
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
df['Light_Conditions']
for feature in df.dtypes[df.dtypes == 'object'].index:
    df[feature] = df[feature].astype('category')
#from sklearn.preprocessing import LabelEncoder
for feature in df.dtypes[df.dtypes == 'category'].index:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature]) #encoding data for model training
x = df[['Light_Conditions', 'Weather_Conditions', 'Accident_Severity']]
sns.pairplot(x)
df['Accident_Severity'] = y
df
corr_matrix = df.corr()
corr_matrix['Accident_Severity'].sort_values(ascending=False)
corr_matrix['Accident_Severity']
df2 = df.copy()
#keeping only significant features having more than 3% positive or negative correlation
for feature in df.dtypes[df.dtypes != 'object'].index:
    if(corr_matrix['Accident_Severity'][feature] < 0.03 and corr_matrix['Accident_Severity'][feature] > -0.03):
        del df2[feature]
df2
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
x = df2.iloc[:, :-1] ## independent features
y = df2.iloc[:,-1] ## dependent features
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
from sklearn.model_selection import cross_val_score
score = cross_val_score(log_reg, x, y, cv = 5)
score.mean()
