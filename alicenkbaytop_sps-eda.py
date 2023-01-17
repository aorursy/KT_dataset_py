import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns # visualization

import matplotlib.pyplot as plt # visualization

import missingno as msno # visualizatin for missing values



import warnings

warnings.filterwarnings("ignore") # ignore warnings



from sklearn.model_selection import train_test_split # train and test split



from sklearn.impute import KNNImputer # filling missing data with KNN method



from sklearn.preprocessing import LabelEncoder # filling missing categorical values with label encoder method

import category_encoders as ce



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load file 

df = pd.read_csv("/kaggle/input/protein-data-set/pdb_data_no_dups.csv")
df.head(3)
df.info()
df.describe().T
df.publicationYear[df.publicationYear==201.0]=np.nan
df.phValue[df.phValue>14]=np.nan
df.drop("pdbxDetails", axis=1, inplace = True)
df.classification.nunique()
df.experimentalTechnique.nunique()
df.structureId.nunique()
df.macromoleculeType.nunique()
df.crystallizationMethod.nunique()
df.structureId.value_counts(ascending=False).head(10)
# Missing Value Table

def missing_value_table(df):

    missing_value = df.isna().sum().sort_values(ascending=False)

    missing_value_percent = 100 * df.isna().sum()//len(df)

    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)

    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})

    cm = sns.light_palette("lightgreen", as_cmap=True)

    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)

    return missing_value_table_return

  

missing_value_table(df)
sns.pairplot(df)

sns.set(style="ticks", color_codes=True)
corr = df.corr()

plt.figure(figsize=(12,5))

sns.heatmap(corr, annot=True)
sns.lmplot(x = "densityPercentSol", y = "densityMatthews", line_kws={'color': 'red'}, data = df);

plt.title("densityMatthews-densityPercentSol",color = 'darkblue',fontsize=15)

plt.show()
density_data = df[['densityMatthews','densityPercentSol']] 

sns.pairplot(density_data);

plt.title("densityMatthews-densityPercentSol",color = 'darkblue',fontsize=15)

plt.show()
plt.figure(figsize=(20,18))

ex = df.macromoleculeType.value_counts(ascending=False)[:5]

figureObject, axesObject = plt.subplots() 

explode = (0.2, 0.5, 0.5, 0.5, 0.5)

plt.title("Macro Molecule Type",color = 'darkblue',fontsize=15)



axesObject.pie(ex.values,

               labels   = ex.index,

               shadow   = True,                       

               explode  = explode,

               autopct  = '%.1f%%',

               wedgeprops = { 'linewidth' : 3,'edgecolor' : "orange" })                              

             

axesObject.axis('equal') 



plt.show() 
experimentalTechnique=df["experimentalTechnique"].value_counts(ascending=False)[:5]

plt.figure(figsize=[10,5])

plt.plot(experimentalTechnique, color="#588da8", linestyle="--", linewidth=3, label = "experimentalTechnique")

plt.title("Experimental Technique-Frequency",color = 'darkblue',fontsize=15)

plt.xlabel("Experimental Technique")

plt.ylabel("Frequency")

plt.show()
classification = df.classification.value_counts()[:10]

plt.figure(figsize=(12,5))

sns.barplot(x=classification.index, y=classification.values, palette="dark")

plt.xticks(rotation='vertical')

plt.ylabel('Number of Classification')

plt.xlabel('Classification Types')

plt.title('Top 10 Classification',color = 'darkblue',fontsize=15);
def ph(ph):

    if ph < 7 :

        ph = 'Acidic'

    elif ph > 7:

        ph = 'Base'

    else:

        ph = 'Neutral'

    return ph
df_ph = df.dropna(subset=["phValue"])

df_ph['pH'] = df_ph['phValue'].apply(ph)

labels = df_ph['pH'].value_counts().index

values = df_ph['pH'].value_counts().values
fig1, ax1 = plt.subplots()

ax1.pie(values, labels=labels, autopct='%1.1f%%')

ax1.axis('equal')

plt.title("Acid-Base-Neutral Balance",color = 'darkblue',fontsize=15)

plt.show()
plt.figure(figsize=(12,5))

sns.scatterplot(x=df.publicationYear.value_counts().sort_index().index, y=df.publicationYear.value_counts().sort_index().values)

plt.xticks(rotation='vertical')

plt.ylabel('Frequency')

plt.xlabel('Years')

plt.title('Publication Distribution by Years',color = 'darkblue',fontsize=15);
y = df[["classification"]] # target

x = df.drop("classification", axis=1)



x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42, test_size=0.2)
columns = x_train.select_dtypes(["int","float64","int64"])



del columns["phValue"] # delete unneccessary value

del columns["publicationYear"] # delete unneccessary value
lower_and_upper = {} # storage

x_train_copy = x_train.copy() # train copy 



for col in columns.columns: # outlier detect

    q1 = x_train[col].describe()[4] # Q1 = Quartile 1 median 25 

    q3 = x_train[col].describe()[6] # Q3 = Quartile 3 median 75 

    iqr = q3-q1  #IQR Q3 -Q1

    

    lower_bound = q1-(1.5*iqr)

    upper_bound = q3+(1.5*iqr)

    

    lower_and_upper[col] = (lower_bound, upper_bound)

    x_train_copy.loc[(x_train_copy.loc[:,col]<lower_bound),col]=lower_bound*0.75

    x_train_copy.loc[(x_train_copy.loc[:,col]>upper_bound),col]=upper_bound*1.25

    

lower_and_upper
x_test_copy = x_test.copy() # train copy   



for col in columns.columns:

    x_test_copy.loc[(x_test_copy.loc[:,col]<lower_and_upper[col][0]),col]=lower_and_upper[col][0]*0.75

    x_test_copy.loc[(x_test_copy.loc[:,col]>lower_and_upper[col][1]),col]=lower_and_upper[col][1]*1.25
msno.bar(df, figsize=(15,8), sort='descending');
msno.matrix(df)

plt.title("Missing Value",color = 'darkblue',fontsize=15)

plt.show()
msno.heatmap(df)

plt.title("Missing Value Correlation HeatMap",color = 'darkblue',fontsize=15)

plt.show()
x_train_copy['macromoleculeType'].fillna(x_train_copy['macromoleculeType'].mode()[0], inplace=True) # fill missing data with mode
x_test_copy['macromoleculeType'].fillna(x_test_copy['macromoleculeType'].mode()[0], inplace=True) # fill missing data with mode
x_train_resol_std, x_train_resol_mean = x_train_copy.resolution.std(), x_train_copy.resolution.mean() # mean and standard deviation

random = np.random.uniform(x_train_resol_std, x_train_resol_mean, 113120) # 113120 numbers of rows 

x_train.resolution = x_train.resolution.mask(x_train.resolution.isnull(), random)
x_test_resol_std, x_test_resol_mean = x_test_copy.resolution.std(), x_test_copy.resolution.mean() # mean and standard deviation

random = np.random.uniform(x_test_resol_std, x_test_resol_mean, 28281) # 28281 numbers of rows 

x_test.resolution = x_test.resolution.mask(x_test.resolution.isnull(), random)
imputer = KNNImputer(n_neighbors=5)

imputer.fit_transform([x_train.crystallizationTempK])
imputer = KNNImputer(n_neighbors=5)

imputer.fit_transform([x_train.densityMatthews])
imputer = KNNImputer(n_neighbors=5)

imputer.fit_transform([x_train.densityPercentSol])
imputer = KNNImputer(n_neighbors=5)

imputer.fit_transform([x_train.phValue])
imputer = KNNImputer(n_neighbors=5)

imputer.fit_transform([x_train.resolution])
imputer = KNNImputer(n_neighbors=5)

imputer.fit_transform([x_train.publicationYear])
imputer = KNNImputer(n_neighbors=5)

imputer.fit_transform([x_test.crystallizationTempK])
imputer = KNNImputer(n_neighbors=5)

imputer.fit_transform([x_test.densityMatthews])
imputer = KNNImputer(n_neighbors=5)

imputer.fit_transform([x_test.densityPercentSol])
imputer = KNNImputer(n_neighbors=5)

imputer.fit_transform([x_test.phValue])
imputer = KNNImputer(n_neighbors=5)

imputer.fit_transform([x_test.resolution])
imputer = KNNImputer(n_neighbors=5)

imputer.fit_transform([x_test.publicationYear])