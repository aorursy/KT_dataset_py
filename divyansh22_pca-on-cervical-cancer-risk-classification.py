### Primary necessary imports and reading the data.



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

df_raw = pd.read_csv("../input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv")
df_raw.info() ### get info about the data
df_nan = df_raw.replace('?', np.nan) 
df_nan.isnull().sum() ### check for null values in every column.
df = df_nan #temporary save
df1 = df.apply(pd.to_numeric)
df1.info()
####filling NaN values with median for continous variables and 0/1 for discrete variables.



df['Number of sexual partners'] = df['Number of sexual partners'].fillna(df['Number of sexual partners'].median())

df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].median())

df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())

df['Smokes'] = df['Smokes'].fillna(1)

df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())

df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].median())

df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)

df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].median())

df['IUD'] = df['IUD'].fillna(0)

df['IUD (years)'] = df['IUD (years)'].fillna(0)

df['STDs'] = df['STDs'].fillna(1)

df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())

df['STDs:condylomatosis'] = df['STDs:condylomatosis'].fillna(df['STDs:condylomatosis'].median())

df['STDs:cervical condylomatosis'] = df['STDs:cervical condylomatosis'].fillna(df['STDs:cervical condylomatosis'].median())

df['STDs:vaginal condylomatosis'] = df['STDs:vaginal condylomatosis'].fillna(df['STDs:vaginal condylomatosis'].median())

df['STDs:vulvo-perineal condylomatosis'] = df['STDs:vulvo-perineal condylomatosis'].fillna(df['STDs:vulvo-perineal condylomatosis'].median())

df['STDs:syphilis'] = df['STDs:syphilis'].fillna(df['STDs:syphilis'].median())

df['STDs:pelvic inflammatory disease'] = df['STDs:pelvic inflammatory disease'].fillna(df['STDs:pelvic inflammatory disease'].median())

df['STDs:genital herpes'] = df['STDs:genital herpes'].fillna(df['STDs:genital herpes'].median())

df['STDs:molluscum contagiosum'] = df['STDs:molluscum contagiosum'].fillna(df['STDs:molluscum contagiosum'].median())

df['STDs:AIDS'] = df['STDs:AIDS'].fillna(df['STDs:AIDS'].median())

df['STDs:HIV'] = df['STDs:HIV'].fillna(df['STDs:HIV'].median())

df['STDs:Hepatitis B'] = df['STDs:Hepatitis B'].fillna(df['STDs:Hepatitis B'].median())

df['STDs:HPV'] = df['STDs:HPV'].fillna(df['STDs:HPV'].median())

df['STDs: Time since first diagnosis'] = df['STDs: Time since first diagnosis'].fillna(df['STDs: Time since first diagnosis'].median())

df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(df['STDs: Time since last diagnosis'].median())
####filling NaN values with dummy values for categorical variables.



df = pd.get_dummies(data=df, columns=['Smokes','Hormonal Contraceptives','IUD','STDs',

                                      'Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Citology','Schiller'])
df.isnull().sum()
df_final = df #temporary save
df.describe()
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



X = df.drop('Biopsy', axis=1)

y = df["Biopsy"]



X = StandardScaler().fit_transform(X)  # Standardizing the values in X.



pca = PCA(0.80)  # Changes in variance percentage can be made here.

prin_comp = pca.fit_transform(X)

principalDf = pd.DataFrame(data = prin_comp)



print(principalDf)



print('\nEigenvalues \n%s' %pca.explained_variance_)

print('Eigenvectors \n%s' %pca.components_)
def scree_plot():

    from matplotlib.pyplot import figure, show

    from matplotlib.ticker import MaxNLocator



    ax = figure().gca()

    ax.plot(pca.explained_variance_)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel('Principal Component')

    plt.ylabel('Eigenvalue')

    plt.axhline(y=1, linewidth=1, color='r', alpha=0.5)

    plt.title('Scree Plot of PCA: Component Eigenvalues')

    show()



scree_plot()
finalDf = pd.concat([principalDf, df[["Biopsy"]]], axis = 1)

finalDf
pca2 = PCA(n_components = 2)  # Changes can be made here.

prin_comp2 = pca2.fit_transform(X)

principalDf = pd.DataFrame(data = prin_comp2

             , columns = ['PC 1', 'PC 2'])



finalDf2 = pd.concat([principalDf2, df[['Biopsy']]], axis = 1)



import matplotlib.pyplot as plt



fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

targets = [0, 1]

colors = ['r', 'g']

for target, color in zip(targets,colors):

    indicesToKeep = finalDf2['Biopsy'] == target

    ax.scatter(finalDf2.loc[indicesToKeep, 'PC 1']

               , finalDf2.loc[indicesToKeep, 'PC 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()