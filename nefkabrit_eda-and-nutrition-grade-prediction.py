import numpy as np , matplotlib as plt 

%pylab inline

from scipy import stats

from scipy.stats import chi2_contingency



import pandas as pd

%matplotlib inline

import seaborn as sns



from sklearn import preprocessing , decomposition , neighbors

from sklearn.cross_validation import train_test_split

from sklearn import neighbors

import random



import warnings

warnings.filterwarnings('ignore')
# Data load

RawData_Nutri = pd.read_csv('../input/en.openfoodfacts.org.products.tsv',  sep='\t')

RawData_Nutri.shape
RawData_Nutri.describe()
plt.figure(figsize=(13, 40))

plt.rcParams['axes.facecolor'] = 'black'

plt.rc('grid', color='#202020')

(RawData_Nutri.isnull().mean(axis=0)*100).plot.barh(color ="#FF6600")

plt.xlim(xmax=100)

plt.title("Missing values rate",fontsize=18)

plt.xlabel("percentage",fontsize=14)
# Keep only consistent features (less of 60% NaN values)

de = RawData_Nutri.isnull().mean(axis=0)

l = []

for i in range(0,len(de)):

    if de[i] < 0.6:

        templist = list(de[de==de[i]].index) 

        for i in range (0,len(templist)):

            l.append(templist[i])



variable_consistante = list(set(l)) 

Dnutri_NewFeat = RawData_Nutri.loc[:, lambda df: variable_consistante] 

Dnutri_NewFeat.shape
# This list contains unwanted features

NoList = ["code","url","states_en", "countries_tags","additives","brands",

          "last_modified_datetime","creator","additives_tags",

          "states","states_tags","ingredients_text","created_datetime",

          "serving_size","created_t","nutrition-score-uk_100g","countries",

          "last_modified_t","brands_tags","additives_en",

          "ingredients_that_may_be_from_palm_oil_n"]



for i in range (0,len(NoList)):

    variable_consistante.remove(NoList[i])



Dnutri_NewFeat = RawData_Nutri.loc[:, lambda df: variable_consistante]

Dnutri_NewFeat.shape
#variable_consistante
l = ["product_name","countries_en","nutrition_grade_fr","nutrition-score-fr_100g"]

featlist = list(Dnutri_NewFeat)

for i in range(0,len(l)):

    featlist.remove(l[i])



# Replace NaN value by 0 for nuremic features.

for i in range(len(featlist)):

    Dnutri_NewFeat[featlist[i]].fillna(0, inplace=True)
# Replace NaN value by Unknow for categorial features.

Dnutri_NewFeat["countries_en"].fillna("Unknow", inplace=True)

Dnutri_NewFeat["product_name"].fillna("Unknow", inplace=True)
Dnutri_Nano = Dnutri_NewFeat.dropna(axis=0, how='any') 

Dnutri_Nano = Dnutri_Nano.sort_values(by=["nutrition_grade_fr"] , ascending=[True])

Dnutri_Nano.shape
plt.figure(figsize=(13,4))

(Dnutri_Nano.notnull().mean(axis=0)*100).plot.barh(color ="#33CC66")

plt.xlim(xmax=100)

plt.title("Not null value rate (Dnutri_Nano) ")
Dnutri_score_less = Dnutri_NewFeat[pd.isnull(Dnutri_NewFeat['nutrition_grade_fr'])]

Dnutri_score_less.shape
plt.figure(figsize=(13,3))

(Dnutri_score_less.isnull().mean(axis=0)*100).plot.barh(color ="#33CCFF")

plt.xlim(xmax=100)

plt.title("Missing values rate (Dnutri_score_less)")
def boxplot_univ (feature,plotColor="#CC9900"):

    """

    Generates a boxplot from a given features and color

    """

    plt.figure(figsize=(8,3)) 

    plt.rc('grid', color='#202020') 

    plt.rc('axes', facecolor='black')

    plt.rc('text', color='black')

    sns.boxplot(data=Dnutri_Nano, y=feature, color=plotColor) 
boxplot_univ("energy_100g")

plt.ylim(0, 5000)



boxplot_univ("fat_100g","#FFCC33")

plt.ylim (0, 200)



boxplot_univ("sugars_100g","#33CCFF")

plt.ylim (-50, 150)



boxplot_univ("salt_100g","#F5F5DC")

plt.ylim (0, 10)



boxplot_univ("fiber_100g","#33CC33")

plt.ylim (0, 100)



boxplot_univ("additives_n","purple")

plt.ylim (0, 20)



boxplot_univ("proteins_100g","red")

plt.ylim (0, 100)



boxplot_univ("calcium_100g","#CCCCCC")

plt.ylim (0, 0.5)
#Outliers Treatment

Dnutri_Nano.loc[Dnutri_Nano.energy_100g > 4000, 'energy_100g'] = 4000

Dnutri_Nano.loc[Dnutri_Nano.fat_100g > 100, 'fat_100g'] = 100

Dnutri_Nano.loc[Dnutri_Nano.carbohydrates_100g > 100, 'carbohydrates_100g'] = 100

Dnutri_Nano.loc[Dnutri_Nano.sugars_100g > 100, 'sugars_100g'] = 100

Dnutri_Nano.loc[Dnutri_Nano.sugars_100g < 0, 'sugars_100g'] = 0

Dnutri_Nano.loc[Dnutri_Nano.salt_100g > 100, 'salt_100g'] = 100

Dnutri_Nano.loc[Dnutri_Nano.sodium_100g > 100, 'sodium_100g'] = 100

Dnutri_Nano.loc[Dnutri_Nano.fiber_100g >100, 'fiber_100g'] = 100

Dnutri_Nano.loc[Dnutri_Nano.proteins_100g >100, 'proteins_100g'] = 100

Dnutri_Nano.loc[Dnutri_Nano.proteins_100g < 0, 'proteins_100g'] = 0
# Delete features

Dnutri_Nano = Dnutri_Nano.drop('trans-fat_100g',1)

Dnutri_Nano = Dnutri_Nano.drop('ingredients_from_palm_oil_n',1)
nutriGrd = Dnutri_Nano['nutrition_grade_fr'].value_counts(normalize=True)

plt.figure(figsize=(6, 6))

pie(nutriGrd.values, labels=nutriGrd.index,

                autopct='%1.1f%%', shadow=True, startangle=90)

title('Nutrigrade Rate')

show()
def boxplot_multiv (feature,plotColor="#CC9900"):

    """

    Generate boxplot from nutrition_grade_fr and a given feature

    """

    plt.figure(figsize=(15, 4)) 

    plt.rc('grid', color='#202020') 

    plt.rc('axes', facecolor='black')

    plt.rc('text', color='black')

    sns.boxplot(data=Dnutri_Nano, x="nutrition_grade_fr",y=feature, color=plotColor)
# bivariate boxplot

boxplot_multiv("energy_100g")

boxplot_multiv("sugars_100g","#33CCFF")

boxplot_multiv("fat_100g","yellow")

boxplot_multiv("additives_n","purple")

plt.ylim (0,10)

boxplot_multiv("fiber_100g","green")

plt.ylim (0,15)

boxplot_multiv("salt_100g","white")

plt.ylim (0,5)

boxplot_multiv("proteins_100g","red")

plt.ylim (0,30)
corr = Dnutri_Nano.corr()

corr = corr.round(1)

plt.figure(figsize=(10, 9))

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True



sns.set(font_scale=1.2)

with sns.axes_style("white"):

    sns.heatmap(corr,  annot = True ,vmax=1, cmap="RdBu_r",square=True, mask=mask)
l = ["vitamin-c_100g","vitamin-a_100g","iron_100g","calcium_100g","cholesterol_100g",

    "salt_100g"]



for i in range(0,len(l)):

    Dnutri_Nano = Dnutri_Nano.drop(l[i],1)
usa = Dnutri_Nano[Dnutri_Nano["countries_en"]== "United States"]['nutrition_grade_fr'].value_counts()

fr = Dnutri_Nano[Dnutri_Nano["countries_en"]== "France"]['nutrition_grade_fr'].value_counts()

swi = Dnutri_Nano[Dnutri_Nano["countries_en"]== "Switzerland"]['nutrition_grade_fr'].value_counts()

germ = Dnutri_Nano[Dnutri_Nano["countries_en"]== "Germany"]['nutrition_grade_fr'].value_counts()
# Create contingence dataframe

contingence = {'USA' : pd.Series(usa, index=usa.index),

               'France' : pd.Series(fr, index=fr.index),

               'Suisse' : pd.Series(swi, index=swi.index),

               'Germany' : pd.Series(germ, index=germ.index)

              }

# Contingence table

khi_cont = pd.DataFrame(contingence)

khi_cont
khi_array = khi_cont.as_matrix(columns=None)



chi2, pvalue, degrees, expected = chi2_contingency(khi_array,correction=True)

expected
chi2, degrees, pvalue
def splitDataFrameList(df, target_column, separator):

    '''Split rows with several countries

    '''

    def splitListToRows(row, row_accumulator, target_column, separator):

        split_row = row[target_column].split(separator)

        for s in split_row:

            new_row = row.to_dict()

            new_row[target_column] = s

            row_accumulator.append(new_row)

    new_rows = []

    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))

    new_df = pd.DataFrame(new_rows)

    return new_df



Dnutri_Nano = splitDataFrameList(Dnutri_Nano, "countries_en", ",")
index_list = Dnutri_Nano["countries_en"].value_counts().index

index_list = index_list.drop(['United States', 'France', 'Switzerland', 

                              'Germany'])

for i in index_list:

    Dnutri_Nano["countries_en"].replace({i : "Other_Pays" }, inplace=True)
#One hot Encoding

Dnutri_Nano = pd.get_dummies(Dnutri_Nano, columns=["countries_en"], prefix=["From"])
badFood =  ((Dnutri_Nano["sugars_100g"]) + (Dnutri_Nano["sodium_100g"]*5) + (Dnutri_Nano["saturated-fat_100g"])) / (Dnutri_Nano["fiber_100g"]+0.1)

goodFood =  (Dnutri_Nano["fiber_100g"])/(Dnutri_Nano["saturated-fat_100g"]+0.1)



Dnutri_Nano["badFood"]= badFood

Dnutri_Nano["goodFood"]= goodFood



for i in ["badFood","goodFood"]:

    if i == "badFood":

        colordist = "red"

    else:

        colordist = "green"

    plt.rcParams['axes.facecolor'] = 'black'

    plt.rc('grid', color='#202020')

    sns.distplot(Dnutri_Nano[i], kde=True, color=colordist)

    plt.show()
sns.set(font_scale=4)

flatui = ["green", "#99FF33", "#FFFF33", "#FF6600", "#FF0000"]

plt.rcParams['axes.facecolor'] = 'black'

plt.rc('grid', color='#202020')

sns.pairplot(Dnutri_Nano[["goodFood","badFood","nutrition-score-fr_100g","nutrition_grade_fr"]],

             hue="nutrition_grade_fr", diag_kind="kde",size =12, palette=flatui)
corr = Dnutri_Nano[["goodFood","badFood","nutrition-score-fr_100g"]].corr()

corr = corr.round(1)

plt.figure(figsize=(7, 6))



sns.set(font_scale=1.3)

with sns.axes_style("white"):

    sns.heatmap(corr,  annot = True ,vmax=1, cmap="BrBG",square=True)
# little preprocessing

headers = list(Dnutri_Nano)

index = Dnutri_Nano["product_name"]

nutrigrade = Dnutri_Nano["nutrition_grade_fr"]

D = Dnutri_Nano.drop("nutrition_grade_fr",1)

D = D.drop("product_name",1)

D = D.drop("nutrition-score-fr_100g",1)
# Data Standardizing

std_scale = preprocessing.StandardScaler().fit(D)

nutri_scaled = std_scale.transform(D)
# Run PCA

pca = decomposition.PCA()

pca.fit(nutri_scaled)

print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())
plt.figure(figsize=(12, 7))

sns.set(font_scale=2)

plt.rcParams['axes.facecolor'] = 'black'

plt.rc('grid', color='#202020')



plt.step(range(16), pca.explained_variance_ratio_.cumsum(), where='mid',color="#66FFFF")

sns.barplot(np.arange(1,17),pca.explained_variance_ratio_,palette="PuBuGn_d")
pca2 = decomposition.PCA(n_components=8)

pca2.fit(nutri_scaled)
pcs = pca2.components_

def PCA_plot (components,comp1,comp2):

    plt.figure(figsize=(12, 12))

    for i, (x, y) in enumerate(zip(components[comp1, :], components[comp2, :])):

        # Display origine segment (x, y)

        plt.plot([0, x], [0, y], color='#00FFFF')

        

        plt.text(x, y, D.columns[i], fontsize='12', color='#FFFF99')



    plt.plot([-1, 1], [0, 0], color='grey', ls='--')

    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    plt.xlim([-1, 1])

    plt.ylim([-1, 1])



# To assign each individual a color corresponding to their nutrition grade 

# This will be useful when vizualing

conv = Dnutri_Nano["nutrition_grade_fr"].replace('a', "green")

conv = conv.replace('b', "#99FF33")

conv = conv.replace('c', "#FFFF33")

conv = conv.replace('d', "#FF6600")

conv = conv.replace('e', "#FF0000")

Dnutri_Nano['nutrigrade_num'] = pd.Series(conv, index=Dnutri_Nano.index)



def scatterP_c (x,y):

    plt.figure(figsize=(12, 12))

    X_projected = pca2.transform(nutri_scaled)



    plt.scatter(X_projected[:, x], X_projected[:, y],

    c=Dnutri_Nano.get('nutrigrade_num'))

    plt.xlim([-15, 40])

    plt.ylim([-25, 40])

    plt.rcParams['axes.facecolor'] = 'k'

    plt.rc('grid', color='#202020')
scatterP_c(0,1)

PCA_plot(pcs,0,1)
D = D.drop("From_Germany",1)

D = D.drop("From_Switzerland",1)

D = D.drop("From_Other_Pays",1)
#Sampling

NutriSpl = (random.sample(list(D.index),150000))



data_entry = D.loc[NutriSpl]

#data_entry = data_entry.drop("nutrigrade_num",1)

data_target = nutrigrade.loc[NutriSpl]
print(data_entry.shape)

print(data_target.shape)
#Testing set / Trainning set

Xtrain, Xtest, ytrain, ytest = train_test_split(data_entry, data_target, train_size=0.8,random_state=1)

# Training with KNN

# Optimisation du score

error_list = []

for k in range(2,15):

    knn = neighbors.KNeighborsClassifier(k) # training !

    error_list.append(100*(1-knn.fit(Xtrain, ytrain).score(Xtest,ytest))) # compute error on testing set

    

# Display KNN performence in term of K

plt.figure(figsize=(15, 15))

plt.plot(range(2,15), error_list,'go-', markersize =8)

plt.ylabel('error (%)')

plt.xlabel('k')

plt.show()
NutriSpl = (random.sample(list(D.index),230000))



data_entry = D.loc[NutriSpl]

data_target = nutrigrade.loc[NutriSpl]



Xtrain, Xtest, ytrain, ytest = train_test_split(data_entry, data_target, train_size=0.8,random_state=1)



# Training  KNN

knn = neighbors.KNeighborsClassifier(n_neighbors=3)

knn.fit(Xtrain, ytrain) 

error = (1 - knn.score(Xtest, ytest))*100  # compute error on testing set

error
# Apply the same treatment on our testing set.

Dnutri_score_less = splitDataFrameList(Dnutri_score_less, "countries_en", ",")



index_list = Dnutri_score_less["countries_en"].value_counts().index

index_list = index_list.drop(['United States','France', 'Switzerland','Germany'])

for i in index_list:

    Dnutri_score_less["countries_en"].replace({i : "Other_Pays" }, inplace=True)

Dnutri_score_less = pd.get_dummies(Dnutri_score_less, columns=["countries_en"], prefix=["From"])



Dnutri_score_less = Dnutri_score_less.drop("vitamin-c_100g",1)

Dnutri_score_less = Dnutri_score_less.drop("vitamin-a_100g",1)

Dnutri_score_less = Dnutri_score_less.drop("iron_100g",1)

Dnutri_score_less = Dnutri_score_less.drop("calcium_100g",1)

Dnutri_score_less = Dnutri_score_less.drop("cholesterol_100g",1)

Dnutri_score_less = Dnutri_score_less.drop("salt_100g",1)

Dnutri_score_less = Dnutri_score_less.drop("ingredients_from_palm_oil_n",1)

Dnutri_score_less = Dnutri_score_less.drop("product_name",1)

Dnutri_score_less = Dnutri_score_less.drop("nutrition-score-fr_100g",1)

Dnutri_score_less = Dnutri_score_less.drop("nutrition_grade_fr",1)

Dnutri_score_less = Dnutri_score_less.drop("trans-fat_100g",1)

Dnutri_score_less = Dnutri_score_less.drop("From_Germany",1)

Dnutri_score_less = Dnutri_score_less.drop("From_Switzerland",1)

Dnutri_score_less = Dnutri_score_less.drop("From_Other_Pays",1)



badFood =  ((Dnutri_score_less["sugars_100g"]) + (Dnutri_score_less["sodium_100g"]*5) + (Dnutri_score_less["saturated-fat_100g"])) / (Dnutri_score_less["fiber_100g"]+0.1) 

goodFood =  (Dnutri_score_less["fiber_100g"])/(Dnutri_score_less["saturated-fat_100g"]+0.1)



Dnutri_score_less["badFood"]= badFood

Dnutri_score_less["goodFood"]= goodFood



dd = Dnutri_score_less[D.columns.tolist()]
# Prediction

knn.predict(dd)
Dnutri_score_less["nutrition_grade_fr"] = knn.predict(dd)

Dnutri_score_less.head(5)