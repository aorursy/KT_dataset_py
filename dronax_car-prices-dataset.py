import warnings

warnings.filterwarnings('ignore')



#importing the libraries

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

import seaborn as sns



#pour afficher tous les colonnes d'un tableau

pd.set_option('display.max_columns', None)
data = pd.read_csv('../input/CarPrice_Assignment.csv')

print("Dimension of our data set is: ")

print(data.shape)

data.head()
data.info()
#Chaque élément du colonne CarName sera diviser en deux String, et on va garder seulement le premier

CompanyName = data['CarName'].apply(lambda x : x.split(' ')[0])



#Insérer la nouvelle variable comme colonne dans notre dataset

data.insert(3,"CompanyName",CompanyName)



#Supprimer la colonne CarModel

data.drop(['CarName'],axis=1,inplace=True)



#Supprimer la colonne CarID, car elle n'a aucune effet sur notre dataset

data.drop(['car_ID'],axis=1,inplace=True)



data.head()
def get_variable_type(element) :

    """

     Vérifier que les colonnes sont de variable continue ou catégorique.

     L'hypothèse est que si:

                  nombre unique <20 alors on suppose c'est catégorique

                  nombre unique> = 20 et dtype = [int64 ou float64] alors on suppose c'est continu

     """

    if element==0:

        return "Not Known"

    elif element < 20 and element!=0 :

        return "Categorical"

    elif element >= 20 and element!=0 :

        return "Contineous"

    

def predict_variable_type(metadata_matrix):

    metadata_matrix["Variable_Type"] = metadata_matrix["Valeurs_Uniques_Count"].apply(get_variable_type).astype(str)

    metadata_matrix["frequency"] = metadata_matrix["Null_Count"] - metadata_matrix["Null_Count"]

    metadata_matrix["frequency"].astype(int)

    return metadata_matrix 



def get_meta_data(dataframe) :

    """

     Méthode pour obtenir des métadonnées sur n'importe quel dataset transmis

    """

    metadata_matrix = pd.DataFrame({

                    'Datatype' : dataframe.dtypes.astype(str), # types de données de colonnes

                    'Non_Null_Count': dataframe.count(axis = 0).astype(int), # nombre total d'éléments dans les colonnes

                    'Null_Count': dataframe.isnull().sum().astype(int), # total des valeurs nulles dans les colonnes

                    'Null_Percentage': dataframe.isnull().sum()/len(dataframe) * 100, # pourcentage de valeurs nulles

                    'Valeurs_Uniques_Count': dataframe.nunique().astype(int) # nombre de valeurs uniques

                     })

    

    metadata_matrix = predict_variable_type(metadata_matrix)

    return metadata_matrix



def list_potential_categorical_type(dataframe,data) :

    print("*********colonnes de type de données catégoriques potentielles*********")

    metadata_matrix_categorical = dataframe[dataframe["Variable_Type"] == "Categorical"]

    

    length = len(metadata_matrix_categorical)

    if length == 0 :

        print("Aucune colonne catégorique dans un jeu de données donné.")  

    else :    

        metadata_matrix_categorical = metadata_matrix_categorical.filter(["Datatype","Valeurs_Uniques_Count"])

        metadata_matrix_categorical.sort_values(["Valeurs_Uniques_Count"], axis=0,ascending=False, inplace=True)

        col_to_check = metadata_matrix_categorical.index.tolist()

        name_list = []

        values_list = []

        

        for name in col_to_check :

            name_list.append(name)

            values_list.append(data[name].unique())

        

        temp = pd.DataFrame({"index":name_list,"Valeurs_Uniques":values_list})

        metadata_matrix_categorical = metadata_matrix_categorical.reset_index()

        metadata_matrix_categorical = pd.merge(metadata_matrix_categorical,temp,how='inner',on='index')

        display(metadata_matrix_categorical.set_index("index"))
metadata = get_meta_data(data)



#List potential columns of categorical variables

list_potential_categorical_type(metadata,data)
data.CompanyName.unique()
data = data.replace(to_replace ="maxda", value ="mazda") 

data = data.replace(to_replace ="porcshce", value ="porsche") 

data = data.replace(to_replace ="toyouta", value ="toyota") 

data = data.replace(to_replace ="vokswagen", value ="volkswagen") 

data = data.replace(to_replace ="vw", value ="volkswagen")

data = data.replace(to_replace ="Nissan", value ="nissan")
data.CompanyName.unique()
plt.title('Car Price Spread')

sns.boxplot(y=data.price)

plt.show()

print(data.price.describe())
plt.title('Car Price Distribution Plot')

sns.distplot(data.price)

plt.show()
print(data.price.describe())
import scipy

from scipy.stats.stats import pearsonr



def pairplot(x_axis,y_axis) :

    sns.pairplot(data,x_vars=x_axis,y_vars=y_axis,height=4,aspect=1,kind="scatter")

    plt.show()
#Determiner la variable indépendante

y_vars=['price']
x_vars=['wheelbase','curbweight','boreratio']

pairplot(x_vars,y_vars)

print("At first glance, the 3 variables are positively correlated but spread at higher values.")



p1=data['wheelbase']

p2=data['curbweight']

p3=data['boreratio']



pearson_coeff, p_value = pearsonr(p1,data['price'])

print('\nWe can make sure of this by looking at the Coefficient of Correlation')

print('\nCoefficient of Correlation between Price and wheelbase:',pearson_coeff*100,'%')



pearson_coeff, p_value = pearsonr(p2,data['price'])

print('Correlation coefficient between Price and curbweight:',pearson_coeff*100,'%')



pearson_coeff, p_value = pearsonr(p3,data['price'])

print('Correlation coefficient between Price and boreratio: ',pearson_coeff*100,'%')
x_vars=['carlength','carwidth', 'carheight']

pairplot(x_vars,y_vars)

print("Carlength and Carwidth are more correlated than carheight which is more spread out but positive.")



p1=data['carlength']

p2=data['carwidth']

p3=data['carheight']



pearson_coeff, p_value = pearsonr(p1,data['price'])

print('\nWe can make sure of this by looking at the Coefficient of Correlation')

print('\nCorrelation coefficient between Price and carlength:',pearson_coeff*100,'%')



pearson_coeff, p_value = pearsonr(p2,data['price'])

print('Correlation coefficient between Price and carwidth: ',pearson_coeff*100,'%')



pearson_coeff, p_value = pearsonr(p3,data['price'])

print('Correlation coefficient between Price and carheight: ',pearson_coeff*100,'%')
x_vars=['enginesize','horsepower','stroke']

pairplot(x_vars,y_vars)

print("Enginesize and Horsepower are positively correlated, but Stroke is more spread out (may not be related).")



p1=data['enginesize']

p2=data['horsepower']

p3=data['stroke']



pearson_coeff, p_value = pearsonr(p1,data['price'])

print('\nWe can make sure of this by looking at the Coefficient of Correlation')

print('\nCorrelation coefficient between Price and enginesize: ',pearson_coeff*100,'%')



pearson_coeff, p_value = pearsonr(p2,data['price'])

print('Correlation coefficient between Price and horsepower: ',pearson_coeff*100,'%')



pearson_coeff, p_value = pearsonr(p3,data['price'])

print('Correlation coefficient between Price and stroke: ',pearson_coeff*100,'%')
x_vars=['compressionratio','peakrpm',"symboling"]

pairplot(x_vars,y_vars)

print("Compressionratio, Peakrpm and symboling are not correlated.")



p1=data['compressionratio']

p2=data['peakrpm']

p3=data['symboling']



pearson_coeff, p_value = pearsonr(p1,data['price'])

print('\nWe can make sure of this by looking at the Coefficient of Correlation')

print('\nCorrelation coefficient between Price and compressionratio: ',pearson_coeff*100,'%')



pearson_coeff, p_value = pearsonr(p2,data['price'])

print('Correlation coefficient between Price and peakrpm: ',pearson_coeff*100,'%')



pearson_coeff, p_value = pearsonr(p3,data['price'])

print('Correlation coefficient between Price and symboling: ',pearson_coeff*100,'%')
x_vars=['citympg', 'highwaympg']

pairplot(x_vars,y_vars)

print('Citympg & Highwaympg are negatively correlated.\nThe more prices get lower, the higher the distances get, which means that the cheapest cars have better mileage than expensive cars.')



p1=data['citympg']

p2=data['highwaympg']



pearson_coeff, p_value = pearsonr(p1,data['price'])

print('\nWe can make sure of this by looking at the Coefficient of Correlation')

print('\nCorrelation coefficient between Price and citympg: ',pearson_coeff*100,'%')



pearson_coeff, p_value = pearsonr(p2,data['price'])

print('Correlation coefficient between Price and highwaympg: ',pearson_coeff*100,'%')
def heatmap(x,y,dataframe):

    sns.heatmap(dataframe.corr(),cmap="OrRd",annot=True)

    plt.show()
heatmap(20,12,data)
dimension_col_list = ['wheelbase', 'carlength', 'carwidth','curbweight']



heatmap(10,10,data.filter(dimension_col_list))
performance_col_list = ['enginesize','boreratio','horsepower']

heatmap(10,10,data.filter(performance_col_list))
performance_col_list = ['citympg','highwaympg']

heatmap(10,10,data.filter(performance_col_list))
plt.figure(figsize=(20,9))



plt.xticks(rotation = 90)

order = data['CompanyName'].value_counts(ascending=False).index

sns.countplot(x='CompanyName', data=data, order=order)



plt.show()
plt.figure(figsize=(20, 12))



plt.subplot(2,3,1)

sns.boxplot(x = 'fueltype', y = 'price', data = data)



plt.subplot(2,3,2)

plt.title('Fuel Type Histogram')

order = data['fueltype'].value_counts(ascending=False).index

sns.countplot(x='fueltype', data=data, order=order)



plt.show()
plt.figure(figsize=(20, 12))



plt.subplot(2,3,1)

sns.boxplot(x = 'aspiration', y = 'price', data = data)



plt.subplot(2,3,2)

plt.title('Aspiration Histogram')

order = data['aspiration'].value_counts(ascending=False).index

sns.countplot(x='aspiration', data=data, order=order)



plt.show()
plt.figure(figsize=(20, 12))



plt.subplot(2,3,1)

sns.boxplot(x = 'doornumber', y = 'price', data = data)



plt.subplot(2,3,2)

plt.title('Door Number Histogram')

order = data['doornumber'].value_counts(ascending=False).index

sns.countplot(x='doornumber', data=data, order=order)



plt.show()
plt.figure(figsize=(20, 12))



plt.subplot(2,3,1)

sns.boxplot(x = 'enginelocation', y = 'price', data = data)



plt.subplot(2,3,2)

plt.title('Engine Location Histogram')

order = data['enginelocation'].value_counts(ascending=False).index

sns.countplot(x='enginelocation', data=data, order=order)



plt.show()
plt.subplot(2,3,1)

sns.boxplot(x='carbody',y='price',data = data)



plt.subplot(2,3,2)

plt.title('Car Body Histogram')

order = data['carbody'].value_counts(ascending=False).index

sns.countplot(x='carbody', data=data, order=order)



plt.show()
plt.subplot(2,3,1)

sns.boxplot(x='fuelsystem',y='price',data = data)



plt.subplot(2,3,2)

plt.title('Fuel System Histogram')

order = data['fuelsystem'].value_counts(ascending=False).index

sns.countplot(x='fuelsystem', data=data, order=order)



plt.show()
plt.subplot(2,3,1)

sns.boxplot(x='enginetype',y='price',data = data)



plt.subplot(2,3,2)

plt.title('Engine Type Histogram')

order = data['enginetype'].value_counts(ascending=False).index

sns.countplot(x='enginetype', data=data, order=order)



plt.show()
plt.subplot(2,3,1)

sns.boxplot(x='cylindernumber',y='price',data = data)



plt.subplot(2,3,2)

plt.title('Cylinder Number Histogram')

order = data['cylindernumber'].value_counts(ascending=False).index

sns.countplot(x='cylindernumber', data=data, order=order)



plt.show()
plt.subplot(2,3,1)

sns.boxplot(x = 'drivewheel', y = 'price', data = data)



plt.subplot(2,3,2)

plt.title('DriveWheel Histogram')

order = data['drivewheel'].value_counts(ascending=False).index

sns.countplot(x='drivewheel', data=data, order=order)



plt.show()
plt.subplot(2,3,1)

sns.boxplot(x=data.symboling, y=data.price)





plt.subplot(2,3,2)

plt.title('Symboling Histogram')

order = data['symboling'].value_counts(ascending=False).index

sns.countplot(x='symboling', data=data, order=order)



plt.show()
metadata_matrix_dataframe = get_meta_data(data)

list_potential_categorical_type(metadata_matrix_dataframe,data)
data = data.drop(['carheight' ,'stroke' ,'compressionratio' ,'peakrpm' ,'carlength' ,'carwidth' ,'curbweight' ,'enginesize' ,'highwaympg'], axis=1)

data.head()
def binary_dummy_replace(x) :

     return x.map({"gas":1,"diesel":0,

                   "std":1,"turbo":0,

                   "two":1, "four":0,

                   "front": 1, "rear": 0})

def dummies(x,df):  

    temp = pd.get_dummies(df[x], prefix=x, drop_first = True)

    

    #l = temp.columns.values

    #for nm in l:

        #newt=x+"_"+nm

        #temp.rename({nm: Replace_Name(x)+"_"+nm}, axis=1, inplace=True)

        

    #print(temp.columns.values)

        

    df = pd.concat([df, temp], axis = 1)

    df.drop([x], axis = 1, inplace = True)

    return df
data = dummies('symboling',data)

data = dummies('CompanyName',data)

data = dummies('fueltype',data)

data = dummies('aspiration',data)

data = dummies('doornumber',data)

data = dummies('carbody',data)

data = dummies('drivewheel',data)

data = dummies('enginelocation',data)

data = dummies('enginetype',data)

data = dummies('cylindernumber',data)

data = dummies('fuelsystem',data)
data.head()
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

cars_train, cars_test= train_test_split(data, train_size=0.67, test_size=0.33, random_state = 0)
from sklearn.preprocessing import StandardScaler,scale

#we create an object of the class StandardScaler

sc = StandardScaler() 



col_to_scale = ['wheelbase','boreratio','horsepower','citympg','price',]



cars_train[col_to_scale] = sc.fit_transform(cars_train[col_to_scale])

cars_test[col_to_scale] = sc.fit_transform(cars_test[col_to_scale])



cars_train.head()
y_train = cars_train.loc[:,cars_train.columns == 'price']



X_train = cars_train.loc[:, cars_train.columns != 'price']
y_test = cars_test.loc[:,cars_test.columns == 'price']



X_test = cars_test.loc[:, cars_test.columns != 'price']
# Making predictions

import statsmodels.api as sm 



lm = sm.OLS(y_train,X_train).fit()



y_pred=lm.predict(X_test)
resid = y_test - y_pred.to_frame('price')
fig = plt.figure(figsize=(9,6))

sns.distplot(resid, bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)

plt.show()
plt.figure(figsize=(9,9))

plt.scatter(y_pred, resid)

plt.hlines(0,-2,4)

plt.suptitle('Residuals vs Predictions', fontsize=16)

plt.xlabel('Predictions')

plt.ylabel('Residuals')
from scipy import stats



def normality_of_residuals_test(model):

    '''

    Function to establish the normal QQ graph of the residues and perform the Anderson-Darming statistical test to study the normality of the residuals.

    

    Arg:

    * model - OLS models adapted from statsmodels

    '''

    sm.ProbPlot(lm.resid).qqplot(line='s');

    plt.title('Q-Q plot');



    ad = stats.anderson(lm.resid, dist='norm')

    

    print(f'----Anderson-Darling test ---- \nstatistic: {ad.statistic:.4f}, critical value of 5%: {ad.critical_values[2]:.4f}')

    

normality_of_residuals_test(lm)
plt.figure(figsize=(15,9))

plt.scatter(resid.index, resid.values)

plt.hlines(0,0,200)

plt.suptitle('Residuals by order', fontsize=16)

plt.xlabel('Order')

plt.ylabel('Residuals')
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(resid))
import statsmodels.tsa.api as smt



acf = smt.graphics.plot_acf(resid, lags=40 , alpha=0.05)

acf.show()
%matplotlib inline

%config InlineBackend.figure_format ='retina'

import seaborn as sns 

import matplotlib.pyplot as plt

import statsmodels.stats.api as sms

from statsmodels.compat import lzip

sns.set_style('darkgrid')

sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)



def homoscedasticity_test(model):

    '''

    Fonction de test de l'homoscédasticité des résidus dans un modèle de régression linéaire.



    Il compare les valeurs résiduelles aux valeurs prédites et exécute les tests de Goldfeld-Quandt.

    

    Args:

    * model - fitted OLS model from statsmodels

    '''

    fitted_vals = model.predict()

    resids = model.resid



    #fit_reg=False

    sns.regplot(x=fitted_vals, y=resids, lowess=True, line_kws={'color': 'red'})

    plt.suptitle('Résidus vs Prédictions', fontsize=16)

    plt.xlabel('Prédictions')

    plt.ylabel('Résidus')



    print('\n----Goldfeld-Quandt test ----')

    name = ['F statistic', 'p-value']

    test = sms.het_goldfeldquandt(lm.resid, lm.model.exog)

    print(lzip(name, test))

    print('\n----Residuals plots ----')



homoscedasticity_test(lm)
fig = plt.figure(figsize=(11,5))

plt.scatter(y_test,y_pred)

plt.xlabel('y_test', fontsize=18)

plt.ylabel('y_pred', fontsize=16)



#Regression Line function

f = lambda x: x



# x values of line to plot

x = np.array(y_test)



# plot fit

plt.plot(x,f(x),lw=2.5, c="orange")

from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)
print(lm.summary())
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
regression = LinearRegression()

regression.fit(X_train,y_train)
rfe = RFE(regression,10)

rfe = rfe.fit(X_train,y_train)
for z in range(len(X_train.columns)):

    print(X_train.columns[z],'\t\t\t',rfe.support_[z])
col = X_train.columns[rfe.support_]

for x in col:

    print(x)
X_train_rfe = X_train[X_train.columns[rfe.support_]]

X_train_rfe.head()
import statsmodels.api as sm 



def color_code_vif_values(val):

    """

    Take a scalar and return a string with the property css 'color: red' for 10, black otherwise.

    """

    if val > 10 : color = 'red' 

    elif val > 5 and val <= 10 : color = 'blue'

    elif val > 0 and val <= 5 : color = 'darkgreen'

    else : color = 'black'

    return 'color: %s' % color



def drop_col(dataframe,col_to_drop) :

    dataframe.drop([col_to_drop],axis=1,inplace=True)

    return dataframe



def display_vif(x) :

    #Calculer les VIFs pour le nouveau modèle

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif = pd.DataFrame()

    X = x

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.set_index("Features")

    vif = vif.sort_values(by = "VIF", ascending = False)

    df = pd.DataFrame(vif.VIF).style.applymap(color_code_vif_values)

    display(df)

    

model_count = 0



def statsmodel_summary(y_var,x_var) :

    global model_count

    model_count = model_count + 1

    text = "*****MODEL - " + str(model_count)

    print(text)

    

    x_var_const = sm.add_constant(x_var) # adding constant

    lm = sm.OLS(y_var,x_var_const).fit() # calculating the fit

    print(lm.summary()) # print summary for analysis

    display_vif(x_var_const.drop(['const'],axis=1))

    return x_var_const , lm
lm = statsmodel_summary(y_train,X_train_rfe)
X_train_rfe = X_train_rfe.drop(["carbody_sedan"], axis = 1)

X_train_rfe.head()
lm = statsmodel_summary(y_train,X_train_rfe)
X_train_rfe = X_train_rfe.drop(["carbody_wagon"], axis = 1)

X_train_rfe.head()
lm = statsmodel_summary(y_train,X_train_rfe)
X_train_rfe = X_train_rfe.drop(["CompanyName_porsche"], axis = 1)

X_train_rfe.head()
lm = statsmodel_summary(y_train,X_train_rfe)
#Array containing names of important variables

final_features = list(X_train_rfe.columns)



#Filter the test dataset

X_test_new = X_test.filter(final_features)



X_test_new.head()
# Making predictions

lm = sm.OLS(y_train,X_train_rfe).fit()



y_pred=lm.predict(X_test_new)
resid = y_test - y_pred.to_frame('price')
plt.figure(figsize=(15,9))

plt.scatter(resid.index, resid.values)

plt.hlines(0,0,200)
print(durbin_watson(resid))
import statsmodels.tsa.api as smt



acf = smt.graphics.plot_acf(resid, lags=40 , alpha=0.05)

acf.show()
homoscedasticity_test(lm)
fig = plt.figure(figsize=(9,6))

sns.distplot(resid, bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)

plt.show()
plt.scatter(y_pred, resid)

plt.hlines(0,-2,4)
normality_of_residuals_test(lm)
fig = plt.figure(figsize=(11,5))

plt.scatter(y_test,y_pred)

plt.xlabel('y_test', fontsize=18)

plt.ylabel('y_pred', fontsize=16)



#Regression Line function

f = lambda x: x



# x values of line to plot

x = np.array(y_test)



# plot fit

plt.plot(x,f(x),lw=2.5, c="orange")
from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)
print(lm.summary())