# Load libraries.
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)

pd.set_option('display.max_columns', None)
# Importing the data and displaying some rows
df = pd.read_csv("../input/malnutrition-across-the-globe/country-wise-average.csv")
country_average = df
display(country_average.head(10))

df = pd.read_csv("../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv")
health_system = df


country_average.count()
#Showing the NaNs
country_average.isna().sum()
# View the second dataset
health_system.count()

# Counting the NaNs
health_system.isna().sum()
# Removed one column containing redundant values
health_system_1=health_system.drop(['Province_State'],axis=1)
# Showing all variables-- distribution
health_system_1.describe()

#Drow Income Classification
import numpy as np
import matplotlib.pyplot as plt

pk_colors = ['#A8B820',  # Bug,
             '#705848',  # Dark,
             '#7038F8',  # Dragon
             '#F8D030',  # Electric
        
             ]

helthSys_cnt = country_average["Income Classification"].value_counts(sort=False).sort_index()
helthSys_cnt = pd.concat([helthSys_cnt, pd.DataFrame(pk_colors,
                                           index=helthSys_cnt.index,
                                           columns=["Colors"])], axis=1)
helthSys_cnt.sort_values("Income Classification", inplace=True)
helthSys_cnt_bar = helthSys_cnt.plot(kind='barh', y="Income Classification", color=helthSys_cnt.Colors,
                                           legend=False, figsize=(8, 8))
helthSys_cnt_bar.set_title("Number of Country\nIncome Classification",
                                           fontsize=16, weight="bold")

helthSys_cnt_bar.set_xlabel("Number of Country")

# We created the *fill_NaN* function to fill up the Country Region column with the information from the World Bank Name column.
# The Country Region column contains NaN
def rellenar_NaN(row):
    if pd.isnull(row['Country_Region']):
        val = row['World_Bank_Name']
    else:
        val = row['Country_Region']
    return val

# We apply the fill_NaN function and assign it to the Country Region column.
health_system_1['Country_Region']=health_system_1.apply(rellenar_NaN, axis=1)
# We check it out
health_system_1['Country_Region'].isna().sum()
# We filter the countries that do not have information in the next column.
health_system_1 = health_system_1[health_system_1['Health_exp_per_capita_USD_2016'].notnull()]
health_system_1_numericas=health_system_1.select_dtypes(include=['number'])
health_system_1_categoricas=health_system_1.select_dtypes(include=['object'])
# View categorical variables
health_system_1_categoricas
# Encoding of the column of *country names*
health_system_1_categoricas['Country_Region_num']=health_system_1_categoricas.Country_Region.astype('category').cat.codes

# View of numerical variables
health_system_1_numericas.isna().sum()
# Let´s look at how many columns without *health system* information are there...
filtered_df= pd.merge(health_system_1_categoricas, health_system_1_numericas, left_index=True, right_index=True)
filtered_df.isna().sum()
# Correlations. Show a mini plot of Current
my_plot = filtered_df.plot("per_capita_exp_PPP_2016", "Health_exp_pct_GDP_2016", kind="scatter")
plt.show()
# Correlations. Show a mini plot
my_plot = filtered_df.plot("Nurse_midwife_per_1000_2009-18", "Health_exp_pct_GDP_2016", kind="scatter")
plt.show()
filtered_df_2=filtered_df[['Country_Region','Health_exp_pct_GDP_2016']]
filtered_df_2=filtered_df_2.head(5)
# Correlation mini Bar Plot
splot=filtered_df_2.plot(kind='bar',stacked=True,title="Region and current health as a GDP")
splot.set_xlabel("Country_Region")
# the countries in the *country average* file to capital letters in order to cross the datasets
filtered_df['Country_Region'] = filtered_df['Country_Region'].str.upper()
dataset_completeness=filtered_df.merge(country_average, left_on='Country_Region', right_on='Country')
dataset_completeness.isna().sum()
# Removing Country column
dataset_completeness.drop(['Country'],axis=1)
# Pint ---- Correlation Matrix
import seaborn as sns

Var_Corr = dataset_completeness.corr()

Var_Corr

sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns)

dataset_completeness['Country_Region'] = dataset_completeness['Country_Region'].str.upper() 
# Change the column name for *Population*
dataset_completeness_1 = dataset_completeness.rename(columns={'''U5 Population ('000s)''': 'Population'})
dataset_completeness_1['pctje_Muerte_por_malnutricion'] =dataset_completeness_1['Severe Wasting']*0.3

dataset_completeness_1['Num_Muerte_por_malnutricion'] =(dataset_completeness_1['pctje_Muerte_por_malnutricion']/100)*dataset_completeness_1['Population']
# We assume  4% cap to define like country_severewasted in this target variable.
def label_race (row):
   if row['Severe Wasting'] >= 3.0 :
      return 1
   else:
      return 0
# we apply the function label_race to datset
dataset_completeness_1['Pais_malnutrido']=dataset_completeness_1.apply (lambda row: label_race(row), axis=1)
country_name=dataset_completeness_1['Country_Region']
dataset_completeness_reg_lineal=dataset_completeness_1
# Remove the columns wich containt categoric variables  
dataset_completeness_1=dataset_completeness_1.drop(['Severe Wasting','Country_Region','World_Bank_Name','Country'], axis=1)
dataset_completeness_1.isna().sum()
#Fill up with 0 all NaN
dataset_completeness_1 = dataset_completeness_1.fillna(0)
#import libraries for Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
dataset_completeness_1.columns
# Apply dTypes to look up variables type
dataset_completeness_1.dtypes
# Choose target and predictor variables
from sklearn.model_selection import train_test_split

y=dataset_completeness_1['Pais_malnutrido']

x=dataset_completeness_1.drop(['Pais_malnutrido'], axis=1)

X_train,X_test, y_train, y_test=train_test_split( x, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# Showing y_test of target variable
y_test
lr = LogisticRegression()
lr.fit(X_train,y_train)
# Show the prediction of logistic Regretion
predictions = lr.predict(X_test)
print(predictions)
predictions
lr.score(X_test,y_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))
# Show the Confusion Matrix
print(confusion_matrix(y_test, predictions))

lista_col=X_train.columns.tolist()
from matplotlib import pyplot
# get importance
importance = lr.coef_[0]
# summarize feature importance

for i,v in enumerate(importance):
    
    print(lista_col[i]+', Score: '+str( "%.2f" % (v,)))

pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
#dataset_completeness=dataset_completeness.drop(['Overweight','pctje_Muerte_por_malnutricion', 'Num_Muerte_por_malnutricion'], axis=1)
# We choose only Helth-Systems variables this time
from sklearn.model_selection import train_test_split

y=dataset_completeness_1['Pais_malnutrido']

x=dataset_completeness_1.drop(['Pais_malnutrido','pctje_Muerte_por_malnutricion','Num_Muerte_por_malnutricion'
                           ,'Overweight','Stunting','Underweight','Wasting','Income Classification','Country_Region_num'], axis=1)

X_train,X_test, y_train, y_test=train_test_split( x, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


lr = LogisticRegression()
lr.fit(X_train,y_train)

predictions = lr.predict(X_test)
print(predictions)

lr.score(X_test,y_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))
# Show the Confusion Matrix
print(confusion_matrix(y_test, predictions))

# We convert the xtrain column array into a list to use the function for
lista_col=X_train.columns.tolist()
from matplotlib import pyplot
# get importance
importance = lr.coef_[0]
# summarize feature importance

for i,v in enumerate(importance):
    
    print(lista_col[i]+', Score: '+str( "%.2f" % (v,)))
    
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
# Join up ytest, la xtest, the target and the prediction. Transform the prediction array to dataframe; do the some with ytest.
# merge function brings together the predictive variables and the objective variable ytest.
# Then the index is reset out to cross with the prediction
df = pd.DataFrame(data=predictions,  columns=["pred"])

y_test=y_test.to_frame()

prediction=pd.merge(X_test, y_test, left_index=True, right_index=True)

prediction.reset_index(inplace=True)

prediction_completeness=pd.merge(prediction, df, left_index=True, right_index=True)


prediction_completeness
prediction_completeness.to_excel('prediction_completeness.xlsx')
def label_race (row):
   if row['Pais_malnutrido'] == 1 :
      return 'Malnutrido'
   else:
      return 'Nutrido'
dataset_completeness_1['Pais_malnutrido']=dataset_completeness_1.apply (lambda row: label_race(row), axis=1)
y=dataset_completeness_1['Pais_malnutrido']

x=dataset_completeness_1.drop(['Pais_malnutrido','pctje_Muerte_por_malnutricion','Num_Muerte_por_malnutricion'], axis=1)

X_train,X_test, y_train, y_test=train_test_split( x, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
X_test
# Load the Library
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
# Take predict on the xtest and see the score it had obtained
y_pred=rfc.predict(X_test)
df_pred = pd.DataFrame(y_pred)
rfc.score(X_test,y_test)

# Take the function Matrix imporatamos to see prediction
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
# We choose the important variables
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# We create a list with an array of predictions.
nom_columnas=X_test.columns.tolist()
# Let`s see the ranking of Feature
print("Feature ranking:")

for f in range(X_test.shape[1]):
    print(nom_columnas[f]+" (%f)" % (  importances[indices[f]]))

from sklearn.model_selection import train_test_split
# Creamos un dataset paralelo para hacer el random forrest 
dataset_completo_rf=dataset_completeness_1.drop(['Country_Region_num'], axis=1)
dataset_completeness_1=dataset_completeness_1.drop(['Country_Region_num'], axis=1)

# 114 Paises
# 102 Nutridos
# 12 Malnutridos
# Evaluamos el numero de paises malnutridos y nutridos que tenemos
print(dataset_completo_rf['Pais_malnutrido'].value_counts())

# Balanceamos el modelo para quedarnos con el mismo número de malnutridos que nutridos
dataset_completo_rf_1 = dataset_completo_rf.groupby('Pais_malnutrido')
dataset_completo_rf_1 = pd.DataFrame(dataset_completo_rf_1.apply(lambda x: x.sample(dataset_completo_rf_1.size().min()).reset_index(drop=True)))

# Evaluamos los cambios realizados
print(dataset_completo_rf_1['Pais_malnutrido'].value_counts())
# Utilizamos el nuevo dataset para lanzar el random forrest
y=dataset_completo_rf_1['Pais_malnutrido']

x=dataset_completo_rf_1.drop(['Pais_malnutrido','pctje_Muerte_por_malnutricion','Num_Muerte_por_malnutricion'], axis=1)

X_train,X_test, y_train, y_test=train_test_split( x, y, test_size=0.35, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
rfc = RandomForestClassifier()
predictor=rfc.fit(X_train, y_train)

y_pred=rfc.predict(X_test)
df_pred = pd.DataFrame(y_pred)

importances = rfc.feature_importances_
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
nom_columnas=X_test.columns.tolist()
# Print the feature ranking
print("Feature ranking:")

for f in range(X_test.shape[1]):
    print(nom_columnas[f]+" (%f)" % (  importances[indices[f]]))
# Draw the desission tree to understand the Random Forest clasification
nom_features=X_test.columns.tolist()
nom_features_y=y_test.tolist()
estimator = rfc.estimators_[5]
from sklearn.tree import export_graphviz
import graphviz
# Export as dot file
graph = export_graphviz(estimator, 
                feature_names = nom_features,
                class_names = nom_features_y,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
graphviz.Source(graph)
dataset_completeness_reg_lineal.columns
dataset_completeness_reg_lineal
# Take exclude the categorical variables
dataset_completeness_reg_lineal=dataset_completeness_reg_lineal.drop(['Country_Region','World_Bank_Name','Country','Country_Region_num','pctje_Muerte_por_malnutricion',
       'Num_Muerte_por_malnutricion', 'Pais_malnutrido'], axis=1)
#Load necessary libraries for Linear Regresion
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Plot all variables-- distribution 
dataset_completeness_reg_lineal.hist()
plt.show()
# Take start Linear Regression Model
regr = linear_model.LinearRegression()

dataset_completeness_reg_lineal=dataset_completeness_reg_lineal.fillna(0)
dataset_completeness_reg_lineal
# Take the target
y=dataset_completeness_reg_lineal['Severe Wasting']

# Taking out the target of dataset X 

x=dataset_completeness_reg_lineal.drop(['Severe Wasting'], axis=1)

#split of datset xtrain e ytrain
X_train,X_test, y_train, y_test=train_test_split( x, y, test_size=0.4, random_state=42)


# Training the model
regr.fit(X_train, y_train)
 
# We make the predictions that ultimately one line (in this case, being 2D)
y_pred = regr.predict(X_test)
 
# Let's see the coefficients obtained. In our case, they will be the Tangent
print('Coefficients: \n', regr.coef_)
# This is the value where the Y axis cuts (in X=0)
print('Independent term: \n', regr.intercept_)
# Mean Square Error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Variance Score. The best score is 1.0
print('Variance score: %.2f' % r2_score(y_test, y_pred))
y_pred
# Let's repeat the same operation transforming the prediction into a data frame
df_pred = pd.DataFrame(data=y_pred,  columns=["pred"])
# How many elements do we have in the test (evaluation) set?
y_test.count()
y_test=y_test.to_frame()
# join up together
prediction=pd.merge(X_test, y_test, left_index=True, right_index=True)
# We create the indox to prediction
prediction.reset_index(inplace=True)
# Lets predict on dataset 
prediction_completeness=pd.merge(prediction, df_pred, left_index=True, right_index=True)
# Show prediction completeness
prediction_completeness
prediction_completeness.to_excel('dataset_reg_lineal_pred.xlsx')
X_test.columns
#Show the hyperparameters of the model
regr
#
regr.coef_
df_columns = pd.DataFrame(X_test.columns.tolist(),  columns =['columns'])
df_coef = pd.DataFrame(data=regr.coef_, columns=["coeficientes"])
df_coeficientes = pd.merge(df_columns, df_coef, left_index=True, right_index=True)
df_coeficientes.to_excel('coeficientes_reg_lineal.xlsx')
# Decision Tree of LinealRegression
from sklearn.tree import DecisionTreeRegressor
# hiperparameter
regr_1 = DecisionTreeRegressor(max_depth=5)

# Separate ytest and xpred from dataset
y=dataset_completeness_reg_lineal['Severe Wasting']

x=dataset_completeness_reg_lineal.drop(['Severe Wasting'], axis=1)

X_train,X_test, y_train, y_test=train_test_split( x, y, test_size=0.4, random_state=42)


# Training the model
regr_1.fit(X_train, y_train)
 
# Making the prediction that ultimately one line 
y_pred = regr_1.predict(X_test)
 
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Variance Score. The best score is 1.0
print('Variance score: %.2f' % r2_score(y_test, y_pred))

print('Score: %.2f' % regr_1.score(X_test, y_test))

regr_1.get_params()
#Plot linear regression
dataset_completeness_reg_lineal.columns
# Pinting the Linear regression of *Level of current health expenditure expressed as a percentage of GDP *.
x_draw=dataset_completeness_reg_lineal['Health_exp_pct_GDP_2016']

X_test_draw=X_test['Health_exp_pct_GDP_2016']
plt.figure()
#plt.scatter(x_draw, y, s=20, edgecolor="black",
 #           c="darkorange", label="data")
plt.plot(X_test, y_test, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()