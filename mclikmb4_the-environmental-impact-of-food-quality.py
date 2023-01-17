import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_eco_all = pd.read_csv( "../input/ecology-related-data-from-openfoodfacts/openfoodfacts_Eco.csv")
df_eco_all =df_eco_all.drop(['Unnamed: 0'], axis=1)
df_eco_all = df_eco_all.rename({'pnns_groups_2': 'catPNNS'}, axis=1) #easier name
df_eco_all.head()
df_eco_all['nutrition_grade_fr'].unique()
# plot eventual correlations for the main dataframe containing data relative to quality and environmental impact
import seaborn as sns
#sns.set(color_codes=True)
#sns.set(style="white", palette="muted")

#sns.pairplot(df_eco_all)
# some correlation can be observed between nutrition score and carbon foodprint
# no obvious correlations can be observed between the two main ecological parameters: carbon footprint and number of palm oil ingredients
# moreover, the columns still contain zero and Nan values
# based on this we split the database and analize the subset after a further cleaning
carbon_categories = ['product_name','nutrition_grade_fr_n','nutrition_grade_fr','catPNNS', 'carbon-footprint_100g']
df_carbon = df_eco_all[carbon_categories]
df_carbon = df_carbon.dropna(subset=['carbon-footprint_100g', 'nutrition_grade_fr']) 
df_carbon = df_carbon.sort_values(by=['carbon-footprint_100g'], ascending=False)
df_carbon['nutrition_grade_fr'].unique()
print(df_carbon.shape)
df_carbon[:25]
df_carbon = df_carbon[df_carbon['carbon-footprint_100g'] > 1]  # eliminate unrealistic values
print(df_carbon.shape) # overview

df_carbon['nutrition_grade_fr'].unique()
# as it could also be observed in the matrix above, higher nutrition score have an as an average a higher carbon footprint. We plot again the graph below.
# and we estimate the visual observation with a simple univariate linear regression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import linear_model
X0 = np.matrix([np.ones(df_carbon.shape[0]), df_carbon['nutrition_grade_fr_n'] ]).T
y0 = np.matrix([df_carbon['carbon-footprint_100g']]).T

regr = linear_model.LinearRegression() 
regr.fit(X0, y0)
estimated_carbon_foodprint = regr.predict(X0)
accuracy0 = regr.score(X0, y0)
print(accuracy0)


plt.scatter(df_carbon['nutrition_grade_fr_n'],df_carbon['carbon-footprint_100g'] )
plt.plot(df_carbon['nutrition_grade_fr_n'],estimated_carbon_foodprint, color = 'tomato' )
plt.xlabel('French nutritional score', color="red", fontsize = 14)
plt.ylabel('Carbon footprint', color="red", fontsize = 14)
plt.savefig("Quality carbon0.png", format="PNG")

# Now redo the correlation after grouping by nutritional grade average score
df_carbon_impact = df_carbon.drop(columns=['catPNNS','product_name','nutrition_grade_fr'])
df_carbon_impact = df_carbon_impact.groupby(['nutrition_grade_fr_n']).mean()
df_carbon_impact = df_carbon_impact.sort_values(by=['nutrition_grade_fr_n'], ascending=False)
df_carbon_impact.reset_index(inplace=True)

df_carbon_impact
grades =['A','B','C','D']
df_carbon_impact['french food grade'] =grades

df_carbon_impact
# this grouping indicates that the two variables are well correlated
X1 = np.matrix([np.ones(df_carbon_impact.shape[0]), df_carbon_impact['nutrition_grade_fr_n'] ]).T
y1 = np.matrix([df_carbon_impact['carbon-footprint_100g']]).T

regr = linear_model.LinearRegression() 
regr.fit(X1, y1)
estimated_carbon_foodprint_byGrade = regr.predict(X1)
accuracy12 = regr.score(X1, y1)
print(accuracy12)
# the evident correlation is shown by the graph below
plt.figure(figsize=(8,5))

x_coords = df_carbon_impact['nutrition_grade_fr_n']
y_coords = df_carbon_impact['carbon-footprint_100g']
for i,type in enumerate(grades):
    x = x_coords[i]
    y = y_coords[i]
    plt.scatter(x, y, marker='o', color='blue')
    plt.text(x+0.3, y+0.3, type, fontsize=9)

plt.plot(df_carbon_impact['nutrition_grade_fr_n'],estimated_carbon_foodprint_byGrade, color = 'g' )
plt.title('Environmental impact by food quality', color="red", fontsize = 14)
plt.xlabel('French nutrition grade average score', color="red", fontsize = 14)
plt.ylabel('Carbon footprint (100g)', color="red", fontsize = 14)


plt.show()

plt.savefig("Quality carbon footprint.png", format="PNG")


palm_categories = ['nutrition_grade_fr_n', 'ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n']
df_palm = df_eco_all[palm_categories]

#df_palm = df_palm[df_palm['ingredients_from_palm_oil_n'] > 0] 
#df_palm = df_palm[df_palm['ingredients_that_may_be_from_palm_oil_n'] > 0] 
df_palm = df_palm.sort_values(by=['ingredients_from_palm_oil_n'], ascending=False)

df_palm = df_palm.dropna() 
df_palm = df_palm.groupby(['nutrition_grade_fr_n']).mean() 

df_palm = df_palm.reset_index(drop=False)

print(df_palm.shape)
df_palm
# prepare data
X2 = df_palm.iloc[:, 1:3].values
y2 = df_palm.iloc[:, 0].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.5, random_state = 42)

regr.fit(X_train, y_train)
y_pred = regr.predict(X_test) # estimation of nutrition score based on fit

accuracy2 = regr.score(X_test, y_test)
print('score = ',accuracy2)
print('')
print(X2)
regr = linear_model.LinearRegression() 
regr.fit(X2, y2)
y_pred = regr.predict(X2)
accuracy21 = regr.score(X2, y2)
print(accuracy21)
numerical_categories = ['nutrition_grade_fr_n','carbon-footprint_100g','ingredients_from_palm_oil_n','ingredients_that_may_be_from_palm_oil_n']
df_eco = df_eco_all[numerical_categories]
df_eco = df_eco_all.dropna(subset=['carbon-footprint_100g', 'nutrition_grade_fr_n'])

df_eco = df_eco.fillna(value = 0)
df_eco = df_eco.groupby(['nutrition_grade_fr_n']).mean() 
df_eco = df_eco.sort_values(by=['nutrition_grade_fr_n'], ascending=True)
print(df_eco.shape)
df_eco = df_eco.reset_index(level=0, inplace=False) # gets grades to the first column
df_eco
#linear fit
X3 = df_eco.iloc[:, 1:4].values
y3 = df_eco.iloc[:, 0].values


regr.fit(X3, y3)
y_pred = regr.predict(X3) # estimation of nutrition score based on fit

accuracy3 = regr.score(X3, y3)
print(accuracy3)
ax1 = sns.distplot(y3, hist=False, color="r", label="Actual Nutritional Grade Value")
sns.distplot(y_pred, hist=False, color="b", label="Fitted Nutritional Grade Value" , ax=ax1)
features = ['ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n', 'carbon-footprint_100g',] # select the features to transform

# Separating out the features
x = df_eco.loc[:, features].values
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x) # scale those features
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
principalComponents = pca.fit_transform(x)

principal_eco_Df = pd.DataFrame(data = principalComponents , columns = ['environmental impact'])


print('the new component contains',pca.explained_variance_ratio_ * 100, '% of the information from the two variables')
# 
df_eco['environmental impact'] = principal_eco_Df['environmental impact']
df_eco.head()
X4 = np.matrix([np.ones(df_eco.shape[0]), df_eco['environmental impact'] ]).T
y4 = np.matrix([df_eco['nutrition_grade_fr_n']]).T

regr = linear_model.LinearRegression() 
regr.fit(X4, y4)
predicted_quality = regr.predict(X4)
accuracy4 = regr.score(X4, y4)
print(accuracy4)
plt.figure(figsize=(8,5))

x_coords = df_eco['environmental impact']
y_coords = df_eco['nutrition_grade_fr_n']
for i,type in enumerate(grades):
    x = x_coords[i]
    y = y_coords[i]
    plt.scatter(x, y, marker='o', color='blue')
    plt.text(x+0.3, y+0.3, type, fontsize=9)

plt.plot(df_eco['environmental impact'],predicted_quality, color = 'g' )
plt.title('Environmental impact by food quality', color="red", fontsize = 14)
plt.xlabel('Environmental impact (palm oil content and carbon footprint)', color="red", fontsize = 14)
plt.ylabel('French nutrition grade average score', color="red", fontsize = 14)


plt.show()

plt.savefig("Quality all.png", format="PNG")
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with few decision trees: too low, underfitting, too high overfiting and heavy processing. Since data are so scarce we keep it low

rf = RandomForestRegressor(n_estimators = 10, random_state = 42) 
rf.fit(X3, y3)
predicted_quality = rf.predict(X3)
accuracy5 = rf.score(X3, y3)
print(accuracy5)
plt.figure(figsize=(8,5))

plt.scatter(x_coords, y_coords)
plt.plot(X3, predicted_quality)
plt.title('Random Forest', color="red", fontsize = 14)
plt.xlabel('Environmental impact (palm oil content and carbon footprint)', color="red", fontsize = 14)
plt.ylabel('French nutrition grade average score', color="red", fontsize = 14)
df_carbon.head()
palm_categories = ['product_name','catPNNS','nutrition_grade_fr_n', 'ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n']
df_palm = df_eco_all[palm_categories]
df_palm = df_palm[df_palm['ingredients_from_palm_oil_n'] > 0] 
df_palm = df_palm[df_palm['ingredients_that_may_be_from_palm_oil_n'] > 0]

df_palm['palm_oil_tot'] = df_palm['ingredients_from_palm_oil_n'] + df_palm['ingredients_that_may_be_from_palm_oil_n']
df_palm = df_palm.dropna() 
df_palm = df_palm.sort_values(by=['palm_oil_tot'], ascending=False)
df_palm[:15]
df_palm = df_palm.groupby(['catPNNS']).mean()
df_palm = df_palm.sort_values(by=['palm_oil_tot'], ascending=False)
df_palm[:-10]
# here a visualization of average environmental impact (carbon footprint)
# and quality for each category according to the french national health programme
df_carbon_cat = df_carbon[['nutrition_grade_fr_n','catPNNS', 'carbon-footprint_100g']]
df_carbon_cat = df_carbon_cat[df_carbon_cat.catPNNS != 'unknown']

df_carbon_cat = df_carbon_cat.groupby(['catPNNS']).mean()
df_carbon_cat = df_carbon_cat.sort_values(by=['carbon-footprint_100g'], ascending=False)
df_carbon_cat = df_carbon_cat.reset_index(drop=False)
df_carbon_cat[:-10]
print(df_carbon_cat.shape)
df_carbon_cat.describe()
nutritionalcategories = ['catPNNS','nutrition_grade_fr_n']
df_nutcat = df_eco_all[nutritionalcategories]
df_nutcat[:10]
df_nutcat.describe()

df_nutcat = df_nutcat.groupby(['catPNNS']).mean()
df_nutcat = df_nutcat.sort_values(by=['nutrition_grade_fr_n'], ascending=True)
df_nutcat = df_nutcat.reset_index(drop=False)
df_nutcat.head()

plt.figure(figsize=(8,10))
plt.barh(df_nutcat['catPNNS'], df_nutcat['nutrition_grade_fr_n'])
plt.title('Nutritional value by PNNS food category', color="red", fontsize = 14)
plt.ylabel('Average nutrition score', color="red", fontsize = 14)
plt.xticks(rotation='vertical')
plt.rcParams['figure.constrained_layout.use'] = True
plt.savefig("PNNS Category based nutrition.png", format="PNG", dpi = 100)
df_e = df_eco_all[df_eco_all['nutrition_grade_fr'] == "e"]
df_e_reduced = df_e[['nutrition_grade_fr_n','catPNNS']]
df_e_reduced = df_e_reduced.groupby(['catPNNS']).count()
df_e_reduced = df_e_reduced.sort_values(by=['nutrition_grade_fr_n'], ascending= False)
df_e_reduced = df_e_reduced.reset_index(drop=False)
df_e_reduced.head()
df_e_reduced = df_e_reduced[df_e_reduced.catPNNS != 'unknown'] 

df_e_reduced = df_e_reduced[df_e_reduced['nutrition_grade_fr_n'] > 1000] # filter noisy values

total = np.sum(df_e_reduced.loc[:,'nutrition_grade_fr_n':].values)
print("tot: ",total)
df_e_reduced['percentage'] = (df_e_reduced['nutrition_grade_fr_n']/total)*100
df_e_reduced

# pie chart E grade
plt.figure(figsize=(10,10))
labels = df_e_reduced['catPNNS']
sizes = df_e_reduced['percentage']


fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig("Category E.png", format="PNG", dpi = 100)
plt.show()
df_a = df_eco_all[df_eco_all['nutrition_grade_fr'] == "a"]
df_a_reduced = df_a[['nutrition_grade_fr_n','catPNNS']]
df_a_reduced = df_a_reduced.groupby(['catPNNS']).count()
df_a_reduced = df_a_reduced.sort_values(by=['nutrition_grade_fr_n'], ascending= False)
df_a_reduced = df_a_reduced.reset_index(drop=False)
df_a_reduced = df_a_reduced[df_a_reduced.catPNNS != 'unknown'] 
df_a_reduced = df_a_reduced[df_a_reduced['nutrition_grade_fr_n'] > 1000]
total = np.sum(df_a_reduced.loc[:,'nutrition_grade_fr_n':].values)
print("tot: ",total)
df_a_reduced['percentage'] = (df_a_reduced['nutrition_grade_fr_n']/total)*100
df_a_reduced
plt.figure(figsize=(10,10)) # pie chart A grade
labels = df_a_reduced['catPNNS']
sizes = df_a_reduced['percentage']


fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig("Category A.png", format="PNG", dpi = 100)
plt.show()

df_nutriscore = df_eco_all.groupby(['nutrition_grade_fr']).count()

df_nutriscore = df_nutriscore.reset_index(drop=False)
df_nutriscore.head()
total1 = np.sum(df_nutriscore.loc[:,'product_name':].values)
print("tot: ",total)
df_nutriscore['percentage'] = (df_nutriscore['product_name']/total)*100
plt.figure(figsize=(10,10)) # pie chart A grade
labels = df_nutriscore['nutrition_grade_fr']
sizes = df_nutriscore['percentage']


fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig("eco_cat.png", format="PNG", dpi = 100)
plt.show()
df_cat = df_eco_all.groupby(['nutrition_grade_fr']).count()

df_nutriscore = df_nutriscore.reset_index(drop=False)
df_nutriscore.head()
df_nutcat1 = df_eco_all.groupby(['catPNNS']).count()
df_nutcat1 = df_nutcat1.sort_values(by=['product_name'], ascending=False)
df_nutcat1 = df_nutcat1.reset_index(drop=False)
df_nutcat1.head()
df_nutcat1.tail()
df_nutcat1 = df_nutcat1[df_nutcat1['product_name'] < 150000] 
df_nutcat1 = df_nutcat1[df_nutcat1['product_name'] > 5000] 
total2 = np.sum(df_nutcat1.loc[:,'product_name':].values)
print("tot: ",total)
df_nutcat1['percentage'] = (df_nutcat1['product_name']/total)*100

plt.figure(figsize=(10,10)) # pie chart A grade
labels = df_nutcat1['catPNNS']
sizes = df_nutcat1['percentage']


fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig("Category main.png", format="PNG", dpi = 100)
plt.show()
df_meat = df_carbon[df_carbon['catPNNS'] == 'meat']
df_meat = df_meat.sort_values(by=['carbon-footprint_100g'], ascending=False)
df_meat = df_meat[['product_name', 'carbon-footprint_100g','nutrition_grade_fr']]
df_meat
df_carbon_d = df_carbon[df_carbon['nutrition_grade_fr'] == 'd']
df_carbon_d = df_carbon_d.sort_values(by=['carbon-footprint_100g'], ascending=False)
df_carbon_d[:5]
box_d = plt.boxplot(df_carbon_d['carbon-footprint_100g'], showmeans=True)
plt.ylabel("Carbon footprint", color="red", fontsize = 14)
plt.xticks([1], ['Nutriscore D'], color="red", fontsize = 14)
plt.savefig("Carbon footprint D.png", format="PNG", dpi = 100)
df_carbon.shape
#remove outliers in class D
df_carbon1 = df_carbon[df_carbon['product_name'] != 'Suprême Noir Authentique']
df_carbon1 = df_carbon1[df_carbon1['product_name'] != 'Viande des grisons']
df_carbon1 = df_carbon1[df_carbon1['carbon-footprint_100g'] < 1500] 
df_carbon1.shape # only three values lost: very few outliers above 1500
df_carbon_impact = df_carbon1.drop(columns=['catPNNS','product_name','nutrition_grade_fr'])
df_carbon_impact = df_carbon_impact.groupby(['nutrition_grade_fr_n']).mean()
df_carbon_impact = df_carbon_impact.sort_values(by=['nutrition_grade_fr_n'], ascending=False)
df_carbon_impact.reset_index(inplace=True)
grades =['A','B','C','D']
df_carbon_impact['french food grade'] =grades

X1 = np.matrix([np.ones(df_carbon_impact.shape[0]), df_carbon_impact['nutrition_grade_fr_n'] ]).T
y1 = np.matrix([df_carbon_impact['carbon-footprint_100g']]).T

regr = linear_model.LinearRegression() 
regr.fit(X1, y1)
estimated_carbon_foodprint_byGrade = regr.predict(X1)
accuracy13 = regr.score(X1, y1)
print(accuracy13)
plt.figure(figsize=(8,5))

x_coords = df_carbon_impact['nutrition_grade_fr_n']
y_coords = df_carbon_impact['carbon-footprint_100g']
for i,type in enumerate(grades):
    x = x_coords[i]
    y = y_coords[i]
    plt.scatter(x, y, marker='o', color='blue')
    plt.text(x+0.3, y+0.3, type, fontsize=9)

plt.plot(df_carbon_impact['nutrition_grade_fr_n'],estimated_carbon_foodprint_byGrade, color = 'g' )
plt.title('Environmental impact by food quality', color="red", fontsize = 14)
plt.xlabel('French nutrition grade average score', color="red", fontsize = 14)
plt.ylabel('Carbon footprint (100g)', color="red", fontsize = 14)

plt.savefig("Quality carbon footprint improved.png", format="PNG")

plt.show()


