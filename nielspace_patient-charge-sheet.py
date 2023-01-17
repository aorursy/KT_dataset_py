import pandas as pd

insurance = pd.read_csv("../input/insurance/insurance.csv")
insurance.head()
insurance.isna().sum()/len(insurance)
insurance.describe()
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#just a trail

sns.distplot(insurance.age)
print(np.std(np.log(insurance.charges)))

sns.distplot(np.log(insurance.charges))
insurance['age_cat'] = np.nan

lst = [insurance]
lst
for col in lst:

    col.loc[(col['age'] >= 18) & (col['age'] <= 35), 'age_cat'] = 'Young Adult'

    col.loc[(col['age'] > 35) & (col['age'] <= 55), 'age_cat'] = 'Senior Adult'

    col.loc[col['age'] > 55, 'age_cat'] = 'Elder'

    
print(lst)
age_cat = insurance.age_cat.map({'Young Adult':0, 

 'Senior Adult':1,

 'Elder':2})

labels = insurance["age_cat"].unique()

amount = insurance["age_cat"].value_counts().tolist()
my_circle=plt.Circle( (0,0), 0.7, color='white')



plt.figure(figsize=(10,10))

plt.pie(amount, labels=labels, colors=['red','green','blue'])



p=plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
plt.figure(figsize=(15,10))

sns.distplot(insurance.bmi)

plt.show()
plt.figure(figsize=(10,10))

sns.heatmap(insurance.corr())

plt.show()



print('*'*100)

print(insurance.corr())
young_adults = insurance["bmi"].loc[insurance["age_cat"] == "Young Adult"].values

senior_adult = insurance["bmi"].loc[insurance["age_cat"] == "Senior Adult"].values

elders = insurance["bmi"].loc[insurance["age_cat"] == "Elder"].values
plt.figure(figsize=(10,10))

sns.boxplot(data= [young_adults, senior_adult, elders])

import statsmodels.api as sm

from statsmodels.formula.api import ols





moore_lm = ols("bmi ~ age_cat", data=insurance).fit()

print(moore_lm.summary())
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
#sex

label.fit(insurance.sex.drop_duplicates())

insurance.sex = label.transform(insurance.sex)

insurance.sex.head()
#smoker or non-smoker

insurance.smoker = label.fit_transform(insurance.smoker)

insurance.smoker.head()
#region



insurance.region = label.fit_transform(insurance.region)

insurance.region.head()
insurance.describe()
insurance.corr()
plt.figure(figsize=(15,10))

sns.heatmap(insurance.corr())

plt.show()
moore_lm = ols("charges ~ smoker", data=insurance).fit()

print(moore_lm.summary())
plt.figure(figsize=(10,10))

sns.distplot(insurance.charges)

plt.show()
insurance.loc[(insurance.smoker == 1)].charges
f = plt.figure(figsize=(20,10))



ax = f.add_subplot(121)

sns.distplot(insurance.loc[(insurance.smoker == 1)].charges, ax=ax)

ax.set_title('Smokers')





ax = f.add_subplot(122)

sns.distplot(insurance.loc[(insurance.smoker == 0)].charges, color='r', ax = ax)

ax.set_title('Non-Smokers')
plt.figure(figsize=(15,10))

sns.catplot(x='smoker', kind='count', hue = 'sex', palette='PuBuGn_r', data=insurance)

plt.show()
f = plt.figure(figsize=(20,20))



ax = f.add_subplot(211)

sns.boxenplot(x = 'age', y='charges', hue='sex', data=insurance, ax=ax)



ax = f.add_subplot(212)

sns.scatterplot(x = 'charges', y='age', hue='smoker', data=insurance, ax=ax)



plt.show()
plt.figure(figsize=(10,10))

sns.distplot(insurance.age, color='r')
insurance["weight_condition"] = np.nan

lst = [insurance]



for col in lst:

    col.loc[col["bmi"] < 18.5, "weight_condition"] = "Underweight"

    col.loc[(col["bmi"] >= 18.5) & (col["bmi"] < 24.986), "weight_condition"] = "Normal Weight"

    col.loc[(col["bmi"] >= 25) & (col["bmi"] < 29.926), "weight_condition"] = "Overweight"

    col.loc[col["bmi"] >= 30, "weight_condition"] = "Obese"
f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,8))



# I wonder if the cluster that is on the top is from obese people

sns.stripplot(x="age_cat", y="charges", data=insurance, ax=ax1, linewidth=1, palette="Reds")

ax1.set_title("Relationship between Charges and Age")





sns.stripplot(x="age_cat", y="charges", hue="weight_condition", data=insurance, ax=ax2, linewidth=1, palette="Set2")

ax2.set_title("Relationship of Weight Condition, Age and Charges")



sns.stripplot(x="smoker", y="charges", hue="weight_condition", data=insurance, ax=ax3, linewidth=1, palette="Set2")

ax3.legend_.remove()

ax3.set_title("Relationship between Smokers and Charges")



plt.show()
import seaborn as sns

sns.set(style="ticks")

pal = ["#FA5858", "#58D3F7"]



sns.pairplot(insurance, hue="smoker", palette=pal)

plt.title("Smokers")
f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,8))

sns.scatterplot(x="bmi", y="charges", hue="weight_condition", data=insurance, palette="Set1", ax=ax1)

ax1.set_title("Relationship between Charges and BMI by Weight Condition")



sns.scatterplot(x="bmi", y="charges", hue="smoker", data=insurance, palette="Set1", ax=ax2)

ax2.set_title("Relationship between Charges and BMI by Smoking Condition")

sns.scatterplot(x='children', y='age', data=insurance, hue='charges')
insurance.children.unique()
plt.hist(insurance.children)
sns.barplot(insurance.children, insurance.charges)
plt.figure(figsize=(15,10))

sns.violinplot(x='children', y='charges', data=insurance)
plt.boxplot(insurance.children)

plt.show()
insurance.children.std()
from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=3)
insurance.head()
X = insurance.drop(['age_cat', 'weight_condition' ], axis=1)

y = insurance.charges
cluster.fit(X)
cluster.cluster_centers_
X.values[:,0]
fig = plt.figure(figsize=(12,8))



plt.scatter(X.values[:,2], X.values[:,6], c=cluster.labels_, cmap="Set1_r", s=25)

plt.scatter(cluster.cluster_centers_[:,2] ,cluster.cluster_centers_[:,6], color='black', marker="o", s=250)
plt.figure(figsize=(15,10))

sns.heatmap(insurance.corr())

plt.show()
X = insurance.drop('region', axis=1)
plt.figure(figsize=(15,10))

sns.heatmap(X.corr())

plt.show()
plt.figure(figsize=(15,10))

sns.scatterplot(x='children', y='bmi', hue='weight_condition', data=insurance)

plt.show()
X.std()
plt.figure(figsize=(10, 12))

plt.boxplot(insurance.bmi)

plt.show()
new_bmi = X.bmi.values

q25, q75 = np.percentile(new_bmi, 25), np.percentile(new_bmi, 75)

print(f'Quartile 25: {q25} | Quartile 75: {q75}')

new_bmi_iqr = q75 - q25

print(f'iqr: {new_bmi_iqr}')
new_bmi_cutoff = new_bmi_iqr * 1.5

new_bmi_lower, new_bmi_upper = q25 - new_bmi_cutoff, q75 + new_bmi_cutoff

print('Lower: ', new_bmi_lower)

print('Upper :', new_bmi_upper)
outliers = [x for x in new_bmi if x<new_bmi_lower or x>new_bmi_upper]

outliers, len(outliers)
final_df = X.drop(X[(X.bmi>new_bmi_upper) | (X.bmi<new_bmi_lower)].index)
plt.figure(figsize=(10,15))

plt.boxplot(final_df.bmi)

plt.show()
new_age = X.age.values

q25, q75 = np.percentile(new_age, 25), np.percentile(new_age, 75)

print(f'Quartile 25: {q25} | Quartile 75: {q75}')

new_age_iqr = q75 - q25

print(f'iqr: {new_age_iqr}')



new_age_cutoff = new_age_iqr * 1.5

new_age_lower, new_age_upper = q25 - new_age_cutoff, q75 + new_age_cutoff

print('Lower: ', new_age_lower)

print('Upper :', new_age_upper)



outliers = [x for x in new_age if x<new_age_lower or x>new_age_upper]

outliers, len(outliers)



final_df = X.drop(X[(X.age>new_age_upper) | (X.age<new_age_lower)].index)



plt.figure(figsize=(10,15))

plt.boxplot(final_df.age)

plt.show()
from sklearn.preprocessing import StandardScaler
Scale = StandardScaler()

final_df.bmi = Scale.fit_transform(final_df.bmi.values.reshape(-1,1))

final_df.std()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
import math

def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_test), y_test),

                m.score(X_train, y_train), m.score(X_test, y_test)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
X = final_df.drop(['charges', 'age_cat', 'weight_condition'], axis=1)

y = np.log(final_df.charges)





X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 23, test_size=0.3)
model=LinearRegression()

model.fit(X_train, y_train)
print_score(model)
model.intercept_
model.coef_
plt.figure(figsize=(10,10))

plt.plot(X_train, y_train, 'ro')

plt.plot(X_train,model.coef_[0]*X_train + model.intercept_)

plt.show()
from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor()

model.fit(X_train, y_train)

model.score(X_test, y_test)
model=RandomForestRegressor(n_estimators=25, n_jobs=-1, max_depth=6, max_features=0.5)

model.fit(X_train, y_train)

model.score(X_test, y_test)
rmse(model.predict(X_test), y_test)
from sklearn.tree import export_graphviz

from IPython import display

from io import StringIO

import re
import graphviz

import IPython



def draw_tree(t, df, size=10, ratio=0.6, precision=0):

    """ Draws a representation of a random forest in IPython.

    Parameters:

    -----------

    t: The tree you wish to draw

    df: The data used to train the tree. This is used to get the names of the features.

    """

    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,

                      special_characters=True, rotate=True, precision=precision)

    IPython.display.display(graphviz.Source(re.sub('Tree {',

       f'Tree {{ size={size}; ratio={ratio}', s)))
draw_tree(model.estimators_[0], X, precision=5)
print_score(model)
X.columns
np.exp(model.predict([[30, 0, 0.4, 3, 0]]))
insurance.loc[(insurance.age == 30) & (insurance.bmi<=20)]
feature_importances = pd.DataFrame(model.feature_importances_,

                                   index = X.columns,

                                    columns=['importance']).sort_values('importance',  ascending=False)
feature_importances.plot.barh(figsize=(15,8))