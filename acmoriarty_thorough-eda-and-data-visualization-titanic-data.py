# Importing the libraries

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()  # for plot styling

import numpy as np

import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, mean_squared_error, mean_absolute_error, explained_variance_score
data_titanic = pd.read_csv(r"../input/titanic/train.csv")

data_titanic.head(3)
data_titanic.isna().sum()
tita = data_titanic.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)

tita.head()
tita.info
tita.shape
tita.dtypes
tita.describe
tita.isna().sum()
tita2 = tita.dropna()

tita2.head(2)
tita2['Survived'].value_counts()
tita['Age'].nunique()
# for heat maps, indexing / correaltions needs to be established

plt.figure(figsize=(12,8))

tc = data_titanic.corr()

sns.heatmap(tc,cmap='coolwarm')

plt.title('Correlation of figures in the Titanic Dataset')
tita_num = tita2.drop(["Sex","Embarked"], axis=1)

tita_num.head()
corr = tita_num.corr()

corr
cat_corr = tita2.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)

cat_corr
plt.figure(figsize=(14,8))

sns.heatmap(corr, cmap='hot')
plt.figure(figsize=(14,8))

sns.heatmap(cat_corr)
data_corr = cat_corr['Survived']

golden_features_list = data_corr[abs(data_corr) > 0.4].sort_values(ascending=False)

print("There is {} strongly correlated values with Survived Feature:\n{}".format(len(golden_features_list), golden_features_list))
plt.figure(figsize=(14,8))

sns.heatmap(cat_corr[(cat_corr >= 0.2) | (cat_corr <= -0.1)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
cat_corr['Survived']
plt.rc('font', size=12)

plt.rc('axes', titlesize=18)





plt.figure(figsize=(10,7))

tita2.Survived.groupby(tita2.Sex).sum().plot(kind='pie')

#plt.axis('equal')

plt.title("Survival Percentages by Gender")

plt.show()
labels=np.array(['Pclass', 'SibSp','Parch'])

stats=data_titanic.loc[386,labels].values
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

# close the plot

stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))
plt.rc('font', size=12)

plt.rc('axes', titlesize=18)

fig=plt.figure(figsize=(10,10))



ax = fig.add_subplot(1,2,1, polar=True)

ax.plot(angles, stats, 'o-', linewidth=2)

ax.fill(angles, stats, alpha=0.25)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title([data_titanic.loc[386,"Name"]])

ax.grid(True)
plt.figure(figsize=(8,4))

sns.countplot(tita2['Embarked'], palette="Set1")

plt.title("Port of Embarkation")
tita2.head(2)
sns.pairplot(tita2, corner=True)
plt.figure(figsize=(8,6))

sns.distplot(tita2["Age"], kde=False)

plt.title("Age Distribution Aboard the Titanic")
f, axes = plt.subplots(1, 2, figsize=(12,5))



sns.countplot(tita2['Parch'], orient='v', ax=axes[0])

axes[0].title.set_text('Number of Parents and Children Aboard the Titanic')



sns.countplot(tita2['SibSp'], orient='v', ax=axes[1])

axes[1].title.set_text('# of siblings / spouses aboard the Titanic')



f.tight_layout()
sns.set(style="ticks")

titanic = sns.load_dataset("titanic")

variables = list(titanic.select_dtypes(include=["object", "bool"]).drop(["embarked"], axis=1).columns)  



#We have used the titanic dataset because it is formatted as needed for the boxplot



titanic
titanic.select_dtypes(include=["object", "bool"]).drop(["embarked"], axis=1)
plt.figure(figsize=(15,10))



for i, c in enumerate(variables, 1):

    plt.subplot(2,3,i) 

    g = sns.boxplot(x=c, y="fare",data=titanic.query("fare>0"))

    g.set(yscale="log")
sns.set_style('whitegrid')

sns.jointplot(x='Fare',y='Age',data=data_titanic)
bins= [0,2,4,13,20,40, 110]

labels = ['Infant','Toddler','Kid','Teen','Adult', 'Adult-Adult']

titanic['AgeGroup'] = pd.cut(titanic['age'], bins=bins, labels=labels, right=False)

titanic.head(2)
ages_survive = pd.DataFrame(titanic['AgeGroup'].groupby(titanic['alive']).value_counts())

ages_survive = ages_survive.rename(columns = {'AgeGroup':'Age Group Number'})

ages_survive = ages_survive.reset_index()

ages_survive = ages_survive.rename(columns = {'AgeGroup':'Age Group',"alive":"Survived?"})

ages_survive
plt.figure(figsize=(12,8))

sns.barplot(data = ages_survive, x = 'Age Group', y = 'Age Group Number', hue='Survived?', palette = 'rocket')
import plotly.express as px



fig = px.treemap(ages_survive, path=['Survived?', 'Age Group'], values='Age Group Number',

                  color='Age Group Number', hover_data=['Survived?'],

                  color_continuous_scale='RdBu',

                  )

fig.show()
ages_survive1=ages_survive[:6]

ages_survive1
ages_survive2=ages_survive[6:]

ages_survive2
ages_survive['proportion %']=0

for i in range(5):

    ages_survive['proportion %'][i]=(ages_survive1["Age Group Number"][i]/(ages_survive2["Age Group Number"][i+6] + ages_survive1["Age Group Number"][i])) * 100



for i in range(5):

    ages_survive['proportion %'][i+6]=(ages_survive2["Age Group Number"][i+6]/(ages_survive2["Age Group Number"][i+6] + ages_survive1["Age Group Number"][i])) * 100
from tabulate import tabulate



print(tabulate(ages_survive, tablefmt="pipe", headers=ages_survive.columns))
sns.relplot(x="age", y="fare",kind="line", hue="sex", data=titanic, height=6, aspect=2.5)
plt.figure(figsize=(12,8))

sns.set_style('whitegrid')

sns.distplot(data_titanic["Fare"],kde=False) #without the kde
plt.figure(figsize=(12,8))

sns.boxplot(x='Pclass',y='Age',data=data_titanic)
plt.figure(figsize=(12,8))

sns.set_style('white')

sns.swarmplot(x='class',y='age',data=titanic)
g = sns.FacetGrid(data=data_titanic, col='Sex')

g.map(sns.distplot, 'Age',kde=False)
f, axes = plt.subplots(1, 2, figsize=(12,5))



sns.violinplot(x="who", y="fare", data=titanic, ax=axes[0])

axes[0].title.set_text('Fares for the Individual Gender Groups')



sns.violinplot(x="who", y="age", data=titanic, scale="width", pellete="Set3", ax=axes[1])

axes[1].title.set_text('Ages for the Respective Individual Gender Groups')



f.tight_layout()
f, axes = plt.subplots(1, 2, figsize=(12,5))



sns.violinplot(x="class", y="fare", hue="sex", data=titanic, palette="muted", ax=axes[0])

axes[0].title.set_text('Comparing class, gender and fares')



sns.violinplot(x="alive", y="fare", hue="sex", data=titanic, palette="Set2", ax=axes[1])

axes[1].title.set_text('Comparing gender, fare and survival status')



f.tight_layout()
plt.figure(figsize=(12,7))

sns.scatterplot(x="age", y='fare', hue='class', data=titanic, size="class", sizes=(20, 200),palette="Set2",legend="full")
g = sns.lmplot(x="age", y="fare", hue="sex", palette="Set1",data=titanic,markers=["o", "x"],height=6,aspect=2)
plt.figure(figsize=(12,7))

sns.distplot(titanic['fare'], vertical=True, kde=False)

sns.catplot(x='alive', y= 'age',hue='sex',data=titanic,  kind='bar', ci=None)
tita_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
alive_fare = pd.DataFrame(titanic['fare'].groupby(titanic['alive']).sum()).reset_index()

alive_fare
import plotly.express as px



fig = px.bar(alive_fare, x='alive', y='fare', title="Fare--Survival Breakdown in Interactive Bar Chart")

fig.show()
import squarify # (algorithm for treemap)

plt.figure(figsize=(12,12))

labels=['Third Class','First Class','Second Class']

squarify.plot(sizes=titanic['class'].value_counts(), label=labels, alpha=0.75)



plt.axis('off')

plt.show()
sns.catplot(x='alive', y= 'age',hue='sex',data=titanic,  kind='bar')
import scipy.stats



def find_remove_outlier_iqr(data_sample):

    q1 = np.percentile(data_sample, 25)

    q3 = np.percentile(data_sample, 75)

    

    iqr = q3 - q1

    

    cutoff = iqr * 1.5

    

    lower, upper = q1-cutoff, q3+cutoff

    

    outliers =[]

    outliers_removed = []

    for x in data_sample:

        if x < lower or x > upper:

            outliers.append(x)

        if x > lower and x < upper:

            outliers_removed.append(x)

    return outliers
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



outliers = find_remove_outlier_iqr(titanic["fare"])



out_df = titanic[titanic["fare"].isin(outliers)]

out_df.head(20)
out_df.shape
out_df.describe()
#creating a dataframe without the outliers

no_out_df = titanic[~titanic["fare"].isin(outliers)]

no_out_df.head()
no_out_df.describe()
num_df3 = no_out_df.select_dtypes(include = ['float64', 'int64'])
f, ax = plt.subplots(figsize=(12, 8))

out_corr = num_df3.corr()

sns.heatmap(out_corr,xticklabels=out_corr.columns.values,yticklabels=out_corr.columns.values)
out_corr
# correlation plot

num_df3.corrwith(num_df3.survived).plot.bar(figsize=(10,7),

                                   title='Correlation with Response Variable',

                                   grid = True)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import sklearn.linear_model

import sklearn.ensemble

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier
tita2['Log-Age']=np.log(tita2['Age'])

tita2.head(2)
var_mod = ['Pclass','Sex','Embarked']

lab_enc = LabelEncoder()



for i in var_mod:

    tita2[i] = lab_enc.fit_transform(tita2[i])
tita2.head(2)
tita3 = pd.concat([tita2,pd.get_dummies(tita2['Sex'], prefix='gender')],axis=1)

tita3.drop(['Sex'],axis=1, inplace=True)

tita4 = pd.concat([tita3,pd.get_dummies(tita3['Pclass'], prefix='class')],axis=1)

tita4.drop(['Pclass'],axis=1, inplace=True)

tita5 = pd.concat([tita4,pd.get_dummies(tita4['Embarked'], prefix='Port')],axis=1)

tita5.drop(['Embarked'],axis=1, inplace=True)
tita5.head()
X=tita5.loc[:, tita5.columns != 'Survived']

y=tita5['Survived']
X.ndim
sc_X = StandardScaler()

scaled_X = sc_X.fit_transform(X)
scaled_X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.25, random_state = 0)
decision_model=DecisionTreeClassifier()

decision_model.fit(X_train,y_train)
estimator = []

estimator.append(('cart',decision_model))



svm_model=SVC().fit(X_train,y_train)

estimator.append(('svm',svm_model))



logmodel = LogisticRegression().fit(X_train,y_train)

estimator.append(('logistic', logmodel))



ensemble = VotingClassifier(estimator)

ensemble.fit(X_train,y_train)
pred_dec=decision_model.predict(X_test)



from sklearn.metrics import accuracy_score

accuracy_score(y_test,pred_dec)
svm_pred=svm_model.predict(X_test)

accuracy_score(y_test,svm_pred)
log_pred=logmodel.predict(X_test)

accuracy_score(y_test,log_pred)
Voting_pred=ensemble.predict(X_test)

accuracy_score(y_test,Voting_pred)