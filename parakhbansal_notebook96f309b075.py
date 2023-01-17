#importing all the necessary modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mp
%matplotlib inline
#reading the training dataset
train = pd.read_csv('../input/titanic/train.csv')
#observing the data set as we can see the target survived column is here
train.head()
#reading the test dataset
test = pd.read_csv(r'../input/titanic/test.csv')
#as we can see the target column is not here
test.head()
# Setting up visualisations
sns.set_style(style='white') 
sns.set(rc={
    'figure.figsize':(12,7), 
    'axes.facecolor': 'white',
    'axes.grid': True, 'grid.color': '.9',
    'axes.linewidth': 1.0,
    'grid.linestyle': u'-'},font_scale=1.5)
custom_colors = ["#3498db", "#95a5a6","#34495e", "#2ecc71", "#e74c3c"]
sns.set_palette(custom_colors)
total = pd.concat([train, test], ignore_index=True, sort  = False)
total.info()
#let's check the no. of missing values in each
total.isnull().sum()
pd.DataFrame(total.isnull().sum()).plot.line().set_title("Number of missing values in the given features")
#checking unique values in each column
total.nunique()
fig_pclass = train.Pclass.value_counts().plot.pie().legend(labels=["Class 3","Class 1","Class 2"], loc='center right', bbox_to_anchor=(2.25, 0.5)).set_title("Training Data - People travelling in different classes")


male_pr = round((train[train.Sex == 'male'].Survived == 1).value_counts()[1]/len(train.Sex) * 100, 2)
female_pr = round((train[train.Sex == 'female'].Survived == 1).value_counts()[1]/len(train.Sex) * 100, 2)
sex_perc_df = pd.DataFrame(
    { "Percentage Survived":{"male": male_pr,"female": female_pr},  "Percentage Not Survived":{"male": 100-male_pr,"female": 100-female_pr}})
sex_perc_df.plot.barh().set_title("Percentage of male and female survived and Deceased")
pd.DataFrame(total.Age.describe())

pd.DataFrame(total.Fare.describe())
p = sns.countplot(x = "Embarked", hue = "Survived", data = total, palette=["C1", "C0"])
p.set_xticklabels(["Southampton","Cherbourg","Queenstown"])
p.legend(labels = ["Deceased", "Survived"])
p.set_title("Training Data - Survival based on embarking point.")
#LET'S LABEL ENCODE SEX COLUMN
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()

total['Sex'] = le.fit_transform(total['Sex'])

#let's create dummy columns for emabrked column and then drop the existing column
#also since it has only two missing values we fill those by the ctaegory having highest count
total.Embarked.fillna(total.Embarked.mode()[0], inplace = True)

dumm = pd.get_dummies(total.Embarked, prefix="Emb", drop_first = True)
total = pd.concat([total,dumm], axis=1)
total.drop(['Embarked'],axis=1, inplace=True)
total.head()
total['Salutation'] = total.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
grp = total.groupby(['Sex', 'Pclass'])
grp.Age.apply(lambda x: x.fillna(x.median()))
total.Age.fillna(total.Age.median(), inplace = True)
#fare has 1 missing value so i will fill it with mean
total.Fare.fillna(total.Fare.mean(), inplace=True)
total.head()
#Now Let's drop column that we won't use in prediction
total.drop(['Name','SibSp','Ticket','Cabin','Salutation'],axis=1,inplace=True)
total_test = total[total.Survived.isnull()]
total_test = total_test.drop(['Survived'],axis=1)
total_test.head()
total_train = total.dropna()
y = total_train['Survived']
X = total_train.drop(['Survived'],axis=1)
y = y.astype('int')
from sklearn.model_selection import cross_val_score

#LET'S IMPORT THE MODELS FIRST
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
acc = []
for i in [1,3,5,10]:
    knn = KNeighborsClassifier(n_neighbors=i)
    acc.append(cross_val_score(knn,X,y,cv=5,scoring='accuracy'))
#let's see which k gives us best accuracy
for i in acc:
    print(np.mean(i))
svc = SVC(gamma='auto')
svc_score = cross_val_score(svc, X,y, scoring='accuracy')
print(np.mean(svc_score))
dt = DecisionTreeClassifier()
dt_score = cross_val_score(dt,X,y,scoring='accuracy')
print(np.mean(dt_score))
rf = RandomForestClassifier(criterion='entropy', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf_score = cross_val_score(dt,X,y,scoring='accuracy')
print(np.mean(rf_score))
#let's fit entire data on our random forest model
model = RandomForestClassifier(criterion='entropy', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
model.fit(X,y)
predictions = model.predict(total_test)
predictions.shape
total_test.shape
submission = pd.DataFrame({'PassengerId':total_test.PassengerId,'Survived':predictions})
submission.Survived = submission.Survived.astype(int)
print(submission.shape)
filename = 'Titanic_final.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)
