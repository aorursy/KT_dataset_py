import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as mpl

from matplotlib import pyplot as plt

from wordcloud import WordCloud

from sklearn.metrics import accuracy_score, log_loss

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC, LinearSVC

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder

%matplotlib inline
trd = pd.read_csv('/kaggle/input/titanic/train.csv')

tsd = pd.read_csv('/kaggle/input/titanic/test.csv')



td = pd.concat([trd, tsd], ignore_index=True, sort=False)
td.shape
td.info()
pd.DataFrame(td.isnull().sum()).plot.line().set_title('Number of missing values in the given features')

td.isnull().sum()
sns.heatmap(td.isnull(), cbar=False).set_title('Missing values heatmap')
td.nunique()
(trd.Survived.value_counts(normalize=True) * 100).plot.barh().set_title('Trainig Data - Perceentage of people survived and Deceased')
fig_pclass = trd.Pclass.value_counts().plot.pie().legend(labels=["Class 3","Class 1","Class 2"], loc='center right', bbox_to_anchor=(2.25, 0.5)).set_title("Training Data - People travelling in different classes")
pclass_1_survivor_distribution = round((trd[trd.Pclass == 1].Survived == 1).value_counts()[1]/len(trd[trd.Pclass == 1]) * 100, 2)

pclass_2_survivor_distribution = round((trd[trd.Pclass == 2].Survived == 1).value_counts()[1]/len(trd[trd.Pclass == 2]) * 100, 2)

pclass_3_survivor_distribution = round((trd[trd.Pclass == 3].Survived == 1).value_counts()[1]/len(trd[trd.Pclass == 3]) * 100, 2)

pclass_perc_df = pd.DataFrame(

    { "Percentage Survived":{"Class 1": pclass_1_survivor_distribution,"Class 2": pclass_2_survivor_distribution, "Class 3": pclass_3_survivor_distribution},  

     "Percentage Not Survived":{"Class 1": 100-pclass_1_survivor_distribution,"Class 2": 100-pclass_2_survivor_distribution, "Class 3": 100-pclass_3_survivor_distribution}})

pclass_perc_df.plot.bar().set_title("Training Data - Percentage of people survived on the basis of class")
for x in [1,2,3]:    ## for 3 classes

    trd.Age[trd.Pclass == x].plot(kind="kde")

plt.title("Age density in classes")

plt.legend(("1st","2nd","3rd"))
for x in ["male","female"]:

    td.Pclass[td.Sex == x].plot(kind="kde")

plt.title("Training Data - Gender density in classes")

plt.legend(("Male","Female"))
pclass_perc_df
fig_sex = (trd.Sex.value_counts(normalize = True) * 100).plot.bar()

male_pr = round((trd[trd.Sex == 'male'].Survived == 1).value_counts()[1]/len(trd.Sex) * 100, 2)

female_pr = round((trd[trd.Sex == 'female'].Survived == 1).value_counts()[1]/len(trd.Sex) * 100, 2)

sex_perc_df = pd.DataFrame(

    { "Percentage Survived":{"male": male_pr,"female": female_pr},  "Percentage Not Survived":{"male": 100-male_pr,"female": 100-female_pr}})

sex_perc_df.plot.barh().set_title("Percentage of male and female survived and Deceased")

fig_sex
pd.DataFrame(td.Age.describe())
td['Age_Range'] = pd.cut(td.Age, [0, 10, 20, 30, 40, 50, 60,70,80])

sns.countplot(x = "Age_Range", hue = "Survived", data = td, palette=["C1", "C0"]).legend(labels = ["Deceased", "Survived"])
sns.distplot(td['Age'].dropna(),color='darkgreen',bins=30)
td.SibSp.describe()
ss = pd.DataFrame()

ss['survived'] = trd.Survived

ss['sibling_spouse'] = pd.cut(trd.SibSp, [0, 1, 2, 3, 4, 5, 6,7,8], include_lowest = True)

(ss.sibling_spouse.value_counts()).plot.area().set_title("Training Data - Number of siblings or spouses vs survival count")
x = sns.countplot(x = "sibling_spouse", hue = "survived", data = ss, palette=["C1", "C0"]).legend(labels = ["Deceased", "Survived"])

x.set_title("Training Data - survival based on number of siblings or spouses")
pd.DataFrame(td.Parch.describe())


pc = pd.DataFrame()

pc['survived'] = trd.Survived

pc['parents_children'] = pd.cut(trd.Parch, [0, 1, 2, 3, 4, 5, 6], include_lowest = True)

(pc.parents_children.value_counts()).plot.area().set_title("Training Data - Number of parents/children and survival density")
x = sns.countplot(x = "parents_children", hue = "survived", data = pc, palette=["C1", "C0"]).legend(labels = ["Deceased", "Survived"])

x.set_title("Training Data - Survival based on number of parents/children")
td['Family'] = td.Parch + td.SibSp

td['Is_Alone'] = td.Family == 0
td.Fare.describe()


td['Fare_Category'] = pd.cut(td['Fare'], bins=[0,7.90,14.45,31.28,120], labels=['Low','Mid',

                                                                                      'High_Mid','High'])
x = sns.countplot(x = "Fare_Category", hue = "Survived", data = td, palette=["C1", "C0"]).legend(labels = ["Deceased", "Survived"])

x.set_title("Survival based on fare category")
p = sns.countplot(x = "Embarked", hue = "Survived", data = trd, palette=["C1", "C0"])

p.set_xticklabels(["Southampton","Cherbourg","Queenstown"])

p.legend(labels = ["Deceased", "Survived"])

p.set_title("Training Data - Survival based on embarking point.")
td.Embarked.fillna(td.Embarked.mode()[0], inplace = True)
td['Salutation'] = td.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip()) 

td.Salutation.nunique()

wc = WordCloud(width = 1000,height = 450,background_color = 'white').generate(str(td.Salutation.values))

plt.imshow(wc, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()



td.Salutation.value_counts()
grp = td.groupby(['Sex', 'Pclass'])  

td.Age = grp.Age.apply(lambda x: x.fillna(x.median()))



#If still any row remains

td.Age.fillna(td.Age.median, inplace = True)
sal_df = pd.DataFrame({

    "Survived":

    td[td.Survived == 1].Salutation.value_counts(),

    "Total":

        td.Salutation.value_counts()

})

s = sal_df.plot.barh()
td.Cabin = td.Cabin.fillna('NA')
td = pd.concat([td,pd.get_dummies(td.Cabin, prefix="Cabin"),pd.get_dummies(td.Age_Range, prefix="Age_Range"), pd.get_dummies(td.Embarked, prefix="Emb", drop_first = True), pd.get_dummies(td.Salutation, prefix="Title", drop_first = True),pd.get_dummies(td.Fare_Category, prefix="Fare", drop_first = True), pd.get_dummies(td.Pclass, prefix="Class", drop_first = True)], axis=1)

td['Sex'] = LabelEncoder().fit_transform(td['Sex'])

td['Is_Alone'] = LabelEncoder().fit_transform(td['Is_Alone'])
td.drop(['Pclass', 'Fare','Cabin', 'Fare_Category','Name','Salutation', 'Ticket','Embarked', 'Age_Range', 'SibSp', 'Parch', 'Age'], axis=1, inplace=True)
# Data to be predicted

X_to_be_predicted = td[td.Survived.isnull()]

X_to_be_predicted = X_to_be_predicted.drop(['Survived'], axis = 1)



#Training data

train_data = td

train_data = train_data.dropna()

feature_train = train_data['Survived']

label_train  = train_data.drop(['Survived'], axis = 1)

train_data.shape #891 x 28



##Gaussian

clf = GaussianNB()

x_train, x_test, y_train, y_test = train_test_split(label_train, feature_train, test_size=0.2)

clf.fit(x_train,  np.ravel(y_train))

print("NB Accuracy: "+repr(round(clf.score(x_test, y_test) * 100, 2)) + "%")

result_rf=cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')

print('The cross validated score for GNB is:',round(result_rf.mean()*100,2))

y_pred = cross_val_predict(clf,x_train,y_train,cv=10)

sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="summer")

plt.title('Confusion_matrix for NB', y=1.05, size=15)
##Random forest

clf = RandomForestClassifier(criterion='entropy', 

                             n_estimators=700,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

x_train, x_test, y_train, y_test = train_test_split(label_train, feature_train, test_size=0.2)

clf.fit(x_train,  np.ravel(y_train))

print("RF Accuracy: "+repr(round(clf.score(x_test, y_test) * 100, 2)) + "%")



result_rf=cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')

print('The cross validated score for Random forest is:',round(result_rf.mean()*100,2))

y_pred = cross_val_predict(clf,x_train,y_train,cv=10)

sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="summer")

plt.title('Confusion_matrix for RF', y=1.05, size=15)
result = clf.predict(X_to_be_predicted)

submission = pd.DataFrame({'PassengerId':X_to_be_predicted.PassengerId,'Survived':result})

submission.Survived = submission.Survived.astype(int)

print(submission.shape)

filename = 'Titanic Predictions.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)