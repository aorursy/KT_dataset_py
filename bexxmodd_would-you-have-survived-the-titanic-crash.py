import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/titanic/train.csv')

df.head()
df.dtypes
print("total number of NaN in column 'Age' is:", df['Age'].isna().sum())
fcount = 0

mcount = 0



for i in df['Sex']:

    if i == 'female':

        fcount +=1

    else:

        mcount +=1

        

print('We have: ', fcount, 'females')

print('We have: ', mcount, 'males')
df['Sex'].replace({'female':1, 'male':0}, inplace=True)

df.head()
df['Pclass'].value_counts()
df['Family'] = df['SibSp'] + df['Parch']

df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

df.head()
df['Age'].fillna(29, inplace=True) # replace missing values with average age value

df['Age'].apply(np.ceil) # round age up to the closest integer

df.describe()
plt.figure(figsize=(14, 3.5), dpi=80)



plt.subplot(1,3,1)

plt.hist(df['Age'], color='firebrick', alpha=.85)

plt.xlabel('Age')

plt.ylabel('Frequency')



plt.subplot(1,3,2)

plt.hist(df['Fare'], color='firebrick', alpha=.85)

plt.xlabel('Fare')

plt.ylabel('Frequency')



plt.subplot(1,3,3)

plt.hist(df['Family'], color='firebrick', alpha=.85)

plt.xlabel('Family Size')

plt.ylabel('Frequency')
cclass = pd.get_dummies(pd.Series(list(df['Pclass'])), drop_first=True)

cclass
df = pd.concat([df, cclass], axis=1)

df.rename(columns={2: 'SecondClass', 3: 'ThirdClass'},inplace=True)

df.tail()
df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)



df = df.set_index('PassengerId')

df.head(5)
df.tail(5)
df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False)
df1 = df.copy()

df1['Sex'].replace({1:'Female', 0:'Male'}, inplace=True)

df1[['Sex', 'Survived']].groupby(['Sex'],

                                as_index=True).mean().sort_values(by='Survived', ascending=False)
df[['Family', 'Survived']].groupby(['Family'],

                                   as_index=True).mean().sort_values(by='Survived', ascending=False)
# group ages based on 10 year increments for a table

bins = [0, 10, 20, 30, 40, 50, 60, 90]

labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+']

df['Age_gr'] = pd.cut(df1.Age, bins, labels = labels, include_lowest = True)



df.head()
df[['Age_gr', 'Survived']].groupby(['Age_gr'],

                                as_index=True).mean().sort_values(by='Age_gr', ascending=True)
import matplotlib as mpl 

import matplotlib.pyplot as plt

print('Matplotlib imported!')
df['Fare'] = df['Fare'].replace(0, np.nan)

df['Fare'] = df['Fare'].dropna(how='all', axis=0)

print((df['Fare'] == 0).sum())

df.tail()
female = df[['Sex', 'Age', 'Fare', 'Survived', 'Family']].loc[df['Sex'] == 1]

male = df[['Sex', 'Age', 'Fare', 'Survived', 'Family']].loc[df['Sex'] == 0]

female.describe()
plt.figure(figsize=(14, 3.5), dpi=80)

plt.suptitle('Female & Male', fontsize=16)

labels='Female','Male'



plt.subplot(1, 3, 1)

female['Age'].plot(kind='density', linewidth=3, alpha=.7, color='darkred')

male['Age'].plot(kind='density', linewidth=3, alpha=.7, color='orange')

plt.xlim(0, 90)

plt.ylim(0, 0.05)

plt.xlabel('Age')

plt.legend(labels)



plt.subplot(1, 3, 2)

female['Fare'].plot(kind='density', alpha=.7, color='darkred', linewidth=3)

male['Fare'].plot(kind='density', alpha=.7, color='orange', linewidth=3)

plt.xlim(-10, 500)

plt.ylim(0, .025)

plt.xlabel('Fare')

plt.legend(labels)



plt.subplot(1, 3, 3)

female['Family'].plot(kind='density', alpha=.7, color='darkred', linewidth=3)

male['Family'].plot(kind='density', alpha=.7, color='orange', linewidth=3)

plt.xlim(-1, 10)

plt.ylim(0, .7)

plt.xlabel('Family Size')

plt.legend(labels)
g = sns.catplot(x="Fare", y="Survived", row="Pclass", kind='box', palette='YlOrRd',

                orient="h", height=1.8, aspect=3.5, hue_order='Ascending',

                data=df1)

g.set(xscale='log')

g.set(xlim=(3, 500))
h = sns.catplot(x="Age", y="Survived", row="Pclass", kind='box', palette='YlOrRd',

                orient="h", height=1.8, aspect=3.5, hue_order='Ascending',

                data=df1)

f = sns.catplot(x="Age_gr", y="Survived", col="Pclass", palette='YlOrRd',

                data=df, kind="bar", ci=None, height=6, aspect=1.33)

f.set_axis_labels('', 'Survival Rate')

f.set(ylim=(0,1))
aw = sns.regplot(x='Fare', y='Survived', data=df, logistic=True,

                 line_kws={"color":"orange","alpha":0.3,"lw":4},

                 scatter_kws={"color":"red", "edgecolor":'darkred', 'alpha':.7, "s":100} )

aw.set_ylabel('Survival Probability')

aw.set_xlabel('Fare')

aw.set_title('Logistic Model',fontsize=16)
ax = sns.regplot(x='Age', y='Survived', data=df, logistic=True, marker='o', 

                 line_kws={"color":"orange","alpha":0.3,"lw":4},

                 scatter_kws={"color":"red", "edgecolor":'darkred', 'alpha':.7, "s":100} )

ax.set_ylabel('Survival Probability')

ax.set_xlabel('Age')

ax.set_title('Logistic Model',fontsize=16)
# importing necessary packages

import statsmodels.api as sm

import statsmodels.formula.api as smf

import statsmodels.api as sm

from statsmodels.iolib.summary2 import summary_col

print('statsmodels imported!')
df['lnFare'] = np.log(df['Fare'])

df.head()
logit1 = smf.logit('Survived ~ Sex + Age', data=df).fit()

print(logit1.summary())
logit2 = smf.logit('Survived ~ Sex + Age + lnFare', data=df).fit()
logit3 = smf.logit('Survived ~ Sex + Age + lnFare + Family', data=df).fit()
logit4 = smf.logit('Survived ~ Sex + Age + lnFare + Family + SecondClass + ThirdClass', data=df).fit()
# we build table charachteristics and plug regressions

info_dict={'Pseudo R-squared' : lambda x: f"{x.prsquared:.4f}",

           'No. observations' : lambda x: f"{int(x.nobs):d}"}



results_table = summary_col(results=[logit1, logit2, logit3, logit4],

                            float_format='%0.4f',

                            stars = True,

                            model_names=['Model 1',

                                         'Model 2',

                                        'Model 3',

                                        'Model 4'],

                            info_dict=info_dict,)



results_table.add_title('Logit Regressions - Table')



print(results_table)
print('SURVIVED = ', logit4.params[0].round(4),'+', logit4.params[1].round(4),'* SEX',

      logit4.params[2].round(4),'* AGE +', logit4.params[3].round(4),'* lnFARE', logit4.params[4].round(4), '* FAMILY',

     logit4.params[4].round(4),'* 2NDCLASS', logit4.params[6].round(4), '* 3RDCLASS')
df = df.dropna()

Xset = df[['Sex','Age','lnFare','Family','SecondClass','ThirdClass']]

y = df[['Survived']]



from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(Xset, y)

predictions = logmodel.predict(Xset)
# this function will generate confusion matrix needed to evaluate success rate of our logistic model

from sklearn.metrics import classification_report, confusion_matrix

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Reds):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="grey" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

print(confusion_matrix(y, predictions, labels=[1,0]))
# Compute confusion matrix

cnf_matrix = confusion_matrix(y, predictions, labels=[1,0])

np.set_printoptions(precision=2)





# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['Survived=1','Survived=0'],normalize= False,  title='Confusion matrix')
from sklearn.metrics import classification_report

print(classification_report(y, predictions))
acc_log = round(logmodel.score(Xset, y) * 100, 2)

print('Accuracy of our train data via Logistic model can be summed as: ', acc_log,'%')
sub = pd.read_csv('../input/titanic/gender_submission.csv')

test = pd.read_csv('../input/titanic/test.csv')

test.head()
print('Test dataset shape:', test.shape)

print('Outcomes\' shape:', sub.shape)
# we combine those two datasets

test_model = pd.merge(test, sub, on='PassengerId')

test_model.head()
test_model.drop(['Cabin', 'Ticket'], axis=1, inplace=True)



test_model = test_model.set_index('PassengerId')

test_model.head(10)
# update Sex to dummy variables; male=1 and female=1

test_model['Sex'].replace({'female':1, 'male':0}, inplace=True)

test_model.head()
test_model['Age'].fillna(30, inplace=True) # replace missing values with average values

test_model['Age'].apply(np.ceil) # roung age up to the closest integer

test_model.describe()
test_class1 = pd.get_dummies(pd.Series(list(test_model['Pclass'])), drop_first=True)

test_class1
test_model = test_model.reset_index()

test_model = pd.concat([test_model, test_class1], axis=1)

test_model.rename(columns={2: 'SecondClass', 3: 'ThirdClass'}, inplace=True)

test_model.head()
test_model['Family'] = test_model['SibSp'] + test_model['Parch']

test_model.drop(['SibSp', 'Parch'], axis=1, inplace=True)

test_model.head()
test_model['Fare'] = test_model['Fare'].replace(0, np.nan)

test_model['Fare'] = test_model['Fare'].dropna(how='all', axis=0)

print((df['Fare'] == 0).sum())



test_model['lnFare'] = np.log(test_model['Fare'])

print('Shape: ', test_model.shape)

test_model.tail()
test_model.dropna(inplace=True)

test_model.reset_index(drop=True, inplace=True)

print(test_model.shape)

test_model.tail()
Xtest1 = test_model[['Sex','Age','lnFare','Family','SecondClass','ThirdClass']]

yhat1 = logmodel.predict(Xtest1)

print(yhat1.shape)

print(test_model.shape)
yhat11 = pd.DataFrame(yhat1)

yhat11.tail()
test_model.tail()
finalll = pd.concat([test_model, yhat11], axis=1)

finalll.rename(columns={0: 'Predicted'}, inplace=True)

finalll.tail()
count = 0

wrong = 0

total = 0



for index, row in finalll.iterrows():

    if row['Survived'] == row['Predicted']:

        count += 1

        total += 1

    else:

        wrong += 1

        total += 1

        

print('correct prediction ratio:', count/ total)

print('wrong prediction ratio:', wrong / total )