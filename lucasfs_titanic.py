import pandas as pd

import numpy as np

import plotly as py



py.offline.init_notebook_mode(connected=True)
data = pd.read_csv('../input/train.csv').drop(columns=['Name', 'PassengerId', 'Ticket', 'Cabin']).fillna(0).replace(

    {'Sex': {'male':0, 'female':1}})

data.head()
data.head()
data.info()
data.describe()
# Correlation Matrix

import plotly.figure_factory as ff



py.offline.iplot(

    ff.create_annotated_heatmap(

        z=data.corr(),

        x=list(data),

        y=list(data)

    )

)
import statsmodels.formula.api as smf

import statsmodels.api as sm



anova = smf.ols(formula='Survived ~ C(Pclass) + C(Sex) + Age + Fare + C(Cabin)', data=train).fit()

sm.stats.anova_lm(anova, typ=3)
survived_sex = pd.crosstab(data.Survived, data.Sex)



bar_survived_sex = py.graph_objs.Bar(

    x = data.Sex,

    y = data.Survived.value_counts()

)



bar_survived_pclass = py.graph_objs.Bar(

    x = data.Pclass,

    y = data.Survived.value_counts()

)



fig = py.tools.make_subplots(1, 2)



fig.append_trace(bar_survived_sex, 1, 1)

fig.append_trace(bar_survived_pclass, 1, 2)



py.offline.iplot(fig) 

import sklearn as skl

import sklearn.model_selection

import sklearn.tree

import sklearn.ensemble

import sklearn.discriminant_analysis



kf = skl.model_selection.KFold(n_splits=10)



response = data.Survived.values

features = data.loc[:, 'Pclass':'Fare'].values.tolist()



results = []

models = [

    ['Logistic Regression', skl.linear_model.LogisticRegression(solver='liblinear')],

    ['Decision Tree', skl.tree.DecisionTreeClassifier()],

    ['KNN', skl.neighbors.KNeighborsClassifier()],

    ['SVM', skl.svm.SVC(gamma='scale')],

    ['Random Forest', skl.ensemble.RandomForestClassifier(n_estimators=100)],

    ['ADABoost', skl.ensemble.AdaBoostClassifier()],

    ['LDA', skl.discriminant_analysis.LinearDiscriminantAnalysis()]

]

                         

for method, model in models:

    cv_model = skl.model_selection.cross_val_score(model, features, response, cv=kf)

    results.append((method,

                    np.around(cv_model, decimals=4),

                    np.around(cv_model.mean(), decimals=4),

                    np.around(cv_model.var(), decimals=4)

    ))



boxplot= []



for result in results:

    boxplot.append(

        py.graph_objs.Box(

            y=result[1],

            name=result[0]

        )

    )

    

py.offline.iplot(boxplot)

pd.DataFrame(np.array(results).reshape(len(models), 4), columns=list(['Method', 'Scores', 'Mean', 'Variance'])).style.set_properties(**{'word-wrap': 'break-word'})
test = pd.read_csv('../input/test.csv')

test = test.drop(columns=['Name', 'Ticket'])

test = test.fillna(0)

test = test.replace(

    {'Sex': {'male':0, 'female':1}})



model = skl.ensemble.AdaBoostClassifier().fit(features, response)

predictions = model.predict(test.loc[:, 'Pclass':'Fare'])



kaggle_submission = pd.DataFrame({'PassengerId':test.PassengerId,

                                  'Survived':predictions})

kaggle_submission.to_csv('titanic.csv', index=False)