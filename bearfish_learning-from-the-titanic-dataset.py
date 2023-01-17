import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import altair as alt # for nice visualization (not standard in kaggle, further functions for rendering see below)

alt.renderers.enable('notebook')



import os

print(os.listdir("../input"))
import json  # need it for json.dumps

from IPython.display import HTML



# Create the correct URLs for require.js to find the Javascript libraries

vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + alt.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



altair_paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {paths}

}});

"""



# Define the function for rendering

def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        """Render an altair chart directly via javascript.

        

        This is a workaround for functioning export to HTML.

        (It probably messes up other ways to export.) It will

        cache and autoincrement the ID suffixed with a

        number (e.g. vega-chart-1) so you don't have to deal

        with that.

        """

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay defined and keep track of the unique div Ids

    return wrapped





@add_autoincrement

def render_alt(chart, id="vega-chart"):

    # This below is the javascript to make the chart directly using vegaEmbed

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vegaEmbed) {{

        const spec = {chart};     

        vegaEmbed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

    }});

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )



HTML("".join((

    "<script>",

    workaround.format(paths=json.dumps(altair_paths)),

    "</script>"

)))

train_source = pd.read_csv('../input/train.csv')

train_source.head(n=10)
train = train_source.drop(columns=['Name', 'Ticket']).copy()
train.describe(include='all')
train.Cabin.drop_duplicates().sort_values().head(n=10)
for Pclass in [1, 2, 3]:

    print("Class: " + str(Pclass) )

    print(train[train.Pclass == Pclass].describe(include='all')[['Pclass', 'Cabin']].iloc[0])
train['Deck'] = np.NAN

train.loc[train['Pclass'] == 1, 'Deck'] = 2

train.loc[train['Pclass'] == 2, 'Deck'] = 5

train.loc[train['Pclass'] == 3, 'Deck'] = 6

train.loc[train['Cabin'].str.contains('T', na=False),'Deck'] = 2

train.loc[train['Cabin'].str.contains('G', na=False),'Deck'] = 7

train.loc[train['Cabin'].str.contains('F', na=False),'Deck'] = 6

train.loc[train['Cabin'].str.contains('E', na=False),'Deck'] = 5

train.loc[train['Cabin'].str.contains('D', na=False),'Deck'] = 4

train.loc[train['Cabin'].str.contains('C', na=False),'Deck'] = 3

train.loc[train['Cabin'].str.contains('B', na=False),'Deck'] = 2

train.loc[train['Cabin'].str.contains('A', na=False),'Deck'] = 1
train['Gender'] = 0

train.loc[train['Sex'] == 'female', 'Gender'] = 1



train['Embark'] = 1

train.loc[train['Embarked'] == 'C', 'Embark'] = 2

train.loc[train['Embarked'] == 'Q', 'Embark'] = 3
train.drop(columns=['Sex', 'Cabin', 'Embarked'], inplace=True)
age_imputer = train.groupby(by=['Pclass', 'Gender'])['Age'].mean()

for pclass in [1, 2, 3]:

    for gender in [0, 1]:

        train.loc[train.Age.isna() & (train['Pclass'] == pclass) & (train['Gender'] == gender), 'Age'] = (age_imputer[pclass][gender])
test_source = pd.read_csv('../input/test.csv')

test_source.describe(include='all')
def transform(df):

    

    # Make a copy

    df_transformed = df.copy()

    

    # Drop PassengerID, Name & Ticket

    df_transformed.drop(columns=['Name', 'Ticket'], inplace=True)

    

    # Build featue "Decks"

    df_transformed['Deck'] = np.NAN

    df_transformed.loc[df_transformed['Pclass'] == 1, 'Deck'] = 2

    df_transformed.loc[df_transformed['Pclass'] == 2, 'Deck'] = 5

    df_transformed.loc[df_transformed['Pclass'] == 3, 'Deck'] = 6

    df_transformed.loc[df_transformed['Cabin'].str.contains('T', na=False),'Deck'] = 2

    df_transformed.loc[df_transformed['Cabin'].str.contains('G', na=False),'Deck'] = 7

    df_transformed.loc[df_transformed['Cabin'].str.contains('F', na=False),'Deck'] = 6

    df_transformed.loc[df_transformed['Cabin'].str.contains('E', na=False),'Deck'] = 5

    df_transformed.loc[df_transformed['Cabin'].str.contains('D', na=False),'Deck'] = 4

    df_transformed.loc[df_transformed['Cabin'].str.contains('C', na=False),'Deck'] = 3

    df_transformed.loc[df_transformed['Cabin'].str.contains('B', na=False),'Deck'] = 2

    df_transformed.loc[df_transformed['Cabin'].str.contains('A', na=False),'Deck'] = 1

    

    # Build features "Gender" and "Embark"

    df_transformed['Gender'] = 0

    df_transformed.loc[df_transformed['Sex'] == 'female', 'Gender'] = 1

    df_transformed['Embark'] = 1

    df_transformed.loc[df_transformed['Embarked'] == 'C', 'Embark'] = 2

    df_transformed.loc[df_transformed['Embarked'] == 'Q', 'Embark'] = 3

    

    # Drop "Sex", "Cabin" and "Embarked"

    df_transformed.drop(columns=['Sex', 'Cabin', 'Embarked'], inplace=True)

    

    # Use age_imputer to impute Age

    for pclass in [1, 2, 3]:

        for gender in [0, 1]:

            df_transformed.loc[df_transformed.Age.isna() & (df_transformed['Pclass'] == pclass) & (df_transformed['Gender'] == gender), 'Age'] = (age_imputer[pclass][gender])

            

    # Impute "Fare" based on Pclass

    

    fare_imputer = df_transformed.groupby(by=['Pclass'])['Fare'].mean()

    

    for pclass in [1, 2, 3]:

        df_transformed.loc[df_transformed.Fare.isna() & (df_transformed['Pclass'] == pclass), 'Fare'] = fare_imputer[pclass]

    

    return df_transformed



check_train = transform(train_source)



(train == check_train).describe()
chart11 = alt.Chart(train).mark_bar().encode(

    alt.X(alt.repeat('row'), type='ordinal'),

    alt.Y('count()', axis=alt.Axis(title='Total')),

    color='Survived:N'

).properties(

    width=200,

    height=150

).repeat(

    row=['Pclass', 'Deck', 'Embark', 'Gender', 'SibSp', 'Parch']

)



chart12 = alt.Chart(train).mark_bar().encode(

    alt.X(alt.repeat('row'), type='quantitative', bin=True),

    alt.Y('count()', axis=alt.Axis(title='Total')),

    color='Survived:N'

).properties(

    width=200,

    height=150

).repeat(

    row=['Fare', 'Age']

)



chart21 = alt.Chart(train).mark_bar().encode(

    alt.X(alt.repeat('row'), type='ordinal'),

    alt.Y('count()', stack='normalize', axis=alt.Axis(title='%')),

    color='Survived:N'

).properties(

    width=200,

    height=150,

).repeat(

    row=['Pclass', 'Deck', 'Embark', 'Gender', 'SibSp', 'Parch']

)



chart22 = alt.Chart(train).mark_bar().encode(

    alt.X(alt.repeat('row'), type='quantitative', bin=True),

    alt.Y('count()', stack='normalize', axis=alt.Axis(title='%')),

    color='Survived:N'

).properties(

    width=200,

    height=150

).repeat(

    row=['Fare', 'Age']

)



render_alt(chart11 & chart12 | chart21 & chart22, id='vega-chart')
chart = alt.Chart(train).mark_circle().encode(

    alt.X('Age:Q', bin=True),

    alt.Y('Pclass:O'),

    alt.Size('count()'),

    color=('mean(Survived):Q'),#, stack='normalize')

    tooltip=(['mean(Survived):Q', 'sum(Survived)', 'count()'])

).properties(

    width=300,

    height=300

)



render_alt(chart, id='vega-chart')
chart = alt.Chart(train).mark_circle().encode(

    alt.X('Pclass:O'),

    alt.Y('Embark:O'),

    alt.Size('count()'),

    color=('mean(Survived):Q'),#, stack='normalize')

    tooltip=(['mean(Survived):Q', 'sum(Survived)', 'count()'])

).properties(

    width=300,

    height=300

)



render_alt(chart, id='vega-chart')
from sklearn.model_selection import train_test_split
X = train.iloc[:,2:].values

y = train.iloc[:,1].values

X_train, X_dev, y_train, y_dev = train_test_split(X, y, random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



log_clf = LogisticRegression(solver="liblinear", random_state=42)

rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)

svm_clf = SVC(gamma="auto", random_state=42)



voting_clf = VotingClassifier(

    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],

    voting='hard')
voting_clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score



for clf in (log_clf, rnd_clf, svm_clf, voting_clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_dev)

    print(clf.__class__.__name__, accuracy_score(y_dev, y_pred))
test = transform(test_source)

X_test = test.iloc[:,1:].values

y_pred = voting_clf.predict(X_test)
my_submission = pd.DataFrame({'PassengerId':test.iloc[:,0].values,'Survived':y_pred})
# export as csv file

my_submission.to_csv("sub.csv", index=False)