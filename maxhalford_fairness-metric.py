import pandas as pd



names = pd.read_csv('/kaggle/input/defi-ia-insa-toulouse/categories_string.csv')['0'].to_dict()

jobs = pd.read_csv('/kaggle/input/defi-ia-insa-toulouse/train_label.csv', index_col='Id')['Category']

jobs = jobs.map(names)

jobs = jobs.rename('job')

jobs.head()
genders = pd.read_json('/kaggle/input/defi-ia-insa-toulouse/train.json').set_index('Id')['gender']

genders.head()
people = pd.concat((jobs, genders), axis='columns')

people.head()
counts = people.groupby(['job', 'gender']).size().unstack('gender')

counts
counts['disparate_impact'] = counts[['M', 'F']].max(axis='columns') / counts[['M', 'F']].min(axis='columns')

counts.sort_values('disparate_impact', ascending=False)
counts['disparate_impact'].mean()
def macro_disparate_impact(people):

    counts = people.groupby(['job', 'gender']).size().unstack('gender')

    counts['disparate_impact'] = counts[['M', 'F']].max(axis='columns') / counts[['M', 'F']].min(axis='columns')

    return counts['disparate_impact'].mean()



people.head()
macro_disparate_impact(people)
from sklearn import model_selection



descriptions = pd.read_json('/kaggle/input/defi-ia-insa-toulouse/train.json').set_index('Id')['description']



X_train, X_test, y_train, y_test, gender_train, gender_test = model_selection.train_test_split(

    descriptions,

    jobs,

    genders,

    test_size=.5,

    random_state=42

)
from sklearn import feature_extraction

from sklearn import linear_model

from sklearn import pipeline

from sklearn import preprocessing



model = pipeline.make_pipeline(

    feature_extraction.text.TfidfVectorizer(),

    preprocessing.Normalizer(),

    linear_model.LogisticRegression(multi_class='multinomial')

)



model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred = pd.Series(y_pred, name='job', index=X_test.index)

y_pred.head()
test_people = pd.concat((y_pred, gender_test), axis='columns')

test_people
macro_disparate_impact(test_people)