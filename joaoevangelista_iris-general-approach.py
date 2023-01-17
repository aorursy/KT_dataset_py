import os

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

sns.set_style('whitegrid')

sns.set_palette('Set2')

np.random.seed(1998)
df = pd.read_csv('../input/Iris.csv')
df.head()
plt.figure(figsize=(10,5))

plt.title('Species Variable Counts')

_ = sns.countplot(data=df, x='Species')
# lets see if there is null values

series = df.isnull().sum()

pd.DataFrame({'Column Name': series.index, 'Has Nulls ?': ['Yes' if v > 0 else 'No' for v in series.values]})
fig, axs = plt.subplots(2, 2, figsize=(20, 10))

_ = sns.boxplot(data=df, x='Species', y='SepalWidthCm', ax=axs[0][0])

_ = sns.boxplot(data=df, x='Species', y='SepalLengthCm', ax=axs[0][1])

_ = sns.boxplot(data=df, x='Species', y='PetalWidthCm', ax=axs[1][0])

_ = sns.boxplot(data=df, x='Species', y='PetalLengthCm', ax=axs[1][1])
mean = df.mean()

pd.DataFrame({'Column Name': mean.index, 'Needs Scale ?': ['No' if  1 < v < 10 else 'Yes' for v in mean.values]})
# lets drop the id since it is useless

df.drop('Id', axis=1, inplace=True)
# make a copy to not mess visualizations

edf = df.copy()
corr = df.corr()

ax = sns.heatmap(corr, annot=True, linewidth=0.3, linecolor='w')

_ = plt.xticks(rotation=90)
edf['SepalRatio'] = edf['SepalWidthCm'] / edf['SepalLengthCm'] 

edf['PetalRatio'] = edf['PetalWidthCm'] / edf['PetalLengthCm']
edf.head()
mean = edf.mean()

pd.DataFrame({'Column Name': mean.index, 'Needs Scale ?': ['No' if  1 < v < 10 else 'Yes' for v in mean.values]})
X = edf.drop('Species', axis=1).as_matrix()

y = edf['Species'].values
from sklearn.preprocessing import MinMaxScaler

# scale the features

X_scaler = MinMaxScaler()

X_scaled = X_scaler.fit_transform(X)

from sklearn.preprocessing import LabelEncoder

y_encoder = LabelEncoder()

y_encoded = y_encoder.fit_transform(y)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y) # we use stratify to split equally the classes



params = {'learning_rate': [0.1, 0.2, 0.4, 0.6], 'n_estimators': [100, 200, 300]}



clf = GridSearchCV(GradientBoostingClassifier(), cv=5, param_grid=params)



clf.fit(X_train, y_train)

print('Trained GradientBoostingClassifier')

print('-'*40)

print('Best Features ', clf.best_params_)
score = accuracy_score(y_test, clf.predict(X_test))

print("Accuracy Score {0:.2f}%".format(score * 100))
import joblib

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin



class IrisRatioTransfomer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):

        return self

    

    

    def transform(self, measures_matrix):

        """Given an numpy matrix, it compute the measures of rate betweeen the pairs of information

        Example:

            given: np.ndarray([[1, 2, 3, 4], [1, 2, 3, 4]])

                representing sepal length, sepal width, petal length, petal width

            then: it will return the ration between 2, 1 and 4, 3, appended to the array

                as last positions

        """

        try:

            row_collector = []

            for row in measures_matrix:

                sepal_ratio = row[1] / row[0]

                petal_ratio = row[3] / row[2]

                

                row_collector.append([*row, sepal_ratio, petal_ratio])

            return np.array(row_collector)

        except KeyError as e:

            print('The input data is mal-formed, please check the documentation')

            raise KeyError(e)

    

    

pipeline = Pipeline([

    ('Iris Ratio Transformer', IrisRatioTransfomer()),

    ('Feature Scaler', X_scaler),

    ('Estimator', GradientBoostingClassifier()) # parameters got from the GridSearch, and the default ones

])



# fit the pipeline with raw data, let it do the transformations we did manually

pipeline.fit(df.drop('Species', axis=1).as_matrix(), y_encoder.transform(df['Species'].values))

# save the fitted model with the encoder for decoding

assemble = (pipeline , y_encoder)

# dump the pipeline and the label encoder so we can use at another place

joblib.dump(assemble, 'iris.pkl')

print('ok.')
import joblib



class Measure():

    def __init__(self, sepal_width, sepal_length, petal_width, petal_length):

        self.data = [

            sepal_width,

            sepal_length,

            petal_width,

            petal_length

        ]



        



class Botanist():

    

    def __init__(self):

        "Load the assemble from pipeline step, and assign them to fields acessible from this class"

        

        pipeline, label_encoder = joblib.load('iris.pkl')

        self.pipeline = pipeline

        self.label_encoder = label_encoder

    

    def predict(self, measure):

        """

        Predict the species of Iris based on it's measures.

        This method accepts a single entry, for array computation use

        `bulk_predict`.

        Returns the name of the specie

        """

        return self.bulk_predict([measure])[0]

        

        

    def bulk_predict(self, measures):

        """

        Predict the species of Iris based on it's measures.

        This an array of Measure, for single computation use

        `predict`.

        Returns an array containing the names of the species on

        the same order as defined on the measures array.

        """

        feats = [measure.data for measure in measures]

        predicted = self.pipeline.predict(feats) # we need a matrix

        return self.label_encoder.inverse_transform(predicted)

        

b = Botanist()
b.predict(Measure(5.1,3.5,1.4,0.2))
measures = [

    Measure(6.7, 3.0, 5.2, 2.3), # virginica

    Measure(4.9, 3.0, 1.4, 0.2), # setosa

    Measure(5.9, 3.0, 5.1, 1.8), # virginica

    Measure(5.0, 2.3, 3.3, 1.0), # versicolor

    Measure(4.5, 3.8, 1.7, 0.1) # made up

]



results = b.bulk_predict(measures)

for r in results:

    print('The specie measured is a', r)