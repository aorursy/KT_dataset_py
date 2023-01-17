import math



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
DATA_PATH = '/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv'
df = pd.read_csv(DATA_PATH)
df
df.columns
fig, axes = plt.subplots(4, 3, figsize=(24, 18))



sns.distplot(df['fixed acidity'], ax=axes[0, 0])

sns.distplot(df['volatile acidity'], ax=axes[0, 1])

sns.distplot(df['citric acid'], ax=axes[0, 2])

sns.distplot(df['residual sugar'], ax=axes[1, 0])

sns.distplot(df['chlorides'], ax=axes[1, 1])

sns.distplot(df['free sulfur dioxide'], ax=axes[1, 2])

sns.distplot(df['total sulfur dioxide'], ax=axes[2, 0])

sns.distplot(df['density'], ax=axes[2, 1])

sns.distplot(df['pH'], ax=axes[2, 2])

sns.distplot(df['sulphates'], ax=axes[3, 0])

sns.distplot(df['alcohol'], ax=axes[3, 1])



plt.show()
df = df.loc[:, ['fixed acidity',

                'residual sugar',

                'chlorides',

                'density',

                'pH',

                'quality']]
df.head()
class NaiveBayes(object):

    def __init__(self):

        pass

    

    @staticmethod

    def _compute_mean(list_of_numbers):

        return sum(list_of_numbers) / len(list_of_numbers)

    

    @staticmethod

    def _compute_mean_std(list_of_numbers):

        mean = NaiveBayes._compute_mean(list_of_numbers)

        std = math.sqrt(sum([(number - mean)**2 for number in list_of_numbers]) / (len(list_of_numbers) - 1))

        

        return (mean, std)

            

    

    def fit(self, df, feature_cols=['fixed acidity', 'residual sugar', 'chlorides', 'density', 'pH'], label_col='quality'):

        df = df.loc[:, feature_cols + [label_col]]

        self.feature_cols = feature_cols

        self.label_col = label_col

        self.classes = df[self.label_col].unique()

        self.classes.sort()

        

        self.groups = [df.loc[df[self.label_col] == class_name].drop(self.label_col, axis=1) for class_name in self.classes]

        

        self.class_prior_probs = {class_name: (len(group) / len(df)) for class_name, group in zip(self.classes, self.groups)}

        

        self.params = dict()

        for class_name, group in zip(self.classes, self.groups):

            self.params[class_name] = [self._compute_mean_std(group[feature].tolist()) for feature in self.feature_cols]

    

    @staticmethod    

    def _compute_gaussian_prob(x, mean, std):

        return 1 / (math.sqrt(2 * math.pi) * std) * math.exp(-((x-mean)**2)/(2*(std**2)))

    

    def _compute_prob_given_class(self, row, class_name):

        class_prob = 1

        for feature, (mean, std) in zip(self.feature_cols, self.params[class_name]):

            class_prob *= self._compute_gaussian_prob(row[feature], mean, std)

            

        return class_prob

        

    def predict(self, df):

        predictions = []

        

        for idx, row in df.iterrows():

            best_label, best_value = None, -1

            

            for class_name in self.classes:

                prob_item_given_class = self._compute_prob_given_class(row, class_name)

                class_prior_prob = self.class_prior_probs[class_name]

                

                prob_class_given_item = prob_item_given_class * class_prior_prob

                

                if prob_class_given_item > best_value:

                    best_value = prob_class_given_item

                    best_label = class_name

                    

            predictions.append(best_label)

                    

        return predictions

            
model = NaiveBayes()
df = pd.read_csv(DATA_PATH)
model.fit(df)
model.params
model.predict(df)