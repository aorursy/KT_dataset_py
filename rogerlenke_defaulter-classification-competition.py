import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import metrics, model_selection, tree, neural_network, ensemble, linear_model, svm, neighbors, preprocessing



import matplotlib.pyplot as plt

import seaborn as sns
class Learning():

    model = None

    

    data = []

    prediction_data = []

    train = {}

    test = {}

    

    to_drop = []

    to_dummy = []

    to_binarize = []

    

    def split(self):

        data_temp = self.data.copy()

        

        data_temp = self.preprocess_data_temp(data_temp)

        

        X, y = data_temp.drop(columns=['default payment next month']), data_temp['default payment next month']

        

        self.train['X'], self.test['X'], self.train['y'], self.test['y'] = model_selection.train_test_split(X, y)

        

    def train_model(self, complete=False):

        if complete:

            data_temp = self.data.copy()

            data_temp = self.preprocess_data_temp(data_temp)

            

            X, y = data_temp.drop(columns=['default payment next month']), data_temp['default payment next month']

            

            self.model.fit(X, y)

            

        else:

            self.model.fit(self.train['X'], self.train['y'])

        

    def score(self):

        probs = self.model.predict_proba(self.train['X'])

        probs = probs[:, 1]

        auc = metrics.roc_auc_score(self.train['y'], probs)

        

        p = self.model.predict(self.train['X'])

        

        print('Train score:')

        print(f'Precision: {metrics.precision_score(self.train["y"], p)}')

        print(f'Accurracy: {metrics.accuracy_score(self.train["y"], p)}')

        print(f'Recall: {metrics.recall_score(self.train["y"], p)}')

        print(f'ROC_AUC: {auc}')

        

        p = self.model.predict(self.test['X'])

        

        probs = self.model.predict_proba(self.test['X'])

        probs = probs[:, 1]

        auc = metrics.roc_auc_score(self.test['y'], probs)

        

        print('Test score:')

        print(f'Precision: {metrics.precision_score(self.test["y"], p)}')

        print(f'Accurracy: {metrics.accuracy_score(self.test["y"], p)}')

        print(f'Recall: {metrics.recall_score(self.test["y"], p)}')

        print(f'ROC_AUC: {auc}')

        

    def cross_validate(self):

        data_temp = self.data.copy()

        

        data_temp = self.preprocess_data_temp(data_temp)

        

        X, y = data_temp.drop(columns=['default payment next month']), data_temp['default payment next month']

        

        cv = model_selection.cross_validate(self.model,

            X, y, cv=5, scoring=['precision', 'accuracy', 'recall', 'roc_auc'],

            return_train_score=True)



            

        print('Train score:')

        print(f'Precision: {cv["train_precision"]}')

        print(f'Accurracy: {cv["train_accuracy"]}')

        print(f'Recall: {cv["train_recall"]}')

        print(f'ROC_AUC: {cv["train_roc_auc"]}')

        

        print('Test score:')

        print(f'Precision: {cv["test_precision"]}')

        print(f'Accurracy: {cv["test_accuracy"]}')

        print(f'Recall: {cv["test_recall"]}')

        print(f'ROC_AUC: {cv["test_roc_auc"]}')

        

    def output(self):

        data_temp = self.prediction_data.copy()

        

        data_temp = self.preprocess_data_temp(data_temp)

        

        p = self.model.predict(data_temp)

        

        csv_dict = {'ID': self.prediction_data.ID, 'Default': p}

        

        csv = pd.DataFrame(csv_dict)

        

        csv.to_csv('output.csv', index=False, encoding='utf8')

        

    def preprocess_data_temp(self, data_temp):

        if self.to_dummy:

            data_temp = pd.get_dummies(columns=self.to_dummy, data=data_temp)

            

        if self.to_drop:

            data_temp = data_temp.drop(columns=self.to_drop)

        

        if self.to_binarize:

            binarizer = preprocessing.Binarizer()

            data_temp[self.to_binarize] = binarizer.fit_transform(data_temp[self.to_binarize])

        

        return data_temp
def correlation_plot(dataframe):

    correlations = dataframe.corr(method='pearson')



    fig, ax = plt.subplots(figsize=(14,10))

    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',

                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})

    plt.show();

data = pd.read_csv('../input/train.csv')

prediction_data = pd.concat([pd.read_csv('../input/valid.csv'), pd.read_csv('../input/test.csv')])
data.describe()
data = data.rename(columns={'PAY_0': 'PAY_1'})

prediction_data = prediction_data.rename(columns={'PAY_0': 'PAY_1'})
plt.figure(figsize=(10, 6))



sns.countplot(x='default payment next month', data=data)
non_default = len(data[data['default payment next month'] == 0])

default = len(data[data['default payment next month'] == 1])



print(f'Percentage of non-defaulters: {non_default/(non_default+default)}')

print(f'Percentage of defaulters: {default/(non_default+default)}')
plt.figure(figsize=(10, 6))



sns.countplot(x='EDUCATION', hue='default payment next month', data=data)
data.groupby('EDUCATION').count()
plt.figure(figsize=(10, 6))



sns.countplot(x='SEX', hue='default payment next month', data=data)
plt.figure(figsize=(10, 6))



sns.countplot(x='MARRIAGE', hue='default payment next month', data=data)
data.groupby('MARRIAGE').count()
pay_x = ['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']



data[pay_x].describe()
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(14, 10))



sns.countplot(x='PAY_1', hue='default payment next month', data=data, ax=axs[0][0])

sns.countplot(x='PAY_2', hue='default payment next month', data=data, ax=axs[0][1])

sns.countplot(x='PAY_3', hue='default payment next month', data=data, ax=axs[0][2])

sns.countplot(x='PAY_4', hue='default payment next month', data=data, ax=axs[1][0])

sns.countplot(x='PAY_5', hue='default payment next month', data=data, ax=axs[1][1])

sns.countplot(x='PAY_6', hue='default payment next month', data=data, ax=axs[1][2])
f = ['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',

    'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']



data[f].head(20)
data.loc[data.index == 18][f]
data.loc[data.index == 20989][f]
data.loc[data.index == 17][f]
data.columns
correlation_plot(data)
f = ['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',

    'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'default payment next month', 'LIMIT_BAL']



correlation_plot(data[f])
temp = data.copy()
temp = data.copy()



pay_x = ['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']



for pay in pay_x:

    mask = (temp[pay] >= 0)

    temp = temp.loc[mask]

    

correlation_plot(temp[f])
learning = Learning()

learning.data = data

learning.prediction_data = prediction_data



learning.split()
learning.model = ensemble.RandomForestClassifier(n_estimators=10)

learning.train_model()

learning.score()
learning.model = neighbors.KNeighborsClassifier()

learning.train_model()

learning.score()
learning.model = neural_network.MLPClassifier()

learning.train_model()

learning.score()
mask = (data['default payment next month'] == 1)



defaulters_data = data[mask]



defaulters_amount = len(defaulters_data)



mask = (data['default payment next month'] == 0)



non_defaulters_data = data.loc[mask]



non_defaulters_data = non_defaulters_data.sample(defaulters_amount)
new_data = pd.concat([defaulters_data, non_defaulters_data])
def modify_data(dataframe):

    f = ['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']



    for column in f:

        mask = (dataframe[column] < 0)

        dataframe.loc[mask, column] = 0

        

    return dataframe

        

new_data = modify_data(new_data)
learning = Learning()



learning.data = new_data

learning.prediction_data = prediction_data



learning.to_drop = ['ID']

learning.to_dummy = ['EDUCATION', 'SEX', 'MARRIAGE']

learning.split()
learning.model = ensemble.RandomForestClassifier(n_estimators=10)

learning.train_model()

learning.score()
learning.train_model(complete=True)

learning.cross_validate()
learning.model = ensemble.RandomForestClassifier(n_estimators=200, max_leaf_nodes=200, min_samples_split=42,

                                                 n_jobs=-1)

learning.train_model()

learning.score()
learning.cross_validate()
learning.train_model(complete=True)

learning.score()
learning.train_model(complete=False)

learning.score()
probs = learning.model.predict_proba(learning.test['X'])

probs = probs[:, 1]



fpr, tpr, thresholds = metrics.roc_curve(learning.test['y'], probs)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(fpr, tpr, marker='.')
learning.prediction_data = modify_data(prediction_data)

learning.output()