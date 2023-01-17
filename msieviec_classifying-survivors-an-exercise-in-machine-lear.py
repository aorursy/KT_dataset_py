import numpy as np

import pandas as pd

from statistics import mode



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



full_data = pd.concat([train.drop(columns = 'Survived'), test], 

                       axis = 0, 

                       sort = False)



full_data.head()
full_data.describe()
full_data.shape, train.shape
len(full_data['PassengerId'].unique())
## missing

full_data.isnull().sum()
# fill age by means of sex, class

full_data[['Age','Fare']] = (full_data

                    .groupby(['Sex', 'Pclass'])[['Age','Fare']]

                    .transform(lambda x: 

                        x.fillna(round(np.mean(x)))))



full_data['Embarked'].unique()
# impute most common port

full_data['Embarked'] = (full_data.Embarked

                         .fillna(mode(full_data.Embarked)))
full_data['Cabin'].unique()
# deck (if any)

full_data['Deck'] = (full_data.Cabin

                     .str.extract('([A-Z])(?=\d*$)', 

                                  expand = True)

                     .fillna('X'))
# cabin letter (if any)

full_data['Cabin'] = (full_data.Cabin

                      .str.extract('(\d+)$', expand = True)

                      .fillna(0))

full_data['Cabin'] = full_data['Cabin'].apply(int) # str to int
# title

title_extract = ('(Mr\.|Master|Mrs\.|Miss\.|Rev\.|Capt\.|Col\.|Major\.)')

title_dict = {'Capt\.|Major\.|Col\.' : 'Officer'}

full_data['Title'] = (full_data.Name

                      .str.extract(title_extract)

                      .replace(title_dict, 

                               regex = True)

                      .fillna('Other'))
full_data.groupby('Title').size()
full_data['Ticket'].unique()
# ticketnumber

full_data['TicketNumber'] = (full_data.Ticket

                             .str.extract('(\d+)$', expand = True))

full_data.loc[full_data.TicketNumber.notnull(), 'TicketNumber'] = (full_data

                                                                   .loc[full_data.TicketNumber.notnull(),

                                                                        'TicketNumber']

                                                                   .astype(int)) # str to int



# 4 missing tickets

(full_data.Ticket

 .str.extract('(\d+)$', expand = True)

 .isnull()

 .sum())
full_data['TicketNumber'] = (full_data

                             .groupby(['Pclass', 'Embarked'])['TicketNumber']

                             .transform(lambda x: x.fillna(np.mean(x))))
# new training set for probabilities and visualizations

train_for_vis = pd.concat([train.Survived, full_data[:train.shape[0]]], axis = 1)



import seaborn as sns

import matplotlib.pyplot as plt

sns.set_palette('cool')



# without family more likely to die

sns.catplot(x = 'Parch', 

            y = 'Survived', 

            data = train_for_vis,

            height = 5,

            aspect = 1.5,

            kind = 'bar')

plt.title('Figure 1: Survival by Number of Parents/Children')
sns.catplot(x = 'SibSp', 

            y = 'Survived', 

            data = train_for_vis,

            height = 5,

            aspect = 1.5,

            kind = 'bar')

plt.title('Figure 2: Survival by Number of Siblings/Spouses')
# survival by deck

sns.catplot(x = 'Deck', 

            y = 'Survived', 

            data = train_for_vis, 

            kind = 'bar',

            height = 5,

            aspect = 1.85,

            order = sorted(list(train_for_vis['Deck'].unique())))

plt.title('Figure 3: Survival by Deck')
# survival by sex

sns.catplot(x = 'Sex', 

            y = 'Survived', 

            data = train_for_vis,

            kind = 'bar')

plt.title('Figure 4: Survival by Sex')
# survival by age

g = sns.FacetGrid(train_for_vis, 

                  hue = 'Survived',

                  height = 5,

                  aspect = 1.5,

                  palette = 'cool')

g.map(sns.kdeplot, 'Age', shade = True).add_legend()

plt.title('Figure 5: Survival by Age')
g = sns.FacetGrid(train_for_vis, 

                  hue = 'Survived',

                  height = 5,

                  aspect = 1.5,

                  palette = 'cool')

g.map(sns.kdeplot, 'TicketNumber', shade = True).add_legend()

plt.title('Figure 6: Survival by TicketNumber')
g = sns.FacetGrid(train_for_vis, 

                  hue = 'Survived',

                  height = 5,

                  aspect = 1.5,

                  palette = 'cool')

g.map(sns.kdeplot, 'Cabin', shade = True).add_legend()

plt.title('Figure 7: Survival by Cabin Number')
corr_map = train_for_vis.corr(method = 'pearson')

corr_map = (corr_map

            .transform(lambda x: np.flip(x, 0), 

                       axis = 0))

mask = np.tri(corr_map.shape[0], k = -1).T



plt.figure(figsize = (12, 9))

sns.heatmap(corr_map, 

            annot = True, 

            cmap = 'GnBu_r', 

            mask = np.flip(mask, axis = 1),

            vmin = 0,

            vmax = 1)

plt.title("Figure 8: Pearson Correlation Coefficient")
corr_map = train_for_vis.corr(method = 'spearman')

corr_map = (corr_map

            .transform(lambda x: np.flip(x, 0), 

                       axis = 0))

mask = np.tri(corr_map.shape[0], k = -1).T



plt.figure(figsize = (12, 9))

sns.heatmap(corr_map, 

            annot = True, 

            cmap = 'GnBu_r', 

            mask = np.flip(mask, axis = 1),

            vmin = 0,

            vmax = 1)

plt.title("Figure 9: Spearman's Rho")
full_data['Sex'] = (full_data['Sex']

                    .map({'male' : 0, 'female' : 1}))
def ticket_transform(data):

    if data < 200000:

        data = 0

    elif 200000 <= data < 1000000:

        data = 1

    else:

        data = 2

    return data



full_data['TicketCat'] = (full_data['TicketNumber']

                          .transform(ticket_transform))
train_for_vis = pd.concat([train.Survived, full_data[:train.shape[0]]], axis = 1)



# make ordinal

ticket_by_prob = (train_for_vis

                 .groupby(['TicketCat'])['Survived']

                 .agg('mean')

                 .sort_values(ascending = True)

                 .index)

ticket_map = dict(zip(ticket_by_prob,

                    [x for x in range(len(ticket_by_prob))]))

full_data['TicketCat'] = full_data['TicketCat'].map(ticket_map)
(train_for_vis

    .groupby(['TicketCat'])['Survived']

    .agg('mean')

    .sort_values(ascending = True))
def age_transform(data):

    if data < 17:

        data = 0

    elif 17 <= data < 32:

        data = 1

    elif 32 <= data < 42:

        data = 2

    elif 42 <= data < 60:

        data = 3

    else:

        data = 4

    return data



full_data['AgeCat'] = (full_data['Age']

                       .transform(age_transform))

# make ordinal

train_for_vis = pd.concat([train.Survived, full_data[:train.shape[0]]], axis = 1)

age_by_prob = (train_for_vis

                 .groupby(['AgeCat'])['Survived']

                 .agg('mean')

                 .sort_values(ascending = True)

                 .index)

age_map = dict(zip(age_by_prob,

                    [x for x in range(len(age_by_prob))]))

full_data['AgeCat'] = full_data['AgeCat'].map(age_map)
full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch']



full_data['FamilyCat'] = (full_data['FamilySize']

                          .map({0 : 'Alone',

                                **dict.fromkeys([1, 2, 3], 'Small'),

                                **dict.fromkeys([4, 5, 6, 7, 8, 9, 10], 'Large')}))



train_for_vis = pd.concat([train.Survived, full_data[:train.shape[0]]], axis = 1)



# ordinal

fam_by_prob = (train_for_vis

                 .groupby(['FamilyCat'])['Survived']

                 .agg('mean')

                 .sort_values(ascending = True)

                 .index)

fam_map = dict(zip(fam_by_prob,

                    [x for x in range(len(fam_by_prob))]))

full_data['FamilyCat'] = full_data['FamilyCat'].map(fam_map)
# ordinal title

title_by_prob = (train_for_vis

                 .groupby(['Title'])['Survived']

                 .agg('mean')

                 .sort_values(ascending = True)

                 .index)



title_map = dict(zip(title_by_prob,

                     [x for x in range(len(title_by_prob))]))

full_data['Title'] = full_data['Title'].map(title_map)
# ordinal class

class_by_prob = (train_for_vis

                 .groupby(['Pclass'])['Survived']

                 .agg('mean')

                 .sort_values(ascending = True)

                 .index)

class_map = dict(zip(class_by_prob,

                     [x for x in range(len(class_by_prob))]))

full_data['Pclass'] = full_data['Pclass'].map(class_map)
# ordinal port

port_by_prob = (train_for_vis

                 .groupby(['Embarked'])['Survived']

                 .agg('mean')

                 .sort_values(ascending = True)

                 .index)

port_map = dict(zip(port_by_prob,

                    [x for x in range(len(port_by_prob))]))

full_data['Embarked'] = full_data['Embarked'].map(port_map)
# ordinal deck

deck_by_prob = (train_for_vis

                 .groupby(['Deck'])['Survived']

                 .agg('mean')

                 .sort_values(ascending = True)

                 .index)

deck_map = dict(zip(deck_by_prob,

                    [x for x in range(len(deck_by_prob))]))

full_data['Deck'] = full_data['Deck'].map(deck_map)

to_drop_cols = ['PassengerId', 'Name', 'Age',

                'SibSp', 'Parch', 'Ticket', 

                'TicketNumber', 'FamilySize']



full_data = full_data.drop(columns = to_drop_cols)

full_data.head()
train_clean = pd.concat([train['Survived'], 

                         full_data[:train.shape[0]]], axis = 1)

sns.pairplot(train_clean, 

             hue = 'Survived', 

             vars = list(train_clean.drop(columns = 'Survived').columns))

plt.suptitle('Figure 10: Pairplot of All Variables', y = 1.02)
to_drop_rows = [343, 679, 258, 737] # high fares
full_data[['Cabin', 'Fare']] = (full_data[['Cabin', 'Fare']]

                                .transform(lambda x: np.log(1+x)))
full_data.head()
train_clean = pd.concat([train['Survived'], 

                         full_data[:train.shape[0]]], axis = 1)

train_clean = train_clean.drop(index = to_drop_rows)

test_clean = full_data[(train_clean.shape[0] + len(to_drop_rows)):]

train_clean.shape, test_clean.shape  # minus 4 outliers
train.shape, test.shape
corr_map = train_clean.corr(method = 'spearman')

corr_map = (corr_map

            .transform(lambda x: np.flip(x, 0), 

                       axis = 0))

mask = np.tri(corr_map.shape[0], k = -1).T



plt.figure(figsize = (12, 9))

sns.heatmap(corr_map, 

            annot = True, 

            cmap = 'GnBu_r', 

            mask = np.flip(mask, axis = 1),

            vmin = 0,

            vmax = 1)

plt.title("Figure 11: Spearman's Rho")
from sklearn.model_selection import cross_val_predict, GridSearchCV

from sklearn.metrics import accuracy_score

from collections import namedtuple

from sklearn.utils import shuffle



shuffled = shuffle(train_clean, random_state = 42)

predictors = shuffled.drop(columns = 'Survived')

response = shuffled['Survived']



# accuracy assessment

def get_accuracy(model, resp):

    out = namedtuple('Output', 'Accuracy Predictions')

    pred = cross_val_predict(model, predictors, resp, cv = 10)

    acc = accuracy_score(resp, pred)

    return out(acc, pred)
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(C = 1.75,

                              solver = 'liblinear',

                              fit_intercept = True, 

                              max_iter = 10000)

acc_lr = get_accuracy(model_lr, response)

print(f'Logistic Regression classification accuracy was {acc_lr.Accuracy}')
model_lr.fit(predictors, response)

print('Logistic Regression Coefficients')

features = pd.DataFrame(dict(zip(list(train_clean.drop(columns = 'Survived')), 

                      list(model_lr.coef_[0]))), index = range(1)).sort_values(by = 0, axis = 1, ascending = False)

sns.catplot(data = features,

            kind = 'bar',

            height = 5,

            aspect = 1.5,

            palette = 'magma_r')

plt.title('Figure 12: Logistic Regression Coefficients')
from sklearn.gaussian_process import GaussianProcessClassifier

model_gpc = GaussianProcessClassifier(n_restarts_optimizer = 10,

                                      max_iter_predict = 100000,

                                      random_state = 5)

acc_gpc = get_accuracy(model_gpc, response)

print(f'Gaussian Process classification accuracy was {acc_gpc.Accuracy}')
from sklearn.neighbors import KNeighborsClassifier

model_kn = KNeighborsClassifier(n_neighbors = 9,

                                n_jobs = 2)

acc_kn = get_accuracy(model_kn, response)

print(f'K-Nearest Neighbors classification accuracy was {acc_kn.Accuracy}')
from sklearn.svm import SVC

model_svc = SVC(max_iter = 1000000, 

                C = 0.5,

                kernel = 'rbf',

                gamma = 'scale',

                probability = True,

                random_state = 5)

acc_svc = get_accuracy(model_svc, response)

print(f'Support Vector classification accuracy was {acc_svc.Accuracy}')
import xgboost as xgb

model_xgb = xgb.XGBClassifier(reg_alpha = .35,

                              reg_lambda = .6,

                              nthread = 2,

                              seed = 1)

acc_xgb = get_accuracy(model_xgb, response)

print(f'XGBoost classification accuracy was {acc_xgb.Accuracy}')
model_xgb.fit(predictors, response)

model_xgb.feature_importances_

features = pd.DataFrame(dict(zip(list(train_clean.drop(columns = 'Survived')), 

                      list(model_xgb.feature_importances_))), index = range(1)).sort_values(by = 0, axis = 1, ascending = False)

sns.catplot(data = features,

            kind = 'bar',

            height = 5,

            aspect = 1.5,

            palette = 'magma_r')

plt.title('Figure 13: XGBoost Feature Importances')
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(random_state = 5,

                                  max_depth = 6,

                                  max_features = 8,

                                  n_estimators = 500,

                                  n_jobs = 2)

acc_rf = get_accuracy(model_rf, response)

print(f'Random forest classification accuracy was {acc_rf.Accuracy}')
model_rf.fit(predictors, response)

model_rf.feature_importances_

features = pd.DataFrame(dict(zip(list(train_clean.drop(columns = 'Survived')), 

                      list(model_rf.feature_importances_))), index = range(1)).sort_values(by = 0, axis = 1, ascending = False)

sns.catplot(data = features,

            kind = 'bar',

            height = 5,

            aspect = 1.5,

            palette = 'magma_r')

plt.title('Figure 14: Random Forest Feature Importances')