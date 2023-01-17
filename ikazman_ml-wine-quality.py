import os

import time

import warnings



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import classification_report



import lightgbm as lgb



warnings.filterwarnings('ignore')
np.random.seed(42)
class DataExplorer:

    """

    Reading dataset

    """

    def __init__(self, filedir, filename):

        self.data = pd.read_csv(os.path.join(filedir, filename))

        

    def grouper_quality(self, row):

        """

        Enlarging groups of target features

        """

        quality = row['quality']



        if quality < 5:

            return 3



        elif quality > 6:

            return 1



        else:

            return 2

        

    def generalization(self):

        """

        Greate enlarged groups of target features: third, second and first class wines

        """

        data = self.data.copy()

        data['gen_quality'] = self.data.apply(self.grouper_quality, axis=1)



        return data.drop('quality', axis=1)

    

    def binomizator(self):

        """

        Binominaizing target features

        """

        data = self.data.copy()

        data['bi_quality'] = self.data.quality.apply(lambda x: 1 if x >= 6 else 0)



        return data.drop('quality', axis=1)



    class Reporter():

        """

        Collecting perfomance data and bulids report

        """

        def __init__(self, data, target, features_dict, models, binomial=False):

            """

            Instances for Reporter

            """



            self.final_report = None

            self.best_estimator = []

            self.predictions = []

            self.data = data

            self.target = target

            self.models = models

            self.binomial = binomial

            self.score = f1_score

            self.scoring = 'f1_micro'

            self.random_state = 42

            self.features_dict = features_dict

            self.folds = 5



        def metrics_plot(self, model, model_title, features_valid, target_valid):

            """

            Displays the PR curve and ROC curve

            """



            probabilities_valid = model.predict_proba(features_valid)

            precision, recall, thresholds = precision_recall_curve(target_valid, probabilities_valid[:, 1])

            fpr, tpr, thresholds = roc_curve(target_valid, probabilities_valid[:, 1])



            fig, ax = plt.subplots(1, 2, figsize=(15, 6))

            fig.subplots_adjust(hspace=0.4, wspace=0.4)



            sns.lineplot(recall, precision, drawstyle='steps-post', ax=ax[0])

            ax[0].set_xlabel('Recall')

            ax[0].set_ylabel('Precision')

            ax[0].set_ylim([0.0, 1.05])

            ax[0].set_xlim([0.0, 1.0])

            ax[0].set_title('Precision-Recall Curve ' + model_title)



            sns.lineplot(fpr, tpr, ax=ax[1])

            ax[1].plot([0, 1], [0, 1], linestyle='--')

            ax[1].set_xlim(0, 1)

            ax[1].set_ylim(0, 1)

            ax[1].set_xlabel('False Positive Rate')

            ax[1].set_ylabel('True Positive Rate')

            ax[1].set_title('ROC-curve ' + model_title)



        def auc_roc(self, model, features_valid, target_valid):

            """

            Calculating ROC-AUC

            """



            probabilities_valid = model.predict_proba(features_valid)

            probabilities_one_valid = probabilities_valid[:, 1]

            auc_roc = roc_auc_score(target_valid, probabilities_one_valid)



            return auc_roc



        def grid_search(self, model, param_grid, x_features, y_features):

            """

            GridSearchCV

            """

            kfold = KFold(n_splits=self.folds, shuffle=True,

                          random_state=self.random_state)

            grid_model = GridSearchCV(model, param_grid=param_grid,

                                      scoring=self.scoring, cv=kfold,

                                      verbose=1, n_jobs=-1, )

            grid_model.fit(x_features, y_features)

            best_estimator = grid_model.best_estimator_

            return best_estimator



        def data_spliter(self, features):

            """

            Splitting data into training and test in a ratio of 60:40

            """

            x_train, x_test, y_train, y_test = train_test_split(self.data[features], 

                                                                self.data[self.target], 

                                                                train_size=0.6, 

                                                                stratify=self.data[self.target],

                                                                random_state=self.random_state)



            return x_train, y_train, x_test, y_test



        def reporter(self):



            started = time.time()

            report = []

            estimators = []

            predictions = []

            score_name = str(self.score).split(' ')[1]

            models = self.models



            for key in self.features_dict:



                features = self.features_dict[key]



                print('Features set - ', key)



                x_train, y_train, x_test, y_test = self.data_spliter(features)

                

                x_train = np.log(x_train, where=x_train>0)

                x_test = np.log(x_test, where=x_test>0)



                scaler = StandardScaler()



                x_train = scaler.fit_transform(x_train)

                x_test = scaler.transform(x_test)



                print('Dataset was splitted into training and test in a ratio of 60:40 and scaled using StandardScaler. \n')

                print('Shapes: \n')

                print('- train ', x_train.shape, y_train.shape)

                print('- test ', x_test.shape, y_test.shape)

                print('\n')



                print('Report: ')



                for model in models:

                    started_local = time.time()

                    print('\n', model[0], '\n')

                    grid_search = self.grid_search(model[1], model[2], x_train, y_train)

                    print(grid_search)

                    ended_local = time.time()

                    predicted_test = np.ravel(grid_search.predict(x_test))

                    test_score = self.score(y_test, predicted_test, average='micro')



                    report.append((model[0], test_score, ended_local-started_local, key))

                    estimators.append((model[0], grid_search))

                    predictions.append((model[0], predicted_test))

                    if self.binomial == True:

                        self.metrics_plot(grid_search, model[0], x_test, y_test)

                    print('\n', 'Classification report for ' + model[0], '\n\n', classification_report(y_test, predicted_test))



                print('--------------------------------------------------------------------')

                print('--------------------------------------------------------------------')

                print('--------------------------------------------------------------------')

                print('--------------------------------------------------------------------')

            self.final_report = pd.DataFrame(report, columns=['model', score_name + '_test', 'seconds_to_fit', 'features_key'])

            self.best_estimator = pd.DataFrame(estimators, columns=['model', 'grid_params'])

            self.predictions = pd.DataFrame(predictions, columns=['model', 'test_predictions'])

            ended = time.time()

            print('Cross-validation training and parameter search completed in {} sec.'.format(round(ended-started, 2)))
explorer = DataExplorer('/kaggle/input/red-wine-quality-cortez-et-al-2009/', 'winequality-red.csv')
dict_of_features_combinations = {'all_features': ['fixed acidity', 'volatile acidity', 'citric acid', 

                                                   'residual sugar', 'chlorides', 'free sulfur dioxide', 

                                                   'total sulfur dioxide', 'density', 'pH', 'sulphates', 

                                                   'alcohol'],

                                 

                                 'most_important': ['volatile acidity', 'residual sugar', 'sulphates', 'alcohol']}
models_list = []
lg = lgb.LGBMClassifier(random_state=42, class_weight='balanced')



param_grid = {'boosting_type': ['gbdt'],

              'num_leaves': [10, 20, 30],

              'num_iterations': [600],

              'learning_rate': [0.01, 0.0001],

              'max_depth': [10, 30, 100]}



models_list.append(('LGBMClassifier', lg, param_grid))
explorer.data
multi_report = explorer.Reporter(explorer.data, 'quality', dict_of_features_combinations, models_list)
multi_report.reporter()
multi_report.final_report
df_three_classed = explorer.generalization()
df_three_classed
gen_report = explorer.Reporter(df_three_classed, 'gen_quality', dict_of_features_combinations, models_list)
gen_report.reporter()
gen_report.final_report
df_binomial = explorer.binomizator()
bi_report = explorer.Reporter(df_binomial, 'bi_quality', dict_of_features_combinations, models_list, binomial=True)
bi_report.reporter()
bi_report.final_report