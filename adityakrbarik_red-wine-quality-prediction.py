# Importing Packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_rows', 3000)

pd.set_option('display.max_columns', 1500)



from scipy.stats import skew

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import MinMaxScaler

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import CategoricalNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_auc_score, roc_curve, classification_report, auc

from sklearn.model_selection import KFold, cross_val_score



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head(10)
cols = df.columns

print(f'Name of the Columns in the Dataset are : \n\n {np.array(cols)}')
df.info()
Rating_Count = df['quality'].value_counts().sort_index()

fig, ax = plt.subplots(figsize = [20, 10])

fig.subplots_adjust(top = 0.93)

fig.suptitle('Counts of Each Rating of Red Wine in the Dataset', size = 25, fontweight = 'bold')

sns.barplot(Rating_Count.index, Rating_Count.values, ax = ax)

for index_, value_ in enumerate(Rating_Count):

    ax.text(index_, value_ + 12, str(value_), color = 'black', fontweight = 'bold', size = 15)

ax.set_xlabel('Quality Rating of Red Wine (from 0 to 10)', size = 20)

ax.set_ylabel('Frequency', size = 20)

plt.show()
df['quality_new'] = df['quality'].apply(lambda x: 'Good' if x >= 7 else 'Bad')

quality = df.pop('quality') # keeping aside the original quality column as Backup.
fig, ax = plt.subplots(figsize = [18, 8])

fig.suptitle('Proportion of different Wine Quality in the Data', size = 25, fontweight = 'bold')

ax.pie(df['quality_new'].value_counts(), labels = list(df['quality_new'].value_counts().index), 

       autopct = '%1.1f%%', textprops = {'fontsize': 22}, pctdistance = 0.5)



# Draw Circle

centre_circle = plt.Circle((0,0),0.75,fc = 'white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)



# Equal Aspect Ratio ensures that Pie is drawn as a Circle

ax.axis('equal') 

plt.show()
X, Y = df.drop('quality_new', axis = 1), df['quality_new']
Y.replace({'Good' : 1, 'Bad' : 0}, inplace = True)
X.describe()
fig, axs = plt.subplots(nrows = 4, ncols = 3, figsize = [6.4 * 3, 4.8 * 4])

fig.subplots_adjust(hspace = .15, wspace = .15, top = 0.93)

axs[-1, -1].axis('off')

axs = axs.ravel()

fig.suptitle('Boxplot for different Physicochemical Variables in the Data', size = 25, fontweight = 'bold', y = 0.98)

for i, col in enumerate(X.columns):

    sns.boxplot(y = col, data = X, ax = axs[i])

    axs[i].set_xlabel(col, size = 15)

plt.show()
# Function to return Indices of Outliers 

def indicies_of_outliers(x): 

    Q1, Q3 = x.quantile([0.25, 0.75]) 

    IQR = Q3 - Q1

    lower_limit = Q1 - (1.5 * IQR)

    upper_limit = Q3 + (1.5 * IQR)

    return np.where((x > upper_limit) | (x < lower_limit))[0] 
outlier_indices = set()

for col in X.columns:

    outlier_indices = set(outlier_indices | set(indicies_of_outliers(X[col])))

print(f'Percentage of Outlier Removal is {len(outlier_indices)/X.shape[0]*100:.2f} %.')

X.drop(outlier_indices, axis = 0, inplace = True)

Y.drop(outlier_indices, axis = 0, inplace = True)
fig, axs = plt.subplots(nrows = 4, ncols = 3, figsize = [6.4 * 3, 4.8 * 4])

fig.subplots_adjust(hspace = .25, wspace = .15, top = 0.93)

axs[-1, -1].axis('off')

axs = axs.ravel()

fig.suptitle('Distribution plot for different Physicochemical Variables in the Original Data', size = 25, fontweight = 'bold', y = 0.98)

for i, col in enumerate(X.columns):

    sns.distplot(X[col], ax = axs[i])

    axs[i].set_xlabel(f'{col} with skewness : {skew(X[col].dropna()):.2f}', size = 15)

plt.show()
print(f'Number of Strictly Positive Values in each Column : \n\n{(X > 0).sum()}')
Power_Transform = PowerTransformer(method = 'yeo-johnson')

fig, axs = plt.subplots(nrows = 4, ncols = 3, figsize = [6.4 * 3, 4.8 * 4])

fig.subplots_adjust(hspace = .25, wspace = .15, top = 0.93)

axs[-1, -1].axis('off')

axs = axs.ravel()

fig.suptitle('Distribution plot for different Physicochemical Variables in the Transformd Data', size = 25, fontweight = 'bold', y = 0.98)

for i, col in enumerate(X.columns):

    if abs(skew(df[col])) > 0.75:

        Power_Transform.fit(np.array(X[col]).reshape(-1,1))

        X[col] = Power_Transform.transform(np.array(X[col]).reshape(-1,1))

    sns.distplot(X[col], ax = axs[i])

    axs[i].set_xlabel(f'{col} with skewness : {skew(X[col].dropna()):.2f}', size = 15)

plt.show()
scaler = MinMaxScaler()

fig, axs = plt.subplots(nrows = 4, ncols = 3, figsize = [6.4 * 3, 4.8 * 4])

fig.subplots_adjust(hspace = .25, wspace = .15, top = 0.93)

axs[-1, -1].axis('off')

axs = axs.ravel()

fig.suptitle('Distribution plot for different Physicochemical Variables in the Scaled Transformed Data', size = 25, fontweight = 'bold', y = 0.98)

for i, col in enumerate(X.columns):

    X[col] = scaler.fit_transform(np.array(X[col]).reshape(-1,1))

    sns.distplot(X[col], ax = axs[i])

    axs[i].set_xlabel(f'{col} with skewness : {skew(X[col].dropna()):.2f}', size = 15)

plt.show()
plt.figure(figsize = [20, 10])

sns.heatmap(X.corr(), annot = True, vmin = -1, vmax = 1)

plt.show()
corr_cols = [('fixed acidity', 'citric acid'), ('fixed acidity', 'density'), ('free sulfur dioxide', 'total sulfur dioxide'), 

             ('fixed acidity' , 'pH'), ('volatile acidity', 'citric acid'), ('density', 'alcohol')]

fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = [6.4 * 3, 4.8 * 3])

fig.subplots_adjust(hspace = .15, wspace = .15, top = 0.93)

fig.suptitle('Scatter Plots of the pairs of Variables having Significant Correlation among themselves', size = 25, fontweight = 'bold')

axs = axs.ravel()

for i, corr_col in enumerate(corr_cols):

    sns.scatterplot(x = corr_col[0], y = corr_col[1], data = X, ax = axs[i])

plt.show()
fig_1, axs = plt.subplots(nrows = 4, ncols = 3, figsize = [6.4 * 3, 4.8 * 4])

fig_1.subplots_adjust(hspace = .15, wspace = .15, top = 0.93)

axs[-1, -1].axis('off')

axs = axs.ravel()

fig_1.suptitle('Boxplot for different Physicochemical Variables in the Final Data', size = 25, fontweight = 'bold', y = 0.98)

for i, col in enumerate(X.columns):

    sns.boxplot(y = col, data = X, ax = axs[i])

    axs[i].set_xlabel(col, size = 15)

plt.show()
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25)
def Model_Fit(Predictor, Response, Model, Model_Params = None, Imbalanced_Class = False, Imbalanced_Classification_Sampling_Technique = None, 

              Test_Percentage = None, Stratification = False, Grid_Search = False, Print_Scores = False):

    """

    This function will help to fit a given Classification Model with given Predictor and Response variables .

    

    Parameters

    ----------

    Predictor : pandas.core.frame.DataFrame <Required>

        A pandas dataframe containing all the predictor variables.

        

    Response : pandas.core.series.Series <Required>

        A pandas series object containing the response variable.

        

    Model : str <Required>

        The name of the model (without parentheses) in string format.

        

    Model_Params : dict , default = None

        A dictionary conatining keys as the exact name of model (without parentheses) 

        and the values will be the parameters of the correspondoing model. 

        It is required while using 'Grid_Search = true'.

        An example is :

                        model_params = {

                            'SVC' : {

                                    'C'                : list(np.arange(0.5, 1.5, 0.1)), 

                                    'kernel'           : ['linear', 'poly', 'rbf', 'sigmoid'], 

                                    'gamma'            : list(np.arange(0.5, 1.5, 0.1)), 

                                    'class_weight'     : ['balanced', None]

                                    }

                            }

                            

    Imbalanced_Class : bool, default = False

        Whether to apply Imbalanced Classification Sampling Technique or not.

        

    Imbalanced_Classification_Sampling_Technique = str, default = None

        The name of the Imbalanced Classification Sampling Technique (without parentheses) in string format.

        It is required while using 'Imbalanced_Class = true'.

        

    Test_Percentage : float, default = None

         A fraction of the dataset which will be used as test set to evaluate the prediction of the model.

         

    Stratification : bool, default = False

        Whether to use Stratified Sampling for splliting the data into train and test or not.

        

    Grid_Search : bool, default = False

        Whether to use GridSearchCV to select the best model params out the given model params or not.

    

    Print_Scores : bool, default = False

        Whether to print the different evaluation metrics for the prediction of the model or not.

    """

    X, Y, test_per = Predictor, Response, float(Test_Percentage)

    

    if Test_Percentage == None:

        while(True):

            test_per = input('Please give the percentage of Test Data \n (Value must be between 0 and 1) : ') 

            if float(test_per) > 0 and float(test_per) < 1:

                Test_Percentage = test_per

                break

            else:

                continue

    

    Models = {

        'LogisticRegression' : LogisticRegression(), 'KNeighborsClassifier' : KNeighborsClassifier(), 

        'DecisionTreeClassifier' : DecisionTreeClassifier(), 'RandomForestClassifier' : RandomForestClassifier(), 'SVC' : SVC(), 

        'LinearDiscriminantAnalysis' : LinearDiscriminantAnalysis(), 'QuadraticDiscriminantAnalysis' : QuadraticDiscriminantAnalysis(), 

        'CategoricalNB' : CategoricalNB(), 'SGDClassifier' : SGDClassifier()

    }

    Imbalanced_Classification_Sampling_Techniques = {

        'RandomUnderSampler' : RandomUnderSampler(), 'RandomOverSampler' : RandomOverSampler(), 'SMOTE' : SMOTE(), 'ADASYN' : ADASYN()

    }

    if Imbalanced_Class:

            if Imbalanced_Classification_Sampling_Technique in Imbalanced_Classification_Sampling_Techniques:

                imb_ = Imbalanced_Classification_Sampling_Techniques[Imbalanced_Classification_Sampling_Technique]

                X, Y = imb_.fit_resample(X, Y)

            else:

                print('Please Enter a Valid Imbalanced Classification Sampling Techniques. '

                      f'Available Techniques are : \n\t {list(Imbalanced_Classification_Sampling_Techniques.keys())}')

    if Stratification:

        Split_ = StratifiedShuffleSplit(n_splits = 1, test_size = test_per)

        for Train_, Test_ in Split_.split(X, Y):

            X_Train, X_Test = X.iloc[Train_].reset_index(drop = True), X.iloc[Test_].reset_index(drop = True)

            Y_Train, Y_Test = Y.iloc[Train_].reset_index(drop = True), Y.iloc[Test_].reset_index(drop = True)

    else:

        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = test_per)

            

    if Model in Models.keys():

        kf = KFold(n_splits = 10, shuffle = True).get_n_splits(pd.concat((X_Train, Y_Train), axis = 1).values)

        if Grid_Search:

            model_ = GridSearchCV(Models[Model], Model_Params[Model], scoring = 'roc_auc', cv = kf)

        else:

            model_ = Models[Model]

        model_.fit(X_Train, Y_Train)

        Y_Pred = model_.predict(X_Test)



        Area_Under_ROC_Curve = roc_auc_score(Y_Test , Y_Pred)



        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = [20, 6])

        fig.subplots_adjust(wspace = .25, top = 0.8)

        axs = axs.ravel()

        fig.suptitle(f'For the Predictions using {Model} Model', size = 25, fontweight = 'bold')

        fpr, tpr, _ = roc_curve(Y_Test , Y_Pred)

        axs[0].plot(fpr, tpr, 'k-', label = f'AUC : {roc_auc_score(Y_Test , Y_Pred)}')

        axs[0].plot([0, 1], [0, 1], 'k--')

        axs[0].set_xlim([-0.05, 1.05])

        axs[0].set_ylim([-0.05, 1.05])

        axs[0].set_xlabel('False Positive Rate', size = 15)

        axs[0].set_ylabel('True Positive Rate', size = 15)

        axs[0].set_title('Receiver Operating Characteristic (ROC) Curve', size = 20)

        axs[0].legend(loc = 'lower right')



        sns.heatmap(pd.DataFrame(confusion_matrix(Y_Test , Y_Pred)), annot = True, fmt = 'd', cmap = 'Blues', ax = axs[1])

        axs[1].set_xlabel('Predicted Class', size = 15)

        axs[1].set_ylabel('Actual Class', size = 15)

        axs[1].set_title('Confusion Matrix', size = 20)

        plt.show()

        

        if Print_Scores:

            print(

                'For the Final Model, \n\n'

                f' Accuracy   :  {accuracy_score(Y_Test , Y_Pred)}\n'

                f' Precision  :  {precision_score(Y_Test , Y_Pred)}\n'

                f' Recall     :  {recall_score(Y_Test , Y_Pred)}\n'

                f' F1 Score   :  {f1_score(Y_Test , Y_Pred)}\n\n'

                'and a detail Classification Report is given below: \n\n'

                f'{classification_report(Y_Test, Y_Pred, target_names = ["Good Wine (1)", "Bad Wine (0)"], digits = 8)}'

            )

        Output = {'Model Name'                                     :   Model, 

                  'Fitted Model'                                   :   model_, 

                  'Imbalanced Classification Sampling Technique'   :   Imbalanced_Classification_Sampling_Technique, 

                  'Area Under ROC Curve'                           :   Area_Under_ROC_Curve}

        return(Output)

    else:

        print(f'Please Enter a Valid Model Name. Available Models are : \n\t {list(Models.keys())}')
Models = ['LogisticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier', 'SVC', 

          'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis', 'SGDClassifier']
Imbalanced_Classification_Sampling_Techniques = ['RandomUnderSampler', 'RandomOverSampler', 'SMOTE', 'ADASYN']
Model_Params = {

    'LogisticRegression'              :   {

                                           'C'                : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 

                                           'l1_ratio'         : list(np.arange(0.0, 1.1, 0.1)), 

                                           'penalty'          : ['l1', 'l2', 'elasticnet']

                                          }, 

    'KNeighborsClassifier'            :   {

                                           'n_neighbors'      : list(range(2, 10, 1)), 

                                           'weights'          : ['uniform', 'distance'], 

                                           'algorithm'        : ['auto', 'ball_tree', 'kd_tree', 'brute'], 

                                           'p'                : list(range(0, 3, 1))

                                          }, 

    'DecisionTreeClassifier'          :   {

                                           'criterion'        : ['gini', 'entropy'], 

                                           'splitter'         : ['best', 'random'],  

                                           'class_weight'     : ['balanced', None]

                                          }, 

    'RandomForestClassifier'          :   {

                                           'n_estimators'     : list(range(100, 400, 100)), 

                                           'criterion'        : ['gini'], 

                                           'class_weight'     : ['balanced', None]

                                          }, 

    'SVC'                             :   {

                                           'C'                : list(np.arange(0.5, 1.5, 0.1)), 

                                           'kernel'           : ['linear', 'poly', 'rbf', 'sigmoid'], 

                                           'gamma'            : list(np.arange(0.5, 1.5, 0.1)), 

                                           'class_weight'     : ['balanced', None]

                                          }, 

    'LinearDiscriminantAnalysis'      :   {

                                           'solver'           : ['svd', 'lsqr', 'eigen'], 

                                           'shrinkage'        : ['auto', None]

                                          }, 

    'QuadraticDiscriminantAnalysis'   :   {}, 

    'SGDClassifier'                   :   {

                                           'loss'             : ['hinge', 'log', 'squared_hinge', 'modified_huber', 'perceptron'], 

                                           'penalty'          : ['l2', 'l1', 'elasticnet'], 

                                           'class_weight'     : ['balanced', None], 

                                           'early_stopping'   : [True], 

                                           'n_iter_no_change' : [5]

                                          }

}
Final_Model_Name, Final_Fitted_Model, Final_IMB, Final_AUC = None, None, None, 0

for Model in Models:

    for IMB_Sampling_Technique in Imbalanced_Classification_Sampling_Techniques:

        print(f'Model : {Model} \nImbalanced Classification Sampling Technique : {IMB_Sampling_Technique} \n\n')

        Model_Info = Model_Fit(X, Y, Test_Percentage = 0.25, Model = Model, Model_Params = Model_Params, Imbalanced_Class = True, 

                               Imbalanced_Classification_Sampling_Technique = IMB_Sampling_Technique, 

                               Stratification = True, Grid_Search = False, Print_Scores = False)

        Model_Name_, Fitted_Model_, IMB_Sampling_Technique_Name, AUC_ = Model_Info.values()

        if AUC_ > Final_AUC: 

            Final_Model_Name, Final_Fitted_Model, Final_IMB, Final_AUC = Model_Name_, Fitted_Model_, IMB_Sampling_Technique_Name, AUC_
Final_Model_Info = Model_Fit(X, Y, Test_Percentage = 0.25, Model = Final_Model_Name, Model_Params = Model_Params, 

                             Imbalanced_Class = True, Imbalanced_Classification_Sampling_Technique = Final_IMB, 

                             Stratification = True, Grid_Search = True, Print_Scores = True)
def calc_auc_score(model, n_split):

    kf = KFold(n_splits = n_split, shuffle = True).get_n_splits(pd.concat((X, Y), axis = 1).values)

    auc = cross_val_score(model, X, Y, scoring = "roc_auc", cv = kf)

    return(auc)
auc_score = calc_auc_score(model = Final_Model_Info["Fitted Model"], n_split = 10)
print(

    'So, for the Final Model : \n\n'

    f'\t Imbalanced Classification Sampling Technique   :   {Final_Model_Info["Imbalanced Classification Sampling Technique"]}\n'

    f'\t Model Name                                     :   {Final_Model_Info["Model Name"]}\n'

    f'\t Best Set of Model Parameters                   :   {Final_Model_Info["Fitted Model"].best_params_}\n'

    f'\t Area Under ROC Curve                           :   {Final_Model_Info["Area Under ROC Curve"]}\n'

    f'\t Mean of Cross Validation Score ("roc_auc")     :   {auc_score.mean()}\n'

    )