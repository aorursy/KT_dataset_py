# Importing all required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

%matplotlib inline



from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import RFE



from statistics import mode

import statsmodels.api as sm

from sklearn import metrics

from sklearn.metrics import confusion_matrix



from sklearn import metrics

from sklearn.metrics import confusion_matrix, classification_report



import statsmodels.api as sm
#Common function for printing confusion matrix statistics

def print_CFstats(y_test, y_pred):

    CF = metrics.confusion_matrix(y_test, y_pred)

    print(CF)

    TP = CF[0,0]

    TN = CF[1,1]

    FP = CF[0,1]

    FN = CF[1,0]



    # accuracy

    print("Accuracy : ", round(metrics.accuracy_score(y_test, y_pred)*100,2),'%')



    # precision

    print("Precision : ", round(metrics.precision_score(y_test, y_pred)*100,3),'%')



    # recall/sensitivity

    print("Sensitivity / Recall: ", round(metrics.recall_score(y_test, y_pred)*100,2),'%')

def display_model_stats(curmodel, printop=False):

    #genrate a Dataframe extracting feature names and their p-values

    df = gen_pvaluesdf(curmodel)



    #Variance Inflation Factor for detecting Multicollinearity

    df_vif = calculate_vif(X_c)

    #Merge the 2 dataframes to create one df and sort on significance of pvalue along with VIF info

    #Also, display the R-square and Adjustd R-square

    df_p_vif = pd.merge(df_vif, df)



    if printop == True:



        print('='*33)

        print('| Num Features in the model: |', len(ChosenFeatureList))

        print('='*33)

        

        print('| R-squared:', round(curmodel.rsquared,3),

              ' | Adjusted R-squared:', round(curmodel.rsquared_adj,3), '|')

        print('='*48)



        df_p_vif.sort_values(by='pval', ascending=False)

        display(df_p_vif)
epsilon = 0.000001

veryLargeNum = 1/epsilon



def calculate_vif(input_data, showonlygtr3=False):

    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])

    x_vars=input_data

    xvar_names=input_data.columns

    for i in range(0,xvar_names.shape[0]):

        y=x_vars[xvar_names[i]] 

        x=x_vars[xvar_names.drop(xvar_names[i])]

        rsq=sm.OLS(y,x).fit().rsquared

        

        #To avoid % by 0 at runtime

        if (1-rsq) < epsilon:

            vif=veryLargeNum

        else:

            vif=round(1/(1-rsq),2)

            

        vif_df.loc[i] = [xvar_names[i], vif]

        df_2disp = vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)

    if showonlygtr3:

        return df_2disp.loc[df_2disp.Vif >= 3]

    else :

        return df_2disp
#geenrate a Dataframe extracting feature names and their p-values

def gen_pvaluesdf(statmodel):

    df_pvalues = pd.DataFrame(round(statmodel.pvalues,2).reset_index(name='pvalues'))

    varnames = np.array(df_pvalues.iloc[:, 0])

    pvalues  = np.array(df_pvalues.iloc[:, 1])

    df = pd.DataFrame(varnames, columns=['Var'])

    df['pval'] = pvalues

    df['Significant'] = np.where((df.pval < 0.05), 'Yes', 'No')

    return df
#Common functions which can be reused by callers to plot %wise graphs



def get_percentage_cnt(df, col):   

    t1 = pd.DataFrame(df.groupby(col)[col].count().rename('cnt%'))

    t1.reset_index(inplace=True)

    t1['cnt%'] = round((t1['cnt%'] * 100) / t1['cnt%'].sum(),2)

    return t1



def myplot(x, y, df):

    ax = sns.barplot(x=x, y=y, data=df)

    for p in ax.patches:

        ax.annotate('{0:.1f}%'.format(p.get_height()), (p.get_x()+0.1, p.get_height()))
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

fname = os.path.join(dirname, filename)



#Importing dataset

bc = pd.read_csv(fname)
#Examine the data

bc.head()
bc.drop(bc.columns[[0]], axis=1, inplace=True)

bc.drop(bc.columns[len(bc.columns)-1], axis=1, inplace=True)

bc.head()
print('='*20)

print('|', "{0:^15}".format('Statistics:'), '|')

print('='*85)

print('|', "{0:^25}".format('Column name'), '|', "{0:^25}".format('% Unique Values'),

      '|', "{0:^25}".format('%Missing values'), '|')

print('='*85)



for col in bc:

    print("| {0:^25}".format(col), '|', "{0:^25.2%}".format(round(bc[col].nunique()/len(bc[col]),2)), '|',

          "{0:^25.2%}".format(round(bc[col].isnull().sum()/len(bc[col]),2)), '|')



print('='*85)



bc.shape
cols_with_zerodata_inpercent = round(100*(bc.isnull().sum()/len(bc.index)), 2)



print("Columns with 100% missing data:")

colnames = bc.columns[cols_with_zerodata_inpercent==100].values

print("Num col:", len(colnames), colnames)



cols_with_nonuniq_values = bc.nunique()

colnamesnouniq = bc.columns[(cols_with_nonuniq_values <= 1)].values

print("Columns with just 1 unique value:", len(colnamesnouniq), colnamesnouniq)



print("#Duplicate Rows are :")

print(bc.duplicated().sum())

bc.drop_duplicates(keep=False, inplace=True)



bc.head(1)
#Get list of categorical Features

categorical_feature_list_orig = list(map(str, bc.columns[bc.dtypes == object]))



#Get list of continous Features

continous_feature_list_orig = list(map(str, bc.columns[bc.dtypes != object]))



print('='*21)

print('Total Features:', len(categorical_feature_list_orig) + len(continous_feature_list_orig))

print('='*21)

print('Categorical Features:')

print('='*21)

print(categorical_feature_list_orig)

print('='*19)

print('Continous Features:')

print('='*19)

print(continous_feature_list_orig)
print('='*82)

print('|', "{0:^23}".format('Name'),

    '|', "{0:^7}".format('Type'),

    '|', "{0:^6}".format('Min'),

    '|', "{0:^9}".format('Max'),

    '|', "{0:^5}".format('Mean'),

    '|', "{0:^5}".format('Medan'), 

    '|', "{0:^5}".format('Std_D'), '|'

   )

print('='*82)



for col in continous_feature_list_orig:

    print("| {0:^23}".format(col), '|',

          "{0:^7}".format(str(bc.loc[:,col].dtype)), '|', "{0:^6}".format(round(bc.loc[:,col].min(),2)), '|',

          "{0:^9}".format(round(bc.loc[:,col].max())), '|', "{0:^5}".format(round(bc.loc[:,col].mean(),2)), '|',

          "{0:^5}".format(round(bc.loc[:,col].median())), '|', "{0:^5}".format(round(bc.loc[:,col].std(),2)), '|',

          )              



print('='*82)
print('='*82)

print('| Categorical Variable |', "{0:^55}".format('Possible Categorical Values'), '|')

print('='*82)



for catdata in categorical_feature_list_orig:

    print('|', "{0:^20}".format(catdata), '|', "{0:^55}".format(str(bc[catdata].unique())), '|')

print('='*82)

plt.figure(figsize=(20,5))



dual_categories = ['diagnosis']



plt_cnt = 1;

for feature in dual_categories:

    plt.subplot(1,4,plt_cnt)

    tempdf = get_percentage_cnt(bc, feature)

    myplot(feature, 'cnt%', tempdf)

    plt_cnt += 1



plt.show()
#Continous Features

Continous_feature_list   = list(map(str, bc.columns[bc.dtypes != object]))



#Categorical Features

Categorical_feature_list = list(map(str, bc.columns[bc.dtypes == object]))
d_asp = pd.DataFrame(pd.get_dummies(bc[Categorical_feature_list]))



#Print shapes of all DFs

print('Original shape of bc:', bc.shape)

print('Original shape of Categorical Features DF:', d_asp.shape)



#concatinate the dataframe with get_dummies to original DataFrame

bc = pd.concat([bc, d_asp], axis=1)



#check the shape to confirm features are added

print('Modified shape of bc:', bc.shape)



bc.head()
'''

"Categorical_feature_list" now contains the original Categorical Features 

We can safely drop original Categorical Features from Dataframe as these are no longer necessary

'''

insurance_dummies = bc.drop(Categorical_feature_list, axis=1)



insurance_dummies.head(3)
f, ax = plt.subplots(figsize=(10, 8))



corr = insurance_dummies.corr()



sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),

            square=True, ax=ax)

bc.drop(Categorical_feature_list, axis=1, inplace=True)



dummycolumns = ['diagnosis_B']



bc.drop(dummycolumns, axis=1, inplace=True)



bc.head()



class_y = bc['diagnosis_M']
print('Num of categorical columns remaining are:', len(bc.columns[bc.dtypes == object]))
bcmean = bc.mean()

bcstd = bc.std()



norm_cp = (bc - bc.mean()) / bc.std()

norm_cp.head(5)
#Declare a Training set of size 80%

mtrnsz = 0.8



#Declare a Testing set of size 20%

mtstsz = (1 - mtrnsz)



#Init a random seed value to be used throughout

rsz = 100



#Dependent Feature for our model

Dependant_feature = ['diagnosis_M']

y_feature = bc['diagnosis_M']



#Total independant features of the model

Independent_features = list(set(norm_cp.columns) - set(Dependant_feature))

X_features = norm_cp[Independent_features]
X_features = norm_cp[Independent_features]

ChosenFeatureList = X_features



X_train, X_test, y_train, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz,

                                                    test_size=mtstsz, random_state=rsz)



X_c = sm.add_constant(X_train)

lm = sm.OLS(y_train,X_c).fit()

#Print Variance Inflation Factor for detecting Multicollinearity

display_model_stats(lm, True)
# using various kernels, C=1, default value of gamma

kernels_2try = [ 'linear', 'poly', 'rbf']



num_models = 1



for kernels in kernels_2try :

    print('==============================')

    print('Kernel : ', kernels, ", Model #:", num_models)

    print('==============================')

    

    model = SVC(C = 1, kernel=kernels)

    model.fit(X_train, y_train)

    

    y_pred = model.predict(X_test)

    print_CFstats(y_test, y_pred)

    num_models += 1

    print('==============================\n')
X_features = norm_cp[Independent_features]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz,

                                                test_size=mtstsz, random_state=rsz)



logreg = LogisticRegression()



num_features = [20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]



for numbr_feat in num_features :

    rfe = RFE(logreg, numbr_feat) # running RFE

    rfe = rfe.fit(X_train, y_train)

    

    top_features = []

    ChosenFeatureList = []

    

    for is_valid, feature in zip(list(rfe.support_), list(X_train.columns)) :

        if True == is_valid :

            #print(feature)

            top_features.append(feature)

    

    ChosenFeatureList = top_features

    X_features = norm_cp[top_features]

    

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz,

                                                    test_size=mtstsz, random_state=rsz)



    print('==============================')

    print('Kernel : linear, #Features:', numbr_feat)

    print('==============================')



    model = SVC(C = 1, kernel='linear')

    model.fit(X_train, y_train)



    y_pred = model.predict(X_test)

    print_CFstats(y_test, y_pred)

    

    X_c = sm.add_constant(X_train)

    lm = sm.OLS(y_train,X_c).fit()

    #Print Variance Inflation Factor for detecting Multicollinearity

    display_model_stats(lm, True)

    

    print('='*41)

    print('\n\n')
X_features = norm_cp[Independent_features]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz,

                                                test_size=mtstsz, random_state=rsz)



rfe = RFE(logreg, 3) # running RFE

rfe = rfe.fit(X_train, y_train)



top_features_3 = []

for is_valid, feature in zip(list(rfe.support_), list(X_train.columns)) :

    if True == is_valid :

        top_features_3.append(feature)



X_features = norm_cp[top_features_3]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz,

                                                test_size=mtstsz, random_state=rsz)



print('Final features:')

print('='*15)

for feat in top_features_3:

    print(feat)

model_lin_3 = SVC(C = 1, kernel='linear')

model_lin_3.fit(X_train, y_train)



y_pred = model_lin_3.predict(X_test)



print('\nStatistics:')

print('='*30)

print_CFstats(y_test, y_pred)

print('='*30)
X_features = norm_cp[Independent_features]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz,

                                                test_size=mtstsz, random_state=rsz)



rfe = RFE(logreg, 7) # running RFE

rfe = rfe.fit(X_train, y_train)



top_features = []

for is_valid, feature in zip(list(rfe.support_), list(X_train.columns)) :

    if True == is_valid :

        top_features.append(feature)



X_features = norm_cp[top_features]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz,

                                                test_size=mtstsz, random_state=rsz)



print('Final features:')

print('='*15)

for feat in top_features:

    print(feat)

model_lin = SVC(C = 1, kernel='linear')

model_lin.fit(X_train, y_train)



y_pred = model_lin.predict(X_test)



print('\nStatistics:')

print('='*30)

print_CFstats(y_test, y_pred)

print('='*30)
f, ax = plt.subplots(figsize=(10, 8))

corr = round(norm_cp[top_features].corr(),3)

sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# creating a KFold object

folds = KFold(n_splits = 25, shuffle = True, random_state = 4)



# specify range of hyperparameters

# Set the parameters by cross-validation



hyper_params = [ {

                    'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10],

                    'C': [1, 25, 50, 100, 1000]

                 }

               ]



# specify model

model = SVC(kernel="rbf")



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = model, 

                        param_grid = hyper_params,

                        scoring = 'recall',

                        cv = folds,

                        verbose = 1,

                        return_train_score=True,

                       )

# fit the model

model_cv.fit(X_train, y_train)       



# run the model on test data

y_pred = model_cv.predict(X_test)



# scores of GridSearch CV

gscores_recall = model_cv.cv_results_
#Common plotting code to plot the various combinations of Gamma and C for the given score(recall)

def plot_gamma(cv_results, gamma_val, ymax):

    gamma_01 = cv_results[cv_results['param_gamma']==gamma_val]



    plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

    plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

    plt.xlabel('C')

    plt.ylabel('Recall')

    titlestr = "gamma=" + str(gamma_val)

    plt.title(titlestr)

    plt.ylim([0.20, ymax])

    plt.legend(['test recall', 'train recall'], loc='upper left')

    plt.xscale('log')
# cv results

cv_results = pd.DataFrame(model_cv.cv_results_)



# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# # plotting

plt.figure(figsize=(16,8))



gama2run = [0.0001, 0.001, 0.01, 0.1, 1, 10]

plotnum = [231, 232, 233, 234, 235, 236]

ymax = [1.1, 1.1, 1.1, 1.1, 1.1, 1.1]

idx = 0



for val in gama2run:

    #print(val, plotnum[idx])

    plt.subplot(plotnum[idx])

    plot_gamma(cv_results, val, ymax[idx])

    idx += 1



plt.show()
best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
# model

model_rbf = SVC(C=50, gamma=0.1, kernel="rbf")



model_rbf.fit(X_train, y_train)

    

y_pred = model_rbf.predict(X_test)

print_CFstats(y_test, y_pred)