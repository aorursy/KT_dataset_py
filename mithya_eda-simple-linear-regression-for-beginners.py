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



import statsmodels.api as sm
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
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



fname = os.path.join(dirname, filename)

# Any results you write to the current directory are saved as output.



#Importing dataset

insurance = pd.read_csv(fname)
#Examine the data

insurance.head()
#insurance.drop(insurance.columns[0], axis=1, inplace=True)

insurance.head()
print('='*20)

print('|', "{0:^15}".format('Statistics:'), '|')

print('='*85)

print('|', "{0:^25}".format('Column name'), '|', "{0:^25}".format('% Unique Values'),

      '|', "{0:^25}".format('%Missing values'), '|')

print('='*85)



for col in insurance:

    print("| {0:^25}".format(col), '|', "{0:^25.2%}".format(round(insurance[col].nunique()/len(insurance[col]),2)), '|',

          "{0:^25.2%}".format(round(insurance[col].isnull().sum()/len(insurance[col]),2)), '|')



print('='*85)



insurance.shape
cols_with_zerodata_inpercent = round(100*(insurance.isnull().sum()/len(insurance.index)), 2)



print("Columns with 100% missing data:")

colnames = insurance.columns[cols_with_zerodata_inpercent==100].values

print("Num col:", len(colnames), colnames)



cols_with_nonuniq_values = insurance.nunique()

colnamesnouniq = insurance.columns[(cols_with_nonuniq_values <= 1)].values

print("Columns with just 1 unique value:", len(colnamesnouniq), colnamesnouniq)



insurance.drop_duplicates(keep=False, inplace=True)



insurance.head(1)
#Get list of categorical Features

categorical_feature_list_orig = list(map(str, insurance.columns[insurance.dtypes == object]))



#Get list of continous Features

continous_feature_list_orig = list(map(str, insurance.columns[insurance.dtypes != object]))



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

          "{0:^7}".format(str(insurance.loc[:,col].dtype)), '|', "{0:^6}".format(round(insurance.loc[:,col].min())), '|',

          "{0:^9}".format(round(insurance.loc[:,col].max())), '|', "{0:^5}".format(round(insurance.loc[:,col].mean())), '|',

          "{0:^5}".format(round(insurance.loc[:,col].median())), '|', "{0:^5}".format(round(insurance.loc[:,col].std())), '|',

          )              



print('='*82)
print('='*82)

print('| Categorical Variable |', "{0:^55}".format('Possible Categorical Values'), '|')

print('='*82)



for catdata in categorical_feature_list_orig:

    print('|', "{0:^20}".format(catdata), '|', "{0:^55}".format(str(insurance[catdata].unique())), '|')

print('='*82)

plt.figure(figsize=(20,5))



dual_categories = ['sex', 'smoker', 'region']



plt_cnt = 1;

for feature in dual_categories:

    plt.subplot(1,4,plt_cnt)

    tempdf = get_percentage_cnt(insurance, feature)

    myplot(feature, 'cnt%', tempdf)

    plt_cnt += 1



plt.show()
#Continous Features

Continous_feature_list   = list(map(str, insurance.columns[insurance.dtypes != object]))



#Categorical Features

Categorical_feature_list = list(map(str, insurance.columns[insurance.dtypes == object]))
d_asp = pd.DataFrame(pd.get_dummies(insurance[Categorical_feature_list]))



#Print shapes of all DFs

print('Original shape of insurance:', insurance.shape)

print('Original shape of Categorical Features DF:', d_asp.shape)



#concatinate the dataframe with get_dummies to original DataFrame

insurance = pd.concat([insurance, d_asp], axis=1)



#check the shape to confirm features are added

print('Modified shape of insurance:', insurance.shape)



insurance.head()
'''

"Categorical_feature_list" now contains the original Categorical Features 

We can safely drop original Categorical Features from Dataframe as these are no longer necessary

'''

insurance_dummies = insurance.drop(Categorical_feature_list, axis=1)



insurance_dummies.head(3)
f, ax = plt.subplots(figsize=(10, 8))



corr = insurance_dummies.corr()



sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),

            square=True, ax=ax)

insurance.drop(Categorical_feature_list, axis=1, inplace=True)



dummycolumns = ['sex_female', 'smoker_no', 'region_southwest']



insurance.drop(dummycolumns, axis=1, inplace=True)



insurance.head()
print('Num of categorical columns remaining are:', len(insurance.columns[insurance.dtypes == object]))
norm_cp = (insurance - insurance.mean()) / insurance.std()

norm_cp.head(5)
# Let's see the correlation matrix. This gives us a clear idea of what might be the features and their corelation

plt.figure(figsize = (12,8))     # Size of the figure



df_cor = pd.DataFrame(norm_cp[Continous_feature_list])



cor_mat = round(df_cor.corr(),1)



#obtain list of Continous Independent Features (CIF) based on their importance

df = pd.DataFrame(cor_mat.charges.sort_values(ascending=False))

df.reset_index(inplace=True)



#Continous Independant Feature

CIF = list(df.iloc[1:,0])



ax = plt.axes()

sns.heatmap(cor_mat, annot=True, ax=ax)

ax.set_title('Heatmap Continous Features')

plt.show()
# Let's see the correlation matrix. This gives us a clear idea of what might be the features and their corelation

plt.figure(figsize = (12,8))     # Size of the figure



df_cor = pd.DataFrame(norm_cp)



#Dropping all Continous Features for plotting purposes...

for feat in Continous_feature_list:

    if feat == 'charges':

        continue;

    df_cor.drop(feat, axis=1, inplace=True)



ax = plt.axes()   

cor_mat = round(df_cor.corr(),1)

sns.heatmap(cor_mat, annot = True)

ax.set_title('Heatmap Continous Features')

plt.show()



#Declare a Training set of size 80%

mtrnsz = 0.8



#Declare a Testing set of size 20%

mtstsz = (1 - mtrnsz)



#Init a random seed value to be used throughout

rsz = 100



#Dependent Feature for our model

Dependant_feature = ['charges']

y_feature = norm_cp[Dependant_feature]



#Total independant features of the model

Independent_features = list(set(norm_cp.columns) - set(Dependant_feature))



#Keeps track of the current feature to be tested

FeatureToTest = []



#Keeps track of chosen features

ChosenFeatureList = []
def display_model_stats(curmodel, printop=False):

    #genrate a Dataframe extracting feature names and their p-values

    df = gen_pvaluesdf(curmodel)



    #Variance Inflation Factor for detecting Multicollinearity

    df_vif = calculate_vif(X)

    #Merge the 2 dataframes to create one df and sort on significance of pvalue along with VIF info

    #Also, display the R-square and Adjustd R-square

    df_p_vif = pd.merge(df_vif, df)



    if printop == True:



        print('='*33)

        print('| Current Features in the model: |')

        print('='*33)

        print(ChosenFeatureList)

        

        print('='*19)

        print('| Important Stats: |')

        print('='*48)

        print('| R-squared:', round(curmodel.rsquared,3),

              ' | Adjusted R-squared:', round(curmodel.rsquared_adj,3), '|')

        print('='*48)

        df_p_vif.sort_values(by='pval', ascending=False)

        display(df_p_vif)



        print('='*18)

        print('| Tested Feature: |')

        print('='*18)

        print(FeatureToTest)       
#Total numer of models tested to arrive at solution

numModelsTested = 0



#Keeps track of features being tested

FeatureListToTest = []



#Keeps track of chosen features

ChosenFeatureList = []



numModelsTested += 1

cont_featurenum = 0



FeatureToTest = CIF[cont_featurenum]

FeatureListToTest.append(FeatureToTest)



#We could have chosen any arbitrary list of Continous features to begin with but as

#the below features are very highly corelated with price of a car; hence choosing to start with these

X_features = norm_cp[FeatureListToTest]



X, X_test, y, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz, test_size=mtstsz, random_state=rsz)

X = sm.add_constant(X)

lm1 = sm.OLS(y,X).fit()



display_model_stats(lm1, True)
#Add the chosen feature to list of chosenfeatures

ChosenFeatureList.append(FeatureToTest)



numModelsTested += 1

cont_featurenum += 1



FeatureToTest = CIF[cont_featurenum]

FeatureListToTest.append(FeatureToTest)



#We could have chosen any arbitrary list of Continous features to begin with but as

#the below features are very highly corelated with price of a car; hence choosing to start with these

X_features = norm_cp[FeatureListToTest]



X, X_test, y, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz, test_size=mtstsz, random_state=rsz)

X = sm.add_constant(X)

lm2 = sm.OLS(y,X).fit()



display_model_stats(lm2, True)
#Add the chosen feature to list of chosenfeatures

ChosenFeatureList.append(FeatureToTest)



numModelsTested += 1

cont_featurenum += 1



FeatureToTest = CIF[cont_featurenum]

FeatureListToTest.append(FeatureToTest)



#We could have chosen any arbitrary list of Continous features to begin with but as

#the below features are very highly corelated with price of a car; hence choosing to start with these

X_features = norm_cp[FeatureListToTest]



X, X_test, y, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz, test_size=mtstsz, random_state=rsz)

X = sm.add_constant(X)

lm3 = sm.OLS(y,X).fit()



display_model_stats(lm3, True)
FeatureListToTest.remove('children')

ChosenFeatureList
CatIF = ['sex_male', 'smoker_yes', 'region_northeast', 'region_northwest', 'region_southeast']
numModelsTested += 1

cat_var_num = 0



FeatureToTest = CatIF[cat_var_num]

FeatureListToTest.append(FeatureToTest)



#We could have chosen any arbitrary list of Continous features to begin with but as

#the below features are very highly corelated with price of a car; hence choosing to start with these

X_features = norm_cp[FeatureListToTest]



X, X_test, y, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz, test_size=mtstsz, random_state=rsz)

X = sm.add_constant(X)

lm6 = sm.OLS(y,X).fit()



display_model_stats(lm6, True)
FeatureListToTest.remove('sex_male')

numModelsTested += 1



cat_var_num += 1

FeatureToTest = CatIF[cat_var_num]



FeatureListToTest.append(FeatureToTest)



#We could have chosen any arbitrary list of Continous features to begin with but as

#the below features are very highly corelated with price of a car; hence choosing to start with these

X_features = norm_cp[FeatureListToTest]



X, X_test, y, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz, test_size=mtstsz, random_state=rsz)

X = sm.add_constant(X)

lm6_1 = sm.OLS(y,X).fit()



display_model_stats(lm6_1, True)
ChosenFeatureList.append(FeatureToTest)

numModelsTested += 1



FeatureToTest = ['region_northeast', 'region_northwest', 'region_southeast']

FeatureListToTest.extend(FeatureToTest)



#We could have chosen any arbitrary list of Continous features to begin with but as

#the below features are very highly corelated with price of a car; hence choosing to start with these

X_features = norm_cp[FeatureListToTest]



X, X_test, y, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz, test_size=mtstsz, random_state=rsz)

X = sm.add_constant(X)

lm6_2 = sm.OLS(y,X).fit()



display_model_stats(lm6_2, True)
ChosenFeatureList
FeatureToTest = FeatureListToTest



X_features = norm_cp[ChosenFeatureList]



X, X_test, y, y_test = train_test_split(X_features, y_feature, random_state=0)



X = sm.add_constant(X)

lm_final = sm.OLS(y,X).fit()



#Print summary

print(lm_final.summary())



#Print Variance Inflation Factor for detecting Multicollinearity

display_model_stats(lm_final, True)
#Total number of models tested are:



print('Total num of Models tested are:', numModelsTested)

print('The final set of important features are:', ChosenFeatureList)

from sklearn.ensemble import RandomForestRegressor



X, X_test, y, y_test = train_test_split(X_features, y_feature, train_size=mtrnsz, test_size=mtstsz, random_state=rsz)



forest = RandomForestRegressor(n_estimators = 20,

                              criterion = 'mse',

                              random_state = 0,

                              n_jobs = -1)



forest.fit(X, y)

forest_train_pred = forest.predict(X)

forest_test_pred = forest.predict(X_test)



print('MSE train data: %.3f, MSE test data: %.3f' % (



mean_squared_error(y, forest_train_pred),



mean_squared_error(y_test, forest_test_pred)))



print('R2 train data: %.3f, R2 test data: %.3f' % (



r2_score(y, forest_train_pred),

r2_score(y_test, forest_test_pred)))



#Use Test data-set to evaluate the model

X_test = sm.add_constant(X_test)

Predicted_charges = lm_final.predict(X_test)



#Obtain Mean Squared Error

mse = mean_squared_error(y_test.charges, Predicted_charges)

#Obtain Rsquare based score

r_squared = r2_score(y_test.charges, Predicted_charges)



print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
#Actual vs Predicted

c = [i for i in range(1,len(y_test)+1,1)]

fig = plt.figure(figsize=(20,5))

plt.plot(c,y_test.charges, color="blue", linewidth=1.5, linestyle="-")

plt.plot(c,Predicted_charges, color="red",  linewidth=1.5, linestyle="-")

fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('Price', fontsize=16)                               # Y-label
# Error terms

c = [i for i in range(1,len(y_test)+1,1)]

fig = plt.figure(figsize=(20,5))

plt.plot(c,y_test.charges-Predicted_charges, color="blue", linewidth=1.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('Y_test.price-Predicted_price', fontsize=16)                # Y-label
