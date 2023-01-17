# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sympy

import scipy

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import sklearn.metrics as metrics

import statsmodels.api as stats

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/Purchase_Likelihood.csv")

df.head()
target = df["A"].astype("category")

predictor = df[["group_size","homeowner","married_couple"]].astype('category')
def build_mnlogit (fullX, y, debug = 'N'):

    # Number of all parameters

    nFullParam = fullX.shape[1]



    # Number of target categories

    y_category = y.cat.categories

    nYCat = len(y_category)



    # Find the non-redundant columns in the design matrix fullX

    reduced_form, inds = sympy.Matrix(fullX.values).rref()



    # These are the column numbers of the non-redundant columns

    if (debug == 'Y'):

        print('Column Numbers of the Non-redundant Columns:')

        print(inds)



    # Extract only the non-redundant columns for modeling

    X = fullX.iloc[:, list(inds)]



    # The number of free parameters

    thisDF = len(inds) * (nYCat - 1)



    # Build a multionomial logistic model

    logit = stats.MNLogit(y, X)

    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

    thisParameter = thisFit.params

    thisLLK = logit.loglike(thisParameter.values)



    if (debug == 'Y'):

        print(thisFit.summary())

        print("Model Parameter Estimates:\n", thisParameter)

        print("Model Log-Likelihood Value =", thisLLK)

        print("Number of Free Parameters =", thisDF)



    # Recreat the estimates of the full parameters

    workParams = pd.DataFrame(np.zeros(shape = (nFullParam, (nYCat - 1))))

    workParams = workParams.set_index(keys = fullX.columns)

    fullParams = pd.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)

    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)



    # Return model statistics

    return (thisLLK, thisDF, fullParams)

group_size_dummies = pd.get_dummies(predictor[["group_size"]].astype('category'))

homeowner_dummies = pd.get_dummies(predictor[["homeowner"]].astype('category'))

married_couple_dummies = pd.get_dummies(predictor[["married_couple"]].astype('category'))
#Intercept only



designX = pd.DataFrame(target.where(target.isnull(), 1))

LLK0, DF0, fullParams0 = build_mnlogit (designX, target, debug = 'Y')
#Intercept + Group Size

designX = stats.add_constant(group_size_dummies, prepend=True)

LLK_1R, DF_1R, fullParams_1R = build_mnlogit (designX, target, debug = 'N')

testDev = 2 * (LLK_1R - LLK0)

testDF = DF_1R - DF0

testPValue = scipy.stats.chi2.sf(testDev, testDF)

print('Deviance Chi=Square Test')

print('Chi-Square Statistic = ', testDev)

print('  Degreee of Freedom = ', testDF)

print('        Significance = ', testPValue)
# Intercept + Group Size + HomeOwner

designX = group_size_dummies

designX = designX.join(homeowner_dummies)

designX = stats.add_constant(designX, prepend=True)

LLK_1R_1J, DF_1R_1J, fullParams_1R_1J = build_mnlogit (designX, target, debug = 'N')

testDev = 2 * (LLK_1R_1J - LLK_1R)

testDF = DF_1R_1J - DF_1R

testPValue = scipy.stats.chi2.sf(testDev, testDF)

print('Deviance Chi=Square Test')

print('Chi-Square Statistic = ', testDev)

print('  Degreee of Freedom = ', testDF)

print('        Significance = ', testPValue)

# Intercept + Group Size + HomeOwner + Married Couple

designX = group_size_dummies

designX = designX.join(homeowner_dummies)

designX = designX.join(married_couple_dummies)

designX = stats.add_constant(designX, prepend=True)

LLK_1R_1J_M, DF_1R_1J_M, fullParams_1R_1J = build_mnlogit (designX, target, debug = 'N')

testDev = 2 * (LLK_1R_1J_M - LLK_1R_1J)

testDF = DF_1R_1J_M - DF_1R_1J

testPValue = scipy.stats.chi2.sf(testDev, testDF)

print('Deviance Chi=Square Test')

print('Chi-Square Statistic = ', testDev)

print('  Degreee of Freedom = ', testDF)

print('        Significance = ', testPValue)
def create_interaction (inDF1, inDF2):

    name1 = inDF1.columns

    name2 = inDF2.columns

    outDF = pd.DataFrame()

    for col1 in name1:

        for col2 in name2:

            outName = col1 + " * " + col2

            outDF[outName] = inDF1[col1] * inDF2[col2]

    return(outDF)
# Intercept + Group Size + HomeOwner + Married Couple + GroupSize*Homeowner

designX = group_size_dummies

designX = designX.join(homeowner_dummies)

designX = designX.join(married_couple_dummies)



xRJ = create_interaction(group_size_dummies, homeowner_dummies)

designX = designX.join(xRJ)



designX = stats.add_constant(designX, prepend=True)

LLK_2RJ_m, DF_2RJ_m, fullParams_2RJ = build_mnlogit (designX, target, debug = 'N')

testDev = 2 * (LLK_2RJ_m - LLK_1R_1J_M)

testDF = DF_2RJ_m - DF_1R_1J_M

testPValue = scipy.stats.chi2.sf(testDev, testDF)

print('Deviance Chi=Square Test')

print('Chi-Square Statistic = ', testDev)

print('  Degreee of Freedom = ', testDF)

print('        Significance = ', testPValue)
designX = group_size_dummies

designX = designX.join(homeowner_dummies)

designX = designX.join(married_couple_dummies)

xRJ = create_interaction(group_size_dummies, homeowner_dummies)

designX = designX.join(xRJ)



xHM = create_interaction(homeowner_dummies,married_couple_dummies)

designX = designX.join(xHM)



designX = stats.add_constant(designX, prepend=True)
m = [4,6,8,10,12,14,15,16,18,19,20]

#designX.columns[n]

print("Aliased parameters in the model")

for i in m:

    print(designX.columns[i])
# Intercept + Group Size + HomeOwner + Married Couple + GroupSize*Homeowner + Homeowner*MarriedCouple



LLK_2RJ_HM, DF_2RJ_HM, fullParams_2RJ = build_mnlogit (designX, target, debug = 'Y')

testDev = 2 * (LLK_2RJ_HM - LLK_2RJ_m)

testDF = DF_2RJ_HM - DF_2RJ_m

testPValue = scipy.stats.chi2.sf(testDev, testDF)

print('Deviance Chi=Square Test')

print('Chi-Square Statistic = ', testDev)

print('  Degreee of Freedom = ', testDF)

print('        Significance = ', testPValue)
import math

FeatureImportance = [4.347870389027117e-210,4.306457217534288e-19,5.512105969198056e-52,4.13804354648637e-16 ]

for i in FeatureImportance:

    print(-math.log10(i))
logit = stats.MNLogit(target, designX)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
group = [1,2,3,4]

home = [0,1]

married = [0,1]

combi = []



for i in home:

    for j in group:

        for k in married:

            combi.append([i,j,k])

            

combinat = pd.DataFrame(combi,columns=["home","group","married"])



combi_group_dummies = pd.get_dummies(combinat[["group"]].astype('category'))

combi_home_dummies = pd.get_dummies(combinat[["home"]].astype('category'))

combi_married_dummies = pd.get_dummies(combinat[["married"]].astype('category'))



gh_combi = create_interaction(combi_group_dummies, combi_home_dummies)

gh_combi = pd.get_dummies(gh_combi)



hm_combi = create_interaction(combi_home_dummies,combi_married_dummies)

hm_combi = pd.get_dummies(hm_combi)



fullX = combi_group_dummies

fullX = fullX.join(combi_home_dummies)

fullX = fullX.join(combi_married_dummies)

fullX = fullX.join(gh_combi)

fullX = fullX.join(hm_combi)

fullX = stats.add_constant(fullX, prepend=True)
combinat
predicted_probabilities = thisFit.predict(fullX)

predicted_probabilities = pd.DataFrame.join(combinat,predicted_probabilities)

predicted_probabilities
predicted_probabilities[1]/predicted_probabilities[0]
combi[9]
one,two,three = 0,0,0

for i in df.A:

    if i == 0:

        one+=1

    elif i == 1:

        two+=1

    else:

        three+=1

counts = [one,two,three]



proabs = []

for i in counts:

    proabs.append(i/len(df.A))



data = pd.DataFrame(data = counts , columns = ["Count"])

data["Class Probabilites"] = proabs

data
pd.crosstab(df.A,df.group_size)
pd.crosstab(df.A,df.homeowner)
pd.crosstab(df.A,df.married_couple)
import scipy.stats as ss

def cramers_v(confusion_matrix):

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0,(phi2 - ((k-1)*(r-1))/(n-1)))

    rcorr = r - ((r-1)**2)/(n-1)

    kcorr = k - ((k-1)**2)/(n-1)

    print(np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1))))
confusion_matrix = pd.crosstab(df.A,df.group_size)

cramers_v(confusion_matrix)
confusion_matrix = pd.crosstab(df.A,df.married_couple)

cramers_v(confusion_matrix)
confusion_matrix = pd.crosstab(df.A,df.homeowner)

cramers_v(confusion_matrix)
data_count = df.groupby('A').count()['group_size']

data_prop = data_count / df.shape[0]

data_grouped = pd.DataFrame({'A': data_count.index, 

                                    'Count': data_count.values, 

                                    'Class probabilities': data_prop.values})



crosstab_group_size = pd.crosstab(df.A, df.group_size, margins = False, dropna = False)



crosstab_homeowner = pd.crosstab(df.A, df.homeowner, margins = False, dropna = False)



crosstab_married_couple = pd.crosstab(df.A, df.married_couple, margins = False, dropna = False)
def predict_probabilities(predictors):

    prob_0 = ((data_grouped['Count'][0] / data_grouped['Count'].sum()) * 

                   (crosstab_group_size[predictors[0]][0] / crosstab_group_size.loc[[0]].sum(axis=1)[0]) * 

                   (crosstab_homeowner[predictors[1]][0] /crosstab_homeowner.loc[[0]].sum(axis=1)[0]) * 

                   (crosstab_married_couple[predictors[2]][0] / crosstab_married_couple.loc[[0]].sum(axis=1)[0]))

    prob_1 = ((data_grouped['Count'][1] / data_grouped['Count'].sum()) * 

                   (crosstab_group_size[predictors[0]][1] / crosstab_group_size.loc[[1]].sum(axis=1)[1]) * 

                   (crosstab_homeowner[predictors[1]][1] / crosstab_homeowner.loc[[1]].sum(axis=1)[1]) * 

                   (crosstab_married_couple[predictors[2]][1] / crosstab_married_couple.loc[[1]].sum(axis=1)[1]))

    prob_2 = ((data_grouped['Count'][2] / data_grouped['Count'].sum()) * 

                   (crosstab_group_size[predictors[0]][2] / crosstab_group_size.loc[[2]].sum(axis=1)[2]) * 

                   (crosstab_homeowner[predictors[1]][2] / crosstab_homeowner.loc[[2]].sum(axis=1)[2]) * 

                   (crosstab_married_couple[predictors[2]][2] / crosstab_married_couple.loc[[2]].sum(axis=1)[2]))

    sum_of_probs = prob_0 + prob_1 + prob_2

    valid_prob_0 = prob_0 / sum_of_probs

    valid_prob_1 = prob_1 / sum_of_probs

    valid_prob_2 = prob_2 / sum_of_probs



    return [valid_prob_0, valid_prob_1, valid_prob_2]
group_sizes = sorted(list(df.group_size.unique()))

homeowners = sorted(list(df.homeowner.unique()))

married_couples = sorted(list(df.married_couple.unique()))

n = pd.DataFrame(combinations,columns = ["group_size","homeowner","married_couples"])



import itertools

combinations = list(itertools.product(group_sizes, homeowners, married_couples))



nb_probabilities = []

for combination in combinations:

    temp = [predict_probabilities(combination)]

    nb_probabilities.extend(temp)

final = pd.DataFrame.join(pd.DataFrame(nb_probabilities,columns = ["0","1","2"]),n)

final
final["1"]/final["0"]
max(final["1"]/final["0"])
final.iloc[7]