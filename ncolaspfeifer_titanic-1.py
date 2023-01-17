# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



## for plotting

import matplotlib.pyplot as plt

import seaborn as sns



#for statistical tests

import scipy

import statsmodels.formula.api as smf

import statsmodels.api as sm



## for machine learning

import sklearn

from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition, neural_network



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

train_file = "../input/titanic/train.csv"

traindf = pd.read_csv(train_file)

traindf.columns = traindf.columns.str.lower()

print(traindf.columns)



testdf = pd.read_csv("../input/titanic/test.csv")

testdf.columns = testdf.columns.str.lower()

#testdf.head()

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#test predictability of features using statistical hypothesis testing

#our target feature (survived) is categorical, so we use the following tests when assessing each feature type:

    #categorical: chi^2 test with Cramer's V correlation coefficient

    #numerical: ANOVA F-test

#based on https://towardsdatascience.com/machine-learning-with-python-classification-complete-tutorial-d2c99dc524ec





"""

Determines the type of a variable in a given dataframe.

@param variable variable we want to know the type

@param df pandas DataFrama containing values for the variable

@returns "numerical" | "categorical"

"""

def detect_feature_type(variable, df):

    values = df[variable].unique()

    dtype = df[variable].dtype

    max_categ_values = 10

    if len(values) > max_categ_values and dtype != 'O':

        return "numerical"

    return "categorical"



predictive_features = set()

non_predictive_features = set()

categorical_features = set()

numerical_features = set()



"""

Uses the ANOVA F-test  to calculate if the sets created by the different values of 

    the numerical feature, under the target feature, differ significantly.

@param feature name of the numerical feature we want to measure predictability

@param Y name of the target feature

@param traindf pandas DataFrame containing the data we need to use

@returns statistical significance p value

"""

def is_numerical_feature_predictive(feature, Y, traindf):

    model = smf.ols(Y + " ~ " + feature, data=traindf).fit()

    signif_result = sm.stats.anova_lm(model)

    p_value = round(signif_result["PR(>F)"][0], 3)

    #result = "Predictive" if p_value < 0.05 else "Non-Predictive"

    return p_value < 0.05, p_value





"""

Uses the chi2 statistical hypothesis test to calculate if the datasets created by 

    the different values of the categorical feature splits the data, under the 

    target feature, is statistically significant.

    Also report Cramer's V correlation coefficient.

@param feature name of the categorical feature we want to measure predictability

@param Y name of the target feature

@param traindf pandas DataFrame containing the data we need to use

@returns statistical significance p value,

         cramer's V correlation coefficient

"""

def is_categorical_feature_predictive(feature, Y, traindf):

    #cross_tab = pd.crosstab(index=traindf[predictor], columns=traindf[Y])

    #crosstab execution time is worse than groupby and pivot_table according to

    #https://ramiro.org/notebook/pandas-crosstab-groupby-pivot/

    contingency_table = traindf.groupby([feature, Y]).count().unstack().fillna(0)

    #print(contingency_table)

    chi2_test = scipy.stats.chi2_contingency(contingency_table)

    chi2, p_value = chi2_test[0], chi2_test[1]

    n = contingency_table.sum().sum()

    phi2 = chi2/n

    r, k = contingency_table.shape

    phi2_corr = max(0, phi2 - ( (k-1) * (r-1)) / (n-1))

    r_corr = r - ((r-1) ** 2) / (n-1)

    k_corr = k - ((k-1) ** 2) / (n-1)

    coeff = np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))

    coeff, p_value = round(coeff, 3), round(p_value, 3)

    #result = "Predictive" if p_value < 0.05 else "Non-Predictive"

    return p_value < 0.05, p_value, coeff





#Y is the variable we want to predict

#survived is categorical

Y = "survived"

for feature in (set(traindf.columns) - set(Y)):

    if feature == Y:

        continue

    var_type = detect_feature_type(feature, traindf)

    if var_type == "numerical":

        #numerical feature

        numerical_features.add(feature)

        predictive, p_value = is_numerical_feature_predictive(feature, Y, traindf)

        if predictive:

            text_result = "Predictive"

            predictive_features.add(feature)

        else:

            text_result = "Non-Predictive"

            non_predictive_features.add(feature)

        print("{var_name} is {result} (Anova F p-value: {p_value})".format(

            var_name=feature, result=text_result, p_value=round(p_value, 3)))

    if var_type == "categorical":

        #categorical feature

        categorical_features.add(feature)

        predictive, p_value, coeff = is_categorical_feature_predictive(feature, Y, traindf)

        if predictive:

            text_result = "Predictive"

            predictive_features.add(feature)

        else:

            text_result = "Non-Predictive"

            non_predictive_features.add(feature)

        print("{var_name} is {result} (chi2 p-value: {p_value}). Cramer Correlation: {cramer}".format(

            var_name=feature, result=text_result, p_value=p_value, cramer=coeff))



print("Non-predictive features: {}".format(str(non_predictive_features)))

print("Predictive features: {}".format(str(predictive_features)))
#feature extraction



#cabin section extraction



"""

Extracts the feature cabin_section from the cabin feature in the dataframe.

    Also reports if the extracted feature is statistically significant in

    predicting the target feature

@Y target feature

@param df pandas DataFrame containing the cabin column

@returns the given dataframe,

         a boolean informing if the extracted feature is statistically significant

             in predicting the target feature

"""

def extract_cabin_section_feature(Y, df):

    cabin_sect = "cabin_section"

    df[cabin_sect] = df["cabin"].apply(lambda x: str(x)[0])



    predictive, p_value, coeff = is_categorical_feature_predictive(cabin_sect, Y, traindf)

    text_result = "Predictive" if predictive else "Non-Predictive"

    print("{var_name} is {result} (chi2 p-value: {p_value}). Cramer Correlation: {cramer}".format(

        var_name=cabin_sect, result=text_result, p_value=p_value, cramer=coeff))

    return df, predictive



traindf, predictive = extract_cabin_section_feature(Y, traindf)

if predictive:

    predictive_features.add("cabin_section")

categorical_features.add("cabin_section")



#may extract name title later
"""

@param df pandas dataframe containing at least columns fare and pclass

@param class_means dataframe with columns pclass and fare. 

    fare contains the average fair for the pclass

@returns dataframe with nan and 0 values of fare substituted by the row's pclass average fare

"""

def fix_missing_fare_values(df, class_means):

    df.fare = df.fare.map(lambda x: np.nan if x==0 else x)

    df.fare = df[['fare', 'pclass']].apply(

        lambda x: class_means.fare[x['pclass']] if pd.isnull(x['fare']) else x['fare'], axis=1 )

    return df



#split the original traindf in train and test data for internal testing

#because Kaggle only allows 10 submissions per day and I need to test many more times

smol_traindf, smol_testdf = model_selection.train_test_split(traindf, test_size=0.3)



#uncomment this line for a submission run!!!!!!!!!!!!!!!

#smol_traindf, smol_testdf = traindf, testdf



internal_test = False

if Y in smol_testdf.columns:

    internal_test = True

    test_answers = smol_testdf[Y]



if "cabin_section" not in smol_testdf.columns:

    smol_testdf = extract_cabin_section_feature(smol_testdf)



#filter non-predictive features

smol_traindf = smol_traindf[predictive_features | {Y}]

smol_testdf = smol_testdf[predictive_features]





#replace nan age values for only train set average age

#should we use mean instead?

avgAge = smol_traindf["age"].median()

smol_traindf["age"] = smol_traindf["age"].fillna(avgAge)

smol_testdf["age"] = smol_testdf["age"].fillna(avgAge)





#replace nan and 0 valures in fare

#should we use mean instead?

class_means = smol_traindf.pivot_table('fare', index='pclass', aggfunc='median')

#traindf[["fare", "pclass"]].groupby("pclass").hist(bins=10)

smol_traindf = fix_missing_fare_values(smol_traindf, class_means)

smol_testdf = fix_missing_fare_values(smol_testdf, class_means)





"""

@param feature categorical feature to encode as one hot

@param df pandas dataframe containing the feature

@returns pandas dataframe without feature, but with n-1 new one hot encoded features (0,1)

    representing feature

"""

def encode_one_hot(feature, df):

    if feature not in df.columns:

        return df

    #turns a categorical feature into n-1 dummies1

    dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(feature, axis=1)

    return df





#features we do not want to one hot encode because they are ordinal

ordinal_categorical_features = ["parch", "sibsp"]    



#change categorical features to one hot encoding

#does this trigger the curse of dimensionality?

for feature in categorical_features:

    if feature == Y or feature in ordinal_categorical_features:

        continue

    smol_traindf = encode_one_hot(feature, smol_traindf)

    smol_testdf = encode_one_hot(feature, smol_testdf)





"""

@param df pandas dataframe

@param Y target feature

@returns dataframe with all features linearly scaled into (0,1)

"""

def scale_features(df, Y):

    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    removed_Y = False

    if Y in df.columns:

        output = df[Y]

        df = df.drop(Y, axis=1)

        removed_Y = True

    X = scaler.fit_transform(df)

    df = pd.DataFrame(X, columns=df.columns, index=df.index)

    if removed_Y:

        df[Y] = output

    return df





#scale numerical data to be in the range [0,1]

smol_traindf = scale_features(smol_traindf, Y)

smol_testdf = scale_features(smol_testdf, Y)





#cabin_section == T and G are not so common, so sometimes, during internal testing, one of the sets

#    may have no occurence of these values, so we need to manually insert their one hot encoded columns

#cabin_section G comes right after section F

g_loc = smol_traindf.columns.get_loc("cabin_section_F") + 1

if "cabin_section_G" not in smol_testdf.columns:

    smol_testdf.insert(loc=g_loc, column="cabin_section_G", value=pd.Series(index=smol_testdf.index, dtype=np.float32).fillna(0.0))

if "cabin_section_G" not in smol_traindf.columns:

    smol_traindf.insert(loc=g_loc, column="cabin_section_G", value=pd.Series(index=smol_traindf.index, dtype=np.float32).fillna(0.0))

#cabin_section T comes right after section G

t_loc = smol_traindf.columns.get_loc("cabin_section_G") + 1

if "cabin_section_T" not in smol_testdf.columns:

    smol_testdf.insert(loc=t_loc, column="cabin_section_T", value=pd.Series(index=smol_testdf.index, dtype=np.float32).fillna(0.0))

if "cabin_section_T" not in smol_traindf.columns:

    smol_traindf.insert(loc=t_loc, column="cabin_section_T", value=pd.Series(index=smol_traindf.index, dtype=np.float32).fillna(0.0))



predictors = []

#predictors.append(lambda : neural_network.MLPClassifier(hidden_layer_sizes=(15, 15), max_iter=3000, early_stopping=True))

predictors.append(lambda : ensemble.RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1))

predictors.append(lambda : ensemble.GradientBoostingClassifier())

#predictors.append(lambda : sklearn.tree.DecisionTreeClassifier())

for predictor in predictors:

    #naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

    model = predictor()

    print(model)

    model.fit(smol_traindf.drop(Y, axis=1), smol_traindf[Y])

    predictions = model.predict(smol_testdf)

    #predictions = model.predict(smol_testdf.drop(Y, axis=1))



    if internal_test:

        #internal test on accuracy, not gonna submit to kaggle

        #TODO also compute F1 score

        correct = predictions == test_answers

        total = len(correct)

        print("Correctly predicted : {}%".format(len(correct[correct == True]) / total * 100))

        print("Incorrectly predicted : {}%\n".format(len(correct[correct == False]) / total * 100))

    else:

        best_output = pd.DataFrame({'passengerid': testdf.passengerid, 'survived': predictions})

        output.to_csv('my_submission.csv', index=False)

        print("Your submission was successfully saved!")
#feature selection

#lets see the corelation between features





corr_matrix = traindf.drop(Y, axis=1).copy()

for col in corr_matrix.columns:

    if corr_matrix[col].dtype == "O":

        corr_matrix[col] = corr_matrix[col].factorize(sort=True)[0]



corr_matrix = corr_matrix.corr(method="pearson")

sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f',

            cmap="YlGnBu", cbar=True, linewidths=0.5)

plt.title("pearson correlation")
from sklearn.ensemble import RandomForestClassifier



y = traindf["survived"]



#extracting new variables

traindf["family_size"] = traindf["parch"] + traindf["sibsp"]

traindf["fare_per_person"] = traindf["fare"]/(traindf["family_size"]+1)



features = ["pclass", "sex", "sibsp", "parch"]

X = pd.get_dummies(traindf[features])

X_test = pd.get_dummies(testdf[features])

print(X_test)

#print(np.isnan(X_test.age))



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'passengerid': testdf.passengerid, 'survived': predictions})

print((output == best_output)[Y])

#output.to_csv('my_submission.csv', index=False)

#print("Your submission was successfully saved!")
def proportion_survived(discrete_var):

    by_var = traindf.groupby([discrete_var, "survived"])

    #counts the amount of entries for each combination of values from survived and discrete_var

    table = by_var.size()

    #turns the series into a dataframe and indexes it by survived, discrete_var

    table = table.unstack()

    #get the rations by dividing values by total

    normedtable = table.div(table.sum(1), axis=0)

    return normedtable

    



import matplotlib.pylab as plt

discrete_vars = ["sex", "pclass", "embarked"]

fig1, axes1 = plt.subplots(3, 1)

for i in range(len(discrete_vars)):

    table = proportion_survived(discrete_vars[i])

    table.plot(kind='barh', stacked=True, ax=axes1[i])

fig1.show()
#plotting survived ratio considering gender and class

fig2, axes2 = plt.subplots(2,3)

genders = traindf.sex.unique()

classes = traindf.pclass.unique()



def normrgb(rgb):   #this converts rgb codes into the format matplotlib wants

    rgb = [float(x)/255 for x in rgb]

    return rgb



darkpink, lightpink =normrgb([255,20,147]), normrgb([255,182,193])

darkblue, lightblue = normrgb([0,0,128]),normrgb([135,206, 250])

for gender in genders:

    for pclass in classes:

        if gender=='male':

            colorscheme = [lightblue, darkblue] #blue for boys

            row=0

        else:

            colorscheme = [lightpink, darkpink] #pink for a girl

            row=1

        group = traindf[(traindf.sex==gender)&(traindf.pclass==pclass)]

        group = group.groupby(['embarked', 'survived']).size().unstack()

        group = group.div(group.sum(1), axis=0)

        group.plot(kind='barh', ax=axes2[row, (int(pclass)-1)], color=colorscheme, stacked=True, legend=False).set_title('Class '+ str(pclass)).axes.get_xaxis().set_ticks([])

        

plt.subplots_adjust(wspace=0.4, hspace=1.5)

fhandles, flabels = axes2[1,2].get_legend_handles_labels()

mhandles, mlabels = axes2[0,2].get_legend_handles_labels()

plt.figlegend(fhandles, ('die', 'live'), title='Female', loc='center', bbox_to_anchor=(0.06, 0.40, 1.1, .102))



plt.figlegend(mhandles, ('die', 'live'), 'center', title='Male',bbox_to_anchor=(-0.15, 0.40, 1.1, .102))



fig2.show()