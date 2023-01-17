import pandas as pd

import seaborn as sns

import numpy as np

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import r2_score

from matplotlib import pyplot as plt
filepath = '../input/Admission_Predict_Ver1.1.csv'

df = pd.read_csv(filepath)

df.columns
df.rename(columns = {'Chance of Admit ': 'Chance of Admit', 'LOR ': 'LOR'}, inplace = True)

df.columns
corr = df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(10, 8))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center= .6,

            square=True, linewidths=2, cbar_kws={"shrink": .75})

plt.savefig(fname = 'corr graph', dpi = 400)

plt.show()
def isValuable(df, compareTo, plim, pcoefLim):

    """

    Determines whether certain elements of inputed DataFrame are statistically relevant to the 'comparedto' column

    df = DataFrame

    compareTo = column to compare

    plim = limit on p value to be considered relevant

    pcoefLim = limit on pcoef value to be considered relevant

    """

    dictDf = df.to_dict('series')

    effectors = []

    for column in df:

        if (df[column].dtype != 'object'):

            if not (df[column].equals(df[compareTo])):

                pcoef, p = stats.pearsonr(df[compareTo], df[column])

                print("{} compared to {}".format(column, compareTo))

                print(("The pcoef is: {}\nThe p_value is: {}").format(pcoef, p))

                if (pcoef > pcoefLim) or (pcoef < -pcoefLim) and (p < plim):

                    effectors.append(column)

                    sns.regplot(x = df[column], y = df[compareTo], data=df, marker = '.')

                    plt.ylim(0,)

                    plt.show()

                    print(".......ADDED.......")

                else:

                    print(".....NOT ADDED.....")

                print("")

        else:

            columnName = df[column].name

            grouped = df[[columnName, compareTo]].groupby(columnName)

            uniqueElements = dictDf[column].unique()

            elements = []

            for thing in uniqueElements:

                try:

                    specialGroup = grouped.get_group(thing)[compareTo]

                    elements.append(specialGroup)

                except:

                    pass

            f, p = stats.f_oneway(*elements)

            print(("After ANOVA Analysis of {} compared to {}:\nF Value: {}\nP Value: {}").format(columnName, compareTo, f, p))

            if (p < plim):

                effectors.append(column)

                sns.boxplot(x= df[column], y=df[compareTo], data=df)

                plt.show()

                print(".......ADDED.......")

            print("")

    return effectors

effectors = isValuable(df, 'Chance of Admit', .01, .5)

print('Valuable:', effectors)

X = df[effectors]

y = df['Chance of Admit']
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = .3, random_state = 1)

MLR = LinearRegression()

MLR.fit(XTrain, yTrain)

r2 = MLR.score(XTest, yTest)

print(("Split R^2 Score: {:.5}").format(r2))

MLR.fit(X, y)

r2 = MLR.score(X, y)

print(("Full R^2 Score:  {:.5}").format(r2))
#to avoid data conversion warnings

X = X.astype('float64')

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = .3, random_state = 1)

#creating pipeline

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(degree=3)), ('model',LinearRegression())]

pipe = Pipeline(Input)

pipe.fit(XTrain, yTrain)

yhat = pipe.predict(XTest)

#getting R^2

r2_p = r2_score(yTest, yhat)

print("R^2 Score: ", r2_p)
# "Perfect student"

input = [[340, 120, 1, 5, 5, 10, 1]]

#          1    2   3  4  5  6   7

# 1 - 'GRE Score' out of 340 

# 2 - 'TOEFL Score' out of 120

# 3 - 'University Rating' out of 5 (1 is top 20%, 5 bottom 20%)

# 4 - 'SOP' - Statement of Purpose strength out of 10

# 5 - 'LOR' - Letter of Rec strength out of 10

# 6 - 'CGPA'- Cumulative GPA out of 10

# 7 - 'Research' - research or no research 0 or 1

prediction = MLR.predict(input)

print("The odds that you get into a no. {} ranked school is: {:.3%}".format(input[0][2], prediction[0]))