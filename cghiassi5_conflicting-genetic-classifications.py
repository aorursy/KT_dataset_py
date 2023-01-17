import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import sklearn as sk

import fractions

import re

import seaborn as sns
dataframe = pd.read_csv("../input/clinvar_conflicting.csv", dtype={0: object, 38: str, 40: object})
print(dataframe.shape)

dataframe.head()
#plotting a histogram of the different values of class to see if it's skewed

g = dataframe.groupby('CLASS').size()

g.plot(kind = 'bar')

proportion_conflicting = g[1]/g.sum()

print(g)



print("The fraction of classifications that are conflicting is {}".format(proportion_conflicting))
#Testing Categorical Variables



grouped = dataframe.groupby('CLNVC')

grouped_class = grouped['CLASS'].agg(np.mean)



print("Proportion of conflicting classifications by mutation {} \n".format(

    grouped_class))



#comparing proportion of each mutation conflicting to average

print("Proportion of conflicting classifications by mutation compared to average for data set {}\n".format(

    grouped_class.apply(lambda grouped_class: grouped_class - proportion_conflicting)))

grouped.size()



grouped_class.plot(kind = "bar", ylim = [0,1], title = 'Proportion of conflicting classifications by mutation', figsize = (20,10))

#IMPACT



grouped_impact = dataframe.groupby('IMPACT')

grouped_impact_class = grouped_impact['CLASS'].agg(np.mean)



print("Proportion of conflicting classifications by impact level {} \n \n".format(

    grouped_impact_class))



#comparing proportion of each IMPACT level conflicting to average

print("Proportion of conflicting classifications by impact level compared to average for data set{}\n".format(

    grouped_impact_class.apply(lambda grouped_impact_class: grouped_impact_class - proportion_conflicting)))



print(grouped_impact.size())



grouped_impact_class.plot(kind = "bar", ylim = [0,1], title = 'Proportion of conflicting classifications by impact level',

                          figsize = (20,10))
#creating function for correlation plot

def plot_corr(df,size=10):

    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.



    Input:

        df: pandas DataFrame

        size: vertical and horizontal size of the plot'''

    corr = df.corr()



    # Set up the matplotlib figure

    f, ax = plt.subplots(figsize=(20, 10))



    # Generate a custom diverging colormap

    cmap = sns.diverging_palette(220, 10, as_cmap=True)



    # Draw the heatmap with the mask and correct aspect ratio

    sns.heatmap(corr, cmap=cmap, vmax= 1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

    

#dropping all columns with over 20% NA

df=dataframe.replace({'':np.nan})

df = df.dropna(thresh=0.8*len(df), axis=1)

df.columns

#dropping all columns that clearly don't play a role in outcome (ex: clinical name of diseases)

df = df.drop(["CLNDISDB", "CLNDN", "Feature", 'Consequence', 'BIOTYPE', 'SYMBOL', 'Feature_type', 'ORIGIN'], axis = 1)

#removing all non numerical values

df_numeric = df.drop(['Amino_acids', 'Codons', 'MC', "CLNHGVS", 'REF', 'ALT', 'CLNVC', 'Allele', 'IMPACT', 'CHROM'], axis = 1)



#converting variables with numeric values listed as strings to numeric

for i in ["Protein_position", "CDS_position", "cDNA_position"]:

    df_numeric[i] = pd.to_numeric(df_numeric[i], errors = 'coerce')

    



#converting EXON to numeric values

df_numeric.EXON.fillna('0', inplace=True)

df_numeric['variant_exon'] = df_numeric.EXON.apply(lambda x: [float(s) for s in re.findall(r'\b\d+\b', x)][0])

df_numeric = df_numeric.drop(["EXON"], axis = 1)



df_numeric.dropna(axis = 0, inplace = True)

df_numeric.head()
plot_corr(df_numeric, size = 20)
#splitting data into training and test sets

from sklearn.model_selection import train_test_split

df_numeric_predictors = df_numeric.drop(["CLASS"], axis = 1)

df_numeric_outcome = df_numeric["CLASS"]

X_train, X_test, y_train, y_test = train_test_split(df_numeric_predictors, df_numeric_outcome, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

#training the model using training sets

regr = LogisticRegression().fit(X_train, y_train)

#testing the model using testing sets

y_pred = regr.predict(X_test)



y_array = np.array(y_test)

print('Accuracy score: %.2f' % accuracy_score(y_array, y_pred))

from sklearn import svm

clf = svm.SVC(gamma = 0.001)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Accuracy score: %.2f' % accuracy_score(y_test, y_pred))