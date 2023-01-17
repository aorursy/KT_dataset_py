#Import libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import sklearn



import sklearn.linear_model



from sklearn.preprocessing import LabelEncoder



from matplotlib import pyplot as plt



from sklearn.linear_model import LogisticRegression







#Load data

mushrooms_df = pd.read_csv('../input/mushrooms.csv')



mushrooms_df.rename(columns={'class':'classes'}, inplace=True)



#del mushrooms_df['veil-type']







#Prepare data using LabelEncoder library

labelencoder=LabelEncoder()



for col in mushrooms_df.columns:

    mushrooms_df[col]=labelencoder.fit_transform(mushrooms_df[col])





''' 

Alternatively I used also a code of this type for all features of mushrooms's file

#Prepare data

total_rowsn = len(mushrooms_df.index)



#class: edible=e, poisonous=p

for i in range(0, total_rowsn):

    if mushrooms_df.iloc[i, 0] == 'e':

        mushrooms_df.iloc[i, 0] = 1  # edible

    else:mushrooms_df.iloc[i, 0] = 0  # poisonous



and so on ....



'''





#correlation matrix

corr = mushrooms_df.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})





plt.show(block=True)



'''

Plotting the correlation 

matrix we discover thatthe vail-type is a feature 

not relevat for our classification. Indeed its values 

are all "p" and this doesn't contribute to our predictive model.



The correaltion matrix helps to discover this type of 

feature without a previous knowledge of the source data file.



'''



#split train and test data

train_df = mushrooms_df.sample(frac=0.7,random_state=200)



test_df = mushrooms_df.drop(train_df.index)



test_rowsn = len(test_df.index)



train_rowsn = len(train_df.index)





X_train = train_df.drop("classes", axis=1)



Y_train = train_df.iloc[:, 0]



X_test = test_df.drop("classes", axis=1)



Y_test = test_df.iloc[:, 0]





#Logistic Regression:



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)



# accuracy of the Logisti Regression algorithm

print ("accuracy:")



print(acc_log)





'''

we use the logistic regression to analyze the features and what are more important in classificaction



these are the most important features and we should use only them to check our algorithm



veil-color  6.099109

gill-size   6.812580



also they are in red color in the correlation matrix



'''

coeff_df = pd.DataFrame(train_df.columns.delete(0))



coeff_df.columns = ['Feature']



coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



result = coeff_df.sort_values(by=['Correlation'], ascending=[True])







print("Features importance:")



print(result)



#Original data

df_origin = pd.DataFrame(Y_test)



#Prediction result

df_pred = pd.DataFrame(Y_pred)





print("End")