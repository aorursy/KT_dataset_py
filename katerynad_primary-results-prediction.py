import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
%matplotlib inline
#-----------------------------------------------------------------------------------------------
def prediction_ols (X_train, Y_train, X_test, Y_test,normalize):
    """
    The function gets train and test data sets, normalize flag (True/False)
    and returns predictions for test and train data sets as well as coefficients
    based on ordinary least squares prediction method
    """
    # Print shapes of the training and testing data sets
    print ("Shapes of the training and testing data sets")
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


    #Create our regression object
    lreg = LinearRegression(normalize=normalize)

    #do a linear regression, except only on the training
    lreg.fit(X_train,Y_train)

    print("The estimated intercept coefficient is %.2f " %lreg.intercept_)
    print("The number of coefficients used was %d " % len(lreg.coef_))



    # Set a DataFrame from the Facts
    coeff_df = DataFrame(X_train.columns)
    coeff_df.columns = ["Fact"]


    # Set a new column lining up the coefficients from the linear regression
    coeff_df["Coefficient"] = pd.Series(lreg.coef_)


    # Show
    #coeff_df

    #highest correlation between a fact and fraction votes
    print ("Highest correlation fact: %s is %.9f" % (cf_dict.loc[coeff_df.iloc[coeff_df["Coefficient"].idxmax()]["Fact"],"description"], coeff_df.iloc[coeff_df["Coefficient"].idxmax()]["Coefficient"]) )
    #sns_plot = sns.jointplot(coeff_df.iloc[coeff_df["Coefficient"].idxmax()]["Fact"],"Fraction Votes",pd.merge(X_test,pd.DataFrame(Y_test), right_index=True, left_index=True),kind="scatter")


    #Predictions on training and testing sets
    pred_train = lreg.predict(X_train)
    pred_test = lreg.predict(X_test)

    # The mean square error
    print("Fit a model X_train, and calculate MSE with Y_train: %.6f"  % np.mean((Y_train - pred_train) ** 2))
    print("Fit a model X_test, and calculate MSE with Y_test: %.6f"  %np.mean((Y_test - pred_test) ** 2))
    #Explained variance score: 1 is perfect prediction
    print("Variance score: %.2f" % lreg.score(X_test, Y_test))

    
    return pred_test,coeff_df,pred_train
#-----------------------------------------------------------------------------------------------
def prediction_ridge (X_train, Y_train, X_test, Y_test,alpha,normalize):
    """
    The function gets train and test data sets, normalize flag (True/False)
    and returns predictions for test and train data sets as well as coefficients
    based on Ridge method
    """
    # Print shapes of the training and testing data sets
    print ("Shapes of the training and testing data sets")
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    #Create our regression object

    lreg = Ridge (alpha = alpha,normalize=normalize)

    #do a linear regression, except only on the training
    lreg.fit(X_train,Y_train)

    print("The estimated intercept coefficient is %.2f " %lreg.intercept_)
    print("The number of coefficients used was %d " % len(lreg.coef_))



    # Set a DataFrame from the Facts
    coeff_df = DataFrame(X_train.columns)
    coeff_df.columns = ["Fact"]


    # Set a new column lining up the coefficients from the linear regression
    coeff_df["Coefficient"] = pd.Series(lreg.coef_)


    # Show
    #coeff_df

    #highest correlation between a fact and fraction votes
    print ("Highest correlation fact: %s is %.9f" % (cf_dict.loc[coeff_df.iloc[coeff_df["Coefficient"].idxmax()]["Fact"],"description"], coeff_df.iloc[coeff_df["Coefficient"].idxmax()]["Coefficient"]) )

    #sns_plot = sns.jointplot(coeff_df.iloc[coeff_df["Coefficient"].idxmax()]["Fact"],"Fraction Votes",pd.merge(X_test,pd.DataFrame(Y_test), right_index=True, left_index=True),kind="scatter")


    #Predictions on training and testing sets
    pred_train = lreg.predict(X_train)
    pred_test = lreg.predict(X_test)

    # The mean square error
    print("Fit a model X_train, and calculate MSE with Y_train: %.6f"  % np.mean((Y_train - pred_train) ** 2))
    print("Fit a model X_test, and calculate MSE with Y_test: %.6f"  %np.mean((Y_test - pred_test) ** 2))

    #Explained variance score: 1 is perfect prediction
    print("Variance score: %.2f" % lreg.score(X_test, Y_test))

    return pred_test,coeff_df,pred_train
#-----------------------------------------------------------------------------------------------
def prediction_lasso (X_train, Y_train, X_test, Y_test,alpha,normalize):
    """
    The function gets train and test data sets, normalize flag (True/False)
    and returns predictions for test and train data sets as well as coefficients
    based on Lasso method
    """
    # Print shapes of the training and testing data sets
    print ("Shapes of the training and testing data sets")
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    #Create our regression object

    lreg = Lasso (alpha = alpha,normalize=normalize)

    #do a linear regression, except only on the training
    lreg.fit(X_train,Y_train)

    print("The estimated intercept coefficient is %.2f " %lreg.intercept_)
    print("The number of coefficients used was %d " % len(lreg.coef_))



    # Set a DataFrame from the Facts
    coeff_df = DataFrame(X_train.columns)
    coeff_df.columns = ["Fact"]


    # Set a new column lining up the coefficients from the linear regression
    coeff_df["Coefficient"] = pd.Series(lreg.coef_)


    # Show
    #coeff_df

    #highest correlation between a fact and fraction votes
    print ("Highest correlation fact: %s is %.9f" % (cf_dict.loc[coeff_df.iloc[coeff_df["Coefficient"].idxmax()]["Fact"],"description"], coeff_df.iloc[coeff_df["Coefficient"].idxmax()]["Coefficient"]) )

    #sns_plot = sns.jointplot(coeff_df.iloc[coeff_df["Coefficient"].idxmax()]["Fact"],"Fraction Votes",pd.merge(X_test,pd.DataFrame(Y_test), right_index=True, left_index=True),kind="scatter")


    #Predictions on training and testing sets
    pred_train = lreg.predict(X_train)
    pred_test = lreg.predict(X_test)

    # The mean square error
    print("Fit a model X_train, and calculate MSE with Y_train: %.6f"  % np.mean((Y_train - pred_train) ** 2))
    print("Fit a model X_test, and calculate MSE with X_test and Y_test: %.6f"  %np.mean((Y_test - pred_test) ** 2))

    #Explained variance score: 1 is perfect prediction
    print("Variance score: %.2f" % lreg.score(X_test, Y_test))

    return pred_test,coeff_df,pred_train
#-----------------------------------------------------------------------------------------------
def prediction_BayesianRidge (X_train, Y_train, X_test, Y_test,normalize):
    """
    The function gets train and test data sets, normalize flag (True/False)
    and returns predictions for test and train data sets as well as coefficients
    based on BayesianRidge method
    """
    # Print shapes of the training and testing data sets
    print ("Shapes of the training and testing data sets")
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    #Create our regression object

    lreg = BayesianRidge(normalize=normalize)

    #do a linear regression, except only on the training
    lreg.fit(X_train,Y_train)

    print("The estimated intercept coefficient is %.2f " %lreg.intercept_)
    print("The number of coefficients used was %d " % len(lreg.coef_))



    # Set a DataFrame from the Facts
    coeff_df = DataFrame(X_train.columns)
    coeff_df.columns = ["Fact"]


    # Set a new column lining up the coefficients from the linear regression
    coeff_df["Coefficient"] = pd.Series(lreg.coef_)


    # Show
    #coeff_df

    #highest correlation between a fact and fraction votes
    print ("Highest correlation fact: %s is %.9f" % (cf_dict.loc[coeff_df.iloc[coeff_df["Coefficient"].idxmax()]["Fact"],"description"], coeff_df.iloc[coeff_df["Coefficient"].idxmax()]["Coefficient"]) )

    #sns_plot = sns.jointplot(coeff_df.iloc[coeff_df["Coefficient"].idxmax()]["Fact"],"Fraction Votes",pd.merge(X_test,pd.DataFrame(Y_test), right_index=True, left_index=True),kind="scatter")


    #Predictions on training and testing sets
    pred_train = lreg.predict(X_train)
    pred_test = lreg.predict(X_test)

    # The mean square error
    print("MSE with X_train and Y_train: %.6f"  % np.mean((Y_train - pred_train) ** 2))
    print("MSE with X_test and Y_test: %.6f"  %np.mean((Y_test - pred_test) ** 2))

    #Explained variance score: 1 is perfect prediction
    print("Variance score: %.2f" % lreg.score(X_test, Y_test))

    return pred_test,coeff_df,pred_train
#-----------------------------------------------------------------------------------------------
def residual_plot(Y_train, pred_train, Y_test, pred_test, candidate, test_dataset_name, train_dataset_name,method):
    """
    The function builds the residual plot to show the difference between the observed value 
    of the dependent variable (y) and the predicted value (ŷ)
    """
    
    # Scatter plot the training data
    train = plt.scatter(pred_train,(Y_train-pred_train),c="b",alpha=0.5)
    # Scatter plot the testing data
    test = plt.scatter(pred_test,(Y_test-pred_test),c="r",alpha=0.5)
    # Plot a horizontal axis line at 0
    plt.hlines(y=0,xmin=-0.5,xmax=1)

    #Labels
    plt.legend((train,test),("Train","Test"),loc="lower left")
    plt.title("Residual Plots for %s  using %s method " % (candidate, method) )
   
#-----------------------------------------------------------------------------------------------
def vis_results(Y_train, pred_train, Y_test, pred_test, candidate, test_dataset_name, train_dataset_name,method):
    """
    The function builds residual and joinplots for the predicted results
    """
    #prediction in csv
    CandidateFractionPrediction=pd.DataFrame(Y_test)
    CandidateFractionPrediction["Prediction"]=pred_test
    CandidateFractionPrediction=pd.merge(CandidateFractionPrediction,facts[["area_name","state_abbreviation"]], right_index=True, left_index=True)
    CandidateFractionPrediction=CandidateFractionPrediction.reset_index()
    CandidateFractionPrediction.columns=["fips", "Fraction Votes", "Prediction", "county", "state"]
    CandidateFractionPrediction[CandidateFractionPrediction["state"]=="CA"]
    
    #joinplot
    sns_plot = sns.jointplot("Fraction Votes","Prediction",CandidateFractionPrediction,kind="scatter")
    #residual plot
    plt.close()
    residual_plot(Y_train, pred_train, Y_test, pred_test, candidate, test_dataset_name, train_dataset_name,method)
    
    #-----------------------------------------------------------------------------------------------
def get_data(candidate):
    """
    Builds the source data for a candidate (name, string)
    """
    # source data
    pr=pd.read_csv("../input/primary_results.csv")
    #pivoting and drop Null values for clean and easy analysis
    pr_piv= pr[["fips", "candidate","fraction_votes"]].pivot_table(index="fips", columns="candidate", values="fraction_votes")
    pr_piv.drop(" No Preference", axis=1, inplace=True)
    pr_piv.drop(" Uncommitted", axis=1, inplace=True)

    #merge fraction votes and facts to have a complete data set for all counties for each candidate
    pr_facts=pd.merge(pr_piv, facts, right_index=True, left_index=True)

    Candidate_data=pr_facts[[candidate,"PST045214", "PST040210", "PST120214", "POP010210", "AGE135214","AGE295214", "AGE775214", "SEX255214", "RHI125214", "RHI225214","RHI325214", "RHI425214", "RHI525214", "RHI625214", "RHI725214","RHI825214", "POP715213", "POP645213", "POP815213","VET605213", "LFE305213", "HSG010214", "HSG445213","HSG096213", "HSG495213", "HSD410213", "HSD310213", "INC910213","INC110213", "PVY020213", "BZA010213", "BZA110213", "BZA115213","NES010213", "SBO001207", "SBO315207", "SBO115207", "SBO215207","SBO515207", "SBO415207", "SBO015207", "MAN450207", "WTN220207","RTN130207", "RTN131207", "AFN120207", "BPS030214", "LND110210","POP060210"]]
    Candidate_data=Candidate_data.dropna()
    Candidate_data.columns=["Fraction Votes","PST045214", "PST040210", "PST120214", "POP010210", "AGE135214","AGE295214", "AGE775214", "SEX255214", "RHI125214", "RHI225214","RHI325214", "RHI425214", "RHI525214", "RHI625214", "RHI725214","RHI825214", "POP715213", "POP645213", "POP815213","VET605213", "LFE305213", "HSG010214", "HSG445213","HSG096213", "HSG495213", "HSD410213", "HSD310213", "INC910213","INC110213", "PVY020213", "BZA010213", "BZA110213", "BZA115213","NES010213", "SBO001207", "SBO315207", "SBO115207", "SBO215207","SBO515207", "SBO415207", "SBO015207", "MAN450207", "WTN220207","RTN130207", "RTN131207", "AFN120207", "BPS030214", "LND110210","POP060210"]
    Candidate_fractions=Candidate_data["Fraction Votes"]
    Candidate_facts=Candidate_data.drop("Fraction Votes", 1)

    return Candidate_facts,Candidate_fractions
#-----------------------------------------------------------------------------------------------
def run(candidate):
    """
    Runs prediction and visualizes results for a candidate (name, string)
    """
    print ("----------%s PRIMARY RESULTS PREDICTION------------" %candidate)

    test_dataset_name="Test part"
    train_dataset_name="Train part"

    #get source data
    Candidate_facts,Candidate_fractions=get_data(candidate)

    #separate to training and test data sets
    X_train, X_test, Y_train, Y_test = train_test_split(Candidate_facts,Candidate_fractions)

    #Prediction
    #multi
    print ("----------------------Ordinary least square  ------------")
    pred_test,coeff_df,pred_train=prediction_ols (X_train, Y_train, X_test, Y_test,False)
    vis_results(Y_train, pred_train, Y_test, pred_test, candidate, test_dataset_name, train_dataset_name,'ols')
    
    print ("------------------------------Ridge 0.01 Normalize ------------")
    pred_test,coeff_df,pred_train=prediction_ridge(X_train, Y_train, X_test, Y_test, 0.01,False)
    #vis_results(Y_train, pred_train, Y_test, pred_test, candidate, test_dataset_name, train_dataset_name,'ridge')
    print ("------------------------------Lasso 0.0001-----------------------")
    pred_test,coeff_df,pred_train=prediction_lasso(X_train, Y_train, X_test, Y_test, 0.0001,True)
    #vis_results(Y_train, pred_train, Y_test, pred_test, candidate, test_dataset_name, train_dataset_name,'lasso')
    print ("---------------------------BayesianRidge-----------------------")
    pred_test,coeff_df,pred_train=prediction_BayesianRidge(X_train, Y_train, X_test, Y_test,False)
    #vis_results(Y_train, pred_train, Y_test, pred_test, candidate, test_dataset_name, train_dataset_name,'br')

#facts and  dictionary
cf_dict=pd.read_csv("../input/county_facts_dictionary.csv")
cf_dict=cf_dict.set_index("column_name")
facts=pd.read_csv("../input/county_facts.csv")
facts=facts.set_index("fips")
print ("Hillary Clinton and Bernie Sanders fraction votes are most correlated with the county facts. The variance is above 0.5-0.6 for these 2 candidates.") 
print ("The mean square errors between the training and testing data sets are pretty close")
print ("No structure or pattern in the residual plots")
print (".........................................................................................")
print ("The quality of the predicted values for the rest of the candidates is low with 0.2 - 0.4 and less varience values.")
print (".........................................................................................")
print ("Ordinary least squares method works perfectly fine fo the data. The rest of the method can give a slightly better results but not very significant")
print ("Lasso method works better with normalized data")
print ("=========================================================================================")
print ("Please see more details for Hillary Clinton as example below")
print ("=========================================================================================")
run("Hillary Clinton")
#run("Bernie Sanders")
#run("Donald Trump")
#run("Marco Rubio")
#run("Ted Cruz")