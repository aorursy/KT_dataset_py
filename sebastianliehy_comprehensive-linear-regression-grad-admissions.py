import itertools

from tqdm import tqdm, tnrange, tqdm_notebook

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np 

import seaborn as sns

from scipy import stats

import statsmodels.api as sm

from statsmodels.formula.api import ols

from statsmodels.graphics.regressionplots import *

from sklearn.linear_model import LinearRegression, TheilSenRegressor

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error, r2_score

from IPython.core.interactiveshell import InteractiveShell





InteractiveShell.ast_node_interactivity = "all"  # make multiple printed outputs possible from one cell



# Read data



df = pd.read_csv("../input/Admission_Predict.csv")



df.shape  # small dataset



df.info()
df.head(n=10)
df.describe()
df.nunique()
df.drop(["Serial No."],axis=1,inplace=True)

coa = df[["Chance of Admit "]]

df.drop(["Chance of Admit "],axis=1,inplace=True)

df["Chance_of_Admit"] = coa

lor = df[["LOR "]]

df.drop(["LOR "],axis=1,inplace=True)

df["LOR"] = lor

gre = df[["GRE Score"]]

df.drop(["GRE Score"],axis=1,inplace=True)

df["GRE_Score"] = gre

toe = df[["TOEFL Score"]]

df.drop(["TOEFL Score"],axis=1,inplace=True)

df["TOEFL_Score"] = toe

ur = df["University Rating"] 

df.drop(["University Rating"],axis=1,inplace=True)

df["University_Rating"] = ur
df.head()
def heatMap(df, mirror):



   # Create Correlation df

   corr = df.corr()

   # Plot figsize

   fig, ax = plt.subplots(figsize=(10, 10))

   # Generate Color Map

   colormap = sns.diverging_palette(220, 10, as_cmap=True)

   

   if mirror == True:

      #Generate Heat Map, allow annotations and place floats in map

      sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")

      #Apply xticks

      plt.xticks(range(len(corr.columns)), corr.columns);

      #Apply yticks

      plt.yticks(range(len(corr.columns)), corr.columns)

      #show plot



   else:

      # Drop self-correlations

      dropSelf = np.zeros_like(corr)

      dropSelf[np.triu_indices_from(dropSelf)] = True

      # Generate Color Map

      colormap = sns.diverging_palette(220, 10, as_cmap=True)

      # Generate Heat Map, allow annotations and place floats in map

      sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)

      # Apply xticks

      plt.xticks(range(len(corr.columns)), corr.columns);

      # Apply yticks

      plt.yticks(range(len(corr.columns)), corr.columns)

   # show plot

   plt.show()



heatMap(df,True)
sns.pairplot(df)
## plots to help ascertain if linear regression assumptions being violated



full_X = df.drop(["Chance_of_Admit"],axis=1)

full_y = df[["Chance_of_Admit"]]





def LR_assump(X,y):

    cX = sm.add_constant(X)

    ols_model = sm.OLS(y, cX)



    ols_results = ols_model.fit()

    infl = ols_results.get_influence()

    measuredf = infl.summary_frame()

    Tresid = measuredf["student_resid"]

    residuals = ols_results.resid

    fig, axs = plt.subplots(2, 2,figsize=(20,20))

    axs[0,0].set_title('Normal QQ Plot')

    qq = stats.probplot(residuals,dist="norm",plot=axs[0,0])  # to suppress array being printed

    axs[0,1].set_title("Index Plot of Studentised Residuals")

    axs[0,1].set_ylabel('Tresid')

    axs[0,1].set_xlabel('Index')

    axs[0,1].axhline(y=2,xmin=0,xmax=500)  # cutoffs for studentised resid, find outliers this way

    axs[0,1].axhline(y=-2,xmin=0,xmax=500)

    sns.scatterplot(np.arange(len(Tresid)), Tresid, ax=axs[0,1])

    axs[1,0].set_title("Histogram of Studentised Residuals")

    axs[1,0].set_xlabel('Tresid')

    axs[1,0].set_ylabel('Frequency')

    sns.distplot(Tresid,ax=axs[1,0])

    fv = ols_results.fittedvalues

    axs[1,1].set_title("Plot of Studentised Residuals vs Fitted Values")

    axs[1,1].set_ylabel('Tresid')

    axs[1,1].set_xlabel('Fitted Values')

    axs[1,1].axhline(y=2,xmin=0,xmax=500)  # cutoffs for studentised resid, find outliers this way

    axs[1,1].axhline(y=-2,xmin=0,xmax=500)

    sns.scatterplot(fv, Tresid, ax=axs[1,1])

LR_assump(full_X,full_y)
# Measures to determine influential points

m = ols(formula='Chance_of_Admit ~ GRE_Score + TOEFL_Score + SOP + LOR + CGPA + Research + University_Rating', data=df).fit()

infl = m.get_influence()

fig, axs = plt.subplots(2, 2,figsize=(20,20))



# Set up values

(c, p) = infl.cooks_distance

(d, p) = infl.dffits

lev = infl.hat_matrix_diag





axs[0,0].set_title('Index Plots of Cook\'s Distance')

axs[0,0].set_ylabel('Cook\'s Distance Value')

axs[0,0].set_xlabel('Index')

sns.scatterplot(np.arange(len(c)), c, ax=axs[0,0])



axs[0,1].set_title('Index Plots of DFITS')

axs[0,1].set_ylabel('DFITS Value')

axs[0,1].set_xlabel('Index')

dfitsCutOff = 2*(np.sqrt((df.shape[1])/(df.shape[0]-df.shape[1]-2)))

dfitsCutOff

axs[0,1].axhline(y=0.255,xmin=0,xmax=500)

sns.scatterplot(np.arange(len(d)), d, ax=axs[0,1])





axs[1,0].set_title('Index Plots of Leverage Values')

levcutoff = 2*(7+1)/full_X.shape[0]

axs[1,0].set_ylabel('Leverage Values')

axs[1,0].set_xlabel('Index')

axs[1,0].axhline(y=levcutoff,xmin=0,xmax=500)

sns.scatterplot(np.arange(len(lev)), lev, ax=axs[1,0])



Pi = lev/(1-lev)

SSE = m.mse_total * len(full_y)

di = m.resid/(SSE**(1/2))

Di=(7+1)/(1-lev)*(di**2/(1-di**2))





axs[1,1].set_title('Potential Residual Plot')

axs[1,1].set_xlabel('Potential Function Values')

axs[1,1].set_ylabel('Scaled Residuals')

sns.scatterplot(Pi, Di, ax=axs[1,1])
# tank yew stats models 



fig = plt.figure(figsize=(12, 20))

fig = sm.graphics.plot_ccpr_grid(m, fig=fig)
# Full model



X = df.drop(["Chance_of_Admit"],axis=1)

y = df[["Chance_of_Admit"]]



lr = LinearRegression()

lr.fit(X,y)

pred_y = lr.predict(X)

r2_score(y, pred_y)  # initial r2 score



# Build step forward feature selection



lr = LinearRegression()



sfs1 = SFS(estimator=lr, 

           k_features=(1, 7),

           forward=True, 

           floating=False, 

           scoring='r2',

           cv=10)



sfs1.fit(X, y)



print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))

print('all subsets:\n', sfs1.subsets_)

plot_sfs(sfs1.get_metric_dict(), kind='std_err');   # lower R^2 than above with cross validation
# may not be accurate mallows cp 



full_X = df.drop(["Chance_of_Admit"],axis=1)

full_y = df[["Chance_of_Admit"]]

fX_train, fX_test, fy_train, fy_test = train_test_split(full_X,y, test_size=0.3,train_size=0.7,random_state=42)





def Cp(y,pred,p,n,model):

    model.fit(full_X,full_y.values.ravel())

    SSEfull = mean_squared_error(full_y,model.predict(full_X)) * len(full_y)   # subset Sum squared error

    sigma_hat_squared = SSEfull/(len(full_y)-len(full_X.columns)-1)

    SSEp = mean_squared_error(y,pred) * len(y)

    return (SSEp/sigma_hat_squared) - n + 2*p    ## closer to 0 the better 

    


def LRfit(X,y):

    #Fit linear regression model and return RSS and R squared values

    

    lr = LinearRegression(fit_intercept = True)

    lr.fit(X,y)

    pred = lr.predict(X)

    RSS = mean_squared_error(y,pred) * len(y)

    SSE = ((y-pred)**2)["Chance_of_Admit"].sum()

    SST = ((y-y.mean())**2)["Chance_of_Admit"].sum()

    R_squared = 1 -(SSE/SST)

    adjR_sq = 1 - (1-R_squared)*((X.shape[0]-1)/(X.shape[0]-len(X.columns)-1))

    p = len(X.columns)

    n = len(y)

    cp = abs(Cp(y,pred,p,n,LinearRegression(fit_intercept = True)) - p - 1) 

    return RSS,SSE, R_squared, adjR_sq, cp





#Init variables



RSS_list, SSE_list, R_squared_list, adjR_squared_list, Cp_list, feature_list = [],[],[],[],[],[]

num_features = []



#Looping over k = 1 to k = 7 features in X

for k in tnrange(1,len(full_X.columns) + 1, desc = 'Loop...'):



    #Looping over all possible combinations

    for combo in itertools.combinations(full_X.columns,k):

        results = LRfit(full_X[list(combo)],full_y)

        RSS_list.append(results[0])

        SSE_list.append(results[1])                  #Append lists

        R_squared_list.append(results[2])

        adjR_squared_list.append(results[3])

        Cp_list.append(results[4])

        feature_list.append(combo)

        num_features.append(len(combo))   



#Store in DataFrame

LRdf = pd.DataFrame({'num_features': num_features,"RSS": RSS_list,'SSE': SSE_list,'R_squared':R_squared_list,

                     'Adj R_squared':adjR_squared_list,"|Cp-p-1|": Cp_list,'features':feature_list})

LRdf.sort_values("|Cp-p-1|",axis=0,ascending=True)
## Define function to fit Theil Sen for easier, repeatable subset fitting



def TSfit(X,y):

    ts = TheilSenRegressor(random_state=0)

    ts.fit(X,y.values.ravel())

    pred = ts.predict(X)

    SSE = mean_squared_error(y.values.ravel(),pred) * len(y)

    SST = ((y-y.mean())**2)["Chance_of_Admit"].sum()

    R_squared = 1 -(SSE/SST)

    adjR_sq = 1 - (1-R_squared)*((X.shape[0]-1)/(X.shape[0]-len(X.columns)-1))

    p = len(X.columns)

    n = len(y)

    cp = abs(Cp(y,pred,p,n,TheilSenRegressor(random_state=0)) - p - 1) 

    return SSE, R_squared, adjR_sq, cp





#Init variables



SSE_list, R_squared_list, adjR_squared_list, Cp_list, feature_list = [],[],[],[],[]

num_features = []



#Looping over k = 1 to k = 7 features in X

for k in tnrange(1,len(full_X.columns) + 1, desc = 'Loop...'):



    #Looping over all possible combinations

    for combo in itertools.combinations(full_X.columns,k):

        results = TSfit(full_X[list(combo)],y)

        

        SSE_list.append(results[0])                  #Append lists

        R_squared_list.append(results[1])

        adjR_squared_list.append(results[2])

        Cp_list.append(results[3])

        feature_list.append(combo)

        num_features.append(len(combo))   



#Store in DataFrame

TSdf = pd.DataFrame({'num_features': num_features,'SSE': SSE_list,'R_squared':R_squared_list, 'Adj R_squared':adjR_squared_list,"|Cp-p-1|": Cp_list,'features':feature_list})

TSdf.sort_values("|Cp-p-1|",axis=0,ascending=True)

full_X = df.drop(["Chance_of_Admit"],axis=1)

full_y = df[["Chance_of_Admit"]]**2



LR_assump(full_X,full_y)
RSS_list, SSE_list, R_squared_list, adjR_squared_list, Cp_list, feature_list = [],[],[],[],[],[]

num_features = []



#Looping over k = 1 to k = 7 features in X

for k in tnrange(1,len(full_X.columns) + 1, desc = 'Loop...'):



    #Looping over all possible combinations

    for combo in itertools.combinations(full_X.columns,k):

        results = LRfit(full_X[list(combo)],full_y)

        RSS_list.append(results[0])

        SSE_list.append(results[1])                  #Append lists

        R_squared_list.append(results[2])

        adjR_squared_list.append(results[3])

        Cp_list.append(results[4])

        feature_list.append(combo)

        num_features.append(len(combo))   



#Store in DataFrame

LRdf = pd.DataFrame({'num_features': num_features,"RSS": RSS_list,'SSE': SSE_list,'R_squared':R_squared_list,

                     'Adj R_squared':adjR_squared_list,"|Cp-p-1|": Cp_list,'features':feature_list})

LRdf.sort_values("|Cp-p-1|",axis=0,ascending=True)
SSE_list, R_squared_list, adjR_squared_list, Cp_list, feature_list = [],[],[],[],[]

num_features = []



#Looping over k = 1 to k = 7 features in X

for k in tnrange(1,len(full_X.columns) + 1, desc = 'Loop...'):



    #Looping over all possible combinations

    for combo in itertools.combinations(full_X.columns,k):

        results = TSfit(full_X[list(combo)],full_y)

        

        SSE_list.append(results[0])                  #Append lists

        R_squared_list.append(results[1])

        adjR_squared_list.append(results[2])

        Cp_list.append(results[3])

        feature_list.append(combo)

        num_features.append(len(combo))   



#Store in DataFrame

TSdf = pd.DataFrame({'num_features': num_features,'SSE': SSE_list,'R_squared':R_squared_list, 'Adj R_squared':adjR_squared_list,"|Cp-p-1|": Cp_list,'features':feature_list})

TSdf.sort_values("|Cp-p-1|",axis=0,ascending=True)

X = df.drop(["Chance_of_Admit"],axis=1)

y = df[["Chance_of_Admit"]]



X = X[["CGPA", "LOR", "GRE_Score", "TOEFL_Score"]]



ts = TheilSenRegressor(random_state=0)

ts.fit(X,y.values.ravel())

ts.coef_