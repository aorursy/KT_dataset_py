# Loading the packages

import numpy as np

import pandas as pd 

from sklearn.preprocessing import StandardScaler

from scipy import stats

import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

import seaborn as sns 

from sklearn.cluster import KMeans

from sklearn.feature_selection import RFE

sns.set(style="ticks", color_codes=True)
# Loading the training dataset

df_train = pd.read_csv("../input/train.csv")
y = df_train["target"]

# We exclude the target and id columns from the training dataset

df_train.pop("target");

df_train.pop("id")

colnames1 = df_train.columns
scaler = StandardScaler()

scaler.fit(df_train)

X = scaler.transform(df_train)

df_train = pd.DataFrame(data = X, columns=colnames1)   # df_train is standardized 
normal_vars = [] # This list will contain the explanatory variables 

                 # that follow a normal distribution



counter = 0 

for c in df_train.columns:

    

    df = df_train[c]

    index_0 = y == 0.0

    index_1 = y == 1.0 

    df_0 = df[index_0]

    df_1 = df[index_1]

    

    if stats.kstest(df_0, 'norm')[1] >= 0.5 and stats.kstest(df_1, 'norm')[1] >= 0.5:        

        print("Variable: {} follows a normal distribution on condition of the classes".format(c))                

        counter += 1

        normal_vars.append(c)

        

        

print("Number of variables that follow a normal distribution {}".format(counter))

        

        
def plot_hist(list_var):

    

    for l in list_var:

        df = df_train[l]

        index_0 = y == 0.0

        index_1 = y == 1.0 

        df_0 = df[index_0]

        df_1 = df[index_1]



        

        n, bins, patches = plt.hist(x=df_0, bins='auto', color='#0504aa',

                            alpha=0.7, rwidth=0.85); 

        plt.grid(axis='y', alpha=0.75)

        plt.xlabel('Value')

        plt.title('Histogram of the variable ' + l + ' on condition of the class 0 ')

        plt.show()

        

        

        n, bins, patches = plt.hist(x=df_1, bins='auto', color='#0504aa',

                            alpha=0.7, rwidth=0.85); 

        plt.grid(axis='y', alpha=0.75)

        plt.xlabel('Value')

        plt.title('Histogram of the variable ' + l + ' on condition of the class 1 ')

        plt.show()

        

        

list_var = ["3", "5", "6"]     



plot_hist(list_var)
df_train = df_train[normal_vars]

X = df_train.values

colnames2 = df_train.columns
predictors = colnames2



def calculate_cor_mat(predictors, X):

    X = X[predictors]

    X = X.values

    correlation_matrix = np.corrcoef(X.T)

    print(correlation_matrix)

        

calculate_cor_mat(predictors, df_train)
def plot_pair_plot(predictors, X):

    X = X.loc[:, predictors]

    g = sns.pairplot(X) 



plot_pair_plot(predictors[0:10], df_train) # we select only a 

                                           # subset of the predictors





random_forest_predictos = ["33", "279", "272", 

                           "83", "237", "241", 

                           "91", "199", "216", 

                           "19", "65", "141", "70", "243", "137", "26", "90"]



predictors = list(set(predictors) & set(random_forest_predictos))

print(predictors)




def five_num(X):

    

    quartiles = np.percentile(X, [25, 50, 75])

    data_min, data_max = X.min(), X.max()

    print("Minimum: {}".format(data_min))

    print("Q1: {}".format(quartiles[0]))

    print("Median: {}".format(quartiles[1]))

    print("Q3: {}".format(quartiles[2]))

    print("Maximum: {}".format(data_max))    





def fit_discriminant(predictors, X):

    

    X = X[predictors]

    X = X.values 

    

    skf = StratifiedKFold(n_splits=10)

    skf.get_n_splits(X, y)

    

    

    train_auc = []

    valid_auc = []

    

    for train_index, test_index in skf.split(X, y):

        

        model = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')

        model.fit(X[train_index], y[train_index])    

        

        y_train = y[train_index]

        y_test = y[test_index]

    

        y_train_predict = model.predict_proba(X[train_index])

        y_train_predict = y_train_predict[:,1]

        y_test_predict = model.predict_proba(X[test_index], )

        y_test_predict = y_test_predict[:,1]           

        

        train_auc.append(roc_auc_score(y_train, y_train_predict))

        valid_auc.append(roc_auc_score(y_test, y_test_predict))

        

    n_bins = 5



    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True);

    ax1.hist(train_auc, bins=n_bins);

    ax1.set_title("Histogram of AUC training")

    ax2.hist(valid_auc, bins=n_bins);

    ax2.set_title("Histogram of AUC validation")  

    

    print("Five numbers Training AUC\n")

    five_num(np.array(train_auc))

    print("\nFive numbers Valid AUC\n")

    five_num(np.array(valid_auc))



    
fit_discriminant(predictors, df_train)
# We fit the model with the whole training dataset

model = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')

model.fit(df_train[predictors], y)
df_test = pd.read_csv("../input/test.csv")

df_test.pop("id");

X = df_test 

X = scaler.transform(X)

df_test = pd.DataFrame(data = X, columns=colnames1)   # df_train is standardized 

X = df_test[predictors]

del df_test

y_pred = model.predict_proba(X)

y_pred = y_pred[:,1]

# submit prediction

smpsb_df = pd.read_csv("../input/sample_submission.csv")

smpsb_df["target"] = y_pred

smpsb_df.to_csv("discrimant_analysisii.csv", index=None)
