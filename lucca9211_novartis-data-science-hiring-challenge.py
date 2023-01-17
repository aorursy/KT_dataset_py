import pandas as pd

import numpy as np

## Importing seaborn, matplotlab and scipy modules. 

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from scipy import stats

import matplotlib.style as style

style.use('fivethirtyeight')





from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

import scipy

from sklearn.linear_model import LogisticRegression

import optuna
# Read the Dataset

train_df = pd.read_csv("../input/Train.csv", index_col="INCIDENT_ID")

test_df = pd.read_csv("../input/Test.csv", index_col="INCIDENT_ID")

submit_df = pd.read_csv("../input/sample_submission.csv", index_col="INCIDENT_ID")
# First Five training data

train_df.head()
# First Five test data

test_df.head()
train_df = train_df.drop('DATE', axis =1)

test_df = test_df.drop('DATE', axis =1)
# describe training dataset

train_df.describe()

# describe test dataset

test_df.describe()
print (f"Train has {train_df.shape[0]} rows and {train_df.shape[1]} columns")

print (f"Test has {test_df.shape[0]} rows and {test_df.shape[1]} columns")
def plot_fn(df, feature):



    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(12,8))

    ## creating a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    #gs = fig3.add_gridspec(3, 3)



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

    

plot_fn(train_df, 'X_7')

plot_fn(train_df, 'X_8')

plot_fn(train_df, 'X_9')

plot_fn(train_df, 'X_11')

plot_fn(train_df, 'X_13')

plot_fn(train_df, 'X_14')
#skewness and kurtosis

print("Skewness: " + str(train_df['MULTIPLE_OFFENSE'].skew()))

print("Kurtosis: " + str(train_df['MULTIPLE_OFFENSE'].kurt()))

# Function for Calculating missing data ratio in feature columns

def missing_ratio(data_mis):

    data_mis = (data_mis.isnull().sum() / len(data_mis)) * 100

    data_mis = data_mis.drop(data_mis[data_mis == 0].index).sort_values(ascending=False)

    data_mis = pd.DataFrame({'Percentage' :data_mis})

    data_mis['Id'] = data_mis.index

    data_mis.reset_index(drop=True,level=0, inplace=True)

    return data_mis#.head()
# Plot the missing feature columns by ratio

def missing_graph(mis):

    with sns.axes_style('whitegrid'):

        g = sns.catplot(x='Id', y='Percentage', data=mis,

                        aspect=1.5, height=8,kind="bar")



        g.set_xlabels('Features')

        g.fig.suptitle("Percentage of Missing Data")

    

        g.set_xticklabels(rotation=45, horizontalalignment='right')

# calculate percentage of missing data

train_mis = missing_ratio(train_df)

test_mis = missing_ratio(test_df)

train_mis
# missing ratio graph in training data

missing_graph(train_mis)





# missing ratio graph in test data

missing_graph(test_mis)



train_df['X_12'] = train_df['X_12'].fillna(-1)

test_df['X_12'] = test_df['X_12'].fillna(-1)
## Plot fig sizing. 

style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))

## Plotting heatmap. 



# Generate a mask for the upper triangle (taken from seaborn example gallery)

mask = np.zeros_like(train_df.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(train_df.corr(), 

            cmap=sns.diverging_palette(20, 220, n=200), 

            mask = mask, 

            annot=True, 

            center = 0, 

           );

## Give title. 

plt.title("Heatmap of all the Features", fontsize = 30);



Ytrain=train_df['MULTIPLE_OFFENSE']

train_df=train_df[list(test_df)]

all_data=pd.concat((train_df, test_df))

print(train_df.shape, test_df.shape, all_data.shape)



encoded=pd.get_dummies(all_data, columns=all_data.columns, sparse=True)

encoded=encoded.sparse.to_coo()

encoded=encoded.tocsr()
Xtrain=encoded[:len(train_df)]

Xtest=encoded[len(train_df):]





kf=StratifiedKFold(n_splits=10)





def objective(trial):

    C=trial.suggest_loguniform('C', 10e-10, 10)

    model=LogisticRegression(C=C, class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)

    score=-cross_val_score(model, Xtrain, Ytrain, cv=kf, scoring='roc_auc').mean()

    return score

study=optuna.create_study()





study.optimize(objective, n_trials=20)



print(study.best_params)



#print(-study.best_value)

params=study.best_params
params['C']
model=LogisticRegression(C=params['C'], class_weight='balanced',max_iter=10000, solver='lbfgs', n_jobs=-1)

model.fit(Xtrain, Ytrain)

predictions=model.predict_proba(Xtest)[:,1]

# submit_df['MULTIPLE_OFFENSE']=predictions

# submit_df.to_csv('submission.csv')

# submit_df.head()

predictions =np.round(predictions)
ID = test_df.index
submission = pd.DataFrame({'INCIDENT_ID':ID, 'MULTIPLE_OFFENSE': predictions})
submission.head()

submission.to_csv('submission.csv', index=False)
