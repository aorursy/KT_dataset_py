

# Data manipulation

import pandas as pd

import numpy as np



# Data visualization

import matplotlib.pyplot as plt

import seaborn as sb

from pandas.plotting import scatter_matrix





# Machine Learning Algorithms

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor



# Model Selection and Evaluation

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV





# Performance

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

# For Missing Values

from sklearn.impute import SimpleImputer







fifa_raw_dataset = pd.read_csv('../input/fifa_data/data.csv')



fifa_raw_dataset.info()

fifa_raw_dataset.head()

fifa_raw_dataset.info()



fifa_raw_dataset.shape

features = ['Age','International Reputation', 'Overall', 'Potential', 'Reactions', 'Composure', 'Value', 'Wage', 'Release Clause']

fifa_dataset = fifa_raw_dataset[[*features]]

fifa_dataset.head()





fifa_dataset.info()




#parse string for millions and thousands to numeric values

def parseValue(x):

    x = str(x).replace('€', '')

    if('M' in str(x)):

        x = str(x).replace('M', '')

        x = float(x) * 1000000

    elif('K' in str(x)):

        x = str(x).replace('K', '')

        x = float(x) * 1000

    return float(x)





fifa_dataset['Value'] = fifa_dataset['Value'].apply(parseValue)

fifa_dataset['Wage'] = fifa_dataset['Wage'].apply(parseValue)

fifa_dataset['Release Clause'] = fifa_dataset['Release Clause'].apply(parseValue)



fifa_dataset.head()

print(fifa_dataset.describe())



# Value ditribution



#Fig 1 , all hists

fifa_dataset.hist(bins = 50 , figsize = (20,20),color='Purple',histtype= "bar" ,label= 'Fifa Features')

plt.show()



#Fig 2 value plot

plt.figure(1, figsize=(18, 7))

sb.set(style="darkgrid")

sb.countplot( x= 'Value', data=fifa_dataset)

plt.title('Value distribution of all players')

plt.show()



#ahow corr matrix

corr_matrix = fifa_dataset.corr()

print(corr_matrix.shape)

print(corr_matrix["Value"].sort_values(ascending=False))



#Fig 3 corr

plt.figure(figsize=(20 , 10))

hm = sb.heatmap(fifa_dataset[["Value", "International Reputation", "Overall",

              "Potential", "Reactions", "Composure","Wage","Release Clause","Age"]].corr() , cmap= 'coolwarm' , annot = True , linewidth = .5)

plt.show(hm)



#Fig 4 age plot



sb.set(style ="dark", palette="colorblind", color_codes=True)

x = fifa_dataset['Age']

plt.figure(figsize=(12,8))

ax = sb.distplot(x, bins = 58, kde = False, color='g')

ax.set_xlabel(xlabel="Player\'s age", fontsize=16)

ax.set_ylabel(ylabel='Number of players', fontsize=16)

ax.set_title(label='Histogram of players age', fontsize=20)

plt.show()
#Fig 5  plotting a pie chart to represent share of international repuatation



labels = ['1', '2', '3', '4', '5']

sizes = fifa_dataset['International Reputation'].value_counts()

colors = plt.cm.copper(np.linspace(0, 1, 5))

explode = [0.1, 0.1, 0.2, 0.5, 0.9]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True)

plt.title('International Repuatation for the Football Players', fontsize = 20)

plt.legend()

plt.show()



# to make this notebook's output identical at every run

np.random.seed(42)

#Split the dataset into TRAIN and TEST set. Giving 20% of data to test set.

train_set, test_set = train_test_split(fifa_dataset, test_size=0.2, random_state=42)

print('Train',' ','Test')

print(len(train_set),'+',len(test_set),'=',len(train_set)+len(test_set))



l = list(train_set['Value'] == 0)

print('Zeros in output label: ',len([v for v in l if v==True] ))

print('\nNaN values in following features:')

print(train_set.isnull().any())



# Doing imputation

train_set = train_set.replace(0, pd.np.nan)

imputer = SimpleImputer(strategy="median")

print(imputer.fit(train_set))

print(imputer.statistics_)





tf = imputer.transform(train_set)

fifa_dataset_tf = pd.DataFrame(tf, columns=fifa_dataset.columns)

print(fifa_dataset_tf.head())
# No NULL value present after imputation.

print(fifa_dataset_tf.isnull().any())



# drop labels for training set



fifa_dataset_features = fifa_dataset_tf.drop("Value", axis=1)

fifa_dataset_labels = fifa_dataset_tf["Value"].copy()



#LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(fifa_dataset_features, fifa_dataset_labels)

print(lin_reg)

# RMSE

fifa_dataset_predictions = lin_reg.predict(fifa_dataset_features)

lin_mse = mean_squared_error(fifa_dataset_labels, fifa_dataset_predictions)

lin_rmse = np.sqrt(lin_mse)

print(lin_rmse)

#Accuracy

score_lin = r2_score(fifa_dataset_labels, fifa_dataset_predictions)

print('Accuracy:',format(score_lin*100,'.2f'),'%')



#DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)

tree_reg.fit(fifa_dataset_features, fifa_dataset_labels)

print(tree_reg)

# rmse

fifa_dataset_predictions = tree_reg.predict(fifa_dataset_features)

tree_mse = mean_squared_error(fifa_dataset_labels, fifa_dataset_predictions)

tree_rmse = np.sqrt(tree_mse)

print(tree_rmse)

#Accuracy

score_tree = r2_score(fifa_dataset_labels, fifa_dataset_predictions)

print('Accuracy:',format(score_tree*100,'.2f'),'%')





#RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

forest_reg.fit(fifa_dataset_features, fifa_dataset_labels)

print(forest_reg)

#rmse

fifa_dataset_predictions = forest_reg.predict(fifa_dataset_features)

forest_mse = mean_squared_error(fifa_dataset_labels, fifa_dataset_predictions)

forest_rmse = np.sqrt(forest_mse)

print(forest_rmse)

#Accuracy

score_forst = r2_score(fifa_dataset_labels, fifa_dataset_predictions)

print('Accuracy:',format(score_forst*100,'.2f'),'%')



#GradientBoostingRegressor

gradient_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)

gradient_reg.fit(fifa_dataset_features, fifa_dataset_labels)

print(gradient_reg)

#rmse

fifa_dataset_predictions = gradient_reg.predict(fifa_dataset_features)

gradient_mse = mean_squared_error(fifa_dataset_labels, fifa_dataset_predictions)

gradient_rmse = np.sqrt(gradient_mse)

print(gradient_rmse)

#Accuracy

score_gradient= r2_score(fifa_dataset_labels, fifa_dataset_predictions)

print('Accuracy:',format(score_gradient*100,'.2f'),'%')





#MLPRegressor

mlp_reg = MLPRegressor(random_state=42)

mlp_reg.fit(fifa_dataset_features, fifa_dataset_labels)

print(mlp_reg)

#MLPRegressor rmse

fifa_dataset_predictions = mlp_reg.predict(fifa_dataset_features)

mlp_mse = mean_squared_error(fifa_dataset_labels, fifa_dataset_predictions)

mlp_rmse = np.sqrt(mlp_mse)

print(mlp_rmse)

#MLP Accuracy

score_MLP = r2_score(fifa_dataset_labels, fifa_dataset_predictions)

print('Accuracy:',format(score_MLP*100,'.2f'),'%')



#SVR Regressor

svr_reg = SVR()

svr_reg.fit(fifa_dataset_features, fifa_dataset_labels)

print(svr_reg)

#rmse

fifa_dataset_predictions = svr_reg.predict(fifa_dataset_features)

svr_mse = mean_squared_error(fifa_dataset_labels, fifa_dataset_predictions)

svr_rmse = np.sqrt(svr_mse)

print(svr_rmse)

#Accuracy

score_SVR = r2_score(fifa_dataset_labels, fifa_dataset_predictions)

print('Accuracy:',format(score_SVR*100,'.2f'),'%')





#Plotting scores

train_accu=[score_lin,score_tree,score_forst,score_gradient,score_MLP,score_SVR]

col={'Train Accuracy':train_accu}

models=['Linear','Decision Tree','Random Forest','GradientBoosting','MLP','SVR']

df_acc= pd.DataFrame(data=col,index=models)

print(df_acc)





df_acc.plot(kind='bar',color = "orange", ec="skyblue")



plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.title('Models Comparison')

plt.show()



#plotting rmse

train_rmse=[lin_rmse,tree_rmse,forest_rmse,gradient_rmse,mlp_rmse,svr_rmse]

col={'Train RMSE':train_rmse}

models=['Linear','Decision Tree','Random Forest', 'GradientBoosting','MLP','SVR']

df_rmse= pd.DataFrame(data=col,index=models)

print(df_rmse)







df_rmse.plot(kind='bar',color = "blue", ec="skyblue")



plt.xlabel("Models")

plt.ylabel("RMSE")

plt.title('Models Comparison')

plt.show()


#tree_reg_ cross validation

scores = cross_val_score(tree_reg, fifa_dataset_features, fifa_dataset_labels,

                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())



display_scores(tree_rmse_scores)



#lin_reg cross validation

lin_scores = cross_val_score(lin_reg, fifa_dataset_features, fifa_dataset_labels,

                             scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)





#forest_reg cross validation

forest_scores = cross_val_score(forest_reg, fifa_dataset_features, fifa_dataset_labels,

                             scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)



#gradiant_reg cross validation

gradient_scores = cross_val_score(gradient_reg, fifa_dataset_features, fifa_dataset_labels,

                             scoring="neg_mean_squared_error", cv=10)

gradient_rmse_scores = np.sqrt(-gradient_scores)

display_scores(gradient_rmse_scores)





#mlp_reg cross validation

mlp_scores = cross_val_score(mlp_reg, fifa_dataset_features, fifa_dataset_labels,

                             scoring="neg_mean_squared_error", cv=10)

mlp_rmse_scores = np.sqrt(-mlp_scores)

display_scores(mlp_rmse_scores)





#svr_reg cross validation

svr_scores = cross_val_score(svr_reg, fifa_dataset_features, fifa_dataset_labels,

                             scoring="neg_mean_squared_error", cv=10)

svr_rmse_scores = np.sqrt(-svr_scores)

display_scores(svr_rmse_scores)



#plotting rmse

val_rmse=[lin_rmse_scores.mean(), tree_rmse_scores.mean(),forest_rmse_scores.mean(),gradient_rmse_scores.mean(),

          mlp_rmse_scores.mean(),svr_rmse_scores.mean()]

col={'Validation RMSE':val_rmse}

models=['Linear','Decision Tree','Random Forest', 'GradientBoosting','MLP','SVR']

df_rmse= pd.DataFrame(data=col,index=models)

print(df_rmse)



df_rmse.plot(kind='bar',color = "blue", ec="skyblue")



plt.xlabel("Models")

plt.ylabel("RMSE")

plt.title('Models Comparison')

plt.show()





#Plotting scores

val_accu=[lin_scores.mean(), scores.mean(), forest_scores.mean(), gradient_scores.mean(), mlp_scores.mean(), svr_scores.mean()]

col={'Validation Accuracy':val_accu}

models=['Linear','Decision Tree','Random Forest','GradientBoosting','MLP','SVR']

df_acc= pd.DataFrame(data=col,index=models)

print(df_acc)



df_acc.plot(kind='bar',color = "purple", ec="skyblue")



plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.title('Models Comparison')

plt.show()
