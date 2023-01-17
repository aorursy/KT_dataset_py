# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import scipy as sp
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import operator
warnings.filterwarnings('ignore')
data  = pd.read_csv('../input/fifa19/data.csv')
print(data.columns)
data = data.drop(columns = ['ID', 'Unnamed: 0','Work Rate'], axis = 1)
metrics_columns = ['Overall', 'Potential', 'Value', 'Wage', 'Special', 'International Reputation',
                  'Release Clause']
data['Body Type'].unique()
data['Body Type'] = data['Body Type'].replace('Messi', 'Normal')
data['Body Type'] = data['Body Type'].replace('Courtois', 'Lean')
data['Body Type'] = data['Body Type'].replace('Shaqiri', 'Stocky')
data['Body Type'] = data['Body Type'].replace('Akinfenwa', 'Stocky')
data['Body Type'] = data['Body Type'].replace('C. Ronaldo', 'Lean')
data['Body Type'] = data['Body Type'].replace('PLAYER_BODY_TYPE_25', 'Normal')
data['Body Type'] = data['Body Type'].replace('Neymar', 'Lean')
physical_columns = ['Body Type', 'Height', 'Weight', 'Preferred Foot', 'Acceleration',
                    'SprintSpeed', 'Jumping', 'Stamina', 'Strength']
#Converting value, wage and release clause 
def conversion(in_euros):
    try:
        final = float(in_euros[1:-1])
        if in_euros[-1] == "M":
            final = final*1000000
        elif in_euros[-1] == "K":
            final = final*1000
    except:
        final = np.nan
    return final

data.Value = data.Value.apply(conversion)
data.Wage = data.Wage.apply(conversion)
data["Release Clause"] = data["Release Clause"].apply(conversion)
plt.figure(figsize = (12,12))
sns.pairplot(data, x_vars = physical_columns, y_vars = metrics_columns)
plt.show()
plt.figure(figsize = (10,10))
sns.heatmap(data[physical_columns+metrics_columns].corr(), linewidth = 3, linecolor = 'grey')
plt.show()
data_age = data.dropna(subset = ['Age'], axis = 0)
fig, axis = plt.subplots(1, 2, figsize = (12,8))
axis[0].hist(data_age.Age)
axis[1].violinplot(data_age.Age)

plt.show()
data_age.Age.describe()
age_group = []
i_20 = 0
i_20_30 = 0
i_30 = 0
for i in range(0, data.Age.size):
    if data.Age[i] < 20:
        age_group.append('Less than 20 years')
        i_20 = i_20+ 1
    elif data.Age[i] > 30:
        age_group.append('more than 30 years')
        i_30 = i_30 +1
    else:
        age_group.append('In between 20 and 30 years')
        i_20_30 = i_20_30+ 1
print('number of players below age 20: ',i_20,'\n number of players inbetween 20 and 30: ',i_20_30,
     '\n number of players above 30 years: ',i_30)
print('Percentage of players who are in between age of 20 and 30:', i_20_30*100/(i_20+i_30+i_20_30))
plt.figure(figsize = (8,8))
sns.scatterplot(data.Age, data.Overall, hue = age_group)
plt.xlabel('Age of the player', color = 'yellow')
plt.ylabel('Overall outcome from the player', color = 'yellow')
plt.yticks(color = 'yellow')
plt.xticks(color = 'yellow')
plt.title('Age vs Overall', color = 'yellow', fontsize = 20)
plt.show()
countries_in_order = np.array(data.Nationality.value_counts().index)
countries_num_players = data.Nationality.value_counts()
plt.figure(figsize = (8,8))
plt.bar(countries_in_order[0:20], countries_num_players[0:20],)
plt.xticks(rotation = 'vertical', color = 'yellow')
plt.yticks(color = 'yellow')
plt.xlabel("Countries", color = 'green', fontsize = 20)
plt.ylabel("Number of players", color = 'green', fontsize = 19)
plt.title("Number of players produced by the countries", color = 'green', fontsize = 20)
plt.show()
Overall_country = []
Overall_country_average = []
for k in range(0, data.Nationality.value_counts().size):
    Overall_sum = 0
    for i in range(0, data.Nationality.size):
        if data.Nationality[i] == countries_in_order[k]:
            Overall_sum = Overall_sum + data.Overall[i]
    Overall_country.append(Overall_sum)
    Overall_country_average.append(Overall_sum/countries_num_players[k])
plt.figure(figsize = (12,8))
plt.grid(color = 'yellow')
plt.plot(countries_in_order[0:20], Overall_country_average[0:20], marker = 'o', linewidth = 5,
         markersize = 12, markerfacecolor = 'white', rasterized = True)
plt.xticks(color = 'yellow', rotation = 90, fontsize = 12)
plt.yticks(color = 'yellow', fontsize = 12)
plt.xlabel('Countries (top 20)', color = 'green', fontsize = 15)
plt.ylabel('Average Overall rating of the players', color = 'green', fontsize = 15)
plt.title('Average ratings of players in different countries', color = 'green', fontsize = 20)
plt.show()

Overall_country_experienced = []
for k in range(0, 30):
    Overall_sum = 0
    for i in range(0, data.Nationality.size):
        if ((data.Nationality[i]  == countries_in_order[k]) and (data.Age[i] >= 30)) :
            Overall_sum = Overall_sum + 1
    Overall_country_experienced.append(Overall_sum)
plt.figure(figsize = (10,10))
plt.pie( Overall_country_experienced[0:25],labels =countries_in_order[0:25], rotatelabels = True)
plt.title('Number of experienced players in each country', color = 'green', fontsize = 20)
plt.show()
plt.figure(figsize = (8,8))
sns.distplot(data['Value'].dropna(), kde = False)
plt.yscale('log')
plt.xlabel('Value of the player in euros',color = 'green', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Disturbution of Value of the players on log scale', fontsize = 20, color = 'green')
plt.show()
plt.figure(figsize = (8,8))
plt.grid(color = 'yellow')
sns.distplot(data['Wage'].dropna(), kde = True, rug = True, hist = False)
plt.yscale('log')
plt.xlabel('Wage of the player in euros',color = 'green', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Disturbution of Wage of the players on log scale', fontsize = 20, color = 'green')
plt.show()
plt.figure(figsize = (8,8))
plt.grid(color = 'yellow')
sns.distplot(data['Release Clause'].dropna(), kde = True, hist = True, color = 'black')
plt.yscale('log')
plt.xlabel('Release Clause of the player in euros',color = 'green', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Disturbution of Release Clause of the players on log scale', fontsize = 20, color = 'green')
plt.show()
sns.jointplot(data.Age, data.Value, height = 8, color = 'black')
plt.yscale('log')
plt.show()

sns.jointplot(x = 'Overall', y = 'Release Clause', data = data, kind = 'scatter', height = 10)
plt.yscale('log')
plt.show()
Position_simplified = data.Position.replace(['LS', 'RS', 'LF', 'RF', 'LAM', 'RAM','LW', 'RW', 'LCM',
                                            'RCM', 'LM', 'RM', 'LDM', 'RDM','LWB', 'RWB', 'LCB',
                                            'RCB', 'LB', 'RB'], ['ST', 'ST', 'CF', 'CF', 'CAM',
                                                                'CAM', 'LRW', 'LRW', 'CM', 'CM',
                                                                'LRM', 'LRM', 'CDM', 'CDM', 'LRWB',
                                                                'LRWB', 'CB', 'CB', 'LRB', 'LRB'])
data = data.assign(Position_Simplified = Position_simplified)
plt.figure(figsize = (9,9))
for k in data.Position_Simplified.unique():
    sns.distplot(data.Overall[data.Position_Simplified == k], hist = False, label = k)
plt.xlabel('Overall', color = 'yellow', fontsize = 15)
plt.yticks(color = 'yellow')
plt.xticks(color = 'yellow')
plt.title('Disturbution of different soccer positions', color = 'green', fontsize = 20)
plt.show()
plt.figure(figsize = (15,8))
plt.grid(color = 'black')
sns.boxenplot(data.Position_Simplified, data.Overall)
sns.stripplot(data.Position_Simplified, data.Overall, color = 'grey')
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.xlabel('Positions', color = 'yellow', fontsize = 15)
plt.ylabel('Overall Rating', color = 'yellow', fontsize =  15)
plt.title('Position v/s Overall rating (detailed)', color = 'green', fontsize = 20)
plt.figure(figsize = (12, 5))
plt.grid(color = 'black')
sns.violinplot(data.Position_Simplified, data.Wage)
plt.yscale('log')
plt.xlabel('Positions of Players', color = 'yellow', fontsize = 15)
plt.ylabel('Wage', color = 'yellow', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Position vs Wage of the players', color = 'green', fontsize = 20)
plt.show()
plt.figure(figsize = (8,8))
plt.grid(color = 'black')
plt.hist(data.Overall[data.Position == 'GK'], color = 'red')
plt.xlabel('Overall rating', color = 'yellow', fontsize = 15)
plt.ylabel('Number of players', color = 'yellow', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Disturbution of Goalkeepers based on their overall rating', color = 'green', fontsize = 20)
plt.show()
data.columns
All_skills = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
              'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
              'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
              'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
              'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
              'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
              'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
Evaluation_metrics = ['Value', 'Wage', 'Special', 'Release Clause', 'Overall', 'Potential']
correlate_GK = data[data.Position == 'GK'][Evaluation_metrics + All_skills]
correlate_GK = correlate_GK.corr()
correlate_GK = correlate_GK.drop(Evaluation_metrics, axis = 1)
correlate_GK = correlate_GK.drop(All_skills, axis = 0)
plt.figure(figsize = (20, 8))
sns.heatmap(correlate_GK, linewidth = 0.5,annot = True, linecolor = 'white')
plt.xticks(color = 'yellow', fontsize = 15)
plt.yticks(color = 'yellow', fontsize = 15)
plt.xlabel('Various skills of the players', color = 'yellow', fontsize = 20)
plt.ylabel('Evaluation Metrics of the players', color = 'yellow', fontsize = 20)
plt.title('Correlation of Skills and Evaluation metrics of the players', color = 'green', fontsize = 25)
plt.show()
goalie_skills = ['Reactions', 'GKDiving','GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
plt.figure(figsize = (8,8))
sns.pairplot(data[data.Position == 'GK'], x_vars = goalie_skills, y_vars = Evaluation_metrics)
plt.show()
from sklearn.model_selection import train_test_split
GK_regression_data = data[data.Position == 'GK']
GK_regression_data = GK_regression_data[goalie_skills + ['Overall']]
GK_regression_data = GK_regression_data.dropna(axis = 0)
GK_X_train, GK_X_test, GK_Y_train, GK_Y_test = train_test_split(GK_regression_data[goalie_skills],
                                                                GK_regression_data['Overall'],
                                                                test_size = 0.25, random_state = 42)
print( GK_X_train.shape, GK_Y_train.shape, GK_X_test.shape, GK_Y_test.shape)
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
elastic_net_CV = ElasticNetCV(fit_intercept = True, precompute = False)
elastic_net_CV.fit(GK_X_train, GK_Y_train)
print('the alpha value is: ', elastic_net_CV.alpha_ ,
      ' and the value of l1 ratio: ' , elastic_net_CV.l1_ratio_)
elastic_net = ElasticNet(alpha = 0.1372597751010709, l1_ratio = 0.5, fit_intercept = True, normalize = False,
                        precompute = False, max_iter = 1000, copy_X = True, tol = 0.0001, warm_start = False,
                        positive = False, random_state = None, selection = 'cyclic')
elastic_net.fit(GK_X_train, GK_Y_train)
elastic_net.get_params()
from sklearn.metrics import mean_squared_error
GK_Y_predict = elastic_net.predict(GK_X_test)
mean_squared_error(GK_Y_test, GK_Y_predict)
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression(fit_intercept = True, normalize = False, copy_X = True, n_jobs = None)
linear_regression.fit(GK_X_train, GK_Y_train)
GK_Y_predict = linear_regression.predict(GK_X_test)
print(mean_squared_error(GK_Y_test, GK_Y_predict))
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha = 0.1372597751010709, l1_ratio = 0.5, fit_intercept = True, normalize = False,
                        precompute = False, max_iter = 1000, copy_X = True, tol = 0.0001, warm_start = False,
                        positive = False, random_state = None, selection = 'cyclic')
Y_predict = np.zeros(GK_X_test.shape)
mse = np.zeros((np.size(GK_X_train, 1), ))
for i in range(0, np.size(GK_X_train, 1)):
    X_train = np.array(GK_X_train)
    Y_train = GK_Y_train
    X_train = X_train[:, i]
    X_train = X_train[:, np.newaxis]
    elastic_net.fit(X_train, Y_train)
    X_test = np.array(GK_X_test)
    Y_test = GK_Y_test
    X_test = X_test[:, i]
    X_test = X_test[:, np.newaxis]
    Y_predict[:, i] = elastic_net.predict(X_test)
    mse[i] = mean_squared_error(Y_predict[:, i], Y_test)

fig , axes = plt.subplots(np.size(GK_X_train, 1), 1, figsize = (8, 8*np.size(GK_X_train,1)))
for i in range(0 , np.size(GK_X_train, 1)):
    axes[i].set_xlabel(list(GK_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(GK_X_train.columns)[i]+' v/s Overall average of player', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    x_show = np.array(GK_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], GK_Y_test)
plt.show()
GK_regression_data['Overall'].describe()
plt.figure(figsize = (8, 8))
plt.xlabel(list(GK_X_train.columns)[0], color = 'green', fontsize = 15)
plt.ylabel('overall', color = 'green', fontsize = 15)
plt.title(list(GK_X_train.columns)[0]+' v/s Overall average with grouping', color = 'yellow', fontsize = 20)
plt.grid('True')
x_show = np.array(GK_X_test)
plt.plot(x_show[:, 0], Y_predict[:, 0], color = 'black', linewidth = 3)
plt.scatter(x_show[:, 0], GK_Y_test)
plt.hlines(69, 30, 90, color = 'green')
plt.hlines(64, 30, 90, color = 'green')
plt.hlines(59, 30, 90, color = 'green')
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.show()
fig , axes = plt.subplots(np.size(GK_X_train, 1), 1, figsize = (8, 8*np.size(GK_X_train,1)))
for i in range(0 , np.size(GK_X_train, 1)):
    axes[i].set_xlabel(list(GK_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(GK_X_train.columns)[i]+' v/s Overall rating with grouping', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    axes[i].hlines(69, 30, 90, color = 'green')
    axes[i].hlines(64, 30, 90, color = 'green')
    axes[i].hlines(59, 30, 90, color = 'green')
    x_show = np.array(GK_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], GK_Y_test)
plt.show()
CB_data = data[data.Position_Simplified == 'CB']
plt.figure(figsize = (8,8))
plt.grid(color = 'black')
plt.hist(CB_data.Overall, color = 'red')
plt.xlabel('Overall rating', color = 'yellow', fontsize = 15)
plt.ylabel('Number of players', color = 'yellow', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Disturbution of Center Defenders based on their overall rating', color = 'green', fontsize = 20)
plt.show()
correlate_CB = CB_data[Evaluation_metrics + All_skills]
correlate_CB = correlate_CB.corr()
correlate_CB = correlate_CB.drop(Evaluation_metrics, axis = 1)
correlate_CB = correlate_CB.drop(All_skills, axis = 0)
plt.figure(figsize = (20, 8))
sns.heatmap(correlate_CB, linewidth = 0.5,annot = True, linecolor = 'white')
plt.xticks(color = 'yellow', fontsize = 15)
plt.yticks(color = 'yellow', fontsize = 15)
plt.xlabel('Various skills of the players', color = 'yellow', fontsize = 20)
plt.ylabel('Evaluation Metrics of the players', color = 'yellow', fontsize = 20)
plt.title('Correlation of Skills and Evaluation metrics of the centre defenders', color = 'green', fontsize = 25)
plt.show()
CB_skills = ['HeadingAccuracy', 'ShortPassing', 'LongPassing', 'BallControl', 'Reactions', 'Aggression',
             'Interceptions', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle']
plt.figure(figsize = (20,20))
sns.pairplot(CB_data, x_vars = CB_skills, y_vars = Evaluation_metrics)
plt.show()
CB_regression_data = CB_data[CB_skills + ['Overall']]
CB_regression_data = CB_regression_data.dropna(axis = 0)
CB_X_train, CB_X_test, CB_Y_train, CB_Y_test = train_test_split(CB_regression_data[CB_skills],
                                                                CB_regression_data['Overall'],
                                                                test_size = 0.25, random_state = 42)
print(CB_X_train.shape, CB_X_test.shape, CB_Y_train.shape, CB_Y_test.shape)
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
Lasso_CV = LassoCV(eps = 0.001, n_alphas = 100, fit_intercept = True, copy_X = True)
Lasso_CV.fit(CB_X_train, CB_Y_train)
print('the alpha parameter calculated is', Lasso_CV.alpha_, 'The intercept calculated', Lasso_CV.intercept_)
lasso = Lasso(alpha = 0.12548123432868297, fit_intercept = True, precompute = False, warm_start = False)
lasso.fit(CB_X_train, CB_Y_train)
lasso.get_params()
CB_Y_predict = lasso.predict(CB_X_test)
print(mean_squared_error(CB_Y_test, CB_Y_predict))
from sklearn.linear_model import Lasso
elastic_net = Lasso(alpha = 0.12548123432868297, fit_intercept = True, normalize = False,
                        precompute = False, max_iter = 1000, copy_X = True, tol = 0.0001, warm_start = False,
                        positive = False, random_state = None, selection = 'cyclic')
Y_predict = np.zeros(CB_X_test.shape)
mse = np.zeros((np.size(CB_X_train, 1), ))
for i in range(0, np.size(CB_X_train, 1)):
    X_train = np.array(CB_X_train)
    Y_train = CB_Y_train
    X_train = X_train[:, i]
    X_train = X_train[:, np.newaxis]
    elastic_net.fit(X_train, Y_train)
    X_test = np.array(CB_X_test)
    Y_test = CB_Y_test
    X_test = X_test[:, i]
    X_test = X_test[:, np.newaxis]
    Y_predict[:, i] = elastic_net.predict(X_test)
    mse[i] = mean_squared_error(Y_predict[:, i], Y_test)
fig , axes = plt.subplots(np.size(CB_X_train, 1), 1, figsize = (8, 8*np.size(CB_X_train,1)))
for i in range(0 , np.size(CB_X_train, 1)):
    axes[i].set_xlabel(list(CB_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(CB_X_train.columns)[i]+' v/s Overall average of CB defender',
                      color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    x_show = np.array(CB_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], CB_Y_test)
plt.show()
CB_data['Overall'].describe()
fig , axes = plt.subplots(np.size(CB_X_train, 1), 1, figsize = (8, 8*np.size(CB_X_train,1)))
for i in range(0 , np.size(CB_X_train, 1)):
    axes[i].set_xlabel(list(CB_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(CB_X_train.columns)[i]+' v/s Overall rating with grouping', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    axes[i].hlines(71, 30, 90, color = 'green')
    axes[i].hlines(67, 30, 90, color = 'green')
    axes[i].hlines(63, 30, 90, color = 'green')
    x_show = np.array(CB_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], CB_Y_test)
plt.show()
print('There are', (data.Position_Simplified == 'LRWB').sum() ,
      'fielders in the dataset who play as right or left wing fielders')
plt.figure(figsize = (8,8))
sns.distplot(data.Overall[data.Position_Simplified == 'LRWB'])
plt.title('Distribution of Overall rating of all the L/R wing Backs', color = 'yellow', fontsize = 20)
plt.ylabel('Distribution', color = 'green', fontsize = 15)
plt.xlabel('Overall rating', color = 'green', fontsize = 15)
plt.xticks(color = 'red', fontsize = 10)
plt.yticks(color = 'red', fontsize = 10)
plt.show()
correlate_LRWB = data[data.Position_Simplified == 'LRWB'][All_skills + Evaluation_metrics]
correlate_LRWB = correlate_LRWB.corr()
correlate_LRWB = correlate_LRWB.drop(Evaluation_metrics, axis = 1)
correlate_LRWB = correlate_LRWB.drop(All_skills, axis = 0)
plt.figure(figsize = (20,8))
sns.heatmap(correlate_LRWB, linecolor = 'white', linewidth = 0.5, annot = True)
plt.xticks(color = 'green', fontsize = 15)
plt.yticks(color = 'green', fontsize = 15)
plt.xlabel('skills of the players', color = 'yellow', fontsize = 20)
plt.ylabel('Evaluation metrics', color = 'yellow', fontsize = 20)
plt.title('correlation b/w evaluation and skills of wing backs', color = 'yellow', fontsize = 25)
plt.show()

CDM_data = data[data.Position_Simplified == 'CDM']
plt.figure(figsize = (8,8))
plt.grid(color = 'red')
plt.hist(CDM_data.Overall, color = 'blue')
plt.xlabel('Overall rating', color = 'yellow', fontsize = 15)
plt.ylabel('Number of players', color = 'yellow', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Center defensive Mids based on their overall rating', color = 'green', fontsize = 20)
plt.show()
correlate_CDM = CDM_data[Evaluation_metrics + All_skills]
correlate_CDM = correlate_CDM.corr()
correlate_CDM = correlate_CDM.drop(Evaluation_metrics, axis = 1)
correlate_CDM = correlate_CDM.drop(All_skills, axis = 0)
plt.figure(figsize = (20, 8))
sns.heatmap(correlate_CDM, linewidth = 0.5, linecolor = 'white', annot = True)
plt.xlabel('Skills of the defense mids', color = 'yellow', fontsize = 20)
plt.ylabel('Evaluation Metrics of the players', color = 'yellow', fontsize = 20)
plt.xticks(color = 'green', fontsize = 15)
plt.yticks(color = 'green', fontsize = 15)
plt.title('Correlation map of the centre defensive fielders', color = 'yellow', fontsize = 25)
plt.show()
CDM_skills = ['ShortPassing', 'LongPassing', 'BallControl', 'Reactions', 'Interceptions', 'Vision',
              'Composure', 'StandingTackle']
plt.figure(figsize = (8,8))
sns.pairplot(CDM_data, x_vars = CDM_skills, y_vars = Evaluation_metrics)
plt.show()
CDM_X_train, CDM_X_test, CDM_Y_train, CDM_Y_test = train_test_split(CDM_data[CDM_skills],
                                                                    CDM_data['Overall'], test_size = 0.25,
                                                                    random_state = 42)
print(CDM_X_train.shape, CDM_X_test.shape, CDM_Y_train.shape, CDM_Y_test.shape)
from sklearn.linear_model import RidgeCV
ridgeCV = RidgeCV(alphas = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000), fit_intercept = True, 
                  normalize = False, scoring = None)
ridgeCV.fit(CDM_X_train, CDM_Y_train)
print('the alpha value is: ', ridgeCV.alpha_ ,
      ' and the value of cpfficient matrix is: ' , ridgeCV.coef_)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 1000, fit_intercept = True, normalize = False, max_iter = None, solver = 'auto')
ridge.fit(CDM_X_train, CDM_Y_train)
ridge.get_params()
CDM_Y_predict = ridge.predict(CDM_X_test)
print('The mean squared error is ', mean_squared_error(CDM_Y_predict, CDM_Y_test))
lr = LinearRegression(fit_intercept = True)
Y_predict = np.zeros(CDM_X_test.shape)
mse = np.zeros((np.size(CDM_X_train, 1), ))
for i in range(0, np.size(CDM_X_train, 1)):
    X_train = np.array(CDM_X_train)
    Y_train = CDM_Y_train
    X_train = X_train[:, i]
    X_train = X_train[:, np.newaxis]
    lr.fit(X_train, Y_train)
    X_test = np.array(CDM_X_test)
    Y_test = CDM_Y_test
    X_test = X_test[:, i]
    X_test = X_test[:, np.newaxis]
    Y_predict[:, i] = elastic_net.predict(X_test)
    mse[i] = mean_squared_error(Y_predict[:, i], CDM_Y_test)
fig , axes = plt.subplots(np.size(CDM_X_train, 1), 1, figsize = (8, 8*np.size(CDM_X_train,1)))
for i in range(0 , np.size(CDM_X_train, 1)):
    axes[i].set_xlabel(list(CDM_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(CDM_X_train.columns)[i]+' v/s Overall average of player', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    x_show = np.array(CDM_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], CDM_Y_test)
plt.show()
CDM_data['Overall'].describe()
fig , axes = plt.subplots(np.size(CDM_X_train, 1), 1, figsize = (8, 8*np.size(CDM_X_train,1)))
for i in range(0 , np.size(CDM_X_train, 1)):
    axes[i].set_xlabel(list(CDM_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(CDM_X_train.columns)[i]+' v/s Overall rating with grouping', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    axes[i].hlines(71, 30, 90, color = 'green')
    axes[i].hlines(67, 30, 90, color = 'green')
    axes[i].hlines(64, 30, 90, color = 'green')
    x_show = np.array(CDM_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], CDM_Y_test)
plt.show()
LRB_data = data[data.Position_Simplified == 'LRB']
plt.figure(figsize = (8,8))
plt.grid(color = 'yellow')
plt.hist(LRB_data.Overall, color = 'grey')
plt.xlabel('Overall rating', color = 'yellow', fontsize = 15)
plt.ylabel('Number of players', color = 'yellow', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Disturbution of Full backs based on their overall rating', color = 'green', fontsize = 20)
plt.show()
correlate_LRB = LRB_data[Evaluation_metrics + All_skills]
correlate_LRB = correlate_LRB.corr()
correlate_LRB = correlate_LRB.drop(Evaluation_metrics, axis = 1)
correlate_LRB = correlate_LRB.drop(All_skills, axis = 0)
plt.figure(figsize = (20, 8))
sns.heatmap(correlate_LRB, linewidth = 0.5,annot = True, linecolor = 'white')
plt.xticks(color = 'yellow', fontsize = 15)
plt.yticks(color = 'yellow', fontsize = 15)
plt.xlabel('Various skills of the players', color = 'yellow', fontsize = 20)
plt.ylabel('Evaluation Metrics of the players', color = 'yellow', fontsize = 20)
plt.title('Correlation of Skills and Evaluation metrics of the centre Ful backs', color = 'green', fontsize = 25)
plt.show()
LRB_skills = ['Crossing', 'ShortPassing', 'LongPassing', 'BallControl', 'Reactions', 'Dribbling',
             'Interceptions', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle']
plt.figure(figsize = (20,20))
sns.pairplot(LRB_data, x_vars = LRB_skills, y_vars = Evaluation_metrics)
plt.show()
LRB_regression_data = LRB_data[LRB_skills + ['Overall']]
LRB_regression_data = LRB_regression_data.dropna(axis = 0)
LRB_X_train, LRB_X_test, LRB_Y_train, LRB_Y_test = train_test_split(LRB_regression_data[LRB_skills],
                                                                LRB_regression_data['Overall'],
                                                                test_size = 0.25, random_state = 42)
print(LRB_X_train.shape, LRB_X_test.shape, LRB_Y_train.shape, LRB_Y_test.shape)
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(eps = 0.001, alphas = None, fit_intercept = True, selection = 'cyclic')
lasso_cv.fit(LRB_X_train, LRB_Y_train)
print('The value of alpha of this model: ', lasso_cv.alpha_ , '\nThe value of intercept: ', lasso_cv.intercept_)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.05140238284526512, precompute = False, warm_start = False, selection = 'cyclic')
lasso.fit(LRB_X_train, LRB_Y_train)
lasso.get_params()
LRB_Y_predict = lasso.predict(LRB_X_test)
print('The squared error obtained is: ', mean_squared_error(LRB_Y_predict, LRB_Y_test))
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.05140238284526512, fit_intercept = True, normalize = False,
                        precompute = False, max_iter = 1000, copy_X = True, tol = 0.0001, warm_start = False,
                        positive = False, random_state = None, selection = 'cyclic')
Y_predict = np.zeros(LRB_X_test.shape)
mse = np.zeros((np.size(LRB_X_train, 1), ))
for i in range(0, np.size(LRB_X_train, 1)):
    X_train = np.array(LRB_X_train)
    Y_train = LRB_Y_train
    X_train = X_train[:, i]
    X_train = X_train[:, np.newaxis]
    elastic_net.fit(X_train, Y_train)
    X_test = np.array(LRB_X_test)
    Y_test = LRB_Y_test
    X_test = X_test[:, i]
    X_test = X_test[:, np.newaxis]
    Y_predict[:, i] = elastic_net.predict(X_test)
    mse[i] = mean_squared_error(Y_predict[:, i], Y_test)
fig , axes = plt.subplots(np.size(LRB_X_train, 1), 1, figsize = (8, 8*np.size(LRB_X_train,1)))
for i in range(0 , np.size(LRB_X_train, 1)):
    axes[i].set_xlabel(list(LRB_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(LRB_X_train.columns)[i]+' v/s Overall average of player', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    x_show = np.array(LRB_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], LRB_Y_test)
plt.show()
LRB_data.Overall.describe()
fig , axes = plt.subplots(np.size(LRB_X_train, 1), 1, figsize = (8, 8*np.size(LRB_X_train,1)))
for i in range(0 , np.size(LRB_X_train, 1)):
    axes[i].set_xlabel(list(LRB_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(LRB_X_train.columns)[i]+' v/s Overall rating with grouping', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    axes[i].hlines(70, 20, 90, color = 'green')
    axes[i].hlines(66, 20, 90, color = 'green')
    axes[i].hlines(62, 20, 90, color = 'green')
    x_show = np.array(LRB_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], LRB_Y_test)
plt.show()
CM_data = data[data.Position_Simplified == 'CM']
plt.figure(figsize = (8,8))
plt.grid(color = 'red')
plt.hist(CM_data.Overall, color = 'green')
plt.xlabel('Overall rating', color = 'yellow', fontsize = 15)
plt.ylabel('Number of players', color = 'yellow', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Disturbution of Center Midfielders based on their overall rating', color = 'White', fontsize = 20)
plt.show()
correlate_CM = CM_data[Evaluation_metrics + All_skills]
correlate_CM = correlate_CM.corr()
correlate_CM = correlate_CM.drop(Evaluation_metrics, axis = 1)
correlate_CM = correlate_CM.drop(All_skills, axis = 0)
plt.figure(figsize = (20, 8))
sns.heatmap(correlate_CM, linewidth = 0.5,annot = True, linecolor = 'white')
plt.xticks(color = 'yellow', fontsize = 15)
plt.yticks(color = 'yellow', fontsize = 15)
plt.xlabel('Various skills of the players', color = 'yellow', fontsize = 20)
plt.ylabel('Evaluation Metrics of the players', color = 'yellow', fontsize = 20)
plt.title('Correlation of Skills and Evaluation metrics of the centre Midfielders', color = 'green', fontsize = 25)
plt.show()
CM_skills = ['Crossing', 'ShortPassing', 'Dribbling', 'LongPassing', 'BallControl', 'Reactions',
             'ShotPower', 'LongShots', 'Positioning', 'Vision', 'Composure']
plt.figure(figsize = (20,20))
sns.pairplot(CM_data, x_vars = CM_skills, y_vars = Evaluation_metrics)
plt.show()
CM_regression_data = CM_data[CM_skills + ['Overall']]
CM_regression_data = CM_regression_data.dropna(axis = 0)
CM_X_train, CM_X_test, CM_Y_train, CM_Y_test = train_test_split(CM_regression_data[CM_skills],
                                                                CM_regression_data['Overall'],
                                                                test_size = 0.25, random_state = 42)
print(CM_X_train.shape, CM_X_test.shape, CM_Y_train.shape, CM_Y_test.shape)
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
Lasso_CV = LassoCV(eps = 0.001, n_alphas = 100, fit_intercept = True, copy_X = True)
Lasso_CV.fit(CM_X_train, CM_Y_train)
print('the alpha parameter calculated is', Lasso_CV.alpha_, 'The intercept calculated', Lasso_CV.intercept_)
lasso = Lasso(alpha = 0.13895669171744096, fit_intercept = True, precompute = False, warm_start = False)
lasso.fit(CM_X_train, CM_Y_train)
lasso.get_params()
CM_Y_predict = lasso.predict(CM_X_test)
print(mean_squared_error(CM_Y_test, CM_Y_predict))
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.13895669171744096, fit_intercept = True, normalize = False,
                        precompute = False, max_iter = 1000, copy_X = True, tol = 0.0001, warm_start = False,
                        positive = False, random_state = None, selection = 'cyclic')
Y_predict = np.zeros(CM_X_test.shape)
mse = np.zeros((np.size(CM_X_train, 1), ))
for i in range(0, np.size(CM_X_train, 1)):
    X_train = np.array(CM_X_train)
    Y_train = CM_Y_train
    X_train = X_train[:, i]
    X_train = X_train[:, np.newaxis]
    lasso.fit(X_train, Y_train)
    X_test = np.array(CM_X_test)
    Y_test = CM_Y_test
    X_test = X_test[:, i]
    X_test = X_test[:, np.newaxis]
    Y_predict[:, i] = lasso.predict(X_test)
    mse[i] = mean_squared_error(Y_predict[:, i], Y_test)
fig , axes = plt.subplots(np.size(CM_X_train, 1), 1, figsize = (8, 8*np.size(CM_X_train,1)))
for i in range(0 , np.size(CM_X_train, 1)):
    axes[i].set_xlabel(list(CM_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(CM_X_train.columns)[i]+' v/s Overall average of Center Midfielder',
                      color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    x_show = np.array(CM_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], CM_Y_test)
plt.show()
CM_data['Overall'].describe()
fig , axes = plt.subplots(np.size(CM_X_train, 1), 1, figsize = (8, 8*np.size(CM_X_train,1)))
for i in range(0 , np.size(CM_X_train, 1)):
    axes[i].set_xlabel(list(CM_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(CM_X_train.columns)[i]+' v/s Overall rating with grouping', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    axes[i].hlines(71, 30, 90, color = 'green')
    axes[i].hlines(66, 30, 90, color = 'green')
    axes[i].hlines(61, 30, 90, color = 'green')
    x_show = np.array(CM_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], CM_Y_test)
plt.show()
LRM_data = data[data.Position_Simplified == 'LRM']
plt.figure(figsize = (8,8))
plt.grid(color = 'red')
plt.hist(LRM_data.Overall, color = 'blue')
plt.xlabel('Overall rating', color = 'yellow', fontsize = 15)
plt.ylabel('Number of players', color = 'yellow', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Disturbution of Left/Right Midfielders based on their overall rating', color = 'White', fontsize = 20)
plt.show()
correlate_LRM = LRM_data[Evaluation_metrics + All_skills]
correlate_LRM = correlate_LRM.corr()
correlate_LRM = correlate_LRM.drop(Evaluation_metrics, axis = 1)
correlate_LRM = correlate_LRM.drop(All_skills, axis = 0)
plt.figure(figsize = (20, 8))
sns.heatmap(correlate_LRM, linewidth = 0.5,annot = True, linecolor = 'white')
plt.xticks(color = 'yellow', fontsize = 15)
plt.yticks(color = 'yellow', fontsize = 15)
plt.xlabel('Various skills of the players', color = 'yellow', fontsize = 20)
plt.ylabel('Evaluation Metrics of the players', color = 'yellow', fontsize = 20)
plt.title('Correlation of Skills and Evaluation metrics of the left/right Midfielders', color = 'green', fontsize = 25)
plt.show()
LRM_skills = ['Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'BallControl', 'Reactions', 'LongShots', 'Positioning', 'Vision', 'Composure']
plt.figure(figsize = (20,20))
sns.pairplot(LRM_data, x_vars = LRM_skills, y_vars = Evaluation_metrics)
plt.show()
LRM_regression_data = LRM_data[LRM_skills + ['Overall']]
LRM_regression_data = LRM_regression_data.dropna(axis = 0)
LRM_X_train, LRM_X_test, LRM_Y_train, LRM_Y_test = train_test_split(LRM_regression_data[LRM_skills],
                                                                LRM_regression_data['Overall'],
                                                                test_size = 0.25, random_state = 42)
print(LRM_X_train.shape, LRM_X_test.shape, LRM_Y_train.shape, LRM_Y_test.shape)
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
Lasso_CV = LassoCV(eps = 0.001, n_alphas = 100, fit_intercept = True, copy_X = True)
Lasso_CV.fit(LRM_X_train,LRM_Y_train)
print('the alpha parameter calculated is', Lasso_CV.alpha_, 'The intercept calculated', Lasso_CV.intercept_)
lasso = Lasso(alpha = 0.13895669171744096, fit_intercept = True, precompute = False, warm_start = False)
lasso.fit(LRM_X_train, LRM_Y_train)
lasso.get_params()
LRM_Y_predict = lasso.predict(LRM_X_test)
print(mean_squared_error(LRM_Y_test, LRM_Y_predict))
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.13895669171744096, fit_intercept = True, normalize = False,
                        precompute = False, max_iter = 1000, copy_X = True, tol = 0.0001, warm_start = False,
                        positive = False, random_state = None, selection = 'cyclic')
Y_predict = np.zeros(LRM_X_test.shape)
mse = np.zeros((np.size(LRM_X_train, 1), ))
for i in range(0, np.size(LRM_X_train, 1)):
    X_train = np.array(LRM_X_train)
    Y_train = LRM_Y_train
    X_train = X_train[:, i]
    X_train = X_train[:, np.newaxis]
    lasso.fit(X_train, Y_train)
    X_test = np.array(LRM_X_test)
    Y_test = LRM_Y_test
    X_test = X_test[:, i]
    X_test = X_test[:, np.newaxis]
    Y_predict[:, i] = lasso.predict(X_test)
    mse[i] = mean_squared_error(Y_predict[:, i], Y_test)
fig , axes = plt.subplots(np.size(LRM_X_train, 1), 1, figsize = (8, 8*np.size(LRM_X_train,1)))
for i in range(0 , np.size(LRM_X_train, 1)):
    axes[i].set_xlabel(list(LRM_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(LRM_X_train.columns)[i]+' v/s Overall average of Left/Right Midfielder',
                      color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    x_show = np.array(LRM_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], LRM_Y_test)
plt.show()
LRM_data['Overall'].describe()
fig , axes = plt.subplots(np.size(LRM_X_train, 1), 1, figsize = (8, 8*np.size(LRM_X_train,1)))
for i in range(0 , np.size(LRM_X_train, 1)):
    axes[i].set_xlabel(list(LRM_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(LRM_X_train.columns)[i]+' v/s Overall rating with grouping', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    axes[i].hlines(71, 20, 90, color = 'green')
    axes[i].hlines(67, 20, 90, color = 'green')
    axes[i].hlines(62, 20, 90, color = 'green')
    x_show = np.array(LRM_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], LRM_Y_test)
plt.show()
CAM_data = data[data.Position_Simplified == 'CAM']
plt.figure(figsize = (8,8))
plt.grid(color = 'black')
plt.hist(CAM_data.Overall, color = 'red')
plt.xlabel('Overall rating', color = 'yellow', fontsize = 15)
plt.ylabel('Number of players', color = 'yellow', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Disturbution of Center Defenders based on their overall rating', color = 'green', fontsize = 20)
plt.show()
correlate_CAM = CAM_data[Evaluation_metrics + All_skills]
correlate_CAM = correlate_CAM.corr()
correlate_CAM = correlate_CAM.drop(Evaluation_metrics, axis = 1)
correlate_CAM = correlate_CAM.drop(All_skills, axis = 0)
plt.figure(figsize = (20, 8))
sns.heatmap(correlate_CAM, linewidth = 0.5,annot = True, linecolor = 'white')
plt.xticks(color = 'yellow', fontsize = 15)
plt.yticks(color = 'yellow', fontsize = 15)
plt.xlabel('Various skills of the players', color = 'yellow', fontsize = 20)
plt.ylabel('Evaluation Metrics of the players', color = 'yellow', fontsize = 20)
plt.title('Correlation of Skills and Evaluation metrics of the attacking midfielders', color = 'green', fontsize = 25)
plt.show()
CAM_skills = ['Crossing', 'Finishing', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'LongPassing',
              'BallControl', 'Reactions', 'ShotPower', 'LongShots', 'Positioning', 'Vision',
              'Aggression', 'Composure']
plt.figure(figsize = (20,30))
sns.pairplot(CAM_data, x_vars = CAM_skills, y_vars = Evaluation_metrics)
plt.show()
CAM_regression_data = CAM_data[CAM_skills + ['Overall']]
CAM_regression_data = CAM_regression_data.dropna(axis = 0)
CAM_X_train, CAM_X_test, CAM_Y_train, CAM_Y_test = train_test_split(CAM_regression_data[CAM_skills],
                                                                CAM_regression_data['Overall'],
                                                                test_size = 0.25, random_state = 42)
print(CAM_X_train.shape, CAM_X_test.shape, CAM_Y_train.shape, CAM_Y_test.shape)
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
Lasso_CV = LassoCV(eps = 0.001, n_alphas = 100, fit_intercept = True, copy_X = True)
Lasso_CV.fit(CAM_X_train, CAM_Y_train)
print('the alpha parameter calculated is', Lasso_CV.alpha_, 'The intercept calculated', Lasso_CV.intercept_)
lasso = Lasso(alpha = 0.3171430552809601, fit_intercept = True, precompute = False, warm_start = False)
lasso.fit(CAM_X_train, CAM_Y_train)
lasso.get_params()
CAM_Y_predict = lasso.predict(CAM_X_test)
print(mean_squared_error(CAM_Y_test, CAM_Y_predict))
from sklearn.linear_model import Lasso
elastic_net = Lasso(alpha = 0.12548123432868297, fit_intercept = True, normalize = False,
                        precompute = False, max_iter = 1000, copy_X = True, tol = 0.0001, warm_start = False,
                        positive = False, random_state = None, selection = 'cyclic')
Y_predict = np.zeros(CAM_X_test.shape)
mse = np.zeros((np.size(CAM_X_train, 1), ))
for i in range(0, np.size(CAM_X_train, 1)):
    X_train = np.array(CAM_X_train)
    Y_train = CAM_Y_train
    X_train = X_train[:, i]
    X_train = X_train[:, np.newaxis]
    elastic_net.fit(X_train, Y_train)
    X_test = np.array(CAM_X_test)
    Y_test = CAM_Y_test
    X_test = X_test[:, i]
    X_test = X_test[:, np.newaxis]
    Y_predict[:, i] = elastic_net.predict(X_test)
    mse[i] = mean_squared_error(Y_predict[:, i], Y_test)
fig , axes = plt.subplots(np.size(CAM_X_train, 1), 1, figsize = (8, 8*np.size(CAM_X_train,1)))
for i in range(0 , np.size(CAM_X_train, 1)):
    axes[i].set_xlabel(list(CAM_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(CAM_X_train.columns)[i]+' v/s Overall average of CB defender',
                      color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    x_show = np.array(CAM_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], CAM_Y_test)
plt.show()
CAM_data['Overall'].describe()
fig , axes = plt.subplots(np.size(CAM_X_train, 1), 1, figsize = (8, 8*np.size(CAM_X_train,1)))
for i in range(0 , np.size(CAM_X_train, 1)):
    axes[i].set_xlabel(list(CAM_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(CAM_X_train.columns)[i]+' v/s Overall rating with grouping', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    axes[i].hlines(71, 30, 90, color = 'green')
    axes[i].hlines(67, 30, 90, color = 'green')
    axes[i].hlines(63, 30, 90, color = 'green')
    x_show = np.array(CAM_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], CAM_Y_test)
plt.show()
LRW_data = data[data.Position_Simplified == 'LRW']
plt.figure(figsize = (8,8))
plt.grid(color = 'red')
plt.hist(LRW_data.Overall, color = 'yellow')
plt.xlabel('Overall rating', color = 'yellow', fontsize = 15)
plt.ylabel('Number of players', color = 'yellow', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Center defensive Mids based on their overall rating', color = 'green', fontsize = 20)
plt.show()
correlate_LRW = LRW_data[Evaluation_metrics + All_skills]
correlate_LRW = correlate_LRW.corr()
correlate_LRW = correlate_LRW.drop(Evaluation_metrics, axis = 1)
correlate_LRW = correlate_LRW.drop(All_skills, axis = 0)
plt.figure(figsize = (20, 8))
sns.heatmap(correlate_LRW, linewidth = 0.5,annot = True, linecolor = 'white')
plt.xticks(color = 'yellow', fontsize = 15)
plt.yticks(color = 'yellow', fontsize = 15)
plt.xlabel('Various skills of the players', color = 'yellow', fontsize = 20)
plt.ylabel('Evaluation Metrics of the players', color = 'yellow', fontsize = 20)
plt.title('Correlation of Skills and Evaluation metrics of the centre Ful backs', color = 'green', fontsize = 25)
plt.show()
LRW_skills = ['Crossing', 'Finishing', 'Dribbling', 'ShortPassing', 'Curve', 'LongPassing',
              'BallControl', 'Reactions', 'LongShots', 'Positioning', 'Composure', 'Vision']
plt.figure(figsize = (20,20))
sns.pairplot(LRW_data, x_vars = LRW_skills, y_vars = Evaluation_metrics)
plt.show()
LRW_regression_data = LRW_data[LRW_skills + ['Overall']]
LRW_regression_data = LRW_regression_data.dropna(axis = 0)
LRW_X_train, LRW_X_test, LRW_Y_train, LRW_Y_test = train_test_split(LRW_regression_data[LRW_skills],
                                                                LRW_regression_data['Overall'],
                                                                test_size = 0.25, random_state = 42)
print(LRW_X_train.shape, LRW_X_test.shape, LRW_Y_train.shape, LRW_Y_test.shape)
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(eps = 0.001, alphas = None, fit_intercept = True, selection = 'cyclic')
lasso_cv.fit(LRW_X_train, LRW_Y_train)
print('The value of alpha of this model: ', lasso_cv.alpha_ , '\nThe value of intercept: ', lasso_cv.intercept_)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.11275707234321862, precompute = False, warm_start = False, selection = 'cyclic')
lasso.fit(LRW_X_train, LRW_Y_train)
lasso.get_params()
LRW_Y_predict = lasso.predict(LRW_X_test)
print('The squared error obtained is: ', mean_squared_error(LRW_Y_predict, LRW_Y_test))
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.11275707234321862, fit_intercept = True, normalize = False,
                        precompute = False, max_iter = 1000, copy_X = True, tol = 0.0001, warm_start = False,
                        positive = False, random_state = None, selection = 'cyclic')
Y_predict = np.zeros(LRW_X_test.shape)
mse = np.zeros((np.size(LRW_X_train, 1), ))
for i in range(0, np.size(LRW_X_train, 1)):
    X_train = np.array(LRW_X_train)
    Y_train = LRW_Y_train
    X_train = X_train[:, i]
    X_train = X_train[:, np.newaxis]
    elastic_net.fit(X_train, Y_train)
    X_test = np.array(LRW_X_test)
    Y_test = LRW_Y_test
    X_test = X_test[:, i]
    X_test = X_test[:, np.newaxis]
    Y_predict[:, i] = elastic_net.predict(X_test)
    mse[i] = mean_squared_error(Y_predict[:, i], Y_test)
fig , axes = plt.subplots(np.size(LRW_X_train, 1), 1, figsize = (8, 8*np.size(LRW_X_train,1)))
for i in range(0 , np.size(LRW_X_train, 1)):
    axes[i].set_xlabel(list(LRW_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(LRW_X_train.columns)[i]+' v/s Overall average of player', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    x_show = np.array(LRW_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], LRW_Y_test)
plt.show()
LRW_data.Overall.describe()
fig , axes = plt.subplots(np.size(LRW_X_train, 1), 1, figsize = (8, 8*np.size(LRW_X_train,1)))
for i in range(0 , np.size(LRW_X_train, 1)):
    axes[i].set_xlabel(list(LRW_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(LRW_X_train.columns)[i]+' v/s Overall rating with grouping', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    axes[i].hlines(71, 20, 90, color = 'green')
    axes[i].hlines(66, 20, 90, color = 'green')
    axes[i].hlines(63, 20, 90, color = 'green')
    x_show = np.array(LRW_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], LRW_Y_test)
plt.show()
data['Position_Simplified'] = data['Position_Simplified'].replace('ST', 'Forwards')
data['Position_Simplified'] = data['Position_Simplified'].replace('CF', 'Forwards')
Forwards_data = data[data.Position_Simplified == 'Forwards']
plt.figure(figsize = (8,8))
plt.grid(color = 'black')
plt.hist(Forwards_data.Overall, color = 'red')
plt.xlabel('Overall rating', color = 'yellow', fontsize = 15)
plt.ylabel('Number of players', color = 'yellow', fontsize = 15)
plt.xticks(color = 'yellow')
plt.yticks(color = 'yellow')
plt.title('Disturbution of forwards based on their overall rating', color = 'green', fontsize = 20)
plt.show()
correlate_Forwards = Forwards_data[Evaluation_metrics + All_skills]
correlate_Forwards = correlate_Forwards.corr()
correlate_Forwards = correlate_Forwards.drop(Evaluation_metrics, axis = 1)
correlate_Forwards = correlate_Forwards.drop(All_skills, axis = 0)
plt.figure(figsize = (20, 8))
sns.heatmap(correlate_Forwards, linewidth = 0.5,annot = True, linecolor = 'white')
plt.xticks(color = 'yellow', fontsize = 15)
plt.yticks(color = 'yellow', fontsize = 15)
plt.xlabel('Various skills of the players', color = 'yellow', fontsize = 20)
plt.ylabel('Evaluation Metrics of the players', color = 'yellow', fontsize = 20)
plt.title('Correlation of Skills and Evaluation metrics of the Forwards', color = 'green', fontsize = 25)
plt.show()
Forwards_skills = ['Finishing', 'ShortPassing', 'Volleys', 'Dribbling', 'BallControl', 'Reactions',
                   'ShotPower', 'LongShots', 'Positioning', 'Composure']
plt.figure(figsize = (20,20))
sns.pairplot(Forwards_data, x_vars = Forwards_skills, y_vars = Evaluation_metrics)
plt.show()
Forwards_regression_data = Forwards_data[Forwards_skills + ['Overall']]
Forwards_regression_data = Forwards_regression_data.dropna(axis = 0)
Forwards_X_train, Forwards_X_test, Forwards_Y_train, Forwards_Y_test = train_test_split(Forwards_regression_data[Forwards_skills],
                                                                                        Forwards_regression_data['Overall'],
                                                                                        test_size = 0.25, random_state = 42)
print(Forwards_X_train.shape, Forwards_X_test.shape, Forwards_Y_train.shape, Forwards_Y_test.shape)
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
Lasso_CV = LassoCV(eps = 0.001, n_alphas = 100, fit_intercept = True, copy_X = True)
Lasso_CV.fit(Forwards_X_train, Forwards_Y_train)
print('the alpha parameter calculated is', Lasso_CV.alpha_, 'The intercept calculated', Lasso_CV.intercept_)
lasso = Lasso(alpha = 0.06811647800000001, fit_intercept = True, precompute = False, warm_start = False)
lasso.fit(Forwards_X_train, Forwards_Y_train)
lasso.get_params()
Forwards_Y_predict = lasso.predict(Forwards_X_test)
print(mean_squared_error(Forwards_Y_test, Forwards_Y_predict))
from sklearn.linear_model import Lasso
elastic_net = Lasso(alpha = 0.06811647800000001, fit_intercept = True, normalize = False,
                        precompute = False, max_iter = 1000, copy_X = True, tol = 0.0001, warm_start = False,
                        positive = False, random_state = None, selection = 'cyclic')
Y_predict = np.zeros(Forwards_X_test.shape)
mse = np.zeros((np.size(Forwards_X_train, 1), ))
for i in range(0, np.size(Forwards_X_train, 1)):
    X_train = np.array(Forwards_X_train)
    Y_train = Forwards_Y_train
    X_train = X_train[:, i]
    X_train = X_train[:, np.newaxis]
    elastic_net.fit(X_train, Y_train)
    X_test = np.array(Forwards_X_test)
    Y_test = Forwards_Y_test
    X_test = X_test[:, i]
    X_test = X_test[:, np.newaxis]
    Y_predict[:, i] = elastic_net.predict(X_test)
    mse[i] = mean_squared_error(Y_predict[:, i], Y_test)
fig , axes = plt.subplots(np.size(Forwards_X_train, 1), 1, figsize = (8, 8*np.size(Forwards_X_train,1)))
for i in range(0 , np.size(Forwards_X_train, 1)):
    axes[i].set_xlabel(list(Forwards_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(Forwards_X_train.columns)[i]+' v/s Overall average of Forwards',
                      color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    x_show = np.array(Forwards_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], Forwards_Y_test)
plt.show()
Forwards_data['Overall'].describe()
fig , axes = plt.subplots(np.size(Forwards_X_train, 1), 1, figsize = (8, 8*np.size(Forwards_X_train,1)))
for i in range(0 , np.size(Forwards_X_train, 1)):
    axes[i].set_xlabel(list(Forwards_X_train.columns)[i], color = 'green', fontsize = 15)
    axes[i].set_ylabel('overall', color = 'green', fontsize = 15)
    axes[i].set_title(list(Forwards_X_train.columns)[i]+' v/s Overall rating with grouping', color = 'yellow', fontsize = 20)
    axes[i].grid('True')
    axes[i].hlines(71, 30, 90, color = 'green')
    axes[i].hlines(66, 30, 90, color = 'green')
    axes[i].hlines(62, 30, 90, color = 'green')
    x_show = np.array(Forwards_X_test)
    axes[i].plot(x_show[:, i], Y_predict[:, i], color = 'black', linewidth = 3)
    axes[i].scatter(x_show[:, i], Forwards_Y_test)
plt.show()