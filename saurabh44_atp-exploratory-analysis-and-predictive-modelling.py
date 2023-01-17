# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as px
px.init_notebook_mode(connected=True)
px.offline.init_notebook_mode(connected=True)
import plotly.express as px
from urllib.request import urlopen  
import os.path as osp
import os
import logging
import zipfile
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error 
  
logging.getLogger().setLevel('INFO')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def download_file(url_str, path):
    url = urlopen(url_str)
    output = open(path, 'wb')       
    output.write(url.read())
    output.close()  
    
def extract_file(archive_path, target_dir):
    zip_file = zipfile.ZipFile(archive_path, 'r')
    zip_file.extractall(target_dir)
    zip_file.close()
BASE_URL = 'http://tennis-data.co.uk'
DATA_DIR = "tennis_data"
ATP_DIR = './{}/ATP'.format(DATA_DIR)
WTA_DIR = './{}/WTA'.format(DATA_DIR)

ATP_URLS = [BASE_URL + "/%i/%i.zip" % (i,i) for i in range(2000,2019)]
WTA_URLS = [BASE_URL + "/%iw/%i.zip" % (i,i) for i in range(2007,2019)]

os.makedirs(osp.join(ATP_DIR, 'archives'), exist_ok=True)
os.makedirs(osp.join(WTA_DIR, 'archives'), exist_ok=True)

for files, directory in ((ATP_URLS, ATP_DIR), (WTA_URLS, WTA_DIR)):
    for dl_path in files:
        logging.info("downloading & extracting file %s", dl_path)
        archive_path = osp.join(directory, 'archives', osp.basename(dl_path))
        download_file(dl_path, archive_path)
        extract_file(archive_path, directory)
    
ATP_FILES = sorted(glob("%s/*.xls*" % ATP_DIR))
WTA_FILES = sorted(glob("%s/*.xls*" % WTA_DIR))

df_atp = pd.concat([pd.read_excel(f) for f in ATP_FILES], ignore_index=True)
df_wta = pd.concat([pd.read_excel(f) for f in WTA_FILES], ignore_index=True)

logging.info("%i matches ATP in df_atp", df_atp.shape[0])
logging.info("%i matches WTA in df_wta", df_wta.shape[0])
df_atp['Lsets'].replace('`1', 1,inplace = True)
dist_client_dfs=df_atp.groupby(["Winner"])["Winner"].count().reset_index(name="count")
dist_client_dfs.sort_values(by=['count'], ascending=False, inplace=True)
dist_client_dfs=dist_client_dfs.nlargest(10, ['count'])
print(dist_client_dfs.head(3))
fig = px.bar(dist_client_dfs, x="Winner", y="count", orientation='v',text="count")
fig.show()
df_atp['Wsets'].fillna(0,inplace=True)
df_atp['Lsets'].fillna(0,inplace=True)
df_Federer_Winner = df_atp[(df_atp.Winner == "Federer R.")] 
df_Federer_Loser = df_atp[ (df_atp.Loser == "Federer R.")]
print("Total number of sets Roger Federer won: ",df_Federer_Winner['Wsets'].sum() + df_Federer_Loser['Lsets'].sum())
df_Federer_Loser_1617 = df_atp[(df_atp.Loser == "Federer R.") & ((df_atp["Date"].dt.year == 2016) | 
                                                                 (df_atp["Date"].dt.year == 2017))]
df_Federer_Winner_1617 = df_atp[(df_atp.Winner == "Federer R.") & ((df_atp["Date"].dt.year == 2016) | 
                                                                   (df_atp["Date"].dt.year == 2017))]
print("Number of sets Roger Federer won in 2016 and 2017: ",df_Federer_Winner_1617['Wsets'].sum() + 
                                                              df_Federer_Loser_1617['Lsets'].sum())

def previous_w_percentage(player,date, df_atp):
    minimum_played_games = 2
    df_previous  = df_atp[df_atp["Date"] < date]
    previous_wins = df_previous[df_previous["Winner"] == player].shape[0]
    previous_losses = df_previous[df_previous["Loser"] == player].shape[0]
    
    if  minimum_played_games > (previous_wins + previous_losses):
        return 0
    return previous_wins / (previous_wins + previous_losses)
df_atp["winner_previous_win_percentage"] = df_atp.apply(
    lambda row: previous_w_percentage(
        row["Winner"],
        row["Date"],
        df_atp
    ),
    axis=1
)
df_atp.to_csv("atp_previous_win_percentage.csv")
grandslams = df_atp[['Date','Tournament','Series', 'Round', 'Winner']]
grandslams = grandslams[(grandslams.Series == 'Grand Slam') & (grandslams.Round == 'The Final')]
grandslams['Titles'] = grandslams.groupby('Winner').cumcount().astype(int) + 1

winners = grandslams.groupby('Winner')['Tournament'].count()
winners = winners.reset_index()
winners = winners.sort_values(['Tournament'], ascending=False)
# winners
fig = px.bar(winners, x="Winner", y="Tournament", orientation='v',text="Tournament",title='Grand Slams won since 2000')
fig.show()
winners_grandslam = grandslams.groupby(['Winner', 'Tournament']).count()
winners_grandslam = winners_grandslam.reset_index()
# winners_grandslam
fig = px.bar(winners_grandslam, x="Winner", y="Titles", color="Tournament", barmode="group")
fig.show()
type_surface = df_atp[['Surface', 'Winner', 'Loser']]

type_surface_w = type_surface[['Surface', 'Winner']]
type_surface_l = type_surface[['Surface', 'Loser']]
type_surface_w.columns = ['Surface', 'Player']
type_surface_l.columns = ['Surface', 'Player']

type_surface_w['idx'] = range(1, len(type_surface_w) + 1)
type_surface_l['idx'] = range(1, len(type_surface_l) + 1)

type_surface_w = type_surface_w.groupby(['Surface', 'Player']).count()
type_surface_w = type_surface_w.reset_index()
type_surface_w.columns = ['Surface', 'Player', 'Won']

type_surface_l = type_surface_l.groupby(['Surface', 'Player']).count()
type_surface_l = type_surface_l.reset_index()
type_surface_l.columns = ['Surface', 'Player', 'Lost']

type_surface = pd.merge(type_surface_w, type_surface_l, on=['Surface', 'Player'])

type_surface['total_play'] = type_surface['Won'] + type_surface['Lost']

type_surface['perc_win'] = round(type_surface['Won'] / type_surface['total_play'],4)*100

type_surface = type_surface[type_surface.total_play > 50]

# type_surface.sort_values(by='perc_win', ascending=False).head(30)
hard = type_surface[type_surface.Surface == 'Hard'].sort_values(by='perc_win', ascending = False).head(10)
fig1 = px.bar(hard, x='Player', y='perc_win',
             hover_data=['Won', 'Lost'], color='perc_win',
             labels={'perc_win':'Win Percentage'}, height=400,title='Best players on Hard surface')
fig1.show()
grass = type_surface[type_surface.Surface == 'Grass'].sort_values(by='perc_win', ascending = False).head(10)
fig1 = px.bar(grass, x='Player', y='perc_win',
             hover_data=['Won', 'Lost'], color='perc_win',
             labels={'perc_win':'Win Percentage'}, height=400,title='Best players on Grass surface')
fig1.show()
clay = type_surface[type_surface.Surface == 'Clay'].sort_values(by='perc_win', ascending = False).head(10)
fig = px.bar(clay, x='Player', y='perc_win',
             hover_data=['Won', 'Lost'], color='perc_win',
             labels={'perc_win':'Win Percentage'}, height=400,title='Best players on Clay surface')
fig.show()
df_atp["Winner_position"] = df_atp.apply(lambda row: 1 if row["Winner"] > row["Loser"] else 0, axis=1)
df_atp[["Winner", "Loser", "Winner_position"]].head(5)
print(df_atp[df_atp["Winner_position"] == 1].shape[0])
print(df_atp[df_atp["Winner_position"] == 0].shape[0])
df_atp_X = df_atp.loc[:, ['AvgL', 'AvgW', 'B&WL', 'B&WW', 'B365L', 'B365W', 'CBL', 'CBW', 'EXL', 'EXW', 'GBL', 'GBW', \
    'IWL', 'IWW','LBL', 'LBW', 'LRank', 'MaxL', 'MaxW', 'PSL', 'PSW', \
    'SBL', 'SBW', 'SJL', 'SJW', 'UBL', 'UBW', 'WRank','Best of', \
    "Date",'ATP','Series','Court','Surface', 'Winner_position']]
df_atp_X.iloc[:, :-4] = df_atp_X.apply(pd.to_numeric, errors='coerce') 

df_atp_X["WRank"] = df_atp_X["WRank"].fillna(df_atp_X["WRank"].max())
df_atp_X["LRank"] = df_atp_X["LRank"].fillna(df_atp_X["LRank"].max())


cols_1=["AvgL", "AvgW", "B&WL", "B&WW", "B365L", "B365W", "CBL", "CBW", "EXL", "EXW", "GBL", "GBW", "IWL", "IWW",
    "LBL", "LBW", "MaxL", "MaxW", "PSL", "PSW", "SBL", "SBW", "SJL", "SJW", "UBL", "UBW"]
df_atp_X[cols_1]=df_atp_X[cols_1].fillna(1.0)

# df_atp_X.isnull().sum()
df_atp_X["P1Rank"] = df_atp_X.apply(lambda row: row["WRank"] if row["Winner_position"] == 1 else row["LRank"], axis=1)
df_atp_X["P0Rank"] = df_atp_X.apply(lambda row: row["WRank"] if row["Winner_position"] == 0 else row["LRank"], axis=1)
df_atp_X=df_atp_X.drop("WRank", axis=1)
df_atp_X=df_atp_X.drop("LRank", axis=1)
for cols in ( ('AvgL', 'AvgW'), ('B&WL', 'B&WW'), ('B365L', 'B365W'), ('CBL', 'CBW'), ('EXL', 'EXW'), ('GBL', 'GBW'), \
    ('IWL', 'IWW'),('LBL', 'LBW'), ('MaxL', 'MaxW'), ('PSL', 'PSW'), \
    ('SBL', 'SBW'), ('SJL', 'SJW'), ('UBL', 'UBW')):
    suffix=cols[1][:-1]
    df_atp_X["P1"+suffix] = df_atp_X.apply(lambda row: row[cols[1]] if row["Winner_position"] == 1 else row[cols[0]], axis=1)
    df_atp_X["P0"+suffix] = df_atp_X.apply(lambda row: row[cols[1]] if row["Winner_position"] == 0 else row[cols[0]], axis=1)
    df_atp_X=df_atp_X.drop(cols[0], axis=1)
    df_atp_X=df_atp_X.drop(cols[1], axis=1) 
column_names_for_onehot = df_atp_X.columns[3:6]
encoded_atp_df = pd.get_dummies(df_atp_X, columns=column_names_for_onehot, drop_first=True)

encoded_atp_df['Date'] = pd.to_datetime(encoded_atp_df['Date'], format = '%Y-%m-%dT', errors = 'coerce')
encoded_atp_df['Date_year'] = encoded_atp_df['Date'].dt.year
encoded_atp_df['Date_month'] = encoded_atp_df['Date'].dt.month
encoded_atp_df['Date_week'] = encoded_atp_df['Date'].dt.week
encoded_atp_df['Date_day'] = encoded_atp_df['Date'].dt.day
encoded_atp_df=encoded_atp_df.drop("Date", axis=1)
f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(encoded_atp_df.drop("Winner_position", axis=1).corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

corr_matrix = encoded_atp_df.drop("Winner_position", axis=1).corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
print("Features to be dropped: ",to_drop)
# Drop features 
encoded_atp_df.drop(to_drop, axis=1, inplace=True)
# encoded_atp_df.head()
from pandas.plotting import scatter_matrix
encoded_atp_df.hist()
plt.gcf().set_size_inches(25, 25)
sns.set(color_codes=True)
encoded_atp_df = encoded_atp_df.loc[:,encoded_atp_df.apply(pd.Series.nunique) != 1]
year_to_predict = 2017

df_train = encoded_atp_df.iloc[df_atp[df_atp["Date"].dt.year != year_to_predict].index]
df_test = encoded_atp_df.iloc[df_atp[df_atp["Date"].dt.year == year_to_predict].index]

X_train = df_train.drop(["Winner_position"], axis=1)
y_train = df_train["Winner_position"]

X_test = df_test.drop(["Winner_position"], axis=1)
y_test = df_test["Winner_position"]

print("Training Set Shape: ",X_train.shape,  y_train.shape)
print("Test Set Shape:     ",X_test.shape,  y_test.shape)
sc = StandardScaler()  
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 5, stop = 40, num = 5)]
# number of features at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(4, 32, num = 6)]
max_depth.append(None)
criterion=['entropy','gini']
# Method of selecting samples for training each tree
bootstrap = [True,False]
# create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth,
 'criterion': criterion,
 'bootstrap': bootstrap  
 }
# Random search of parameters
rfc_random = RandomizedSearchCV(RandomForestClassifier(), param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2,
                                random_state=42, n_jobs = -1)
# Fit the model
rfc_random.fit(X_train_scaled, y_train)
# print results
print("Best Parameters for Random Forest: ",rfc_random.best_params_)
names_of_classifier = ["Random Forest","Decision Tree","Linear SVM","K-Nearest Neighbors",  "SVM-RBF", "AdaBoost"]

classifier = [
    RandomForestClassifier(n_estimators= 31, criterion="gini", bootstrap = False, max_depth=4, max_features = 'auto',class_weight="balanced"),
    DecisionTreeClassifier(max_depth=9),
    SVC(kernel="linear", C=0.03),
    KNeighborsClassifier(6, n_jobs=-1),
    SVC(gamma=3, C=1),
    AdaBoostClassifier()]

for name, classifier in zip(names_of_classifier, classifier):
    classifier.fit(X_train_scaled,y_train)
    
    y_predict=classifier.predict(X_test_scaled)
    y_Train_predict=classifier.predict(X_train_scaled)
    print("Classifier: ",name)
    print("\nAccuracy for Test Set: ",accuracy_score(y_test, y_predict))
    print( "Mean Squared Error for Test Set: ",round(mean_squared_error(y_test,y_predict), 3))
    print("Confusion matrix for Test Set \n",confusion_matrix(y_test,y_predict))
    print(classification_report(y_test,y_predict))
    fpr, tpr, thresholds= metrics.roc_curve(y_test,y_predict)
    auc = metrics.roc_auc_score(y_test,y_predict, average='macro', sample_weight=None)
    print("ROC Curve for for Test Set \n")
    sns.set_style('darkgrid')
    sns.lineplot(fpr,tpr,color ='blue')
    plt.show()
    
    
    print("\nAccuracy for Train Set: ",accuracy_score(y_train, y_Train_predict))
    print( "Mean Squared Error for Train Set: ",round(mean_squared_error(y_train,y_Train_predict), 3))
    print("Confusion matrix for Train Set \n",confusion_matrix(y_train,y_Train_predict))
    print(classification_report(y_train,y_Train_predict))
    fpr_train, tpr_train, thresholds_train= metrics.roc_curve(y_train,y_Train_predict)
    auc_train = metrics.roc_auc_score(y_train,y_Train_predict, average='macro', sample_weight=None)
    print("ROC Curve for for Train Set \n")
    sns.set_style('darkgrid')
    sns.lineplot(fpr_train,tpr_train,color ='red')
    plt.show()
    
    print("--------------------------------------xxx--------------------------------------\n\n")
    