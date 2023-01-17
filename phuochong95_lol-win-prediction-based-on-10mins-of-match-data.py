# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, log_loss
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import dataset
df= pd.read_csv("/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
df.head(20)
#Checking general description of dataframe
df.info()
#Check win rate of blue
blue_wr= df["blueWins"].sum()/df.shape[0]
print("win rate of blues is: ", blue_wr)
#Create a set of new columns
df1= df.copy()
df1["WardPlaceDiff"]= df1["blueWardsPlaced"]- df1["redWardsPlaced"]
df1["WardsDestroyedDiff"]= df1["blueWardsDestroyed"]-df1["redWardsDestroyed"]
df1["FirstBloodDiff"]= df1["blueFirstBlood"]-df1["redFirstBlood"]
df1["KillDiff"]= df1["blueKills"]-df1["redKills"]
df1["DeathsDiff"]= df1["blueDeaths"]-df1["redDeaths"]
df1["AssistsDiff"]= df1["blueAssists"]-df1["redAssists"]
df1["EliteMonstersDiff"]= df1["blueEliteMonsters"]-df1["redEliteMonsters"]
df1["DragonsDiff"]= df1["blueDragons"]-df1["redDragons"]
df1["HeraldDiff"]= df1["blueHeralds"]-df1["redHeralds"]
df1["TowersDestroyedDiff"]= df1["blueTowersDestroyed"]-df1["redTowersDestroyed"]
df1["AvgLevelDiff"]= df1["blueAvgLevel"]-df1["redAvgLevel"]
df1["TotalMinionsKilledDiff"]= df1["blueTotalMinionsKilled"]-df1["redTotalMinionsKilled"]
df1["TotalJungleMinionsKilledDiff"]= df1["blueTotalJungleMinionsKilled"]- df1["redTotalJungleMinionsKilled"]
df1["CSPerMinDiff"]= df1["blueCSPerMin"]-df1["redCSPerMin"]
df1["GoldPerMinDiff"]= df1["blueGoldPerMin"]- df1["redGoldPerMin"]
columns = list(map(lambda x: ("Diff" in x) or (x=="blueWins"), df1.columns))
df_diff= df1.loc[:, columns]
df_diff= df_diff.drop(["redGoldDiff", "redExperienceDiff", "DeathsDiff"], axis=1)
df_diff
#Checking correlation
df_diff_cormat = df_diff.corr()

#Masking null value to half upper matrix table
lower_triangle_mask = np.triu(np.ones(df_diff_cormat.shape)).astype(np.bool)
df_diff_cormat_lower = df_diff_cormat.mask(lower_triangle_mask)
                              
plt.figure(figsize=(20,10))
sns.heatmap(df_diff_cormat_lower, annot= True, cmap= "RdBu_r")
#Choosing most correlated features with "BlueWins" 
feat = ["blueGoldDiff", "blueExperienceDiff", "FirstBloodDiff", "KillDiff","AssistsDiff","EliteMonstersDiff", "AvgLevelDiff","TotalMinionsKilledDiff","CSPerMinDiff","GoldPerMinDiff"]
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
x= df_diff[feat]
y=df_diff["blueWins"]

#Split data to train and test dataset
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
#Fitting model
LR= LogisticRegression()

LR.fit(x_train,y_train)

#Checking accuracy
y_pred= LR.predict(x_test)
accuracy= accuracy_score(y_test,y_pred)
print("accuracy score is: ", accuracy)