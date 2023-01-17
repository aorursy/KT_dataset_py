# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
# models packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.svm import SVC #Support vector machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision tree
from sklearn.ensemble import VotingClassifier #Ensemble learning
# other packages
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression,f_classif
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("/kaggle/input/road-traffic-injuries-deaths-catalonia-201020/Accidents_de_tr_nsit_amb_morts_o_ferits_greus_a_Catalunya.csv")
# Check first five rows
data.head()
# check distribution of whole dataset
data.describe()
# Check the data type of each column
data.info()
# Find out columns which have empty cells
columns_list = data.isnull().sum()
columns_list[columns_list>0]
# Drop columns
data = data.drop(columns=['D_REGULACIO_PRIORITAT','D_SUBTIPUS_TRAM','D_TITULARITAT_VIA','D_TRACAT_ALTIMETRIC'])
# Add values for other column 'pk',find a most frequency value for this empty cell
most_frequency_value = data['pk'].value_counts().idxmax()
data['pk'].fillna(most_frequency_value,inplace=True)
# Add values for other column 'D_CARACT_ENTORN',fill the empty cells according to the distribution of options
ratio = data['D_CARACT_ENTORN'].value_counts()/data['D_CARACT_ENTORN'].value_counts().sum() # distribution of options
total_empty = data['D_CARACT_ENTORN'].isnull().sum() # the total number of empty cells
assignation = (ratio*total_empty)//0.93 # assignation for each options
for i in range(len(assignation)): # fill the empty cells
    data['D_CARACT_ENTORN'].fillna(assignation.index[i],limit=int(assignation[i]),inplace=True)
# Add values for other column 'D_CIRCULACIO_MESURES_ESP',fill the empty cells with option 'No n'hi ha', as this option has 16347/16774 in this column
value = data['D_CIRCULACIO_MESURES_ESP'].value_counts().index[0]
data['D_CIRCULACIO_MESURES_ESP'].fillna(value,inplace=True)
# Add values for other column 'C_VELOCITAT_VIA',fill the empty cells according to the distribution of options
ratio = data['C_VELOCITAT_VIA'].value_counts()/data['C_VELOCITAT_VIA'].value_counts().sum() # distribution of options
total_empty = data['C_VELOCITAT_VIA'].isnull().sum() # the total number of empty cells
assignation = (ratio*total_empty)//1 # assignation for each options
for j,k in enumerate(assignation): # fill the empty cells,unlucky 7 empty cells can not be filled with the distribution of options
    if k>0.0:
        data['C_VELOCITAT_VIA'].fillna(assignation.index[j],limit=int(k),inplace=True)
data['C_VELOCITAT_VIA'].fillna(data['C_VELOCITAT_VIA'].value_counts().idxmax(),inplace=True)# fill the 7 empty cells with most frequency option
# Add values for other column 'D_CARRIL_ESPECIAL',fill the empty cells with option 'No n'hi ha', as this option has 14980/15942 in this column
value = data['D_CARRIL_ESPECIAL'].value_counts().index[0]
data['D_CARRIL_ESPECIAL'].fillna(value,inplace=True)
# Add values for other column 'C_VELOCITAT_VIA',fill the empty cells according to the distribution of options
ratio = data['D_SENTITS_VIA'].value_counts()/data['D_SENTITS_VIA'].value_counts().sum() # distribution of options
total_empty = data['D_SENTITS_VIA'].isnull().sum() # the total number of empty cells
assignation = (ratio*total_empty)//0.178 # assignation for each options
assignation.sum()
for j,k in enumerate(assignation): # fill the empty cells,unlucky 59 empty cells can not be filled with the distribution of options
    if k>0.0:
        data['D_SENTITS_VIA'].fillna(assignation.index[j],limit=int(k),inplace=True)
data['D_SENTITS_VIA'].fillna(data['D_SENTITS_VIA'].value_counts().idxmax(),inplace=True)# fill the 59 empty cells with most frequency option
# Check the data again, all of columns have no empty cells
data.isnull().sum()
# Begin to deal with object type columns
data.info()
# Column 'F_UNIT_DESC_IMPLICADES' only has one option:0,so drop this column
data = data.drop(columns=['F_UNIT_DESC_IMPLICADES'])
# Column 'D_SUBZONA' almostly has same meaning with 'zona', so drop this column
data = data.drop(columns=['D_SUBZONA'])
# Replace all of object columns with number
le = LabelEncoder()
object_list = data.dtypes
for i in object_list[object_list==object].index:
    temp = le.fit(data[i])
    data[i] = temp.transform(data[i])
# Check the data type again
data.info()
# Check relationship through correlation factors
corr=data.corr()
fig = plt.figure(figsize=(20,20))
r = sns.heatmap(corr, cmap='Purples')
r.set_title("Correlation ")
# For the feature 'zona',the impact factors as fellow, 'D_TIPUS_VIA ' and 'pk' are the top 2 features
score_ranking = corr.sort_values(by=["zona"],ascending=False).iloc[0].sort_values(ascending=False)
fig = plt.figure(figsize=(20,20))
abs(score_ranking[1:]).sort_values(ascending=False).plot(kind = 'bar')
abs(score_ranking).sort_values(ascending=False)
sns.pointplot(y="pk", x="zona", data=data, color='purple')