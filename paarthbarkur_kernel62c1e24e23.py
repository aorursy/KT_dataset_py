# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


matches=pd.read_csv("/kaggle/input/ipl/matches.csv")
conditions = [matches["venue"] == "Rajiv Gandhi International Stadium, Uppal",matches["venue"] == "Maharashtra Cricket Association Stadium",

              matches["venue"] == "Saurashtra Cricket Association Stadium", matches["venue"] == "Holkar Cricket Stadium",

              matches["venue"] == "M Chinnaswamy Stadium",matches["venue"] == "Wankhede Stadium",

              matches["venue"] == "Eden Gardens",matches["venue"] == "Feroz Shah Kotla",

              matches["venue"] == "Punjab Cricket Association IS Bindra Stadium, Mohali",matches["venue"] == "Green Park",

              matches["venue"] == "Punjab Cricket Association Stadium, Mohali",matches["venue"] == "Dr DY Patil Sports Academy",

              matches["venue"] == "Sawai Mansingh Stadium", matches["venue"] == "MA Chidambaram Stadium, Chepauk", 

              matches["venue"] == "Newlands", matches["venue"] == "St George's Park" , 

              matches["venue"] == "Kingsmead", matches["venue"] == "SuperSport Park",

              matches["venue"] == "Buffalo Park", matches["venue"] == "New Wanderers Stadium",

              matches["venue"] == "De Beers Diamond Oval", matches["venue"] == "OUTsurance Oval", 

              matches["venue"] == "Brabourne Stadium",matches["venue"] == "Sardar Patel Stadium", 

              matches["venue"] == "Barabati Stadium", matches["venue"] == "Vidarbha Cricket Association Stadium, Jamtha",

              matches["venue"] == "Himachal Pradesh Cricket Association Stadium",matches["venue"] == "Nehru Stadium",

              matches["venue"] == "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",matches["venue"] == "Subrata Roy Sahara Stadium",

              matches["venue"] == "Shaheed Veer Narayan Singh International Stadium",matches["venue"] == "JSCA International Stadium Complex",

              matches["venue"] == "Sheikh Zayed Stadium",matches["venue"] == "Sharjah Cricket Stadium",

              matches["venue"] == "Dubai International Cricket Stadium",matches["venue"] == "M. A. Chidambaram Stadium",

              matches["venue"] == "Feroz Shah Kotla Ground",matches["venue"] == "M. Chinnaswamy Stadium",

              matches["venue"] == "Rajiv Gandhi Intl. Cricket Stadium" ,matches["venue"] == "IS Bindra Stadium",matches["venue"] == "ACA-VDCA Stadium"]

values = ['Hyderabad', 'Mumbai', 'Rajkot',"Indore","Bengaluru","Mumbai","Kolkata","Delhi","Mohali","Kanpur","Mohali","Pune","Jaipur","Chennai","Cape Town","Port Elizabeth","Durban",

          "Centurion",'Eastern Cape','Johannesburg','Northern Cape','Bloemfontein','Mumbai','Ahmedabad','Cuttack','Jamtha','Dharamshala','Chennai','Visakhapatnam','Pune','Raipur','Ranchi',

          'Abu Dhabi','Sharjah','Dubai','Chennai','Delhi','Bengaluru','Hyderabad','Mohali','Visakhapatnam']

matches['city'] = np.where(matches['city'].isnull(),

                              np.select(conditions, values),

                              matches['city'])



#Removing records having null values in "winner" column

matches=matches[matches["winner"].notna()]
for team in matches["team1"].unique():

    print(team)
matches["team2"]=matches["team2"].replace("Rising Pune Supergiant","Rising Pune Supergiants")

matches["team1"]=matches["team1"].replace("Rising Pune Supergiant","Rising Pune Supergiants")

matches["winner"]=matches["winner"].replace("Rising Pune Supergiant","Rising Pune Supergiants")

matches["toss_winner"]=matches["toss_winner"].replace("Rising Pune Supergiant","Rising Pune Supergiants")

encoder= LabelEncoder()

matches["team1"]=encoder.fit_transform(matches["team1"])

matches["team2"]=encoder.fit_transform(matches["team2"])

matches["winner"]=encoder.fit_transform(matches["winner"].astype(str))

matches["toss_winner"]=encoder.fit_transform(matches["toss_winner"])

matches["venue"]=encoder.fit_transform(matches["venue"])
matches.loc[matches["winner"]==matches["team1"],"team1_win"]=1

matches.loc[matches["winner"]!=matches["team1"],"team1_win"]=0





matches.loc[matches["toss_winner"]==matches["team1"],"team1_toss_win"]=1

matches.loc[matches["toss_winner"]!=matches["team1"],"team1_toss_win"]=0





matches["team1_bat"]=0

matches.loc[(matches["team1_toss_win"]==1) & (matches["toss_decision"]=="bat"),"team1_bat"]=1
prediction_df=matches[["team1","team2","team1_toss_win","team1_bat","team1_win","venue"]]

X=["team1","team2","team1_toss_win","team1_bat","venue"]

y=matches.team1_win

#finding the higly correlated features

correlated_features = set()

correlation_matrix = prediction_df.drop('team1_win', axis=1).corr()



for i in range(len(correlation_matrix.columns)):

    for j in range(i):

        if abs(correlation_matrix.iloc[i, j]) > 0.9:

            column = correlation_matrix.columns[i]

            correlated_features.add(column)

            

prediction_df.drop(columns=correlated_features)
X_train,X_test,y_train,y_test=train_test_split(X,y)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "/kaggle/input/ipl/matches.csv"]).decode("utf8"))



matches=pd.read_csv('/kaggle/input/ipl/matches.csv')

matches.info()
matches[pd.isnull(matches["winner"])]
matches['winner'].fillna("Draw",inplace=True)
matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',

                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',

                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']

                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)



encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},

          'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},

          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},

          'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'Draw':14}}

matches.replace(encode, inplace=True)

matches.head()
matches[pd.isnull(matches['city'])]
matches['city'].fillna('Dubai',inplace=True)
matches.describe()
dicVal = encode['winner']

print(dicVal['RCB']) #key value

print(list(dicVal.keys())[list(dicVal.values()).index(2)])

matches=matches[['team1','team2','city','toss_decision','toss_winner','venue','winner']]
matches.head()


df = pd.DataFrame(matches)

df.describe()
temp1=df['toss_winner'].value_counts(sort=True)

temp2=df['winner'].value_counts(sort=True)



print('No of toss winners by each team')

for idx, val in temp1.iteritems():

   print('{} -> {}'.format(list(dicVal.keys())[list(dicVal.values()).index(idx)],val))

print('No of match winners by each team')

for idx, val in temp2.iteritems():

   print('{} -> {}'.format(list(dicVal.keys())[list(dicVal.values()).index(idx)],val))
df['winner'].hist( bins=50)
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(121)

ax1.set_xlabel('toss_winner')

ax1.set_ylabel('Count of toss winners')

ax1.set_title("toss winners")

temp1.plot(kind='bar')



ax2 = fig.add_subplot(122)

temp2.plot(kind = 'bar')

ax2.set_xlabel('winner')

ax2.set_ylabel('Probability of winning match by winning toss')

ax2.set_title("Probability of match winning by winning toss")
df.apply(lambda x: sum(x.isnull()),axis=0)
from sklearn.preprocessing import LabelEncoder

var_mod = ['city','toss_decision','venue']

le = LabelEncoder()

for i in var_mod:

    df[i] = le.fit_transform(df[i])

df.dtypes
#Import models from scikit learn module:

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold   

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import metrics



#Generic function for making a classification model and accessing performance:

def classification_model(model, data, predictors, outcome):

  model.fit(data[predictors],data[outcome])

  

  predictions = model.predict(data[predictors])

  

  accuracy = metrics.accuracy_score(predictions,data[outcome])

  print('Accuracy : %s' % '{0:.3%}'.format(accuracy))



  kf = KFold(data.shape[0], n_folds=5)

  error = []

  for train, test in kf:

    train_predictors = (data[predictors].iloc[train,:])

    

    train_target = data[outcome].iloc[train]

    

    model.fit(train_predictors, train_target)

    

    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

 

  print('Cross-Validation Score : %s' % '{0:.3%}'.format(np.mean(error)))



  model.fit(data[predictors],data[outcome])
from sklearn.ensemble import RandomForestRegressor

outcome_var=['winner']

predictor_var = ['team1','team2','toss_winner']

model = LogisticRegression()

classification_model(model, df,predictor_var,outcome_var)
df.head(2)
model = RandomForestClassifier(n_estimators=100)

outcome_var = ['winner']

predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']

classification_model(model, df,predictor_var,outcome_var)
#'team1', 'team2', 'venue', 'toss_winner','city','toss_decision'

team1='RCB'

team2='KKR'

toss_winner='RCB'



input=[dicVal[team1],dicVal[team2],'14',dicVal[toss_winner],'2','1']

input = np.array(input).reshape((1, -1))

output=model.predict(input)

print(list(dicVal.keys())[list(dicVal.values()).index(output)])
#feature importances: If we ignore teams, Venue seems to be one of important factors in determining winners 

#followed by toss winning, city

imp_input = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)

print(imp_input)