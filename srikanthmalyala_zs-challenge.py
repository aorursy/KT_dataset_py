# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
cr7_data = pd.read_csv("../input/data.csv")
columns= cr7_data.columns

print(columns)

cr7_data.head()
#Removing Columns Which are not required for prediction 

Final_data=cr7_data.drop(columns=["match_event_id","Unnamed: 0","knockout_match.1","game_season","team_name","date_of_game",'type_of_shot','type_of_combined_shot',

                                    "distance_of_shot.1","match_id","team_id","remaining_min.1","power_of_shot.1","remaining_sec.1",'shot_id_number'])

Cleaned_data=Final_data.dropna(subset=["is_goal"])
Cleaned_data.head()
Cleaned_data=Cleaned_data.replace(to_replace='Left Side(L)',value = 0)

Cleaned_data=Cleaned_data.replace(to_replace='Left Side Center(LC)',value = 1)

Cleaned_data=Cleaned_data.replace(to_replace='Right Side Center(RC)',value = 2)

Cleaned_data=Cleaned_data.replace(to_replace='Center(C)',value = 3)

Cleaned_data=Cleaned_data.replace(to_replace='Right Side(R)',value = 4)

Cleaned_data=Cleaned_data.replace(to_replace='Mid Ground(MG)',value = 5)



Cleaned_data=Cleaned_data.replace(to_replace='Mid Range',value = 0)

Cleaned_data=Cleaned_data.replace(to_replace='Goal Area',value = 1)

Cleaned_data=Cleaned_data.replace(to_replace='Goal Line',value = 2)

Cleaned_data=Cleaned_data.replace(to_replace='Penalty Spot',value = 3)

Cleaned_data=Cleaned_data.replace(to_replace='Mid Ground Line',value = 4)

Cleaned_data=Cleaned_data.replace(to_replace='Right Corner',value = 5)

Cleaned_data=Cleaned_data.replace(to_replace='Left Corner',value = 6)





Cleaned_data=Cleaned_data.replace(to_replace='Less Than 8 ft.',value = 4)

Cleaned_data=Cleaned_data.replace(to_replace='8-16 ft.',value = 12)

Cleaned_data=Cleaned_data.replace(to_replace='16-24 ft.',value = 20)

Cleaned_data=Cleaned_data.replace(to_replace='24+ ft.',value = 28)

Cleaned_data
#finding percentage of data missing in each column

percent_missing=Cleaned_data.isnull().sum()*100/len(Cleaned_data)

missing_values_df=pd.DataFrame({"column_name":Cleaned_data.columns,

                               'percent_missing':percent_missing})







missing_values_df.sort_values('percent_missing', inplace=True)

missing_values_df

Cleaned_data=pd.get_dummies(Cleaned_data,columns=["home/away",'range_of_shot','lat/lng'])
Cleaned_data.head()
Cleaned_data=Cleaned_data.dropna(how='any',axis=0)

X = Cleaned_data.drop(['is_goal'],axis=1)

y = Cleaned_data['is_goal']
print(len(Cleaned_data))
from sklearn.model_selection import cross_val_score, train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=7)
from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
# Fit the model with best_tree_size. Fill in argument to make optimal size

final_model = DecisionTreeRegressor(max_leaf_nodes =80, random_state=2)



# fit the final model

final_model.fit(X_train, y_train)

mean_absolute_error(final_model.predict(X_test),y_test)



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error

model=LogisticRegression()

model.fit(X_train,y_train)

pred=model.predict(X_test)

mean_absolute_error(pred,y_test)
from sklearn.ensemble import RandomForestClassifier

model_rf=RandomForestClassifier()

model_rf.fit(X_train,y_train)

pred_rf=model_rf.predict(X_test)

mean_absolute_error(pred_rf,y_test)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

model_gb=GradientBoostingClassifier(n_estimators=40)

model_gb.fit(X_train,y_train)

pred_gb=model_gb.predict(X_test)

print(mean_absolute_error(y_test,pred_gb))

print(accuracy_score(y_test,pred_gb))
pred_data=Final_data[Final_data['is_goal'].isnull()]

pred_data_1=pred_data.drop(['is_goal'],axis=1)

pred_data_1=pd.get_dummies(pred_data_1,columns=['home/away','range_of_shot','lat/lng'])



pred_data_1=pred_data_1.replace(to_replace='Left Side(L)',value = 0)

pred_data_1=pred_data_1.replace(to_replace='Left Side Center(LC)',value = 1)

pred_data_1=pred_data_1.replace(to_replace='Right Side Center(RC)',value = 2)

pred_data_1=pred_data_1.replace(to_replace='Center(C)',value = 3)

pred_data_1=pred_data_1.replace(to_replace='Right Side(R)',value = 4)

pred_data_1=pred_data_1.replace(to_replace='Mid Ground(MG)',value = 5)



pred_data_1=pred_data_1.replace(to_replace='Mid Range',value = 0)

pred_data_1=pred_data_1.replace(to_replace='Goal Area',value = 1)

pred_data_1=pred_data_1.replace(to_replace='Goal Line',value = 2)

pred_data_1=pred_data_1.replace(to_replace='Penalty Spot',value = 3)

pred_data_1=pred_data_1.replace(to_replace='Mid Ground Line',value = 4)

pred_data_1=pred_data_1.replace(to_replace='Right Corner',value = 5)

pred_data_1=pred_data_1.replace(to_replace='Left Corner',value = 6)





pred_data_1=pred_data_1.replace(to_replace='Less Than 8 ft.',value = 4)

pred_data_1=pred_data_1.replace(to_replace='8-16 ft.',value = 12)

pred_data_1=pred_data_1.replace(to_replace='16-24 ft.',value = 20)

pred_data_1=pred_data_1.replace(to_replace='24+ ft.',value = 28)





pred_data_2=pred_data_1.dropna(how='any',axis=0)

pred_data_2

model_rf.predict(pred_data_2)
array=model_gb.predict(pred_data_2)
final_pred = pd.DataFrame(array,columns=["is_goal"])
final_pred.head()
submission=cr7_data[cr7_data['is_goal'].isnull()]

submission.columns
submission=submission.drop(columns=['Unnamed: 0', 'match_event_id', 'location_x', 'location_y',

       'remaining_min', 'power_of_shot', 'knockout_match', 'game_season',

       'remaining_sec', 'distance_of_shot', 'area_of_shot',

       'shot_basics', 'range_of_shot', 'team_name', 'date_of_game',

       'home/away', 'lat/lng', 'type_of_shot',

       'type_of_combined_shot', 'match_id', 'team_id', 'remaining_min.1',

       'power_of_shot.1', 'knockout_match.1', 'remaining_sec.1','is_goal',

       'distance_of_shot.1'],)

submission=submission.dropna(subset=['shot_id_number'],axis=0)
submission.head()
submission.to_csv('new.csv')

final_pred.to_csv('new.csv')