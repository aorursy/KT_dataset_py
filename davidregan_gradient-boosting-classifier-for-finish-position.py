import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.ensemble import  GradientBoostingClassifier # classifier
df = pd.read_csv("../input/greyhound-racing-uk-predict-finish-position/data_final.csv")
df.head()
df.info()
# Features
features = ['Trap', 'BSP', 'Time_380', 'Finish_Recent', 'Finish_All', 'Stay_380',\
            'Races_All','Odds_Recent','Odds_380', 'Distance_Places_All', 'Dist_By',\
            'Races_380', 'Odds','Last_Run','Early_Time_380', 'Early_Recent' ,\
            'Distance_All', 'Wins_380', 'Grade_380','Finish_380','Early_380',\
            'Distance_Recent', 'Public_Estimate','Wide_380', 'Favourite']
# Target
target = ['Finished']
df[features].corr()
df[['BSP','Odds','Public_Estimate','Finished']].corr()
features.remove('Odds')
features.remove('Public_Estimate')
print(features)
print("\nThere are now",len(features),"features remaining.")
train=df.sample(frac=0.80,random_state=10) #random state is a seed value
test=df.drop(train.index)
# train_X, train_y
train_X = train[features]
train_y = train[target]

# test_X, test_y
test_X = test[features]
test_y = test[target]
# Create model
model = GradientBoostingClassifier(n_estimators = 10, max_features = None, min_samples_split = 2)
model.fit(train_X, train_y.values.ravel())
# evaluate the model on TRAINING DATA
accuracy = model.score(train_X, train_y)
print('    Training Model Accuracy:    ' + str(round(accuracy*100,2)) + '%')
# evaluate the model on Test data
accuracy = model.score(test_X, test_y)
print('    Test Model Accuracy:  ' + str(round(accuracy*100,2)) + '%')
# evaluate the market on Test data
# the feature 'Public_Estimate' gives the market prediction of finish position for each greyhound.
market_data = list(zip(test['Public_Estimate'], test['Finished']))
total = len(list(market_data))
count=0
for val in market_data:
    if val[0] == val[1]:
        count+=1
print('    Test Market Accuracy:      ' + str(round(count/total,3)*100) + '%')  # - - test   