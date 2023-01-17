from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
import pandas as pd
import numpy as np
from scipy import stats

athletes_only = pd.read_csv("../input/athletes.csv")
leaderboard = pd.read_csv("../input/leaderboard.15.csv")
athletes_only.drop(['name'], axis=1, inplace = True)
athletes_only.head()
leaderboard.head()
#merging the two df to get the full data set
leaderboard = leaderboard[['athlete_id', 'score']]
leaderboard = leaderboard.drop_duplicates(subset = ['athlete_id'], keep = 'first')
athletes = athletes_only.merge(leaderboard, left_on='athlete_id', right_on='athlete_id', how='inner')

#using the athletes only df, finding the numeric data and seting NaN to median of the column
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
athletesNum = athletes.select_dtypes(include=numerics)
athletesNum = athletesNum.fillna(athletesNum.median())

athletesNum.head()
athletesNum.describe()
athletesNum.drop(athletesNum[athletesNum.weight > 400].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.weight < 80].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.height > 100].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.fran > 400].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.helen > 1000].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.grace > 500].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.filthy50 > 3000].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.fgonebad > 500].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.run400 > 200].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.run5k > 5000].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.candj > 500].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.snatch > 300].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.deadlift > 500].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.backsq > 500].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.pullups > 100].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.deadlift <= 0].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.backsq <= 0].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.pullups <= 0].index, inplace=True)
athletesNum.drop(athletesNum[athletesNum.score > 1000].index, inplace=True)
athletesNum.loc[athletesNum.age == 0] = athletes.age.median()
col = athletesNum.columns.tolist()
athletesNum.drop_duplicates(subset=col, inplace = True)
athletesNum.drop(['athlete_id','weight', 'height', 'age'], axis = 1, inplace = True)
athletesNum.head()
athletesNum.describe()
#value we are trying to predict
labels = np.array(athletesNum['score'])
#remove score
athletesT = athletesNum.drop('score', axis = 1)
#feature name for later use
feature_list = list(athletesT.columns)
#convert to numpy array
features = np.array(athletesT)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# Split the data into training and testing sets    
trainX, testX, trainY, testY = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', trainX.shape)
print('Training Labels Shape:', trainY.shape)
print('Testing Features Shape:', testX.shape)
print('Testing Labels Shape:', testY.shape)
# Instantiate model with 10 decision trees
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
# Train the model on training data
rf.fit(trainX, trainY)
# Use the forest's predict method on the test data
pred = rf.predict(testX)
print(stats.describe(pred))
from sklearn.metrics import mean_squared_error
errors = abs(pred - testY)
mape = 100 * (errors / testY)
accuracy = 100 - np.mean(mape)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Mean Squared Error:',round(mean_squared_error(testY, pred), 2))
print('Accuracy:', round(accuracy, 2), '%')
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

