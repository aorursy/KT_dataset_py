import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd

filename = '/kaggle/input/mobile-legend-data.xlsx'
df = pd.read_excel(filename)
df.head()
heroArray = ['Akai', 'Aldous', 'Alice', 'Alpha', 'Alucard', 'Angela', 'Argus', 'Atlas', 'Aurora', 'Badang', 'Balmond', 'Bane', 'Baxia', 'Belerick', 'Bruno', 'Carmilla', 'Cecilion', 'Change', 'Chou', 'Claude', 'Clint', 'Cyclops', 'Diggie', 'Dyrroth', 'Esmeralda', 'Estes', 'Eudora', 'Fanny', 'Faramis', 'Franco', 'Freya', 'Gatotkaca', 'Gord', 'Granger', 'Grock', 'Guinevere', 'Gusion', 'Hanabi', 'Hanzo', 'Harith', 'Harley', 'Hayabusa', 'Helcurt', 'Hilda', 'Hylos', 'Irithel', 'Jawhead', 'Johnson', 'Kadita', 'Kagura', 'Kaja', 'Karina', 'Karrie', 'Khufra', 'Kimmy', 'Lancelot', 'Lapu-Lapu', 'Layla', 'Leomord', 'Lesley', 'Ling', 'Lolita', 'Lunox', 'Luo Yi', 'Lylia', 'Martis', 'Masha', 'Minotaur', 'Minsitthar', 'Miya', 'Moskov', 'Nana', 'Natalia', 'Odette', 'Pharsa', 'Popol And Kupa', 'Rafaela', 'Roger', 'Ruby', 'Saber', 'Selena', 'Silvanna', 'Sun', 'Terizla', 'Thamuz', 'Tigreal', 'Uranus', 'Vale', 'Valir', 'Vexana', 'Wanwan', 'XBorg', 'Yi Sun-Shin', 'Yu Zhong', 'Zhask', 'Zilong']
def convertHero (var, arr):
  for count, i in enumerate(arr, start=1):
    if i.casefold() == var.casefold():
      return count
df['rh_1'] = df['rh_1'].apply(lambda x: convertHero(x, heroArray))
df['rh_2'] = df['rh_2'].apply(lambda x: convertHero(x, heroArray))
df['rh_3'] = df['rh_3'].apply(lambda x: convertHero(x, heroArray))
df['rh_4'] = df['rh_4'].apply(lambda x: convertHero(x, heroArray))
df['rh_5'] = df['rh_5'].apply(lambda x: convertHero(x, heroArray))

df['dh_1'] = df['dh_1'].apply(lambda x: convertHero(x, heroArray))
df['dh_2'] = df['dh_2'].apply(lambda x: convertHero(x, heroArray))
df['dh_3'] = df['dh_3'].apply(lambda x: convertHero(x, heroArray))
df['dh_4'] = df['dh_4'].apply(lambda x: convertHero(x, heroArray))
df['dh_5'] = df['dh_5'].apply(lambda x: convertHero(x, heroArray))
df.isnull().any()
roleArray = ['Tank', 'Assassin', 'Marksman', 'Fighter', 'Support', 'Mage']

def convertRole (var, arr):
  for count, role in enumerate(arr, start = 1):
    if role.casefold() == var.casefold():
      return count
    else:
      pass
df['rh_1_role'] = df['rh_1_role'].apply(lambda x: convertRole(x, roleArray))
df['rh_2_role'] = df['rh_2_role'].apply(lambda x: convertRole(x, roleArray))
df['rh_3_role'] = df['rh_3_role'].apply(lambda x: convertRole(x, roleArray))
df['rh_4_role'] = df['rh_4_role'].apply(lambda x: convertRole(x, roleArray))
df['rh_5_role'] = df['rh_5_role'].apply(lambda x: convertRole(x, roleArray))

df['dh_1_role'] = df['dh_1_role'].apply(lambda x: convertRole(x, roleArray))
df['dh_2_role'] = df['dh_2_role'].apply(lambda x: convertRole(x, roleArray))
df['dh_3_role'] = df['dh_3_role'].apply(lambda x: convertRole(x, roleArray))
df['dh_4_role'] = df['dh_4_role'].apply(lambda x: convertRole(x, roleArray))
df['dh_5_role'] = df['dh_5_role'].apply(lambda x: convertRole(x, roleArray))
df.isnull().any()
import seaborn as sns

sns.countplot(df['outcome'], label = 'Count')
df['game_duration'] = pd.to_datetime(df['game_duration'], format='%M:%S')
df['duration_min'] = df['game_duration'].dt.minute
df['duration_second'] = df['game_duration'].dt.second
def getRadiantGoldperMin (x):
  return (x['r_total_gold'] / (x['duration_min'] + (x['duration_second'] / 60)))

def getDireGoldperMin (x):
  return (x['r_total_gold'] / (x['duration_min'] + (x['duration_second'] / 60)))
df['r_gold_min'] = df.apply(getRadiantGoldperMin, axis=1)
df['d_gold_min'] = df.apply(getDireGoldperMin, axis=1)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df['outcome'])

df['outcome'] = le.transform(df['outcome'])
target = df.outcome
X = df.drop(['r_total_gold', 'd_total_gold', 'game_duration', 'outcome', 'duration_min', 'duration_second'], axis=1)
#https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

pred = cross_val_predict(LogisticRegression(), X, target, cv=5)
print (metrics.accuracy_score(target, pred))

print (metrics.classification_report(target, pred))
#https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

from sklearn.model_selection import cross_val_score

pred = cross_val_score(LogisticRegression(), X, target, cv=2)
print('Prediction: ', pred)

print('Accuracy: %0.2f (+/- %0.2f' % (pred.mean(), pred.std() * 2))
from sklearn.model_selection import ShuffleSplit

n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=2, test_size=0.3, random_state=0)
cross_val_score(LogisticRegression(), X, target, cv=cv)