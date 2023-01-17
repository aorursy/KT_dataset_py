import pandas as pd



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import f1_score



import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/fish-market/Fish.csv')
data.columns
data = data.rename(columns = {'Length1':'Vertical_Length', 'Length2':'Diagonal_Length', 

                       'Length3':'Cross_Length','Width':'Diagonal_Width'})
data.head()
data.describe()
data.info()
hist_plot = data.hist(bins = 50, figsize = (10, 10))
data['Species'].value_counts() / len(data) * 100
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data['Species']):    

    strat_train_set = data.loc[train_index]    

    strat_test_set = data.loc[test_index]
strat_train_set['Species'].value_counts() / len(strat_train_set) * 100
strat_test_set['Species'].value_counts() / len(strat_test_set) * 100
features = [i for i in data.columns if i != 'Species']
X = strat_train_set[features]

cor_matrix = X.corr()

cor_matrix['Weight'].sort_values(ascending = False)
X_train = strat_train_set[features]

X_test = strat_test_set[features]



y_train = strat_train_set['Species']

y_test = strat_test_set['Species']
model = LogisticRegression()

model.fit(X_train, y_train)
f1 = f1_score(model.predict(X_test), y_test, average = 'micro')

print(f'F1 score of model: {f1}')