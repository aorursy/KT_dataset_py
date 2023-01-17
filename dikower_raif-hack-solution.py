import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
data = pd.read_csv("../input/merged_data.csv")
# Any results you write to the current directory are saved as output.
np.random.seed(seed=42)
data.head(10)
Nans = (100 - (data.isnull().sum()/data.shape[0]*100)).sort_values()
Nans.plot("bar", title="The filling percentage of columns")
to_plot = {}
for feature in data.columns:
    count = len(data[feature].unique())
    if count < 10:
        to_plot[feature] = count
to_plot = pd.Series(to_plot).sort_values(ascending=True)
to_plot.plot("bar", title="The number of unique values per feature")
to_drop = ["CadastralNumber", "43", "44", "20", "21", "48", "49"]
print("Before -",data.shape)
for column in data.columns:
    if data[column].isnull().sum() > data.shape[0] * 0.3 or len(data[column].unique()) == 1:
        to_drop.append(column)
data = data.drop(to_drop, axis=1).dropna()
print("After -",data.shape)
plt.bar(data["33"], data["Cost m2"])
plt.show()
data.head(4)
import math
def transformation(value, transform='square'): 
    if transform == 'log':
        return 0 if math.log1p(value) == None else max(0, min(10 ** 3, math.log1p(value))) 
    elif transform == 'sqrt':
        return math.sqrt(value + 3.0 / 8) 
    elif transform == 'square':
        return value ** 2 
    elif transform == 'sin':
        return math.sin(value)
    elif transform == 'cos': 
        return math.cos(value)
data["sum1"] = data["17"] + data["7"]
columns = data.drop(["Cost m2", "Cost"], axis=1).columns
possible_transformations = ["log", "sqrt", "square", "sin", "cos"]
for column in columns:
    for transform in possible_transformations:
        data[f"{column}_{transform}"] = data[column].apply(transformation,  args=(transform,))
print(data.columns)
import seaborn as sns
from string import ascii_letters
sns.set(style="white")

corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, vmax=.3, center=0, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})

def remove_collinear_features(x, threshold):
    y = x['Cost m2']
    x = x.drop(['Cost m2'], axis=1)
    # Считаем матрицу кореляций
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Сравниваем фичу каждую с каждой
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # Если кореляция больше, чем трешхолд
            if val >= threshold:
                drop_cols.append(col.values[0])

    # Удаляем по одному из каждой пары корелируемых
    drops = set(drop_cols)
    x = x.drop(drops, axis=1)
    
    # Возвращаем таргет в таблицу
    x['Cost m2'] = y
    return x

data = remove_collinear_features(data, 0.6)
corr = data.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, vmax=.3, center=0, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Onehot encoding
from sklearn.preprocessing import OneHotEncoder
encode_columns = [column for column in data.columns if len(data[column].unique()) < 6]
print(encode_columns)
enc = OneHotEncoder()
for column in encode_columns:
    X = enc.fit_transform(data[column].values.reshape(-1, 1)).toarray()
    OneHot = pd.DataFrame(X, columns = [f"{column}_{i}" for i in range(X.shape[1])])
    data = pd.concat([data, OneHot], axis=1)
data = data.drop(encode_columns, axis=1)

from sklearn.model_selection import train_test_split
Y = data["Cost m2"]
X = data.drop(["Cost m2", "Cost"], axis=1)
X_train, X_test_val, Y_train, Y_test_val = train_test_split(X, Y, test_size=0.1, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_test_val, Y_test_val, test_size=0.5, random_state=42)
import catboost as cb
cb_model = cb.CatBoostRegressor(
)
cb_model.fit(
    X_train, Y_train,
    use_best_model=True,
    eval_set=cb.Pool(X_val, Y_val),
    logging_level="Verbose",  # 'Silent', 'Verbose', 'Info', 'Debug'
    early_stopping_rounds=1,
    plot=True
)
# print(cb_model.score(X_test, Y_test))
cb_model.save_model("trained_model", format="cbm")
feature_importances = pd.Series(cb_model.feature_importances_, index=X_train.columns).sort_values(ascending=True)
feature_importances.plot(kind="bar")
cb_model.score(X_test, Y_test)
prediction = np.round(cb_model.predict(X_test), 2)
pd.DataFrame({"prediction": prediction, "test": Y_test}).head(1000)