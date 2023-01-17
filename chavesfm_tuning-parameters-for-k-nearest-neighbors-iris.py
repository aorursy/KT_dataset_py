import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.neighbors import KNeighborsClassifier # Classification

from sklearn.preprocessing import StandardScaler # Standardize features

from sklearn.model_selection import train_test_split # Split dataset

from sklearn.metrics import classification_report, accuracy_score # Useful metrics model



import matplotlib.pyplot as plt # Visualization

import seaborn as sns # Visualization



import warnings # Ignoe warnings

warnings.filterwarnings("ignore")



sns.set_style('whitegrid') # Set plot style

%matplotlib inline 



import os

print(os.listdir("../input"))

iris_df = pd.read_csv("../input/Iris.csv")

iris_df['Target'] = iris_df['Species'].map({specie:value for value, specie in enumerate(iris_df['Species'].unique())})

iris_df.head()
iris_df.info()
iris_df.isnull().sum()
features_on_off = {'Id':False,

                   'SepalLengthCm':True,

                   'SepalWidthCm':True,

                   'PetalLengthCm':True,

                   'PetalWidthCm':True,

                   'Species':False,

                   'Target':False}



features_on = [feature for feature, state in features_on_off.items() if state]
g = sns.PairGrid(data=iris_df, hue='Species', diag_sharey=False, vars=features_on)

g.map_upper(sns.scatterplot)

g.map_diag(sns.distplot)

g.map_lower(sns.kdeplot)

g.add_legend()
std_scaler = StandardScaler().fit(iris_df[features_on])

scaled_features = std_scaler.transform(iris_df[features_on])
scaled_features = pd.DataFrame(data=scaled_features, columns=features_on)

scaled_features.head()
scaled_features.describe()
X = scaled_features[features_on]

y = iris_df['Target']
n_neighbors = np.arange(1,26)

p_values = np.arange(1,11)

error_rate = {p:{'train':[], 'test':[]} for p in p_values}

n_samples = 200



for k in n_neighbors:

    train_error, test_error = 0, 0

    for p_value in p_values:

        for s in range(n_samples):

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)        

            knn = KNeighborsClassifier(n_neighbors=k, p=p_value)

            knn.fit(X_train, y_train)               

            train_error += np.mean(knn.predict(X_train) != y_train)        

            test_error += np.mean(knn.predict(X_test) != y_test)

        error_rate[p_value]['train'].append(train_error/n_samples)

        error_rate[p_value]['test'].append(test_error/n_samples)
best_k_p = {}

for p_value in p_values:    

    poly_coeff = np.polyfit(1/n_neighbors[8:], error_rate[p_value]['test'][8:], 2)

    poly = lambda x: poly_coeff[0]*x**2 + poly_coeff[1]*x + poly_coeff[2]

    k_min = -poly_coeff[1]/(2*poly_coeff[0])

    error_min = poly(k_min)        

    pair_k_error = list(zip(1/n_neighbors, error_rate[p_value]['test']))    

    best_k_p[p_value] = int(1/sorted(pair_k_error, key=lambda pair: np.sqrt(np.power(pair[0]-k_min,2) + np.power(pair[1]-error_min,2)))[0][0])
for p, k in best_k_p.items():

    print(f'Power of Minkowski metric: {p}; Best number of neighbors: {k}')
def plot_error(neighbors, train_error, test_error, best_k, title, ax):

    

    k_inv = 1.0/neighbors



    min_x, max_x = min(k_inv), max(k_inv)

    min_y, max_y = min(min(train_error, test_error)), max(max(train_error, test_error))



    ax.plot(k_inv, train_error, color='black', marker='o', markerfacecolor='blue', markeredgecolor='white', markersize=10, label='Train Error')

    ax.plot(k_inv, test_error, color='black', marker='o', markerfacecolor='lime', markeredgecolor='white', markersize=10, label='Test Error')



    ax.vlines(1/best_k, min_y-0.015, max_y+0.025, label=f'The Bias-Variance Trade-Oﬀ (k={best_k})', color='green', linestyle='--', linewidth=5)



    ax.set_xlim((min_x-0.05, max_x+0.01))

    ax.set_ylim((min_y-0.005, max_y+0.005))



    x_right = np.linspace(1/best_k, max_x+0.05, 10)

    ax.fill_between(x_right, len(x_right)*[min_y-0.015], len(x_right)*[max_y+0.025], alpha=0.5, color='red', label='High Variance Low Bias')



    x_left = np.linspace(min_x-0.15, 1/best_k, 10)

    ax.fill_between(x_left, len(x_left)*[min_y-0.015], len(x_left)*[max_y+0.025], alpha=0.5, color='orange', label='Low Variance High Bias')



    ax.set_xlabel('$1/k$', fontsize=20)

    ax.set_ylabel('Error', fontsize=20)

    

    ax.set_title(title, fontsize=20)



    ax.legend(loc='best', fontsize=15)    
fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(20,30), sharey=False)



p = 1

for row in range(5):

    for col in range(2):

        plot_error(n_neighbors, error_rate[p]['train'], error_rate[p]['test'], best_k_p[p], f'$p={p}$', ax[row][col])

        p+=1

        

plt.tight_layout()        
min_error = {'train':[], 'test':[]}



for p, k in best_k_p.items():

    min_error['train'].append(error_rate[p]['train'][k])

    min_error['test'].append(error_rate[p]['test'][k])
plt.figure(figsize=(12,8))



plt.plot(1/np.array(p_values), min_error['train'], marker='o', markersize=10, linestyle='--', label='Train Error')

plt.plot(1/np.array(p_values), min_error['test'], marker='o',markersize=10, linestyle='--', label='Test Error')



plt.xlabel('1/p', fontsize=20)

plt.ylabel('Error', fontsize=20)



plt.legend(loc='best', fontsize=15)
best_p = np.array(min_error['test']).argmin() + 1

best_k = best_k_p[best_p]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=best_k, p=best_p)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f'Accuracy: {100*accuracy_score(y_pred, y_test):.0f}%')

print(classification_report(y_test, y_pred))
from eli5.sklearn import PermutationImportance # Feature importance

from eli5 import show_weights



perm = PermutationImportance(knn).fit(X_test, y_test)

show_weights(perm, feature_names = features_on)
xx, yy = np.meshgrid(np.linspace(X['PetalLengthCm'].min()-0.1, X['PetalLengthCm'].max()+0.1, 500),

                     np.linspace(X['PetalWidthCm'].min()-0.1, X['PetalWidthCm'].max()+0.1, 500))
z = knn.predict(np.c_[np.zeros(xx.size), np.zeros(xx.size), xx.ravel(), yy.ravel()])

z = z.reshape(xx.shape)
plt.figure(figsize=(12,8))

plt.scatter(x='PetalLengthCm', y='PetalWidthCm', data=X, c=iris_df['Target'], cmap='viridis', edgecolors='black', s=100)

plt.contourf(xx, yy, z, cmap='viridis', alpha=0.5)

plt.xlabel('Petal Length', fontsize=20)

plt.ylabel('Petal Width', fontsize=20)

plt.title(f'k={best_k}; p={best_p}', fontsize=20)