import pandas as pd

import numpy as np





import matplotlib.pyplot as plt

import seaborn as sns





%matplotlib inline



# set seaborn style to white

sns.set_style("white")
df=pd.read_csv('../input/Parking_Lyon_Kaggle.csv')
df.shape
pd.options.display.max_columns = None # me montre toutes mes columns

df.head()


Parking_Payant = (df['reglementation'] == 'Payant').sum()

Parking_Gratuit = (df['reglementation'] == 'Gratuit').sum()

Parking_ZoneBleu = (df['reglementation'] == 'Zone bleue').sum()





proportions = [Parking_Payant, Parking_Gratuit, Parking_ZoneBleu]





plt.pie(



    proportions,

    



    labels = ['Payant', 'Gratuit','Zone bleue'],

    



    shadow = False,

    



    colors = ['blue','orange', 'green'],

    



    explode = (0.15 , 0, 0.15 ),

    

   

    startangle = 90,

    



    autopct = '%1.1f%%'

    )





plt.axis('equal')





plt.title("Reglementation Proportion",loc='left')





plt.tight_layout()

plt.show()


lm = sns.lmplot(x = 'capacitepmr', y = 'capacite', data = df, hue = 'reglementation', fit_reg=False)





lm.set(title = 'capacite x capacitepmr')





axes = lm.axes

axes[0,0].set_ylim(-5,)

axes[0,0].set_xlim(-5,40)


df["capacite"] = df.capacite.sort_values(ascending = False)

df["capacite"]





binsVal = np.arange(0,600,10)

binsVal





plt.hist(df["capacite"], bins = binsVal)





plt.xlabel('Capacite')

plt.ylabel('Frequency')

plt.title('Capacite du Parking')





plt.show()


Gaba= sns.distplot(df['gabarit']);





Gaba.set(xlabel = 'Value', ylabel = 'Frequency', title = "Gabarit Total")





sns.despine()
sns.jointplot(x ="capacitepmr", y ="parkingtempsreel", data = df)
sns.stripplot(x = "capacitepmr", y = "capacite", hue = "reglementation", data = df, jitter = True);
g = sns.FacetGrid(df, col = "reglementation", hue = "usage")

g.map(plt.scatter, "capacitepmr", "capacite", alpha =1)





g.add_legend();
import folium

import json
colors = {'Gratuit' : 'orange', 'Payant' : 'blue','Zone bleue':'green'}

location=[45.76848099905716,4.837471]

map_osm = folium.Map(location, zoom_start=15)



df.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]], 

                                              radius=10, fill_color=colors[row['reglementation']], popup=row['reglementation'])

                                             .add_to(map_osm), axis=1)



map_osm
df.groupby('reglementation').size()


Parking = df.groupby('nom').sum()





Parking = Parking.sort_values(by = 'capacite',ascending = False)[0:10]





Parking['capacite'].plot(kind='bar')





plt.xlabel('Parking')

plt.ylabel('Capacite')

plt.title('10 Parking with more orders')





plt.show()
df.plot(kind="scatter", x="longitude", y="latitude")

df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

            s=df["capacite"]/100, label="capacite", figsize=(10,7),

            c="capacitepmr", cmap=plt.get_cmap("jet"), colorbar=True,

            )



plt.legend()

df.isnull().sum()
sns.set(style = 'whitegrid', context = 'notebook')

cols = ["parkingtempsreel", "gabarit", "capacite", "capacite2rm", "capacitevelo", "capaciteautopartage", "capacitepmr"]

sns.pairplot(df[cols], height = 2.5)

plt.show()
data2 = df[["parkingtempsreel", "gabarit", "capacite", "capacite2rm", "capacitevelo", "capaciteautopartage", "capacitepmr"]]

sns.set(rc={'figure.figsize':(12,9)})



corr = data2.corr(method="pearson")

sns.heatmap(corr)


data2 = df[["parkingtempsreel", "gabarit", "capacite", "capacite2rm", "capacitevelo", "capaciteautopartage", "capacitepmr"]]



sns.set(rc={'figure.figsize':(12,9)})



correlation_matrix = np.corrcoef(data2.values.T)



hm = sns.heatmap(data = correlation_matrix,

                 annot = True,

                 square = True,

                 fmt='.2f', 

                 yticklabels=data2.columns,

                 xticklabels=data2.columns)



plt.figure(figsize=(20,5))



features = ['capacitepmr', 'capaciteautopartage']



for i, col in enumerate(features):

    plt.subplot(1, len(features), i + 1)

    plt.scatter(df[col], df['capacite'])

    plt.title(col)

    plt.xlabel(col)

    plt.ylabel('capacite')
class LinearRegressionGD(object):

    

    def __init__(self, alpha=0.001, n_iter=20):

        self.alpha = alpha

        self.n_iter = n_iter

    

    def fit(self, X, y):

        self.beta_ = np.zeros(1 + X.shape[1])

        self.cost_ = [] 

        

        for i in range(self.n_iter):

            output = self.net_input(X)

            errors = (y - output)

            self.beta_[1:] += self.alpha * X.T.dot(errors)

            self.beta_[0]  += self.alpha * errors.sum()

            cost = (errors ** 2).sum() / 2.0

            self.cost_.append(cost)

        return self

    

    def net_input(self, X):

        return np.dot(X, self.beta_[1:]) + self.beta_[0]

    

    def predict(self, X):

        return self.net_input(X)
X = df[['capacitepmr']].values

y = df['capacite'].values



from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

sc_y = StandardScaler()



X_std = sc_x.fit_transform(X)

y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr_model = LinearRegressionGD()

lr_model.fit(X_std, y_std)


plt.figure(figsize=(4,3))

plt.plot(range(1, lr_model.n_iter+1), lr_model.cost_)

plt.ylabel('SSE')

plt.xlabel('Epoch')

plt.tight_layout()
def lin_regplot(X, y, model, xlabel='', ylabel=''):

    plt.scatter(X, y, c='blue')

    plt.plot(X, model.predict(X), color='red')

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    return None



plt.figure(figsize=(5,3))

lin_regplot(X_std, y_std, lr_model, xlabel = 'capacitepmr (standardized)', ylabel = 'capacite (standardized)')
num_space_std = sc_x.transform(np.array([[5.0]]))



space_std = lr_model.predict(num_space_std) 



print("Space in 5 square: %.3f" % sc_y.inverse_transform(space_std))
print('Slope: %.3f' %lr_model.beta_[1])

print('Intercept: %.3f' %lr_model.beta_[0])
from sklearn.linear_model import LinearRegression



lmodel = LinearRegression()

lmodel.fit(X, y)
print('Slope: %.3f' %lmodel.coef_[0], '\nIntercept: %.3f' % lmodel.intercept_)
plt.figure(figsize=(5,3))

lin_regplot(X, y, lmodel, 'capacitepmr', 'capacite')
from sklearn.linear_model import RANSACRegressor



ransac = RANSACRegressor(base_estimator=LinearRegression(), 

                         max_trials=100, 

                         min_samples=50, 

                         residual_threshold=5.0, 

                         random_state=10)

ransac.fit(X,y)
inlier_mask = ransac.inlier_mask_

outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3,10, 1)

line_y_ransac = ransac.predict(line_X[:, np.newaxis])



plt.figure(figsize=(10,7))

plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', label = 'Inliers')

plt.scatter(X[outlier_mask], y[outlier_mask], c='green', label='Outliers', marker='s')

plt.plot(line_X, line_y_ransac, color='red')

plt.xlabel('capacitepmr')

plt.ylabel('capacite')

plt.legend(loc='lower right')



print('Slope: %.3f' %ransac.estimator_.coef_[0], '\nIntercept: %.3f' %ransac.estimator_.intercept_)
from sklearn.model_selection import train_test_split



X = pd.DataFrame(np.c_[df['capaciteautopartage'], df['capacitepmr']], columns = ['capaciteautopartage','capacitepmr'])

Y = df['capacite']



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, random_state = 10)



print("X_train = ", X_train.shape, "X_test = ", X_test.shape, 

      "Y_train = ", Y_train.shape, "Y_test=",   Y_test.shape)
lmodel = LinearRegression()

lmodel.fit(X_train, Y_train)
y_train_predicted = lmodel.predict(X_train)

y_test_predicted  = lmodel.predict(X_test)
plt.figure(figsize=(6,5))

plt.scatter(y_train_predicted, y_train_predicted - Y_train, c='blue', label='Training data')

plt.scatter(y_test_predicted, y_test_predicted - Y_test, c='green', marker='s', label='Test data')

plt.xlabel('Predicted values')

plt.ylabel('Residuals')

plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')

plt.xlim([-10,50])

plt.ylim([-30,20])

plt.legend(loc='upper left')
from sklearn.metrics import mean_squared_error, r2_score



mse_train = mean_squared_error(Y_train, y_train_predicted)

mse_test = mean_squared_error(Y_test, y_test_predicted)



print('MSE train: %.3f, test: %.3f' %(mse_train, mse_test))

print('RMSE train: %.3f, RMSE test: %.3f' % (np.sqrt(mse_train), np.sqrt(mse_test)))



print('R2 score train: %.3f, R2 score test: %.3f' %(r2_score(Y_train, y_train_predicted), 

                                                    r2_score(Y_test, y_test_predicted)))