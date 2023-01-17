import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import r2_score
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head(2)
df.info()
def display_missing_val(df):
    missing_val = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_val / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([missing_val, missing_percent], axis=1, keys=['Total','Percent'])
    display(missing_data)
display_missing_val(df)
plt.figure(figsize=(10,8))

sns.scatterplot(data=df, x='room_type', y='price')

plt.xlabel("Room Type", size=13)
plt.ylabel("Price", size=13)
plt.title("Room Type vs Price",size=15, weight='bold')
plt.figure(figsize=(10,8))

sns.scatterplot(
    data=df,
    x="room_type", y="price",
    hue="neighbourhood_group", size="neighbourhood_group",
    sizes=(50, 100), palette="Spectral"
)

plt.xlabel("Room Type", size=13)
plt.ylabel("Price", size=13)
plt.title("Room Type vs Price vs Neighbourhood Group",size=15, weight='bold')
plt.figure(figsize=(10,8))

neighbourhood_group = df.neighbourhood_group.unique()

sns.set_palette("Spectral")

for neighbourhood in neighbourhood_group:
    sns.lineplot(
        data= df[df.neighbourhood_group == neighbourhood],
        x='price', y='number_of_reviews', 
        label = neighbourhood
    )


plt.xlabel("Price", size=13)
plt.ylabel("Number of Reviews", size=13)
plt.title("Price vs Number of Reviews vs Neighbourhood Group",size=15, weight='bold')
plt.figure(figsize=(10,8))

sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood_group)

plt.ioff()
plt.figure(figsize=(10,8))

sns.distplot(df.price, fit=norm)

plt.title("Price Distribution Plot",size=15, weight='bold')
# +1 because division by zero caused problem
df.loc[:, ['log_price']] = np.log1p(df.price)
plt.figure(figsize=(10,8))

sns.distplot(df.log_price, fit=norm)

plt.title("Log-Price Distribution Plot",size=15, weight='bold')
plt.figure(figsize=(10,8))

stats.probplot(df.log_price, plot=plt)

plt.show()
categorical = ['neighbourhood_group','neighbourhood','room_type']

for cat in categorical:
    df.loc[:, [cat]] = df[cat].astype("category").cat.codes
df.head(1)
selected_df = df.copy().drop(columns=['id', 'name','host_id','host_name','last_review','price'])
selected_df.info()
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

selected_df.loc[:, ['reviews_per_month']] = imp.fit_transform(selected_df[['number_of_reviews']]).ravel()
display_missing_val(selected_df)
plt.figure(figsize=(15,10))

palette = sns.diverging_palette(240, 10, n=9)

corr = selected_df.corr(method='pearson')

sns.heatmap(corr, annot=True,
            fmt=".2f", cmap=palette,
            vmax=.3, center=0,
            square=True, linewidths=.5,
            cbar_kws={"shrink": .5}).set(ylim=(11, 0)
            )

plt.title("Correlation Matrix",size=15, weight='bold')
residual_plot = selected_df.copy()

residual_target = residual_plot.pop('log_price')
color = {
    0 : ['red','orange'],
    1 : ['yellow','green'],
    2 : ['blue','slateblue'],
    3 : ['purple','deeppink'],
    4 : ['pink','silver'],
    5 : ['grey','black'],
}

def plot_axes(row,col):
    if col < 1:
        return row, col + 1, color[row][col]
    return row + 1, 0 , color[row][col]
f, axes = plt.subplots(5, 2, figsize=(15, 20))

rows, cols = 0, 0
ax_color = color[rows][cols]

for col in residual_plot.columns:
    sns.residplot(
        residual_plot[col], residual_target,
        ax=axes[rows, cols], color=ax_color,
        lowess=True, 
        scatter_kws={'alpha': 0.5}, 
        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}
    )
    
    rows, cols, ax_color = plot_axes(rows, cols)
    
    
plt.setp(axes, yticks=[])
plt.tight_layout()
residual_plot = StandardScaler().fit_transform(residual_plot)
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(residual_plot, residual_target, test_size=0.8, random_state=21)
le = LabelEncoder()

feature_model = ExtraTreesClassifier(n_estimators=50)
feature_model.fit(X_train_res,le.fit_transform(y_train_res))
plt.figure(figsize=(7,7))

feat_importances = pd.Series(feature_model.feature_importances_, index=selected_df.iloc[:,:-1].columns)
feat_importances.nlargest(10).plot(kind='barh')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(residual_plot, residual_target, test_size=0.2, random_state=42)
def make_model_pipeline(model, tuned_parameters):
    return GridSearchCV(model, tuned_parameters, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
model = LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

lin_reg_model = make_model_pipeline(model,parameters)
lin_reg_model.fit(X_train,y_train)
lin_reg_model.best_score_ 
lin_reg_pred = lin_reg_model.predict(X_test)
model = Ridge()
parameters = {'alpha':[1, 0.1, 0.01, 0.001, 0.0001, 0], 'normalize':[True,False]}

ridge_model = make_model_pipeline(model,parameters)
ridge_model.fit(X_train,y_train)
ridge_model.best_score_
ridge_pred = ridge_model.predict(X_test)
model = Lasso()
parameters = {'alpha':[1, 0.1, 0.01, 0.001, 0.0001], 'normalize':[True,False]}

lasso_model = make_model_pipeline(model,parameters)
lasso_model.fit(X_train,y_train)
lasso_model.best_score_
lasso_pred = lasso_model.predict(X_test)
model = ElasticNet()
parameters = {'alpha':[1, 0.1, 0.01, 0.001, 0.0001], 'normalize':[True,False]}

elastic_net_model = make_model_pipeline(model,parameters)
elastic_net_model.fit(X_train,y_train)
elastic_net_model.best_score_
elastic_net_pred = elastic_net_model.predict(X_test)
models = [('Linear Regression', lin_reg_pred),
          ('Ridge', ridge_pred),
          ('Lasso', lasso_pred),
          ('Elastic Net', elastic_net_pred)]

for model in models:
    
    print(f'-------------{model[0]}-----------\n')
    
    print('MAE: %f'% mean_absolute_error(y_test, model[1]))
    print('RMSE: %f'% np.sqrt(mean_squared_error(y_test, model[1])))   
    print('R2 %f\n' % r2_score(y_test, model[1]))
