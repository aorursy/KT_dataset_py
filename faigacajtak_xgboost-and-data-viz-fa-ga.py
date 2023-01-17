import pandas as pd

from pandas.plotting import scatter_matrix



import itertools



import numpy as np 

from numpy import loadtxt



from scipy.stats import spearmanr



from scipy import stats

from scipy.stats import boxcox

from scipy.stats import skew



import category_encoders as ce



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



import missingno as msno



from sklearn import neighbors

from sklearn import linear_model



from sklearn import model_selection, preprocessing



from sklearn.preprocessing import StandardScaler, Imputer



from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate



from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, mutual_info_regression



from sklearn import model_selection, preprocessing

from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

import xgboost

from sklearn.preprocessing import PolynomialFeatures



from bokeh.plotting import figure

from bokeh.io import output_notebook, show

from bokeh.models.tools import HoverTool

from bokeh.transform import factor_cmap

from bokeh.palettes import Viridis

from bokeh.models import ColumnDataSource

output_notebook()



%matplotlib inline
df = pd.read_csv("../input/train.csv", nrows = 1_000_000)
def first_info(data):

    '''

    # Affiche un résumé complet du dataset

    '''

    print(data.head())

    print(data.info())

    print(data.describe())

    

first_info(df)
#************************************************************************************

#                                                                                   *

#                               Nettoyer les données                                *

#                                                                                   *

#*************************************************************************************
#************************** TRAITEMENT DES VALEURS MANQUANTES ***********************



def count_nan_values (data):

    '''

    # Affiche un tableau des valeurs manquantes avec les pourcentages

    '''

    total = data.isnull().sum().sort_values(ascending=False)

    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    print(missing_data.head(20))

    

print(count_nan_values(df))
def haversine(lon1, lat1, lon2, lat2, earth_radius=6367):

    """

    Calculer les kilomètres parcourus en fonction des longétudes et lattitudes

    Tous les arguments doivent être de même longueur.

    

    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2



    c = 2 * np.arcsin(np.sqrt(a))

    km = earth_radius * c

    return km



df['distance'] = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude','dropoff_latitude']].apply(lambda x: haversine(x[1], x[0], x[3], x[2]), axis=1)
print(df.distance.head())
df_test = df[df.distance == 0]

len(df_test)
x = df_test.distance

y = df_test.fare_amount

hover = HoverTool(tooltips=[("(distance,prix)", "(@x, @y)")])



p = figure(plot_width=600, plot_height=400)

p.circle(x,y, size=3, color="navy", alpha=0.5)

p.add_tools(hover)

show(p)
#Nous remplaçons les observations égales à 0 dans les coordonnées en valeurs manquantes.

coord = ['pickup_longitude','pickup_latitude', 

         'dropoff_longitude', 'dropoff_latitude']



for i in coord :

    df[i] = df[i].replace(0,np.nan)

    

print("")    

print("En tenant compte des zéros dans les coordonnées")

print("***********************************************")

print(count_nan_values(df))
msno.matrix(df)

msno.heatmap(df);
df.dropna(inplace=True)

count_nan_values(df)
#************************************* TRAITEMENT OUTLIERS **************************
plt.figure(figsize=(10,10))

sns.scatterplot(x="distance", y="fare_amount", data=df)

plt.show()
def delete_outlier(data):

    return data[(data.fare_amount > 0) & 

            (df.pickup_longitude > -78) & (df.pickup_longitude < -68) &

            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &

            (df.dropoff_longitude > -78) & (df.dropoff_longitude < -68) &

            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) &

            (df.passenger_count > 0) & (df.passenger_count < 8)]



df_out = delete_outlier(df)



print("Nombre des observations supprimées:",len(df) - len(df_out))
plt.figure(figsize=(12,6))

sns.distplot(df_out.fare_amount, bins=100);

plt.xlim(0,100)

plt.title("Histogramme du prix");
plt.figure(figsize=(10,10))

sns.scatterplot(x="distance", y="fare_amount", data=df_out)

plt.show()
#************************************************************************************

#                                                                                   *

#                               Data Visualization                                  *

#                                                                                   *

#************************************************************************************
#*********************************** Times series  **********************************
def extract_date (dataset):

    '''

    # Transformation de la variable date en datetime et extraction des années, mois, jours

    '''

    dataset.loc[:,'pickup_datetime'] = pd.to_datetime(dataset.loc[:,'pickup_datetime'])

    

    dataset['month'] = pd.DatetimeIndex(dataset['pickup_datetime']).month

    dataset['month_name'] = dataset['month'].map({1:"Janvier",2:"Fevrier",3:"Mars",4:"Avril",

                                                 5:"Mai",6:"Juin",7:"Juillet",8:"Aout",

                                                 9:"Septembre",10:"Octobre",11:"Novembre",12:"Decembre"})

    

    dataset['year']= pd.DatetimeIndex(dataset['pickup_datetime']).year

    dataset["month_year"] = dataset["year"].astype(str) + " - " + dataset["month_name"]

    

    dataset['day']=pd.DatetimeIndex(dataset['pickup_datetime']).weekday

    dataset["day_name"] = dataset["day"].replace([0,1,2,3,4,5,6],

                                            ["Lundi","Mardi","Mercredi","Jeudi",

                                             "Vendredi","Samedi","Dimanche"])

    

    dataset['hour']=pd.DatetimeIndex(dataset['pickup_datetime']).hour

    

    dataset = dataset.sort_values(by = "pickup_datetime", ascending = False)



extract_date(df_out)
grouped = df_out.groupby('day_name')['distance', 'year'].count()



source = ColumnDataSource(grouped)

day = source.data['day_name'].tolist()



d = figure(x_range=day)

pal = Viridis[7]

color_map = factor_cmap(field_name='day_name',palette=pal, factors=day)



d.vbar(x='day_name', top='year', source=source, width=0.70, color=color_map)

d.add_tools(HoverTool(tooltips= [("Total", "@distance")]))



d.title.text ='Nombre de course selon les jours'

d.xaxis.axis_label = 'Jour de la semaine'

d.yaxis.axis_label = 'Nombre de course'



show(d)
plt.figure(figsize=(15,7))

sns.lineplot(x="hour", y="fare_amount", data=df_out, hue='day_name', 

             hue_order=["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"], 

             ci=None, marker="o", palette=pal);
trips_hr = df_out["hour"].value_counts().reset_index()

trips_hr.columns = ["hour","count"]

trips_hr = trips_hr.sort_values(by = "hour",ascending = True)

x=trips_hr['hour']

y=trips_hr['count']



# Instanciation de la figure

p = figure(plot_width= 800, plot_height=500)

p.line(x, y, line_color="black")



r1 = p.circle(x, y, legend='count')

r1.glyph.size=10

r1.glyph.fill_alpha=0.2



r2 = p.circle(x,y, size=20, hover_color = 'navy', hover_alpha=0.4, line_color=None, 

              line_width=0, fill_alpha=0.05, legend='count', fill_color='navy')



p.add_tools(HoverTool(tooltips= [("index", "$index"), ("(count)", "($y)")], renderers=[r2]))



# Changement des labels de l'axe 

p.xaxis.axis_label = "Heure de la journée"

p.yaxis.axis_label = "Nombre de course"



# Changements de la couleur des axes

p.xaxis.major_label_text_color = "navy"

p.yaxis.major_label_text_color = "navy"



# Changements sur la grille

p.xgrid.grid_line_color = None

p.ygrid.band_fill_alpha = 0.05

p.ygrid.band_fill_color = "silver"



show(p)
trip_count = df_out.groupby(["year","month","month_name"])["month_year"].value_counts().to_frame()

trip_count.columns = ["count"]

trip_count = trip_count.reset_index()

xi=trip_count['month_year']

yi=trip_count['count']



fig, ax = plt.subplots(figsize=(40, 20))

sns.lineplot(xi, yi, marker="o", palette=pal, color="navy")



ax.set(xlabel="Date", ylabel="Value", )

ax.set_xticklabels(labels=trip_count["month_year"], rotation=45);
df_out = df_out[df_out.year != 2015]
plt.figure(figsize=(10,10))

sns.countplot(x="day_name", data=df_out, 

              order=["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"], hue="year", palette=pal )

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x="passenger_count", data=df_out, palette=pal)

plt.title("Distribution du nombre de passager");
arr_hist2, edges2 = np.histogram(df_out['fare_amount'], 

                               bins = int(64.5/2), 

                               range = [0, 100])

prix_course = pd.DataFrame({'prix': arr_hist2, 

                       'left': edges2[:-1], 

                       'right': edges2[1:]})



prix_course['f_interval'] = ['%d to %d $' % (left, right) for left, right in zip(prix_course['left'], prix_course['right'])]

source_prix = ColumnDataSource(prix_course)



e = figure(plot_height = 800, plot_width = 800, title = 'Histogramme du nombre de course selon les prix', 

           x_axis_label = 'Prix ($)]', 

           y_axis_label = 'Nombre de course')



e.quad(bottom=0, top='prix', left='left', right='right', source=source_prix, fill_color='lightgray', 

       line_color='dimgray', fill_alpha = 0.75, hover_fill_alpha = 1.0, hover_fill_color = 'olive')



e.xaxis.major_label_text_color = "olive"

e.yaxis.major_label_text_color = "olive"



e.add_tools(HoverTool(tooltips = [('Interval', '@f_interval'),('(prix,nombre)', '($x, $y)')]))

show(e)
arr_hist, edges = np.histogram(df_out['distance'], 

                               bins = int(18/0.2), 

                               range = [0, 18])

course = pd.DataFrame({'dist_course': arr_hist, 

                       'left': edges[:-1], 

                       'right': edges[1:]})



course['f_interval'] = ['%d to %d km' % (left, right) for left, right in zip(course['left'], course['right'])]

src = ColumnDataSource(course)



h = figure(plot_height = 800, plot_width = 800, title = 'Histogramme du nombre de course selon la distance', 

           x_axis_label = 'Distance (km)]', 

           y_axis_label = 'Nombre de course')



h.quad(bottom=0, top='dist_course', left='left', right='right', source=src, fill_color='aliceblue', 

       line_color='navy', fill_alpha = 0.75, hover_fill_alpha = 1.0, hover_fill_color = 'mediumseagreen')



h.add_tools(HoverTool(tooltips = [('Interval', '@f_interval'),('(distance,nombre)', '($x, $y)')]))

show(h)
#************************************************************************************

#                                                                                   *

#                               Sélection des variables                             *

#                                                                                   *

#************************************************************************************
def list_matrix (data, var='var'):

    '''

        # Affiche une liste décroissante des corrélations avec la variable cible. 

        '''

    corr_matrix = data.corr(method='pearson')

    print(corr_matrix[var].sort_values(ascending=False))



list_matrix(df_out, var='fare_amount')
def corr_matrix(data):

    '''

        # Affiche une matrice de corrélation de toutes les variables

        '''

    corr_data=data.corr(method='pearson')

    sns.set(style="white")

    

    mask = np.zeros_like(corr_data, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    

    f, ax = plt.subplots(figsize=(40, 40))

    

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    

    sns.heatmap(corr_data, mask=mask, cmap=cmap, annot=True, vmax=.3, center=0,

                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    

corr_matrix(df_out)
# Suppression des variables inutiles 

df_fare = df_out.drop(["pickup_longitude", "pickup_latitude",

                       "dropoff_longitude", "dropoff_latitude",

                        "pickup_datetime", "key"], axis=1)
def plot_correlation_test (cols, colors, data, nrows=3, ncols=2, var="var"):

    plt.rcParams['figure.figsize'] = [25, 15]



    fig, ax = plt.subplots(nrows, ncols)



    ax=ax.flatten()



    j=0



    for i in ax:

        if j==0:

            i.set_ylabel(var)

        i.scatter(data[cols[j]], data[var],  alpha=0.5, color=colors[j])

        i.set_xlabel(cols[j])

        i.set_title('Pearson: %s'%data.corr().loc[cols[j]][var].round(2)+' Spearman: %s'%data.corr(method='spearman').loc[cols[j]][var].round(2))

        j+=1



    plt.show()

    

cols = ['distance', 'day', 'month','year',"passenger_count","hour"]

colors=['#415952', '#f35134', '#243AB5', '#243AB5', 'olive', 'salmon']



plot_correlation_test(cols, colors, df_fare, nrows=3, ncols=2, var="fare_amount")
data = df_fare.drop(["fare_amount", "month_name", "month_year", "day_name"], axis=1)

target = df_fare.fare_amount



X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
feature_names = list(df_fare.columns.values)

sel = SelectKBest(score_func=f_regression, k=4)

sel.fit(X_train, y_train)

df_new = sel.transform(X_train)

mask = sel.get_support()

vector_names = list(X_train.columns[sel.get_support()])

print(vector_names)



plt.matshow(mask.reshape(1,-1), cmap = 'gray_r')

plt.xlabel('Axe des features');

plt.show();
#************************************************************************************

#                                                                                   *

#                                Construction des modèles                           *

#                                                                                   *

#************************************************************************************
def scale_data (data):

    '''

        # Standardisation des données avec le score Z

        '''

    data.copy()

    scaler = preprocessing.StandardScaler().fit(data)

    data_scaled = scaler.transform(data)

    columns = data.columns

    data_scaled = pd.DataFrame(data=data_scaled, columns=columns)

    

    return (data_scaled)



X_train_scaled = scale_data(X_train)

X_test_scaled = scale_data(X_test)
#*********************************** Regression Linéaire ****************************

lr = LinearRegression()

lr.fit(X_train_scaled, y_train)



def print_coefs (regression, data):

    coeffs = list(regression.coef_)

    coeffs.insert(0, regression.intercept_)



    feats = list(data.columns)

    feats.insert(0, 'intercept')



    print(pd.DataFrame({'valeur estimée': coeffs}, index = feats))



print_coefs(lr, data)

print(lr.score(X_train_scaled, y_train))
def rmse(predictions, targets):

    return print("RMSE:",np.sqrt(((predictions - targets) ** 2).mean()))



pred_test = lr.predict(X_test_scaled)

rmse(pred_test, y_test)
def QQ_plot (regression, X_train, y_train):

    pred_train = regression.predict(X_train)

    residus = pred_train - y_train

    residus_norm = (residus - residus.mean())/residus.std()

    stats.probplot(residus_norm, plot=plt)

    plt.show()



QQ_plot(lr, X_train_scaled, y_train)  
#*********************************** Gradient Boosting ******************************



params = { 'n_estimators': 1000,

          'learning_rate' : 0.01, # Résultat d'une recherche par cadrillage

          'max_depth' : 2, # Résultat d'une recherche par cadrillage

          'loss' : 'ls' # Résultat d'une recherche par cadrillage

         }



model_est = GradientBoostingRegressor(**params, random_state=10)

model_est.fit(X_train_scaled, y_train)



pred_est = model_est.predict(X_test_scaled)
pred_est_train = model_est.predict(X_train_scaled)

rmse(pred_est, y_test)

rmse(pred_est_train, y_train)
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



for i, pred_est in enumerate(model_est.staged_predict(X_test_scaled)):

    test_score[i] = model_est.loss_(y_test, pred_est)

    

y = model_est.train_score_  



p = figure(plot_width= 800, plot_height=500)

p1 = p.line(np.arange(params['n_estimators']) + 1, y, line_color="mediumspringgreen")

p.add_tools(HoverTool(tooltips = [("(train_score)","($y)")], renderers=[p1]))



p2 = p.line(np.arange(params['n_estimators']) + 1, test_score, line_color='greenyellow')

p.add_tools(HoverTool(tooltips = [("(test_score)","($test_score)")], renderers=[p2]))



# Changement des labels de l'axe 

p.xaxis.axis_label = "Boosting iterations"

p.yaxis.axis_label = "Deviance"



# Changements de la couleur des axes

p.xaxis.major_label_text_color = "lightgray"

p.yaxis.major_label_text_color = "lightgray"



# Changements sur la grille

p.xgrid.grid_line_color = None

p.ygrid.band_fill_alpha = 0.05

p.ygrid.band_fill_color = "silver"



show(p)

def plot_features_importances (model, data):

    " Fonction pour afficher les variables les plus importantes du modèle. "

    feature_importance = model.feature_importances_



    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    sorted_idx = np.argsort(feature_importance)

    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 1, 1)

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    plt.yticks(pos, data.columns[sorted_idx])

    plt.xlabel('Relative Importance')

    plt.title('Variable Importance')

    plt.show()



plot_features_importances(model_est, data)
#************************************ XGBoost Regressor *****************************



params_xgb = { 'eta': 0.01, # Résultat d'une recherche par cadrillage 

          'max_depth' : 3, # Résultat d'une recherche par cadrillage 

          'objective' : 'reg:linear'} # Résultat d'une recherche par cadrillage 



model_xgb = xgboost.XGBRegressor(**params_xgb, random_state=27, n_jobs=-1)

model_xgb.fit(X_train_scaled, y_train)

pred_xgb = model_xgb.predict(X_test_scaled)
pred_xgb_train = model_xgb.predict(X_train_scaled)

rmse(pred_xgb, y_test)

rmse(pred_xgb_train, y_train)
plot_features_importances(model_xgb, data)
QQ_plot(model_xgb, X_train_scaled, y_train)  