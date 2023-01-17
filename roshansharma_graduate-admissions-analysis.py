# for some basic operations

import numpy as np

import pandas as pd



# for data visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# for advanced visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff



# for providing path

import os

print(os.listdir('../input/'))
# reading the data



data = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

data2 = pd.read_csv('../input/Admission_Predict.csv')



# getting the shapes of the datasets

print("Shape of data1: ", data.shape)

print("Shape of data2 :", data2.shape)
# combining both the datasets as they have same columns



data = pd.concat([data, data2])



# getting the shape of new dataset

data.head()
# describing the dataset



data.describe()
# checking if the data contains any NULL values



data.isnull().any().any()
# Average GRE Score of the students



gre = data['GRE Score'].mean()

toefl = data['TOEFL Score'].mean()

cgpa = data['CGPA'].mean()



# sop - statement of purpose

# lor - letter of recommendation



sop = np.round(data['SOP'].mean())

lor = np.round(data['LOR '].mean())



research = np.round(data['Research'].mean())

uni_rating = np.round(data['University Rating'].mean())



# printing the results



print("The Average Score for GRE is {:.2f}".format(gre))

print("The Average Score for TOEFL is {:.2f}".format(toefl))

print("The Average CGPA is {:.2f}".format(cgpa))

print("The Average Number for Statement of Purpose is", sop)

print("The Average Number for Recommendation letters among the students is", lor)

print("The Average Number of Research done by students is ", research)

print("The Average University Ratings of Different Students is ", uni_rating)
# Average GRE Score of the students



gre = data['GRE Score'].min()

toefl = data['TOEFL Score'].min()

cgpa = data['CGPA'].min()



# sop - statement of purpose

# lor - letter of recommendation



sop = np.round(data['SOP'].min())

lor = np.round(data['LOR '].min())



research = np.round(data['Research'].min())

uni_rating = np.round(data['University Rating'].min())



# printing the results



print("The Minimum Score for GRE is {:.2f}".format(gre))

print("The Minimum Score for TOEFL is {:.2f}".format(toefl))

print("The Minimum CGPA is {:.2f}".format(cgpa))

print("The Minimum Number for Statement of Purpose is", sop)

print("The Minimum Number for Recommendation letters among the students is", lor)

print("The Minimum Number of Research done by students is ", research)

print("The Minimum University Ratings of Different Students is ", uni_rating)
# Average GRE Score of the students



gre = data['GRE Score'].max()

toefl = data['TOEFL Score'].max()

cgpa = data['CGPA'].max()



# sop - statement of purpose

# lor - letter of recommendation



sop = np.round(data['SOP'].max())

lor = np.round(data['LOR '].max())



research = np.round(data['Research'].max())

uni_rating = np.round(data['University Rating'].max())



# printing the results



print("The Maximum Score for GRE is {:.2f}".format(gre))

print("The Maximum Score for TOEFL is {:.2f}".format(toefl))

print("The Maximum CGPA is {:.2f}".format(cgpa))

print("The Maximum Number for Statement of Purpose is", sop)

print("The Maximum Number for Recommendation letters among the students is", lor)

print("The Maximum Number of Research done by students is ", research)

print("The Maximum University Ratings of Different Students is ", uni_rating)
# looking at the variations of gre score among the students



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (18, 7)

plt.style.use('_classic_test')



sns.distplot(data['GRE Score'], color = 'blue')

plt.title('Variations in GRE Score', fontsize = 20)

plt.xlabel('GRE Score')

plt.ylabel('count')

plt.show()
# looking at the variations of CGPA among the students



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (18, 7)

plt.style.use('_classic_test')



sns.distplot(data['CGPA'], color = 'violet')

plt.title('Variations in CGPA', fontsize = 20)

plt.xlabel('CGPA Score')

plt.ylabel('count')

plt.show()
# looking at the variations of toefl score among the students



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (18, 7)

plt.style.use('_classic_test')



sns.distplot(data['TOEFL Score'], color = 'red')

plt.title('Variations in TOEFL Score', fontsize = 30)

plt.xlabel('TOEFL Score')

plt.ylabel('count')

plt.show()
# looking at the variations of LOR among the students



plt.rcParams['figure.figsize'] = (18, 9)

plt.style.use('dark_background')



sns.countplot(data['LOR '], palette = 'PuBu')

plt.title('Variations in Letter of Recommendations', fontsize = 30)

plt.xlabel('LOR Score')

plt.ylabel('count')

plt.show()
# looking at the variations of SOP among the students



plt.rcParams['figure.figsize'] = (18, 9)

plt.style.use('dark_background')



sns.countplot(data['SOP'], palette = 'Wistia')

plt.title('Variations of SOP among the students', fontsize = 30)

plt.xlabel('SOP Score')

plt.ylabel('count')

plt.show()
# making a pie chart for the analysis of students rather they did research or not.



data_re = data['Research'].value_counts()



label_re = data_re.index

size_re = data_re.values



colors = ['aqua', 'gold']



trace = go.Pie(

         labels = label_re, values = size_re, marker = dict(colors = colors), name = 'Research', hole = 0.3)



df = [trace]



layout1 = go.Layout(

           title = 'Research work done or not')

fig = go.Figure(data = df, layout = layout1)

py.iplot(fig)
# making a donut chart for the analysis of students with different university ratings



data_ur = data['University Rating'].value_counts()



label_re = data_ur.index

size_re = data_ur.values





trace = go.Pie(

         labels = label_re,

         values = size_re,

         marker = dict(colors = ['gold' 'lightgreen', 'orange', 'yellow', 'pink']),

         name = 'University Ratings',

         hole = 0.2)



df2 = [trace]



layout1 = go.Layout(

           title = 'University Ratings of the Students')

fig = go.Figure(data = df2, layout = layout1)

py.iplot(fig)
# Analyzing the Chances of Admission



plt.rcParams['figure.figsize'] = (20, 10)

plt.style.use('fivethirtyeight')



stemlines = plt.stem(data['Chance of Admit '])

plt.setp(stemlines, color = 'violet', linewidth = 0.3)

plt.title('Chances of Students According to RowID', fontsize = 30)

plt.xlabel('Students in serial Order', fontsize = 15)

plt.ylabel('Chances of a student to get an Admission', fontsize = 15)

plt.show()


trace = go.Box(

            x = data['University Rating'],

            y = data['GRE Score'],

            name = 'University Rating vs GRE Score',

            marker = dict(

                  color = 'rgb(145, 165, 5)')

)

                   



df= [trace]



layout = go.Layout(

    boxmode = 'group',

    title = 'University Ratings vs GRE Score',

    

)



fig = go.Figure(data = df, layout = layout)

py.iplot(fig)
plt.rcParams['figure.figsize'] = (18, 9)

plt.style.use('ggplot')



sns.boxenplot(data['University Rating'], data['TOEFL Score'], palette = 'RdPu')

plt.title('University Ratings vs TOEFL Score', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (18, 9)

plt.style.use('ggplot')



sns.swarmplot(data['University Rating'], data['CGPA'], palette = 'twilight')

plt.title('University Ratings vs CGPA', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (18, 9)

plt.style.use('ggplot')



sns.violinplot(data['University Rating'], data['Chance of Admit '], palette = 'copper')

plt.title('University Ratings vs Chance of Admission', fontsize = 20)

plt.show()


# prepare data



data2 = data.loc[:,["GRE Score", "TOEFL Score", "Chance of Admit "]]

data2["index"] = np.arange(1,len(data2)+1)



# scatter matrix

fig = ff.create_scatterplotmatrix(data2, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
# create trace 1 that is 3d scatter



trace1 = go.Scatter3d(

    x = data['GRE Score'],

    y = data['TOEFL Score'],

    z = data['Chance of Admit '],

    mode = 'markers',

    marker=dict(

        size = 8,

        color = 'gold',

        opacity = 0.5    

    )

)



df3 = [trace1]



layout = go.Layout(

    title = 'Character vs Gender vs Alive or not',

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=df3, layout=layout)

iplot(fig)
# plotting a pair plot to see the correlations



plt.rcParams['figure.figsize'] = (20, 21)

plt.style.use('ggplot')



sns.pairplot(data, hue = 'Chance of Admit ', palette = 'husl')

plt.title('Pair plot for the data', fontsize = 20)

plt.show()
# plotting a heat map



# heatmap between GRE Score and Chance of Admission

plt.rcParams['figure.figsize'] = (20, 30)

plt.style.use('ggplot')



# let's remove the serial no. from the data

data = data.drop(['Serial No.'], axis = 1)



sns.heatmap(data.corr(), cmap = 'spring')

plt.title('Correlation Plot', fontsize = 20)

plt.show()
# splitting the data into dependent and independent datasets



x = data.iloc[:, :-1]

y = data.iloc[:, -1]



# getting the shapes of x and y

print("Shape of x: ", x.shape)

print("Shape of y: ", y.shape)
# splitting into train and test



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



print("Shape of x_train :", x_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of y_test :", y_test.shape)
# standard Scaling



from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
# plotting the dist plot of admissions with normal curve



from scipy import stats

from scipy.stats import norm



plt.rcParams['figure.figsize'] = (10, 7)

sns.distplot(data['Chance of Admit '], fit = norm)



# getting the mu, and sigma related to the curve

mu, sigma = norm.fit(data['Chance of Admit '])

plt.legend(["mu {:.2f} and sigma {:.2f}".format(mu, sigma)], loc = 2)



plt.title('Distribution of Chances of Admissions', fontsize = 20)

plt.show()
# plotting the QQ Plot



from scipy import stats



plt.rcParams['figure.figsize'] = (7, 5)

stats.probplot(data['Chance of Admit '], plot = plt)

plt.show()
from sklearn.linear_model import LinearRegression



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



linreg = LinearRegression()

linreg.fit(x_train, y_train)



linreg_pred = linreg.predict(x_test)



mse = mean_squared_error(y_test, linreg_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, linreg_pred)



print("Root Mean Squared Error : ",rmse)

print("R-Squared Error:", r2)

from sklearn.svm import SVR



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



svr = SVR(kernel = 'linear')

svr.fit(x_train, y_train)



svr_pred = svr.predict(x_test)



mse = mean_squared_error(y_test, svr_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, svr_pred)



print("Root Mean Squared Error : ",rmse)

print("R-Squared Error:", r2)
from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



rfr = RandomForestRegressor()

rfr.fit(x_train, y_train)



rfr_pred = linreg.predict(x_test)



mse = mean_squared_error(y_test, rfr_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, rfr_pred)



print("Root Mean Squared Error : ",rmse)

print("R-Squared Error:", r2)
from xgboost.sklearn import XGBRegressor



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



xgb = XGBRegressor()

xgb.fit(x_train, y_train)



xgb_pred = xgb.predict(x_test)



mse = mean_squared_error(y_test, xgb_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, xgb_pred)



print("Root Mean Squared Error : ",rmse)

print("R-Squared Error:", r2)
from sklearn.ensemble import ExtraTreesRegressor



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



etr = ExtraTreesRegressor()

etr.fit(x_train, y_train)



etr_pred = etr.predict(x_test)



mse = mean_squared_error(y_test, etr_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, etr_pred)



print("Root Mean Squared Error : ",rmse)

print("R-Squared Error:", r2)
# stacked predictions



stacked_predictions = np.column_stack((linreg_pred, svr_pred, rfr_pred))



# specifying the meta model

meta_model = LinearRegression()

meta_model.fit(stacked_predictions, y_test)



# final predictions

stacked_predictions = (meta_model.predict(stacked_predictions))



mse = mean_squared_error(y_test, stacked_predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, stacked_predictions)



print("Root Mean Squared Error : ",rmse)

print("R-Squared Error:", r2)

Boosted_predictions = stacked_predictions*0.2 + xgb_pred*0.3 + etr_pred*0.5



mse = mean_squared_error(y_test, Boosted_predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, Boosted_predictions)



print("Root Mean Squared Error : ",rmse)

print("R-Squared Error:", r2)