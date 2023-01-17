# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# Visualisation Credits 



# 1) Janio Martinez Backman

# Distribution of Charges, Age Analysis, Weight Status vs Charges,Obesity and Smoking using plotly



# 2) Dandelion 

# -Distribution of Age using seaborn
import numpy as np

import pandas as pd

import pylab
# Plotly Packages

from plotly import tools

import plotly.plotly as py

import plotly.figure_factory as ff

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
# Matplotlib and Seaborn

import matplotlib.pyplot as plt

import seaborn as sns

from string import ascii_letters
# Statistical Libraries

from scipy.stats import norm

from scipy.stats import skew

from scipy.stats.stats import pearsonr

from scipy import stats
# Regression Modeling

import statsmodels.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std
# Plotly offline is surprisingly inconsistent regarding when iplot works/ does not work, increasing data rate limit 

# hels. If not a permanent and solution would be using the online api with user credentials and links to plots



# jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10





# Path for installing new packages (<the output> -m pip install pandas)

from sys import executable

print(executable)



# Install packages to the python 3 kernel in jupyter notebook

# //anaconda/envs/ipykernel_py3/bin/python -m pip install <package-name>
#df = pd.read_csv("insurance.csv")

df = pd.read_csv("../input/insurance.csv")

df.head()



# Let's store the original dataframe in another variable.

original_df = df.copy()
# Distribution of Medical Charges



# Types of Distributions: We have a right skewed distribution in which most patients are being charged 

# between  2000− 12000. Using Logarithms: Logarithms helps us have a normal distribution which could help us in 

# a number of different ways such as outlier detection



charge_dist = df["charges"].values

logcharge = np.log(df["charges"])



trace0 = go.Histogram(

    x=charge_dist,

    histnorm='probability',

    name="Charges Distribution",

    marker = dict(

        color = '#FA5858',

    )

)

trace1 = go.Histogram(

    x=logcharge,

    histnorm='probability',

    name="Charges Distribution using Log",

    marker = dict(

        color = '#58FA82',

    )

)



fig = tools.make_subplots(rows=2, cols=1,

                          subplot_titles=('Charge Distribution','Log Charge Distribution'),

                         print_grid=False)



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 2, 1)



fig['layout'].update(showlegend=True, title='Charge Distribution', bargap=0.05)

iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
# Age Analysis:



# Turning Age into Categorical Variables:

# Young Adult: from 18 - 35

# Senior Adult: from 36 - 55

# Elder: 56 or older

# Share of each Category: Young Adults (42.9%), Senior Adults (41%) and Elder (16.1%)



df['age_cat'] = np.nan

lst = [df]



for col in lst:

    col.loc[(col['age'] >= 18) & (col['age'] <= 35), 'age_cat'] = 'Young Adult'

    col.loc[(col['age'] > 35) & (col['age'] <= 55), 'age_cat'] = 'Senior Adult'

    col.loc[col['age'] > 55, 'age_cat'] = 'Elder'

    

    

labels = df["age_cat"].unique().tolist()

amount = df["age_cat"].value_counts().tolist()



colors = ["#ff9999", "#b3d9ff", " #e6ffb3"]



trace = go.Pie(labels=labels, values=amount,

               hoverinfo='label+percent', textinfo='value', 

               textfont=dict(size=20),

               marker=dict(colors=colors, 

                           line=dict(color='#000000', width=2)))



data = [trace]

layout = go.Layout(title="Amount by Age Category")



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic_pie_chart')



plt.figure(figsize=(12,5))

plt.title("Distribution of age")

ax = sns.distplot(df["age"], color = 'g')

plt.show()
# Weight Status: 

# https://www.cancer.org/cancer/cancer-causes/diet-physical-activity/body-weight-and-cancer-risk/adult-bmi.html  

# Turning BMI into Categorical Variables:  

# Under Weight: Body Mass Index (BMI)  <  18.5 

# Normal Weight: Body Mass Index (BMI)  ≥  18.5 and Body Mass Index (BMI)  <  24.9 

# Overweight: Body Mass Index (BMI)  ≥  25 and Body Mass Index (BMI)  <  29.9 

# Obese: Body Mass Index (BMI)  >  30



df["weight_condition"] = np.nan

lst = [df]



for col in lst:

    col.loc[col["bmi"] < 18.5, "weight_condition"] = "Underweight"

    col.loc[(col["bmi"] >= 18.5) & (col["bmi"] < 24.986), "weight_condition"] = "Normal Weight"

    col.loc[(col["bmi"] >= 25) & (col["bmi"] < 29.926), "weight_condition"] = "Overweight"

    col.loc[col["bmi"] >= 30, "weight_condition"] = "Obese"

    

df.head()
# Weight Status vs Charges



# Overweight: Notice how there are two groups of people that get significantly charged more than the other group of overweight people.

# Obese: Same thing goes with the obese group, were a significant group is charged more than the other group.



fig = ff.create_facet_grid(

    df,

    x='age',

    y='charges',

    color_name='weight_condition',

    show_boxes=False,

    marker={'size': 10, 'opacity': 1.0},

    colormap={'Underweight': 'rgb(208, 246, 130)', 'Normal Weight': 'rgb(166, 246, 130)',

             'Overweight': 'rgb(251, 232, 238)', 'Obese': 'rgb(253, 45, 28)'}

)

251, 232, 238





fig['layout'].update(title="Weight Status vs Charges", width=800, height=600, plot_bgcolor='rgb(251, 251, 251)', 

                     paper_bgcolor='rgb(255, 255, 255)')





iplot(fig, filename='facet - custom colormap')
# What Percentage of Obese that Smoked Paid aBove Average from the total obese patients?

# 79% of Obese were non-smokers while the 21% left were smokers



total_obese = len(df.loc[df["weight_condition"] == "Obese"])



obese_smoker_prop = len(df.loc[(df["weight_condition"] == "Obese") & (df["smoker"] == "yes")])/total_obese

obese_smoker_prop = round(obese_smoker_prop, 2)



obese_nonsmoker_prop = len(df.loc[(df["weight_condition"] == "Obese") & (df["smoker"] == "no")])/total_obese

obese_nonsmoker_prop = round(obese_nonsmoker_prop, 2)





# Average charge by obese_smokers and obese_nonsmoker

charge_obese_smoker = df.loc[(df["weight_condition"] == "Obese") & (df["smoker"] == "yes")].mean().iloc[3]

charge_obese_nonsmoker = df.loc[(df["weight_condition"] == "Obese") & (df["smoker"] == "no")].mean().iloc[3]





print("The percentage of obese smokers is ",obese_smoker_prop * 100)

print("The average charge for an obese smoker is ", round(charge_obese_smoker,2))

print("************************************************************")

print("The percentage of obese non-smokers is ",obese_nonsmoker_prop * 100)

print("The average charge for an obese non-smoker is ", round(charge_obese_nonsmoker,2))
# Two subplots one with weight condition and the other with smoker.



f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,8))

sns.scatterplot(x="bmi", y="charges", hue="weight_condition", data=df, palette="Set1", ax=ax1)

ax1.set_title("Relationship between Charges and BMI by Weight Condition")

ax1.annotate('Obese Cluster \n (Does this cluster has \n the Smoking Attribute?)', xy=(37, 50000), xytext=(30, 60000),

            arrowprops=dict(facecolor='black'),

            fontsize=12)

sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df, palette="Set1", ax=ax2)

ax2.set_title("Relationship between Charges and BMI by Smoking Condition")

ax2.annotate('Obese Smoker Cluster ', xy=(35, 48000), xytext=(20, 60000),

            arrowprops=dict(facecolor='black'),

            fontsize=12)

ax2.annotate('The Impact of Smoking to \n Charges on other \n Weight Conditions ', xy=(25, 26000), xytext=(17, 40000),

            arrowprops=dict(facecolor='black'),

            fontsize=12)

plt.show()
# Separation in Charges between Obese Smokers vs Non-Obese Smokers

# In this chart we can visualize how can separate obese smokers and obese non-smokers into different clusters 

# of groups. Therefore, we can say that smoking is a characteristic that definitely affects patient's charges.





# Creating a Scatter Plot with all the Obese



obese_smoker = df.loc[(df["weight_condition"] == "Obese") & (df["smoker"] == "yes")]

obese_nonsmoker = df.loc[(df["weight_condition"] == "Obese") & (df["smoker"] == "no")]





trace0 = go.Scatter(

    x = obese_smoker["age"].values,

    y = obese_smoker["charges"].values,

    name = 'Smokers',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = '#DF0101',

        line = dict(

            width = 2,

            color = 'rgb(0, 0, 0)'

        )

    )

)



trace1 = go.Scatter(

    x = obese_nonsmoker["age"].values,

    y = obese_nonsmoker["charges"].values,

    name = 'Non-Smokers',

    mode = 'markers',

    marker = dict(

        size = 10,

        color = '#00FF40',

        line = dict(

            width = 2,

        )

    )

)



data = [trace0, trace1]



layout = dict(title = 'Clear Separation between Obese Smokers and Non-Smokers in Charges',

              yaxis = dict(zeroline = False,

                          title="Patient Charges",

                          titlefont=dict(size=16)),

              xaxis = dict(zeroline = False,

                          title="Age of the Patient",

                          titlefont=dict(

                          size=16))

             )



fig = dict(data=data, layout=layout)

iplot(fig, filename='styled-scatter')
# Predicting cost of treatment with linear regression



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

#new_df = original_df.copy()



# Sex



le.fit(df.sex.drop_duplicates()) 

df.sex = le.transform(df.sex)



# Smoking Staus



le.fit(df.smoker.drop_duplicates()) 

df.smoker = le.transform(df.smoker)



# Region



le.fit(df.region.drop_duplicates()) 

df.region = le.transform(df.region)



# Age Category (Our created variable)



le.fit(df.age_cat.drop_duplicates()) 

df.age_cat = le.transform(df.age_cat)



# Weight Condition (Our created variable)



le.fit(df.weight_condition.drop_duplicates()) 

df.weight_condition = le.transform(df.weight_condition)



df.head()
# x = df.drop(['charges'], axis = 1)

x = df[['sex','children','smoker','age','bmi']]

y = df.charges



x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)

lr = LinearRegression().fit(x_train,y_train)



y_train_pred = lr.predict(x_train)

y_test_pred = lr.predict(x_test)



score = round(lr.score(x_test,y_test)*100,2)



# model evaluation

rmse = mean_squared_error(y_test, y_test_pred)

r2 = r2_score(y_test, y_test_pred)



# printing values

print('Slope:' ,lr.coef_)

print('Intercept:', lr.intercept_)

print('Root mean squared error: ', rmse)

print('R2 score: ', r2)



# predicted values



plt.plot(x_test,y_test_pred, color='red')

plt.show()
# This Linear Regression yields graph with many dimensions which we cannot plot since the dimension of the 

# graph increases as our features increase. In your case, X has four features. Scatter plot takes argument with 

# only one feature in X and only one class in y. We try taking only one feature for X (a continuous variable like age 

# or bmi) and plot a scatter plot. By doing so we will be able to study the effect of each feature on the dependent 

# variable (which is easier to comprehend than multidimensional plots). 





f, ax = plt.subplots()

f.set_figheight(15)

f.set_figwidth(15)

f.suptitle('Effect of Age and BMI on Charges', fontsize=20)





#Age

plt.subplot(2,2,1)

plt.scatter(x_test['age'],y_test, color="blue")

plt.xlabel("Age")

plt.ylabel("Charges")

plt.subplot(2,2,2)

plt.scatter(x_test['age'],y_test_pred, color="red")

plt.xlabel("Age")

plt.ylabel("Predicted Charges")



#BMI



plt.subplot(2,2,3)

plt.scatter(x_test['bmi'],y_test, color="blue")

plt.xlabel("BMI")

plt.ylabel("Charges")

plt.subplot(2,2,4)

plt.scatter(x_test['bmi'],y_test_pred,color="red")

plt.xlabel("BMI")

plt.ylabel("Predicted Charges")

plt.show()
# Predicting Insurance cost using Regression in Deep Learning



from keras.models import Sequential

from keras.layers import Dense



# There are two different keras versions of tensorflow and pure keras. They don't not work together. 

# You have to change everything to one version



x = df[['sex','children','smoker','age','bmi']]

y = df.charges



x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)



#model=Sequential()

#model.add(Dense(1,activation='relu',kernel_initializer='uniform',input_dim=4))

#model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#model.fit(x_train,y_train,epochs=10,batch_size=1)

#predictions=model.predict(x_test)

#print(predictions)



def build_model():

  model = Sequential([

    # Input Layer

    Dense(12, input_dim=5, activation='relu'),

    # Hidden Layers

    Dense(8, activation='relu'),

    Dense(4, activation='relu'),

    # Output Layer

   Dense(1, activation='linear')

  ])





  model.compile(loss='mse',

                optimizer='rmsprop',

                metrics=['accuracy'])

  return model



model = build_model()

model.summary()
history=model.fit(x_train,y_train,epochs=300,batch_size=5)

y_test_pred=model.predict(x_test)
score=round(r2_score(y_test,y_test_pred)*100,2)

print("Our Artificial Neural Network Model predicts the cost of treatment with around",score,"% accuracy \n")



x = range(335)

fig = plt.figure()

ax1 = fig.add_subplot(111)



ax1.scatter(x, y_test, s=10, c='b', marker="s", label='Real Values')

ax1.scatter(x, y_test_pred, s=10, c='r', marker="o", label='Predicted')

plt.legend(loc='upper left');

plt.show()



#The training set is fixed, but we set the initial weights of the neural network to a random value in a small range, 

# so each time you train the network you get slightly different results. 
from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor(n_estimators=50,max_depth=None,min_samples_split=4,min_samples_leaf=2)

regressor.fit(x_train,y_train)

score = round(regressor.score(x_test,y_test),2)*100

print("Our Random Forest Regression Model predicts the cost of treatment with around",score,"% accuracy \n")



x = range(335)

fig = plt.figure()

ax1 = fig.add_subplot(111)



ax1.scatter(x, y_test, s=10, c='b', marker="s", label='Real Values')

ax1.scatter(x, y_test_pred, s=10, c='r', marker="o", label='Predicted')

plt.legend(loc='upper left');

plt.show()
#  This algorithm is also a great choice, if you need to develop a model in a short period of time. On top of that, 

# it provides a pretty good indicator of the importance it assigns to your features. Random Forests are also very 

# hard to beat in terms of performance. A more accurate prediction requires more trees, which results 

# in a slower model.



# In the healthcare domain it is used to identify the correct combination of components in medicine and to analyze 

# a patient’s medical history to identify diseases.



# When we use LabelEncoder the model will interpret the data to be in some kind of order, 0 < 1 < 2. 

# This is true for bmi, smoking status, age category and weight condtion, but not for things like region (city,country)

# To overcome this problem, we use One Hot Encoder.



# In eandom forest algorithm it is very easy to measure the relative importance of each feature on the prediction. 

# A general rule in machine learning is that the more features you have, the more likely your model will suffer from 

# overfitting and vice versa.



importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(regressor.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head()



# We could go read the research papers on the random forest and try to theorize the best hyperparameters, 

# but a more efficient use of our time is just to try out a wide range of values and see what works



# n_estimators = number of trees in the foreset

# max_features = max number of features considered for splitting a node

# max_depth = max number of levels in each decision tree

# min_samples_split = min number of data points placed in a node before the node is split

# min_samples_leaf = min number of data points allowed in a leaf node

# bootstrap = method for sampling data points (with or without replacement)

importances.plot.bar()

plt.show()