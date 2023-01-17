# importing libraries 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import plotly.express as px 

import plotly.graph_objs as go 

%matplotlib inline
# creating the dataframe

df = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")

df.head()

# checking the missing values in the dataframe

df.isna().sum()
# visualising the missin values in the dataframe

plt.figure(figsize = (8 , 6))

sns.heatmap(df.isnull() , yticklabels = False)
# correncting the names of the columns

df.columns = ['country', 'year', 'status', 'life_expectancy', 'adult_mortality',

       'infant_deaths', 'alcohol', 'percentage_expenditure', 'hepatitis_b',

       'measles', 'bmi', 'under_five_deaths', 'polio', 'total_expenditure',

       'diphtheria', 'hiv/aids', 'gdp', 'population',

       'thinness 1-19 years', 'thinness 5-9 years',

       'income_composition_of_resources', 'schooling']
description1 = df.describe()

# filling the life expectancy and adult mortality with the mean values as it has a very less number of missing values 

df["life_expectancy"] = df["life_expectancy"].fillna(value = df["life_expectancy"].mean())

df["adult_mortality"] = df["adult_mortality"].fillna(value = df["adult_mortality"].mean())

df.isna().sum()
# checking the correlation matrix to get the relation between the different features 

corelation = df.corr()

corelation

plt.figure(figsize = (8 , 6))

sns.heatmap(df.corr() , annot = True)
# filling the missing values of alcohol 

# is can be seen that the alcohol column is mostly correlated with the schooling column

# so using the schooling column as reference to fill the missing values 

# min value of schooling is 0 and max value is 20.7

# cutting the dataframe into 4 parts [0 , 5 , 10  , 15 , 21]



df["group"] = pd.cut(df["schooling"] , bins = (0 , 5 , 7.5 , 10 , 15 , 21 ) , labels = ["g1" , "g2" , "g3" , "g4" , "g5"])

df["group"].value_counts()



grouped = df.groupby(df.group)["alcohol"].mean()

grouped

# this gives the average value for each group
# filling the missing values in the alcohol column of the dataframe

# we will not use NaN instead we will use 0.01(min) value instead of that

def impute_alcohol(col):

    a = col[0]

    s = col[1]

    if pd.isnull(a):

        if (s<5):

            return 1.56

        elif (5<=s<7.5):

            return 1.33

        elif (7.5<=s<10):

            return 2.36

        elif(10<=s<15):

            return 4.40

        elif (s>=15):

            return 9.00

       

    else :

        return a



df["alcohol"] = df[["alcohol" , 'schooling']].apply(impute_alcohol , axis = 1)



# filling the remaining alcohol rows using the mean

df["alcohol"] = df["alcohol"].fillna(value = df["alcohol"].mean())



# checking that either every value is filled in the alcohol column or not

print(df["alcohol"].isna().sum())



# finally dropping the additional group column that  we created 

df = df.drop(["group"] , axis = 1)
# filling up the bmi column 

# bmi is highly corelated with life_expectancy column

# life expectancy ranges from 36.3 to 89



df["group"] = pd.cut(df["life_expectancy"] , bins = (30 ,40 , 50 , 60 , 70 , 80 , 90) , labels = ["g1" , "g2" , "g3" , "g4" , "g5" , "g6"])

df["group"].value_counts()





grouped = df.groupby(df.group)["bmi"].mean()

grouped


def impute_bmi(col):

    b = col[0]

    l = col[1]

    if pd.isnull(b):

        if (l<40):

            return 30.70

        elif (40<=l<50):

            return 19.18

        elif (50<=l<60):

            return 19.12

        elif(60<=l<70):

            return 32.97

        elif (70<=l<80):

            return 46.82

        elif (l>= 80):

            return 50.79



    else :

        return b



df["bmi"] = df[["bmi" , 'life_expectancy']].apply(impute_bmi , axis = 1)





# checking that either every value is filled in the bmi column or not

print(df["bmi"].isna().sum())



# finally dropping the additional group column that  we created 

df = df.drop(["group"] , axis = 1)
def impute_schooling(col):

    s = col[0]

    l = col[1]

    if pd.isnull(s):

        if (l<40):

            return 7.65

        elif (40<=l<50):

            return 8.15

        elif (50<=l<60):

            return 8.21

        elif(60<=l<70):

            return 10.54

        elif (70<=l<80):

            return 13.48

        elif (l>= 80):

            return 16.51



    else :

        return s



df["schooling"] = df[["schooling" , 'life_expectancy']].apply(impute_schooling , axis = 1)





# checking that either every value is filled in the bmi column or not

print(df["schooling"].isna().sum())







df["schooling"].describe()
# polio and diphtheria has very less number of missing values 

# filling it with the mean value of the column will not have much effect on the performance

df = df.fillna(value = {

    "polio" : df["polio"].mean() , 

    "diphtheria" : df["diphtheria"].mean()

})



# checking whether the values have been filled or not

df["polio"].isna().sum() , df["diphtheria"].isna().sum()


def impute_hepatitis(col):

    h = col[0]

    d = col[1]

    if pd.isnull(h):

        

        if (d<50):

            return 37.60

        elif (50<=d<60):

            return 53.00

        elif (60<=d<70):

            return 57.93

        elif(70<=d<80):

            return 66.34

        elif (80<=d<90):

            return 78.84

        elif (d>= 90):

            return 91.05



    else :

        return h



df["hepatitis_b"] = df[["hepatitis_b" , 'diphtheria']].apply(impute_hepatitis , axis = 1)





# checking that either every value is filled in the bmi column or not

print(df["hepatitis_b"].isna().sum())





df["hepatitis_b"].describe()
# filling thinness columns with the mean values as it has a lesser number of 

df["thinness 1-19 years"]  = df["thinness 1-19 years"].fillna(value = df["thinness 1-19 years"].mean())

df["thinness 5-9 years"]  = df["thinness 5-9 years"].fillna(value = df["thinness 5-9 years"].mean())



df["thinness 1-19 years"].isna().sum() , df["thinness 5-9 years"].isna().sum()
# filling income_composition_of_resources column 

# this column is highly related to schooling column (0.80)

df["group"] = pd.cut(df["schooling"] , bins = (0 , 5 , 7.5 , 10 , 15 , 21 ) , labels = ["g1" , "g2" , "g3" , "g4" , "g5"])

df["group"].value_counts()



grouped = df.groupby(df.group)["income_composition_of_resources"].mean()

grouped

def impute_income(col):

    i = col[0]

    s = col[1]

    if pd.isnull(i):

        if (s<5):

            return 0.26

        elif (5<=s<7.5):

            return 0.37

        elif (7.5<=s<10):

            return 0.45

        elif(10<=s<15):

            return 0.66

        elif (s>=15):

            return 0.84

       

    else :

        return i



df["income_composition_of_resources"] = df[["income_composition_of_resources" , 'schooling']].apply(impute_income , axis = 1)



# checking that either every value is filled in the alcohol column or not

print(df["income_composition_of_resources"].isna().sum())



# finally dropping the additional group column that  we created 

df = df.drop(["group"] , axis = 1)
# filling the missing values in total expenditure columns

# total_expenditure is not highly corelated to any of the features but among them it is best connected with the alcohol column 

# alcohol ranges from 0.01 to 17.87





df["group"] = pd.cut(df["alcohol"] , bins = (0 , 5 , 10 , 15  , 20) , labels = ["g1" , "g2" , "g3" , "g4"])

df.group.value_counts()





grouped = df.groupby(df["group"])["total_expenditure"].mean()

grouped
def impute_expenditure(col):

    t = col[0]

    a = col[1]

    if pd.isnull(t):

        if (a<5):

            return 5.37

        elif (5<=a<10):

            return 6.71

        elif (10<=a<15):

            return 6.88

        elif(a>15):

            return 5.81

        

    else :

        return t



df["total_expenditure"] = df[["total_expenditure" , 'alcohol']].apply(impute_expenditure , axis = 1)



# checking that either every value is filled in the alcohol column or not

print(df["total_expenditure"].isna().sum())



# finally dropping the additional group column that  we created 

df = df.drop(["group"] , axis = 1)



df["total_expenditure"].describe()
bins = [0 , 1250 , 2500 , 3750 , 7500 , 8750 , 10000 , 11250 , 12500, 15000 , 17500 ,20000]

labels =  ["g1" ,"g2", "g3" ,"g4" ,"g5", "g6", "g7" ,"g8", "g9" ,"g10" ,"g11" ]

df["group"] = pd.cut(df["percentage_expenditure"] , bins = bins , labels = labels)



grouped = df.groupby(df["group"])["gdp"].mean()

grouped
# gdp is very highly corelated with percentage expenditure

def impute_GDP(c):

    g=c[0]

    p=c[1]

    if pd.isnull(g):

        if p<=1250:

            return 2617.56

        elif 1250<p<=2500:

            return 18457.32

        elif 2500<p<=3750:

            return 28719.60

        elif 3750<p<=7500:

            return 39217.12

        elif 7500<p<=8750:

            return 48372.83

        elif 8750<p<=10000:

            return 54822.89

        elif 10000<p<=11250:

            return 58842.19

        elif 11250<p<=12500:

            return 67018.11

        elif 12500<p<=15000:

            return 76305.27

        elif 15000<p<=17500:

            return 105214.53

        elif p>17500:

            return 91186.03

    else:

        return g

    

df['gdp']=df[['gdp','percentage_expenditure']].apply(impute_GDP,axis=1)



# checking that either every value is filled in the alcohol column or not

print(df["gdp"].isna().sum())



# finally dropping the additional group column that  we created 

df = df.drop(["group"] , axis = 1)



df["gdp"].describe()

# polulation is corelated with infant death column

bins = []

j = 0

for i in range (0,2100 , 400):

    bins.append(i)

    

df["group"] = pd.cut(df["infant_deaths"] , bins = bins)

df.group.value_counts()



grouped = df.groupby(df.group)["population"].mean()

grouped
def impute_pop(col):

    p = col[0]

    i = col[1]

    if pd.isnull(p):

        if i<400:

            return 1.228551e+07

        elif (400<=i<800):

            return 5.975911e+07

        elif (800<=i<1200):

            return 2.810998e+08

        elif(1200<=i<1600):

            return 8.088425e+08

        elif(i>=1600):

            return 5.095718e+07

    else:

        return p





df['population']=df[['population','infant_deaths']].apply(impute_pop,axis=1)



# checking that either every value is filled in the alcohol column or not

print(df["population"].isna().sum())



# finally dropping the additional group column that  we created 

df = df.drop(["group"] , axis = 1)



df["population"].describe()
# data is completed

df.isna().sum()
df.columns
# scatter plot

status = df["status"].unique()

status = list(status)





fig = px.scatter(data_frame = df , 

                x = "infant_deaths" , 

                y = "life_expectancy" , 

                size = "adult_mortality", 

                size_max = 10, 

                color = "status" , 

                opacity = 0.8 , 

                template = "seaborn" , 

                hover_name = df["country"] , 

                hover_data = [df["schooling"] , df["population"] , df["total_expenditure"]] , 

                marginal_x = "rug" , 

                marginal_y = "histogram" , 

                range_color = (0,10), 

                color_discrete_map = {"Developed" : "rgb(255,76,78)" , 

                                      "Developing" : "rgb(98,78,150)"} ,

#                 color_continuous_scale="Darkmint" , 

                category_orders = {"status" : ["Developed" , "Developing"]} , 

                height = 550 ,

                  width = 800

                

                ) 





fig.update_layout(

    title='Infant Deaths vs Life Expectancy',

    xaxis=dict(

        title='Infant Deaths',

        gridcolor='white',

        type='log',

        gridwidth=2,

    ),

    yaxis=dict(

        title='Life Expectancy (years)',

        gridcolor='white',

        type = "log" , 

        gridwidth=2,

    ),

    

    paper_bgcolor='rgb(235, 235, 235)',

    plot_bgcolor='rgb(243, 243, 243)', 

    

)

fig.show()

# overlay histogram

fig = px.histogram(data_frame = df ,

                  x = "schooling" , 

                  color = "status" , 

                  barmode = "overlay" , 

                  marginal = "rug" , 

                  opacity = 0.6, 

                  hover_name = "status",

                  template = "seaborn" , 

#                   histnorm = "probability density" ,    

                   color_discrete_map = dict(Developed = "#26828e" , Developing = "#cf4446")

                  )



# fixing the layout of the plot

fig.update_layout(

    title='Overlay Histogram',

    xaxis=dict(

        title='Schooling',

        gridcolor='white',

        gridwidth=2,

    ),

    yaxis=dict(

        title='count',

        gridcolor='white', 

        gridwidth=2,

    ),

    

    paper_bgcolor='rgb(230, 230 , 230)',

    plot_bgcolor='rgb(243, 243, 243)', 

    

#     for grouped histogram you can use following two additional parmeter

#     bargap = 0.2 , 

#     bargroupgap = 0.1

    

)



fig.show()
# grouped histogram

bins = []

for i in range (35 , 90 , 5):

    bins.append(i)

    

    

fig = px.histogram(data_frame = df ,

                  x = "life_expectancy" , 

                  color = "status" , 

                  barmode = "group" , 

                  marginal = "rug" , 

                  hover_name = "status",

                  template = "seaborn" , 

#                   histnorm = "probability density" ,    

                   color_discrete_map = dict(Developed = "#bd3786" , Developing = "#cf4446") , 

                   nbins = 11 , 

                   range_x = (35 , 90) , 

                   opacity = 0.6, 

                  )



fig.update_layout(

    title = "Grouped Histogram" , 

    

    xaxis = dict (

        title = "Life Expectancy" , 

        gridcolor = "white" , 

        gridwidth = 2

    ) , 

    yaxis = dict (

        title = "Count" , 

        gridcolor = "white" , 

        gridwidth = 2

    ) , 

    paper_bgcolor = 'rgb(230, 230 , 230)' , 

    plot_bgcolor = 'rgb(243, 243 , 243)',

    bargap = 0.1,

    bargroupgap = 0.1,

    

)



fig.show()
# pie chart 

grouped = df.groupby(df['country'])['population'].mean()



grouped = pd.DataFrame(index = df["country"].unique() , data = grouped)

grouped = grouped.sort_values(by = "population" , ascending = False)

grouped = grouped.head(10)

 

fig = px.pie(data_frame = grouped , 

            names = grouped.index , 

            values = "population" , 

            template = "seaborn" , 

             opacity = 0.8 , 

            color_discrete_sequence=px.colors.sequential.Cividis , 

            hole = 0.5 , 

#             color_discrete_map = , 

             

            )



fig.update_traces (pull= 0.05 , textinfo = "percent+label" , rotation = 90)



fig.update_layout(

    title = "Pie Chart" , 

    paper_bgcolor = 'rgb(230, 230 , 230)' , 

    plot_bgcolor = 'rgb(243, 243 , 243)',

    annotations=[dict(text='Mean Population', font_size=20, showarrow=False)]

)



fig.show()
# violin plot

df["life_type"] = pd.cut(df["life_expectancy"] , 

                        bins = (0, 50 , 65 , 75 , 85 , 100) , 

                        labels = ("Bad" , "Average" , "Good" , "Very Good" , "Excellent"))



fig = px.violin(data_frame = df , 

            x = "status" , 

            y = "total_expenditure" , 

            template = "seaborn" , 

            color_discrete_sequence = px.colors.sequential.Plasma ,

#             color_discrete_map = {"Developing" : "#9e2f7f" , "Developed" : "#26828e"} , 

            box = True ,

            points = "outliers" , 

            hover_name = "country" , 

            hover_data = ["life_type" ,"life_expectancy" , "percentage_expenditure"] , 

#             animation_frame = "life_type" , 

#             animation_group = "status"

            )





fig.update_layout(title = "Violin plot" , 

                 xaxis = dict(title = "Country Status" , 

                             gridcolor = "white" , 

                             gridwidth = 2) , 

                 yaxis = dict(title = "Total Expenditure" ,

                             gridcolor = "white" , 

                             gridwidth = 2) , 

                 paper_bgcolor = 'rgb(230, 230 , 230)' , 

                 plot_bgcolor = 'rgb(243, 243 , 243)' 

                 )

fig.show()



df = df.drop("life_type", axis = 1)
# stacked histogram

# how is life expectancy distributed

bins = []

for i in range (35 , 90 , 5):

    bins.append(i)

    

    

fig = px.histogram(data_frame = df ,

                  x = ["thinness 1-19 years" , "thinness 5-9 years"], 

                  opacity = 0.6, 

#                   barmode = "relative" ,  

                  color_discrete_map = {"thinness 1-19 years" : "#440f76" , 

                                       "thinness 5-9 years" : "#26828e"} ,

                  marginal = "rug" , 

                  nbins = 9 ,

                  range_x = (0,30)                   

                  )



fig.update_layout(

    title = "Stacked Histogram" , 

    

    xaxis = dict (

        title = "Thinness" , 

        gridcolor = "white" , 

        gridwidth = 2

    ) , 

    yaxis = dict (

        title = "Count" , 

        gridcolor = "white" , 

        gridwidth = 2

    ) , 

    paper_bgcolor = 'rgb(230, 230 , 230)' , 

    plot_bgcolor = 'rgb(243, 243 , 243)',

    bargap = 0.1,

    bargroupgap = 0.1,

    

)



fig.show()

# animated scatter plot

df["life_type"] = pd.cut(df["life_expectancy"] , bins = (0 , 50 , 65 , 75 , 85 , 100) , 

                        labels = ["Bad" , "Average" , "Good" , "Very Good" , "Excellent"])



fig = px.scatter(data_frame = df , 

                x = "hepatitis_b" , 

                y = "life_expectancy", 

                color = "life_type" , 

#                 color_discrete_sequence = px.colors.sequential.Plasma, 

                template = "seaborn", 

                color_discrete_map = {

                    "Bad" : "#fc67fd",

                    "Average" : "#35b779", 

                    "Good" : px.colors.sequential.Inferno[4], 

                    "Very Good" : "#f1605d", 

                    "Excellent" : "#bd3786"

                } , 

                log_x = True , 

                size_max = 15 , 

                size = "alcohol" , 

                marginal_x = "rug" , 

                marginal_y = "histogram" , 

                hover_name = "country"  ,

                animation_frame = "year"

                )



fig.update_layout(title = "Animated Scatterplot" , 

                 xaxis = dict(title = "Hepatitis B" ,

                             gridwidth = 2 , 

                             gridcolor = "white") ,

                 yaxis = dict(title = "Life Expectancy" , 

                             gridcolor = "white" , 

                             gridwidth = 2) , 

                 paper_bgcolor = 'rgb(230, 230 , 230)' ,

                 plot_bgcolor = 'rgb(243, 243 , 243)')
# .animated box plot

df["schooling_type"] = pd.cut(df["schooling"] , bins = (-1 ,5 , 10 , 15 , 22) , 

                             labels = ("Bad" , "Good" , "Very Good", "Excellent"))



fig = px.box(data_frame = df , 

            x = "schooling_type" , 

            y = 'income_composition_of_resources' , 

#             notched = True

            points = "suspectedoutliers" , 

            color = "status" , 

#             color_discrete_sequence = px.colors.sequential.Plasma 

            category_orders = dict(schooling_type = ["Bad" , "Good" , "Very Good" , "Excellent"]) , 

            template = "seaborn" , 

#             log_y = True 

             boxmode = "group" , 

             hover_name = "status" , 

             animation_frame = "year"

            )



fig.update_layout(title = "Animated BoxPlot" , 

                 xaxis = dict (title = "Schooling Type" , 

                              gridcolor = "white",

                              gridwidth = 2) , 

                 yaxis = dict (title = "Income Composition of resources" ,

                              gridcolor = "white" , 

                              gridwidth = 2) , 

                 paper_bgcolor = 'rgb(230, 230 , 230)',

                 plot_bgcolor = 'rgb(243, 243 , 243)' , 

#                  margin={"r":0,"t":0,"l":0,"b":0} 

                 )



fig.show()
# animated scatter plot

fig = px.scatter_3d(data_frame = df , 

                   x = "population" , 

                   y = "percentage_expenditure" , 

                   z = "total_expenditure" , 

                   size = "alcohol" , 

                   template = "seaborn" ,

                   color = "life_type" , 

                   animation_frame = "year" , 

                   size_max = 30 , 

                   opacity = 0.7 , 

                   width = 800 , 

                   height = 600 ,  

                   hover_name = "country" , 

                   hover_data = ["year" , "life_expectancy" , "income_composition_of_resources"], 

                   labels = {

                       "total_expenditure" : "Total Expenditure" , 

                       "percentage_expenditure" : "Percentage Expenditure" , 

                       "population" : "Population",

                       "life_type" : "Life Type"

                   }

                   )

                    



fig.update_layout(title = "Animated 3D Scatter Plot" , 

                 paper_bgcolor = 'rgb(230, 230 , 230)',

                 plot_bgcolor = 'rgb(243, 243 , 243)' , 

                 )
# Sunburst Plot

gdf = df.groupby(df["year"])["hiv/aids"].max()



ha = pd.DataFrame(columns = df.columns)

for i in gdf:

    a = df[df["hiv/aids"] == i]

    ha = ha.append(a)

    

ha = ha.drop_duplicates(subset = ["hiv/aids"])

ha = ha.sort_values (by = "year" , ascending = False)



fig = px.sunburst(data_frame = ha , 

                path = ["year" , "country"] , 

                  values = "hiv/aids" , 

                  color = "measles" , 

                  template = "seaborn" , 

                 color_discrete_sequence = px.colors.sequential.Viridis , 

                 color_continuous_scale= px.colors.sequential.Viridis ) 



fig.update_layout(title = "SuburstPlot" ,  

                 paper_bgcolor = 'rgb(230, 230 , 230)',

                 plot_bgcolor = 'rgb(243, 243 , 243)' , 

                 )

fig.update_traces(branchvalues = "total")



fig.show()

# removing life_type and schooling_type column

df = df.drop(["life_type" , "schooling_type"] , axis = 1)
# we have 2 categorical type features ..converting it to one hot encoding 

df_countries = pd.get_dummies(df["country"] )

df_status = pd.get_dummies(df["status"] )



# concating it to the original dataframe 

df = df.drop(["status" , "country"] , axis = 1)

df = pd.concat([df , df_countries , df_status] , axis = 1)

df.shape
from sklearn.preprocessing import MinMaxScaler



x = df.drop(["life_expectancy"] , axis = 1)

y = df["life_expectancy"]

y = np.array(y).reshape(-1,1)





scaler_x = MinMaxScaler(feature_range = (0,1))

scaled_x = scaler_x.fit_transform(x)

scaled_x = pd.DataFrame(columns = x.columns , data = scaled_x)

# splitting the data into training and testing sets

from sklearn.model_selection import train_test_split



x_train , x_test , y_train , y_test = train_test_split(scaled_x , y , test_size = 0.1)



y_train = np.reshape(y_train , (2644 , ))

y_test = np.reshape(y_test , (294 , ))

x_train.shape , x_test.shape , y_train.shape , y_test.shape

# making an empty list that will store the result of various ml models

from sklearn.metrics import r2_score



def models(x_train , x_test , y_train , y_test) : 

    

    scores = []

    

    from sklearn.linear_model import Lasso

    lr = Lasso(alpha = 0.001 , max_iter = 5000)

    lr.fit(x_train , y_train)

    lr_predict = lr.predict(x_test)

    scores.append({

        "Model" : "Lasso" , 

        "Score" : r2_score( y_test ,lr_predict)

    })

    

    from sklearn.linear_model import Ridge

    rr = Ridge()

    rr.fit(x_train , y_train)

    rr_predict = rr.predict(x_test)

    scores.append({

        "Model" : "Ridge" , 

        "Score" : r2_score( y_test ,lr_predict)

    })

    

    from sklearn.linear_model import TheilSenRegressor

    tr = TheilSenRegressor()

    tr.fit(x_train , y_train)

    tr_predict = tr.predict(x_test)

    scores.append({

        "Model" : "TheilSenRegressor" , 

        "Score" : r2_score( y_test ,tr_predict)

    })

     

    from sklearn.linear_model import HuberRegressor

    hr = HuberRegressor(max_iter = 5000)

    hr.fit(x_train , y_train)

    hr_predict = hr.predict(x_test)

    scores.append({

        "Model" : "HuberRegressor" , 

        "Score" : r2_score( y_test ,hr_predict)

    })

    

    from sklearn.svm import SVR

    svr = SVR(kernel = "poly")

    svr.fit(x_train , y_train)

    svr_predict = svr.predict(x_test)

    scores.append({

        "Model" : "SVR" , 

        "Score" : r2_score( y_test , svr_predict)

    })

    

    from sklearn.tree import DecisionTreeRegressor

    dtr = DecisionTreeRegressor()

    dtr.fit(x_train , y_train)

    dtr_predict = dtr.predict(x_test)

    scores.append({

        "Model" : "DecisionTreeRegressor" , 

        "Score" : r2_score( y_test , dtr_predict)

    })

    

    from sklearn.ensemble import RandomForestRegressor

    rfr = RandomForestRegressor()

    rfr.fit(x_train , y_train)

    rfr_predict = rfr.predict(x_test)

    scores.append({

        "Model" : "RandomForestRegressor" , 

        "Score" : r2_score( y_test , rfr_predict)

    })

    

    import xgboost as xgb

    xboost = xgb.XGBRegressor(n_estimators = 200)

    xboost.fit(x_train, y_train)

    xboost_predict = xboost.predict(x_test)

    scores.append({

        "Model" : "XGBRegressor" , 

        "Score" : r2_score(y_test , xboost_predict)

    })

    return  scores



results = models(x_train , x_test, y_train , y_test)

results = pd.DataFrame(results)
# visualising the accuracy of the different models

fig = px.bar(data_frame = results , 

      x = "Model" , 

      y = "Score" , 

      opacity = 0.5 , 

      color_discrete_sequence = px.colors.sequential.Cividis , 

      hover_name = "Model" ,)



fig.update_layout(title = "Results" , 

                 xaxis = dict(title = "Models" , 

                             gridcolor = "white" , 

                             gridwidth = 2) , 

                 yaxis = dict(title = "r2 Score" , 

                             gridcolor = "white" , 

                             gridwidth = 2) ,

                 paper_bgcolor = "rgb(230 , 230 , 230)", 

                 plot_bgcolor = "rgb(243 , 243 , 243)", 

                                    )


