import warnings

warnings.filterwarnings('ignore')
import pandas as pd

bl_fri = pd.read_csv("../input/black-friday/BlackFriday.csv", header = 'infer')
print(bl_fri.shape)
print(bl_fri.info())
bl_fri.head()
def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(bl_fri)
#Replacing the NaN with 0 in columns Product_Category_3 and Product_Category_2

bl_fri['Product_Category_2'] = bl_fri['Product_Category_2'].fillna(0)

bl_fri['Product_Category_3'] = bl_fri['Product_Category_3'].fillna(0)
from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())



gen_q = """

select Gender, count(distinct User_ID) as cnt

From bl_fri

GROUP BY Gender;

"""



gen_df = pysqldf(gen_q)
import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



fig = {

  "data": [

    {

      "values": gen_df.cnt,

      "labels": gen_df.Gender,

      "domain": {"x": [0, .5]},

      "hoverinfo":"label+percent",

      "hole": .3,

      "type": "pie"

    },],

 "layout": {

        "title":"Gender Proportion on Black Friday"

    }

}



iplot(fig)
from pandasql import sqldf

pysqldf = lambda q: sqldf(q, globals())



age_q = """

select Age, count(distinct User_ID) as cnt

From bl_fri

GROUP BY Age;

"""



age_df = pysqldf(age_q)
fig = {

  "data": [

    {

      "values": age_df.cnt,

      "labels": age_df.Age,

      "domain": {"x": [0, .5]},

      "hoverinfo":"label+percent",

      "hole": .3,

      "type": "pie"

    },],

 "layout": {

        "title":"Age Group Proportion on Black Friday"

    }

}



iplot(fig)
am_q = """

select Age,  Marital_Status, count(distinct User_ID) as cnt

From bl_fri

GROUP BY Age,  Marital_Status;

"""



am_df = pysqldf(am_q)



am_df_m = am_df[am_df.Marital_Status == 1]

am_df_nm = am_df[am_df.Marital_Status == 0]
trace1 = go.Bar(

                x = am_df_m.Age,

                y = am_df_m.cnt,

                name = "Married")



trace2 = go.Bar(

                x = am_df_nm.Age,

                y = am_df_nm.cnt,

                name = "Single")



data = [trace1, trace2]

layout = go.Layout(barmode = "group", title = "Buying volume - Marriage Vs Age")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
#Product Category 1

gen_prod_1 = """

select Gender, Product_Category_1, count(Product_Category_1) as cnt

From bl_fri

GROUP BY Gender, Product_Category_1;

"""



gen_prod_1_df = pysqldf(gen_prod_1)



gen_prod_1_m = gen_prod_1_df[gen_prod_1_df.Gender == 'M']

gen_prod_1_f = gen_prod_1_df[gen_prod_1_df.Gender == 'F']
fig = {

  "data": [

    {

      "values": gen_prod_1_m.cnt,

      "labels": gen_prod_1_m.Product_Category_1,

      "domain": {"x": [0, .48]},

      "name": "Product Category 1",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": gen_prod_1_f.cnt,

      "labels": gen_prod_1_f.Product_Category_1,

      "domain": {"x": [.52, 1]},

      "name": "Product Category 1",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Product Category 1 Vs Gender",

        "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Male",

                "x": 0.20,

                "y": 0.5

            },

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Female",

                "x": 0.8,

                "y": 0.5

            }

        ]

    }

}



iplot(fig)
#Product Category 2

gen_prod_2 = """

select Gender, Product_Category_2, count(Product_Category_2) as cnt

From bl_fri

where Product_Category_2 != 0

GROUP BY Gender, Product_Category_2;

"""



gen_prod_2_df = pysqldf(gen_prod_2)



gen_prod_2_m = gen_prod_2_df[gen_prod_2_df.Gender == 'M']

gen_prod_2_f = gen_prod_2_df[gen_prod_2_df.Gender == 'F']
fig = {

  "data": [

    {

      "values": gen_prod_2_m.cnt,

      "labels": gen_prod_2_m.Product_Category_2,

      "domain": {"x": [0, .48]},

      "name": "Product Category 2",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": gen_prod_2_f.cnt,

      "labels": gen_prod_2_f.Product_Category_2,

      "domain": {"x": [.52, 1]},

      "name": "Product Category 2",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Product Category 2 Vs Gender",

        "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Male",

                "x": 0.20,

                "y": 0.5

            },

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Female",

                "x": 0.8,

                "y": 0.5

            }

        ]

    }

}



iplot(fig)
#Product Category 3

gen_prod_3 = """

select Gender, Product_Category_3, count(Product_Category_3) as cnt

From bl_fri

where Product_Category_3 != 0

GROUP BY Gender, Product_Category_3;

"""



gen_prod_3_df = pysqldf(gen_prod_3)



gen_prod_3_m = gen_prod_3_df[gen_prod_3_df.Gender == 'M']

gen_prod_3_f = gen_prod_3_df[gen_prod_3_df.Gender == 'F']
fig = {

  "data": [

    {

      "values": gen_prod_3_m.cnt,

      "labels": gen_prod_3_m.Product_Category_3,

      "domain": {"x": [0, .48]},

      "name": "Product Category 3",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": gen_prod_3_f.cnt,

      "labels": gen_prod_3_f.Product_Category_3,

      "domain": {"x": [.52, 1]},

      "name": "Product Category 3",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Product Category 3 Vs Gender",

        "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Male",

                "x": 0.20,

                "y": 0.5

            },

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Female",

                "x": 0.8,

                "y": 0.5

            }

        ]

    }

}



iplot(fig)
gen_occ = """

select Gender, Occupation, count(Occupation) as cnt

From bl_fri

GROUP BY Gender, Occupation;

"""



gen_occ_df = pysqldf(gen_occ)



gen_occ_df_m = gen_occ_df[gen_occ_df.Gender == 'M']

gen_occ_df_f = gen_occ_df[gen_occ_df.Gender == 'F']
fig = {

  "data": [

    {

      "values": gen_occ_df_m.cnt,

      "labels": gen_occ_df_m.Occupation,

      "domain": {"x": [0, .48]},

      "name": "Occupation",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": gen_occ_df_f.cnt,

      "labels": gen_occ_df_f.Occupation,

      "domain": {"x": [.52, 1]},

      "name": "Occupation",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Occupation Vs Gender",

        "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Male",

                "x": 0.20,

                "y": 0.5

            },

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Female",

                "x": 0.8,

                "y": 0.5

            }

        ]

    }

}



iplot(fig)
import numpy as np

city = """

select distinct User_ID, City_Category, Stay_In_Current_City_Years, Age

From bl_fri

GROUP BY User_ID, City_Category, Stay_In_Current_City_Years, Age;

"""



city_df = pysqldf(city)



city_df['Years'] = np.where(city_df['Stay_In_Current_City_Years']=='4+', '4', city_df['Stay_In_Current_City_Years'])



city_df['Years'] = city_df['Years'].astype(str).astype(int)
city_avg = """

select City_Category, Age, round(avg(Years),2) as Avg_yrs

From city_df

GROUP BY City_Category, Age;

"""



city_avg_df = pysqldf(city_avg)



city_avg_A_df = city_avg_df[city_avg_df.City_Category == 'A']

city_avg_B_df = city_avg_df[city_avg_df.City_Category == 'B']

city_avg_C_df = city_avg_df[city_avg_df.City_Category == 'C']
trace_A = go.Bar(

                x = city_avg_A_df.Age,

                y = city_avg_A_df.Avg_yrs,

                name = "City - A")



trace_B = go.Bar(

                x = city_avg_B_df.Age,

                y = city_avg_B_df.Avg_yrs,

                name = "City - B")



trace_C = go.Bar(

                x = city_avg_C_df.Age,

                y = city_avg_C_df.Avg_yrs,

                name = "City - C")



data = [trace_A, trace_B, trace_C]

layout = go.Layout(barmode = "group", title = "Average yrs of stay - City Vs Age")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
ad_dollar = """

select Age, Gender, sum(Purchase) as Dollars

From bl_fri

GROUP BY Age, Gender;

"""



ad_dollar_df = pysqldf(ad_dollar)



ad_dollar_m = ad_dollar_df[ad_dollar_df.Gender == 'M']

ad_dollar_f = ad_dollar_df[ad_dollar_df.Gender == 'F']
trace_M = go.Bar(

                x = ad_dollar_m.Age,

                y = ad_dollar_m.Dollars,

                name = "Male")



trace_F = go.Bar(

                x = ad_dollar_f.Age,

                y = ad_dollar_f.Dollars,

                name = "Female")



data = [trace_M, trace_F]

layout = go.Layout(barmode = "group", title = "Purchase in Dollar - Age Vs Gender")

fig = go.Figure(data = data, layout = layout)

iplot(fig)