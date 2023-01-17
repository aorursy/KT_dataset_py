import pandas as pd

import seaborn as sns

import numpy as np
df = pd.read_csv("/kaggle/input/crimes-in-boston/crime.csv",encoding='latin1')

df
df = df[["OFFENSE_CODE_GROUP","DISTRICT","OCCURRED_ON_DATE","STREET"]]

df["DATE"] = pd.to_datetime(df["OCCURRED_ON_DATE"],format="%Y-%m-%d").dt.date

df
!pip install progplot
#import barwriter

from progplot import BarWriter
#create the barwriter object

bw = BarWriter()
help(bw.set_data)
df
bw.set_data(data=df, category_col="OFFENSE_CODE_GROUP", timeseries_col="DATE", value_col="DISTRICT", groupby_agg="count", resample_agg="sum", output_agg="cumsum", resample = "1w")

help(bw.set_display_settings)
bw.set_display_settings(time_in_seconds=30, video_file_name = "total_crimes_by_type.mp4")

help(bw.set_chart_options)
bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="magma", 

                     title="Top 10 Crimes by Total Offences from <mindatetime> to <currentdatetime>",dateformat="%Y-%m-%d", 

                     y_label="Offence", 

                     use_top_x=20, display_top_x=10,

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end")

bw.test_chart(30)
bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="bone", # <--------Change 

                     title="Top 10 Crimes by Total Offences from <mindatetime> to <currentdatetime>",dateformat="%Y-%m-%d", 

                     y_label="Offence", 

                     use_top_x=20, display_top_x=10,

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end")

bw.test_chart(30)
bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="bone",

                     title="Top 10 Crimes by Total Offences from <mindatetime> to <currentdatetime>",dateformat="%Y-%m-%d", 

                     y_label="Offence", 

                     use_top_x=10, display_top_x=10,

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end",

                     sort=False) # <-------- SORTED?

bw.test_chart(30)
bw.write_video()

bw.show_video()
bw.create_gif()

bw.show_gif()
# we will set the groupby to count. - this will count the crimes for each unique date.



# we will resample per week and calculate the MEAN to get the mean daily value per week



# we will use "4rolling" for the output_agg this will give us the mean over 4 windows to smooth the result.



# AS THERE IS A LOT OF DATA THIS COULD TAKE SOME TIME TO RUN.



bw.set_data(data=df, category_col="STREET", timeseries_col="DATE", value_col="DISTRICT", groupby_agg="count", resample = "1w", resample_agg="mean", output_agg="4rolling")

bw.set_display_settings(time_in_seconds=30, video_file_name = "mean_daily_crimes_by_street_.mp4")
bw.set_chart_options(x_tick_format="{:,.2f}", #<---- add two decimals to the formatting as mean will most likely product floats

                     palette="bone",

                     title="Mean Daily Crimes by Street from <rollingdatetime> to <currentdatetime>",dateformat="%Y-%m-%d", ##   <-------- change 

                     y_label="Offence", 

                     use_top_x=15, display_top_x=10,

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end",

                     sort=True) # <-------- SORTED?

bw.test_chart(30)
bw.write_video()

bw.show_video()
bw.set_chart_options(x_tick_format="{:,.2f}", #<---- add two decimals to the formatting as mean will most likely product floats

                     palette="bone",

                     title="Mean Daily Crimes by Street from <rollingdatetime> to <currentdatetime>",dateformat="%Y-%m-%d", ##   <-------- change 

                     y_label="Offence", 

                     use_top_x=15, display_top_x=10,

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end",

                     sort=False) # <-------- SORTED?

bw.test_chart(30)
bw.write_video()

bw.show_video()
bw.set_data(data=df, category_col="OFFENSE_CODE_GROUP", timeseries_col="DATE", value_col="DISTRICT", groupby_agg="count", resample_agg="sum", output_agg="cumsum", resample = "1w")

bw.set_display_settings(time_in_seconds=30, video_file_name = "total_crimes_by_type_picture.mp4")
# gather the images



!wget https://raw.githubusercontent.com/lewis-morris/progplot/master/icons/investigate.png

!wget https://raw.githubusercontent.com/lewis-morris/progplot/master/icons/medical.jpg

!wget https://raw.githubusercontent.com/lewis-morris/progplot/master/icons/theft.jpg

!wget https://raw.githubusercontent.com/lewis-morris/progplot/master/icons/accident.jpg
img_dict = {"Investigate Person":"./investigate.png",

"Medical Assistance":"./medical.jpg",

"Larceny":"./theft.jpg",

"Motor Vehicle Accident Response":"./accident.jpg"}
bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="bone", # <--------Change 

                     title="Top 4 Crimes by Total Offences from <mindatetime> to <currentdatetime>",dateformat="%Y-%m-%d", 

                     y_label="Offences", 

                     use_top_x=4, display_top_x=4,

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end",

                     convert_bar_to_image=True,  ## <-------- set to true

                     image_dict=img_dict  ## <------ input image dictionary

                    )

bw.test_chart(30)
bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="bone",

                     title="Top 10 Crimes by Total Offences from <mindatetime> to <currentdatetime>",dateformat="%Y-%m-%d", 

                     y_label="Offence", 

                     use_top_x=10, display_top_x=10, 

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end",

                     convert_bar_to_image=True,  ## <-------- set to true

                     image_dict=img_dict  ## <------ input image dictionary

                    )

bw.test_chart(30)
## Just to note - THIS IS SLOW

## writing with images was 1) a headache to code 2) an efficiency nightmare.

## Maybe oneday I will try and speed it up, but for now it is what it is.



bw.write_video()

bw.show_video()