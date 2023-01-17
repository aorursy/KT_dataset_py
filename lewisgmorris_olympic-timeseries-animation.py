import pandas as pd

import seaborn as sns

import numpy as np
#read the data



df = pd.read_csv("/kaggle/input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")

df
df = df[df["Season"]=="Summer"]
#read the region data

noc = pd.read_csv("/kaggle/input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv")

noc
#merge the two 

olymp_df = df.merge(noc,left_on="NOC",right_on="NOC")

olymp_df
#select categories we want 

olymp_df = olymp_df[["region","Age","Height","Weight","Year","Sport","Medal"]]

olymp_df
#fix the medal column - using get dummies you can do this easily.



olymp_df = pd.concat([olymp_df,pd.get_dummies(olymp_df["Medal"])],axis=1)



olymp_df["Total"] = olymp_df["Bronze"] + olymp_df["Gold"] + olymp_df["Silver"]



olymp_df.drop("Medal",axis=1, inplace=True)



olymp_df
#convert year to a date 

olymp_df["Year"] = pd.to_datetime(olymp_df["Year"],format="%Y")

olymp_df
!pip install progplot
#import barwriter

from progplot import BarWriter
#create the barwriter object

bw = BarWriter()
help(bw.set_data)
olymp_df
bw.set_data(data=olymp_df, category_col="region", timeseries_col="Year", value_col="Total", groupby_agg="sum", resample_agg="sum", output_agg="cumsum", resample = "4y")
help(bw.set_display_settings)
bw.set_display_settings(time_in_seconds=30, video_file_name = "total_medals_by_country.mp4")
help(bw.set_chart_options)
bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="Pastel1", 

                     title="Top 10 Countries by Total Medals from <mindatetime> to <currentdatetime>",dateformat="%Y", 

                     y_label="State", 

                     use_top_x=20, display_top_x=10,

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end")

bw.test_chart(30)
bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="Pastel1", 

                     title="Top 5 Countries by Total Medals from <mindatetime> to <currentdatetime>",dateformat="%Y", 

                     y_label="State", 

                     use_top_x=20, display_top_x=5,

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end",

                     squeeze_lower_x="1000") # <----------- HERE either enter the percentace lower than the minimum data value you want the x value to be. OR the absolute value i.e 1000.

bw.test_chart(14)
bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="bone", # <------- change palette 

                     title="Top 5 Countries by Total Medals from <mindatetime> to <currentdatetime>",dateformat="%Y", 

                     y_label="State", 

                     use_top_x=20, display_top_x=5,

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end",

                     sort=False) # <------- set to stop the categories re-arranging.

bw.test_chart(14)
bw.write_video()

bw.show_video()
bw.create_gif()

bw.show_gif()
olymp_df = olymp_df.dropna()

olymp_df
#set the data - mean groupby / mean resample / rolling output agg to smooth the results.

bw.set_data(data=olymp_df, category_col="region", timeseries_col="Year", value_col="Age", groupby_agg="mean", resample_agg="mean", output_agg="4rolling", resample = "4y")



#similar display as normal

bw.set_display_settings(time_in_seconds=45, video_file_name = "mean_age_by_country.mp4")

bw.set_chart_options(x_tick_format="{:,.2f}",

                     palette="Pastel1",

                     title="Top 10 Rolling Mean Age <rollingdatetime> to <currentdatetime>",dateformat="%Y",

                     y_label="State",

                     use_top_x=10, display_top_x=10,

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end")

bw.test_chart(20)

bw.set_chart_options(x_tick_format="{:,.2f}",

                     palette="Pastel1",

                     title="Top 10 Rolling Mean Age <rollingdatetime> to <currentdatetime>",dateformat="%Y",

                     y_label="State",

                     use_top_x=10, display_top_x=10,

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end",

                     squeeze_lower_x="16")  # <------------



bw.test_chart(20)
bw.write_video()

bw.show_video()