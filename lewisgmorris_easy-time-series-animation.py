import pandas as pd

import seaborn as sns

import numpy as np
!pip install progplot
#import barwriter

from progplot import BarWriter
#create the barwriter object

bw = BarWriter()
df = pd.read_csv("/kaggle/input/corona-virus-report/usa_county_wise.csv")

df["Date"] = pd.to_datetime(df["Date"])

df
help(bw.set_data)
bw.set_data(data=df, category_col="Province_State", timeseries_col="Date", value_col="Deaths", groupby_agg="sum", resample="1d",resample_agg="sum", output_agg=None)
help(bw.set_display_settings)
bw.set_display_settings(time_in_seconds=45, video_file_name = "deathsbystate.mp4")
help(bw.set_chart_options)
bw.set_chart_options(use_data_labels=None)
bw.test_chart(100)
#We've set the format of the ticks, the 



bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="Pastel1", 

                     title="Top 15 States by Total Deaths from <mindatetime> to <currentdatetime>",dateformat="%Y-%d-%m", 

                     y_label="State", 

                     use_top_x=30, display_top_x=15,

                     border_size=2, border_colour=(0.3,0.3,0.3),

                     font_scale=1.3,

                     use_data_labels="end")

bw.test_chart(100)



from progplot import palettes

palettes()
bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="bone", ### <-------------- Just change this value

                     title="Top 15 States by Total Deaths from <mindatetime> to <currentdatetime>",dateformat="%Y-%d-%m", 

                     y_label="State", 

                     use_top_x=30, display_top_x=15,

                     border_size=2, border_colour=(0.12,0.12,0.12),

                     font_scale=1.3,

                     use_data_labels="end")

bw.test_chart()
help(bw.write_video)
bw.write_video(limit_frames=10)

bw.show_video()
bw.write_video()

bw.show_video()
bw.create_gif()
bw.show_gif()
bw.set_data(data=df, category_col="Province_State", timeseries_col="Date", value_col="Confirmed", groupby_agg="sum", resample="1d",resample_agg="sum", output_agg=None)  # <--- value_col



bw.set_display_settings(time_in_seconds=45, video_file_name = "casesbystate.mp4") # <--- video_file_name



bw.set_chart_options(x_tick_format="{:,.0f}", dateformat="%Y-%m-%d", 

                     palette="copper", 

                     title="Top 15 States by Total Cases <mindatetime> to <currentdatetime>", y_label="State", # <--- title

                     use_top_x=30, display_top_x=15,

                     border_size=2, border_colour=(0.12,0.12,0.12),

                     font_scale=1.6, title_font_size=18,x_label_font_size=16,

                     use_data_labels="end")  

bw.test_chart(100)
bw.write_video()

bw.show_video()
df_country = pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")

df_country["Date"] = pd.to_datetime(df_country["Date"])

df_country
# download a zip of flag images

!wget "https://flagpedia.net/data/flags/w320.zip"
# unzip images

import zipfile

with zipfile.ZipFile("w320.zip", 'r') as zip_ref:

    zip_ref.extractall("./icons/flags/")
codes = pd.read_html("https://www.iban.com/country-codes",attrs = {'id': 'myTable'})

codes[0]
df_country = df_country.merge(codes[0],left_on="Country/Region", right_on="Country")

df_country = df_country[["Country/Region","Date","Confirmed","Deaths","Alpha-2 code"]]

df_country
countries = list(df_country.loc[:,"Country/Region"].unique())

codes = list(df_country.loc[:,"Alpha-2 code"].unique())

                                  
image_dict = {country:f"./icons/flags/{str(code).lower()}.png" for country,code in zip(countries,codes)}

image_dict
bw.set_data(df_country, "Country/Region", "Date", "Deaths", resample="1d", groupby_agg="sum", resample_agg="sum",output_agg=None)



bw.set_display_settings(time_in_seconds=45, video_file_name = "deathsbycountrywithflag.mp4")



bw.set_chart_options(x_tick_format="{:,.0f}", dateformat="%Y-%m-%d", 

                     palette="summer", 

                     title="Top 15 Countries by Total Deaths <mindatetime> to <currentdatetime>",

                     use_top_x=15, display_top_x=15,

                     border_size=2, border_colour=(0.12,0.12,0.12),

                     font_scale=1.6,

                     use_data_labels="end", convert_bar_to_image=True,image_dict=image_dict)  # <--- Add image_dict and set convert_bar_to_image=True

bw.test_chart(100)
bw.write_video()

bw.show_video()
df_mean = df.dropna()

df_mean
df_mean[(df_mean["Admin2"]=="Autauga") & (df_mean["Province_State"]=="Alabama")]
df_mean = df_mean.groupby(["Province_State","Date"]).sum().reset_index()

df_mean
## used to reverse the cumsum 



df_new = None

df_list = []



for i, (x,y) in enumerate(df_mean.groupby(["Province_State"])):

    y.reset_index(drop=True, inplace=True)

    

    print(f"\r{i}/{len(df_mean['Province_State'].drop_duplicates())}",end="")



    for itm in np.arange(len(y)-1,0,-1):

        y.loc[itm,"Deaths"] -= y.loc[itm-1,"Deaths"]

        y.loc[itm,"Confirmed"] -= y.loc[itm-1,"Confirmed"]

    df_list.append(y)

    

df_new = pd.concat(df_list)
df_new.tail(30)
bw.set_data(data=df_new, category_col="Province_State", timeseries_col="Date", value_col="Deaths", groupby_agg="sum", resample="1d",resample_agg="sum", output_agg="7rolling")





bw.set_display_settings(time_in_seconds=45, video_file_name = "rolling_daily.mp4")

bw.set_chart_options(x_tick_format="{:,.2f}", dateformat="%Y-%m-%d", 

                     palette="bone", 

                     title="Daily Rolling Average Deaths from <rollingdatetime> to <currentdatetime>", y_label="State", 

                     use_top_x=15, display_top_x=15,

                     border_size=2, border_colour=(0.12,0.12,0.12),

                     font_scale=1.3,

                     use_data_labels="end")



bw.test_chart(80)
bw.write_video()
help(bw.set_chart_options)
bw.set_data(data=df, category_col="Province_State", timeseries_col="Date", value_col="Deaths", groupby_agg="sum", resample="1d",resample_agg="sum", output_agg=None)

bw.set_display_settings(time_in_seconds=45, video_file_name = "test.mp4")


bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="Pastel1", 

                     title="Top 15 States by Total Deaths from <mindatetime> to <currentdatetime>",dateformat="%Y-%d-%m", 

                     y_label="State", 

                     use_top_x=30, display_top_x=15,

                     border_size=6, border_colour=(0.3,0.8,0.3), # <-- border size / colour

                     font_scale=1.3,

                     use_data_labels="end")

bw.test_chart(100)


bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="Pastel1", 

                     title="Top 15 States by Total Deaths from <mindatetime> to <currentdatetime>",dateformat="%Y-%d-%m", 

                     y_label="State", 

                     use_top_x=15, display_top_x=15,

                     border_size=3, border_colour=(0.3,0.3,0.3), # <-- border size / colour

                     font_scale=1.3,

                     use_data_labels="base")

bw.test_chart(100)


bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="magma", 

                     title="Top 15 States by Total Deaths from <mindatetime> to <currentdatetime>",dateformat="%Y-%d-%m", 

                     y_label="State", 

                     use_top_x=15, display_top_x=15,

                     border_size=3, border_colour=(0.3,0.3,0.3),

                     font_scale=.7,

                     use_data_labels="base",

                     seaborn_style="darkgrid",

                     seaborn_context="paper") # <-- paper

bw.test_chart(100)


bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="tab10_r", 

                     title="Top 15 States by Total Deaths from <mindatetime> to <currentdatetime>",dateformat="%Y-%d-%m", 

                     y_label="State", 

                     use_top_x=15, display_top_x=15,

                     border_size=3, border_colour=(0.3,0.3,0.3), # <-- border size / colour

                     font_scale=1.3,

                     use_data_labels="end",

                     sort=False)

bw.test_chart(100)


bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="gist_earth_r", 

                     title="Top 15 States by Total Deaths from <mindatetime> to <currentdatetime>",dateformat="%Y-%d-%m", 

                     y_label="State", 

                     use_top_x=15, display_top_x=15,

                     border_size=3, border_colour=(0.3,0.3,0.3), # <-- border size / colour

                     font_scale=1.5,

                     use_data_labels="end",

                     figsize=(10,12),

                     dpi=120)



bw.test_chart(100)


bw.set_chart_options(x_tick_format="{:,.0f}",

                     palette="terrain", 

                     title="Top 15 States by Total Deaths from <mindatetime> to <currentdatetime>",dateformat="%Y-%d-%m", 

                     y_label="State", 

                     use_top_x=15, display_top_x=15,

                     border_size=3, border_colour=(0.3,0.3,0.3), # <-- border size / colour

                     font_scale=1.5,

                     use_data_labels="end",

                     figsize=(20,10),

                     dpi=85)



bw.test_chart(100)