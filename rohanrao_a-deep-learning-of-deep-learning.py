## importing packages

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



import pycountry



from bokeh.io import output_notebook, show

from bokeh.layouts import column, row

from bokeh.models import LinearAxis

from bokeh.palettes import Spectral11

from bokeh.plotting import figure

from bokeh.models.ranges import Range1d



from plotly import graph_objects as go



output_notebook()



## reading data

df = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv", skiprows=[1])

df_2018 = pd.read_csv("/kaggle/input/kaggle-survey-2018/freeFormResponses.csv", skiprows=[1])



## creating numeric feature for salary

dict_salary = dict({"$0-999": 500, "1,000-1,999": 1500, "2,000-2,999": 2500,

                    "3,000-3,999": 3500, "4,000-4,999": 4500, "5,000-7,499": 6250,

                    "7,500-9,999": 8750, "10,000-14,999": 12500, "15,000-19,999": 17500,

                    "20,000-24,999": 22500, "25,000-29,999": 27500, "30,000-39,999": 35000,

                    "40,000-49,999": 45000, "50,000-59,999": 55000, "60,000-69,999": 65000,

                    "70,000-79,999": 75000, "80,000-89,999": 85000, "90,000-99,999": 95000,

                    "100,000-124,999": 112500, "125,000-149,999": 137500, "150,000-199,999": 175000,

                    "200,000-249,999": 225000, "250,000-299,999": 275000, "300,000-500,000": 400000,

                    "> $500,000": 500000})

df["salary"] = df.Q10.map(dict_salary)



## creating numeric feature for expense

dict_expense = dict({"$0 (USD)": 0, "$1-$99": 50, "$100-$999": 550, "$1000-$9,999": 5500,

                     "$10,000-$99,999": 55000, "> $100,000 ($USD)": 100000})

df["expense"] = df.Q11.map(dict_expense)



## tagging practitioner types

df["practitioner_type"] = "Non-DL Practitioner"

df.loc[~(df.Q24_Part_6.isna() & df.Q24_Part_7.isna() & df.Q24_Part_8.isna() & df.Q24_Part_9.isna() & df.Q24_Part_10.isna()), "practitioner_type"] = "DL Practitioner"

df.loc[(df.Q24_Part_1.isna() & df.Q24_Part_2.isna() & df.Q24_Part_3.isna() & df.Q24_Part_4.isna() & df.Q24_Part_5.isna() &

        df.Q24_Part_6.isna() & df.Q24_Part_7.isna() & df.Q24_Part_8.isna() & df.Q24_Part_9.isna() & df.Q24_Part_10.isna()), "practitioner_type"] = "Unknown"



## splitting dataset by type

df_dl = df[df.practitioner_type == "DL Practitioner"]

df_nondl = df[df.practitioner_type == "Non-DL Practitioner"]

df_dl_nondl = df[df.practitioner_type != "Unknown"]

v = figure(plot_width = 700, plot_height = 300, x_range = np.unique(df.practitioner_type.values), title = "Practitioner Distribution")

v.vbar(x = np.unique(df.practitioner_type.values), top = df.practitioner_type.value_counts().sort_index().values, width = 0.9, color = Spectral11[1], legend_label = "# Participants")

v.legend.location = "top_center"

v.legend.click_policy = "hide"



show(v)

v = figure(plot_width = 700, plot_height = 300, x_range = np.unique(df_dl.Q1.values), title = "Age Distribution")

v.vbar(x = np.unique(df_dl.Q1.values), top = df_dl.Q1.value_counts().sort_index().values, width = 0.9, color = Spectral11[1], legend_label = "# DL Practitioners")

v.line(np.unique(df_dl.Q1.values), df_dl.Q1.value_counts().sort_index().values * 100 / df_dl_nondl.Q1.value_counts().sort_index().values, color = Spectral11[10], legend_label = "% DL Practitioners", y_range_name="Percentages")

v.extra_y_ranges = {"Percentages": Range1d(start = 20, end = 80)}

v.add_layout(LinearAxis(y_range_name = "Percentages"), "right")

v.legend.location = "top_right"

v.legend.click_policy = "hide"



show(v)

v = figure(plot_width = 700, plot_height = 400, x_range = np.unique(df_dl.Q4.values), title = "Education Distribution")

v.vbar(x = np.unique(df_dl.Q4.values), top = df_dl.Q4.value_counts().sort_index().values, width = 0.9, color = Spectral11[1], legend_label = "# DL Practitioners")

v.line(np.unique(df_dl.Q4.values), df_dl.Q4.value_counts().sort_index().values * 100 / df_dl_nondl.Q4.value_counts().sort_index().values, color = Spectral11[10], legend_label = "% DL Practitioners", y_range_name="Percentages")

v.extra_y_ranges = {"Percentages": Range1d(start = 20, end = 80)}

v.add_layout(LinearAxis(y_range_name = "Percentages"), "right")

v.legend.location = "top_right"

v.legend.click_policy = "hide"

v.xaxis.major_label_orientation = 145



show(v)

v = figure(plot_width = 700, plot_height = 300, x_range = np.unique(df_dl.Q2.values), title = "Gender Distribution")

v.vbar(x = np.unique(df_dl.Q2.values), top = df_dl.Q2.value_counts().sort_index().values, width = 0.9, color = Spectral11[8], legend_label = "# DL Practitioners")

v.line(np.unique(df_dl.Q2.values), df_dl.Q2.value_counts().sort_index().values * 100 / df_dl_nondl.Q2.value_counts().sort_index().values, color = Spectral11[1], legend_label = "% DL Practitioners", y_range_name="Percentages")

v.extra_y_ranges = {"Percentages": Range1d(start = 20, end = 80)}

v.add_layout(LinearAxis(y_range_name = "Percentages"), "right")

v.legend.location = "top_right"

v.legend.click_policy = "hide"



show(v)

## mapping country codes

def get_country_code(country_name):

    """

    Mapping country name to 3-digit country code.

    """

    

    if country_name == "Russia":

        country_name = "Russian Federation"

    if country_name == "South Korea":

        country_name = "Korea, Republic of"

    if country_name == "Hong Kong (S.A.R.)":

        country_name = "Hong Kong"

    if country_name == "Taiwan":

        country_name = "Taiwan, Province of China"    

    if country_name == "Republic of Korea":

        country_name = "Democratic People's Republic of Korea"

    if country_name == "Iran, Islamic Republic of...":

        country_name = "Iran, Islamic Republic of"

    

    country_data = pycountry.countries.get(name=country_name)

    

    if country_data is None:

        country_data = pycountry.countries.get(official_name=country_name)

    

    if country_data is None:

        return np.nan

    return country_data.alpha_3



df_dl_country = pd.DataFrame(df_dl.Q3.value_counts()).reset_index().rename(columns={"index": "country", "Q3": "dl_count"})

df_dl_country["country_code"] = df_dl_country.country.apply(lambda x: get_country_code(x))



f = go.Figure(data=go.Choropleth(

    locations=df_dl_country.country_code,

    z=df_dl_country.dl_count,

    locationmode="ISO-3",

    text=df_dl_country.country,

    colorscale="Blues",

    autocolorscale=False,

    marker_line_width=0.5,

    colorbar_tickprefix="#",

    colorbar_title="# DL Practitioners"

))



f.update_layout(

    title={

        "text": "Global # DL Practitioners",

        "y":0.9,

        "x":0.475,

        "xanchor": "center",

        "yanchor": "top"}

)



f.show()
df_dl_country.sort_values("dl_count", ascending=False).head(10)
df_dl_nondl_country = pd.DataFrame(df_dl_nondl.Q3.value_counts()).reset_index().rename(columns={"index": "country", "Q3": "dl_nondl_count"})

df_dl_nondl_country["country_code"] = df_dl_nondl_country.country.apply(lambda x: get_country_code(x))

df_dl_nondl_country = df_dl_nondl_country.merge(df_dl_country[["country_code", "dl_count"]], how="left", on="country_code")

df_dl_nondl_country["dl_percentage"] = round(df_dl_nondl_country.dl_count * 100 / df_dl_nondl_country.dl_nondl_count)



f = go.Figure(data=go.Choropleth(

    locations=df_dl_nondl_country.country_code,

    z=df_dl_nondl_country.dl_percentage,

    locationmode="ISO-3",

    text=df_dl_nondl_country.country,

    colorscale="Blues",

    autocolorscale=False,

    marker_line_width=0.5,

    colorbar_ticksuffix="%",

    colorbar_title="% DL Practitioners"

))



f.update_layout(

    title={

        "text": "Global % DL Practitioners",

        "y":0.9,

        "x":0.475,

        "xanchor": "center",

        "yanchor": "top"}

)



f.show()
df_dl_nondl_country.sort_values("dl_percentage", ascending=False).head(10)
df_dl_role = pd.DataFrame(df_dl.Q5.value_counts()).reset_index().rename(columns={"index": "Role", "Q5": "dl_count"})

df_dl_nondl_role = pd.DataFrame(df_dl_nondl.Q5.value_counts()).reset_index().rename(columns={"index": "Role", "Q5": "dl_nondl_count"})



df_dl_role = df_dl_role.merge(df_dl_nondl_role)

df_dl_role["dl_percentage"] = df_dl_role.dl_count * 100 / df_dl_role.dl_nondl_count

df_dl_role.sort_values("dl_percentage", ascending=False, inplace=True)



f = figure(x_range=df_dl_role.Role.values, plot_width=700, plot_height=300, title="Role Distribution")

f.vbar(x=df_dl_role.Role.values, top=df_dl_role.dl_count.values, width=0.9, color=Spectral11[9], legend_label="# DL Practitioners")

f.line(df_dl_role.Role.values, df_dl_role.dl_percentage.values, color=Spectral11[1], legend_label="% DL Practitioners", y_range_name="Percentages")

f.extra_y_ranges = {"Percentages": Range1d(start=20, end=80)}

f.add_layout(LinearAxis(y_range_name="Percentages"), "right")

f.legend.location="top_right"

f.legend.click_policy="hide"

f.xaxis.major_label_orientation=145

show(f)
import seaborn as sns

f = sns.FacetGrid(df_dl_nondl, col="practitioner_type")

f.map(plt.hist, "salary")

f.add_legend()

plt.show()
bp = sns.boxplot(x="practitioner_type", y="salary", data=df_dl_nondl, palette="Set2").set_title("Salary Distribution")
import seaborn as sns

f = sns.FacetGrid(df_dl_nondl, col="practitioner_type")

f.map(plt.hist, "expense")

f.add_legend()

plt.show()
bp = sns.boxplot(x="practitioner_type", y="expense", data=df_dl_nondl, palette="Set3").set_title("Expense Distribution")
media_list = {

    "Twitter": sum(~df_dl.Q12_Part_1.isna()) * 100 / df_dl.shape[0],

    "HackerNews": sum(~df_dl.Q12_Part_2.isna()) * 100 / df_dl.shape[0],

    "Reddit": sum(~df_dl.Q12_Part_3.isna()) * 100 / df_dl.shape[0],

    "Kaggle": sum(~df_dl.Q12_Part_4.isna()) * 100 / df_dl.shape[0],

    "Forums": sum(~df_dl.Q12_Part_5.isna()) * 100 / df_dl.shape[0],

    "YouTube": sum(~df_dl.Q12_Part_6.isna()) * 100 / df_dl.shape[0],

    "Podcasts": sum(~df_dl.Q12_Part_7.isna()) * 100 / df_dl.shape[0],

    "Blogs": sum(~df_dl.Q12_Part_8.isna()) * 100 / df_dl.shape[0],

    "Journals": sum(~df_dl.Q12_Part_9.isna()) * 100 / df_dl.shape[0],

    "Slack": sum(~df_dl.Q12_Part_10.isna()) * 100 / df_dl.shape[0],

    "None": sum(~df_dl.Q12_Part_11.isna()) * 100 / df_dl.shape[0],

    "Other": sum(~df_dl.Q12_Part_12.isna()) * 100 / df_dl.shape[0]

}



df_dl_media = pd.DataFrame.from_dict(media_list, orient="index", columns=["media_percentage"]).reset_index().rename(columns={"index": "media"})

df_dl_media.sort_values("media_percentage", ascending=False, inplace=True)



platform_list = {

    "Udacity": sum(~df_dl.Q13_Part_1.isna()) * 100 / df_dl.shape[0],

    "Coursera": sum(~df_dl.Q13_Part_2.isna()) * 100 / df_dl.shape[0],

    "edX": sum(~df_dl.Q13_Part_3.isna()) * 100 / df_dl.shape[0],

    "DataCamp": sum(~df_dl.Q13_Part_4.isna()) * 100 / df_dl.shape[0],

    "DataQuest": sum(~df_dl.Q13_Part_5.isna()) * 100 / df_dl.shape[0],

    "Kaggle": sum(~df_dl.Q13_Part_6.isna()) * 100 / df_dl.shape[0],

    "Fast.ai": sum(~df_dl.Q13_Part_7.isna()) * 100 / df_dl.shape[0],

    "Udemy": sum(~df_dl.Q13_Part_8.isna()) * 100 / df_dl.shape[0],

    "LinkedIn": sum(~df_dl.Q13_Part_9.isna()) * 100 / df_dl.shape[0],

    "University": sum(~df_dl.Q13_Part_10.isna()) * 100 / df_dl.shape[0],

    "None": sum(~df_dl.Q13_Part_11.isna()) * 100 / df_dl.shape[0],

    "Other": sum(~df_dl.Q13_Part_12.isna()) * 100 / df_dl.shape[0]

}



df_dl_platform = pd.DataFrame.from_dict(platform_list, orient="index", columns=["platform_percentage"]).reset_index().rename(columns={"index": "platform"})

df_dl_platform.sort_values("platform_percentage", ascending=False, inplace=True)



f1 = figure(x_range=df_dl_media.media, plot_width=700, plot_height=400, title="Media Distribution")

f1.vbar(x=df_dl_media.media, top=df_dl_media.media_percentage, width=0.9, color=Spectral11[6], legend_label="% Media Usage")

f1.legend.location="top_right"

f1.legend.click_policy="hide"

f1.xaxis.major_label_orientation=145



f2 = figure(x_range=df_dl_platform.platform, plot_width=700, plot_height=400, title="Platform Distribution")

f2.vbar(x=df_dl_platform.platform, top=df_dl_platform.platform_percentage, width=0.9, color=Spectral11[6], legend_label="% Platform Usage")

f2.legend.location="top_right"

f2.legend.click_policy="hide"

f2.xaxis.major_label_orientation=145



show(column(f1, f2))
ide_list = {

    "Jupyter": sum(~df_dl.Q16_Part_1.isna()) * 100 / df_dl.shape[0],

    "RStudio": sum(~df_dl.Q16_Part_2.isna()) * 100 / df_dl.shape[0],

    "PyCharm": sum(~df_dl.Q16_Part_3.isna()) * 100 / df_dl.shape[0],

    "Atom": sum(~df_dl.Q16_Part_4.isna()) * 100 / df_dl.shape[0],

    "MATLAB": sum(~df_dl.Q16_Part_5.isna()) * 100 / df_dl.shape[0],

    "VisualStudio": sum(~df_dl.Q16_Part_6.isna()) * 100 / df_dl.shape[0],

    "Spyder": sum(~df_dl.Q16_Part_7.isna()) * 100 / df_dl.shape[0],

    "Vim": sum(~df_dl.Q16_Part_8.isna()) * 100 / df_dl.shape[0],

    "Notepad++": sum(~df_dl.Q16_Part_9.isna()) * 100 / df_dl.shape[0],

    "SublimeText": sum(~df_dl.Q16_Part_10.isna()) * 100 / df_dl.shape[0],

    "None": sum(~df_dl.Q16_Part_11.isna()) * 100 / df_dl.shape[0],

    "Other": sum(~df_dl.Q16_Part_12.isna()) * 100 / df_dl.shape[0]

}



df_dl_ide = pd.DataFrame.from_dict(ide_list, orient="index", columns=["ide_percentage"]).reset_index().rename(columns={"index": "ide"})

df_dl_ide.sort_values("ide_percentage", ascending=False, inplace=True)



notebook_list = {

    "KaggleNotebooks": sum(~df_dl.Q17_Part_1.isna()) * 100 / df_dl.shape[0],

    "GoogleColab": sum(~df_dl.Q17_Part_2.isna()) * 100 / df_dl.shape[0],

    "MicrosoftAzureNotebooks": sum(~df_dl.Q17_Part_3.isna()) * 100 / df_dl.shape[0],

    "GoogleCloudNotebooks": sum(~df_dl.Q17_Part_4.isna()) * 100 / df_dl.shape[0],

    "Paperspace/Gradient": sum(~df_dl.Q17_Part_5.isna()) * 100 / df_dl.shape[0],

    "FloydHub": sum(~df_dl.Q17_Part_6.isna()) * 100 / df_dl.shape[0],

    "Binder/JupyterHub": sum(~df_dl.Q17_Part_7.isna()) * 100 / df_dl.shape[0],

    "IBMWatsonStudio": sum(~df_dl.Q17_Part_8.isna()) * 100 / df_dl.shape[0],

    "CodeOcean": sum(~df_dl.Q17_Part_9.isna()) * 100 / df_dl.shape[0],

    "AWSNotebooks": sum(~df_dl.Q17_Part_10.isna()) * 100 / df_dl.shape[0],

    "None": sum(~df_dl.Q17_Part_11.isna()) * 100 / df_dl.shape[0],

    "Other": sum(~df_dl.Q17_Part_12.isna()) * 100 / df_dl.shape[0]

}



df_dl_notebook = pd.DataFrame.from_dict(notebook_list, orient="index", columns=["notebook_percentage"]).reset_index().rename(columns={"index": "notebook"})

df_dl_notebook.sort_values("notebook_percentage", ascending=False, inplace=True)



language_list = {

    "Python": sum(~df_dl.Q18_Part_1.isna()) * 100 / df_dl.shape[0],

    "R": sum(~df_dl.Q18_Part_2.isna()) * 100 / df_dl.shape[0],

    "SQL": sum(~df_dl.Q18_Part_3.isna()) * 100 / df_dl.shape[0],

    "C": sum(~df_dl.Q18_Part_4.isna()) * 100 / df_dl.shape[0],

    "C++": sum(~df_dl.Q18_Part_5.isna()) * 100 / df_dl.shape[0],

    "Java": sum(~df_dl.Q18_Part_6.isna()) * 100 / df_dl.shape[0],

    "Javascript": sum(~df_dl.Q18_Part_7.isna()) * 100 / df_dl.shape[0],

    "TypeScript": sum(~df_dl.Q18_Part_8.isna()) * 100 / df_dl.shape[0],

    "Bash": sum(~df_dl.Q18_Part_9.isna()) * 100 / df_dl.shape[0],

    "MATLAB": sum(~df_dl.Q18_Part_10.isna()) * 100 / df_dl.shape[0],

    "None": sum(~df_dl.Q18_Part_11.isna()) * 100 / df_dl.shape[0],

    "Other": sum(~df_dl.Q18_Part_12.isna()) * 100 / df_dl.shape[0]

}



df_dl_language = pd.DataFrame.from_dict(language_list, orient="index", columns=["language_percentage"]).reset_index().rename(columns={"index": "language"})

df_dl_language.sort_values("language_percentage", ascending=False, inplace=True)



f1 = figure(x_range=df_dl_ide.ide, plot_width=700, plot_height=400, title="IDE Distribution")

f1.vbar(x=df_dl_ide.ide, top=df_dl_ide.ide_percentage, width=0.9, color=Spectral11[3], legend_label="% IDE Usage")

f1.legend.location="top_right"

f1.legend.click_policy="hide"

f1.xaxis.major_label_orientation=145



f2 = figure(x_range=df_dl_notebook.notebook, plot_width=700, plot_height=400, title="Notebook Distribution")

f2.vbar(x=df_dl_notebook.notebook, top=df_dl_notebook.notebook_percentage, width=0.9, color=Spectral11[3], legend_label="% Notebook Usage")

f2.legend.location="top_right"

f2.legend.click_policy="hide"

f2.xaxis.major_label_orientation=145



f3 = figure(x_range=df_dl_language.language, plot_width=700, plot_height=400, title="Language Distribution")

f3.vbar(x=df_dl_language.language, top=df_dl_language.language_percentage, width=0.9, color=Spectral11[3], legend_label="% Language Usage")

f3.legend.location="top_right"

f3.legend.click_policy="hide"

f3.xaxis.major_label_orientation=145



show(column(f1, f2, f3))
hardware_list = {

    "CPUs": sum(~df_dl.Q21_Part_1.isna()) * 100 / df_dl.shape[0],

    "GPUs": sum(~df_dl.Q21_Part_2.isna()) * 100 / df_dl.shape[0],

    "TPUs": sum(~df_dl.Q21_Part_3.isna()) * 100 / df_dl.shape[0],

    "None": sum(~df_dl.Q21_Part_4.isna()) * 100 / df_dl.shape[0],

    "Other": sum(~df_dl.Q21_Part_5.isna()) * 100 / df_dl.shape[0]

}



df_dl_hardware = pd.DataFrame.from_dict(hardware_list, orient="index", columns=["hardware_percentage"]).reset_index().rename(columns={"index": "hardware"})

df_dl_hardware.sort_values("hardware_percentage", ascending=False, inplace=True)



model_list = {

    "DenseNN": sum(~df_dl.Q24_Part_6.isna()) * 100 / df_dl.shape[0],

    "CNN": sum(~df_dl.Q24_Part_7.isna()) * 100 / df_dl.shape[0],

    "GAN": sum(~df_dl.Q24_Part_8.isna()) * 100 / df_dl.shape[0],

    "RNN": sum(~df_dl.Q24_Part_9.isna()) * 100 / df_dl.shape[0],

    "TransformerNetworks": sum(~df_dl.Q24_Part_10.isna()) * 100 / df_dl.shape[0],

    "None": sum(~df_dl.Q24_Part_11.isna()) * 100 / df_dl.shape[0],

    "Other": sum(~df_dl.Q24_Part_12.isna()) * 100 / df_dl.shape[0]

}



df_dl_model = pd.DataFrame.from_dict(model_list, orient="index", columns=["model_percentage"]).reset_index().rename(columns={"index": "model"})

df_dl_model.sort_values("model_percentage", ascending=False, inplace=True)



tool_list = {

    "AutoAugmentation": sum(~df_dl.Q25_Part_1.isna()) * 100 / df_dl.shape[0],

    "AutoFeatureSelection": sum(~df_dl.Q25_Part_2.isna()) * 100 / df_dl.shape[0],

    "AutoModelSelection": sum(~df_dl.Q25_Part_3.isna()) * 100 / df_dl.shape[0],

    "AutoModelArchitectureSearch": sum(~df_dl.Q25_Part_4.isna()) * 100 / df_dl.shape[0],

    "AutoHyperparameterTuning": sum(~df_dl.Q25_Part_5.isna()) * 100 / df_dl.shape[0],

    "AutoML": sum(~df_dl.Q25_Part_6.isna()) * 100 / df_dl.shape[0],

    "None": sum(~df_dl.Q25_Part_7.isna()) * 100 / df_dl.shape[0],

    "Other": sum(~df_dl.Q25_Part_8.isna()) * 100 / df_dl.shape[0]

}



df_dl_tool = pd.DataFrame.from_dict(tool_list, orient="index", columns=["tool_percentage"]).reset_index().rename(columns={"index": "tool"})

df_dl_tool.sort_values("tool_percentage", ascending=False, inplace=True)



f1 = figure(x_range=df_dl_hardware.hardware, plot_width=700, plot_height=400, title="Hardware Distribution")

f1.vbar(x=df_dl_hardware.hardware, top=df_dl_hardware.hardware_percentage, width=0.9, color=Spectral11[0], legend_label="% Hardware Usage")

f1.legend.location="top_right"

f1.legend.click_policy="hide"

f1.xaxis.major_label_orientation=145



f2 = figure(x_range=df_dl_model.model, plot_width=700, plot_height=400, title="Model Distribution")

f2.vbar(x=df_dl_model.model, top=df_dl_model.model_percentage, width=0.9, color=Spectral11[0], legend_label="% Model Usage")

f2.legend.location="top_right"

f2.legend.click_policy="hide"

f2.xaxis.major_label_orientation=145



f3 = figure(x_range=df_dl_tool.tool, plot_width=700, plot_height=400, title="Tool Distribution")

f3.vbar(x=df_dl_tool.tool, top=df_dl_tool.tool_percentage, width=0.9, color=Spectral11[0], legend_label="% Tool Usage")

f3.legend.location="top_right"

f3.legend.click_policy="hide"

f3.xaxis.major_label_orientation=145



show(column(f1, f2, f3))
cv_list = {

    "Regular Methods": sum(~df_dl.Q26_Part_1.isna()) * 100 / df_dl.shape[0],

    "Image Segmentation": sum(~df_dl.Q26_Part_2.isna()) * 100 / df_dl.shape[0],

    "Object Detection": sum(~df_dl.Q26_Part_3.isna()) * 100 / df_dl.shape[0],

    "Image Classification": sum(~df_dl.Q26_Part_4.isna()) * 100 / df_dl.shape[0],

    "Image Generation": sum(~df_dl.Q26_Part_5.isna()) * 100 / df_dl.shape[0],

    "None": sum(~df_dl.Q26_Part_6.isna()) * 100 / df_dl.shape[0],

    "Other": sum(~df_dl.Q26_Part_7.isna()) * 100 / df_dl.shape[0]

}



df_dl_cv = pd.DataFrame.from_dict(cv_list, orient="index", columns=["cv_percentage"]).reset_index().rename(columns={"index": "cv"})

df_dl_cv.sort_values("cv_percentage", ascending=False, inplace=True)



f = figure(x_range=df_dl_cv.cv, plot_width=700, plot_height=400, title="Computer Vision techniques Distribution")

f.vbar(x=df_dl_cv.cv, top=df_dl_cv.cv_percentage, width=0.9, color=Spectral11[8], legend_label="% Computer Vision techniques Usage")

f.legend.location="top_right"

f.legend.click_policy="hide"

f.xaxis.major_label_orientation=145



show(f)
nlp_list = {

    "Word Embeddings": sum(~df_dl.Q27_Part_1.isna()) * 100 / df_dl.shape[0],

    "Sequence Models": sum(~df_dl.Q27_Part_2.isna()) * 100 / df_dl.shape[0],

    "Contextualized Embeddings": sum(~df_dl.Q27_Part_3.isna()) * 100 / df_dl.shape[0],

    "Language Models": sum(~df_dl.Q27_Part_4.isna()) * 100 / df_dl.shape[0],

    "None": sum(~df_dl.Q27_Part_5.isna()) * 100 / df_dl.shape[0],

    "Other": sum(~df_dl.Q27_Part_6.isna()) * 100 / df_dl.shape[0]

}



df_dl_nlp = pd.DataFrame.from_dict(nlp_list, orient="index", columns=["nlp_percentage"]).reset_index().rename(columns={"index": "nlp"})

df_dl_nlp.sort_values("nlp_percentage", ascending=False, inplace=True)



f = figure(x_range=df_dl_nlp.nlp, plot_width=700, plot_height=400, title="NLP techniques Distribution")

f.vbar(x=df_dl_nlp.nlp, top=df_dl_nlp.nlp_percentage, width=0.9, color=Spectral11[8], legend_label="% NLP techniques Usage")

f.legend.location="top_right"

f.legend.click_policy="hide"

f.xaxis.major_label_orientation=145



show(f)