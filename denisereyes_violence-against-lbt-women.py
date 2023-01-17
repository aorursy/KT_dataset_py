!pip install -q bokeh==2.2.1
import bokeh.io
bokeh.io.output_notebook()
import bokeh
from bokeh.io import output_notebook,output_file,show
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row, column
from bokeh.models import Div
from bokeh.transform import dodge
import pandas as pd
import numpy as np
lbtwomen = pd.read_csv("../input/queerwomenviolence/violence_against_LBTwomen_EU.csv")
lbtwomen.head()
lbtwomen_burtin = lbtwomen[(lbtwomen["answer"]=="Yes")&((lbtwomen["question_label"]=="Avoid holding hands in public with a same-sex partner for fear of being assaulted, threatened of harassed?")|(lbtwomen["question_label"]=="Avoid locations for fear of being assaulted, threatened or harassed because you are L, G, B, or T?")|(lbtwomen["question_label"]=="Physically/sexually attacked or threatened with violence at home or elsewhere in the last 5 years for any reason?")|(lbtwomen["question_label"]=="Personally harassed by someone or a group for any reason in the last 5 years in a way that really annoyed, offended or upset you?"))]

lbtwomen_burtin_L = pd.DataFrame(lbtwomen_burtin[lbtwomen_burtin["subset"]=="Lesbian"]["percentage"].groupby(lbtwomen_burtin.question_label).mean().reset_index())
lbtwomen_burtin_L["Category"] = "Lesbian"

lbtwomen_burtin_B = pd.DataFrame(lbtwomen_burtin[lbtwomen_burtin["subset"]=="Bisexual women"]["percentage"].groupby(lbtwomen_burtin.question_label).mean().reset_index())
lbtwomen_burtin_B["Category"] = "Bisexual Woman"

lbtwomen_burtin_T = pd.DataFrame(lbtwomen_burtin[lbtwomen_burtin["subset"]=="Woman with a transsexual past"]["percentage"].groupby(lbtwomen_burtin.question_label).mean().reset_index())
lbtwomen_burtin_T["Category"] = "Transgender Woman"

lbtwomen_burtin_L["percentage"] = lbtwomen_burtin_L["percentage"].round(decimals=0)
lbtwomen_burtin_B["percentage"] = lbtwomen_burtin_B["percentage"].round(decimals=0)
lbtwomen_burtin_T["percentage"] = lbtwomen_burtin_T["percentage"].round(decimals=0)

lbtwomen_burtin_L["q"]=["Avoided Holding Hands (Yes%)","Avoided Locations (Yes%)","Personal Harassment (Yes%)","Physical/Sexual Attack (Yes%)"]
lbtwomen_burtin_B["q"]=["Avoided Holding Hands (Yes%)","Avoided Locations (Yes%)","Personal Harassment (Yes%)","Physical/Sexual Attack (Yes%)"]
lbtwomen_burtin_T["q"]=["Avoided Locations (Yes%)","Personal Harassment (Yes%)","Physical/Sexual Attack (Yes%)"]

v = figure(x_range=lbtwomen_burtin_L["q"],y_range=(0,100),plot_height=300,title="", toolbar_location=None,tools="")
v.vbar(x=dodge("q",-0.25,range=v.x_range),top="percentage",width=0.2,source=lbtwomen_burtin_L,color="#FF6565",legend_label="Lesbian Women")
v.vbar(x=dodge("q",0.0,range=v.x_range),top="percentage",width=0.2,source=lbtwomen_burtin_B,color="#C264A5",legend_label="Bisexual Women")
v.vbar(x=dodge("q",0.25,range=v.x_range),top="percentage",width=0.2,source=lbtwomen_burtin_T,color="#FD8ABD",legend_label="Transgender Women")

v.plot_width = 800
v.x_range.range_padding = 0.1
v.xgrid.grid_line_color = None
v.legend.location = "top_left"
v.legend.orientation = "horizontal"



show(column(Div(text="<h1>Violence and Fear Experienced by LBT Women (EU - 2012)<h1>"),v))
lbtwomen_pie = lbtwomen[(lbtwomen["question_label"]=="MOST SERIOUS physical / sexual attack or threat of violence - Gender of the perpetrator(s)?")&((lbtwomen["answer"]=="Male")|(lbtwomen["answer"]=="Both male and female"))]
lbtwomen_pie_L_c = pd.DataFrame(lbtwomen_pie[lbtwomen_pie["subset"]=="Lesbian"]["percentage"].groupby(lbtwomen_pie.CountryCode).sum().reset_index())
lbtwomen_pie_B_c = pd.DataFrame(lbtwomen_pie[lbtwomen_pie["subset"]=="Bisexual women"]["percentage"].groupby(lbtwomen_pie.CountryCode).sum().reset_index())
lbtwomen_pie_T_c = pd.DataFrame(lbtwomen_pie[lbtwomen_pie["subset"]=="Woman with a transsexual past"]["percentage"].groupby(lbtwomen_pie.CountryCode).sum().reset_index())


lbtwomen_pie_LBT = [["by Male",lbtwomen_pie_L_c["percentage"].mean(),lbtwomen_pie_B_c["percentage"].mean(),lbtwomen_pie_T_c["percentage"].mean(),"#FF6565","#C264A5","#FD8ABD"],["Exclusively by Female",100-lbtwomen_pie_L_c["percentage"].mean(),100-lbtwomen_pie_B_c["percentage"].mean(),100-lbtwomen_pie_T_c["percentage"].mean(),"#F4F2F0","#F4F2F0","#F4F2F0"]]
lbtwomen_pie_LBT = pd.DataFrame(lbtwomen_pie_LBT)
lbtwomen_pie_LBT.columns = ("legend_label","Lesbians","Bisexual_Women","Transgender_Women","color_L","color_B","color_T")
lbtwomen_pie_LBT["Lesbians"] = lbtwomen_pie_LBT["Lesbians"].round(decimals=0)
lbtwomen_pie_LBT["Bisexual_Women"] = lbtwomen_pie_LBT["Bisexual_Women"].round(decimals=0)
lbtwomen_pie_LBT["Transgender_Women"] = lbtwomen_pie_LBT["Transgender_Women"].round(decimals=0)

lbtwomen_pie_LBT["L_endangle"] = (lbtwomen_pie_LBT["Lesbians"]/100)*2*np.pi
lbtwomen_pie_LBT["L_startangle"] = (1-(lbtwomen_pie_LBT["Lesbians"]/100))*2*np.pi

L = figure(plot_height=350,title="Lesbian Women",toolbar_location=None,tooltips="@legend_label: @Lesbians %")
L.wedge(x=0,y=1,radius=0.5,start_angle="L_startangle",end_angle="L_endangle",line_color="white",fill_color="color_L",legend="legend_label",source=lbtwomen_pie_LBT)

L.plot_width=400
L.axis.axis_label = None
L.axis.visible = False
L.grid.grid_line_color = None


lbtwomen_pie_LBT["B_endangle"] = (lbtwomen_pie_LBT["Bisexual_Women"]/100)*2*np.pi
lbtwomen_pie_LBT["B_startangle"] = (1-(lbtwomen_pie_LBT["Bisexual_Women"]/100))*2*np.pi

B = figure(plot_height=350,title="Bisexual Women",toolbar_location=None,tooltips="@legend_label: @Bisexual_Women %")
B.wedge(x=0,y=1,radius=0.5,start_angle="B_startangle",end_angle="B_endangle",line_color="white",fill_color="color_B",legend="legend_label",source=lbtwomen_pie_LBT)

B.plot_width=400
B.axis.axis_label = None
B.axis.visible = False
B.grid.grid_line_color = None


lbtwomen_pie_LBT["T_endangle"] = (lbtwomen_pie_LBT["Transgender_Women"]/100)*2*np.pi
lbtwomen_pie_LBT["T_startangle"] = (1-(lbtwomen_pie_LBT["Transgender_Women"]/100))*2*np.pi


T = figure(plot_height=350,title="Transgender Women",toolbar_location=None,tooltips="@legend_label: @Transgender_Women %")
T.wedge(x=0,y=1,radius=0.5,start_angle="T_startangle",end_angle="T_endangle",line_color="white",fill_color="color_T",legend="legend_label",source=lbtwomen_pie_LBT)

T.plot_width=400
T.axis.axis_label = None
T.axis.visible = False
T.grid.grid_line_color = None


show(column(Div(text="<h1>Worst Physical/Sexual Attack or Threat Against LBT Women (EU - 2012)<h1>"),row(L,B,T)))
transf_ = pd.read_csv("../input/queerwomenviolence/transfeminicides_world.csv")
transf_mx = transf_[transf_["Country"]=="Mexico"]
transf_bra = transf_[transf_["Country"]=="Brazil"]
transf_w = transf_[(transf_["Country"]!="Brazil")&(transf_["Country"]!="Mexico")]
transf_.head()
from bokeh.plotting import figure, show, output_file
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models import ColumnDataSource
from bokeh.models.annotations import Title

def coor_to_mercator(df, lon="Longitude", lat="Latitude"):

      k = 6378137
      df["x"] = df[lon] * (k * np.pi/180.0)
      df["y"] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k

      return df

transf_mx_mercator = coor_to_mercator(transf_mx)
transf_bra_mercator = coor_to_mercator(transf_bra)
transf_w_mercator = coor_to_mercator(transf_w)

pd.DataFrame(transf_["Latitude"].groupby(transf_.Country).count()).sort_values(by="Latitude")
tile_provider = get_provider(Vendors.CARTODBPOSITRON_RETINA)
p = figure(plot_width=600,x_range=(-14000000, -3000000), y_range=(-2500000, 2500000),
           x_axis_type="mercator", y_axis_type="mercator")
p.add_tile(tile_provider)
p.plot_height=600

p.inverted_triangle(x="x",y="y",size=5,fill_color="#444444",line_color="#F4F2F0",line_width=0.2,fill_alpha=0.3,source=transf_w_mercator)
p.inverted_triangle(x="x",y="y",size=5,fill_color="#FD8ABD",line_color="#F17AA9",line_width=0.2,fill_alpha=0.3,source=transf_bra_mercator)
p.inverted_triangle(x="x",y="y",size=5,fill_color="#FD8ABD",line_color="#F17AA9",line_width=0.2,fill_alpha=0.3,source=transf_mx_mercator)

show(column(Div(text="<h1>Transgender People Murdered in the America registered by TDoR (1998-2020)<h1>"),p))
