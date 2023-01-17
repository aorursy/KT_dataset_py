#read and inspect the dataframe
import pandas as pd
df_health_facilities =  pd.read_csv("../input/health-facilities-gh.csv")
df_health_facilities.head()
# quick summary of the dataset filtered to show only relevant data.
# freq: the frequency of the top item.
df_health_facilities.describe(include = "all")[["Region","District","FacilityName","Type","Town","Ownership"]].iloc[0:4]
df = df_health_facilities.groupby("Type").count()[["Region"]].reset_index()

df.columns = ["Type","count"]
df = df.sort_values(by="count",ascending=False)
df.head(5)
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource, LabelSet
import math

output_notebook()

#plotting with bokeh... api is quite functional needs a bit of getting used-to. 
#for standard visualisations you are more productive using Tableau, Qlik, PowerBI etc.
source = df
plot = figure(plot_width=800, plot_height=600,x_range=df['Type'][0:20],title="Number Of Health Facilities",tools=["xwheel_zoom","reset","pan"])
plot.vbar(source=ColumnDataSource(source), x="Type",width=0.8, top="count",color="cornflowerblue")

labels = LabelSet(x='Type', y='count', text="count", x_offset=-8,
                  source=ColumnDataSource(source),text_font_size="8pt", text_color="#555555",
                   text_align='left')


plot.add_layout(labels)
plot.xaxis.major_label_orientation = math.pi/2
show(plot)
df_region_by_type = df_health_facilities[["Region","Type","FacilityName"]].groupby(["Region","Type"], as_index=False).count()
df_region_by_type.columns = ["Region","Type","FacilityCount"]

#shaping the data
df_region_by_type_pivot = df_region_by_type.pivot("Region","Type","FacilityCount").fillna(0)
df_region_by_type_pivot["total"] = df_region_by_type_pivot.sum(axis=1)
df_region_by_type_pivot = df_region_by_type_pivot.sort_values("total",ascending=False)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

plt.figure(figsize=(20,6))

plt.yticks(rotation=0)
plt.title("Regions Vs Type of Facilities")
sns.heatmap(df_region_by_type_pivot,annot=True,fmt="0.0f",cmap="RdGy",linewidth=.02)
# Shaping the data.
df_region_by_owner = df_health_facilities[["Region","Ownership","FacilityName"]].groupby(["Region","Ownership"], as_index=False).count()
df_region_by_owner.columns = ["Region","Ownership","FacilityCount"]
df_region_by_owner_pivot = df_region_by_owner.pivot("Region","Ownership","FacilityCount").fillna(0)
df_region_by_owner_pivot["total"] = df_region_by_owner_pivot.sum(axis=1)
df_region_by_owner_pivot = df_region_by_owner_pivot.sort_values("total",ascending=False)

plt.figure(figsize=(20,6))
plt.yticks(rotation=0)
plt.title("Regions Vs Ownership of Facilities")
sns.heatmap(df_region_by_owner_pivot,annot=True,fmt="0.0f",cmap="RdGy",linewidth=.02)
df_owner_by_type = df_health_facilities[["Ownership","Type","FacilityName"]].groupby(["Ownership","Type"], as_index=False).count()
df_owner_by_type.columns = ["Ownership","Type","FacilityCount"]
df_owner_by_type_pivot = df_owner_by_type.pivot("Ownership","Type","FacilityCount").fillna(0)
df_owner_by_type_pivot["total"] = df_owner_by_type_pivot.sum(axis=1)
df_owner_by_type_pivot = df_owner_by_type_pivot.sort_values("total",ascending=False)


plt.figure(figsize=(20,6))

plt.yticks(rotation=0)
plt.title("Regions Vs Ownership of Facilities")
sns.heatmap(df_owner_by_type_pivot,annot=True,fmt="0.0f",cmap="RdGy",linewidth=.02)
from IPython.core.display import display, HTML

display(HTML("""
<style>
#viz1534357692001 {
    height: 800px;
    width: 1000px;
}
</style>
<h2 id="e5"> Explore Ghana Health Facilites Dataset (<a href="https://public.tableau.com/profile/datanix.ds4good#!/vizhome/ghanahealthinfrastructure/Dashboard1?publish=yes">click to view tableau</a>) </h2>
<div class='tableauPlaceholder' id='viz1534357692001' style='position: relative'><noscript><a href='#'>
<img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;QS&#47;QSTMHFD4J&#47;1_rss.png' style='border: none' />
</a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
<param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;QSTMHFD4J' /> <param name='toolbar' value='yes' />
<param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;QS&#47;QSTMHFD4J&#47;1.png' /> 
<param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' />
<param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object>
</div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1534357692001');                    var vizElement = divElement.getElementsByTagName('object')[0];                    
vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    
var scriptElement = document.createElement('script');                    
scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
vizElement.parentNode.insertBefore(scriptElement, vizElement);                
</script>"""))